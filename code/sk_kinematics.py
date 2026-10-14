from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import casadi as ca
import numpy as np


@dataclass
class Frame:
    path: str
    base_path: str
    transform_in_base: np.ndarray


@dataclass
class ScapulothoracicParameters:
    coordinate_names: tuple
    ellipsoid_radii: np.ndarray
    winging_origin: np.ndarray
    winging_axis: np.ndarray


@dataclass
class Joint:
    name: str
    joint_type: str
    parent_frame: Frame
    child_frame: Frame
    transform_axes: tuple
    scapulothoracic: Optional[ScapulothoracicParameters] = None


class Kinematics:
    def __init__(self, model, model_info=None, *, endpoint_marker="endpoint"):
        self.model = model
        self.path = model.getInputFileName() or None
        self.frames = {}
        self.joints = {}
        self.coordinates = {}
        self.coordinate_names = ()
        self.bodies = ()
        self.endpoint_marker_name = endpoint_marker
        self.endpoint_parent_frame = None
        self.endpoint_location = None
        self.endpoint_transform = None
        self.endpoint_function = None
        self._initialize_from_model(model, model_info, endpoint_marker)

    @classmethod
    def from_model(cls, model, model_info=None, *, endpoint_marker="endpoint"):
        return cls(model, model_info, endpoint_marker=endpoint_marker)

    @staticmethod
    def _path(component):
        return component.getAbsolutePathString()

    @staticmethod
    def _vec3(vector):
        return np.asarray([vector[i] for i in range(3)], dtype=float)

    @classmethod
    def _transform_matrix(cls, transform):
        rotation = transform.R()
        translation = transform.p()
        result = np.eye(4)
        result[:3, :3] = [
            [rotation.get(i, j) for j in range(3)]
            for i in range(3)
        ]
        result[:3, 3] = cls._vec3(translation)
        return result

    def _frame(self, physical_frame):
        path = self._path(physical_frame)
        if path not in self.frames:
            base_frame = physical_frame.findBaseFrame()
            self.frames[path] = Frame(
                path=path,
                base_path=self._path(base_frame),
                transform_in_base=self._transform_matrix(
                    physical_frame.findTransformInBaseFrame()
                ),
            )
        return self.frames[path]

    @staticmethod
    def _array_values(values):
        return [values.get(i) for i in range(values.getSize())]

    def _read_transform_function(self, function, coordinate_names, osim, context):
        function_type = function.getConcreteClassName()
        if function_type not in ("LinearFunction", "Constant", "MultiplierFunction"):
            raise NotImplementedError(
                f"Unsupported OpenSim transform function {function_type} on {context}."
            )
        concrete = getattr(osim, function_type).safeDownCast(function)
        if concrete is None:
            raise TypeError(f"Could not access {function_type} for {context}.")

        if function_type == "MultiplierFunction":
            wrapped = concrete.getFunction()
            if wrapped is None:
                raise ValueError(f"MultiplierFunction has no wrapped function on {context}.")
            return (
                "multiplier",
                float(concrete.getScale()),
                self._read_transform_function(wrapped, coordinate_names, osim, context),
            )
        if function_type == "Constant":
            return ("constant", float(concrete.getValue()))

        coefficients = self._array_values(concrete.getCoefficients())
        if len(coefficients) != 2 or len(coordinate_names) != 1:
            raise NotImplementedError(
                f"Only one-coordinate LinearFunction axes are supported; {context}."
            )
        return ("linear", float(coefficients[0]), float(coefficients[1]))

    def _read_scapulothoracic_parameters(self, joint, osim):
        scapulothoracic = osim.ScapulothoracicJoint.safeDownCast(joint)
        if scapulothoracic is None:
            return None
        direction = float(scapulothoracic.get_scapula_winging_axis_direction())
        return ScapulothoracicParameters(
            coordinate_names=tuple(
                joint.get_coordinates(i).getName() for i in range(4)
            ),
            ellipsoid_radii=self._vec3(
                scapulothoracic.get_thoracic_ellipsoid_radii_x_y_z()
            ),
            winging_origin=np.array([
                scapulothoracic.get_scapula_winging_axis_origin(0),
                scapulothoracic.get_scapula_winging_axis_origin(1),
                0.0,
            ]),
            winging_axis=np.array([-np.sin(direction), np.cos(direction), 0.0]),
        )

    def _read_transform_axes(self, joint, osim):
        custom_joint = osim.CustomJoint.safeDownCast(joint)
        if custom_joint is None:
            if joint.getConcreteClassName() in ("WeldJoint", "ScapulothoracicJoint"):
                return ()
            raise NotImplementedError(
                f"Symbolic kinematics does not yet support {joint.getConcreteClassName()} "
                f"('{joint.getName()}'); use a CustomJoint, WeldJoint, or "
                "ScapulothoracicJoint."
            )

        spatial_transform = custom_joint.get_SpatialTransform()
        axes = []
        for axis_index in range(6):
            axis = spatial_transform.getTransformAxis(axis_index)
            axis_vector = self._vec3(axis.getAxis())
            coordinate_names = self._array_values(axis.getCoordinateNamesInArray())
            function_data = self._read_transform_function(
                axis.getFunction(), coordinate_names, osim,
                f"joint '{joint.getName()}', axis '{axis.getName()}'",
            )

            axes.append({
                "name": axis.getName(),
                "axis": axis_vector,
                "coordinates": coordinate_names,
                "function": function_data,
            })
        return tuple(axes)

    def _initialize_from_model(self, model, model_info, endpoint_marker):
        import opensim as osim

        model.finalizeConnections()

        coordinate_set = model.getCoordinateSet()
        all_coordinates = [
            coordinate_set.get(i) for i in range(coordinate_set.getSize())
        ]
        self.coordinates = {
            coordinate.getName(): float(coordinate.get_default_value())
            for coordinate in all_coordinates
        }
        model_free_names = tuple(
            coordinate.getName()
            for coordinate in all_coordinates
            if not coordinate.get_locked()
        )
        self.coordinate_names = (
            model_free_names if model_info is None
            else tuple(model_info.joint_coordinate_names)
        )
        if self.coordinate_names != model_free_names:
            raise ValueError("ModelInfo coordinate order does not match the supplied model")

        body_set = model.getBodySet()
        self.bodies = tuple(
            self._path(body) for body in (
                body_set.get(i) for i in range(body_set.getSize())
            )
        )

        joint_set = model.getJointSet()
        self.joints = {}
        for i in range(joint_set.getSize()):
            joint = joint_set.get(i)
            parent_frame = self._frame(joint.getParentFrame())
            child_frame = self._frame(joint.getChildFrame())
            parsed_joint = Joint(
                name=joint.getName(),
                joint_type=joint.getConcreteClassName(),
                parent_frame=parent_frame,
                child_frame=child_frame,
                transform_axes=self._read_transform_axes(joint, osim),
                scapulothoracic=self._read_scapulothoracic_parameters(joint, osim),
            )
            self.joints[parsed_joint.name] = parsed_joint

        marker_set = model.getMarkerSet()
        marker = next(
            (
                marker_set.get(i) for i in range(marker_set.getSize())
                if marker_set.get(i).getName() == endpoint_marker
            ),
            None,
        )
        if marker is None:
            raise ValueError(f"Model does not contain marker '{endpoint_marker}'.")
        parent_frame = marker.getParentFrame()
        self.endpoint_parent_frame = self._frame(parent_frame)
        self.endpoint_location = self._vec3(marker.get_location())

        q = ca.MX.sym("q", len(self.coordinate_names))
        values = dict(zip(self.coordinate_names, ca.vertsplit(q)))
        transform = self.get_endpoint_transform_symbolic(values)
        self.endpoint_transform = ca.Function("endpoint_transform", [q], [transform])
        self.endpoint_function = ca.Function(
            "endpoint_function", [q], [transform[:3, 3]]
        )

    def list_bodies(self):
        return list(self.bodies)

    @staticmethod
    def _rotation(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        c, s = np.cos(angle), np.sin(angle)
        result = np.eye(4)
        result[:3, :3] = [
            [c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
            [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
            [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)],
        ]
        return result

    @staticmethod
    def _translation(axis, distance):
        result = np.eye(4)
        result[:3, 3] = np.asarray(axis, dtype=float) * distance
        return result

    @staticmethod
    def _function_value(axis_data, coordinates):
        function_data = axis_data["function"]
        scale = 1.0
        while function_data[0] == "multiplier":
            _, factor, function_data = function_data
            scale *= factor
        kind, *parameters = function_data
        if kind == "constant":
            return scale * parameters[0]
        if kind != "linear":
            raise NotImplementedError(f"Unsupported parsed transform function '{kind}'.")
        coordinate = coordinates.get(
            axis_data["coordinates"][0],
            0.0,
        )
        return scale * (parameters[0] * coordinate + parameters[1])

    def _scapulothoracic_transform(self, parameters, coordinates, symbolic):
        rotation = self._rotation_symbolic if symbolic else self._rotation
        q = [coordinates[name] for name in parameters.coordinate_names]
        ellipsoid = (
            rotation([1, 0, 0], q[0])
            @ rotation([0, 1, 0], q[1])
            @ rotation([0, 0, 1], q[2])
        )

        radii = parameters.ellipsoid_radii[[1, 0, 2]]
        ellipsoid[:3, 3] = np.diag(radii) @ ellipsoid[:3, 2]
        alignment = self._rotation([0, 0, 1], np.pi / 2)
        winging = rotation(parameters.winging_axis, q[3])
        origin = parameters.winging_origin
        winging[:3, 3] = origin - winging[:3, :3] @ origin
        return alignment @ ellipsoid @ alignment.T @ winging

    def _joint_transform(self, joint, coordinates, symbolic=False):
        if joint.joint_type == "WeldJoint":
            return ca.MX.eye(4) if symbolic else np.eye(4)
        if joint.joint_type == "ScapulothoracicJoint":
            return self._scapulothoracic_transform(
                joint.scapulothoracic, coordinates, symbolic
            )
        matrix = ca.MX.eye(4) if symbolic else np.eye(4)
        for axis_data in joint.transform_axes:
            value = self._function_value(axis_data, coordinates)
            if axis_data["name"].startswith("rotation"):
                rotation = (
                    self._rotation_symbolic(axis_data["axis"], value)
                    if symbolic else self._rotation(axis_data["axis"], value)
                )
                matrix = matrix @ rotation
            else:
                translation = (
                    self._translation_symbolic(axis_data["axis"], value)
                    if symbolic else self._translation(axis_data["axis"], value)
                )
                matrix[:3, 3] += translation[:3, 3]
        return matrix

    def _edges(self, coordinates, symbolic=False):
        edges = {}
        for joint in self.joints.values():
            parent = joint.parent_frame
            child = joint.child_frame
            joint_transform = self._joint_transform(joint, coordinates, symbolic)
            inverse = ca.inv if symbolic else np.linalg.inv
            transform = (
                parent.transform_in_base
                @ joint_transform
                @ inverse(child.transform_in_base)
            )
            edges.setdefault(parent.base_path, []).append((child.base_path, transform))
            edges.setdefault(child.base_path, []).append((parent.base_path, inverse(transform)))
        return edges

    def get_transform(self, from_frame, to_frame, coordinates=None, **coordinate_values):
        values = {**self.coordinates, **(coordinates or {}), **coordinate_values}
        return self._find_transform(from_frame, to_frame, values, symbolic=False)

    def get_transform_symbolic(self, from_frame, to_frame, coordinates=None):
        values = {**self.coordinates, **(coordinates or {})}
        return self._find_transform(from_frame, to_frame, values, symbolic=True)

    def _find_transform(self, from_frame, to_frame, coordinates, symbolic):
        edges = self._edges(coordinates, symbolic)
        identity = ca.MX.eye(4) if symbolic else np.eye(4)
        queue = [(from_frame, identity)]
        visited = set()
        while queue:
            current, transform = queue.pop(0)
            if current == to_frame:
                return transform
            if current in visited:
                continue
            visited.add(current)
            for next_frame, edge_transform in edges.get(current, []):
                queue.append((next_frame, transform @ edge_transform))
        raise ValueError(f"No transform chain from {from_frame} to {to_frame}")

    def get_endpoint_transform_symbolic(self, coordinates=None):
        if self.endpoint_parent_frame is None:
            raise ValueError("No endpoint marker is configured.")
        parent = self.endpoint_parent_frame
        base_transform = self.get_transform_symbolic(
            "/ground", parent.base_path, coordinates
        )
        marker_offset = ca.MX.eye(4)
        marker_offset[:3, 3] = self.endpoint_location
        return base_transform @ parent.transform_in_base @ marker_offset

    def inverse_kinematics(
        self,
        targets,
        coordinate_names=None,
        accuracy=1e-8,
    ):
        import opensim as osim

        if self.path is None:
            raise ValueError("inverse_kinematics requires a model loaded from an .osim file")
        targets = np.asarray(targets, dtype=float)
        if targets.ndim == 0 or targets.shape[-1] != 3:
            raise ValueError("targets must have shape (..., 3)")
        if coordinate_names is None:
            coordinate_names = self.coordinate_names
        if len(coordinate_names) == 0:
            raise ValueError("coordinate_names must not be empty")

        target_shape = targets.shape[:-1]
        flat_targets = targets.reshape(-1, 3)
        if len(flat_targets) == 0:
            raise ValueError("targets must contain at least one point")

        marker_data = osim.TimeSeriesTableVec3()
        labels = osim.StdVectorString()
        labels.append(self.endpoint_marker_name)
        marker_data.setColumnLabels(labels)
        marker_data.addTableMetaDataString("Units", "m")
        for sample, target in enumerate(flat_targets):
            row = osim.RowVectorVec3(1)
            row[0] = osim.Vec3(*target)
            marker_data.appendRow(float(sample), row)

        model = osim.Model(self.path)
        marker_reference = osim.MarkersReference(marker_data, osim.SetMarkerWeights())
        coordinate_references = osim.SimTKArrayCoordinateReference()
        state = model.initSystem()
        solver = osim.InverseKinematicsSolver(model, marker_reference, coordinate_references)
        solver.setAccuracy(accuracy)
        marker = model.getMarkerSet().get(self.endpoint_marker_name)

        coordinates, errors = [], []
        for sample, target in enumerate(flat_targets):
            state.setTime(float(sample))
            if sample == 0:
                solver.assemble(state)
            else:
                solver.track(state)
            coordinates.append([
                model.getCoordinateSet().get(name).getValue(state)
                for name in coordinate_names
            ])
            model.realizePosition(state)
            actual = self._vec3(marker.getLocationInGround(state))
            errors.append(np.linalg.norm(actual - target))
        return (
            np.asarray(coordinates).reshape(target_shape + (len(coordinate_names),)),
            np.asarray(errors).reshape(target_shape),
        )

    @staticmethod
    def _rotation_symbolic(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis /= np.linalg.norm(axis)
        x, y, z = axis
        c, s = ca.cos(angle), ca.sin(angle)
        result = ca.MX.eye(4)
        result[:3, :3] = ca.vertcat(
            ca.horzcat(c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s),
            ca.horzcat(y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s),
            ca.horzcat(z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)),
        )
        return result

    @staticmethod
    def _translation_symbolic(axis, distance):
        result = ca.MX.eye(4)
        result[:3, 3] = np.asarray(axis, dtype=float) * distance
        return result


if __name__ == "__main__":
    import opensim as osim

    from .msk_mapbuilder import ModelInfo

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "musculoskeletal_models" / "tsm_4DOF" / "tsm_4DOF.osim"
    model = osim.Model(str(model_path))
    info = ModelInfo.from_model(model)
    kinematics = Kinematics.from_model(model, info)
    print("Free coordinates:", info.joint_coordinate_names)
    print("Endpoint transform function:", kinematics.endpoint_transform)
    print("Endpoint function:", kinematics.endpoint_function)
