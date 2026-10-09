import xml.etree.ElementTree as ET
import casadi as ca
import numpy as np
import opensim as osim

class Frame:
    def __init__(
            self, 
            name, 
            parent=None, 
            transform_fn=None
    ):
        self.name = name
        self.parent = parent
        self.transform_fn = transform_fn

class Joint:
    def __init__(self, name, joint_type, parent_frame, child_frame, transform, frames):
        self.name = name
        self.joint_type = joint_type
        self.parent_frame = parent_frame
        self.child_frame = child_frame
        self.transform = transform
        self.frames = frames
        
class Kinematics:
    def __init__(self, path):
            self.path = path

            self.frames = {}
            self.joints = {}
            self.coordinates = {}
            self.transforms = {}

            tree = ET.parse(path)
            self.root = tree.getroot()

    def _parse_frames(self):
        joint_objects = self.root.find(".//JointSet/objects")
        if joint_objects is None:
            return self.frames

        parsed_frames = {}

        for joint in joint_objects:
            joint_name = joint.attrib["name"]
            frames = joint.find("frames")
            if frames is None:
                continue

            for frame_element in frames:
                frame_name = frame_element.attrib["name"]
                full_name = f"{joint_name}/{frame_name}"
                parsed_frames[full_name] = Frame(
                    name=full_name,
                    parent=frame_element.findtext("socket_parent"),
                    transform_fn=frame_element,
                )

        self.frames = parsed_frames
        return self.frames

    def list_bodies(self):
        body_objects = self.root.find(".//BodySet/objects")
        if body_objects is None:
            return []

        return [
            f"/bodyset/{body.attrib['name']}"
            for body in body_objects
        ]

    def _parse_joints(self):
        joint_objects = self.root.find(".//JointSet/objects")
        if joint_objects is None:
            return self.joints

        parsed_joints = {}
        for joint_element in joint_objects:
            joint_type = joint_element.tag
            frames = {
                frame.attrib["name"]: frame
                for frame in joint_element.findall("frames/*")
            }
            if joint_type == "WeldJoint":
                transform = "identity"
            else:
                transform = {}
                spatial_transform = joint_element.find("SpatialTransform")
                if spatial_transform is not None:
                    for axis in spatial_transform:
                        function = axis.find("LinearFunction")
                        if function is None:
                            function = axis.find("Constant")
                        transform[axis.attrib["name"]] = {
                            "axis": [float(value) for value in axis.findtext("axis").split()],
                            "coordinates": axis.findtext("coordinates").split(),
                            "function": function,
                        }

            parsed_joints[joint_element.attrib["name"]] = Joint(
                name=joint_element.attrib["name"],
                joint_type=joint_type,
                parent_frame=joint_element.findtext("socket_parent_frame"),
                child_frame=joint_element.findtext("socket_child_frame"),
                transform=transform,
                frames=frames,
            )

            for coordinate in joint_element.findall("coordinates/Coordinate"):
                value = coordinate.findtext("default_value")
                self.coordinates[coordinate.attrib["name"]] = float(value or 0)

        self.joints = parsed_joints
        return self.joints

    #######################################
    ##         numeric transforms        ##
    #######################################

    @staticmethod
    def _rotation(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        cosine = np.cos(angle)
        sine = np.sin(angle)
        rotation = np.array([
            [cosine + x * x * (1 - cosine), x * y * (1 - cosine) - z * sine,
             x * z * (1 - cosine) + y * sine],
            [y * x * (1 - cosine) + z * sine, cosine + y * y * (1 - cosine),
             y * z * (1 - cosine) - x * sine],
            [z * x * (1 - cosine) - y * sine, z * y * (1 - cosine) + x * sine,
             cosine + z * z * (1 - cosine)],
        ])
        result = np.eye(4)
        result[:3, :3] = rotation
        return result

    @staticmethod
    def _translation(axis, distance):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        result = np.eye(4)
        result[:3, 3] = axis * distance
        return result

    def _frame_transform(self, frame):
        translation = [float(value) for value in frame.findtext("translation").split()]
        orientation = [float(value) for value in frame.findtext("orientation").split()]
        result = np.eye(4)
        result[:3, 3] = translation
        for axis, angle in zip(np.eye(3), orientation):
            result = result @ self._rotation(axis, angle)
        return result

    def _joint_transform(self, joint, coordinates):
        if joint.transform == "identity":
            return np.eye(4)

        result = np.eye(4)
        for axis_name, axis_data in joint.transform.items():
            function = axis_data["function"]
            if function.tag == "LinearFunction":
                slope, intercept = [
                    float(value) for value in function.findtext("coefficients").split()
                ]
                coordinate = coordinates.get(axis_data["coordinates"][0], 0)
                value = slope * coordinate + intercept
            else:
                value = float(function.findtext("value"))

            if axis_name.startswith("rotation"):
                result = result @ self._rotation(axis_data["axis"], value)
            else:
                result[:3, 3] += self._translation(axis_data["axis"], value)[:3, 3]
        return result

    def get_transform(self, from_body, to_body, coordinates=None, **coordinate_values):
        """Return T_from_to: the pose of ``to_body`` expressed in ``from_body``."""
        if not self.joints:
            self._parse_joints()

        coordinates = {
            **self.coordinates,
            **(coordinates or {}),
            **coordinate_values,
        }
        edges = {}
        for joint in self.joints.values():
            parent_frame = joint.frames[joint.parent_frame]
            child_frame = joint.frames[joint.child_frame]
            parent_body = parent_frame.findtext("socket_parent")
            child_body = child_frame.findtext("socket_parent")
            transform = (
                self._frame_transform(parent_frame)
                @ self._joint_transform(joint, coordinates)
                @ np.linalg.inv(self._frame_transform(child_frame))
            )
            edges.setdefault(parent_body, []).append((child_body, transform))
            edges.setdefault(child_body, []).append((parent_body, np.linalg.inv(transform)))

        transform = np.eye(4)
        queue = [(from_body, transform)]
        visited = set()
        while queue:
            current_body, current_transform = queue.pop(0)
            if current_body == to_body:
                return current_transform
            if current_body in visited:
                continue
            visited.add(current_body)
            for next_body, edge_transform in edges.get(current_body, []):
                queue.append((next_body, current_transform @ edge_transform))

        raise ValueError(f"No transform chain from {from_body} to {to_body}")

    def inverse_kinematics(
        self,
        marker_name,
        targets,
        coordinate_names,
        accuracy=1e-8,
    ):
        """Solve marker-based inverse kinematics with OpenSim.

        ``targets`` contains ground-frame marker positions in metres and may
        have any leading shape, such as ``(samples, 3)`` or
        ``(rows, columns, 3)``. Returned coordinates and marker errors preserve
        the leading target dimensions.
        """
        targets = np.asarray(targets, dtype=float)
        if targets.ndim == 0 or targets.shape[-1] != 3:
            raise ValueError("targets must have shape (..., 3)")
        if not coordinate_names:
            raise ValueError("coordinate_names must not be empty")

        target_shape = targets.shape[:-1]
        flat_targets = targets.reshape(-1, 3)
        if len(flat_targets) == 0:
            raise ValueError("targets must contain at least one point")

        marker_data = osim.TimeSeriesTableVec3()
        labels = osim.StdVectorString()
        labels.append(marker_name)
        marker_data.setColumnLabels(labels)
        marker_data.addTableMetaDataString("Units", "m")

        for sample, target in enumerate(flat_targets):
            row = osim.RowVectorVec3(1)
            row[0] = osim.Vec3(*target)
            marker_data.appendRow(float(sample), row)

        model = osim.Model(self.path)
        marker_reference = osim.MarkersReference(
            marker_data,
            osim.SetMarkerWeights(),
        )
        coordinate_references = osim.SimTKArrayCoordinateReference()
        state = model.initSystem()
        solver = osim.InverseKinematicsSolver(
            model,
            marker_reference,
            coordinate_references,
        )
        solver.setAccuracy(accuracy)
        marker = model.getMarkerSet().get(marker_name)

        coordinates = []
        errors = []
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
            marker_position = marker.getLocationInGround(state)
            actual = np.array([marker_position[i] for i in range(3)])
            errors.append(np.linalg.norm(actual - target))

        coordinates = np.asarray(coordinates)
        errors = np.asarray(errors)
        return (
            coordinates.reshape(target_shape + (len(coordinate_names),)),
            errors.reshape(target_shape),
        )

    #######################################
    ##         symbolic transforms       ##
    #######################################

    @staticmethod
    def _rotation_symbolic(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        cosine = ca.cos(angle)
        sine = ca.sin(angle)
        rotation = ca.vertcat(
            ca.horzcat(cosine + x * x * (1 - cosine), x * y * (1 - cosine) - z * sine,
                       x * z * (1 - cosine) + y * sine),
            ca.horzcat(y * x * (1 - cosine) + z * sine, cosine + y * y * (1 - cosine),
                       y * z * (1 - cosine) - x * sine),
            ca.horzcat(z * x * (1 - cosine) - y * sine, z * y * (1 - cosine) + x * sine,
                       cosine + z * z * (1 - cosine)),
        )
        result = ca.MX.eye(4)
        result[:3, :3] = rotation
        return result

    @staticmethod
    def _translation_symbolic(axis, distance):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        result = ca.MX.eye(4)
        result[:3, 3] = axis * distance
        return result

    def _frame_transform_symbolic(self, frame):
        translation = [float(value) for value in frame.findtext("translation").split()]
        orientation = [float(value) for value in frame.findtext("orientation").split()]
        result = self._translation_symbolic([1, 0, 0], translation[0])
        result = result @ self._translation_symbolic([0, 1, 0], translation[1])
        result = result @ self._translation_symbolic([0, 0, 1], translation[2])
        for axis, angle in zip(np.eye(3), orientation):
            result = result @ self._rotation_symbolic(axis, angle)
        return result

    def _joint_transform_symbolic(self, joint, coordinates):
        if joint.transform == "identity":
            return ca.MX.eye(4)

        result = ca.MX.eye(4)
        for axis_name, axis_data in joint.transform.items():
            function = axis_data["function"]
            if function.tag == "LinearFunction":
                slope, intercept = [
                    float(value) for value in function.findtext("coefficients").split()
                ]
                coordinate_name = axis_data["coordinates"][0]
                coordinate = coordinates.get(coordinate_name, self.coordinates.get(coordinate_name, 0))
                value = slope * coordinate + intercept
            else:
                value = float(function.findtext("value"))

            if axis_name.startswith("rotation"):
                result = result @ self._rotation_symbolic(axis_data["axis"], value)
            else:
                result[:3, 3] += self._translation_symbolic(axis_data["axis"], value)[:3, 3]
        return result

    def get_transform_symbolic(self, from_body, to_body, coordinates=None):
        """Return T_from_to: the pose of ``to_body`` expressed in ``from_body``."""

        if not self.joints:
            self._parse_joints()

        coordinates = coordinates or {}
        edges = {}
        for joint in self.joints.values():
            parent_frame = joint.frames[joint.parent_frame]
            child_frame = joint.frames[joint.child_frame]
            parent_body = parent_frame.findtext("socket_parent")
            child_body = child_frame.findtext("socket_parent")
            transform = (
                self._frame_transform_symbolic(parent_frame)
                @ self._joint_transform_symbolic(joint, coordinates)
                @ ca.inv(self._frame_transform_symbolic(child_frame))
            )
            edges.setdefault(parent_body, []).append((child_body, transform))
            edges.setdefault(child_body, []).append((parent_body, ca.inv(transform)))

        queue = [(from_body, ca.MX.eye(4))]
        visited = set()
        while queue:
            current_body, current_transform = queue.pop(0)
            if current_body == to_body:
                return current_transform
            if current_body in visited:
                continue
            visited.add(current_body)
            for next_body, edge_transform in edges.get(current_body, []):
                queue.append((next_body, current_transform @ edge_transform))

        raise ValueError(f"No transform chain from {from_body} to {to_body}")
