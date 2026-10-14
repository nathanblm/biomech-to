from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import sys

os.environ.setdefault("SIMBODY_HOME", str(Path(sys.prefix) / "simbody"))


@dataclass
class MapData:
    joint_coordinates: np.ndarray                           # (n_positions, n_coordinates)
    cartesian_coordinates: np.ndarray                       # (n_positions, 3), ground XYZ, m
    force_magnitudes: np.ndarray                            # (n_magnitudes,), N
    force_directions: np.ndarray                            # (n_directions, 3)
    forces: np.ndarray                                      # (n_forces, 3), ground XYZ, N
    activation_map: np.ndarray                              # (n_positions, n_forces, n_muscles)
    muscle_force_map: np.ndarray                            # same shape, N
    success: np.ndarray                                     # (n_positions, n_forces), bool
    fatigue_map: Optional[np.ndarray] = None                # same shape as activation, s^-1
    fatigue_dominance_map: Optional[np.ndarray] = None      # (n_positions, n_forces)
    muscle_names: tuple = ()                                # Optional, in map-column order

    def __post_init__(self):
        if self.joint_coordinates.ndim != 2:
            raise ValueError("joint_coordinates must be a 2D array.")
        n_positions = len(self.joint_coordinates)
        if self.forces.ndim != 2 or self.forces.shape[1] != 3:
            raise ValueError("forces must have shape (n_forces, 3).")
        n_forces = len(self.forces)
        if self.activation_map.ndim != 3:
            raise ValueError("activation_map must be a 3D array.")
        shape = (n_positions, n_forces, self.activation_map.shape[2])
        expected = {
            "cartesian_coordinates": (n_positions, 3),
            "activation_map": shape, "muscle_force_map": shape,
            "success": shape[:2], "fatigue_map": shape,
            "fatigue_dominance_map": shape[:2],
        }
        for name, target in expected.items():
            array = getattr(self, name)
            if array is not None and array.shape != target:
                raise ValueError(f"{name}: expected {target}, got {array.shape}.")
        if self.success.dtype != np.bool_:
            raise ValueError("success must be a boolean array.")
        if self.force_magnitudes.ndim != 1:
            raise ValueError("force_magnitudes must be a 1D array.")
        if self.force_directions.ndim != 2 or self.force_directions.shape[1] != 3:
            raise ValueError("force_directions must have shape (n_directions, 3).")
        if len(self.force_magnitudes) * len(self.force_directions) != n_forces:
            raise ValueError("Force count does not match magnitude/direction combinations.")
        self.muscle_names = tuple(self.muscle_names)
        if self.muscle_names and len(self.muscle_names) != shape[2]:
            raise ValueError("muscle_names must match the number of muscle columns.")

    @classmethod
    def from_npz(cls, npz_path: Union[Path, str]):
        optional = {"fatigue_map", "fatigue_dominance_map", "muscle_names"}
        values = {}
        with np.load(Path(npz_path), allow_pickle=False) as data:
            for field in fields(cls):
                if field.name in data:
                    values[field.name] = data[field.name]
                elif field.name not in optional:
                    raise ValueError(f"Archive is missing '{field.name}'.")
        return cls(**values)


class SimbodyRenderer:
    def __init__(self, model, endpoint_marker="endpoint"):
        import opensim as osim
        self.osim = osim
        self.model = osim.Model(str(model)) if isinstance(model, (str, Path)) else osim.Model(model)
        self.model.setUseVisualizer(True)
        self.state = self.model.initSystem()
        self.model.realizePosition(self.state)
        self.endpoint_marker = endpoint_marker
        self.visualizer = self.model.getVisualizer().updSimbodyVisualizer()
        self.ground = self.model.getGround().getMobilizedBodyIndex()
        self.point_ids, self.line_ids = [], []
        self.caption_id = None

        # Kept here so importing this module does not require OpenSim.
        class GeometryUpdater(osim.DecorativeGeometryImplementation):
            def __init__(self):
                super().__init__()
                self.points = (osim.Vec3(0), osim.Vec3(0))
                self.text = ""

            def implementLineGeometry(self, line):
                line.setPoint1(self.points[0])
                line.setPoint2(self.points[1])

            def implementTextGeometry(self, text):
                text.setText(self.text)

        self.updater = GeometryUpdater()

    def _vec3(self, values):
        return self.osim.Vec3(*map(float, values))

    def marker_position(self, name):
        position = self.model.getMarkerSet().get(name).getLocationInGround(self.state)
        return np.array([position.get(i) for i in range(3)])

    def _add(self, geometry):
        return self.visualizer.addDecoration(self.ground, self.osim.Transform(), geometry)

    def _set_points(self, points, colors, radius, opacity):
        osim = self.osim
        for i, (point, color) in enumerate(zip(points, colors)):
            if i == len(self.point_ids):
                sphere = osim.DecorativeSphere(1.0)
                sphere.setResolution(0.25)
                self.point_ids.append(self._add(sphere))
            item = self.visualizer.updDecoration(self.point_ids[i])
            item.setTransform(osim.Transform(self._vec3(point)))
            item.setScaleFactors(osim.Vec3(radius, radius, radius))
            item.setColor(self._vec3(color))
            item.setOpacity(opacity)
            item.setRepresentation(osim.DecorativeGeometry.DrawSurface)
        for identifier in self.point_ids[len(points):]:
            self.visualizer.updDecoration(identifier).setRepresentation(osim.DecorativeGeometry.Hide)

    def _set_force(self, force, scale, origin):
        osim = self.osim
        if force is None or np.linalg.norm(force) == 0:
            for identifier in self.line_ids:
                self.visualizer.updDecoration(identifier).setRepresentation(osim.DecorativeGeometry.Hide)
            return
        start = self.marker_position(self.endpoint_marker) if origin is None else origin
        magnitude = np.linalg.norm(force)
        direction = force / magnitude
        end = start + scale * force
        side = np.cross(direction, np.eye(3)[np.argmin(np.abs(direction))])
        side /= np.linalg.norm(side)
        up = np.cross(direction, side)
        head_length = 0.25 * scale * magnitude
        base = end - head_length * direction
        segments = [(start, end)]
        segments += [(end, base + 0.5 * head_length * offset) for offset in (side, -side, up, -up)]
        for i, (a, b) in enumerate(segments):
            self.updater.points = (self._vec3(a), self._vec3(b))
            if i == len(self.line_ids):
                line = osim.DecorativeLine(*self.updater.points)
                line.setLineThickness(4)
                line.setColor(osim.Vec3(0.1, 0.4, 1.0))
                self.line_ids.append(self._add(line))
            else:
                self.visualizer.updDecoration(self.line_ids[i]).implementGeometry(self.updater)
            self.visualizer.updDecoration(self.line_ids[i]).setRepresentation(osim.DecorativeGeometry.DrawSurface)

    def _set_caption(self, caption):
        self.updater.text = caption
        if self.caption_id is None:
            label = self.osim.DecorativeText(caption)
            label.setIsScreenText(True)
            label.setColor(self.osim.Vec3(1.0, 0.4, 0.1))
            self.caption_id = self._add(label)
        else:
            self.visualizer.updDecoration(self.caption_id).implementGeometry(self.updater)

    def show(self, points, colors, caption, *, force=None, radius=0.006,
             opacity=1.0, force_scale=0.01, force_origin=None):
        # Update every decoration first, then display once.
        self._set_points(points, colors, radius, opacity)
        self._set_force(force, force_scale, force_origin)
        self._set_caption(caption)
        self.model.getVisualizer().show(self.state)
        return len(points)


class Visualizer:
    _FIELDS = {
        "activation": ("activation_map", "Activation", "RdYlGn_r", (0, 1)),
        "muscle_force": ("muscle_force_map", "Muscle force [N]", "viridis", None),
        "fatigue": ("fatigue_map", "Fatigue coefficient [s^-1]", "viridis", None),
        "dominance": ("fatigue_dominance_map", "Fatigue dominance", "coolwarm", (-1, 1)),
    }

    def __init__(self, model=None, *, data: Optional[MapData] = None,
                 model_path=None, muscle_names=None,
                 endpoint_marker="endpoint"):
        """Use an OpenSim model for motion export and optional MapData for plots."""
        # Preserve the existing Visualizer(data, model_path=...) calling form.
        if isinstance(model, MapData):
            if data is not None:
                raise ValueError("Map data was supplied twice.")
            data, model = model, None
        if isinstance(model, (str, Path)):
            if model_path is not None:
                raise ValueError("Provide either model or model_path.")
            model_path, model = model, None
        if model is not None and model_path is not None:
            raise ValueError("Provide either model or model_path.")
        self.model = model
        self.data = data
        self.model_path = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else None
        )
        self.endpoint_marker = endpoint_marker
        if muscle_names is None:
            muscle_names = data.muscle_names if data is not None else ()
        self.muscle_names = tuple(muscle_names)
        if data is not None and self.muscle_names and len(self.muscle_names) != data.activation_map.shape[2]:
            raise ValueError("muscle_names must match the map's muscle axis.")
        self._renderer = None
        self._reference_model = None
        self._reference_state = None

    @classmethod
    def from_npz(cls, npz_path: Union[Path, str], **kwargs):
        return cls(data=MapData.from_npz(npz_path), **kwargs)

    def _get_model(self):
        if self.model is None:
            if self.model_path is None:
                raise ValueError("Provide a model or model_path for this operation.")
            import opensim as osim
            self.model = osim.Model(str(self.model_path))
        return self.model

    def save_motion_sto(self, path, time, states):
        """Export unlocked coordinate values and speeds to an OpenSim .sto file.

        states has shape (2 * n_unlocked_coordinates, n_times), with rows
        [q1, speed1, q2, speed2, ...] in the model's CoordinateSet order.
        Rotations are in radians. Labels come from each coordinate's full model
        path; locked coordinates remain at the defaults in the supplied model.
        Export does not initialize a simulation or open a visualizer window.
        """
        model = self._get_model()
        model.finalizeConnections()
        state_labels = [
            f"{coordinate.getAbsolutePathString()}/{kind}"
            for coordinate in model.getCoordinateSet()
            if not coordinate.get_locked()
            for kind in ("value", "speed")
        ]
        time = np.asarray(time, dtype=float)
        states = np.asarray(states, dtype=float)
        if time.ndim != 1 or time.size == 0 or not np.isfinite(time).all():
            raise ValueError("time must be a non-empty, finite 1D array.")
        if np.any(np.diff(time) <= 0):
            raise ValueError("time must be strictly increasing.")
        expected_shape = (len(state_labels), len(time))
        if states.shape != expected_shape:
            raise ValueError(
                f"states must have shape {expected_shape} with alternating "
                "coordinate value/speed rows for the model's unlocked coordinates; "
                f"got {states.shape}."
            )
        if not np.isfinite(states).all():
            raise ValueError("states must contain only finite values.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            file.write("motion\n")
            file.write("version=1\n")
            file.write(f"nRows={len(time)}\n")
            file.write(f"nColumns={len(state_labels) + 1}\n")
            file.write("inDegrees=no\n")
            file.write("endheader\n")
            file.write("time\t" + "\t".join(state_labels) + "\n")
            for k, sample_time in enumerate(time):
                row = np.concatenate(([sample_time], states[:, k]))
                file.write("\t".join(f"{value:.12g}" for value in row) + "\n")
        return path

    def _simbody(self):
        if self._renderer is None:
            self._renderer = SimbodyRenderer(self._get_model(), self.endpoint_marker)
        return self._renderer

    def _marker_position(self, name):
        if self._renderer is not None:
            return self._renderer.marker_position(name)
        if self._reference_model is None:
            import opensim as osim
            self._reference_model = osim.Model(self._get_model())
            self._reference_model.setUseVisualizer(False)
            self._reference_state = self._reference_model.initSystem()
            self._reference_model.realizePosition(self._reference_state)
        marker = self._reference_model.getMarkerSet().get(name)
        position = marker.getLocationInGround(self._reference_state)
        return np.array([position.get(i) for i in range(3)])

    def _muscle_index(self, muscle):
        if isinstance(muscle, str):
            if not self.muscle_names:
                raise ValueError("No saved muscle names; supply muscle_names or use a muscle index.")
            if muscle not in self.muscle_names:
                raise ValueError(f"Unknown muscle {muscle!r}.")
            return self.muscle_names.index(muscle)
        if not isinstance(muscle, (int, np.integer)):
            raise ValueError("Specify muscle as a name or integer index.")
        if not 0 <= muscle < self.data.activation_map.shape[2]:
            raise IndexError("Muscle index is out of range.")
        return int(muscle)

    def show(self, field="activation", *, muscle=None, force_index=0,
             backend="matplotlib", slice_axis=None, slice_value=None,
             tolerance=0.01, reference_marker=None, filter_success=True,
             value_range=None, cmap=None, max_points=None, radius=0.003,
             opacity=0.8, point_size=8, show_force=True, force_scale=0.01,
             force_origin=None, block=True):
        """Display a full cloud or a slice; return (fig, ax) or Simbody point count.

        slice_axis: 'x', 'y' or 'z'. slice_value: absolute ground coordinate,
        or an offset from reference_marker in the model's default pose (metres).
        Matplotlib projects slices onto the other two axes; Simbody retains 3D.
        value_range fixes the colour scale, useful for comparing maps.
        max_points optionally subsamples only the displayed points.
        force_scale is metres/Newton; force_origin is a ground-frame point.
        """
        if self.data is None:
            raise ValueError("Map display requires data; pass MapData or use Visualizer.from_npz().")
        if backend not in ("matplotlib", "simbody"):
            raise ValueError("backend must be 'matplotlib' or 'simbody'.")
        if field != "workspace" and field not in self._FIELDS:
            raise ValueError(f"Unknown field {field!r}; use {tuple(self._FIELDS)} or 'workspace'.")
        if not isinstance(force_index, (int, np.integer)) or not 0 <= force_index < len(self.data.forces):
            raise IndexError("force_index is outside the saved force samples.")
        if (slice_axis is None) != (slice_value is None):
            raise ValueError("Provide slice_axis and slice_value together.")
        if slice_axis is not None and slice_axis not in ("x", "y", "z"):
            raise ValueError("slice_axis must be 'x', 'y' or 'z'.")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative.")
        if not 0 <= opacity <= 1 or not np.isfinite(radius) or radius <= 0:
            raise ValueError("opacity must be in [0, 1] and radius must be positive.")
        if not np.isfinite(force_scale) or force_scale <= 0:
            raise ValueError("force_scale must be finite and positive.")
        if max_points is not None and (not isinstance(max_points, (int, np.integer)) or max_points <= 0):
            raise ValueError("max_points must be a positive integer.")

        points = self.data.cartesian_coordinates
        mask = np.isfinite(points).all(axis=1)
        if filter_success:
            mask &= self.data.success[:, force_index]
        values, norm, colormap = None, None, None
        title = "Workspace"
        if field != "workspace":
            attribute, label, default_cmap, default_range = self._FIELDS[field]
            array = getattr(self.data, attribute)
            if array is None:
                raise ValueError(f"{attribute} has not been saved/computed.")
            title = label
            if field == "dominance":
                if muscle is not None:
                    raise ValueError("Dominance is a group-level map; omit muscle.")
                values = array[:, force_index]
            else:
                index = self._muscle_index(muscle)
                values = array[:, force_index, index]
                name = self.muscle_names[index] if self.muscle_names else f"muscle {index}"
                title += f" | {name}"
            mask &= np.isfinite(values)
            if not np.any(mask):
                raise ValueError("No finite samples satisfy the validity filter.")
            limits = value_range if value_range is not None else default_range
            if limits is None:
                # Full valid map, before slicing/subsampling, in physical units.
                limits = (0.0, max(float(values[mask].max()), 1e-12))
            limits = np.asarray(limits, dtype=float)
            if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError("value_range must be finite (minimum, maximum) with minimum < maximum.")
            norm = Normalize(*limits, clip=True)
            colormap = plt.get_cmap(cmap or default_cmap)

        origin = np.zeros(3) if reference_marker is None else self._marker_position(reference_marker)
        axis = None if slice_axis is None else "xyz".index(slice_axis)
        if axis is not None:
            if not np.isfinite(slice_value):
                raise ValueError("slice_value must be finite.")
            plane = origin[axis] + slice_value
            mask &= np.abs(points[:, axis] - plane) <= tolerance
            title += f" | {slice_axis}={plane:.3f} +/- {tolerance:g} m"
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            raise ValueError("No samples match the selection; check the slice and success filter.")
        if max_points is not None and len(indices) > max_points:
            indices = np.random.default_rng(0).choice(indices, size=max_points, replace=False)
        selected_points = points[indices]
        force = self.data.forces[force_index]
        title += f" | F[{force_index}]={np.array2string(force, precision=2)} N"
        if show_force and not np.isfinite(force).all():
            raise ValueError("Cannot draw a non-finite force vector.")
        if force_origin is not None:
            force_origin = np.asarray(force_origin, dtype=float)
            if force_origin.shape != (3,) or not np.isfinite(force_origin).all():
                raise ValueError("force_origin must be a finite ground-frame (3,) vector.")

        if backend == "simbody":
            colors = np.tile([0.1, 0.4, 1.0], (len(indices), 1)) if values is None else colormap(norm(values[indices]))[:, :3]
            caption = title if norm is None else f"{title} | scale [{norm.vmin:g}, {norm.vmax:g}]"
            return self._simbody().show(
                selected_points, colors, caption, force=force if show_force else None,
                radius=radius, opacity=opacity, force_scale=force_scale, force_origin=force_origin,
            )

        displayed = selected_points - origin
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d") if axis is None else fig.add_subplot(111)
        axes = [0, 1, 2] if axis is None else [i for i in range(3) if i != axis]
        style = {"s": point_size, "alpha": opacity}
        if values is None:
            style["color"] = "tab:blue"
        else:
            style.update(c=values[indices], cmap=colormap, norm=norm)
        scatter = ax.scatter(*(displayed[:, i] for i in axes), **style)
        if values is not None:
            fig.colorbar(scatter, ax=ax, label=label, shrink=0.8)
        reference = "ground" if reference_marker is None else reference_marker
        ax.set_xlabel(f"{'XYZ'[axes[0]]} relative to {reference} [m]")
        ax.set_ylabel(f"{'XYZ'[axes[1]]} relative to {reference} [m]")
        if axis is None:
            ax.set_zlabel(f"Z relative to {reference} [m]")
            span = np.ptp(displayed, axis=0)
            ax.set_box_aspect(np.maximum(span, max(float(span.max()), 1e-3) * 0.01))
        else:
            ax.set_aspect("equal", adjustable="box")
        if show_force and np.linalg.norm(force) > 0:
            # Without a model this is a reference arrow at the selected cloud's centre.
            start = (selected_points.mean(axis=0) if force_origin is None else force_origin) - origin
            vector = force_scale * force
            if axis is None:
                ax.quiver(*start, *vector, color="tab:blue", normalize=False)
            elif np.linalg.norm(vector[axes]) > 0:
                ax.quiver(*start[axes], *vector[axes], angles="xy", scale_units="xy", scale=1, color="tab:blue")
        ax.set_title(title, fontsize=10)
        ax.grid(True)
        fig.tight_layout()
        plt.show(block=block)
        return fig, ax

    def show_activation(self, muscle, force_index=0, **kwargs):
        return self.show("activation", muscle=muscle, force_index=force_index, **kwargs)

    def show_muscle_force(self, muscle, force_index=0, **kwargs):
        return self.show("muscle_force", muscle=muscle, force_index=force_index, **kwargs)

    def show_fatigue(self, muscle, force_index=0, **kwargs):
        return self.show("fatigue", muscle=muscle, force_index=force_index, **kwargs)

    def show_dominance(self, force_index=0, **kwargs):
        return self.show("dominance", force_index=force_index, **kwargs)

    def show_workspace(self, force_index=0, **kwargs):
        kwargs.setdefault("filter_success", False)
        return self.show("workspace", force_index=force_index, **kwargs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--model_path", type=Path, help="Matching .osim file (required for Simbody)")
    parser.add_argument("--field", choices=["workspace", *Visualizer._FIELDS], default="dominance")
    parser.add_argument("--muscle", type=int, help="Muscle column index")
    parser.add_argument("--force", type=int, default=0, help="Flattened force index")
    parser.add_argument("--backend", choices=["matplotlib", "simbody"], default="matplotlib")
    parser.add_argument("--slice-axis", choices=["x", "y", "z"])
    parser.add_argument("--slice-value", type=float)
    parser.add_argument("--reference-marker", help="Slice/reference origin at this marker's default position")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--max-points", type=int, help="Optional display-only subsampling")
    parser.add_argument("--radius", type=float, default=0.006, help="Simbody point radius in metres")
    parser.add_argument("--opacity", type=float, default=1.0, help="Point opacity in the range [0, 1]")
    args = parser.parse_args()
    viewer = Visualizer.from_npz(args.npz_path, model_path=args.model_path)
    viewer.show(args.field, muscle=args.muscle, force_index=args.force, backend=args.backend,
                slice_axis=args.slice_axis, slice_value=args.slice_value, tolerance=args.tolerance,
                reference_marker=args.reference_marker, max_points=args.max_points,
                radius=args.radius, opacity=args.opacity)
    if args.backend == "simbody":
        input("Press Enter to close the viewer...")
