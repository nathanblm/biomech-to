
import os
import sys
from pathlib import Path

os.environ.setdefault("SIMBODY_HOME", str(Path(sys.prefix) / "simbody"))

import numpy as np
import opensim as osim
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "rmr-solver"))

from RMRsolver import RMRsolver
from utilsObjectives import ActSquared

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from itertools import islice
from tqdm import tqdm


class FatigueMap(osim.DecorativeGeometryImplementation):
    def __init__(self, model, endpoint_body=None, endpoint_marker=None, logging=True):
        super().__init__()
        self.model = model
        self.model.setUseVisualizer(True)
        self.state = model.initSystem()

        if endpoint_marker:
            self.endpoint = self.model.getMarkerSet().get(endpoint_marker)
            self.point_type = "marker"
        elif endpoint_body:
            self.endpoint = self.model.getBodySet().get(endpoint_body)
            self.point_type = "body"
        else:
            raise ValueError("Either endpoint_body or endpoint_marker must be provided.")

        self.freeCoordNames = np.array([])          # shape (n_coords,)
        self.freeCoordLims = np.empty((0, 2))       # shape (n_coords, 2)

        self.muscleNum = 0
        self.muscles = np.empty(0)                  # shape (n_muscles,)
        self.muscleNames = np.empty(0)              # shape (n_muscles,)

        self.actuatorNum = 0
        self.actuators = np.empty(0)                # shape (n_actuators,)
        self.actuatorNames = np.empty(0)            # shape (n_actuators,)
        self.actuatorMuscleIndex = []               # indices of actuators which are muscles

        self.coordSampleSize = 10000
        self.freeCoordSamples = np.empty((0, 3))    # shape (n_coordSamples, n_coords)
        self.reachWorkspace = np.empty((0, 3))      # shape (n_coordSamples, 3)

        self.FmagSampleSize = 1
        self.FmagSamples = np.array([])             # shape (n_FmagSamples,)

        self.FdirSampleSize = 1
        self.FdirSamples = np.empty((0, 3))         # shape (n_FdirSamples, 3)

        self.prescribedForces = np.empty(0)         # shape (n_prescribedForces,)
        self.prescribedForceIndexes = np.empty(0)   # shape (n_prescribedForces,)

        self.activations = np.empty((0, 0, 0))      # shape (n_coordSamples, n_forces, n_muscles)

        self._setup_coordinates()
        self._setup_muscles()
        self._setup_actuators()
        self._setup_prescribed_forces()

        if logging:
            print(f"\n--- Model: {model.getName()} ---")
            print(f"\nEndpoint:\n{self.endpoint.getName()} ({self.point_type})")
            print(f"\nFree Coordinates:\n{self.freeCoordNames}")
            print(f"\nCoordinate Limits:\n{self.freeCoordLims}")
            print(f"\nMuscles:\n{self.muscleNames}")
            print(f"\nActuators:\n{self.actuatorNames}")
            print(f"\nMuscle Actuator Indices:\n{self.actuatorMuscleIndex}")
            print(f"\nPrescribed Forces:\n{[force.getName() for force in self.prescribedForces]}")
            print(f"\nPrescribed Force Indices:\n{self.prescribedForceIndexes}")

    def _setup_coordinates(self):
        for i in range(self.model.getCoordinateSet().getSize()):
            coordinate = self.model.getCoordinateSet().get(i)
            if coordinate.get_locked():
                continue

            self.freeCoordNames = np.append(self.freeCoordNames, coordinate.getName())
            self.freeCoordLims = np.vstack((self.freeCoordLims, [coordinate.getRangeMin(), coordinate.getRangeMax()]))

    def _setup_muscles(self):
        muscleSet = self.model.getMuscles()
        self.muscleNum = muscleSet.getSize()
        self.muscles = np.empty(self.muscleNum, dtype=object)
        self.muscleNames = np.empty(self.muscleNum, dtype=object)

        for i in range(self.muscleNum):
            self.muscles[i] = osim.Millard2012EquilibriumMuscle.safeDownCast(muscleSet.get(i))
            self.muscleNames[i] = self.muscles[i].getName()

    def _setup_actuators(self):
        actuatorSet = self.model.getActuators()
        self.actuatorNum = actuatorSet.getSize()
        self.actuators = []
        self.actuatorNames = []
        self.actuatorMuscleIndex = [] # indices of actuators which are muscles

        for i in range(self.actuatorNum):
            self.actuatorNames.append(actuatorSet.get(i).getName())
            self.actuators.append(osim.ScalarActuator.safeDownCast(actuatorSet.get(i)))
            if self.actuatorNames[i] in self.muscleNames:
                self.actuatorMuscleIndex.append(i)

    def _setup_prescribed_forces(self):
        forceSet = self.model.getForceSet()
        self.prescribedForces = []
        self.prescribedForceIndexes = []

        for i in range(forceSet.getSize()):
            force = forceSet.get(i)
            if force.getConcreteClassName() == "PrescribedForce":
                self.prescribedForces.append(force)
                self.prescribedForceIndexes.append(i)

        self.prescribedForces = np.array(self.prescribedForces, dtype=object)
        self.prescribedForceIndexes = np.array(self.prescribedForceIndexes, dtype=int)


    ##################################
    ##           SAMPLING           ##
    ##################################

    def generate_samples(self, 
                         coordSampleSize=10000, 
                         FmagSampleSize=1,
                         FmagMin=25.0,
                         FmagMax=25.0,
                         FdirSampleSize=1):
        self.coordSampleSize = coordSampleSize
        self.FmagSampleSize = FmagSampleSize
        self.FdirSampleSize = FdirSampleSize

        self.sample_reachable_workspace()
        self.sample_force_magnitudes(Fmin=FmagMin, Fmax=FmagMax)
        self.sample_force_directions()

    def sample_reachable_workspace(self):
        rng = np.random.default_rng()

        lower_bounds = self.freeCoordLims[:, 0]
        upper_bounds = self.freeCoordLims[:, 1]

        self.freeCoordSamples = rng.uniform(lower_bounds, upper_bounds, size=(self.coordSampleSize, len(self.freeCoordNames)))

        self.evaluate_workspace()

    def evaluate_workspace(self):
        """Compute positions from existing coordinate rows without resampling."""
        workspace = []

        for sample in tqdm(
            self.freeCoordSamples,
            total=len(self.freeCoordSamples),
            desc="Sampling reachable workspace",
            unit="sample",
        ):
            for name, value in zip(self.freeCoordNames, sample):
                coordinate = self.model.getCoordinateSet().get(str(name))
                coordinate.setValue(self.state, float(value), False)

            self.model.realizePosition(self.state)

            if self.point_type == "marker":
                position = self.endpoint.getLocationInGround(self.state)
            elif self.point_type == "body":
                position = self.endpoint.getPositionInGround(self.state)
            workspace.append([position[i] for i in range(3)])

        self.reachWorkspace = np.asarray(workspace, dtype=float)

    def sample_force_magnitudes(self, Fmin=20.0, Fmax=30.0):
        rng = np.random.default_rng()
        self.FmagSamples = rng.uniform(Fmin, Fmax, size=self.FmagSampleSize)

    def sample_force_directions(self):
        rng = np.random.default_rng()
        self.FdirSamples = rng.normal(size=(self.FdirSampleSize, 3))
        norms = np.linalg.norm(self.FdirSamples, axis=1, keepdims=True)
        self.FdirSamples /= norms

    ##################################
    ##           FATIGUE            ##
    ##################################

    def static_activation_map(self, model_path, force_index, workers=4):
        forces = (
            self.FmagSamples[:, None, None]         # shape (n_FmagSamples, 1, 1)
            * self.FdirSamples[None, :, :]          # shape (1, n_FdirSamples, 3)
        ).reshape(-1, 3)                            # shape (n_FmagSamples * n_FdirSamples, 3)

        tasks = (
            (i, j, q, force)
            for i, q in enumerate(self.freeCoordSamples)
            for j, force in enumerate(forces)
        )                                           # shape (n_coordSamples * n_forces, 4)

        shape = (len(self.freeCoordSamples), len(forces))   # shape (n_coordSamples, n_forces)
        self.staticActivations = np.full(
            (*shape, self.muscleNum), np.nan
        )                                                  # shape (n_coordSamples, n_forces, n_muscles)
        self.staticSuccess = np.zeros(shape, dtype=bool)
        self.staticForces = forces

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=init_worker,
            initargs=(model_path, force_index, self.freeCoordNames),
        ) as pool, tqdm(total=shape[0] * shape[1], desc="Static activation map",
                        unit="solve", dynamic_ncols=True) as progress:
            # Keep a small queue; count results in completion order.
            pending = {pool.submit(solve_sample, task)
                       for task in islice(tasks, workers * 2)}
            failed = 0
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    i, j, activation, ok = future.result()
                    self.staticSuccess[i, j] = ok
                    self.staticActivations[i, j] = activation
                    if not ok:
                        failed += 1
                    progress.set_postfix(failed=failed, refresh=False)
                    progress.update(1)
                    task = next(tasks, None)
                    if task is not None:
                        pending.add(pool.submit(solve_sample, task))

        return self.staticActivations, self.staticSuccess

    ##################################
    ##        VISUALIZATION         ##
    ##################################

    def _reset_to_default_state(self):
        """Reset the displayed model to its default coordinate values."""
        for i in range(self.model.getCoordinateSet().getSize()):
            coordinate = self.model.getCoordinateSet().get(i)
            if not coordinate.get_locked():
                coordinate.setValue(
                    self.state, coordinate.getDefaultValue(), False
                )
        self.model.realizePosition(self.state)

    def _endpoint_position_in_ground(self):
        """Return the configured endpoint position in ground coordinates."""
        self._reset_to_default_state()
        if self.point_type == "marker":
            position = self.endpoint.getLocationInGround(self.state)
        else:
            position = self.endpoint.getPositionInGround(self.state)
        return np.asarray([position[i] for i in range(3)], dtype=float)

    def _simbody_points_overlay(self, points, radius, color, opacity):
        self._reset_to_default_state()
        colors = np.broadcast_to(color, (len(points), 3))
        visualizer = self.model.getVisualizer().updSimbodyVisualizer()
        ground = self.model.getGround().getMobilizedBodyIndex()
        decoration_ids = getattr(self, "_workspace_decoration_ids", [])
        for index, point in enumerate(points):
            if index == len(decoration_ids):
                sphere = osim.DecorativeSphere(1.0)
                sphere.setResolution(0.25)
                decoration_ids.append(visualizer.addDecoration(ground, osim.Transform(), sphere))
            decoration = visualizer.updDecoration(decoration_ids[index])
            decoration.setTransform(osim.Transform(osim.Vec3(*map(float, point))))
            decoration.setScaleFactors(osim.Vec3(radius, radius, radius))
            decoration.setColor(osim.Vec3(*map(float, colors[index])))
            decoration.setOpacity(opacity)
            decoration.setRepresentation(
                osim.DecorativeGeometry.Hide if opacity == 0
                else osim.DecorativeGeometry.DrawSurface
            )
        for decoration_id in decoration_ids[len(points):]:
            visualizer.updDecoration(decoration_id).setRepresentation(osim.DecorativeGeometry.Hide)
        self._workspace_decoration_ids = decoration_ids
        self.model.realizePosition(self.state)
        self.model.getVisualizer().show(self.state)
        return len(points)

    def _simbody_force_visual(self, force, scale=0.01, origin=None):
        """Show ``staticForces[force]`` at the endpoint; scale is metres/N.

        The force vector is expressed in ground coordinates and acts on the arm.
        ``origin`` can be supplied in ground coordinates for a custom placement;
        by default the arrow starts at the configured endpoint.  This keeps the
        visualization independent of model-specific base translations.
        """
        vector = self.staticForces[force]
        magnitude = np.linalg.norm(vector)
        self._reset_to_default_state()
        start = self._endpoint_position_in_ground() if origin is None else np.asarray(origin, dtype=float)
        end = start + scale * vector

        visualizer = self.model.getVisualizer().updSimbodyVisualizer()
        ground = self.model.getGround().getMobilizedBodyIndex()
        if not hasattr(self, "_force_decoration_ids"):
            self._force_decoration_ids = []
        # A zero force collapses all five segments to the origin of the arrow.
        segments = [(start, start)] * 5

        if magnitude > 0:
            direction = vector / magnitude
            reference = np.eye(3)[np.argmin(np.abs(direction))]
            side = np.cross(direction, reference)
            side /= np.linalg.norm(side)
            up = np.cross(direction, side)
            head_length = 0.25 * scale * magnitude
            head_base = end - head_length * direction
            segments = [(start, end)]
            for offset in (side, -side, up, -up):
                segments.append((end, head_base + 0.5 * head_length * offset))
        for index, (a, b) in enumerate(segments):
            self._force_line_points = (
                osim.Vec3(*map(float, a)), osim.Vec3(*map(float, b))
            )
            if index == len(self._force_decoration_ids):
                line = osim.DecorativeLine(*self._force_line_points)
                line.setLineThickness(4)
                line.setColor(osim.Vec3(0.1, 0.4, 1.0))
                line.setOpacity(1)
                self._force_decoration_ids.append(
                    visualizer.addDecoration(ground, osim.Transform(), line)
                )

            else:
                visualizer.updDecoration(
                    self._force_decoration_ids[index]
                ).implementGeometry(self)

    def implementLineGeometry(self, line):
        """Callback used by Simbody when an existing arrow line is redrawn."""
        line.setPoint1(self._force_line_points[0])
        line.setPoint2(self._force_line_points[1])

    def _simbody_activation_title(self, muscle, force):
        """Replace the current caption without accumulating screen decorations."""
        visualizer = self.model.getVisualizer().updSimbodyVisualizer()
        magnitude = np.linalg.norm(self.staticForces[force])
        self._activation_caption = f"Muscle: {muscle} | Force {force}: {magnitude:.2f} N"
        if not hasattr(self, "_activation_label_id"):
            label = osim.DecorativeText(self._activation_caption)
            label.setIsScreenText(True)
            label.setColor(osim.Vec3(1.0, 0.4, 0.1))
            ground = self.model.getGround().getMobilizedBodyIndex()
            self._activation_label_id = visualizer.addDecoration(
                ground, osim.Transform(), label
            )
        else:
            visualizer.updDecoration(self._activation_label_id).implementGeometry(
                self
            )

    def implementTextGeometry(self, text):
        """Callback used by Simbody when the existing caption is redrawn."""
        text.setText(self._activation_caption)

    def _workspace_slice(self, y_offset, shoulder_marker, tolerance):
        self._reset_to_default_state()
        shoulder_marker = self.model.getMarkerSet().get(shoulder_marker)
        shoulder_pos = shoulder_marker.getLocationInGround(self.state)
        shoulder_pos = np.array([shoulder_pos[i] for i in range(3)])
        plane = shoulder_pos + np.array([0, y_offset, 0])

        mask = np.abs(self.reachWorkspace[:, 1] - plane[1]) <= tolerance
        return plane, self.reachWorkspace[mask]

    def plt_3D_workspace_visual(self):
        if self.reachWorkspace.size == 0:
            raise ValueError("Reachable workspace has not been sampled yet.")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.reachWorkspace[:, 0], self.reachWorkspace[:, 1], self.reachWorkspace[:, 2], c='b', marker='o')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.title('Reachable Workspace')
        plt.show()

    def plt_2D_workspace_visual(self, y_offset, shoulder_marker="r_acromion", tolerance=0.01):
        plane, points = self._workspace_slice(y_offset, shoulder_marker, tolerance)
        sliced = points - plane

        fig, ax = plt.subplots()
        ax.scatter(sliced[:, 0], sliced[:, 2], s=5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_title(f"X–Z workspace slice")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        plt.show()

    def simbody_3D_workspace_overlay(self, radius=0.002, color=(0.1, 0.4, 1.0), opacity=0.5):
        return self._simbody_points_overlay(self.reachWorkspace, radius, color, opacity)

    def simbody_2D_workspace_overlay(self, y_offset, shoulder_marker="r_acromion",
                                     tolerance=0.01, radius=0.002,
                                     color=(1.0, 0.4, 1.0), opacity=0.9):
        plane, points = self._workspace_slice(y_offset, shoulder_marker, tolerance)
        return self._simbody_points_overlay(points, radius, color, opacity)

    def simbody_3D_activation_visual(self, muscle, force, radius=0.002, opacity=0.5,
                                     filter_success=True):
        """Colour activation with RdYlGn_r on [0, 1]. force is a sample index."""
        if self.staticActivations.size == 0:
            raise ValueError("Static activations have not been computed yet.")

        if muscle not in self.muscleNames:
            raise ValueError(f"Muscle '{muscle}' not found in the model.")

        muscle_index = np.where(self.muscleNames == muscle)[0][0]
        activations = self.staticActivations[:, force, muscle_index]

        mask = np.isfinite(activations)
        if filter_success:
            mask &= self.staticSuccess[:, force]
        points = self.reachWorkspace[mask]
        values = np.clip(activations[mask], 0, 1)
        colors = plt.get_cmap("RdYlGn_r")(values)[:, :3]

        self._simbody_activation_title(muscle, force)
        self._simbody_force_visual(force)
        return self._simbody_points_overlay(points, radius, colors, opacity)

    def simbody_2D_activation_visual(self, muscle, force, y_offset,
                                     shoulder_marker="r_acromion", tolerance=0.01,
                                     radius=0.006, opacity=0.5, filter_success=True):
        """Colour a shoulder-offset slice with RdYlGn_r on [0, 1].

        force is the force-combination index; negative y_offset moves down.
        """
        plane, _ = self._workspace_slice(y_offset, shoulder_marker, tolerance)
        muscle_index = list(self.muscleNames).index(muscle)
        activations = self.staticActivations[:, force, muscle_index]

        mask = (
            (np.abs(self.reachWorkspace[:, 1] - plane[1]) <= tolerance)
            & np.isfinite(activations)
        )
        if filter_success:
            mask &= self.staticSuccess[:, force]
        points = self.reachWorkspace[mask]
        values = np.clip(activations[mask], 0, 1)
        colors = plt.get_cmap("RdYlGn_r")(values)[:, :3]
        self._simbody_activation_title(muscle, force)
        self._simbody_force_visual(force)
        return self._simbody_points_overlay(points, radius, colors, opacity)

##################################
##     PARALLEL PROCESSING      ##
##################################

def init_worker(model_path, force_index, coordinate_names):
    global solver, initial_guess, coordinate_order

    model = osim.Model(model_path)
    model.setUseVisualizer(False)

    solver = RMRsolver(
        model,
        visualize=False,
        constrainActDyn=False,
        prescribedForceIndex=[int(force_index)],
    )

    # Penalizes both muscle and torque actuators equally
    solver.setObjective(ActSquared(np.ones(solver.actuatorNum)))

    initial_guess = solver.x0.copy()
    coordinate_names = list(coordinate_names)
    coordinate_order = [
        coordinate_names.index(name)
        for name in solver.coordinateFreeNames
    ]

def solve_sample(task):
    i, j, q, force = task

    solver.x0 = initial_guess.copy()
    zeros = np.zeros(solver.coordinateFreeNum)

    activations, joint_forces, info = solver.solve(
        time=0.0,
        position=q[coordinate_order],
        speed=zeros,
        acceleration=zeros,
        values_prescribed_forces=np.r_[force, np.zeros(3)],
    )

    muscle_activations = activations[solver.actuatorMuscleIndex]
    return i, j, muscle_activations, info["success"]

if __name__ == "__main__":

    model_str = "ts-model"  # or "ts-model"

    model_configs = {
        "arm26": {
            "model_path": PROJECT_ROOT / "musculoskeletal-models/arm26/arm26.osim",
            "endpoint_kwargs": {"endpoint_marker": "r_radius_styloid"},
            "coord_sample_size": 100000,
            "force_dir_sample_size": 2,
            "force_mag_sample_size": 1,
            "shoulder_marker": "r_acromion",
            "y_offset": -0.1,
            "workers": 12,
        },
        "ts-model": {
            "model_path": PROJECT_ROOT / "musculoskeletal-models/ts-model/TSM_subject_noWeight.osim",
            "endpoint_kwargs": {"endpoint_body": "hand"},
            "coord_sample_size": 100,
            "force_dir_sample_size": 3,
            "force_mag_sample_size": 1,
            "shoulder_marker": "ac",
            "y_offset": -0.1,
            "workers": 12,
        },
    }

    if model_str not in model_configs:
        raise ValueError(f"Unknown model: {model_str}")

    config = model_configs[model_str]
    model_path = str(config["model_path"])
    model = osim.Model(model_path)
    fatigue_map = FatigueMap(model, **config["endpoint_kwargs"])
    fatigue_map.coordSampleSize = config["coord_sample_size"]
    fatigue_map.FmagSampleSize = config["force_mag_sample_size"]
    fatigue_map.FdirSampleSize = config["force_dir_sample_size"]

    results_dir = PROJECT_ROOT / "results" / model_str
    results_dir.mkdir(parents=True, exist_ok=True)
    file_static_act = (
        f"static_act_p{fatigue_map.coordSampleSize}"
        f"_F{fatigue_map.FmagSampleSize}_D{fatigue_map.FdirSampleSize}"
    )
    file_success = (
        f"static_success_p{fatigue_map.coordSampleSize}"
        f"_F{fatigue_map.FmagSampleSize}_D{fatigue_map.FdirSampleSize}"
    )
    file_coord_samples = f"coord_samples_p{fatigue_map.coordSampleSize}"
    file_forces = f"forces_p{fatigue_map.FmagSampleSize}_D{fatigue_map.FdirSampleSize}"

    try:
        fatigue_map.staticActivations = np.load(results_dir / f"{file_static_act}.npy")
        fatigue_map.staticSuccess = np.load(results_dir / f"{file_success}.npy")
        fatigue_map.freeCoordSamples = np.load(results_dir / f"{file_coord_samples}.npy")
        fatigue_map.staticForces = np.load(results_dir / f"{file_forces}.npy")
    except FileNotFoundError:
        fatigue_map.generate_samples(
            coordSampleSize=config["coord_sample_size"],
            FmagSampleSize=config["force_mag_sample_size"],
            FdirSampleSize=config["force_dir_sample_size"],
            FmagMin=20.0,
            FmagMax=30.0
        )
        fatigue_map.static_activation_map(
            model_path,
            force_index=fatigue_map.prescribedForceIndexes[0],
            workers=config["workers"],
        )
        np.save(results_dir / f"{file_static_act}.npy", fatigue_map.staticActivations)
        np.save(results_dir / f"{file_success}.npy", fatigue_map.staticSuccess)
        np.save(results_dir / f"{file_coord_samples}.npy", fatigue_map.freeCoordSamples)
        np.save(results_dir / f"{file_forces}.npy", fatigue_map.staticForces)
    else:
        fatigue_map.evaluate_workspace()

    for i, muscle in enumerate(fatigue_map.muscleNames):
        for force in range(fatigue_map.staticForces.shape[0]):
                if i != 0:
                    input(f"Press Enter to continue to the next muscle ({muscle}) and force ({force})...")
                fatigue_map.simbody_2D_activation_visual(
                    muscle=muscle,
                    force=force,
                    y_offset=config["y_offset"],
                    shoulder_marker=config["shoulder_marker"],
                    tolerance=0.2,
                    radius=0.006,
                    opacity=1.0,
                    filter_success=True,
                )
