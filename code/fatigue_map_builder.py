from dataclasses import dataclass
from pathlib import Path
import numpy as np
import opensim as osim
from tqdm import tqdm
from typing import Optional, Union
import multiprocessing as mp
from itertools import islice
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from rmr_solver import RMRsolver, ActSquared

osim.Logger.setLevelString("Error")

_worker_context = None
_worker_output = None

def _init_worker(model_path, model_info, activation_dynamics=False):
    global _worker_context, _worker_output

    osim.Logger.setLevelString("Error")

    model = osim.Model(str(model_path))
    _worker_context = RMRWorkerContext(
        model=model,
        model_info=model_info,
        activation_dynamics=activation_dynamics,
    )

def _solve_task(task):
    return _worker_context.solve_task(task)

def _cast_millard(muscle):
    cast = osim.Millard2012EquilibriumMuscle.safeDownCast(muscle)
    if cast is None:
        raise TypeError(
            f"Expected Millard2012EquilibriumMuscle for "
            f"'{muscle.getName()}'; got {muscle.getConcreteClassName()}."
        )
    return cast

@dataclass
class ModelInfo:
    joint_coordinate_names: tuple[str, ...]     # len: n_coordinates
    joint_coordinate_limits: np.ndarray         # (n_coordinates, 2)
    muscle_names: tuple[str, ...]               # len: n_muscles
    muscle_max_forces: np.ndarray               # (n_muscles,)
    actuator_names: tuple[str, ...]             # len: n_actuators
    muscle_actuator_indices: np.ndarray         # (n_muscles,)
    prescribed_force_names: tuple[str, ...]     # len: n_prescribed_forces
    prescribed_force_indices: tuple[int, ...]   # len: n_prescribed_forces

    def __str__(self):
        return (
            f"ModelInfo(\n"
            f"  joint_coordinate_names={self.joint_coordinate_names},\n"
            f"  joint_coordinate_limits={self.joint_coordinate_limits},\n"
            f"  muscle_names={self.muscle_names},\n"
            f"  muscle_max_forces={self.muscle_max_forces},\n"
            f"  actuator_names={self.actuator_names},\n"
            f"  muscle_actuator_indices={self.muscle_actuator_indices},\n"
            f"  prescribed_force_names={self.prescribed_force_names},\n"
            f"  prescribed_force_indices={self.prescribed_force_indices}\n"
            f")"
        )

    @classmethod
    def from_model(cls, model):
        coordinate_set = model.getCoordinateSet()
        joint_coordinate_names = tuple(
            coordinate.getName() 
            for coordinate in coordinate_set 
            if not coordinate.get_locked()
        )
        joint_coordinate_limits = np.asarray([
            [coordinate.getRangeMin(), coordinate.getRangeMax()]
            for coordinate in coordinate_set
            if not coordinate.get_locked()
        ])

        muscle_set = model.getMuscles()
        muscle_names = tuple(
            _cast_millard(muscle).getName()
            for muscle in muscle_set
        )
        muscle_max_forces = np.asarray([
            _cast_millard(muscle).getMaxIsometricForce()
            for muscle in muscle_set
        ])

        actuator_set = model.getActuators()
        actuator_names = tuple(
            actuator.getName()
            for actuator in actuator_set
        )
        muscle_actuator_indices = np.asarray([
            actuator_set.getIndex(actuator_name)
            for actuator_name in muscle_names
        ])

        force_set = model.getForceSet()
        prescribed_force_names = tuple(
            force.getName()
            for force in force_set
            if force.getConcreteClassName() == "PrescribedForce"
        )
        prescribed_force_indices = tuple([
            force_set.getIndex(force)
            for force in force_set
            if force.getConcreteClassName() == "PrescribedForce"
        ])
        
        return cls(
            joint_coordinate_names=joint_coordinate_names,
            joint_coordinate_limits=joint_coordinate_limits,
            muscle_names=muscle_names,
            muscle_max_forces=muscle_max_forces,
            actuator_names=actuator_names,
            muscle_actuator_indices=muscle_actuator_indices,
            prescribed_force_names=prescribed_force_names,
            prescribed_force_indices=prescribed_force_indices
        )

@dataclass
class Samples:
    joint_coordinates: np.ndarray               # (n_position_samples, n_coordinates)
    cartesian_coordinates: np.ndarray           # (n_position_samples, 3)
    force_magnitudes: np.ndarray                # (n_magnitudes,)
    force_directions: np.ndarray                # (n_directions, 3)
    forces: np.ndarray                          # (n_magnitudes, n_directions, 3)

    @classmethod
    def empty(cls, n_joint_coordinates):
        return cls(
            joint_coordinates=np.empty((0, n_joint_coordinates)),
            cartesian_coordinates=np.empty((0, 3)),
            force_magnitudes=np.empty((0,)),
            force_directions=np.empty((0, 3)),
            forces=np.empty((0, 0, 3))
        )

@dataclass
class Results:
    activations: np.ndarray                     # (n_samples, n_forces, n_muscles)
    muscle_forces: np.ndarray                   # (n_samples, n_forces, n_muscles)
    success: np.ndarray                         # (n_samples, n_forces)

class RMRWorkerContext:
    def __init__(self, model, model_info, activation_dynamics=False):
        self.model = model
        self.model_info = model_info

        self.solver = RMRsolver(
            model=model,
            visualize=False,
            constrainActDyn=activation_dynamics,
            prescribedForceIndex=self.model_info.prescribed_force_indices,
        )

        # equal weights for all muscles and reserve actuators
        self.muscle_weights = np.ones(len(model_info.muscle_names))
        self.reserve_actuator_weights = np.ones(len(model_info.actuator_names) - len(model_info.muscle_names))
        self.weights = np.concatenate([self.muscle_weights, self.reserve_actuator_weights])

        self.solver.setObjective(ActSquared(self.weights))

        if tuple(self.solver.coordinateFreeNames) != self.model_info.joint_coordinate_names:
            raise ValueError("Solver and ModelInfo coordinate orders differ.")

        self.zeros = np.zeros(len(self.model_info.joint_coordinate_names))

    def solve_sample(self, position, force):
        return self.solver.solve(
            time=0.0,
            position=position,
            speed=self.zeros,
            acceleration=self.zeros,
            values_prescribed_forces=np.r_[
                force,
                np.zeros(3),
            ],
        )

    def solve_task(self, task):
        i, j, position, force = task
        activations, joint_forces, muscle_forces, info = self.solve_sample(
            position, force
        )

        muscle_activations = activations[self.solver.actuatorMuscleIndex]

        return (
            i,
            j,
            muscle_activations,
            muscle_forces,
            info["success"],
        )

class FatigueMapBuilder:
    def __init__(self, model, info: ModelInfo, *, model_path=None):
        self.model = model
        self.info = info
        self.model_path = Path(model_path).resolve()
        self.state = model.initSystem()

        self.samples = Samples.empty(n_joint_coordinates=len(info.joint_coordinate_names))

    @classmethod
    def from_model(cls, model, *, model_path=None):
        return cls(model, ModelInfo.from_model(model), model_path=model_path)

    def generate_position_samples(self, n_samples: int):

        def _sample_joint_coordinates(n_samples: int):
            rng = np.random.default_rng()
            lower_bounds = self.info.joint_coordinate_limits[:, 0]
            upper_bounds = self.info.joint_coordinate_limits[:, 1]
            joint_samples = rng.uniform(
                low=lower_bounds,
                high=upper_bounds,
                size=(n_samples, len(lower_bounds))
            )
            return joint_samples
        
        def _order_joint_samples(joint_samples):
            # to utilize warm starting of the solver 
            order = np.lexsort(joint_samples[:, ::-1].T)
            return joint_samples[order]

        def _compute_cartesian_coordinates(joint_samples: np.ndarray):
            endpoint = self.model.getMarkerSet().get("endpoint")
            cartesian_samples = np.empty((len(joint_samples), 3))

            for sample_index, joint_sample in enumerate(tqdm(
                joint_samples,
                desc="Computing Cartesian coordinates",
                unit="sample")):

                for name, sample_value in zip(self.info.joint_coordinate_names, joint_sample):
                    coordinate = self.model.getCoordinateSet().get(name)
                    coordinate.setValue(self.state, sample_value)

                self.model.realizePosition(self.state)
                cartesian_position = endpoint.getLocationInGround(self.state)
                cartesian_samples[sample_index] = np.array([
                    cartesian_position.get(0),
                    cartesian_position.get(1),
                    cartesian_position.get(2)
                ])

            return cartesian_samples

        joint_samples = _sample_joint_coordinates(n_samples)
        joint_samples = _order_joint_samples(joint_samples)
        cartesian_samples = _compute_cartesian_coordinates(joint_samples)

        self.samples.joint_coordinates = joint_samples
        self.samples.cartesian_coordinates = cartesian_samples

    def generate_force_samples(
            self,
            *,
            n_magnitudes: int = 1,
            n_directions: int = 1,
            magnitude_min: float = 20.0,
            magnitude_max: float = 30.0,
            magnitudes: Optional[Union[int, float, np.ndarray]] = None,
            directions: Optional[np.ndarray] = None
            ):

            rng = np.random.default_rng()
            
            if magnitudes is None:
                magnitudes = rng.uniform(
                    magnitude_min,
                    magnitude_max,
                    size=n_magnitudes,
                )
            else:
                magnitudes = np.asarray(magnitudes, dtype=float).reshape(-1)

            if directions is None:
                directions = rng.normal(size=(n_directions, 3))
            else:
                directions = np.asarray(directions, dtype=float).reshape(-1, 3)

            norms = np.linalg.norm(directions, axis=1)
            if np.any(norms == 0):
                raise ValueError("Directions must be non-zero.")

            directions = directions / norms[:, None]

            self.samples.force_magnitudes = magnitudes
            self.samples.force_directions = directions
            self.samples.forces = (magnitudes[:, None, None] * directions[None, :, :]).reshape(-1, 3)
            
    def compute_activation_and_muscle_force_maps(self, *, workers=4, activation_dynamics=False):
        n_positions = len(self.samples.joint_coordinates)
        n_forces = len(self.samples.forces)
        n_muscles = len(self.info.muscle_names)
        total_tasks = n_positions * n_forces

        activations = np.full((n_positions, n_forces, n_muscles), np.nan)
        muscle_forces = np.full((n_positions, n_forces, n_muscles), np.nan)
        success = np.zeros((n_positions, n_forces), dtype=bool)

        tasks = (
            (i, j, self.samples.joint_coordinates[i], self.samples.forces[j])
            for i in range(n_positions)
            for j in range(n_forces)
        )

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(str(self.model_path), self.info, activation_dynamics),
        ) as pool, tqdm(total=total_tasks, desc="RMR solves", unit="solve") as progress:
            pending = {
                pool.submit(_solve_task, task)
                for task in islice(tasks, workers * 2)
            }

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)

                for future in done:
                    i, j, muscle_activation, muscle_force, ok = future.result()
                    activations[i, j] = muscle_activation
                    muscle_forces[i, j] = muscle_force
                    success[i, j] = ok
                    progress.update(1)

                    next_task = next(tasks, None)
                    if next_task is not None:
                        pending.add(pool.submit(_solve_task, next_task))

        self.results = Results(activations, muscle_forces, success)
        return self.results
              
if __name__ == "__main__":
    from opensim import Model

    ROOT = Path(__file__).parent.parent
    model_path = ROOT / "musculoskeletal_models" / "arm26_3dof" / "arm26.osim"
    model = Model(str(model_path))

    fatigue_map_builder = FatigueMapBuilder.from_model(model, model_path=model_path)

    fatigue_map_builder.generate_position_samples(n_samples=1000000)

    one_direction = np.array([[1.0, 0.0, 0.0]])
    one_magnitude = np.array([10.0])

    #n_magnitudes = 1
    #n_directions = 3

    fatigue_map_builder.generate_force_samples(
        magnitudes=one_magnitude,
        directions=one_direction,
        #n_magnitudes=n_magnitudes,
        #n_directions=n_directions,
    )

    fatigue_map_builder.compute_activation_and_muscle_force_maps(
        workers=12,
        activation_dynamics=False
    )

    print(fatigue_map_builder.results.activations.shape)
