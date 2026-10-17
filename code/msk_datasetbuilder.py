from datetime import datetime
from pathlib import Path
import numpy as np
import opensim as osim
from tqdm import tqdm
from typing import Optional, Union
import multiprocessing as mp
from itertools import islice
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from rmr_solver import RMRsolver, ActSquared
from .visualizer import Visualizer
from .msk_data import ModelInfo, OCPResult, SampleInputs, SampleResults, FatigueParameters, \
      MSKdataset, RMRconfig, FatigueValidationData

osim.Logger.setLevelString("Error")


_worker_context = None
_worker_output = None


def _init_worker(model_path, model_info, config, activation_dynamics=False):
    global _worker_context, _worker_output

    osim.Logger.setLevelString("Error")

    model = osim.Model(str(model_path))
    _worker_context = RMRWorkerContext(
        model=model,
        model_info=model_info,
        config=config,
        activation_dynamics=activation_dynamics,
    )

def _solve_task(task):
    return _worker_context.solve_task(task)

def _scientific_int(value: int) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"

def _force_label(force):
    x, y, z = force
    return f"fx{x:g}_fy{y:g}_fz{z:g}"


class RMRWorkerContext:
    def __init__(self, model, model_info, config, activation_dynamics=False):
        self.model = model
        self.model_info = model_info
        self.config = config

        self.solver = RMRsolver(
            model=model,
            visualize=False,
            constrainActDyn=activation_dynamics,
            prescribedForceIndex=self.model_info.prescribed_force_indices,
        )

        self.solver.setObjective(ActSquared(self.config.weights))

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


class MusculoskeletalDatasetBuilder:
    def __init__(self, model, info: ModelInfo, config: RMRconfig, fatique_params: FatigueParameters, *, model_path=None):
        self.model = model
        self.info = info
        self.config = config
        self.fatigue_params = fatique_params
        
        self.model_path = Path(model_path).resolve() if model_path is not None else None
        self.state = model.initSystem()
        
        self.inputs = SampleInputs.empty(n_joint_coordinates=len(info.joint_coordinate_names))
        self.results = SampleResults.empty()
        self.validation = FatigueValidationData.empty(n_muscles=len(info.muscle_names))

    @classmethod
    def from_model(cls, model, *, model_path=None, agonists=(), antagonists=()):
        info = ModelInfo.from_model(model, agonists=agonists, antagonists=antagonists)
        config = RMRconfig.from_model_info(info)
        fatigue_params = FatigueParameters.from_max_forces(info.muscle_max_forces)
        builder = cls(model, info, config, fatigue_params, model_path=model_path)
        return builder

    @classmethod
    def from_npz(cls, model, npz_path, *, model_path=None):
        builder = cls.from_model(model, model_path=model_path)

        dataset = MSKdataset.from_npz(npz_path)
        builder.info = dataset.info
        builder.inputs = dataset.samples
        builder.results = dataset.results
        builder.fatigue_params = dataset.fatigue_params

        return builder

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

        self.inputs.joint_coordinates = joint_samples
        self.inputs.cartesian_coordinates = cartesian_samples

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

            self.inputs.force_magnitudes = magnitudes
            self.inputs.force_directions = directions
            self.inputs.forces = (magnitudes[:, None, None] * directions[None, :, :]).reshape(-1, 3)
            
    def evaluate_sample_activations_and_muscle_forces(self, *, workers=4, activation_dynamics=False):

        n_positions = len(self.inputs.joint_coordinates)
        n_forces = len(self.inputs.forces)
        n_muscles = len(self.info.muscle_names)
        total_tasks = n_positions * n_forces

        activations = np.full((n_positions, n_forces, n_muscles), np.nan)
        muscle_forces = np.full((n_positions, n_forces, n_muscles), np.nan)
        success = np.zeros((n_positions, n_forces), dtype=bool)

        tasks = (
            (i, j, self.inputs.joint_coordinates[i], self.inputs.forces[j])
            for i in range(n_positions)
            for j in range(n_forces)
        )

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(str(self.model_path), self.info, self.config, activation_dynamics),
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

        self.results.sample_activations = activations
        self.results.sample_muscle_forces = muscle_forces
        self.results.sample_success = success

        return self.results

    def compute_sample_fatigue_rates(self):
        rates = self.results.sample_muscle_forces / self.fatigue_params.capacity[None, None, :]
        self.results.sample_fatigue_rates = rates
        return rates

    def compute_sample_fatigue_dominance(self):
        agonists = self.info.agonist_muscles
        antagonists = self.info.antagonist_muscles
        if not agonists or not antagonists:
            raise ValueError("ModelInfo must define both agonist_muscles and antagonist_muscles.")
        agonist_indices = [self.info.muscle_names.index(name) for name in agonists]
        antagonist_indices = [self.info.muscle_names.index(name) for name in antagonists]

        agonist_fatigue = self.results.sample_fatigue_rates[:, :, agonist_indices].sum(axis=2)
        antagonist_fatigue = self.results.sample_fatigue_rates[:, :, antagonist_indices].sum(axis=2)

        numerator = agonist_fatigue - antagonist_fatigue
        denominator = agonist_fatigue + antagonist_fatigue

        sample_fatigue_dominance = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator != 0
        )

        self.results.sample_fatigue_dominance = sample_fatigue_dominance
        return sample_fatigue_dominance

    def save_msk_dataset(self, *, results_dir=None, tag=None):
        if results_dir is None:
            ROOT = Path(__file__).parent.parent
            results_dir = ROOT / "results" / self.model.getName() / "datasets"

        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        if not tag and self.inputs.forces.shape[0] == 1:
            tag = _force_label(self.inputs.forces[0])

        filename = (
            f"msk_dataset_p{_scientific_int(self.inputs.joint_coordinates.shape[0])}"
            f"_m{self.inputs.force_magnitudes.shape[0]}"
            f"_d{self.inputs.force_directions.shape[0]}"
            f"{'_' + tag if tag else ''}.npz"
        )

        output_path = results_dir / filename
        dataset = MSKdataset(
            info=self.info,
            samples=self.inputs,
            results=self.results,
            fatigue_params=self.fatigue_params,
        )
        return dataset.save_npz(output_path)

    def create_visualizer(self, npz_path, *, endpoint_marker="endpoint"):
        saved = type(self).from_npz(self.model, npz_path, model_path=self.model_path)
        data = MSKdataset(
            info=saved.info,
            samples=saved.inputs,
            results=saved.results,
            fatigue_params=saved.fatigue_params,
        )
        return Visualizer(
            model=self.model,
            data=data,
            endpoint_marker=endpoint_marker,
        )   

    def generate_validation_data(self, ocp_result, *, force_index=0, activation_dynamics=True, filter_succes=True):

        solver = RMRsolver(
            self.model, 
            visualize=True,
            constrainActDyn=activation_dynamics,
            prescribedForceIndex=self.info.prescribed_force_indices,
            )     

        solver.setObjective(ActSquared(self.config.weights))

        positions = ocp_result.states[ocp_result.pos_indices, :-1].T
        speeds = ocp_result.states[ocp_result.pos_indices + 1, :-1].T
        accelerations = ocp_result.controls.T
        N = positions.shape[0]

        f = self.inputs.forces[force_index]
        f_N = np.broadcast_to(f, (N, f.size))

        val_muscle_forces = np.empty((N, len(self.info.muscle_names)))
        val_rates = np.empty((N, len(self.info.muscle_names)))
        val_succes = np.empty(N, dtype=bool)

        for i in range(N):
            time = i * ocp_result.time[1]  # assumes uniform time steps
            _, _, val_muscle_forces[i], info = solver.solve(
                time=time,
                position=positions[i],
                speed=speeds[i],
                acceleration=accelerations[i],
                values_prescribed_forces=np.r_[f_N[i], np.zeros(3)],
            )

            val_succes[i] = info["success"]

        if filter_succes:
            val_muscle_forces = val_muscle_forces[val_succes]

        rates = val_muscle_forces / self.fatigue_params.capacity[None, :]

        self.validation = FatigueValidationData(
            tag=ocp_result.tag,
            force_index=force_index,
            time=ocp_result.time,
            joint_positions=ocp_result.states[ocp_result.pos_indices, :],
            endpoint_forces=np.broadcast_to(f, (N, f.size)),
            muscle_forces=val_muscle_forces,
            fatigue_rates=rates,
            success=val_succes
        )

        return self.validation

    def save_validation_data(self, *, dir=None):
        if dir is None:
            ROOT = Path(__file__).parent.parent
            dir = ROOT / "results" / self.model.getName() / "ocp" / "validation"

        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)


        filename = f"{self.validation.tag}.npz"
        output_path = dir / filename

        return self.validation.save_npz(output_path)


MODEL_CONFIG = {
    "arm26_3DOF": {
        "model_path": "musculoskeletal_models/arm26_3DOF/arm26_3DOF.osim",
        "agonists": ["TRImed", "TRIlat", "TRIlong"],
        "antagonists": ["BICshort", "BIClong", "BRA"],
        "reference_marker": "r_acromion",
    },
    "arm26_4DOF": {
        "model_path": "musculoskeletal_models/arm26_4DOF/arm26_4DOF.osim",
        "agonists": ["TRImed", "TRIlat", "TRIlong"],
        "antagonists": ["BICshort", "BIClong", "BRA"],
        "reference_marker": "r_acromion",
    },
    "tsm_4DOF": {
        "model_path": "musculoskeletal_models/tsm_4DOF/tsm_4DOF.osim",
        "agonists": ["BIC_long", "BIC_brevis"],
        "antagonists": ["TRIlong"],
        "reference_marker": "ac",
    },
}


def build_msk_dataset(model_name: str, *, n_samples=100000, n_magnitudes=1, n_directions=1, magnitude_min=20.0, magnitude_max=30.0, magnitudes=None, directions=None):
    config = MODEL_CONFIG.get(model_name)

    model_path = Path(config["model_path"])

    # Reuse the explicitly configured single-force dataset if it is already saved.
    if magnitudes is not None and directions is not None:
        configured_magnitudes = np.asarray(magnitudes, dtype=float).reshape(-1)
        configured_directions = np.asarray(directions, dtype=float).reshape(-1, 3)
        if len(configured_magnitudes) == len(configured_directions) == 1:
            direction = configured_directions[0]
            force = configured_magnitudes[0] * direction / np.linalg.norm(direction)
            force_tag = _force_label(force)
            dataset_path = (
                Path(__file__).resolve().parent.parent
                / "results" / model_name / "datasets"
                / f"msk_dataset_p{_scientific_int(n_samples)}_m1_d1_{force_tag}.npz"
            )
            if dataset_path.exists():
                MSKdataset.from_npz(dataset_path)
                print(f"Loading existing dataset: {dataset_path}")
                return dataset_path

    model = osim.Model(str(model_path))

    msk_builder = MusculoskeletalDatasetBuilder.from_model(
        model,
        model_path=model_path,
        agonists=config["agonists"],
        antagonists=config["antagonists"],
    )

    msk_builder.generate_position_samples(n_samples=n_samples)
    msk_builder.generate_force_samples(
        magnitudes=magnitudes,
        directions=directions,
        n_magnitudes=n_magnitudes,
        n_directions=n_directions,
    )
    msk_builder.evaluate_sample_activations_and_muscle_forces(
        workers=mp.cpu_count(),
        activation_dynamics=False,
    )
    msk_builder.compute_sample_fatigue_rates()
    msk_builder.compute_sample_fatigue_dominance()
    return msk_builder.save_msk_dataset()

def build_validation_dataset(model_name: str, dataset_npz_path: str, ocp_result_npz_path, *, force_index=0, activation_dynamics=True, filter_succes=True):
    config = MODEL_CONFIG.get(model_name)

    model_path = Path(config["model_path"])
    model = osim.Model(str(model_path))

    msk_builder = MusculoskeletalDatasetBuilder.from_npz(
        model,
        npz_path=dataset_npz_path,
        model_path=model_path,
    )

    from .msk_data import OCPResult
    ocp_result = OCPResult.from_npz(ocp_result_npz_path)

    msk_builder.generate_validation_data(
        ocp_result,
        force_index=force_index,
        activation_dynamics=activation_dynamics,
        filter_succes=filter_succes,
    )
    msk_builder.save_validation_data()

def dataset_demo(model_name: str, npz_path: str, *, endpoint_marker="endpoint"):
    config = MODEL_CONFIG.get(model_name)

    model_path = Path(config["model_path"])
    model = osim.Model(str(model_path))
    ref_marker = config["reference_marker"]

    msk_builder = MusculoskeletalDatasetBuilder.from_npz(
        model,
        npz_path=npz_path,
        model_path=model_path,
    )

    visualizer = msk_builder.create_visualizer(
        npz_path=npz_path,
        endpoint_marker=endpoint_marker,
    )

    # Pick a configured agonist by name; it must be present in the saved model data.
    muscle_name = config["agonists"][0]
    if muscle_name not in visualizer.muscle_names:
        raise ValueError(
            f"Unknown muscle {muscle_name!r}. Available muscles: {visualizer.muscle_names}"
        )
    muscle = muscle_name
    for field in ("activation", "muscle_force"):
        visualizer.show(
            field,
            muscle=muscle,
            force_index=0,
            backend="matplotlib",
            space="cartesian",
            block=False,
        )


    visualizer.show(
        "dominance",
        force_index=0,
        backend="simbody",
        space="cartesian",
        slice_axis="y",
        slice_value=-0.1,
        reference_marker=ref_marker,
        tolerance=0.2,
        opacity=0.9,
        radius=0.008,
        #max_points=5000,
    )


if __name__ == "__main__":
    #dataset_demo("tsm_4DOF", "results/tsm_4DOF/datasets/msk_dataset_p1e4_m1_d1_fx-20_fy0_fz0.npz")
    
    #build_msk_dataset(
    #    "arm26_3DOF", 
    #    n_samples=100, 
    #    n_magnitudes=1, 
    #    n_directions=1, 
    #    magnitude_min=20.0, 
    #    magnitude_max=20.0,
    #    magnitudes=None,
    #    directions=None
    #)

    build_validation_dataset(
        "arm26_3DOF",
        dataset_npz_path="results/arm26_3DOF/datasets/msk_dataset_p1e5_m1_d1_fx-20_fy0_fz0.npz",
        ocp_result_npz_path="results/arm26_3DOF/ocp/results/20261017_140446.npz",
        force_index=0,
        activation_dynamics=True,
        filter_succes=True,
    )
