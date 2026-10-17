from pathlib import Path
import tqdm
import numpy as np

from .msk_data import MSKdataset, ModelInfo, OCPResult, JointStates, MuscleStates

from rmr_solver import RMRsolver, ActSquared
import os
import sys

os.environ.setdefault("SIMBODY_HOME", str(Path(sys.prefix) / "simbody"))


class FatigueModel:
    def __init__(self, dataset: MSKdataset, *, force_index: int = 0):
        self.data = dataset

        # currently only supports nearest-neighbor lookup for joint coordinates.
        # would provide a very rough estimate if endpoint force not exactly the same as the dataset force.
        self.force_index = force_index

        samples, results = dataset.samples, dataset.results
        valid = (
            results.sample_success[:, self.force_index]
            & np.isfinite(samples.joint_coordinates).all(axis=1)
            & np.isfinite(results.sample_fatigue_rates[:, self.force_index, :]).all(axis=1)
            & np.isfinite(results.sample_muscle_forces[:, self.force_index, :]).all(axis=1)
        )
        self.valid_indices = np.flatnonzero(valid)

    @classmethod
    def from_npz(cls, npz_path: Path, *, force_index: int = 0):
        return cls(MSKdataset.from_npz(npz_path), force_index=force_index)

    def evaluate_nn_batch(self, q: np.ndarray):                                                 # shape (N, n_coordinates)
        qs = self.data.samples.joint_coordinates[self.valid_indices]                            # shape (n_valid_samples, n_coordinates)
        diffs = qs[:, :, None] - q[None, :, :]                                                  # shape (n_valid_samples, n_coordinates, N)
        dists = np.linalg.norm(diffs, axis=1)                                                   # shape (n_valid_samples, N)
        nn_indices = self.valid_indices[np.argmin(dists, axis=0)]                               # shape (N,)
        rates = self.data.results.sample_fatigue_rates[nn_indices, self.force_index, :]         # shape (N, n_muscles)
        forces = self.data.results.sample_muscle_forces[nn_indices, self.force_index, :]        # shape (N, n_muscles)
        return rates, forces

    def simulate_muscle_states(self, X_joints, X_muscles, dt, N, rates=None, forces=None):
        if rates is None and forces is None:
            rates, forces = self.evaluate_nn_batch(X_joints)         # shape (N, n_muscles)


        params = self.data.fatigue_params
        recovery_rates = params.recovery / params.capacity

        for k in range(N - 1):
            previous = X_muscles[:, k]

            active = forces[k] >= params.threshold

            derivative = np.where(
                active,
                (1.0 - previous) * rates[k],
                -previous * recovery_rates,
            )

            X_muscles[:, k + 1] = np.clip(
                previous + dt * derivative,
                0.0,
                1.0,
            )           

        return X_muscles           # shape (n_muscles, N)
    

MODEL_CONFIG = {
    "arm26_3DOF": {
        "model_path": "musculoskeletal_models/arm26_3DOF/arm26_3DOF.osim",
        "dataset_npz_path": "results/arm26_3DOF/datasets/msk_dataset_p1e5_m1_d1_fx-20_fy0_fz0.npz",
        "ocp_result_npz_path": "results/arm26_3DOF/ocp/results/20261017_140446.npz",
        "ocp_validation_npz_path": "results/arm26_3DOF/ocp/validation/20261017_140446.npz",
    },
    "arm26_4DOF": {
        "model_path": "musculoskeletal_models/arm26_4DOF/arm26_4DOF.osim",
        "dataset_npz_path": "results/arm26_4DOF/datasets/msk_dataset_p1e5_m1_d1_fx-20_fy0_fz0.npz",
        "ocp_result_npz_path": "results/arm26_4DOF/ocp/results/20261017_112917.npz",
        "ocp_validation_npz_path": "results/arm26_4DOF/ocp/validation/20261017_112917.npz",
    },
    "tsm_4DOF": {   # opensimAD does not support a joint in the TSM model
        "model_path": "musculoskeletal_models/tsm_4DOF/tsm_4DOF.osim",
        "dataset_npz_path": "results/tsm_4DOF/datasets/msk_dataset_p1e5_m1_d1_fx-20_fy0_fz0.npz",
        "ocp_result_npz_path": "results/tsm_4DOF/ocp/results/20261017_112917.npz",
        "ocp_validation_npz_path": "results/tsm_4DOF/ocp/validation/20261017_112917.npz",
    },
}



def compare_muscle_state_trajectories(model_name: str, force_index: int = 0, initial_fatigue=None):
    fatigue_model = FatigueModel.from_npz(MODEL_CONFIG[model_name]["dataset_npz_path"], force_index=force_index)

    from .msk_data import OCPResult, FatigueValidationData, JointStates
    ocp_result = OCPResult.from_npz(MODEL_CONFIG[model_name]["ocp_result_npz_path"])
    validation_data = FatigueValidationData.from_npz(MODEL_CONFIG[model_name]["ocp_validation_npz_path"])

    import opensim as osim
    model = osim.Model(MODEL_CONFIG[model_name]["model_path"])
    joint_states = JointStates.from_model(model)

    if validation_data.muscle_states is None:
        validation_data.muscle_states = fatigue_model.simulate_muscle_states(
            X_joints=ocp_result.states[ocp_result.pos_indices, :].T,
            X_muscles=ocp_result.states[joint_states.size:, :].copy(),
            dt=ocp_result.time[1] - ocp_result.time[0],
            N=ocp_result.time.size,
            rates=validation_data.fatigue_rates,
            forces=validation_data.muscle_forces,
        )
        validation_data.save_npz(MODEL_CONFIG[model_name]["ocp_validation_npz_path"])

    from .visualizer import Visualizer
    visualizer = Visualizer(model=model, data=fatigue_model.data)
    return visualizer.compare_muscle_state_trajectories(
        ocp_result,
        validation_data,
    )


if __name__ == "__main__":
    model_name = "arm26_3DOF"
    compare_muscle_state_trajectories(model_name, force_index=0)
