from pathlib import Path

import numpy as np

from .msk_data import MSKdataset, OCPResult


class FatigueModel:
    def __init__(self, dataset: MSKdataset, *, force_index: int = 0):
        self.data = dataset
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

    def evaluate_nn(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return rates and forces, each (n_muscles,), at the nearest valid sample."""
        q = np.asarray(q, dtype=float)
        qs = self.data.samples.joint_coordinates[self.valid_indices]
        index = self.valid_indices[np.argmin(np.sum((qs - q) ** 2, axis=1))]
        return (
            self.data.results.sample_fatigue_rates[index, self.force_index, :],
            self.data.results.sample_muscle_forces[index, self.force_index, :],
        )

    def eval_muscle_state_trajectory(
        self, ocp_result: OCPResult, *, initial_fatigue=None,
    ) -> np.ndarray:
        n_time = len(ocp_result.time)
        n_muscles = len(self.data.info.muscle_names)
        params = self.data.fatigue_params

        initial = np.zeros(n_muscles) if initial_fatigue is None else np.asarray(initial_fatigue, dtype=float)
        fatigue = np.empty((n_muscles, n_time))
        fatigue[:, 0] = initial
        recovery_rates = params.recovery / params.capacity

        for i, dt in enumerate(np.diff(ocp_result.time)):
            q = ocp_result.states[ocp_result.pos_indices, i]
            rates, muscle_forces = self.evaluate_nn(q)
            previous = fatigue[:, i]
            active = muscle_forces >= params.threshold
            derivative = np.where(
                active,
                (1 - previous) * rates,
                -previous * recovery_rates,
            )
            fatigue[:, i + 1] = np.clip(previous + dt * derivative, 0.0, 1.0)

        return fatigue

    def evaluate_nn_batch(self, q: np.ndarray):                                                 # shape (N, n_coordinates)
        qs = self.data.samples.joint_coordinates[self.valid_indices]                            # shape (n_valid_samples, n_coordinates)
        diffs = qs[:, :, None] - q[None, :, :]                                                  # shape (n_valid_samples, n_coordinates, N)
        dists = np.linalg.norm(diffs, axis=1)                                                   # shape (n_valid_samples, N)
        nn_indices = self.valid_indices[np.argmin(dists, axis=0)]                               # shape (N,)
        rates = self.data.results.sample_fatigue_rates[nn_indices, self.force_index, :]         # shape (N, n_muscles)
        forces = self.data.results.sample_muscle_forces[nn_indices, self.force_index, :]        # shape (N, n_muscles)
        return rates, forces

    def simulate_muscle_states(self, X_joints, X_muscles, dt, N):

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
    

    def plot_fatigue_trajectory(self, ocp_result: OCPResult, *, initial_fatigue=None):
        import matplotlib.pyplot as plt

        fatigue = self.eval_muscle_state_trajectory(ocp_result, initial_fatigue=initial_fatigue)
        plt.figure()
        plt.plot(ocp_result.time, fatigue.T)
        plt.xlabel("Time (s)")
        plt.ylabel("Fatigue State")
        plt.title("Muscle Fatigue Trajectory")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    model_name = "arm26_4DOF"
    dataset_name = "msk_dataset_p1e4_m1_d1_fx-20_fy0_fz0.npz"
    trajectory_name = "results_20261016_130651.npz"
    force_index = 0
    initial_fatigue = None

    dataset_path = root / "results" / model_name / "datasets" / dataset_name
    trajectory_path = root / "results" / model_name / "ocp" / trajectory_name

    fatigue = FatigueModel.from_npz(dataset_path, force_index=force_index)
    trajectory = OCPResult.from_npz(trajectory_path)
    states = fatigue.eval_muscle_state_trajectory(trajectory, initial_fatigue=initial_fatigue)
    print(f"Evaluated fatigue states shape: {states.shape}")

    fatigue.plot_fatigue_trajectory(trajectory, initial_fatigue=initial_fatigue)
