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

    def evaluate_with_rmr_batch(self, 
                                model: str, 
                                x: np.ndarray, 
                                u: np.ndarray, 
                                dt: float,
                                info: ModelInfo, 
                                joint_states: JointStates,
                                muscle_states: MuscleStates,
                                *, 
                                activation_dyamics=True
                            ):
        
        solver = RMRsolver(
            model, 
            visualize=False,
            constrainActDyn=activation_dyamics,
            prescribedForceIndex=info.prescribed_force_indices,
            )

        muscle_weights = np.ones(len(info.muscle_names))
        reserve_actuator_weights = np.ones(len(info.actuator_names) - len(info.muscle_names))
        weights = np.concatenate([muscle_weights, reserve_actuator_weights])

        solver.setObjective(ActSquared(weights))
        
        positions = x[:-1, joint_states.pos_indices]
        speeds = x[:-1, joint_states.pos_indices + 1]
        accelerations = u
        N = positions.shape[0]

        # we assume force is equal to the constant force from the dataset
        f = self.dataset.samples.forces[self.force_index]
        f_N = np.broadcast_to(f, (x.shape[0], f.size))

        
        muscle_forces = np.empty((N, len(info.muscle_names)))
        succes = np.empty(N, dtype=bool)

        for i in range(N):
            _, _, forces_i, solve_info = solver.solve(
                time=i * dt,
                position=positions[i],
                speed=speeds[i],
                acceleration=accelerations[i],
                values_prescribed_forces=np.r_[f_N[i], np.zeros(3)]
            )

            muscle_forces[i] = forces_i
            succes[i] = solve_info["success"]

        mf_valid = muscle_forces[succes]

        rates = mf_valid / self.data.fatigue_params.capacity[None, :]
        return rates, muscle_forces

    def evaluate_with_rmr_batch_(self, model, info, ocp_result, *, activation_dyamics=True):
        solver = RMRsolver(
            model, 
            visualize=True,
            constrainActDyn=activation_dyamics,
            prescribedForceIndex=info.prescribed_force_indices,
            )

        muscle_weights = np.ones(len(info.muscle_names))
        reserve_actuator_weights = np.ones(len(info.actuator_names) - len(info.muscle_names))
        weights = np.concatenate([muscle_weights, reserve_actuator_weights])

        solver.setObjective(ActSquared(weights))

        positions = ocp_result.states[ocp_result.pos_indices, :-1].T
        speeds = ocp_result.states[ocp_result.pos_indices + 1, :-1].T
        accelerations = ocp_result.controls.T
        N = positions.shape[0]

        # we assume force is equal to the constant force from the dataset
        f = self.data.samples.forces[self.force_index]
        f_N = np.broadcast_to(f, (ocp_result.states.shape[1], f.size))

        muscle_forces = np.empty((N, len(info.muscle_names)))
        succes = np.empty(N, dtype=bool)

        for i in range(N):
            _, _, muscle_forces[i], info = solver.solve(
                time=i * (ocp_result.time[1] - ocp_result.time[0]),
                position=positions[i],
                speed=speeds[i],
                acceleration=accelerations[i],
                values_prescribed_forces=np.r_[f_N[i], np.zeros(3)]
            )

            succes[i] = info["success"]

        mf_valid = muscle_forces[succes]
        rates = mf_valid / self.data.fatigue_params.capacity[None, :]

        return rates, muscle_forces
    

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

    def plot_muscle_state_difference(
        self,
        time,
        muscle_states_approx,
        muscle_states_rmr,
        *,
        muscle_names=None,
        show=True,
    ):
        """Plot RMR minus approximate muscle states over time.

        State arrays may be shaped ``(n_muscles, n_time)`` or
        ``(n_time, n_muscles)``. The returned difference uses the former shape.
        """
        import matplotlib.pyplot as plt

        time = np.asarray(time, dtype=float)
        approx = np.asarray(muscle_states_approx, dtype=float)
        rmr = np.asarray(muscle_states_rmr, dtype=float)
        n_muscles = len(self.data.info.muscle_names)

        if approx.ndim != 2 or rmr.ndim != 2:
            raise ValueError("Muscle state arrays must be 2D.")
        if approx.shape != rmr.shape:
            raise ValueError(
                f"Muscle state arrays must have the same shape; got {approx.shape} and {rmr.shape}."
            )
        if approx.shape[0] != n_muscles and approx.shape[1] == n_muscles:
            approx, rmr = approx.T, rmr.T
        if approx.shape[0] != n_muscles or approx.shape[1] != len(time):
            raise ValueError(
                f"Expected states shaped ({n_muscles}, {len(time)}), got {approx.shape}."
            )
        if not np.isfinite(time).all() or not np.isfinite(approx).all() or not np.isfinite(rmr).all():
            raise ValueError("Time and muscle states must contain only finite values.")

        difference = rmr - approx
        names = tuple(muscle_names) if muscle_names is not None else self.data.info.muscle_names
        if len(names) != n_muscles:
            raise ValueError(f"muscle_names must contain {n_muscles} names.")

        fig, ax = plt.subplots(figsize=(8, 5))
        for index, name in enumerate(names):
            ax.plot(time, difference[index], label=name)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("RMR - approximate fatigue state")
        ax.set_title("Muscle-state difference")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize="small")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax, difference


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    model_name = "arm26_3DOF"
    dataset_name = "msk_dataset_p1e5_m1_d1_fx-20_fy0_fz0.npz"
    trajectory_name = "results_20261017_112917.npz"
    force_index = 0
    initial_fatigue = None

    dataset_path = root / "results" / model_name / "datasets" / dataset_name
    trajectory_path = root / "results" / model_name / "ocp" / trajectory_name

    fatigue = FatigueModel.from_npz(dataset_path, force_index=force_index)
    ocp_result = OCPResult.from_npz(trajectory_path)
    muscle_states_approx = fatigue.eval_muscle_state_trajectory(ocp_result, initial_fatigue=initial_fatigue)


    import opensim as osim
    model = osim.Model(str(root / "musculoskeletal_models" / model_name / f"{model_name}.osim"))

    rates, forces = fatigue.evaluate_with_rmr_batch_(
        model=model,
        info=fatigue.data.info,
        ocp_result=ocp_result,
        activation_dyamics=True
    )

    joint_states = JointStates.from_model(model)
    model_states = MuscleStates.from_model(model)

    muscle_states_rmr = fatigue.simulate_muscle_states(
        X_joints=ocp_result.states[joint_states.size:, :],
        X_muscles=ocp_result.states[:joint_states.size, :],
        dt=ocp_result.time[1] - ocp_result.time[0],
        N=ocp_result.states.shape[1],
        rates=rates,
        forces=forces
    )

    fatigue.plot_muscle_state_difference(
        time=ocp_result.time,
        muscle_states_approx=muscle_states_approx,
        muscle_states_rmr=muscle_states_rmr,
    )

