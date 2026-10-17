from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import casadi as ca
import numpy as np

import opensim as osim
from .sk_kinematics import Kinematics
from .visualizer import Visualizer
from .msk_fatiguemodel import FatigueModel
from .msk_data import ModelInfo, OCPResult

from .msk_data import JointStates, MuscleStates

@dataclass
class OCPconfig:
    N: int
    dt: float
    control_lim: np.ndarray
    w_control: np.ndarray
    w_tracking: np.ndarray

    @classmethod
    def from_model(cls, model, horizon, timestep, *, control_lim=None, w_control=None, w_tracking=None):
        info = ModelInfo.from_model(model)

        if control_lim is None:
            control_lim = np.full(len(info.joint_coordinate_names), np.inf)
        if w_control is None:
            w_control = np.full(len(info.joint_coordinate_names), 1e-3)
        if w_tracking is None:
            w_tracking = np.full(3, 1e4)

        return cls(
            N=horizon,
            dt=timestep,
            control_lim=control_lim,
            w_control=w_control,
            w_tracking=w_tracking
        )

class MusculoskeletalOCP:
    def __init__(self, kinematics, dynamics, fatigue, reference, joint_states, muscle_states, x, u, conf, N=None, x_init=None):
        self.kinematics = kinematics
        self.dynamics = dynamics
        self.fatigue = fatigue
        self.reference = reference
        self.conf = conf
        self.ocp_result = OCPResult.empty()

        self.muscle_states = muscle_states
        self.joint_states = joint_states

        self.x = x                          # [q1, v1, q2, v2, ..., qn, vn, f1, f2, ..., fm]
        self.u = u                          # [a1, a2, ..., an]
        self.x_init = x_init

        self.opti = ca.Opti()

        self.X = self.opti.variable(self.x.size, self.conf.N)
        self.U = self.opti.variable(self.u.size, self.conf.N - 1)

    @classmethod
    def from_model(
        cls,
        model,
        dynamics_func: Path,
        fatigue: FatigueModel,
        reference,
        horizon,
        timestep,
        x_init=None,
    ):
        kinematics = Kinematics(model)
        joint_states = JointStates.from_model(model)
        muscle_states = MuscleStates.from_model(model)
        dynamics_func = Path(dynamics_func)
        dynamics = ca.Function.load(str(dynamics_func))
        conf = OCPconfig.from_model(model, horizon, timestep)

        x = np.concatenate((joint_states.x, muscle_states.fatigue))
        u = joint_states.accelerations

        if x_init is None:
            positions, _ = kinematics.inverse_kinematics(reference[:, 0])
            velocities = np.zeros_like(positions)
            muscle_state = np.zeros_like(muscle_states.fatigue)
            joint_state = np.column_stack((positions, velocities)).ravel()
            x_init = np.concatenate((joint_state, muscle_state))
            print(f"Initial state not provided. Using default: {x_init}")

        return cls(kinematics, dynamics, fatigue, reference, joint_states, muscle_states, x, u, conf, x_init=x_init)

    def solve(self):
        endpoint_FK = self.kinematics.endpoint_function
        n = self.joint_states.x.size
        m = self.muscle_states.fatigue.size
        position_indices = np.arange(0, n, 2)

        # intial state constraint
        self.opti.subject_to(self.X[:, 0] == self.x_init)

        # joint limits
        for i, row in enumerate(position_indices):
            lower, upper = self.joint_states.position_limits[i]
            self.opti.subject_to(
                self.opti.bounded(lower, self.X[row, :], upper)
            )

        # control limits
        for i in range(len(self.u)):
            lower = -self.conf.control_lim[i]
            upper = self.conf.control_lim[i]
            self.opti.subject_to(self.opti.bounded(lower, self.U[i, :], upper))

        # cost function
        cost = 0
        for k in range(self.conf.N):
            pos_error = endpoint_FK(self.X[position_indices, k]) - self.reference[:, k]
            cost += ca.sumsqr(pos_error)

        # dynamics rk4 integration
        for k in range(self.conf.N - 1):
            xk = self.X[:n, k]
            uk = self.U[:, k]

            f = lambda state: self.dynamics(ca.vertcat(state, uk))
            k1 = f(xk)
            k2 = f(xk + self.conf.dt * k1 / 2)
            k3 = f(xk + self.conf.dt * k2 / 2)
            k4 = f(xk + self.conf.dt * k3)

            self.opti.subject_to(
                self.X[:n, k + 1]
                == xk + self.conf.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            )   

        # TODO: fatigue dynamics

        
        self.opti.minimize(cost)
        self.opti.solver("ipopt", {"print_time": False}, {"print_level": 5})
        solution = self.opti.solve()

        endpoint_positions = np.asarray(endpoint_FK(solution.value(self.X[position_indices, :])))
    
        self.ocp_result = OCPResult(
            time=np.arange(self.conf.N) * self.conf.dt,
            states=solution.value(self.X),
            pos_indices=position_indices,
            controls=solution.value(self.U),
            endpoint_positions=endpoint_positions,
            reference=self.reference,
            objective=solution.value(self.opti.f)
        )

        # simulate muscle states because not yet implemented in the OCP
        self.ocp_result.states[n:, :] = self.fatigue.simulate_muscle_states(
            self.ocp_result.states[position_indices, :],
            self.ocp_result.states[n:, :],
            dt=self.conf.dt,
            N=self.conf.N,
        )

        return self.ocp_result

    def save_result(self, model_name: str, *, dir=None, tag=None):
        if dir is None:
            ROOT = Path(__file__).parent.parent
            dir = ROOT / "results" / model_name / "ocp" / "results"
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        if tag is None:
            tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.ocp_result.tag = tag

        filename = dir / f"{tag}.npz"
        return self.ocp_result.save_npz(filename)


MODEL_CONFIG = {
    "arm26_3DOF": {
        "model_path": "musculoskeletal_models/arm26_3DOF/arm26_3DOF.osim",
        "dataset_npz_path": "results/arm26_3DOF/datasets/msk_dataset_p1e5_m1_d1_fx-20_fy0_fz0.npz",
        "dynamics_func_path": "musculoskeletal_models/arm26_3DOF/fd_dynamics/F.casadi",
        "ocp_result_npz_path": "results/arm26_3DOF/ocp/results/20261017_140446.npz",
    },
    "arm26_4DOF": {
        "model_path": "musculoskeletal_models/arm26_4DOF/arm26_4DOF.osim",
        "dataset_npz_path": "results/arm26_4DOF/datasets/msk_dataset_p1e5_m1_d1_fx-20_fy0_fz0.npz",
        "dynamics_func_path": "musculoskeletal_models/arm26_4DOF/fd_dynamics/F.casadi",
        "ocp_result_npz_path": "results/arm26_4DOF/ocp/results/20261017_112917.npz",

    },
    "tsm_4DOF": {   # opensimAD does not support a joint in the TSM model
        "model_path": "musculoskeletal_models/tsm_4DOF/tsm_4DOF.osim",
        "dataset_npz_path": "results/tsm_4DOF/datasets/msk_dataset_p1e5_m1_d1_fx-20_fy0_fz0.npz",
    },
}


def solve_ocp(model_name: str, force_index: int = 0, reference=None, horizon=100, timestep=0.1, save_motion=False):
    fatigue_model = FatigueModel.from_npz(MODEL_CONFIG[model_name]["dataset_npz_path"], force_index=force_index)

    import opensim as osim
    model = osim.Model(MODEL_CONFIG[model_name]["model_path"])

    if reference is None:
        time = np.arange(horizon) * timestep
        reference = np.vstack((
            0.3 * np.ones(horizon),
            0.6 * np.ones(horizon),
            0.2 + 0.3 * np.sin(2.0 * np.pi * time / 10.0),
        ))

    ocp = MusculoskeletalOCP.from_model(
        model=model,
        dynamics_func=MODEL_CONFIG[model_name]["dynamics_func_path"],
        fatigue=fatigue_model,
        reference=reference,
        horizon=horizon,
        timestep=timestep,
    )

    result = ocp.solve()
    ocp.save_result(model_name=model_name)

    if save_motion:
        visualizer = Visualizer(model)
        root = Path(__file__).parent.parent
        filename = f"{model.getName()}_{datetime.now():%Y%m%d_%H%M%S}.sto"
        motion_file = visualizer.save_motion_sto(
            root / "results" / f"{model.getName()}/motions/{filename}",
            result.time,
            result.states[:ocp.joint_states.size, :],
        )

    
    return result

def visualize_ocp_result(model_name: str, ocp_result_path: str, *, block=True):
    import opensim as osim
    model = osim.Model(MODEL_CONFIG[model_name]["model_path"])
    ocp_result = OCPResult.from_npz(ocp_result_path)

    fatigue_model = FatigueModel.from_npz(MODEL_CONFIG[model_name]["dataset_npz_path"])
    visualizer = Visualizer(model, data=fatigue_model.data)
    visualizer.visualize_ocp_result(
        ocp_result,
        slice_axis="y",
        slice_value=-0.25,
        reference_marker="r_acromion",
        tolerance=0.05,
        opacity=0.9,
        radius=0.008,
        block=block,
    )


if __name__ == "__main__":

    model_name = "arm26_3DOF"

    solve_ocp(model_name, force_index=0, horizon=100, timestep=0.1, save_motion=True)

    #ocp_result_path = MODEL_CONFIG[model_name]["ocp_result_npz_path"]
    #visualize_ocp_result(model_name, ocp_result_path)
