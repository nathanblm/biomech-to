from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

import casadi as ca
import numpy as np

import opensim as osim
from .sk_kinematics import Kinematics
from .visualizer import Visualizer

@dataclass
class JointStates:
    names: tuple[str, ...]                  # len(n_coordinates)
    positions: np.ndarray                   # (n_coordinates,)
    position_limits: np.ndarray             # (n_coordinates, 2)
    velocities: np.ndarray                  # (n_coordinates,)
    accelerations: np.ndarray               # (n_coordinates,)

    @classmethod
    def from_model(cls, model):
        from .msk_mapbuilder import ModelInfo
        info = ModelInfo.from_model(model)

        return cls(
            names=info.joint_coordinate_names,
            positions=np.zeros(len(info.joint_coordinate_names)),
            position_limits=info.joint_coordinate_limits,
            velocities=np.zeros(len(info.joint_coordinate_names)),
            accelerations=np.zeros(len(info.joint_coordinate_names)),
        )

    @property
    def size(self) -> int:
        return self.positions.size + self.velocities.size

    @property
    def x(self) -> np.ndarray:
        return np.column_stack((self.positions, self.velocities)).ravel()

@dataclass
class MuscleStates:
    names: tuple[str, ...]                  # len(n_muscles)
    fatigue: np.ndarray                     # (n_muscles,)

    @classmethod
    def from_model(cls, model):
        from .msk_mapbuilder import ModelInfo
        info = ModelInfo.from_model(model)

        return cls(
            names=info.muscle_names,
            fatigue=np.zeros(len(info.muscle_names)),
        )

    @property
    def size(self) -> int:
        return len(self.names)

@dataclass
class OCPconfig:
    N: float
    dt: float
    control_lim: np.ndarray
    w_control: np.ndarray
    w_tracking: np.ndarray

    @classmethod
    def from_model(cls, model, horizon, timestep, *, control_lim=None, w_control=None, w_tracking=None):
        from .msk_mapbuilder import ModelInfo
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

@dataclass
class OCPResult:
    time: np.ndarray
    states: np.ndarray                      # (n_states, n_time_points)
    controls: np.ndarray                    # (n_controls, n_time_points - 1)
    endpoint_positions: np.ndarray          # (n_endpoint_dims, n_time_points)
    reference: np.ndarray                   # (n_endpoint_dims, n_time_points)
    objective: float

    @property
    def rms_tracking_error(self) -> float:
        error = self.endpoint_positions - self.reference
        return float(np.sqrt(np.mean(np.sum(error**2, axis=0))))


class MusculoskeletalOCP:
    def __init__(self, kinematics, dynamics, sample_fatigue_rates, reference, joint_states, muscle_states, x, u, conf, N=None, x_init=None):
        self.kinematics = kinematics
        self.dynamics = dynamics
        self.sample_fatigue_rates = sample_fatigue_rates
        self.reference = reference
        self.conf = conf

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
        sample_fatigue_rates,
        reference,
        horizon,
        timestep,
        x_init=None,
    ):
        kinematics = Kinematics(model)
        dynamics = ca.Function.load(str(dynamics_func))
        joint_states = JointStates.from_model(model)
        muscle_states = MuscleStates.from_model(model)
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

        return cls(kinematics, dynamics, sample_fatigue_rates, reference, joint_states, muscle_states, x, u, conf, x_init=x_init)

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

        endpoint_positions = np.array([endpoint_FK(solution.value(self.X[position_indices, :]))[:, k] for k in range(self.conf.N)])
        return OCPResult(
            time=np.linspace(0, self.conf.N * self.conf.dt, self.conf.N),
            states=solution.value(self.X[:self.joint_states.size, :]),
            controls=solution.value(self.U),
            endpoint_positions=endpoint_positions,
            reference=self.reference,
            objective=solution.value(self.opti.f)
        )



if __name__ == "__main__":
    root = Path(__file__).parent.parent

    model_name = "arm26_3DOF"
    model_dir = root / "musculoskeletal_models" / f"{model_name}"
    results_dir = root / "results" / f"{model_name}"
    archive_name = "p1e5_m1_d1_fx-20_fy0_fz0.npz"
    model = osim.Model(str(model_dir / f"{model_name}.osim"))

    horizon = 100
    timestep = 0.1
    time = np.linspace(0.0, horizon * timestep, horizon)

    reference = np.vstack((
        0.4 * np.ones(100),
        0.6 * np.ones(100),
        0.2 + 0.1 * np.sin(2 * np.pi * time / 2.0),
    ))

    archive = np.load( results_dir / archive_name )
    sample_fatigue_rates = archive["sample_fatigue_rates"]

    ocp = MusculoskeletalOCP.from_model(
        model=model,
        dynamics_func=model_dir / "fd_dynamics" / "F.casadi",
        sample_fatigue_rates=sample_fatigue_rates,
        reference=reference,
        horizon=100,
        timestep=0.1,
    )

    result = ocp.solve()


    visualizer = Visualizer(model)
    filename = f"{model.getName()}_{datetime.now():%Y%m%d_%H%M%S}.sto"
    motion_file = visualizer.save_motion_sto(
        root / "results" / f"{model.getName()}/motions/{filename}",
        result.time,
        result.states[:ocp.joint_states.size, :],
    )

