# Application models

These OpenSim models are configured for the map builder, RMR solver, endpoint
kinematics, and (where noted) the CasADi optimal-control application.

| Model | Free coordinates | Muscles | Runtime force | `endpoint` body / location (m) | `F.casadi` |
| --- | ---: | ---: | --- | --- | --- |
| [arm26 3-DOF](arm26_3DOF/README.md) | 3 | 6 Millard | `endpoint_external_force` | `r_ulna_radius_hand`: `(0, -0.23559, 0)` | Included |
| [arm26 4-DOF](arm26_4DOF/README.md) | 4 | 6 Millard | `endpoint_external_force` | `r_ulna_radius_hand`: `(0, -0.23559, 0)` | Included |
| [TSM 4-DOF](tsm_4DOF/README.md) | 4 | 33 Millard | `hand_external_force` | `hand`: `(0, 0, 0)` | Not included |

## Application requirements

- The current map-builder/RMR integration expects `Millard2012EquilibriumMuscle`
  muscles, an `endpoint` marker at the locations above, and exactly one
  `PrescribedForce`. It supplies one runtime six-component wrench (force plus
  zero torque).
- Unlocked model coordinates define map samples and application state
  dimensions. Each model's README lists its free and locked coordinates.
- Endpoint kinematics supports the `CustomJoint`, `WeldJoint`, and
  `ScapulothoracicJoint` types used here, with `LinearFunction`, `Constant`,
  and `MultiplierFunction` transform functions.
- The optimal-control application also loads a model-specific
  `fd_dynamics/F.casadi`. It is present for both arm26 models; generate it for
  TSM before using that model with the optimal-control application.
