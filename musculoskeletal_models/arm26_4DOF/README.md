# arm26 4-DOF model

`arm26_4DOF.osim` extends the reduced arm26 model for shoulder axial rotation
and RMR optimization:

- Six `Millard2012EquilibriumMuscle` muscles and four free coordinates:
  `r_shoulder_elev`, `r_shoulder_plane`, `r_shoulder_rot`, and `r_elbow_flex`.
- The axial-rotation coordinate is `r_shoulder_rot`; it rotates about the
  shoulder frame's Y axis and has range `[-π, π]` radians.
- A fixed `0.8 m` upward base offset and reserve actuators for all three
  shoulder coordinates. The axial-rotation reserve actuator keeps the
  acceleration constraint actuated without dedicated muscles for that DOF.
- Added `endpoint_external_force` at the wrist, with ground-frame force
  components and zero applied torque. Its application point uses wrist-marker
  coordinates relative to the forearm body's elbow origin. It is ForceSet
  index `9` in this model; code should select it by name where possible.
- Added the `endpoint` marker on `/bodyset/r_ulna_radius_hand` at local
  `(0, -0.23559, 0) m`.
- Hand and finger meshes are disabled; forearm geometry remains visible.

Inverse dynamics use [antoinefalisse/opensimAD](https://github.com/antoinefalisse/opensimAD).
Forward dynamics use [itbellix/opensimAD](https://github.com/itbellix/opensimAD).
