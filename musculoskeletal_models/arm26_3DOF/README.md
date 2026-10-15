# arm26 3-DOF model

`arm26_3DOF.osim` is a reduced arm26 model configured for the RMR solver:

- Six `Millard2012EquilibriumMuscle` muscles and three free coordinates:
  `r_shoulder_elev`, `r_shoulder_plane`, and `r_elbow_flex` (shoulder elevation,
  shoulder plane, and elbow flexion).
- A fixed `0.8 m` upward base offset and reserve actuators for shoulder
  elevation and plane motion.
- Added `endpoint_external_force` at the wrist. Its force is expressed in the
  ground frame, applied torque is zero, and its application point uses
  wrist-marker coordinates relative to the forearm body's elbow origin.
- Added the `endpoint` marker on `/bodyset/r_ulna_radius_hand` at local
  `(0, -0.23559, 0) m`.
- Hand and finger meshes are disabled; forearm geometry remains visible.

Inverse dynamics use [antoinefalisse/opensimAD](https://github.com/antoinefalisse/opensimAD).
Forward dynamics use [itbellix/opensimAD](https://github.com/itbellix/opensimAD).
