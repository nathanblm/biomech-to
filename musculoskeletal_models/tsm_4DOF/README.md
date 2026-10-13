The `TSM_subject_noWeight.osim` has been adapted as follows:

- Added the `hand_external_force` prescribed force at the hand body origin.
  Force components are expressed in ground coordinates and the applied torque
  is zero.
- Fixed the thorax base pose by locking the six `ground_thorax` coordinates:
  `ground_thorax_rot_x/y/z` and `ground_thorax_tx/y/z`. 
- Added a fixed `+0.8 m` translation in the ground Y direction through the
  `ground_offset` frame, matching the elevated base used by `arm26`.
- Reduced the sampled kinematics to the three glenohumeral coordinates and
  elbow flexion: `plane_elv`, `shoulder_elv`, `axial_rot`, and
  `elbow_flexion`.
- Locked `clav_prot`, `clav_elev`, `scapula_abduction`, `scapula_elevation`, `scapula_upward_rot`, `scapula_winging`, and `pro_sup`.
