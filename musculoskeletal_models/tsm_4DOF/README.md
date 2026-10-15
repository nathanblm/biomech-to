# TSM 4-DOF model

`tsm_4DOF.osim` is adapted from `TSM_subject_noWeight.osim` for four-coordinate
map generation and RMR optimization:

- Free coordinates: `plane_elv`, `shoulder_elv`, `axial_rot`, and `elbow_flexion`.
- Locked coordinates: `ground_thorax_rot_x/y/z`, `ground_thorax_tx/y/z`,
  `clav_prot`, `clav_elev`, `scapula_abduction`, `scapula_elevation`,
  `scapula_upward_rot`, `scapula_winging`, and `pro_sup`.
- The thorax is fixed to ground. A `ground_offset` frame raises the model by
  `0.8 m` along ground Y, matching the arm26 model base height.
- Added `hand_external_force` at the hand body origin. Force components are in
  the ground frame and applied torque is zero.
- Added the `endpoint` marker at the `/bodyset/hand` origin: `(0, 0, 0) m`.

