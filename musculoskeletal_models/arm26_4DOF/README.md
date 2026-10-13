The original `arm26.osim` model used for its simplicity has been adapted with the following features for compatibility with the RMR solver:

- Six `Millard2012EquilibriumMuscle` muscles.
- Four free coordinates: shoulder elevation, shoulder plane, shoulder axial rotation, and elbow flexion.
- The shoulder axial-rotation coordinate is named `r_shoulder_rot`, rotates about the shoulder frame's second (`Y`) axis, and has a range of `[-pi, pi]` radians.
- A fixed base offset of 0.8 m upward.
- Reserve actuators for shoulder elevation, plane motion, and axial rotation to compensate for omitted shoulder muscles.
- An `endpoint_external_force` prescribed force at the wrist, with ground-frame force components and zero applied torque. Its application point uses wrist-marker coordinates relative to the forearm body's elbow origin.
- Because the axial-rotation reserve actuator is part of the model's ForceSet, `endpoint_external_force` is ForceSet index `9` in this four-DOF version. Code that uses `RMRsolver` directly should select that index; `fatigue_maps.py` discovers it by name.
- Disabled hand and finger meshes for simpler visualization; forearm geometry is retained.

The RMR solver discovers all unlocked coordinates automatically. The axial-rotation reserve actuator is included so the acceleration constraint remains actuated even though this reduced model does not include dedicated shoulder muscles for that DOF. No RMR solver source changes are required for the extra coordinate.

Inverse dynamics are generated using the [antoinefalisse/opensimAD repository](https://github.com/antoinefalisse/opensimAD), while forward dynamics are generated using the custom [itbellix/opensimAD repository](https://github.com/itbellix/opensimAD).
