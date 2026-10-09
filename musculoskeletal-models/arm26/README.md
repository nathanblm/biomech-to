The original `arm26.osim` model used for its simplicity has been adapted with the following features for compatibility with the RMR solver:

- Six `Millard2012EquilibriumMuscle` muscles.
- Three free coordinates: shoulder elevation, shoulder plane, and elbow flexion.
- A fixed base offset of 0.8 m upward.
- Reserve actuators for shoulder elevation and plane motion to compensate for omitted shoulder muscles.
- An `endpoint_external_force` prescribed force at the wrist, with ground-frame force components and zero applied torque. Its application point uses wrist-marker coordinates relative to the forearm body's elbow origin.
- Disabled hand and finger meshes for simpler visualization; forearm geometry is retained.

Inverse dynamics are generated using the [antoinefalisse/opensimAD repository](https://github.com/antoinefalisse/opensimAD), while forward dynamics are generated using the custom [itbellix/opensimAD repository](https://github.com/itbellix/opensimAD).