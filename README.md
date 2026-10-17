# Pipeline demo

From the repository root, activate your project environment and run:

```bash
python main.py
```

The script builds and visualizes the muscle fatigue map, optimizes and visualizes a trajectory with the OCP, generates RMR validation data, then compares the map based and RMR based muscle state trajectories. Close the dataset visualization and press Enter when prompted to continue.

Configure the demo at the top of [main.py](main.py):

- `MODEL_NAME`: model key, such as `arm26_3DOF`.
- `N_SAMPLES`: number of sampled joint configurations used to build the map.
- `FORCE_MAGNITUDES` and `FORCE_DIRECTIONS`: endpoint forces included in the map; `FORCE_INDEX` selects one for the OCP and validation.
- `HORIZON` and `TIMESTEP`: OCP horizon length and time step.
- `ACTIVATION_DYNAMICS` and `FILTER_SUCCESS`: options for generating RMR validation data.
- `SHOW_DATASET_VISUALS` and `SHOW_OCP_VISUAL`: enable or skip the two visualization steps.
- `BLOCK_OCP_VISUAL`: set to `True` to wait for the OCP plot window to close before continuing; defaults to `False`.

Increase `N_SAMPLES` for a denser map. Model files and generated dynamics must be available at the paths configured in the pipeline modules.
