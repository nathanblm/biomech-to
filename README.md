# Viewing maps

The visualizer reads an `MSKdataset` saved by `save_msk_dataset()` in a `.npz`
archive and displays a muscle map or the sampled workspace. Choose a force with
`--force` (the example archive has one,
so use `0`) and a map with `--field`:

| Field | Muscle argument | Meaning |
| --- | --- | --- |
| `activation` | `--muscle NAME` or `--muscle INDEX` | Muscle activation |
| `muscle_force` | `--muscle NAME` or `--muscle INDEX` | Muscle force in N |
| `fatigue` | `--muscle NAME` or `--muscle INDEX` | Fatigue coefficient |
| `dominance` | None | Agonist–antagonist fatigue dominance |
| `workspace` | None | Sampled positions |

For muscle fields, select a muscle by its saved name (for example,
`--muscle BIClong`) or by its column index (for example, `--muscle 0`). Dominance is
positive when the agonist group accumulates more fatigue, negative when the
antagonist group does, and near zero when they are similar. The configured
groups are stored as `agonist_muscles` and `antagonist_muscles` in `ModelInfo`.

## Cartesian space

Cartesian plots use Matplotlib by default. For a 3D OpenSim view, pass the
matching model and `--backend simbody`:

```bash
RESULTS=results/tsm_4DOF/datasets/msk_dataset_p1e4_m1_d1_fx-20_fy0_fz0.npz
MODEL=musculoskeletal_models/tsm_4DOF/tsm_4DOF.osim

python3 -m code.visualizer "$RESULTS" --field activation --muscle 0 --force 0
python3 -m code.visualizer "$RESULTS" --model "$MODEL" --backend simbody \
  --field dominance --force 0
```

To show a Cartesian slice in Simbody, provide the model, axis, and position.
`--reference-marker` makes the slice position relative to that marker's default
pose:

```bash
python3 -m code.visualizer "$RESULTS" --model "$MODEL" --backend simbody \
  --field dominance --force 0 \
  --slice-axis y --slice-value -0.1 --reference-marker ac --tolerance 0.05
```

## Joint space

Use `--space joint` to plot the saved joint-coordinate samples. Select two or
three coordinates with `--coordinate-axes`; for a slice, choose another
coordinate with `--slice-coordinate` and `--slice-value`:

```bash
python3 -m code.visualizer "$RESULTS" --space joint \
  --field activation --muscle 0 --force 0 \
  --coordinate-axes 0 1 --slice-coordinate 2 --slice-value 0.5 --tolerance 0.05
```

Coordinate selectors accept indices in the archive's order. Saved coordinate
names can also be used for `--slice-coordinate`. Joint-space plots use Matplotlib.

`--max-points N` subsamples only the display. For Simbody appearance, adjust
`--radius` (metres) and `--opacity` (0–1).

## Visualize an optimization result

```bash
python3 -m code.visualizer results/arm26_3DOF/datasets/msk_dataset_p1e5_m1_d1_fx-20_fy0_fz0.npz \
  --model musculoskeletal_models/arm26_3DOF/arm26_3DOF.osim \
  --ocp-result results/arm26_3DOF/ocp/results_20261016_154356.npz --force 0 \
  --slice-axis y --slice-value 0.6 --tolerance 0.05 --opacity 0.8
```

This opens the Simbody dominance map with the OCP trajectory and a fatigue
chart. `--slice-axis`, `--slice-value`, and `--tolerance` set the slice; opacity
defaults to `0.5` and can be changed with `--opacity`.

The fatigue chart shows all muscles by default. Use `--muscles` to select traces
by saved name or column index, and add `--agonist-antagonist` to overlay the
configured group means
