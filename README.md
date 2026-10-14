
# Viewing maps

The current result archive is:

```text
results/tsm_4DOF/p1e5_m1_d1_fx-20_fy0_fz0.npz
```

Use the visualizer with the matching OpenSim model. The `--force` value is the
flattened force index; for the current archive, use `0`.

```bash
RESULTS=results/tsm_4DOF/p1e5_m1_d1_fx-20_fy0_fz0.npz
MODEL=musculoskeletal_models/tsm_4DOF/tsm_4DOF.osim
```

## Maps

Select a map with `--field`:

```bash
# Muscle activation map
python3 -m code.visualizer "$RESULTS" --model "$MODEL" \
  --backend simbody --field activation --muscle 0 --force 0

# Muscle force map
python3 -m code.visualizer "$RESULTS" --model "$MODEL" \
  --backend simbody --field muscle_force --muscle 0 --force 0

# Fatigue map
python3 -m code.visualizer "$RESULTS" --model "$MODEL" \
  --backend simbody --field fatigue --muscle 0 --force 0

# Agonist–antagonist fatigue dominance map
python3 -m code.visualizer "$RESULTS" --model "$MODEL" \
  --backend simbody --field dominance --force 0
```

For `activation`, `muscle_force`, and `fatigue`, replace `--muscle 0` with the
index of the muscle to inspect. The archive must contain the corresponding map.
The dominance map is group-level, so it does not take `--muscle`.

Positive dominance values indicate greater fatigue accumulation in the
agonist group; negative values indicate greater accumulation in the antagonist
group. Values near zero indicate similar accumulation. The currently configured
groups are agonists `BIC_long`, `BIC_brevis` and antagonist `TRIlong`; update
the groups in `code/mm_builder.py` when generating a different comparison.

To inspect a workspace slice, add for example:

```bash
--slice-axis y --slice-value -0.1 --reference-marker ac --tolerance 0.05
```

For Simbody point appearance, adjust `--radius` (metres) and `--opacity`:

```bash
--radius 0.006 --opacity 1.0
```
