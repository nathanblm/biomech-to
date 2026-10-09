"""Small utilities shared by the biomechanics scripts."""

from pathlib import Path

import numpy as np


def save_motion_sto(path, time, states, state_labels):
    """Save a state trajectory in OpenSim Storage (.sto) format."""
    path = Path(path)
    time = np.asarray(time)
    states = np.asarray(states)

    if states.shape != (len(state_labels), len(time)):
        raise ValueError("states must have shape (number of labels, number of times)")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("motion\n")
        file.write("version=1\n")
        file.write(f"nRows={len(time)}\n")
        file.write(f"nColumns={len(state_labels) + 1}\n")
        file.write("inDegrees=no\n")
        file.write("endheader\n")
        file.write("time\t" + "\t".join(state_labels) + "\n")

        for k, sample_time in enumerate(time):
            row = np.concatenate(([sample_time], states[:, k]))
            file.write("\t".join(f"{value:.12g}" for value in row) + "\n")

    return path
