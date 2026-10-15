from dataclasses import dataclass
from pathlib import Path

import numpy as np

@dataclass
class FatigueMap:
    joint_coordinate_samples: np.ndarray            # (n_samples, n_coordinates)
    joint_coordinate_names: np.ndarray              # (n_coordinates,)
    muscle_fatigue_rates: np.ndarray                # (n_samples, n_muscles)
    muscle_names: np.ndarray                        # (n_muscles,)
    force: np.ndarray                                # (n_samples, 3)

    @classmethod
    def from_npz(cls, npz_path: Path):
        with np.load(npz_path) as data:
            q = data["joint_coordinate_samples"]
