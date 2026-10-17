"""Run the dataset, OCP, RMR validation, and fatigue comparison pipeline."""

import os
from pathlib import Path

from code.msk_datasetbuilder import (
    MODEL_CONFIG as DATASET_CONFIG,
    build_msk_dataset,
    build_validation_dataset,
    dataset_demo,
)
from code.msk_fatiguemodel import (
    MODEL_CONFIG as FATIGUE_CONFIG,
    compare_muscle_state_trajectories,
)
from code.msk_optimization import (
    MODEL_CONFIG as OCP_CONFIG,
    solve_ocp,
    visualize_ocp_result,
)


ROOT = Path(__file__).resolve().parent

# Demo settings. Change these values to configure the pipeline.
MODEL_NAME = "arm26_3DOF"
N_SAMPLES = 10
FORCE_MAGNITUDES = [20.0]
FORCE_DIRECTIONS = [[-1.0, 0.0, 0.0]]
FORCE_INDEX = 0
HORIZON = 100
TIMESTEP = 0.1
ACTIVATION_DYNAMICS = True
FILTER_SUCCESS = True
SHOW_DATASET_VISUALS = True
SHOW_OCP_VISUAL = True
BLOCK_OCP_VISUAL = False


def main():
    # Module configs use paths relative to the repository root.
    os.chdir(ROOT)

    dataset_path = Path(
        build_msk_dataset(
            MODEL_NAME,
            n_samples=N_SAMPLES,
            magnitudes=FORCE_MAGNITUDES,
            directions=FORCE_DIRECTIONS,
        )
    )
    OCP_CONFIG[MODEL_NAME]["dataset_npz_path"] = str(dataset_path)

    if SHOW_DATASET_VISUALS:
        dataset_demo(
            MODEL_NAME,
            str(dataset_path),
            endpoint_marker=DATASET_CONFIG[MODEL_NAME]["reference_marker"],
        )

    ocp_result = solve_ocp(
        MODEL_NAME,
        force_index=FORCE_INDEX,
        horizon=HORIZON,
        timestep=TIMESTEP,
    )
    ocp_result_path = (
        ROOT / "results" / MODEL_NAME / "ocp" / "results" / f"{ocp_result.tag}.npz"
    )

    if SHOW_OCP_VISUAL:
        visualize_ocp_result(
            MODEL_NAME,
            str(ocp_result_path),
            block=BLOCK_OCP_VISUAL,
        )

    build_validation_dataset(
        MODEL_NAME,
        str(dataset_path),
        str(ocp_result_path),
        force_index=FORCE_INDEX,
        activation_dynamics=ACTIVATION_DYNAMICS,
        filter_succes=FILTER_SUCCESS,
    )
    validation_path = (
        ROOT / "results" / MODEL_NAME / "ocp" / "validation" / f"{ocp_result.tag}.npz"
    )

    FATIGUE_CONFIG[MODEL_NAME].update(
        dataset_npz_path=str(dataset_path),
        ocp_result_npz_path=str(ocp_result_path),
        ocp_validation_npz_path=str(validation_path),
    )
    return compare_muscle_state_trajectories(
        MODEL_NAME,
        force_index=FORCE_INDEX,
    )


if __name__ == "__main__":
    main()
