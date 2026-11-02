from pathlib import Path

import h5py
import numpy as np

from tests.fixtures.synthetic_scanm import (
    EXPERIMENT_DAY,
    EXPERIMENT_NUMBER,
    RECORDING_PREFIX,
    RECORDING_SPECS,
    generate_tutorial_dataset,
)


def test_generate_tutorial_dataset(tmp_path: Path):
    data_dir = generate_tutorial_dataset(tmp_path / "synthetic-scanm-data")
    experiment_dir = data_dir / EXPERIMENT_DAY / EXPERIMENT_NUMBER
    pre_dir = experiment_dir / "Pre"

    assert (experiment_dir / "synthetic__left.ini").is_file()
    assert (data_dir / "resources" / "noise.h5").is_file()
    assert len(list(pre_dir.glob("*.h5"))) == 3

    for stimulus, spec in RECORDING_SPECS.items():
        with h5py.File(pre_dir / f"{RECORDING_PREFIX}_{stimulus}_TEST.h5", "r") as h5_file:
            assert set(h5_file) == {
                "OS_Parameters",
                "ROIs",
                "Traces0_raw",
                "Tracetimes0",
                "Triggertimes",
                "Triggervalues",
                "wDataCh0",
                "wDataCh1",
                "wDataCh2",
                "wParamsNum",
            }
            assert h5_file["wDataCh0"].shape == (8, 8, spec["n_frames"])
            assert h5_file["Triggertimes"].shape == (len(spec["trigger_times"]),)
            assert np.array_equal(np.unique(h5_file["ROIs"][:]), [-2, -1, 1])

    total_size = sum(path.stat().st_size for path in data_dir.rglob("*") if path.is_file())
    assert total_size < 3 * 1024 * 1024
