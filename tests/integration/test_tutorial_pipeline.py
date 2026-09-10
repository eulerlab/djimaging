import os

import h5py
import numpy as np
import pytest

from tests.fixtures.random_utils import numpy_seed


pytestmark = pytest.mark.skipif(
    os.environ.get("DJ_TEST_MYSQL") != "1",
    reason="requires the MySQL integration-test service",
)


def test_tutorial_pipeline(tutorial_schema, tutorial_data_dir, dj_test_stores):
    experiment_key = {"experimenter": "synthetic"}

    tutorial_schema.UserInfo().upload_user(
        {
            **experiment_key,
            "data_dir": str(tutorial_data_dir),
            "datatype_loc": 0,
            "animal_loc": 1,
            "region_loc": 2,
            "field_loc": 3,
            "stimulus_loc": 4,
            "cond1_loc": 5,
            "cond2_loc": 6,
            "cond3_loc": 7,
        },
        verbose=0,
    )
    tutorial_schema.RawDataParams().add_default(experimenter_list=["synthetic"])

    tutorial_schema.Experiment().rescan_filesystem(restrictions=experiment_key, verboselvl=0)

    processed = dj_test_stores["processed"]
    external_files_before = {path for path in processed.rglob("*") if path.is_file()}
    tutorial_schema.Field().rescan_filesystem(restrictions=experiment_key, verboselvl=0)
    assert {path for path in processed.rglob("*") if path.is_file()} == external_files_before, (
        "Field channel averages must stay in the database"
    )

    tutorial_schema.Stimulus().add_nostim(skip_duplicates=True)
    tutorial_schema.Stimulus().add_chirp(
        spatialextent=1000,
        stim_name="gChirp",
        alias="chirp_gchirp_globalchirp",
        skip_duplicates=True,
    )
    tutorial_schema.Stimulus().add_chirp(
        spatialextent=300,
        stim_name="lChirp",
        alias="lchirp_localchirp",
        skip_duplicates=True,
    )
    with h5py.File(tutorial_data_dir / "resources" / "noise.h5", "r") as h5_file:
        noise_stimulus = h5_file["stimulusarray"][:].T.astype(int)
    tutorial_schema.Stimulus().add_noise(
        stim_name="noise",
        pix_n_x=20,
        pix_n_y=15,
        pix_scale_x_um=30,
        pix_scale_y_um=30,
        stim_path=tutorial_data_dir / "resources" / "noise.h5",
        stim_trace=noise_stimulus,
        skip_duplicates=True,
    )
    tutorial_schema.Stimulus().add_movingbar(skip_duplicates=True)

    external_files_before = {path for path in processed.rglob("*") if path.is_file()}
    tutorial_schema.Presentation().populate(display_progress=False)
    tutorial_schema.RoiMask().rescan_filesystem(
        restrictions=experiment_key,
        verboselvl=0,
        roi_mask_dir="AutoROIs",
    )
    tutorial_schema.Roi().populate(experiment_key, display_progress=False)

    tutorial_schema.Traces().populate(experiment_key, display_progress=False)
    tutorial_schema.PreprocessParams().add_default(skip_duplicates=True)
    tutorial_schema.PreprocessTraces().populate(experiment_key, display_progress=False)
    tutorial_schema.Snippets().populate(experiment_key, display_progress=False)
    tutorial_schema.Averages().populate(experiment_key, display_progress=False)
    tutorial_schema.ChirpQI().populate(experiment_key, display_progress=False)

    with numpy_seed(42):
        tutorial_schema.OsDsIndexes().populate(experiment_key, display_progress=False)

    external_files_after = {path for path in processed.rglob("*") if path.is_file()}
    assert external_files_after == external_files_before, (
        "Presentation channel averages, ROI masks, traces, snippets, averages, "
        "and response metrics must stay in the database"
    )

    tutorial_schema.OpticDisk().populate(experiment_key, display_progress=False)
    tutorial_schema.RelativeFieldLocation().populate(experiment_key, display_progress=False)
    tutorial_schema.RetinalFieldLocation().populate(experiment_key, display_progress=False)

    expected_counts = {
        tutorial_schema.Experiment(): 1,
        tutorial_schema.Field(): 1,
        tutorial_schema.Presentation(): 3,
        tutorial_schema.RoiMask(): 1,
        tutorial_schema.RoiMask.RoiMaskPresentation(): 3,
        tutorial_schema.Roi(): 2,
        tutorial_schema.Traces(): 6,
        tutorial_schema.PreprocessTraces(): 6,
        tutorial_schema.Snippets(): 4,
        tutorial_schema.Averages(): 4,
        tutorial_schema.ChirpQI(): 2,
        tutorial_schema.OsDsIndexes(): 2,
        tutorial_schema.OpticDisk(): 1,
        tutorial_schema.RelativeFieldLocation(): 1,
        tutorial_schema.RetinalFieldLocation(): 1,
    }
    count_mismatches = {}
    for table, expected_count in expected_counts.items():
        observed_count = len(table & experiment_key)
        if observed_count != expected_count:
            count_mismatches[table.full_table_name] = {
                "expected": expected_count,
                "observed": observed_count,
            }
    assert not count_mismatches, f"Unexpected table counts: {count_mismatches}"

    assert np.all((tutorial_schema.Presentation() & experiment_key).to_arrays("trigger_valid") == 1)
    assert np.all(np.isfinite((tutorial_schema.ChirpQI() & experiment_key).to_arrays("qidx")))
    assert np.all(np.isfinite((tutorial_schema.OsDsIndexes() & experiment_key).to_arrays("ds_index")))
    assert np.all(np.isfinite((tutorial_schema.OsDsIndexes() & experiment_key).to_arrays("os_index")))
