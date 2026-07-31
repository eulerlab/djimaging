from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from djimaging.utils.scanm import read_h5_utils, traces_and_triggers_utils, wparams_utils
from tests.fixtures.synthetic_scanm import PIXEL_DURATION_US, expected_trace_times


def test_compute_traces_synthetic(synthetic_recording_path: Callable[[str], Path]):
    with h5py.File(synthetic_recording_path("chirp"), "r", driver="stdio") as h5_file:
        wparams = read_h5_utils.extract_wparams(h5_file)
        expected_traces, stored_trace_times = read_h5_utils.extract_traces(h5_file)
        os_params = read_h5_utils.extract_os_params(h5_file)
        stack = np.copy(h5_file["wDataCh0"])
        roi_mask = read_h5_utils.extract_roi_mask(h5_file)

    _, traces, _, _ = traces_and_triggers_utils.compute_traces(
        stack=stack, roi_mask=roi_mask, wparams=wparams
    )

    assert_allclose(traces, expected_traces)
    assert_allclose(
        stored_trace_times - os_params["stimulatordelay"] / 1000,
        expected_trace_times(roi_mask, stack.shape[2], precision="line"),
    )


@pytest.mark.parametrize("precision", ["line", "pixel"])
def test_compute_tracetimes_synthetic(
        synthetic_recording_path: Callable[[str], Path], precision: str):
    with h5py.File(synthetic_recording_path("chirp"), "r", driver="stdio") as h5_file:
        wparams = read_h5_utils.extract_wparams(h5_file)
        stack = np.copy(h5_file["wDataCh0"])
        roi_mask = read_h5_utils.extract_roi_mask(h5_file)

    _, _, trace_times, _ = traces_and_triggers_utils.compute_traces(
        stack=stack, roi_mask=roi_mask, wparams=wparams, precision=precision
    )

    assert_allclose(
        trace_times,
        expected_trace_times(roi_mask, stack.shape[2], precision=precision),
    )


@pytest.mark.parametrize("stimulus", ["MB", "chirp", "DN"])
@pytest.mark.parametrize("precision", ["line", "pixel"])
def test_compute_triggertimes_synthetic(
        synthetic_recording_path: Callable[[str], Path], stimulus: str, precision: str):
    with h5py.File(synthetic_recording_path(stimulus), "r", driver="stdio") as h5_file:
        wparams = read_h5_utils.extract_wparams(h5_file)
        expected_times, expected_values = read_h5_utils.extract_triggers(
            h5_file, check_triggervalues=True
        )
        trigger_stack = np.copy(h5_file["wDataCh2"])

    trigger_times, trigger_values = wparams_utils.compute_triggers_from_wparams(
        stack=trigger_stack,
        wparams=wparams,
        precision=precision,
        stimulator_delay=0.,
    )

    # wParamsNum uses ScanM's float32 metadata, which accumulates a few
    # microseconds of rounding over the longest synthetic recording.
    atol = PIXEL_DURATION_US * 1e-6 if precision == "line" else 5e-6
    assert_allclose(trigger_times, expected_times, atol=atol, rtol=0)
    assert_allclose(trigger_values, expected_values)
