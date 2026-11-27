"""Offline loader on the SYNTHETIC ScanM dataset of ``tests/fixtures`` (h5 files) and, opt-in, on real local data.

Real data: set ``DJIMAGING_INSPECTOR_DATA`` to an experiment folder with ``Raw/*.smp`` and ``AutoROIs/*_ROIs.pkl``
(e.g. ``data/20260916/1``) and optionally ``DJIMAGING_INSPECTOR_FIELD`` (default ``GCL0``).
"""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from djimaging.inspector.offline import LocalFieldSource
from djimaging.utils.mask_format_utils import to_igor_format
from tests.fixtures.synthetic_scanm import EXPERIMENT_DAY, EXPERIMENT_NUMBER, RECORDING_PREFIX, expected_trace_times


@pytest.fixture
def no_database(monkeypatch):
    """Fail if anything tries to connect to a database."""
    import datajoint as dj

    def refuse(*args, **kwargs):
        raise AssertionError("the offline inspector must not connect to a database")

    monkeypatch.setattr(dj, 'conn', refuse)
    monkeypatch.setattr(dj.connection, 'conn', refuse, raising=False)


@pytest.fixture
def experiment(tutorial_data_dir, tmp_path):
    """Copy of the synthetic experiment plus AutoROIs mask files: non-consecutive IDs (1, 7) for chirp, the same
    mask shifted by one pixel in x for the moving bar, no mask for the noise recording."""
    src = tutorial_data_dir / EXPERIMENT_DAY / EXPERIMENT_NUMBER
    dst = tmp_path / EXPERIMENT_DAY / EXPERIMENT_NUMBER
    shutil.copytree(src, dst)
    masks = dst / 'AutoROIs'
    masks.mkdir()
    import h5py
    with h5py.File(dst / 'Pre' / f'{RECORDING_PREFIX}_chirp_TEST.h5', 'r') as f:
        igor = f['ROIs'][:].astype(int)
    python = np.where(igor == -1, 1, np.where(igor == -2, 7, 0))
    np.save(masks / f'{RECORDING_PREFIX}_chirp_TEST_ROIs.npy', python)
    np.save(masks / f'{RECORDING_PREFIX}_MB_TEST_ROIs.npy', np.roll(python, 1, axis=0))
    return dst, igor


def test_offline_synthetic_h5(experiment, no_database):
    exp_dir, igor = experiment
    src = LocalFieldSource(exp_dir, field='FIELD0')
    assert not src.from_raw_data
    infos = {i.stim_name: i for i in src.list_presentations()}
    assert set(infos) == {'chirp', 'movingbar', 'noise'}
    assert not infos['noise'].has_mask

    chirp = src.load_presentation(infos['chirp'].name, 'wDataCh0')
    assert chirp.roi_ids.tolist() == [1, 7]
    assert chirp.mask_info['as_field_mask'] == 'same'
    assert chirp.triggers.size == 10 and chirp.quality.metric == 'qidx'
    assert set(chirp.quality.values) == {1, 7} and chirp.quality.reference == pytest.approx(1 / 5)
    # traces = ROI means of the stack at djimaging's line-precision times (expected from the scan geometry)
    times = expected_trace_times(igor, chirp.n_frames)
    for col, roi_id in enumerate([1, 7]):
        trace = chirp.signals['raw'].traces[roi_id]
        np.testing.assert_allclose(trace.times, times[:, col], atol=1e-9)
        pixels = chirp.rois.pixels(roi_id)
        np.testing.assert_allclose(trace.values, chirp.frames[:, pixels].mean(axis=1), rtol=1e-6)

    mb = src.load_presentation(infos['movingbar'].name, 'wDataCh0')
    assert mb.mask_info['as_field_mask'] == 'shifted' and abs(mb.mask_info['shift_dx']) == 1
    assert mb.quality.metric == 'd_qi' and set(mb.quality.extra) == {'ds_index', 'os_index'}
    assert mb.trigger_labels[:3] == ['0°', '180°', '45°'] and mb.trigger_labels[8] == '0°'  # short, no rep suffix
    assert mb.stimulus.conditions == [0, 180, 45, 225, 90, 270, 135, 315] and mb.stimulus.pad_trace
    assert chirp.stimulus.ntrigger_rep == 2 and chirp.stimulus.conditions is None

    noise = src.load_presentation(infos['noise'].name, 'wDataCh0')
    assert noise.rois is None and noise.quality is None and 'no ROI mask' in noise.quality_unavailable
    assert src.load_highres('wDataCh0')[0] is None  # the synthetic dataset has no HighRes file


def test_offline_does_not_write_into_data(experiment, no_database):
    exp_dir, _ = experiment
    before = {p: p.stat().st_mtime_ns for p in Path(exp_dir).rglob('*')}
    src = LocalFieldSource(exp_dir, field='FIELD0')
    for info in src.list_presentations():
        src.load_presentation(info.name, 'wDataCh0')
    after = {p: p.stat().st_mtime_ns for p in Path(exp_dir).rglob('*')}
    assert before == after


REAL = os.environ.get('DJIMAGING_INSPECTOR_DATA')


@pytest.mark.skipif(not REAL, reason='set DJIMAGING_INSPECTOR_DATA to a local experiment folder to run')
def test_offline_real_data(no_database):
    field = os.environ.get('DJIMAGING_INSPECTOR_FIELD', 'GCL0')
    src = LocalFieldSource(REAL, field=field)
    infos = src.list_presentations()
    chirp = next(i for i in infos if i.stim_name == 'chirp')
    pres = src.load_presentation(chirp.name, chirp.channels[0])
    assert pres.triggers.size == 10 and pres.n_frames > 1000
    # ROI IDs are the mask labels (no renumbering), traces exist for every ROI not touching the artifact lines
    mask = to_igor_format(np.asarray(__import__('pickle').load(open(Path(REAL) / 'AutoROIs' / Path(
        pres.mask_info['roi_mask_file']).name, 'rb'))).astype(np.int32))
    assert pres.roi_ids.tolist() == sorted(np.unique(-mask[mask < 0]).tolist())
    assert set(pres.signals['raw'].traces) | set(pres.roi_notes) == set(pres.roi_ids.tolist())
    hr, note = src.load_highres(pres.channel)
    assert hr is not None, note
