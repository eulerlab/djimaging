"""Tests of ScanMRecording (with the synthetic .h5 recordings) and of the .smp readers (with a fake SMP object)."""
import shutil
import warnings
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from djimaging.utils.scanm import read_smp_utils, read_utils, recording
from djimaging.utils.scanm.recording import ScanMRecording
from scanmsupport.scanm.scanm_smp import SMP
from tests.fixtures.synthetic_scanm import FRAME_DT, NX, NY, PIXEL_DURATION_US, STIMULATOR_DELAY_MS
from tests.utils.test_scanm_read_h5 import wparams_num, write_h5


@pytest.fixture
def chirp_path(synthetic_recording_path: Callable[[str], Path]) -> str:
    return str(synthetic_recording_path('chirp'))


@pytest.fixture
def chirp(chirp_path) -> ScanMRecording:
    return ScanMRecording(chirp_path, stimulator_delay=STIMULATOR_DELAY_MS / 1000.)


def test_recording_attributes(chirp):
    assert chirp.filetype == '.h5'
    assert chirp.scan_type == 'xy'
    assert chirp.scan_type_id == 10
    assert chirp.ch_names == ['wDataCh0', 'wDataCh1', 'wDataCh2']
    assert chirp.ch_n_frames == 900
    assert (chirp.pix_nx, chirp.pix_ny, chirp.pix_nz) == (NX, NY, 0)
    assert chirp.real_pixel_duration == pytest.approx(PIXEL_DURATION_US * 1e-6)
    assert chirp.scan_line_duration == pytest.approx(FRAME_DT / NY)
    assert chirp.scan_n_lines == NY
    assert chirp.scan_period == pytest.approx(FRAME_DT)
    assert chirp.scan_frequency == pytest.approx(1 / FRAME_DT)
    assert (chirp.pos_x_um, chirp.pos_y_um, chirp.pos_z_um) == (123, 245, 9)
    assert chirp.pix_dx_um == 0.  # neither setup_id nor scan_width_um
    assert chirp.pix_n_artifact is None
    assert chirp.roi_mask.shape == (NX, NY)
    assert chirp.trigger_times.size == 10
    assert chirp.trigger_valid is None
    assert 'xy: [8 x 8]' in repr(chirp)
    assert chirp.wparams_other == {}


def test_recording_pixel_size_and_artifact(chirp_path):
    rec = ScanMRecording(chirp_path, scan_width_um=80., setup_id=1, npixartifact=2)
    assert rec.pix_dx_um == pytest.approx(80. / NX)
    assert rec.pix_dy_um == rec.pix_dx_um
    assert rec.pix_n_artifact == 2

    rec = ScanMRecording(chirp_path, setup_id=1)
    assert rec.pix_dx_um == pytest.approx(112. / NX)
    assert rec.pix_n_artifact == 1


@pytest.mark.parametrize('repeated_stim, ntrigger_rep, expected', [
    (True, 2, True), (True, 5, True), (True, 3, False), (False, 10, True), (False, 2, False)])
def test_recording_trigger_valid(chirp_path, repeated_stim, ntrigger_rep, expected):
    rec = ScanMRecording(chirp_path, repeated_stim=repeated_stim, ntrigger_rep=ntrigger_rep)
    assert rec.trigger_valid == expected


def test_recording_compute_triggers(chirp):
    triggertimes_file = chirp.trigger_times.copy()
    chirp.compute_triggers(time_precision='pixel', repeated_stim=True, ntrigger_rep=2)
    assert chirp.trigger_threshold == 30_000
    assert_allclose(chirp.trigger_times, triggertimes_file + STIMULATOR_DELAY_MS / 1000., atol=1e-9)
    assert_array_equal(chirp.trigger_values, 60_000)
    assert chirp.trigger_valid
    assert chirp.frame_dt == pytest.approx(FRAME_DT)
    assert chirp.frame_dt_offset.shape == (NX, NY)

    chirp.compute_triggers(ntrigger_rep=0, repeated_stim=False)
    assert chirp.trigger_times.size == 0
    assert chirp.trigger_valid  # 0 triggers expected and found


def test_recording_compute_triggers_fixed_threshold(chirp):
    chirp.compute_triggers(trigger_threshold=70_000)
    assert chirp.trigger_times.size == 0


def test_recording_compute_triggers_needs_delay(chirp_path):
    rec = ScanMRecording(chirp_path)
    with pytest.raises(ValueError, match='stimulator_delay'):
        rec.compute_triggers()


def test_recording_auto_threshold_midpoint(chirp):
    chirp.ch_stacks['wDataCh2'] = np.where(chirp.ch_stacks['wDataCh2'] > 30_000, 20_000, 0).astype(np.uint16)
    chirp.compute_triggers()
    assert chirp.trigger_threshold == 10_000
    assert chirp.trigger_times.size == 10


def test_recording_auto_threshold_high_baseline(chirp):
    chirp.ch_stacks['wDataCh2'] = np.where(chirp.ch_stacks['wDataCh2'] > 30_000, 60_000, 30_000).astype(np.uint16)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        chirp.compute_triggers()
    assert chirp.trigger_threshold == 45_000
    assert chirp.trigger_times.size == 10


def test_recording_equality(chirp_path, chirp):
    other = ScanMRecording(chirp_path)
    assert chirp == other
    assert chirp != 'not a recording'
    other.trigger_times = other.trigger_times + 1.
    assert chirp != other
    other = ScanMRecording(chirp_path)
    other.ch_stacks['wDataCh0'] = other.ch_stacks['wDataCh0'] + 1
    assert chirp != other
    other.ch_names = ['wDataCh0']
    assert chirp != other


def test_recording_wparams_dicts(chirp):
    num = chirp.get_wparams_num_dict()
    assert num['User_dxPix'] == NX
    assert num['XCoord_um'] == 123
    assert set(chirp.get_wparams_num_dict(lower_keys=True)) == {k.lower() for k in num}
    assert set(chirp.get_wparams_str_dict(lower_keys=True)) == {k.lower() for k in chirp.get_wparams_str_dict()}


def test_recording_wparams_num_real_pix_dur_unit(chirp):
    assert chirp.get_wparams_num_dict()['RealPixDur'] == pytest.approx(PIXEL_DURATION_US)


def test_recording_to_h5(chirp, tmp_path):
    out = tmp_path / 'out.h5'
    out.write_text('')
    with pytest.raises(FileExistsError):
        chirp.to_h5(str(out))
    with pytest.warns(UserWarning, match='Triggers are not computed'):
        chirp.trigger_times = None
        chirp.to_h5(str(out), overwrite=True)
    with h5py.File(out, 'r') as f:
        assert {'wDataCh0', 'wDataCh1', 'wDataCh2', 'wParamsNum', 'wParamsStr', 'Triggertimes'} <= set(f.keys())
        assert f['wParamsNum'].attrs['User_dxPix'] == NX
        assert f['Triggertimes'].size == 10

    chirp.to_h5(str(out), overwrite=True, include_wparams_num=False, include_wparams_str=False,
                include_triggers=False)
    with h5py.File(out, 'r') as f:
        assert set(f.keys()) == {'wDataCh0', 'wDataCh1', 'wDataCh2'}


@pytest.mark.xfail(strict=True, reason="BUG: file type is dispatched on the case-sensitive extension; a .H5 file "
                                       "(accepted by get_file_info_df) is read as .smp")
def test_recording_upper_case_extension(chirp_path, tmp_path):
    path = tmp_path / 'SMP_M1_GCL0_chirp.H5'
    shutil.copy(chirp_path, path)
    rec = ScanMRecording(str(path))
    assert rec.ch_n_frames == 900


def test_recording_xz_scan(tmp_path):
    nx, nz, nf = 6, 5, 4
    path = write_h5(tmp_path / 'xz.h5', {'wDataCh0': np.zeros((nx, nz, nf)), 'wDataCh2': np.zeros((nx, nz, nf))},
                    wparams_num(nx=nx, ny=1, nz=nz, User_AspectRatioFr=2.))
    rec = ScanMRecording(path, stimulator_delay=0.)
    assert rec.scan_type == 'xz'
    assert rec.scan_n_lines == nz
    assert rec.pix_dy_um is None
    assert rec.roi_mask is None
    assert rec.trigger_times.size == 0
    assert 'xz: [6 x 5]' in repr(rec)
    assert rec.wparams_other == {'User_AspectRatioFr': 2.}
    rec.compute_triggers()
    assert rec.frame_dt_offset.shape == (nx, nz)
    assert rec.trigger_times.size == 0


def test_recording_xy_aspect_ratio_not_supported(tmp_path):
    path = write_h5(tmp_path / 'xy.h5', {'wDataCh0': np.zeros((4, 4, 2))},
                    wparams_num(nx=4, ny=4, User_AspectRatioFr=2.))
    with pytest.raises(NotImplementedError, match='Aspect ratio'):
        ScanMRecording(path)


def test_recording_missing_roi_mask_or_triggers_raise(tmp_path):
    path = write_h5(tmp_path / 'xy.h5', {'wDataCh0': np.zeros((4, 4, 2))}, wparams_num(nx=4, ny=4))
    with pytest.raises(KeyError, match='ROI'):
        ScanMRecording(path, roi_mask_ignore_not_found=False)
    with pytest.raises(KeyError, match='triggertimes'):
        ScanMRecording(path, triggers_ignore_not_found=False)


# .smp readers with a fake SMP object (no real .smp/.smh file needed)

class FakeSMP(SMP):
    dxFr_pix = 7
    dyFr_pix = 3
    dzFr_pix = None
    dxRetrace_pix = 2
    dxOffs_pix = 1
    scanType = 10

    def __init__(self, data=None, kv=None):
        super().__init__()
        n_frames = 5
        self._data = data if data is not None else {
            0: np.arange(n_frames * 3 * 4, dtype=np.uint16).reshape(n_frames, 3, 4),  # (frames, y, x), cropped
            2: np.zeros((n_frames, 3, 4), dtype=np.uint16),
        }
        self._kvPairDict = kv if kv is not None else {
            'RealPixelDuration_µs': ['float', 1, 5.],
            'AspectRatioFrame': ['float', 1, 1.],
            'Zoom': ['float', 1, 2.],
            'Angle_deg': ['float', 1, 0.],
            'XCoord_um': ['float', 1, 1.],
            'YCoord_um': ['float', 1, 2.],
            'ZCoord_um': ['float', 1, 3.],
            'ZStep_um': ['float', 1, 1.],
        }
        self.loaded = []

    def loadSMH(self, fName, verbose=False):
        self.loaded.append(('smh', fName))

    def loadSMP(self, verbose=False):
        self.loaded.append('smp')

    def getData(self, ch=0, crop=False):
        if ch == 1:
            raise RuntimeError('broken channel')
        return self._data.get(ch)


def test_load_wparams_from_smp_object():
    wparams, raw_file = read_smp_utils.load_wparams(FakeSMP())
    assert isinstance(raw_file, FakeSMP)
    assert wparams['realpixdur'] == 5.
    assert wparams['user_aspectratiofr'] == 1.
    assert 'realpixelduration_µs' not in wparams
    assert (wparams['user_dxpix'], wparams['user_dypix'], wparams['user_dzpix']) == (7, 3, 0)
    assert (wparams['user_npixretrace'], wparams['user_nxpixlineoffs'], wparams['user_scantype']) == (2, 1, 10)

    wparams = read_smp_utils.load_wparams(FakeSMP(), return_file=False, lower_keys=False)
    assert wparams['RealPixDur'] == 5.
    assert wparams['User_dxPix'] == 7


def test_load_wparams_from_path(monkeypatch):
    monkeypatch.setattr(read_smp_utils, 'SMP', FakeSMP)
    wparams, raw_file = read_smp_utils.load_wparams('some/file.smp')
    assert raw_file.loaded == [('smh', 'some/file.smp'), 'smp']
    assert wparams['user_dxpix'] == 7


def test_load_stacks_and_wparams_smp():
    with pytest.warns(UserWarning, match='wDataCh1'):
        ch_stacks, wparams = read_smp_utils.load_stacks_and_wparams(FakeSMP(), ch_names=('wDataCh0', 'wDataCh1'))
    assert ch_stacks['wDataCh0'].shape == (4, 3, 5)  # (x, y, frames)
    assert ch_stacks['wDataCh0'][1, 2, 3] == ((3 * 3) + 2) * 4 + 1
    assert_array_equal(ch_stacks['wDataCh1'], 0)
    assert ch_stacks['wDataCh1'].shape == (4, 3, 5)

    with pytest.warns(UserWarning, match='wDataCh3'):
        ch_stacks, _ = read_smp_utils.load_stacks_and_wparams(FakeSMP(), ch_names=('wDataCh2', 'wDataCh3'))
    assert sorted(ch_stacks) == ['wDataCh2', 'wDataCh3']

    with pytest.raises(AttributeError):  # Primary channel is missing
        read_smp_utils.load_stacks_and_wparams(FakeSMP(), ch_names=('wDataCh3',))


def test_load_all_stacks_and_wparams_smp():
    with pytest.warns(UserWarning, match='wDataCh1'):
        ch_stacks, wparams = read_smp_utils.load_all_stacks_and_wparams(FakeSMP(), ch_max=4)
    assert sorted(ch_stacks) == ['wDataCh0', 'wDataCh2']
    assert ch_stacks['wDataCh2'].shape == (4, 3, 5)


def test_read_utils_load_stacks_from_raw(monkeypatch):
    monkeypatch.setattr(read_smp_utils, 'SMP', FakeSMP)
    ch_stacks, wparams = read_utils.load_stacks('x.smp', from_raw_data=True, ch_names=('wDataCh0',))
    assert ch_stacks['wDataCh0'].shape == (4, 3, 5)
    assert wparams['realpixdur'] == 5.


def test_recording_from_smp(monkeypatch):
    monkeypatch.setattr(read_smp_utils, 'SMP', FakeSMP)
    with pytest.warns(UserWarning, match='wDataCh1'):
        rec = recording.ScanMRecording('Raw/M1_GCL0_chirp.smp', stimulator_delay=0.)
    assert rec.filetype == '.smp'
    assert rec.ch_names == ['wDataCh0', 'wDataCh2']
    assert (rec.pix_nx_full, rec.pix_nx, rec.pix_ny) == (7, 4, 3)
    assert rec.real_pixel_duration == pytest.approx(5e-6)
    assert rec.obj_zoom == 2.
    assert rec.roi_mask is None and rec.trigger_times is None
    rec.compute_frame_times(time_precision='pixel')
    assert rec.frame_dt_offset.shape == (4, 3)
    assert rec.frame_dt == pytest.approx(7 * 3 * 5e-6)
    assert_allclose(rec.frame_dt_offset[:, 0], np.arange(1, 5) * 5e-6)  # one line offset pixel


def test_load_wparams_micro_sign_variants():
    raw_file = FakeSMP()
    raw_file._kvPairDict['RealPixelDuration_�s'] = raw_file._kvPairDict.pop('RealPixelDuration_µs')
    wparams = read_smp_utils.load_wparams(raw_file, return_file=False)
    assert wparams['realpixdur'] == 5.


def test_recording_xyz_repr(tmp_path):
    path = write_h5(tmp_path / 'xyz.h5', {'wDataCh0': np.zeros((4, 4, 2))}, wparams_num(nx=4, ny=4, nz=3))
    rec = ScanMRecording(path)
    assert rec.scan_type == 'xyz'
    assert 'xyz: [4 x 4 x 3]' in repr(rec)


def test_recording_stimulator_delay_from_date_and_setup(chirp_path):
    rec = ScanMRecording(chirp_path, setup_id=1, date='2099-01-01')
    rec.compute_triggers()
    assert rec.stimulator_delay == pytest.approx(recording.setup_utils.get_stimulator_delay('2099-01-01', setupid=1))
    assert rec.trigger_times.size == 10
