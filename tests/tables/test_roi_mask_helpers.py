"""Tests for the pure module-level helpers of djimaging.tables.core.roi_mask."""
import warnings

import numpy as np
import pytest

from djimaging.tables.core import roi_mask as rm


class _Model:
    def __init__(self):
        self.calls = []

    def create_mask_from_data(self, **kw):
        self.calls.append(kw)
        return [[1, 0], [0, 2]]


def _data(ch_names=('wDataCh0', 'wDataCh1'), n_stacks=2):
    stacks = {ch: [np.full((2, 2, 3), i + 10 * c, dtype=float) for i in range(n_stacks)]
              for c, ch in enumerate(ch_names)}
    return dict(stacks=stacks, n_artifact=1, roi_channel='wDataCh0', ch_names=list(ch_names),
                pixel_size_d1_d2=(1., 1.))


@pytest.mark.parametrize('path,expected', [
    ('a/b_ROIs.npy', 'numpy'), ('a/b.NPY', 'numpy'), ('b.pkl', 'pickle'), ('b.pickle', 'pickle'),
    ('b.h5', 'igor_h5'), ('b.HDF5', 'igor_h5'), ('b.txt', 'other'), ('noext', 'other'),
])
def test_roi_mask_source_from_path(path, expected):
    assert rm._roi_mask_source_from_path(path) == expected


def test_get_autorois_model_defaults_and_given():
    m = object()
    assert rm._get_autorois_model(None, 'default_rgc', {'UNet': m}) == ('UNet', m, False)
    assert rm._get_autorois_model(None, 'default_bc', {'CorrRoiMask': m}) == ('CorrRoiMask', m, False)


def test_get_autorois_model_missing_raises():
    with pytest.raises(ValueError):
        rm._get_autorois_model('UNet', 'default_rgc', {'Other': object()})


def test_get_autorois_model_loads_when_none(monkeypatch):
    m = object()
    seen = {}

    def fake_load(kind, models):
        seen.update(kind=kind, models=models)
        return {'UNet': m}

    monkeypatch.setattr(rm, 'load_default_autorois_models', fake_load)
    assert rm._get_autorois_model(None, 'default_rgc', None) == ('UNet', m, True)
    assert seen == dict(kind='default_rgc', models=['UNet'])


def test_get_autorois_model_load_returns_none(monkeypatch):
    monkeypatch.setattr(rm, 'load_default_autorois_models', lambda kind, models: None)
    with pytest.raises(ValueError):
        rm._get_autorois_model('UNet', 'default_rgc', None)


def test_predict_roi_mask_single_stack_with_second_channel():
    model, data = _Model(), _data()
    out = rm._predict_roi_mask(model, data, ref_idx=1)
    assert isinstance(out, np.ndarray) and out.shape == (2, 2)
    kw = model.calls[0]
    assert np.all(kw['ch0_stack'] == 1) and np.all(kw['ch1_stack'] == 11)
    assert kw['n_artifact'] == 1 and 'multiple_stacks' not in kw


def test_predict_roi_mask_single_channel():
    model = _Model()
    rm._predict_roi_mask(model, _data(ch_names=('wDataCh0',)))
    assert model.calls[0]['ch1_stack'] is None


def test_predict_roi_mask_all_stacks():
    model, data = _Model(), _data()
    rm._predict_roi_mask(model, data, use_all_stacks=True)
    kw = model.calls[0]
    assert kw['multiple_stacks'] is True and len(kw['ch0_stack']) == 2 and len(kw['ch1_stack']) == 2


def test_remove_vessel_rois_noop_without_threshold():
    mask = np.array([[1, 1], [0, 0]])
    assert rm._remove_vessel_rois(mask, _data(), 0, None, 0.2, 1) is mask


def test_remove_vessel_rois_warns_without_second_channel():
    mask = np.array([[1, 1], [0, 0]])
    with pytest.warns(UserWarning, match='second channel'):
        out = rm._remove_vessel_rois(mask, _data(ch_names=('wDataCh0',)), 0, 2., 0.2, 1)
    assert out is mask


def test_remove_vessel_rois_delegates(monkeypatch):
    mask = np.array([[1, 1], [0, 0]])
    data = _data()
    seen = {}

    def fake_est(alt, main, n_artifact, alt_z):
        seen.update(alt=alt, main=main, n_artifact=n_artifact, alt_z=alt_z)
        return 'vm'

    def fake_rm(m, vm, max_overlap, connectivity, verbose):
        seen.update(vm=vm, max_overlap=max_overlap, connectivity=connectivity)
        return 'out'

    monkeypatch.setattr(rm, 'estimate_vessel_mask', fake_est)
    monkeypatch.setattr(rm, 'remove_vessel_rois', fake_rm)
    assert rm._remove_vessel_rois(mask, data, 1, 3., 0.5, 2) == 'out'
    assert np.all(seen['alt'] == 11) and np.all(seen['main'] == 1) and seen['alt'].shape == (2, 2)
    assert seen['alt_z'] == 3. and seen['max_overlap'] == 0.5 and seen['connectivity'] == 2


def test_free_memory_runs():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rm._free_memory()


def test_load_stack_data(monkeypatch):
    def fake_load(f, from_raw_data, ch_names):
        return {ch: np.zeros((2, 2, 1)) + len(f) for ch in ch_names}, {}

    monkeypatch.setattr(rm.read_utils, 'load_stacks', fake_load)
    files = ['/d/Pre/a.h5', '/d/Pre/bb.h5']
    stacks, outs = rm.load_stack_data(files, ['wDataCh0', 'wDataCh1'], from_raw_data=False,
                                      roi_mask_dir='ROIs', file_format='pickle')
    assert set(stacks) == {'wDataCh0', 'wDataCh1'}
    assert [len(s) for s in stacks.values()] == [2, 2]
    assert stacks['wDataCh0'][0].mean() == len(files[0]) and stacks['wDataCh0'][1].mean() == len(files[1])
    assert len(outs) == 2 and all(o.endswith('.pkl') for o in outs)
    assert all('ROIs' in o.replace(chr(92), '/') for o in outs)


def test_load_stack_data_empty():
    stacks, outs = rm.load_stack_data([], ['wDataCh0'], from_raw_data=True)
    assert stacks == {'wDataCh0': []} and outs == []


def test_load_default_autorois_models_errors():
    with pytest.raises(ValueError, match='Unknown'):
        rm.load_default_autorois_models(models=['nope'])
    with pytest.raises(NotImplementedError):
        rm.load_default_autorois_models(kind='default_bc', models=['UNet'])


def test_load_default_autorois_models_dispatch(monkeypatch):
    monkeypatch.setattr(rm, '_add_autorois_unet', lambda d: d.__setitem__('UNet', 'u'))
    monkeypatch.setattr(rm, '_add_autorois_corr', lambda d, kind: d.__setitem__('CorrRoiMask', kind))
    assert rm.load_default_autorois_models() == {'UNet': 'u', 'CorrRoiMask': 'default_rgc'}
    assert rm.load_default_autorois_models(models='UNet') == {'UNet': 'u'}
    assert rm.load_default_autorois_models(kind='default_bc') == {'CorrRoiMask': 'default_bc'}
    assert rm.load_default_autorois_models(models=[]) is None


def test_load_default_autorois_models_default_unet_failure_warns(monkeypatch):
    def boom(d):
        raise RuntimeError('x')

    monkeypatch.setattr(rm, '_add_autorois_unet', boom)
    monkeypatch.setattr(rm, '_add_autorois_corr', lambda d, kind: None)
    with pytest.warns(UserWarning, match='UNet'):
        assert rm.load_default_autorois_models() is None
