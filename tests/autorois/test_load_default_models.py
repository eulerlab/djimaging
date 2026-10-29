import pytest

pytest.importorskip("datajoint")
from djimaging.tables.core import roi_mask


@pytest.fixture
def calls(monkeypatch):
    """Replace all model loaders by fakes that record which models were requested."""
    called = []

    def fake(name):
        def loader(autorois_models, *args, **kwargs):
            called.append(name)
            autorois_models[name] = object()

        return loader

    monkeypatch.setattr(roi_mask, '_add_autorois_unet', fake('UNet'))
    monkeypatch.setattr(roi_mask, '_add_autorois_unet_v1', fake('UNet_v1'))
    monkeypatch.setattr(roi_mask, '_add_cellpose', fake('Cellpose3'))
    monkeypatch.setattr(roi_mask, '_add_mediar', fake('MEDIAR'))
    monkeypatch.setattr(roi_mask, '_add_autorois_corr', fake('CorrRoiMask'))
    return called


def test_default_only_loads_latest_unet(calls):
    models = roi_mask.load_default_autorois_models()
    assert list(models) == ['UNet', 'CorrRoiMask']
    assert 'MEDIAR' not in calls and 'Cellpose3' not in calls and 'UNet_v1' not in calls


def test_default_skips_failing_unet_with_warning(calls, monkeypatch):
    def failing(autorois_models):
        raise FileNotFoundError('no weights')

    monkeypatch.setattr(roi_mask, '_add_autorois_unet', failing)
    with pytest.warns(UserWarning, match='no weights'):
        models = roi_mask.load_default_autorois_models()
    assert list(models) == ['CorrRoiMask']


def test_explicit_models_are_loaded(calls):
    models = roi_mask.load_default_autorois_models(models=['UNet', 'Cellpose3', 'MEDIAR'])
    assert list(models) == ['UNet', 'Cellpose3', 'MEDIAR']


def test_explicit_model_failure_raises(calls, monkeypatch):
    def failing(autorois_models):
        raise ImportError('no mediar')

    monkeypatch.setattr(roi_mask, '_add_mediar', failing)
    with pytest.raises(ImportError):
        roi_mask.load_default_autorois_models(models=['MEDIAR'])


def test_unknown_model_raises(calls):
    with pytest.raises(ValueError):
        roi_mask.load_default_autorois_models(models=['Nope'])
