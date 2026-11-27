"""OsDsIndexesTemplate.make on synthetic moving bar snippets (fake relations, no database): angle conventions."""
import numpy as np
import pytest

from djimaging.tables.response.movingbar.orientation import OsDsIndexesTemplate
from tests.tables._fake_dj import FakeRel, make_fake

DIRECTIONS_DEG = np.array([0, 180, 45, 225, 90, 270, 135, 315])
DT = 0.128


def _snippets(gains_fn, n_reps=3):
    times = np.arange(32) * DT
    kernel = np.exp(-((times - 1.7) / 0.45) ** 2)
    gains = gains_fn(np.deg2rad(DIRECTIONS_DEG))
    snippets = [kernel * g * r for r in np.linspace(0.95, 1.05, n_reps) for g in gains]
    return np.stack(snippets, axis=1)


def _osds_table(snippets, reduced_storage=True, snippet_base_dt=5 * 0.128):
    trial_info = [dict(name=d) for d in DIRECTIONS_DEG]
    return make_fake(OsDsIndexesTemplate, ['make'], _reduced_storage=reduced_storage, _n_shuffles=50, _seed=0,
                     stimulus_table=FakeRel(dict(trial_info=trial_info, snippet_base_dt=snippet_base_dt)),
                     snippets_table=FakeRel(dict(snippets_dt=DT, snippets=snippets)))


@pytest.mark.parametrize('pref_deg', [0., 45., 90., 180., 270., 315.])
def test_osds_make_preferred_direction_radians(pref_deg):
    table = _osds_table(_snippets(lambda d: 1 + 0.8 * np.cos(d - np.deg2rad(pref_deg))))
    table.make(dict(roi_id=1))
    entry = table.inserted[0]
    # pref_dir is stored in radians, possibly wrapped to (-pi, pi]
    assert np.mod(entry['pref_dir'], 2 * np.pi) == pytest.approx(np.deg2rad(pref_deg) % (2 * np.pi), abs=1e-6)
    assert entry['ds_index'] > 0.3
    assert entry['os_index'] == pytest.approx(0., abs=1e-6)
    assert entry['ds_pvalue'] < 0.05
    assert 'ds_null' not in entry


@pytest.mark.parametrize('pref_or_deg', [0., 45., 90., 135.])
def test_osds_make_preferred_orientation_radians(pref_or_deg):
    table = _osds_table(_snippets(lambda d: 1 + 0.8 * np.cos(2 * (d - np.deg2rad(pref_or_deg)))),
                        reduced_storage=False)
    table.make(dict(roi_id=1))
    entry = table.inserted[0]
    assert np.mod(entry['pref_or'], np.pi) == pytest.approx(np.deg2rad(pref_or_deg) % np.pi, abs=1e-6)
    assert entry['ds_index'] == pytest.approx(0., abs=1e-6)
    assert entry['os_index'] > 0.3
    assert entry['ds_null'].shape == (50,)
    assert entry['avg_sorted_resp'].shape[1] == 8


def test_osds_make_warns_without_trial_baseline():
    table = _osds_table(_snippets(lambda d: 1 + 0.8 * np.cos(d)), snippet_base_dt=None)
    with pytest.warns(UserWarning, match='snippet_base_dt'):
        table.make(dict(roi_id=1, stim_name='movingbar'))
