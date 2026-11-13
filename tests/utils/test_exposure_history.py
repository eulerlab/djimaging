from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from djimaging.utils import exposure_history as eh

T0 = datetime(2024, 3, 1, 10, 0, 0)


def _recordings():
    rows = [
        dict(rec_name='SMP_GCL0_outline', kind='outline', field='GCL0', rec_start=T0, rec_dur_s=3.,
             absx=0., absy=0., absz=30., zoom=0.65, pixel_size_um=1., nxpix=512, nypix=512, scan_type='xy',
             real_pix_dur_us=1.6),
        dict(rec_name='SMP_GCL0_chirp', kind='response', field='GCL0', rec_start=T0 + timedelta(minutes=1),
             rec_dur_s=30., stim_token='chirp', stim_name='chirp', listed=True, absx=0., absy=0., absz=35.,
             zoom=3., pixel_size_um=1.7, nxpix=64, nypix=64, scan_type='xy', real_pix_dur_us=1.6,
             has_presentation=True, n_traces=40, motion_trust_median=0.93),
        dict(rec_name='SMP_GCL0_wip', kind='response', field='GCL0', rec_start=T0 + timedelta(minutes=2),
             rec_dur_s=60., stim_token='wip', stim_name=None, listed=False, stopped_early=True,
             absx=0., absy=0., absz=35., zoom=3., pixel_size_um=1.7, nxpix=64, nypix=64, scan_type='xy'),
        dict(rec_name='SMP_GCL1_chirp', kind='response', field='GCL1', rec_start=T0 + timedelta(minutes=10),
             rec_dur_s=30., stim_token='chirp', stim_name='chirp', listed=True, absx=200., absy=0., absz=35.,
             zoom=3., pixel_size_um=1.7, nxpix=64, nypix=64, scan_type='xy', has_presentation=False),
        dict(rec_name='SMP_stack', kind='hr', field=None, rec_start=None, rec_dur_s=None, scan_type='xyz'),
    ]
    return pd.DataFrame(rows)


def _stims(offset_s=60.):
    # QDSpy clock: ScanM time - offset
    def qdspy(dt):
        return timedelta(seconds=(dt - datetime.combine(dt.date(), datetime.min.time())).total_seconds() - offset_s)
    return pd.DataFrame([
        dict(log_idx=0, stim_idx=0, stim_file_name='noise_adapt', stim_md5='m_noise', stim_name=None,
             t_start=qdspy(T0 - timedelta(minutes=2)), t_dur_s=90., aborted=False, rec_name=None),
        dict(log_idx=0, stim_idx=1, stim_file_name='RGC_Chirp', stim_md5='m_chirp', stim_name='chirp',
             t_start=qdspy(T0 + timedelta(minutes=1, seconds=2)), t_dur_s=28., aborted=False,
             rec_name='SMP_GCL0_chirp'),
        dict(log_idx=0, stim_idx=2, stim_file_name='wip_stim', stim_md5='m_wip', stim_name=None,
             t_start=qdspy(T0 + timedelta(minutes=2, seconds=1)), t_dur_s=50., aborted=True,
             rec_name='SMP_GCL0_wip'),
    ])


def _levels():
    return pd.DataFrame([
        dict(match_md5='m_chirp', match_file_name=None, match_stim_name='chirp', mean_uv=127.5, mean_green=127.5,
             extent_x_um=1000., extent_y_um=1000.),
        dict(match_md5=None, match_file_name='NOISE_ADAPT', match_stim_name=None, mean_uv=127.5, mean_green=127.5,
             extent_x_um=None, extent_y_um=None),
    ])


@pytest.fixture
def events():
    return eh.build_events(_recordings(), _stims(), clock_offset_s=60., light_levels=_levels())


def _by(events, **kw):
    mask = np.ones(len(events), dtype=bool)
    for k, v in kw.items():
        mask &= (events[k] == v).to_numpy()
    rows = events[mask]
    assert len(rows) == 1, rows
    return rows.iloc[0]


def test_events_order_and_counts(events):
    assert list(events.columns) == eh.EVENT_COLUMNS
    assert (events['event'] == 'laser_scan').sum() == 5
    # 3 logged stimuli + GCL1 chirp from the file name (not in the log)
    assert (events['event'] == 'led_stimulus').sum() == 4
    timed = events['t_start'].dropna()
    assert timed.is_monotonic_increasing
    assert events['t_start'].iloc[-1] is None or pd.isna(events['t_start'].iloc[-1])  # stack without time last
    assert events['t_start_rel_s'].iloc[0] == 0.


def test_stimulus_without_recording_takes_next_position(events):
    e = _by(events, stimulus='noise_adapt')
    assert e['t_start'] == T0 - timedelta(minutes=2)
    assert e['time_source'] == 'qdspy+offset'
    assert e['position_source'] == 'next_recording'
    assert e['field'] == 'GCL0' and e['field_source'] == 'next_recording'
    assert 'position_from_next_recording' in e['incomplete']
    assert e['listed'] == 'unlisted'


def test_logged_stimulus_simultaneous(events):
    e = _by(events, stimulus='RGC_Chirp')
    assert e['position_source'] == 'simultaneous_recording'
    assert e['z_um'] == 35.
    assert e['mean_uv'] == 127.5
    assert e['listed'] == 'listed'
    assert e['incomplete'] == ''


def test_unlisted_stimulus_named_and_flagged(events):
    e = _by(events, stimulus='wip_stim')
    assert e['listed'] == 'unlisted'
    assert e['aborted']
    assert 'light_levels' in e['incomplete']
    laser = _by(events, event='laser_scan', rec_name='SMP_GCL0_wip')
    assert laser['listed'] == 'unlisted'
    assert laser['has_presentation'] is None  # never a presentation, no processing columns
    assert 'stopped_early' in laser['incomplete']


def test_stimulus_from_file_name(events):
    e = _by(events, event='led_stimulus', rec_name='SMP_GCL1_chirp')
    assert e['time_source'] == 'recording'
    assert 'stimulus_from_file_name' in e['incomplete'] and 'not_in_log' in e['incomplete']
    assert e['mean_uv'] == 127.5  # matched by stimulus name
    laser = _by(events, event='laser_scan', rec_name='SMP_GCL1_chirp')
    assert 'no_presentation' in laser['incomplete']


def test_laser_scan_area_and_gaps(events):
    e = _by(events, event='laser_scan', rec_name='SMP_GCL0_chirp')
    assert e['scan_width_um'] == pytest.approx(1.7 * 64)
    assert e['scan_height_um'] == pytest.approx(1.7 * 64)
    stack = _by(events, rec_name='SMP_stack')
    for tag in ('start_time', 'duration', 'field', 'position', 'scan_area'):
        assert tag in stack['incomplete']


def test_prior_exposure(events):
    prior = eh.summarize_prior_exposure(events).set_index('rec_name')
    gcl0 = prior.loc['SMP_GCL0_chirp']
    assert gcl0['prior_laser_s_this_field'] == pytest.approx(3.)
    assert gcl0['prior_led_s_this_field'] == pytest.approx(90.)
    assert gcl0['prior_led_s_this_field_inferred'] == pytest.approx(90.)
    assert gcl0['prior_stimuli_this_field'] == 'noise_adapt*'
    assert 'position_from_next_recording' in gcl0['incomplete']

    gcl1 = prior.loc['SMP_GCL1_chirp']
    assert gcl1['prior_laser_s_other_fields'] == pytest.approx(3. + 30. + 60.)
    assert gcl1['prior_laser_s_this_field'] == 0.
    assert gcl1['prior_led_s'] == pytest.approx(90. + 28. + 50.)
    assert gcl1['prior_led_s_other_fields'] == pytest.approx(168.)
    assert gcl1['prior_led_s_levels_unknown'] == pytest.approx(50.)
    assert gcl1['prior_led_uv_dose'] == pytest.approx(127.5 * (90. + 28.))
    assert gcl1['prior_fields'] == 'GCL0'
    assert 'events_without_time' in gcl1['incomplete']

    assert prior.loc['SMP_stack']['incomplete'] == 'start_time'


def test_overlapping_event_clipped_at_recording_start():
    recs = pd.DataFrame([dict(rec_name='a', kind='response', field='F', rec_start=T0, rec_dur_s=30.,
                              stim_token='x', listed=False, absx=0., absy=0., absz=0.)])
    stims = pd.DataFrame([dict(stim_file_name='x', t_start=timedelta(hours=9, minutes=59, seconds=50),
                               t_dur_s=40., rec_name='a')])
    events = eh.build_events(recs, stims, clock_offset_s=0.)
    prior = eh.summarize_prior_exposure(events)
    assert prior['prior_led_s'].iloc[0] == pytest.approx(10.)


def test_unknown_clock_offset():
    events = eh.build_events(_recordings(), _stims(), clock_offset_s=None)
    e = _by(events, stimulus='RGC_Chirp')
    assert e['time_source'] == 'qdspy_unaligned'
    assert 'clock_offset' in e['incomplete']
    assert e['t_start'] == T0 + timedelta(seconds=2)  # QDSpy clock unchanged
    gaps = eh.experiment_gaps(events, dict(has_stim_log=True, clock_offset_s=None))
    assert any('Clock offset' in g for g in gaps)


def test_no_stim_log():
    events = eh.build_events(_recordings(), None, exp_date=date(2024, 3, 1))
    led = events[events['event'] == 'led_stimulus']
    assert sorted(led['rec_name']) == ['SMP_GCL0_chirp', 'SMP_GCL0_wip', 'SMP_GCL1_chirp']
    assert not led['incomplete'].str.contains('not_in_log').any()
    gaps = eh.experiment_gaps(events, dict(has_stim_log=False))
    assert any('No QDSpy log' in g for g in gaps)
    assert any('No stimulator' in g for g in gaps)
    for g in eh.ALWAYS_GAPS:
        assert g in gaps


def test_lookup_light_levels_order():
    levels = pd.DataFrame([
        dict(match_md5=None, match_file_name='a', match_stim_name=None, mean_uv=1., mean_green=1.),
        dict(match_md5='h', match_file_name=None, match_stim_name=None, mean_uv=2., mean_green=np.nan),
    ])
    assert eh.lookup_light_levels(levels, md5='h', file_name='A')['mean_uv'] == 2.
    assert eh.lookup_light_levels(levels, md5='h')['mean_green'] is None
    assert eh.lookup_light_levels(levels, md5='other', file_name='A')['mean_uv'] == 1.
    assert eh.lookup_light_levels(levels, stim_name='a') is None
    assert eh.lookup_light_levels(None, md5='h') is None


def test_to_display_unknown_vs_not_applicable(events):
    disp = eh.to_display(events)
    led = disp[disp['stimulus'] == 'wip_stim'].iloc[0]
    assert led['zoom'] == eh.NOT_APPLICABLE
    assert led['mean_uv'] == eh.UNKNOWN
    laser = disp[(disp['event'] == 'laser_scan') & (disp['rec_name'] == 'SMP_GCL0_chirp')].iloc[0]
    assert laser['mean_uv'] == eh.NOT_APPLICABLE
    outline = disp[disp['rec_name'] == 'SMP_GCL0_outline'].iloc[0]
    assert outline['listed'] == eh.NOT_APPLICABLE
    assert disp.isna().sum().sum() == 0


def test_csv_roundtrip_and_html(events, tmp_path):
    meta = dict(experiment='2024-03-01_1_Test', setupid='1', has_stim_log=True, clock_offset_s=60.,
                stimulator_id=None)
    files = eh.export_experiment(str(tmp_path / eh.experiment_folder_name(
        dict(date=date(2024, 3, 1), exp_num=1, experimenter='Test'))), events, meta)
    assert files['events'].replace('\\', '/').split('/')[-2] == '2024-03-01_1_Test'

    raw = pd.read_csv(files['events'], keep_default_na=False)
    assert (raw == eh.UNKNOWN).any().any() and (raw == eh.NOT_APPLICABLE).any().any()

    ev, prior, meta_read = eh.read_csv(str(tmp_path / '2024-03-01_1_Test'))
    assert len(ev) == len(events)
    assert ev['x_um'].dtype.kind == 'f'
    assert meta_read['gaps'][:len(eh.ALWAYS_GAPS)] == list(eh.ALWAYS_GAPS)
    assert 'SMP_GCL1_chirp' in set(prior['rec_name'])

    page = eh.render_html(ev, prior, meta_read)
    assert page.startswith('<!doctype html>')
    assert '<svg' in page and 'noise_adapt' in page and "class='unknown'" in page
    with open(files['html'], encoding='utf-8') as f:
        assert '<svg' in f.read()


def test_html_escapes():
    recs = pd.DataFrame([dict(rec_name='<b>x</b>', kind='response', field='F&1', rec_start=T0, rec_dur_s=1.,
                              stim_token='<s>', listed=False, absx=0., absy=0., absz=0.)])
    events = eh.build_events(recs, None)
    page = eh.render_html(events, None, dict(experiment='e'))
    assert '<b>x</b>' not in page and '&lt;b&gt;x&lt;/b&gt;' in page


def test_nwb_roundtrip(events):
    pynwb = pytest.importorskip('pynwb')
    from datetime import timezone
    nwbfile = pynwb.NWBFile(session_description='test', identifier='id',
                            session_start_time=datetime(2024, 3, 1, 9, 0, tzinfo=timezone.utc))
    eh.write_nwb(nwbfile, events, meta=dict(experiment='e', clock_offset_s=60.), this_field='GCL0')
    df, meta = eh.read_nwb(nwbfile)
    assert len(df) == len(events)
    assert meta['experiment'] == 'e'
    assert df['is_this_field'].sum() == (events['field'] == 'GCL0').sum()
    page = eh.render_html(df, None, meta)
    assert '<svg' in page
