import os
import re
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


def test_clock_alignment_in_report(events):
    meta = dict(experiment='e', has_stim_log=True, clock_offset_s=-303.)
    c = eh.clock_alignment(events, meta)
    assert c['status'] == 'aligned' and c['n_logged'] == 3
    assert c['n_in_recording'] == 2 and c['n_fully_in_recording'] == 2 and c['overrunning'] == []
    assert (c['lag_s_min'], c['lag_s_max']) == pytest.approx((1., 2.))

    page = eh.render_html(events, None, meta)
    assert 'Clock alignment (assumed time shift)' in page
    assert '-303.0 s (-5 min 3 s)' in page and '303.0 s (5 min 3 s) ahead of the recording PC clock' in page
    assert 'estimated, not measured' in page
    assert any('shifted by an estimated -303.0 s' in g for g in eh.experiment_gaps(events, meta))

    # A stimulus that outlasts its recording is listed
    stims = _stims()
    stims.loc[1, 't_dur_s'] = 40.  # chirp recording: 30 s, the stimulus starts 2 s after it
    c = eh.clock_alignment(eh.build_events(_recordings(), stims, clock_offset_s=60.), meta)
    assert c['n_fully_in_recording'] == 1
    assert [(r, s) for r, s, _ in c['overrunning']] == [('SMP_GCL0_chirp', 'RGC_Chirp')]
    assert c['overrunning'][0][2] == pytest.approx(12.)

    unaligned = eh.build_events(_recordings(), _stims(), clock_offset_s=None)
    page = eh.render_html(unaligned, None, dict(experiment='e', has_stim_log=True, clock_offset_s=None))
    assert 'NOT aligned' in page
    page = eh.render_html(unaligned, None, dict(experiment='e', has_stim_log=False))
    assert 'No time shift is applied' in page


def test_timeline_abbreviations():
    assert eh._abbreviations(['RGC_MovingBar', 'Chirp', 'Noise_RGC', 'DN', 'chirp1', 'noise_adapt']) == {
        'RGC_MovingBar': 'MB', 'Chirp': 'Ch', 'Noise_RGC': 'No', 'DN': 'DN', 'chirp1': 'Ch1', 'noise_adapt': 'NA'}
    # A name that is its own code keeps it; clashing names take their next candidate, then a number
    assert eh._abbreviations(['MB', 'RGC_MovingBar']) == {'MB': 'MB', 'RGC_MovingBar': 'RMB'}
    # Priority names (logged stimuli) first: a file-name token then gets a number
    assert eh._abbreviations(['MB', 'RGC_MovingBar'], priority=['RGC_MovingBar']) == {
        'RGC_MovingBar': 'MB', 'MB': 'MB~2'}
    assert eh._abbreviations(['Chirp', 'chirp']) == {'Chirp': 'Ch', 'chirp': 'Chi'}
    codes = eh._abbreviations(['Chirp', 'chirp', 'chirpy'])
    assert len(set(codes.values())) == 3


def test_timeline_labels_and_key(events):
    page = eh.render_html(events, None, dict(experiment='e', has_stim_log=True, clock_offset_s=60.))
    key = page[page.index("<div class='tkey'>"):]
    key = key[:key.index('</div></div>')]
    assert 'Recordings (stimulus part of the file name)' in key and 'Stimuli (QDSpy stimulus file)' in key
    assert 'RGC_Chirp' in key and 'noise_adapt' in key  # explained abbreviations of the stimuli
    labels = re.findall(r"<text class='bl[^']*'[^>]*>([^<]+)</text>", page)
    rec_codes = eh._abbreviations(['chirp', 'wip'])  # recordings: stimulus part of the file name
    stim_codes = eh._abbreviations(['noise_adapt', 'RGC_Chirp', 'wip_stim'])  # stimuli: QDSpy stimulus file
    assert set(rec_codes.values()) <= set(labels)
    assert {stim_codes['RGC_Chirp'], stim_codes['wip_stim']} <= set(labels)


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


def _field_recordings():
    rows = [
        dict(rec_name='SMP_C1_GCL0_chirp', kind='response', field='GCL0', rec_start=T0, absx=0., absy=0., absz=35.,
             pixel_size_um=1.5, nxpix=64, nypix=64, scan_type='xy'),
        dict(rec_name='SMP_C1_GCL0_stack', kind='hr', field='GCL0', rec_start=T0 + timedelta(minutes=1), absx=0.,
             absy=0., absz=30., pixel_size_um=0.5, nxpix=512, nypix=512, scan_type='xyz'),
        dict(rec_name='SMP_C1_IPL1_hr', kind='hr', field='IPL1', rec_start=T0 + timedelta(minutes=2), absx=300.,
             absy=-100., absz=20., pixel_size_um=0.5, nxpix=512, nypix=512, scan_type='xy'),
        dict(rec_name='SMP_C1_IPL2_xz', kind='hr', field='IPL2', rec_start=T0 + timedelta(minutes=3), absx=-200.,
             absy=50., absz=20., pixel_size_um=1., nxpix=64, nypix=1, scan_type='xz'),
        dict(rec_name='SMP_C1_od', kind='od', field=None, rec_start=T0 - timedelta(minutes=5), absx=-1000.,
             absy=500., absz=0., scan_type='xy'),
        dict(rec_name='SMP_C1_outline1', kind='outline', field='E1', rec_start=T0 + timedelta(minutes=4), absx=1000.,
             absy=0., absz=0., scan_type='xy'),
        dict(rec_name='SMP_C1_outline2', kind='outline', field='E2', rec_start=T0 + timedelta(minutes=5), absx=1000.,
             absy=800., absz=0., scan_type='xy'),
    ]
    return pd.DataFrame(rows)


def test_build_fields():
    images = {'wDataCh0': np.arange(64 * 64, dtype=float).reshape(64, 64)}
    fields = eh.build_fields(
        _field_recordings(), field_info=dict(GCL0=dict(x_um=1., images=images, image_source='Field.StackAverages')),
        retinal=dict(GCL0=(100., -50., 'RetinalFieldLocation')))
    by_field = fields.set_index('field')
    assert list(fields['field'][:3]) == ['GCL0', 'IPL1', 'IPL2']
    assert by_field.loc['GCL0', 'status'] == 'response' and by_field.loc['IPL1', 'status'] == 'structure'
    assert by_field.loc['GCL0', 'image_rec'] == 'SMP_C1_GCL0_chirp'  # xy response recording first
    assert by_field.loc['GCL0', 'x_um'] == 1.  # from field_info
    assert by_field.loc['GCL0', 'n_response'] == 1 and by_field.loc['GCL0', 'n_structure'] == 1
    assert by_field.loc['IPL1', 'width_um'] == 256. and pd.isna(by_field.loc['IPL2', 'height_um'])
    assert by_field.loc['GCL0', 'ventral_dorsal_pos_um'] == 100.
    assert pd.isna(by_field.loc['IPL1', 'retinal_source'])
    assert set(fields['status']) == set(eh.FIELD_STATUSES)

    page = eh.render_html(eh.build_events(_recordings(), None), None, dict(experiment='e'), fields=fields)
    assert page.count('<image ') == 1 and 'data:image/png;base64,' in page
    assert 'D 100 µm, T 50 µm' in page and '>OD<' in page and '<polyline' in page
    assert "class='card response'" in page and "class='card structure'" in page


def test_field_image():
    stack = np.random.default_rng(0).random((8, 6, 5))
    assert np.allclose(eh.field_image(stack, 'xy'), np.median(stack, axis=2))
    assert np.allclose(eh.field_image(stack, 'xyz'), np.max(stack, axis=2))
    assert eh.field_image(stack[:, :, 0]).shape == (8, 6)


def test_report_exists(events, tmp_path):
    meta = dict(experiment='e', has_stim_log=True, clock_offset_s=60.)
    out = tmp_path / 'exposure_history'
    fields = eh.build_fields(_field_recordings())
    files = eh.export_experiment(str(out), events, meta, fields=fields)
    assert set(files) == set(eh.REPORT_FILES)
    assert all(eh.is_report_file(f) for f in files.values())
    (out / 'notes.txt').write_text('mine')

    with pytest.raises(FileExistsError):
        eh.export_experiment(str(out), events, meta, exists='error')
    with pytest.warns(UserWarning, match='skipped'):
        assert eh.export_experiment(str(out), events, meta) is None  # default: skip
    with pytest.raises(ValueError):
        eh.export_experiment(str(out), events, meta, exists='replace')

    mtime = os.path.getmtime(files['html'])
    assert eh.export_experiment(str(out), events, meta, exists='overwrite') is not None
    assert os.path.getmtime(files['html']) >= mtime
    assert (out / 'notes.txt').read_text() == 'mine'

    # A file with a report name that is not a report is never overwritten, and nothing is written
    (out / eh.REPORT_FILES['html']).write_text('<html>my own page</html>')
    with pytest.raises(FileExistsError, match='not recognised'):
        eh.export_experiment(str(out), events, meta, exists='overwrite')
    assert (out / eh.REPORT_FILES['html']).read_text() == '<html>my own page</html>'


def test_is_report_file_legacy(tmp_path):
    (tmp_path / 'experiment_meta.json').write_text('{"experiment": "e", "gaps": [], "n_recordings": 1}')
    (tmp_path / 'exposure_history.html').write_text('<!doctype html><html><head><title>Light exposure history e')
    (tmp_path / 'exposure_events.csv').write_text('event,t_start,t_end\n')
    (tmp_path / 'prior_exposure.csv').write_text('a,b,c\n')
    assert eh.is_report_file(str(tmp_path / 'experiment_meta.json'))
    assert eh.is_report_file(str(tmp_path / 'exposure_history.html'))
    assert eh.is_report_file(str(tmp_path / 'exposure_events.csv'))
    assert not eh.is_report_file(str(tmp_path / 'prior_exposure.csv'))


def test_html_files_section(events):
    meta = dict(experiment='e', header_path='/data/exp1', unscanned_files=['Other/SMP_x.h5'],
                files=[dict(field='GCL0', rec_name='SMP_GCL0_chirp', kind='response',
                            h5_file='/data/exp1/Pre/SMP_GCL0_chirp.h5',
                            smp_file='/data/exp1/Raw/GCL0_chirp.smp', smh_file=None)])
    page = eh.render_html(events, None, meta)
    textarea = page.split("<textarea", 1)[1].split("</textarea>", 1)[0].split('>', 1)[1]
    assert textarea.split('\n') == ['/data/exp1', '/data/exp1/Raw/GCL0_chirp.smp', '/data/exp1/Pre/SMP_GCL0_chirp.h5',
                                    os.path.join('/data/exp1', 'Other/SMP_x.h5')]
    files_table = page.split('<h2>Files</h2>', 1)[1]
    assert files_table.index('<th>field</th>') < files_table.index('<th>rec_name</th>')
    assert '<td>GCL0</td>' in files_table
    assert '<h2>Files</h2>' not in eh.render_html(events, None, dict(experiment='e'))
