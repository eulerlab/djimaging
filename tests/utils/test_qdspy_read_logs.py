"""Tests of the QDSpy log parser with small hand-written logs (format as written by QDSpy, see QDSpy-log-reader)."""
from datetime import time

import pytest

from djimaging.utils.qdspy.read_logs import parse_stimulus_log

STAGE = "{'scaling_x': 0.7, 'scaling_y': 1.3, 'offset_x': -44, 'offset_y': 61, 'rotation': 0.0}"


def _line(stamp: str, kind: str, msg: str) -> str:
    """One log line: 15 chars time stamp, info type right-aligned in 8 chars, a space, the message."""
    return f"{stamp}{kind:>8} {msg}\n"


def _start(stamp, path, md5='abc'):
    return _line(stamp, 'DATA', f"{{'stimFileName': {path!r}, 'stimState': 'STARTED', 'stimMD5': {md5!r}}}")


def _end(stamp, path, state='FINISHED'):
    return _line(stamp, 'DATA', f"{{'stimFileName': {path!r}, 'stimState': {state!r}}}")


def _frames(stamp, n_frames=600, freq=60.0, n_dropped=0):
    return _line(stamp, 'DATA', f"{{'avgFreq_Hz': {freq}, 'nFrames': {n_frames}, 'nDroppedFrames': {n_dropped}}}")


def _write(tmp_path, lines, name='20250506_100000.log'):
    path = tmp_path / name
    path.write_text(''.join(lines), encoding='utf-8')
    return str(path)


CHIRP = 'C:\\Users\\X\\QDSpy\\Stimuli\\chirp'
NOISE = 'C:\\Users\\X\\QDSpy\\Stimuli\\noise'


def test_parse_two_stimuli(tmp_path):
    log = _write(tmp_path, [
        _line('20250506_100000', 'DEBUG', '... done'),
        _line('20250506_100001', 'INFO', 'Python : v3.6'),
        _start('20250506_100010', CHIRP, md5='md5chirp'),
        _line('20250506_100010', 'DATA', STAGE),
        _line('20250506_100010', 'DATA', "{'nTrials': 3, 'stimFileName': 'chirp'}"),
        _end('20250506_100042', CHIRP),
        _frames('20250506_100042', n_frames=1920, freq=60.0, n_dropped=2),
        _start('20250506_100100', NOISE),
        _line('20250506_100100', 'DATA', "{'SequenceUsed': 4}"),
        _end('20250506_100105', NOISE, state='ABORTED'),
    ])
    stims, stats = parse_stimulus_log(log, verbose=False)

    assert [s['index'] for s in stims] == [0, 1]
    chirp, noise = stims
    assert chirp['stimFileName'] == 'chirp'
    assert chirp['stimPath'] == 'C:/Users/X/QDSpy/Stimuli'
    assert chirp['stimMD5'] == 'md5chirp'
    assert chirp['t_abs_s'] == 10.
    assert chirp['t_since_last_s'] == 10.
    assert chirp['t_start'] == time(10, 0, 10)
    assert chirp['t_end'] == time(10, 0, 42)
    assert chirp['t_dur_s'] == 32.
    assert chirp['aborted'] is False
    assert chirp['t_dur_s_calc'] == pytest.approx(32.)
    assert chirp['nDroppedFrames'] == 2
    assert chirp['params'] == {'nTrials': 3, 'stimFileName': 'chirp'}
    assert chirp['stageInfo']['offset_x'] == -44
    assert 'sequenceUsed' not in chirp

    assert noise['aborted'] is True
    assert noise['t_abs_s'] == 60.
    assert noise['t_since_last_s'] == 18.  # since end of chirp
    assert noise['t_dur_s'] == 5.
    assert noise['sequenceUsed'] == 4
    assert 'params' not in noise
    assert 't_dur_s_calc' not in noise  # no frame statistics logged

    assert stats == dict(nLinesTotal=10, nLinesData=8, nLinesErr=0, nErr=0)


def test_stim_without_path_and_forward_slashes(tmp_path):
    log = _write(tmp_path, [
        _start('20250506_100000', 'chirp'),
        _end('20250506_100001', 'chirp'),
        _start('20250506_100002', 'D:/stimuli/sub/noise'),
        _end('20250506_100003', 'D:\\stimuli\\sub\\noise'),  # same file, other separators
    ])
    stims, stats = parse_stimulus_log(log, verbose=False)
    assert stims[0]['stimFileName'] == 'chirp'
    assert stims[0]['stimPath'] == ''
    assert stims[1]['stimFileName'] == 'noise'
    assert stims[1]['stimPath'] == 'D:/stimuli/sub'
    assert stats['nErr'] == 0


def test_params_last_assignment_wins(tmp_path):
    """Older QDSpy versions emitted params after stage info; the last non-stage dict is used."""
    log = _write(tmp_path, [
        _start('20250506_100000', CHIRP),
        _line('20250506_100000', 'DATA', "{'a': 1}"),
        _line('20250506_100000', 'DATA', STAGE),
        _line('20250506_100000', 'DATA', "{'b': 2}"),
        _end('20250506_100001', CHIRP),
    ])
    stims, _ = parse_stimulus_log(log, verbose=False)
    assert stims[0]['params'] == {'b': 2}
    assert set(stims[0]['stageInfo']) == {'scaling_x', 'scaling_y', 'offset_x', 'offset_y', 'rotation'}


def test_stage_info_with_extra_key_is_params(tmp_path):
    log = _write(tmp_path, [
        _start('20250506_100000', CHIRP),
        _line('20250506_100000', 'DATA', STAGE[:-1] + ", 'extra': 1}"),
        _end('20250506_100001', CHIRP),
    ])
    stims, _ = parse_stimulus_log(log, verbose=False)
    assert 'stageInfo' not in stims[0]
    assert stims[0]['params']['extra'] == 1


def test_errors_are_counted(tmp_path, capsys):
    log = _write(tmp_path, [
        _line('20250506_100000', 'DEBUG', 'start'),
        _line('20250506_100000', 'DATA', "{'orphan': 1}"),  # data w/o start
        _end('20250506_100001', CHIRP),  # end w/o start
        _line('20250506_100002', 'DATA', "{'broken': "),  # not parseable
        _line('20250506_100002', 'DATA', "[1, 2, 3]"),  # not a dict
        _start('20250506_100003', CHIRP),
        _start('20250506_100004', NOISE),  # two consecutive starts: chirp is dropped
        _end('20250506_100005', CHIRP),  # path differs from start (noise)
        'not a log line\n',
        _line('20250506_10000X', 'DATA', "{'bad': 'stamp'}"),
    ])
    stims, stats = parse_stimulus_log(log, verbose=True)
    out = capsys.readouterr().out

    assert len(stims) == 1
    assert stims[0]['stimFileName'] == 'noise'
    assert stims[0]['t_abs_s'] == 4.
    assert stats['nLinesTotal'] == 10
    assert stats['nLinesErr'] == 2
    assert stats['nErr'] == 4
    assert stats['nLinesData'] == 5
    for msg in ['Data w/o start', 'end w/o start', 'parsing line', 'Two consecutive', 'differ']:
        assert msg in out


def test_frame_stats_before_any_stimulus_are_ignored(tmp_path):
    log = _write(tmp_path, [_frames('20250506_100000')])
    stims, stats = parse_stimulus_log(log, verbose=False)
    assert stims == []
    assert stats['nErr'] == 0


def test_unfinished_stimulus_is_not_returned(tmp_path):
    log = _write(tmp_path, [_start('20250506_100000', CHIRP)])
    stims, _ = parse_stimulus_log(log, verbose=False)
    assert stims == []


def test_info_type_is_case_insensitive(tmp_path):
    lines = [_start('20250506_100000', CHIRP).replace('DATA', 'data'), _end('20250506_100001', CHIRP)]
    stims, _ = parse_stimulus_log(_write(tmp_path, lines), verbose=False)
    assert len(stims) == 1


def test_crlf_line_endings(tmp_path):
    path = tmp_path / 'crlf.log'
    path.write_bytes(''.join([_start('20250506_100000', CHIRP), _end('20250506_100001', CHIRP)])
                     .replace('\n', '\r\n').encode('utf-8'))
    stims, stats = parse_stimulus_log(str(path), verbose=False)
    assert len(stims) == 1
    assert stats['nLinesErr'] == 0


def test_zero_frame_rate_does_not_crash(tmp_path):
    log = _write(tmp_path, [
        _start('20250506_100000', CHIRP),
        _end('20250506_100000', CHIRP, state='ABORTED'),
        _frames('20250506_100000', n_frames=0, freq=0, n_dropped=0),
    ])
    stims, _ = parse_stimulus_log(log, verbose=False)
    assert len(stims) == 1
    assert stims[0]['aborted'] is True


def test_last_line_without_newline(tmp_path):
    log = _write(tmp_path, [_start('20250506_100000', CHIRP), _end('20250506_100001', CHIRP).rstrip('\n')])
    stims, stats = parse_stimulus_log(log, verbose=False)
    assert stats['nLinesErr'] == 0
    assert len(stims) == 1


def test_first_line_without_time_stamp(tmp_path):
    log = _write(tmp_path, ['\n', _start('20250506_100000', CHIRP), _end('20250506_100001', CHIRP)])
    stims, _ = parse_stimulus_log(log, verbose=False)
    assert len(stims) == 1
    assert stims[0]['t_abs_s'] == 0.
