import numpy as np
import pytest

from djimaging.autorois.roi_canvas import RoiCanvas as _RoiCanvas
from djimaging.tables.core.channel import ChannelTemplate


def RoiCanvas(**kwargs):
    kwargs.setdefault('output_files', ['mask0.npy', 'mask1.npy'])
    return _RoiCanvas(**kwargs)


def make_stacks(ch_names, n_stims=2, shape=(6, 5, 4)):
    rng = np.random.default_rng(0)
    return {name: [rng.random(shape) + i for _ in range(n_stims)] for i, name in enumerate(ch_names)}


def test_legacy_ch0_ch1_arguments():
    stacks = make_stacks(['a', 'b'])
    canvas = RoiCanvas(ch0_stacks=stacks['a'], ch1_stacks=stacks['b'])
    assert canvas.ch_names == ['ch0', 'ch1'] and canvas.primary_channel == 'ch0'
    assert all(a is b for a, b in zip(canvas.ch0_stacks, stacks['a']))
    assert all(a is b for a, b in zip(canvas.ch1_stacks, stacks['b']))
    assert canvas.secondary_channel == 'ch1'
    assert {'stack: ch0[mean]', 'stack: ch0[std]', 'stack: ch1[mean]', 'stack: ch1[std]'} <= set(canvas.backgrounds)
    assert canvas._selected_bg == 'stack: ch0[mean]'


def test_single_channel():
    canvas = RoiCanvas(ch0_stacks=make_stacks(['a'])['a'])
    assert canvas.ch1_stacks is None and canvas.secondary_channel is None
    assert 'stack: ch1[mean]' not in canvas.backgrounds


def test_three_channels_have_backgrounds_and_primary():
    stacks = make_stacks(['green', 'red', 'blue'])
    canvas = RoiCanvas(stacks=stacks, primary_channel='red')
    for name in stacks:
        assert f'stack: {name}[mean]' in canvas.backgrounds and f'stack: {name}[std]' in canvas.backgrounds
    assert canvas._selected_bg == 'stack: red[mean]'
    assert canvas.ch0_stacks is canvas.stacks['red']
    assert canvas.secondary_channel == 'green' and canvas.ch1_stacks is canvas.stacks['green']
    assert canvas.shifts.shape == (2, 2) and canvas.pres_names == ['stim1', 'stim2']


def test_set_primary_channel_resets_ref_corr_map():
    canvas = RoiCanvas(stacks=make_stacks(['a', 'b', 'c']))
    canvas.ref_corr_map = np.zeros((2, 2))
    canvas.set_primary_channel('c')
    assert canvas.primary_channel == 'c' and canvas.ref_corr_map is None
    assert canvas.ch0_stacks is canvas.stacks['c'] and canvas.secondary_channel == 'a'
    with pytest.raises(ValueError):
        canvas.set_primary_channel('d')


def test_backgrounds_follow_stim_and_shift():
    stacks = make_stacks(['a', 'b'])
    canvas = RoiCanvas(stacks=stacks, shifts=[[0, 0], [1, 0]])
    canvas._selected_stim_idx = 1
    bg = canvas.compute_backgrounds()['stack: b[mean]']
    assert np.isnan(bg).any()  # Shifted
    assert np.nanmean(bg) > 1  # Channel b has offset 1, i.e. it is not channel a


@pytest.mark.parametrize('kwargs', [
    dict(stacks=make_stacks(['a']), ch0_stacks=make_stacks(['a'])['a']),
    dict(),
    dict(stacks={}),
    dict(stacks=make_stacks(['a', 'b']), primary_channel='c'),
    dict(stacks={'a': make_stacks(['a'])['a'], 'b': make_stacks(['b'], n_stims=1)['b']}),
    dict(stacks={'a': make_stacks(['a'])['a'], 'b': make_stacks(['b'], shape=(7, 5, 4))['b']}),
])
def test_invalid_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        RoiCanvas(**kwargs)


class FakeChannelTable:
    def __init__(self, rows=()):
        self.rows = [dict(r) for r in rows]

    def __call__(self):
        return self

    def to_dicts(self):
        return self.rows

    def insert(self, rows, skip_duplicates=False):
        self.rows.extend(rows)

    add_channels = ChannelTemplate.add_channels
    get_functional_channels = ChannelTemplate.get_functional_channels
    get_roi_channel = ChannelTemplate.get_roi_channel
    get_ch_names = ChannelTemplate.get_ch_names


def test_add_channels_default_is_first_channel():
    table = FakeChannelTable()
    table.add_channels([('wDataCh0', 0), ('wDataCh1', 1)])
    assert [r['roi_default'] for r in table.rows] == [1, 0]
    assert [r['is_functional'] for r in table.rows] == [1, 1]  # All channels are functional by default


def test_add_channels_roi_channel():
    table = FakeChannelTable()
    table.add_channels([('wDataCh0', 0), ('wDataCh1', 1), ('wDataCh2', 2)], roi_channel='wDataCh1')
    assert table.get_roi_channel() == 'wDataCh1'
    assert table.get_ch_names() == ['wDataCh1', 'wDataCh0', 'wDataCh2']


def test_add_channels_functional_and_background():
    table = FakeChannelTable()
    table.add_channels([('wDataCh0', 0), ('wDataCh1', 1)], functional=['wDataCh0'])
    assert [r['is_functional'] for r in table.rows] == [1, 0]
    assert table.get_functional_channels() == [('wDataCh0', 0)]
    assert table.get_roi_channel() == 'wDataCh0'


def test_roi_channel_defaults_to_first_functional_channel():
    table = FakeChannelTable()
    table.add_channels([('marker', 0), ('wDataCh1', 1), ('wDataCh2', 2)], functional=['wDataCh2', 'wDataCh1'])
    assert table.get_roi_channel() == 'wDataCh1'
    assert table.get_functional_channels() == [('wDataCh1', 1), ('wDataCh2', 2)]
    assert table.get_ch_names() == ['wDataCh1', 'marker', 'wDataCh2']


def test_roi_channel_can_be_background_channel():
    table = FakeChannelTable()
    table.add_channels([('wDataCh0', 0), ('wDataCh1', 1)], functional=['wDataCh0'], roi_channel='wDataCh1')
    assert table.get_roi_channel() == 'wDataCh1'
    assert table.get_functional_channels() == [('wDataCh0', 0)]


@pytest.mark.parametrize('channels, kwargs', [
    ([], {}),
    ([('wDataCh0', 0)], dict(roi_channel='wDataCh1')),
    ([('wDataCh0', 0)], dict(functional=[])),
    ([('wDataCh0', 0)], dict(functional=['wDataCh1'])),
])
def test_add_channels_invalid(channels, kwargs):
    with pytest.raises(ValueError):
        FakeChannelTable().add_channels(channels, **kwargs)


def test_get_roi_channel_fallbacks():
    rows = [dict(ch_id=1, ch_name='b', roi_default=0, is_functional=1),
            dict(ch_id=0, ch_name='a', roi_default=0, is_functional=0)]
    assert FakeChannelTable(rows).get_roi_channel() == 'b'  # first functional channel if none is flagged

    rows[0]['roi_default'] = rows[1]['roi_default'] = 1
    with pytest.raises(ValueError):
        FakeChannelTable(rows).get_roi_channel()

    with pytest.raises(ValueError):
        FakeChannelTable().get_roi_channel()
