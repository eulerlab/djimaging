import pytest

from djimaging.utils.stimulus_listing import is_listed, is_listed_for, listed_stimuli, split_alias

ALIASES = {
    'chirp': 'chirp_gchirp_globalchirp',
    'movingbar': 'mb_mbar_movingbar_bar',
    'noise': 'dn_noise_dn40m_noise40m',
}


def test_split_alias():
    assert split_alias('Chirp_gChirp__x') == {'chirp', 'gchirp', 'x'}
    assert split_alias('') == set()
    assert split_alias(None) == set()


@pytest.mark.parametrize('token, expected', [
    ('chirp', ['chirp']),
    ('gChirp', ['chirp']),  # case-insensitive
    ('MB', ['movingbar']),
    (' dn ', ['noise']),  # surrounding whitespace is ignored
    ('lChirp', []),  # no partial matches
    ('chir', []),
    ('chirp_mb', []),  # a token is one part of the file name, never several aliases
    ('wipStimulus', []),
    ('', []),
    (None, []),
])
def test_listed_stimuli(token, expected):
    assert listed_stimuli(token, ALIASES) == expected


def test_listed_stimuli_ambiguous_alias():
    aliases = dict(ALIASES, chirp2='chirp_other')
    assert listed_stimuli('chirp', aliases) == ['chirp', 'chirp2']


def test_is_listed_only_for_responses():
    assert is_listed('chirp', ALIASES)
    assert is_listed('chirp', ALIASES, kind='response')
    for kind in ('hr', 'od', 'outline', 'unknown'):
        assert not is_listed('chirp', ALIASES, kind=kind)
    assert not is_listed('wip', ALIASES)


def test_is_listed_for_one_stimulus():
    assert is_listed_for('MBar', 'movingbar', ALIASES['movingbar'])
    assert not is_listed_for('chirp', 'movingbar', ALIASES['movingbar'])
    assert not is_listed_for('chirp', 'chirp', ALIASES['chirp'], kind='hr')


def test_no_stimuli_nothing_listed():
    assert listed_stimuli('chirp', {}) == []
    assert not is_listed('chirp', {})
