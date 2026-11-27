"""The gate: unlisted recordings never become Presentations, and every table with stimulus-specific data depends on
Presentation, so unlisted recordings never get traces, snippets or averages."""
import re

import pandas as pd
import pytest

from djimaging.tables import core
from djimaging.tables.core.presentation import check_listed_for_presentation


def _foreign_keys(table_class, **attrs) -> set:
    table_class = type(f"Test{table_class.__name__}", (table_class,), attrs)
    return set(re.findall(r'^\s*->\s*self\.(\w+)', table_class().definition, flags=re.M))


def test_presentation_depends_on_stimulus():
    field_flags = type("FieldFlags", (), {"incl_region": True, "incl_cond1": True, "incl_cond2": False,
                                          "incl_cond3": False})
    assert 'stimulus_table' in _foreign_keys(core.PresentationTemplate, field_table=field_flags)


@pytest.mark.parametrize('table_class, upstream', [
    (core.TracesTemplate, 'presentation_table'),
    (core.PreprocessTracesTemplate, 'traces_table'),
    (core.SnippetsTemplate, 'preprocesstraces_table'),
    (core.ResampledAveragesTemplate, 'snippets_table'),
])
def test_downstream_tables_depend_on_presentation(table_class, upstream):
    assert upstream in _foreign_keys(table_class)


class _FakeField:
    incl_region = True
    incl_cond1 = False
    incl_cond2 = False
    incl_cond3 = False
    new_primary_keys = ['field', 'region']

    def __call__(self):
        return self

    @staticmethod
    def load_exp_file_info_df(key, filter_kind='response'):
        df = pd.DataFrame([
            dict(field='GCL0', region='LR', stimulus='chirp', kind='response', filepath='a/SMP_M1_LR_GCL0_chirp.h5'),
            dict(field='GCL0', region='LR', stimulus='gChirp', kind='response', filepath='a/SMP_M1_LR_GCL0_gChirp.h5'),
            dict(field='GCL0', region='LR', stimulus='wipStim', kind='response', filepath='a/SMP_M1_LR_GCL0_wip.h5'),
            dict(field='GCL0', region='LR', stimulus='mb', kind='response', filepath='a/SMP_M1_LR_GCL0_mb.h5'),
            dict(field='GCL1', region='LR', stimulus='chirp', kind='response', filepath='a/SMP_M1_LR_GCL1_chirp.h5'),
        ])
        return df[df['kind'] == filter_kind]


class _FakeStimulus:
    def __and__(self, key):
        return self

    @staticmethod
    def fetch1(attr):
        assert attr == 'alias'
        return 'chirp_gchirp'


def test_presentation_only_takes_listed_files():
    presentation = type('TestPresentation', (core.PresentationTemplate,), dict(
        field_table=_FakeField(), stimulus_table=_FakeStimulus(), incl_cond1=False))()
    df = presentation.load_field_stim_file_info_df(dict(field='GCL0', region='LR', stim_name='chirp'))
    assert sorted(df['stimulus']) == ['chirp', 'gChirp']
    assert set(df['stim_name']) == {'chirp'}


@pytest.mark.parametrize('kind', ['response'])
def test_check_listed_for_presentation_accepts_listed(kind):
    row = pd.Series(dict(stimulus='gChirp', kind=kind, filepath='SMP_M1_LR_GCL0_gChirp.h5'))
    check_listed_for_presentation(row, 'chirp', 'chirp_gchirp')


@pytest.mark.parametrize('stimulus, kind', [
    ('wipStim', 'response'),  # unlisted stimulus
    ('mb', 'response'),  # listed, but for another stimulus
    ('chirp', 'hr'),  # not a response recording
    (None, 'response'),
])
def test_check_listed_for_presentation_rejects_unlisted(stimulus, kind):
    row = pd.Series(dict(stimulus=stimulus, kind=kind, filepath='some_file.h5'))
    with pytest.raises(ValueError, match='not listed'):
        check_listed_for_presentation(row, 'chirp', 'chirp_gchirp')
