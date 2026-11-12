"""draw_roi_mask in motion mode opens the canvas on the presentation the initial mask belongs to."""

from djimaging.tables.core.roi_mask import RoiMaskTemplate


class _Rows:
    def __init__(self, row):
        self._row = row

    def fetch1(self, *attrs):
        return tuple(self._row[a] for a in attrs) if attrs else self._row


class _Table:
    def __init__(self, row, primary_key=()):
        self._row, self.primary_key = row, list(primary_key)

    def __and__(self, key):
        return _Rows(self._row)


class _FakeRoiMask:
    _main_presentation_idx = RoiMaskTemplate._main_presentation_idx

    def __init__(self, tmp_path, db_pres=None):
        self.pres_keys = [dict(field=1, stim=s) for s in ('chirp', 'mb', 'ds')]
        raw, pre = tmp_path / 'Raw', tmp_path / 'Pre'
        self.filepaths = [str(raw / f'M1_GCL0_{s}.smp') for s in ('chirp', 'mb', 'ds')]
        self.pre_files = [str(pre / f'SMP_M1_GCL0_{s}.h5') for s in ('chirp', 'mb', 'ds')]
        self.output_files = [str(raw / 'ROIs' / f'M1_GCL0_{s}_ROIs.npy') for s in ('chirp', 'mb', 'ds')]
        self.presentation_table = _Table(None, primary_key=('field', 'stim'))
        self.userinfo_table = _Table(dict(raw_data_dir='Raw', pre_data_dir='Pre', mask_alias='mb_chirp',
                                          highres_alias='hr'))
        self._db_row = None if db_pres is None else dict(field=1, stim=db_pres)

    def __and__(self, key):
        return _Rows(self._db_row)

    def idx(self, src):
        field_files = [f.replace('_ROIs.npy', '_FieldROIs.npy') for f in self.output_files]
        return self._main_presentation_idx(dict(field=1), self.pres_keys, self.filepaths, src,
                                           mask_files=[field_files, self.output_files])


def test_database_mask_uses_its_presentation(tmp_path):
    assert _FakeRoiMask(tmp_path, db_pres='ds').idx('database') == 2


def test_mask_file_uses_its_presentation(tmp_path):
    tab = _FakeRoiMask(tmp_path)
    assert tab.idx(tab.output_files[2]) == 2


def test_field_mask_file_uses_its_presentation(tmp_path):
    tab = _FakeRoiMask(tmp_path)
    assert tab.idx(tab.output_files[0].replace('_ROIs.npy', '_FieldROIs.npy')) == 0


def test_igor_mask_uses_presentation_of_h5_file(tmp_path):
    tab = _FakeRoiMask(tmp_path)
    assert tab.idx(tab.pre_files[2]) == 2


def test_without_mask_falls_back_to_mask_alias(tmp_path):
    tab = _FakeRoiMask(tmp_path)
    assert tab.idx('none') == 1  # 'mb' comes first in mask_alias
    assert tab.idx(str(tmp_path / 'unknown.npy')) == 1
