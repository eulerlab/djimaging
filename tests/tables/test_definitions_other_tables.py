from djimaging.tables import classifier_v2, response, location, receptivefield, motion_correction
from djimaging.utils.dj_utils import get_input_tables, get_class_attributes


def _test_definition(djclass):
    tables = get_input_tables(djclass().definition)
    class_attrs = get_class_attributes(djclass)

    for table in tables:
        table = table.split('.', 1)[0].removesuffix('()')
        assert table == 'master' or table in class_attrs, table

    for class_attr in class_attrs:
        if class_attr.endswith('_table'):
            assert class_attr.lower() == class_attr, 'Use lower case'


# location
def test_definition_RelativeFieldLocationTemplate():
    _test_definition(location.RelativeFieldLocationTemplate)


def test_definition_RetinalFieldLocationTemplate():
    _test_definition(location.RetinalFieldLocationTemplate)


# chirp
def test_definition_ChirpQITemplate():
    _test_definition(response.ChirpQITemplate)


# orientation
def test_definition_OsDsIndexesTemplate():
    _test_definition(response.OsDsIndexesTemplate)


def test_definition_ChirpSurroundTemplateV2():
    _test_definition(response.ChirpSurroundTemplateV2)


# pretrained classifier
def test_definition_ClassifierV2Template():
    _test_definition(classifier_v2.ClassifierV2Template)


def test_definition_Baden16TracesV2Template():
    _test_definition(classifier_v2.Baden16TracesV2Template)


def test_definition_CelltypeAssignmentV2Template():
    _test_definition(classifier_v2.CelltypeAssignmentV2Template)


def test_definition_FastStaParamsTemplate():
    _test_definition(receptivefield.FastStaParamsTemplate)


def test_definition_FastStaTemplate():
    _test_definition(receptivefield.FastStaTemplate)


def test_definition_RfGlmParamsTemplate():
    _test_definition(receptivefield.RfGlmParamsTemplate)


def test_definition_RfGlmTemplate():
    _test_definition(receptivefield.RfGlmTemplate)


def test_definition_RfGlmQualityTemplate():
    _test_definition(receptivefield.RfGlmQualityTemplate)


# motion correction
def test_definition_MotionParamsTemplate():
    _test_definition(motion_correction.MotionParamsTemplate)


def test_definition_MotionReferenceTemplate():
    _test_definition(motion_correction.MotionReferenceTemplate)


def test_definition_MotionDetectionTemplate():
    _test_definition(motion_correction.MotionDetectionTemplate)


def test_definition_MotionRoiMaskTemplate():
    _test_definition(motion_correction.MotionRoiMaskTemplate)
    _test_definition(motion_correction.MotionRoiMaskTemplate.Roi)


def test_definition_MotionTrustTemplate():
    _test_definition(motion_correction.MotionTrustTemplate)
    _test_definition(motion_correction.MotionTrustTemplate.Roi)


def test_motion_params_reject_unknown_and_fill_defaults(monkeypatch):
    inserted = []
    table = motion_correction.MotionParamsTemplate.__new__(motion_correction.MotionParamsTemplate)
    monkeypatch.setattr(type(table), 'insert1', lambda self, row, skip_duplicates=False: inserted.append(row),
                        raising=False)
    table.add(motion_params_id=2, tau_slow_s=5.)
    assert inserted[0]['motion_params']['tau_slow_s'] == 5.
    assert inserted[0]['motion_params']['pooling'] == motion_correction.DEFAULT_MOTION_PARAMS['pooling']
    import pytest
    with pytest.raises(ValueError):
        table.add(motion_params_id=3, not_a_param=1)
    with pytest.raises(ValueError):
        table.add(motion_params_id=4, pooling='field')
