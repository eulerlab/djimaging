from djimaging.tables import optional, response, location


def _test_definition(djclass):
    tables = get_input_tables(djclass().definition)
    class_attrs = get_class_attributes(djclass)

    for table in tables:
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


# rgc_classifier
def test_definition_CellFilterParametersTemplate():
    _test_definition(optional.CellFilterParamsTemplate)


def test_definition_ClassifierTemplate():
    _test_definition(optional.ClassifierTemplate)


def test_definition_ClassifierTrainingDataTemplate():
    _test_definition(optional.ClassifierTrainingDataTemplate)


def test_definition_CelltypeAssignmentTemplate():
    _test_definition(optional.CelltypeAssignmentTemplate)


def test_definition_ClassifierMethodTemplate():
    _test_definition(optional.ClassifierMethodTemplate)


def test_definition_HighResTemplate():
    _test_definition(optional.HighResTemplate)
