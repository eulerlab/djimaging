from djimaging.utils.dj_utils import get_class_attributes, get_input_tables


def _test_definition(djclass):
    tables = get_input_tables(djclass().definition)
    class_attrs = get_class_attributes(djclass)

    for table in tables:
        assert table == 'master' or table in class_attrs, table

    for class_attr in class_attrs:
        if class_attr.endswith('_table'):
            assert class_attr.lower() == class_attr, 'Use lower case'
