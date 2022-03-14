import datajoint as dj


class PlaceholderTable:
    def __and__(self, other):
        return True

    def __mul__(self, other):
        return other

    @classmethod
    def ExpInfo(cls):
        pass

    @classmethod
    def RoiMask(cls):
        pass

    @classmethod
    def FieldInfo(cls):
        pass


def get_class_attributes(class_):
    class_attrs = [attr for attr in class_.__dict__.keys() if attr[:2] != '__']
    return class_attrs


def get_input_tables(definition):
    all_lines = definition.replace(' ', '').split('\n')
    table_lines = [line for line in all_lines if line.startswith('->')]
    tables = [line.replace('->', '').replace('self.', '').replace('self().', '') for line in table_lines]
    return tables


def activate_schema(schema, schema_name=None, create_schema=True, create_tables=True):
    # Based on https://github.com/datajoint/element-lab
    """
    activate(schema, schema_name=None, create_schema=True, create_tables=True)
        :param schema: schema objec
        :param schema_name: schema name on the database server to activate the
                            `lab` element
        :param create_schema: when True (default), create schema in the
                              database if it does not yet exist
        :param create_tables: when True (default), create tables in the
                              database if they do not yet exist
    """
    if schema_name is None:
        schema_name = dj.config.get('schema_name', '')
        assert len(schema_name) > 0, 'Set schema name as parameter or in config file'
    else:
        config_schema_name = dj.config.get('schema_name', '')
        assert len(config_schema_name) == 0 or schema_name == config_schema_name,\
            'Trying to set two different schema names'

    schema.activate(schema_name, create_schema=create_schema, create_tables=create_tables)
