from abc import abstractmethod

import datajoint as dj


def compute_something(data):
    # Implement function
    a, b, c = None, None, data
    return a, b, c


class ExampleTableTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Example table that inherits primary keys from field_table
        -> self.field_table
        ---
        table_column_a : float  # Comment on number
        table_column_b : longblob  # Comment on array
        table_column_c : tinyint unsigned  # use this data type for True (1) or False (0)
        """
        return definition

    @property
    @abstractmethod
    def field_table(self):
        pass  # A place-holder table, must be replaced in schema with real table

    def make(self, key):
        # Fetch something. Use fetch1
        data = (self.field_table() & key).fetch1('table_column_name')

        # Compute something. Ideally do the computation outside make
        a, b, c = compute_something(data)

        # Create new key. Use source key for primary keys.
        new_key = key.copy()
        # Add data to key.
        new_key['table_column_a'] = a
        new_key['table_column_b'] = b
        new_key['table_column_c'] = c

        # Insert data to table
        self.insert1(new_key)
