import os
import warnings
from abc import abstractmethod
from datetime import datetime

import datajoint as dj
import pandas as pd
from matplotlib import pyplot as plt

from djimaging.utils.dj_utils import make_hash


def prepare_dj_config_location_from_table(input_folder):
    stores_dict = {
        "location_table_input": {"protocol": "file", "location": input_folder, "stage": input_folder}}

    # Make sure folders exits
    for store, store_dict in stores_dict.items():
        for name in store_dict.keys():
            if name in ["location", "stage"]:
                assert os.path.isdir(store_dict[name]), f'This must be a folder you have access to: {store_dict[name]}'

    os.environ["DJ_SUPPORT_FILEPATH_MANAGEMENT"] = "TRUE"

    dj_config_stores = dj.config['stores'] or dict()
    dj_config_stores.update(stores_dict)
    dj.config['stores'] = dj_config_stores


class RetinalFieldLocationTableParamsTemplate(dj.Lookup):
    database = ""
    store = "location_table_input"

    @property
    def definition(self):
        definition = """
        table_hash : varchar(32)         # hash of the classifier params config
        ---
        table_path :   attach@{store} # Path to table
        col_experimenter = 'experimenter' : varchar(191)
        col_exp_num = 'exp_num' : varchar(191)
        col_date = 'date' : varchar(191)
        col_field = 'field' : varchar(191)
        col_ventral_dorsal_pos = 'ventral_dorsal_pos' : varchar(191)
        col_temporal_nasal_pos = 'temporal_nasal_pos' : varchar(191)
        """.format(store=self.store)
        return definition

    def add_table(self, table_path, skip_duplicates=False, **col_kw):
        """Add default preprocess parameter to table"""
        key = dict(table_path=table_path)
        key.update(**col_kw)
        key["table_hash"] = make_hash(key)
        self.insert1(key, skip_duplicates=skip_duplicates)


class RetinalFieldLocationFromTableTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
            # location of the recorded fields relative to the optic disk
            # XCoord_um is the relative position from back towards curtain, i.e. larger XCoord_um means closer curtain
            # YCoord_um is the relative position from left to right, i.e. larger YCoord_um means more right

            -> self.field_table
            -> self.params_table
            ---
            ventral_dorsal_pos   :float      # position on the ventral-dorsal axis, greater 0 means dorsal
            temporal_nasal_pos   :float      # position on the temporal-nasal axis, greater 0 means nasal
            """
        return definition

    @property
    def key_source(self):
        try:
            return self.field_table.proj() * self.params_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def params_table(self):
        pass

    @property
    @abstractmethod
    def expinfo_table(self):
        pass

    def make(self, key):
        params = (self.params_table() & key).fetch1()

        df = pd.read_csv(params['table_path'])
        df.date = df.date.apply(lambda x: datetime.date(datetime.strptime(str(x), '%Y%m%d')))
        df.head()

        exp_idx = \
            (df[params['col_experimenter']] == key['experimenter']) & \
            (df[params['col_exp_num']] == key['exp_num']) & \
            (df[params['col_date']] == key['date'])

        if params['col_field'] != '':
            exp_idx &= (df[params['col_field']] == key['field'])

        rows = df[exp_idx]
        assert len(rows) <= 1, f'Multiple rows match key={key}: rows'

        if len(rows) == 0:
            warnings.warn(f'Did not find entry for k={key} in {params["table_path"]}')
            return

        row = rows.iloc[0]

        rfl_key = key.copy()
        rfl_key['ventral_dorsal_pos'] = row[params['col_ventral_dorsal_pos']]
        rfl_key['temporal_nasal_pos'] = row[params['col_temporal_nasal_pos']]
        self.insert1(rfl_key)

    def plot(self, key=None):
        temporal_nasal_pos, ventral_dorsal_pos = self.fetch("temporal_nasal_pos", "ventral_dorsal_pos")
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.scatter(temporal_nasal_pos, ventral_dorsal_pos, label='all')
        if key is not None:
            ktemporal_nasal_pos, kventral_dorsal_pos = (self & key).fetch1(
                "temporal_nasal_pos", "ventral_dorsal_pos")
            ax.scatter(ktemporal_nasal_pos, kventral_dorsal_pos, label='key')
            ax.legend()
        ax.set(xlabel="temporal_nasal_pos", ylabel="ventral_dorsal_pos")
        ax.set_aspect(aspect="equal", adjustable="datalim")
        plt.show()
