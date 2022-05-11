import os
from copy import deepcopy

import datajoint as dj
import h5py
import numpy as np

from djimaging.utils.dj_utils import PlaceholderTable
from djimaging.utils.data_utils import list_h5_files, extract_h5_table


def scan_for_highres_h5_filepath(pre_data_path, field, field_loc, highres_alias):
    h5_files = list_h5_files(folder=pre_data_path, hidden=False, field=field, field_loc=field_loc)
    for filename in h5_files:
        for alias in highres_alias.split('_'):
            for fileinfo in filename.replace('.h5', '').split('_')[field_loc:]:
                if alias.lower() == fileinfo.lower():
                    filepath = os.path.join(pre_data_path, filename)
                    return filepath
    return None


def load_high_res_stack(filepath, data_stack_name):
    with h5py.File(filepath, 'r', driver="stdio") as h5_file:
        stack = np.copy(h5_file[data_stack_name])

        wparams = dict()
        if 'wParamsStr' in h5_file.keys():
            wparams.update(extract_h5_table('wParamsStr', open_file=h5_file, lower_keys=True))
            wparams.update(extract_h5_table('wParamsNum', open_file=h5_file, lower_keys=True))

        # Check stack average
        try:
            nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
            nypix = wparams["user_dypix"]

            assert stack.ndim == 3, 'Stack does not match expected shape'
            assert stack.shape[:2] == (nxpix, nypix), f'Stack shape error: {stack.shape} vs {(nxpix, nypix)}'
        except KeyError:
            pass

    return stack


class HighResTemplate(dj.Computed):

    @property
    def definition(self):
        definition = """
        # High resolution stack information.
        -> self.field_table
        ---
        fromfile : varchar(255)  # Absolute path to file
        stack_average : longblob  # Stack average of the high resolution image
        nframes : int  # Number of frames averaged
        """
        return definition

    field_table = PlaceholderTable
    experiment_table = PlaceholderTable
    userinfo_table = PlaceholderTable

    def make(self, key):
        field = (self.field_table() & key).fetch1("field")
        stim_loc, field_loc, condition_loc = (self.userinfo_table() & key).fetch1(
            "stimulus_loc", "field_loc", "condition_loc")
        data_stack_name, highres_alias = (self.userinfo_table() & key).fetch1("data_stack_name", "highres_alias")

        pre_data_path = os.path.join(
            (self.experiment_table() & key).fetch1('header_path'),
            (self.userinfo_table() & key).fetch1("pre_data_dir"))
        assert os.path.exists(pre_data_path), f"Error: Data folder does not exist: {pre_data_path}"

        filepath = scan_for_highres_h5_filepath(
            pre_data_path=pre_data_path, field=field, field_loc=field_loc, highres_alias=highres_alias)

        if filepath is None:
            return

        stack = load_high_res_stack(filepath, data_stack_name=data_stack_name)

        highres_key = deepcopy(key)
        highres_key["fromfile"] = filepath
        highres_key["stack_average"] = np.mean(stack, 2)
        highres_key["nframes"] = stack.shape[2]

        self.insert1(highres_key)
