import os
import warnings
from abc import abstractmethod
from copy import deepcopy

import datajoint as dj
import numpy as np

from djimaging.utils.alias_utils import match_file, get_field_files
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.plot_utils import plot_field
from djimaging.utils.scanm.read_smp_utils import load_stacks_and_wparams
from djimaging.utils.scanm.setup_utils import get_pixel_size_xy_um
from djimaging.utils.scanm.read_h5_utils import load_stacks_and_wparams


def load_high_res_stack(pre_data_path, raw_data_path, highres_alias,
                        field, field_loc, condition=None, condition_loc=None, allow_raw=True):
    """Scan filesystem for file and load data. Tries to load h5 files first, but may also load raw files."""

    filepath = scan_for_highres_filepath(
        folder=pre_data_path, highres_alias=highres_alias, field=field, field_loc=field_loc, ftype='h5',
        condition=condition, condition_loc=condition_loc)

    if filepath is not None:
        try:
            ch_stacks, wparams = load_stacks_and_wparams(filepath)
            return filepath, ch_stacks, wparams
        except OSError as e:
            warnings.warn(f'OSError when reading file: {filepath}\n{e}')
            pass

    if allow_raw:
        try:
            from scanmsupport.scanm.scanm_smp import SMP
        except ImportError:
            warnings.warn('Custom package `scanmsupport is not installed. Cannot load SMP files.')
            allow_raw = False

    if allow_raw:
        filepath = scan_for_highres_filepath(
            folder=raw_data_path, highres_alias=highres_alias, field=field, field_loc=field_loc - 1, ftype='smp',
            condition=condition, condition_loc=condition_loc)

        if filepath is not None:
            try:
                ch_stacks, wparams = load_stacks_and_wparams(filepath)
                return filepath, ch_stacks, wparams
            except OSError as e:
                warnings.warn(f'OSError when reading file: {filepath}\n{e}')
                pass

    return None, None, None


def scan_for_highres_filepath(folder, field, field_loc, highres_alias, condition=None, condition_loc=None, ftype='h5'):
    """Scan filesystem for files that match the highres alias and are from the same field."""
    if not os.path.isdir(folder):
        return None

    field_files = get_field_files(folder=folder, field=field, field_loc=field_loc, incl_hidden=False, ftype=ftype)

    for file in field_files:
        is_highres = match_file(file, pattern=highres_alias, pattern_loc=None, ftype=ftype)
        is_condition = True if condition is None else \
            match_file(file, pattern=condition, pattern_loc=condition_loc, ftype=ftype)

        if is_highres & is_condition:
            return os.path.join(folder, file)
    return None


class HighResTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # High resolution stack information.
        -> self.field_table
        ---
        fromfile : varchar(255)  # Absolute path to file
        nframes : int  # Number of frames averaged
        absx: float  # absolute position of the center (of the cropped field) in the x axis as recorded by ScanM
        absy: float  # absolute position of the center (of the cropped field) in the y axis as recorded by ScanM
        absz: float  # absolute position of the center (of the cropped field) in the z axis as recorded by ScanM
        nxpix: int  # number of pixels in x
        nypix: int  # number of pixels in y
        nzpix: int  # number of pixels in z
        nxpix_offset: int  # number of offset pixels in x
        nxpix_retrace: int  # number of retrace pixels in x
        zoom: float  # zoom factor used during recording
        pixel_size_um :float  # width / height of a pixel in um        
        """
        return definition

    @property
    def key_source(self):
        try:
            return self.field_table.proj()
        except (AttributeError, TypeError):
            pass

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    class StackAverages(dj.Part):
        @property
        def definition(self):
            definition = """
            # Stack median (over time of the available channels)
            -> master
            ch_name : varchar(255)  # name of the channel
            ---
            ch_average :longblob  # Stack median over time
            """
            return definition

    def make(self, key):
        field = (self.field_table() & key).fetch1("field")
        stim_loc, field_loc, condition_loc = (self.userinfo_table() & key).fetch1(
            "stimulus_loc", "field_loc", "condition_loc")
        highres_alias = (self.userinfo_table() & key).fetch1("highres_alias")
        header_path = (self.experiment_table() & key).fetch1('header_path')
        pre_data_path = os.path.join(header_path, (self.userinfo_table() & key).fetch1("pre_data_dir"))
        raw_data_path = os.path.join(header_path, (self.userinfo_table() & key).fetch1("raw_data_dir"))
        assert os.path.exists(pre_data_path), f"Error: Data folder does not exist: {pre_data_path}"
        setupid = (self.experiment_table().ExpInfo() & key).fetch1("setupid")
        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')

        filepath, ch_stacks, wparams = load_high_res_stack(
            pre_data_path=pre_data_path, raw_data_path=raw_data_path,
            highres_alias=highres_alias, field=field, field_loc=field_loc)

        if (filepath is None) or (data_name not in ch_stacks) or (alt_name not in ch_stacks):
            return

        # Get pixel sizes
        nxpix = wparams["user_dxpix"] - wparams["user_npixretrace"] - wparams["user_nxpixlineoffs"]
        nypix = wparams["user_dypix"]
        nzpix = wparams["user_dzpix"]
        pixel_size_um = get_pixel_size_xy_um(zoom=wparams["zoom"], setupid=setupid, npix=nxpix)

        # Get key
        highres_key = deepcopy(key)
        highres_key["fromfile"] = filepath

        highres_key["absx"] = wparams['xcoord_um']
        highres_key["absy"] = wparams['ycoord_um']
        highres_key["absz"] = wparams['zcoord_um']

        highres_key["nxpix"] = nxpix
        highres_key["nypix"] = nypix
        highres_key["nzpix"] = nzpix
        highres_key["nxpix_offset"] = wparams["user_nxpixlineoffs"]
        highres_key["nxpix_retrace"] = wparams["user_npixretrace"]

        highres_key["zoom"] = wparams["zoom"]
        highres_key["pixel_size_um"] = pixel_size_um

        highres_key["nframes"] = ch_stacks['wDataCh0'].shape[2]

        # get stack avgs
        avg_keys = []
        for name, stack in ch_stacks.items():
            avg_key = deepcopy(key)
            avg_key["ch_name"] = name
            avg_key["ch_average"] = np.median(stack, 2)
            avg_keys.append(avg_key)

        self.insert1(highres_key)
        for avg_key in avg_keys:
            (self.StackAverages & key).insert1(avg_key, allow_direct_insert=True)

    def plot1(self, key=None, figsize=(8, 4)):
        key = get_primary_key(table=self, key=key)

        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        alt_ch_average = (self.StackAverages & key & f'ch_name="{alt_name}"').fetch1('ch_average')
        plot_field(main_ch_average, alt_ch_average, roi_mask=None, title=key, figsize=figsize)
