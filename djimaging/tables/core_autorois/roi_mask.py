import os
import pickle
import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from djimaging.tables.misc.highresolution import load_high_res_stack

from djimaging.autorois.roi_canvas import InteractiveRoiCanvas
from djimaging.autorois.corr_roi_mask_utils import CorrRoiMask

from djimaging.utils import scanm_utils
from djimaging.utils.datafile_utils import as_pre_filepath
from djimaging.utils.dj_utils import get_primary_key, check_unique_one
from djimaging.utils.mask_utils import to_igor_format, to_python_format, to_roi_mask_file, sort_roi_mask_files, \
    load_preferred_roi_mask_igor, load_preferred_roi_mask_pickle, compare_roi_masks
from djimaging.utils.plot_utils import plot_field


class RoiMaskTemplate(dj.Manual):
    database = ""

    @property
    def definition(self):
        definition = """
        # ROI mask
        -> self.field_table
        -> self.raw_params_table
        ---
        -> self.presentation_table
        roi_mask     : blob                   # ROI mask for recording field
        """
        return definition

    class RoiMaskPresentation(dj.Part):
        @property
        def definition(self):
            definition = """
            # ROI Mask
            -> master
            -> self.presentation_table
            ---
            roi_mask      : blob       # ROI mask for presentation field
            as_field_mask : enum("same", "different", "shifted")  # relationship to field mask
            shift_dx=0    : int  # Shift in x
            shift_dy=0    : int  # Shift in y
            """
            return definition

        @property
        @abstractmethod
        def presentation_table(self):
            pass

    @property
    @abstractmethod
    def presentation_table(self):
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

    @property
    @abstractmethod
    def raw_params_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.field_table.proj() * self.raw_params_table().proj()
        except (AttributeError, TypeError):
            pass

    def list_missing_field(self):
        missing_keys = (self.field_table.proj() & (self.presentation_table.proj() - self.proj())).fetch(as_dict=True)
        return missing_keys

    def draw_roi_mask(self, field_key=None, pres_key=None, canvas_width=20, autorois_models='default_rgc',
                      show_diagnostics=True, load_high_res=True, **kwargs):
        if canvas_width <= 0 or canvas_width >= 100:
            raise ValueError(f'canvas_width={canvas_width} must be in (0, 100)%')

        if pres_key is not None:
            field_key = (self.field_table & pres_key).proj().fetch1()
        elif field_key is None:
            field_key = np.random.choice(self.list_missing_field())

        assert field_key is not None, 'No field_key provided and no missing field found.'

        pres_keys = np.array(list((self.presentation_table & field_key).proj()),
                             dtype='object')

        from_raw_data = (self.raw_params_table & field_key).fetch1("from_raw_data")

        n_artifact, pixel_size_um, scan_type = (self.presentation_table() & pres_keys).fetch(
            'npixartifact', 'pixel_size_um', 'scan_type')
        n_artifact = check_unique_one(n_artifact, name='n_artifact')
        pixel_size_um = check_unique_one(pixel_size_um, name='pixel_size_um')
        scan_type = check_unique_one(scan_type, name='scan_type')

        if scan_type == 'xy':
            pixel_size_d1_d2 = (pixel_size_um, pixel_size_um)
        elif scan_type == 'xz':
            z_step_um = (self.presentation_table() & pres_keys).fetch1('z_step_um')
            z_step_um = check_unique_one(z_step_um, name='z_step_um')
            pixel_size_d1_d2 = (pixel_size_um, z_step_um)
        else:
            raise NotImplementedError(scan_type)

        # Sort by relevance
        mask_alias, highres_alias = (self.userinfo_table() & field_key).fetch1("mask_alias", "highres_alias")
        pres_info = np.array([(self.presentation_table & pres_key).fetch1('pres_data_file', 'stim_name', 'condition')
                              for pres_key in pres_keys])
        files, stim_names, conditions = pres_info.T

        # Sort data by relevance
        sort_idxs = sort_roi_mask_files(files, mask_alias=mask_alias, highres_alias=highres_alias, as_index=True)
        pres_keys = pres_keys[sort_idxs]
        files = files[sort_idxs]
        stim_names = stim_names[sort_idxs]
        conditions = conditions[sort_idxs]

        # Load stack data
        ch0_stacks, ch1_stacks, output_files = [], [], []
        for input_file in files:
            data_name, alt_name = (self.userinfo_table & field_key).fetch1('data_stack_name', 'alt_stack_name')

            ch_stacks, wparams = scanm_utils.load_stacks(
                input_file, from_raw_data=from_raw_data, ch_names=(data_name, alt_name))
            ch0_stacks.append(ch_stacks[data_name])
            ch1_stacks.append(ch_stacks[alt_name])
            output_files.append(to_roi_mask_file(input_file))

        # Load high resolution data is possible
        high_res_bg_dict = self.load_high_res_bg_dict(field_key) if load_high_res else dict()

        # Load initial ROI masks
        igor_roi_masks = (self.raw_params_table & field_key).fetch1('igor_roi_masks')
        initial_roi_mask, _ = self.load_initial_roi_mask(field_key=field_key, igor_roi_masks=igor_roi_masks)

        if initial_roi_mask is not None:
            initial_roi_mask = to_python_format(initial_roi_mask)

        # Load default AutoROIs models
        if isinstance(autorois_models, str):
            autorois_models = load_default_autorois_models(autorois_models)

        if len(self & field_key) == 0:
            shifts = None
        else:
            shifts = []
            for pres_key in pres_keys:
                if len(self.RoiMaskPresentation & pres_key) > 0:
                    as_field_mask, shift_dx, shift_dy = \
                        (self.RoiMaskPresentation & pres_key).fetch1("as_field_mask", "shift_dx", "shift_dy")
                    if as_field_mask in ["shifted", "same"]:
                        shifts.append((shift_dx, shift_dy))
                    elif as_field_mask in ["different"]:
                        warnings.warn(f"""
                        Inconsistent ROI mask for field_key=\n{field_key}\nand pres_key=\n{pres_key}\n.
                        This is not supported; GUI will be initialized with Field ROI mask instead.
                        If you really want to define different (i.e. not simply shifted) ROI masks for a field
                        open the GUI for a single Presentation key and save the ROI mask.
                        """)
                        shifts.append((0, 0))

        roi_canvas = InteractiveRoiCanvas(
            stim_names=[f"{stim_name}({condition})" for stim_name, condition in zip(stim_names, conditions)],
            ch0_stacks=ch0_stacks, ch1_stacks=ch1_stacks, n_artifact=n_artifact, bg_dict=high_res_bg_dict,
            main_stim_idx=0, initial_roi_mask=initial_roi_mask, shifts=shifts,
            canvas_width=canvas_width, autorois_models=autorois_models, output_files=output_files,
            pixel_size_um=pixel_size_d1_d2,
            show_diagnostics=show_diagnostics,
            **kwargs,
        )
        print(f"""
        This function returns an InteractiveRoiCanvas object, roi_canvas. 
        To start the GUI, call roi_canvas.start_gui() in a Jupyter Lab cell.
        """)
        return roi_canvas

    def load_initial_roi_mask(self, field_key, igor_roi_masks=str):
        """
        Load initial ROI mask for field.
        First try to load from database.
        Second try pickle file, unless ROI masks should be exclusively loaded from Igor.
        Last try to load from Igor file, unless ROI masks should never be loaded from Igor.
        """

        roi_mask = self.load_field_roi_mask_database(field_key=field_key)

        if roi_mask is not None:
            return roi_mask, 'database'

        if igor_roi_masks != 'yes':
            roi_mask, src_file = self.load_field_roi_mask_pickle(field_key=field_key)
            if roi_mask is not None:
                return roi_mask, src_file

        if igor_roi_masks != 'no':
            roi_mask, src_file = self.load_field_roi_mask_igor(field_key=field_key)
            if roi_mask is not None:
                return roi_mask, src_file

        return None, 'none'

    def load_field_roi_mask_database(self, field_key):
        """Load ROI mask that was generated in DataJoint GUI"""
        database_roi_masks = (self & field_key).fetch("roi_mask")

        if len(database_roi_masks) == 1:
            database_roi_mask = database_roi_masks[0].copy()
        elif len(database_roi_masks) == 0:
            database_roi_mask = None
        else:
            raise ValueError(f'Found multiple ROI masks for key=\n{field_key}')

        return database_roi_mask

    def load_field_roi_mask_pickle(self, field_key) -> (np.ndarray, str):
        mask_alias, highres_alias = (self.userinfo_table() & field_key).fetch1("mask_alias", "highres_alias")
        files = (self.presentation_table() & field_key).fetch("pres_data_file")

        roi_mask, src_file = load_preferred_roi_mask_pickle(
            files, mask_alias=mask_alias, highres_alias=highres_alias)
        if roi_mask is not None:
            print(f'Loaded ROI mask from file={src_file} for files=\n{files}\nfor mask_alias={mask_alias}')

        return roi_mask, src_file

    def load_field_roi_mask_igor(self, field_key) -> (np.ndarray, str):
        mask_alias, highres_alias, raw_data_dir, pre_data_dir = (self.userinfo_table() & field_key).fetch1(
            "mask_alias", "highres_alias", "raw_data_dir", "pre_data_dir")
        files = (self.presentation_table() & field_key).fetch("pres_data_file")

        files = [as_pre_filepath(f, raw_data_dir=raw_data_dir, pre_data_dir=pre_data_dir) for f in files]

        roi_mask, src_file = load_preferred_roi_mask_igor(files, mask_alias=mask_alias, highres_alias=highres_alias)
        if roi_mask is not None:
            print(f'Loaded ROI mask from file={src_file} for files=\n{files}\nfor mask_alias={mask_alias}')

        return roi_mask, src_file

    def rescan_filesystem(self, restrictions: dict = None, verboselvl: int = 0, suppress_errors: bool = False,
                          only_new_fields: bool = True, roi_mask_dir=None, prefix_if_raw='SMP_', drop_smp_prefix=False):
        """Scan filesystem for new ROI masks and add them to the database."""
        if restrictions is None:
            restrictions = dict()

        if only_new_fields:
            restrictions = (self.key_source - self) & restrictions

        err_list = []

        for key in (self.key_source & restrictions):
            try:
                self.add_field_roi_masks(
                    key, roi_mask_dir=roi_mask_dir, prefix_if_raw=prefix_if_raw, drop_smp_prefix=drop_smp_prefix,
                    verboselvl=verboselvl)
            except Exception as e:
                if suppress_errors:
                    warnings.warn(f'Error for key={key}:\n{e}')
                    err_list.append((key, e))
                else:
                    raise e

        return err_list

    def add_field_roi_masks(self, field_key, roi_mask_dir, prefix_if_raw, drop_smp_prefix=False, verboselvl=0):
        if verboselvl > 2:
            print('\nfield_key:', field_key)

        pres_keys = (self.presentation_table.proj() & field_key)
        roi_masks = [self.load_presentation_roi_mask(
            key, roi_mask_dir, prefix_if_raw=prefix_if_raw, drop_smp_prefix=drop_smp_prefix)
            for key in pres_keys]

        data_pairs = zip(pres_keys, roi_masks)

        # Filter out keys without ROI mask
        data_pairs = [(pres_key, roi_mask) for pres_key, roi_mask in data_pairs
                      if roi_mask is not None]

        if len(data_pairs) == 0:
            if verboselvl > 1:
                print('No ROI masks found for field:', field_key)
            if verboselvl > 2:
                print('pres_keys:', [k for k in pres_keys])
            return

        # Filter out keys that are already present
        data_pairs = [(pres_key, roi_mask) for pres_key, roi_mask in data_pairs
                      if pres_key not in self.RoiMaskPresentation().proj()]

        if len(data_pairs) == 0:
            if verboselvl > 1:
                print('Nothing new to add for field:', field_key)
            return

        if verboselvl > 0:
            print(f'Adding {len(data_pairs)} ROI masks for field:', field_key)

        # Find preferred file that should be used as main key.
        mask_alias, highres_alias = (self.userinfo_table & field_key).fetch1("mask_alias", "highres_alias")
        files = [(self.presentation_table & pres_key).fetch1('pres_data_file') for pres_key, roi_mask in data_pairs]
        sort_idxs = sort_roi_mask_files(files, mask_alias=mask_alias, highres_alias=highres_alias, as_index=True)

        main_pres_key, main_roi_mask = data_pairs[sort_idxs[0]]

        self.insert1({**field_key, **main_pres_key, "roi_mask": main_roi_mask}, skip_duplicates=True)
        for pres_key, roi_mask in data_pairs:
            as_field_mask, (shift_dx, shift_dy) = compare_roi_masks(roi_mask, main_roi_mask)
            self.RoiMaskPresentation().insert1(
                {**pres_key, "roi_mask": roi_mask, "as_field_mask": as_field_mask,
                 "shift_dx": shift_dx, "shift_dy": shift_dy},
                skip_duplicates=True)

    def load_presentation_roi_mask(self, key, roi_mask_dir=None, prefix_if_raw='SMP_', drop_smp_prefix=False):
        igor_roi_masks, from_raw_data = (self.raw_params_table & key).fetch1('igor_roi_masks', 'from_raw_data')
        input_file = (self.presentation_table & key).fetch1("pres_data_file")

        if igor_roi_masks == 'yes':
            assert not from_raw_data, 'Inconsistent parameters'
            filesystem_roi_mask = scanm_utils.load_roi_mask_from_h5(filepath=input_file, ignore_not_found=True)
        else:
            if not from_raw_data:
                if drop_smp_prefix:
                    f_dir, f_name = os.path.split(input_file)
                    input_file = os.path.join(f_dir, f_name.replace('SMP_', ''))
                roimask_file = to_roi_mask_file(input_file)
            else:
                raw_data_dir, pre_data_dir = (self.userinfo_table() & key).fetch1("raw_data_dir", "pre_data_dir")

                src_dir = pre_data_dir if roi_mask_dir is None else roi_mask_dir
                input_file = input_file.replace(f'/{raw_data_dir}/', f'/{src_dir}/{prefix_if_raw}')
                roimask_file = to_roi_mask_file(input_file)

            if os.path.isfile(roimask_file):
                with open(roimask_file, 'rb') as f:
                    filesystem_roi_mask = pickle.load(f).copy().astype(np.int32)
                filesystem_roi_mask = to_igor_format(filesystem_roi_mask)
            else:
                filesystem_roi_mask = None

        if len((self.RoiMaskPresentation & key).proj()) > 0:
            database_roi_mask = (self.RoiMaskPresentation & key).fetch1('roi_mask')
            if not np.all(filesystem_roi_mask == database_roi_mask):
                raise ValueError(f'ROI mask for key=\n{key}\nhas been changed on filesystem but not in database.')
            else:
                filesystem_roi_mask = database_roi_mask.copy()

        return filesystem_roi_mask

    def plot1(self, key=None):
        key = get_primary_key(table=self.proj() * self.presentation_table.proj(), key=key)
        npixartifact = (self.field_table & key).fetch1('npixartifact')
        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.presentation_table.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        alt_ch_average = (self.presentation_table.StackAverages & key & f'ch_name="{alt_name}"').fetch1('ch_average')
        roi_mask = (self.RoiMaskPresentation & key).fetch1('roi_mask')
        plot_field(main_ch_average, alt_ch_average, roi_mask=roi_mask, title=key, npixartifact=npixartifact)

    def load_high_res_bg_dict(self, key):
        field = (self.field_table() & key).fetch1("field")
        field_loc = (self.userinfo_table() & key).fetch1("field_loc")
        highres_alias = (self.userinfo_table() & key).fetch1("highres_alias")
        header_path = (self.experiment_table() & key).fetch1('header_path')
        pre_data_path = os.path.join(header_path, (self.userinfo_table() & key).fetch1("pre_data_dir"))
        raw_data_path = os.path.join(header_path, (self.userinfo_table() & key).fetch1("raw_data_dir"))

        try:
            filepath, ch_stacks, wparams = load_high_res_stack(
                pre_data_path=pre_data_path, raw_data_path=raw_data_path,
                highres_alias=highres_alias, field=field, field_loc=field_loc)
        except Exception as e:
            warnings.warn(f'Failed to load high resolution data because of error:\n{e}')
            return dict()

        if ch_stacks is None:
            return dict()

        bg_dict = dict()
        for name, stack in ch_stacks.items():
            bg_dict[f'HR[{name}]'] = np.nanmedian(stack, 2)

        return bg_dict


def load_default_autorois_models(kind='default_rgc'):
    autorois_models = dict()
    _add_autorois_unet(autorois_models, kind=kind)
    _add_autorois_corr(autorois_models, kind=kind)
    return autorois_models if len(autorois_models) > 0 else None


def _add_autorois_unet(autorois_models: dict, kind: str):
    try:
        assert kind in ['default_rgc'], f"Only implemented for RGCs, but kind={kind}"
        from djimaging.autorois.unet import UNet

        config_path = "/gpfs01/euler/data/Resources/AutoROIs/models/UNET_v0.1.0/sd_images.yaml"
        checkpoint_path = "/gpfs01/euler/data/Resources/AutoROIs/models/UNET_v0.1.0/dropout_and_aug_regul.ckpt"
        unet_model = UNet.from_checkpoint(config_path, checkpoint_path)

        autorois_models['UNet'] = unet_model

    except Exception as e:
        warnings.warn(f'Failed to load default AutoROIs models because of error:\n{e}')


def _add_autorois_corr(autorois_models: dict, kind: str):
    if kind == 'default_rgc':
        kws = dict(cut_x=(0, 0), cut_z=(0, 0), min_area_um2=10, max_area_um2=400, n_pix_max=None, line_threshold_q=70,
                   use_ch0_stack=True, grow_use_corr_map=False, grow_threshold=0.1,
                   grow_only_local_thresh_pixels=True)
    elif kind == 'default_bc' or kind == 'default_ac':
        kws = dict(cut_x=(4, 2), cut_z=(8, 2), min_area_um2=0.5, max_area_um2=12.6, n_pix_max=None, line_threshold_q=70,
                   use_ch0_stack=True, grow_use_corr_map=False, grow_threshold=None, line_threshold_min=0.1,
                   grow_only_local_thresh_pixels=True)
    else:
        raise NotImplementedError(kind)
    corr_model = CorrRoiMask(**kws)
    autorois_models['CorrRoiMask'] = corr_model
