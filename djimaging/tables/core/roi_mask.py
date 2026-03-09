from __future__ import annotations

import os
import pickle
import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils.scanm import read_h5_utils, read_utils

from djimaging.autorois.roi_canvas import InteractiveRoiCanvas

from djimaging.utils.filesystem_utils import as_pre_filepath
from djimaging.utils.dj_utils import get_primary_key, check_unique_one
from djimaging.utils.mask_utils import to_roi_mask_file, sort_roi_mask_files, \
    load_preferred_roi_mask_igor, load_preferred_roi_mask_pickle, compare_roi_masks
from djimaging.utils.mask_format_utils import to_igor_format, to_python_format
from djimaging.utils.plot_utils import plot_field


class RoiMaskTemplate(dj.Manual):
    database = ""
    _max_shift = 5

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
        def presentation_table(self) -> dj.Table:
            """Return the presentation table."""
            pass

    @property
    @abstractmethod
    def presentation_table(self) -> dj.Table:
        """Return the presentation table."""
        pass

    @property
    @abstractmethod
    def field_table(self) -> dj.Table:
        """Return the field table."""
        pass

    @property
    @abstractmethod
    def experiment_table(self) -> dj.Table:
        """Return the experiment table."""
        pass

    @property
    @abstractmethod
    def userinfo_table(self) -> dj.Table:
        """Return the user info table."""
        pass

    @property
    @abstractmethod
    def raw_params_table(self) -> dj.Table:
        """Return the raw data parameters lookup table."""
        pass

    @property
    @abstractmethod
    def highres_table(self) -> dj.Table:
        """Return the high-resolution stack table."""
        pass

    @property
    def key_source(self):
        """Return the key source for this manual table."""
        try:
            return self.field_table.proj() * self.raw_params_table().proj()
        except (AttributeError, TypeError):
            pass

    def list_missing_field(self) -> list:
        """Return a list of field keys that have presentations but no ROI mask entry.

        Returns:
            A list of primary key dicts for fields that are missing from this
            ROI mask table.
        """
        missing_keys = (self.field_table.proj() & (self.presentation_table.proj() - self.proj())).fetch(as_dict=True)
        return missing_keys

    def load_field_file_info_df(self, field_key: dict):
        """Load file info for all response and high-resolution files of a field.

        Args:
            field_key: Primary key dict identifying the field.

        Returns:
            A pandas DataFrame with one row per matching file, filtered to the
            field's own primary-key columns.
        """
        file_info_df = self.field_table().load_exp_file_info_df(field_key, filter_kind=['hr', 'response'])
        for new_key in self.field_table().new_primary_keys:
            file_info_df = file_info_df[file_info_df[new_key] == field_key[new_key]]
        return file_info_df

    def draw_roi_mask(
            self,
            field_key: dict = None,
            pres_key: dict = None,
            canvas_width: int = 20,
            autorois_models: str = 'default_rgc',
            show_diagnostics: bool = True,
            load_high_res: bool = True,
            max_shift: int = None,
            roi_mask_dir: str = 'ROIs',
            old_prefix: str = None,
            new_prefix: str = None,
            use_stim_onset: bool = True,
            verbose: bool = True,
            **kwargs,
    ):
        """Launch the interactive ROI-drawing GUI for a field.

        Args:
            field_key: Primary key dict identifying the field. If None and
                pres_key is also None, a random missing field is selected.
            pres_key: Primary key dict of a presentation whose field will be
                used. Takes precedence over field_key when both are given.
            canvas_width: Width of the canvas as a percentage of the screen
                width (must be between 0 and 100 exclusive).
            autorois_models: Either a dict of auto-ROI models or a string key
                ('default_rgc', etc.) to load the default model set.
            show_diagnostics: Whether to show diagnostic panels in the GUI.
            load_high_res: Whether to load high-resolution background stacks.
            max_shift: Maximum allowed pixel shift between presentations.
                Defaults to the table-level ``_max_shift`` value.
            roi_mask_dir: Sub-directory name where ROI mask pickle files are
                stored (e.g. 'ROIs', 'AutoROIs').
            old_prefix: Path prefix to replace when resolving file paths.
            new_prefix: Replacement path prefix.
            use_stim_onset: If True, trim stacks to start at stimulus onset.
            verbose: If True, print status messages.
            **kwargs: Additional keyword arguments forwarded to
                ``InteractiveRoiCanvas``.

        Returns:
            An ``InteractiveRoiCanvas`` instance. Call ``.start_gui()`` on it
            to open the interactive widget.
        """
        if canvas_width <= 0 or canvas_width >= 100:
            raise ValueError(f'canvas_width={canvas_width} must be in (0, 100)%')

        if pres_key is not None:
            field_key = (self.field_table & pres_key).fetch1('KEY')
        elif field_key is None:
            field_key = np.random.choice(self.list_missing_field())

        if field_key is None:
            raise ValueError('No field_key provided and no missing field found.')
        field_key = (self.field_table & field_key).fetch1('KEY')

        from_raw_data = (self.raw_params_table & field_key).fetch1("from_raw_data")

        field_file_info_df = self.load_field_file_info_df(field_key)
        all_filepaths = field_file_info_df[field_file_info_df['kind'] == 'response']['filepath'].values

        pres_keys = []
        filepaths = []

        for f in all_filepaths:
            pres_key_list = ((self.presentation_table & field_key) & dict(pres_data_file=f)).proj().fetch(as_dict=True)
            if len(pres_key_list) == 1:
                pres_keys.append(pres_key_list[0])
                filepaths.append(f)
            elif len(pres_key_list) > 1:
                raise ValueError(f'Found multiple presentation keys for filepath={f}:\n{pres_key_list}')

        pres_names = [{k: v for k, v in pres_key.items() if k not in field_key.keys()}
                      for pres_key in pres_keys]

        # Load stack data
        data_name, alt_name = (self.userinfo_table & field_key).fetch1('data_stack_name', 'alt_stack_name')
        ch0_stacks, ch1_stacks, output_files = load_stack_data(
            files=filepaths, data_name=data_name, alt_name=alt_name, from_raw_data=from_raw_data,
            roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)

        # Some sanity checks
        assert len(pres_names) == len(pres_keys)
        assert len(filepaths) == len(pres_keys)
        assert len(ch0_stacks) == len(pres_keys)
        assert len(ch1_stacks) == len(pres_keys)
        assert len(output_files) == len(pres_keys)

        # Load pixel size and scan type
        n_artifact, pixel_size_um, scan_type = (self.presentation_table() & pres_keys).fetch(
            'npixartifact', 'pixel_size_um', 'scan_type')
        n_artifact = check_unique_one(n_artifact, name='n_artifact')
        pixel_size_um = check_unique_one(pixel_size_um, name='pixel_size_um')
        scan_type = check_unique_one(scan_type, name='scan_type')

        if scan_type == 'xy':
            pixel_size_d1_d2 = (pixel_size_um, pixel_size_um)
        elif scan_type == 'xz':
            z_step_um = (self.presentation_table() & pres_keys).fetch('z_step_um')
            z_step_um = check_unique_one(z_step_um, name='z_step_um')
            pixel_size_d1_d2 = (pixel_size_um, z_step_um)
        else:
            raise NotImplementedError(scan_type)

        # Reduce stacks to after stimulus onset?
        if use_stim_onset:
            triggertimes = (self.presentation_table & pres_keys).fetch('triggertimes')
            scan_frequencies = (self.presentation_table.ScanInfo() & pres_keys).fetch('scan_frequency')

            stim_onset_idxs = [int(np.floor(tt[0] * fs)) if len(tt) > 0 else 0
                               for tt, fs in zip(triggertimes, scan_frequencies)]
            ch0_stacks = [stack[:, :, stim_onset_idx:] for stack, stim_onset_idx in zip(ch0_stacks, stim_onset_idxs)]
            ch1_stacks = [stack[:, :, stim_onset_idx:] for stack, stim_onset_idx in zip(ch1_stacks, stim_onset_idxs)]

        # Load high resolution data is possible
        high_res_bg_dict = self.load_high_res_bg_dict(
            field_key, ch_names=[data_name, alt_name], verbose=verbose) if load_high_res else dict()

        # Load initial ROI masks
        igor_roi_masks = (self.raw_params_table & field_key).fetch1('igor_roi_masks')
        initial_roi_mask, _ = self.load_initial_roi_mask(
            field_key=field_key, igor_roi_masks=igor_roi_masks,
            roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix, verbose=verbose)

        if initial_roi_mask is not None:
            initial_roi_mask = to_python_format(initial_roi_mask)

        # Load default AutoROIs models
        if isinstance(autorois_models, str):
            autorois_models = load_default_autorois_models(autorois_models)

        # Load shifts if available
        shifts = self.init_shifts(field_key, pres_keys)

        roi_canvas = InteractiveRoiCanvas(
            pres_names=pres_names,
            ch0_stacks=ch0_stacks, ch1_stacks=ch1_stacks, n_artifact=n_artifact, bg_dict=high_res_bg_dict,
            main_stim_idx=0, initial_roi_mask=initial_roi_mask, shifts=shifts,
            canvas_width=canvas_width, autorois_models=autorois_models, output_files=output_files,
            pixel_size_um=pixel_size_d1_d2, show_diagnostics=show_diagnostics,
            max_shift=self._max_shift if max_shift is None else max_shift,
            **kwargs,
        )
        if verbose:
            print(f"Returned InteractiveRoiCanvas object. To start GUI, call <enter_object_name>.start_gui().")
        return roi_canvas

    def init_shifts(self, field_key: dict, pres_keys: list) -> list | None:
        """Load saved shift values between the field mask and each presentation mask.

        Args:
            field_key: Primary key dict for the field.
            pres_keys: List of primary key dicts, one per presentation.

        Returns:
            A list of (shift_dx, shift_dy) tuples aligned with pres_keys, or
            None if no ROI mask entry exists yet for the field.
        """
        if len(self & field_key) == 0:
            return None
        shifts = []
        for pres_key in pres_keys:
            if len(self.RoiMaskPresentation & pres_key) > 0:
                as_field_mask, shift_dx, shift_dy = \
                    (self.RoiMaskPresentation & pres_key).fetch1("as_field_mask", "shift_dx", "shift_dy")

                if as_field_mask == "same":
                    assert shift_dx == 0 and shift_dy == 0, 'Shifts are non-zero but should be'
                    shifts.append((0, 0))
                elif as_field_mask == "shifted":
                    shifts.append((shift_dx, shift_dy))
                elif as_field_mask == "different":
                    warnings.warn(f"""
                    Inconsistent ROI mask for field_key=\n{field_key}\nand pres_key=\n{pres_key}\n.
                    This is not supported; Presentation will be initialized with Field ROI mask instead.
                    If you really want to define different (i.e. not simply shifted) ROI masks for a field
                    open the GUI for a single Presentation key and save the ROI mask.
                    """)
                    shifts.append((0, 0))
                else:
                    warnings.warn(f"as_field_mask=\n{as_field_mask}\n is not a valid value. Default to zero shift.")
                    shifts.append((0, 0))
            else:
                warnings.warn(f"Pres_key=\n{pres_key}\n was not in RoiMask, but at least one other field_key was. " +
                              f"Default to zero shift.")
                shifts.append((0, 0))
        return shifts

    def load_initial_roi_mask(self, field_key, igor_roi_masks=str, roi_mask_dir='ROIs', old_prefix=None,
                              new_prefix=None, verbose=True):
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
            roi_mask, src_file = self.load_field_roi_mask_pickle(
                field_key=field_key, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix,
                verbose=verbose)
            if roi_mask is not None:
                return roi_mask, src_file

        if igor_roi_masks != 'no':
            roi_mask, src_file = self.load_field_roi_mask_igor(field_key=field_key, verbose=verbose)
            if roi_mask is not None:
                return roi_mask, src_file

        return None, 'none'

    def load_field_roi_mask_database(self, field_key: dict) -> np.ndarray | None:
        """Load the ROI mask stored in the database for the given field.

        Args:
            field_key: Primary key dict for the field.

        Returns:
            The ROI mask array if exactly one entry exists, or None if no entry
            exists.

        Raises:
            ValueError: If more than one ROI mask is found for the key.
        """
        database_roi_masks = (self & field_key).fetch("roi_mask")

        if len(database_roi_masks) == 1:
            database_roi_mask = database_roi_masks[0].copy()
        elif len(database_roi_masks) == 0:
            database_roi_mask = None
        else:
            raise ValueError(f'Found multiple ROI masks for key=\n{field_key}\n{(self & field_key).fetch("KEY")}')

        return database_roi_mask

    def load_field_roi_mask_pickle(
            self,
            field_key: dict,
            roi_mask_dir: str = 'ROIs',
            old_prefix: str = None,
            new_prefix: str = None,
            verbose: bool = True,
    ) -> tuple[np.ndarray | None, str | None]:
        """Load ROI mask from a pickle file on the filesystem.

        Args:
            field_key: Primary key dict for the field.
            roi_mask_dir: Sub-directory name containing ROI mask pickle files.
            old_prefix: Path prefix to replace when resolving file paths.
            new_prefix: Replacement path prefix.
            verbose: If True, print a message when a mask is loaded.

        Returns:
            A 2-tuple of (roi_mask, src_file) where roi_mask is the loaded
            array (or None if not found) and src_file is the source file path
            (or None if not found).
        """
        mask_alias, highres_alias = (self.userinfo_table() & field_key).fetch1("mask_alias", "highres_alias")
        files = (self.presentation_table() & field_key).fetch("pres_data_file")

        roi_mask, src_file = load_preferred_roi_mask_pickle(
            files, mask_alias=mask_alias, highres_alias=highres_alias,
            roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)
        if roi_mask is not None and verbose:
            print(f'Loaded ROI mask from file={src_file} for files=\n{files}\nfor mask_alias={mask_alias}')

        return roi_mask, src_file

    def load_field_roi_mask_igor(
            self,
            field_key: dict,
            verbose: bool = True,
    ) -> tuple[np.ndarray | None, str | None]:
        """Load ROI mask from an Igor h5 file on the filesystem.

        Args:
            field_key: Primary key dict for the field.
            verbose: If True, print a message when a mask is loaded.

        Returns:
            A 2-tuple of (roi_mask, src_file) where roi_mask is the loaded
            array (or None if not found) and src_file is the source file path
            (or None if not found).
        """
        mask_alias, highres_alias, raw_data_dir, pre_data_dir = (self.userinfo_table() & field_key).fetch1(
            "mask_alias", "highres_alias", "raw_data_dir", "pre_data_dir")
        files = (self.presentation_table() & field_key).fetch("pres_data_file")

        files = [as_pre_filepath(f, raw_data_dir=raw_data_dir, pre_data_dir=pre_data_dir) for f in files]

        roi_mask, src_file = load_preferred_roi_mask_igor(files, mask_alias=mask_alias, highres_alias=highres_alias)
        if roi_mask is not None and verbose:
            print(f'Loaded ROI mask from file={src_file} for files=\n{files}\nfor mask_alias={mask_alias}')

        return roi_mask, src_file

    def rescan_filesystem(self, restrictions: dict = None, verboselvl: int = 0, suppress_errors: bool = False,
                          only_new_fields: bool = True, roi_mask_dir='ROIs', old_prefix=None, new_prefix=None,
                          max_shift=None, auto_fill_pres_keys: bool = False, add_primary_keys=None):
        """Scan filesystem for new ROI masks and add them to the database.
        :param restrictions: Restrictions for field_table
        :param verboselvl: Verbosity level
        :param suppress_errors: Suppress errors
        :param only_new_fields: Only scan for new fields
        :param roi_mask_dir: Directory where ROI masks are stored, typically Raw, Pre or AutoROIs
        :param old_prefix: Prefix that should be replaced in file names
        :param new_prefix: Prefix that should replace old_prefix in file names
        :param max_shift: Maximum allowed shift between Presentations, defaults to Table value
        :param auto_fill_pres_keys: Automatically fill presentation keys with zero shift if they are missing as files
        :param add_primary_keys: Additional primary keys to add to keys. Still experimental!
        """
        if restrictions is None:
            restrictions = dict()

        if only_new_fields:
            restrictions = (self.key_source - self) & restrictions

        err_list = []

        for key in (self.key_source & restrictions):
            try:
                self._add_field_roi_masks(
                    key, auto_fill_pres_keys=auto_fill_pres_keys,
                    roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix,
                    max_shift=max_shift, add_primary_keys=add_primary_keys, verboselvl=verboselvl)
            except Exception as e:
                if suppress_errors:
                    warnings.warn(f'Error for key={key}:\n{e}')
                    err_list.append((key, e))
                else:
                    raise e

        return err_list

    def _add_field_roi_masks(
            self,
            field_key: dict,
            auto_fill_pres_keys: bool = False,
            add_primary_keys: dict = None,
            roi_mask_dir: str = 'ROIs',
            old_prefix: str = None,
            new_prefix: str = None,
            max_shift: int = None,
            verboselvl: int = 0,
    ) -> None:
        """Find and insert ROI mask entries for all presentations of a field.

        Args:
            field_key: Primary key dict for the field.
            auto_fill_pres_keys: If True, presentations without a mask file
                are filled with the main field mask (zero shift).
            add_primary_keys: Optional extra primary key columns to inject into
                every inserted row.
            roi_mask_dir: Sub-directory name containing ROI mask pickle files.
            old_prefix: Path prefix to replace when resolving file paths.
            new_prefix: Replacement path prefix.
            max_shift: Maximum allowed pixel shift between presentations.
                Defaults to the table-level ``_max_shift`` value.
            verboselvl: Verbosity level controlling diagnostic output.
        """
        pres_keys = (self.presentation_table & field_key).fetch('KEY')

        if add_primary_keys:
            pres_keys = [{**pk, **add_primary_keys} for pk in pres_keys]
            field_key = {**field_key, **add_primary_keys}

        if verboselvl > 2:
            print('\nfield_key:', field_key, '\npres_keys:', pres_keys)

        roi_masks = [self._load_presentation_roi_mask(key, roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)
                     for key in pres_keys]

        data_pairs = zip(pres_keys, roi_masks)

        if not auto_fill_pres_keys:
            # Filter out keys without ROI mask
            data_pairs = [(pres_key, roi_mask) for pres_key, roi_mask in data_pairs
                          if roi_mask is not None]
        else:
            data_pairs = list(data_pairs)

        if len(data_pairs) == 0:
            if verboselvl > 1:
                print('No ROI masks found for field:', field_key)
            if verboselvl > 2:
                print('pres_keys:', [k for k in pres_keys])
            return

        # Filter out keys that are already present
        data_pairs = [(pres_key, roi_mask) for pres_key, roi_mask in data_pairs
                      if len(self.RoiMaskPresentation & pres_key) == 0]

        if len(data_pairs) == 0:
            if verboselvl > 1:
                print('Nothing new to add for field:', field_key)
            return

        if len(self & field_key) == 0:
            # Find preferred file that should be used as main key.
            mask_alias, highres_alias = (self.userinfo_table & field_key).fetch1("mask_alias", "highres_alias")
            keys_masks_files = [
                (pres_key, roi_mask, (self.presentation_table & pres_key).fetch1('pres_data_file'))
                for pres_key, roi_mask in data_pairs
                if roi_mask is not None]

            if len(keys_masks_files) == 0:
                if verboselvl > 1:
                    print('No ROI masks found for field:', field_key)
                return
            else:
                if verboselvl > 0:
                    print(f'Adding {len(keys_masks_files)} ROI masks for field:', field_key)

            sort_idxs = sort_roi_mask_files(
                files=[f for k, m, f in keys_masks_files],
                mask_alias=mask_alias, highres_alias=highres_alias, as_index=True)

            main_pres_key, main_roi_mask = keys_masks_files[sort_idxs[0]][:2]
        else:
            main_pres_key = (self.RoiMaskPresentation().proj() & (self & field_key)).fetch1('KEY')
            main_roi_mask = (self.RoiMaskPresentation & main_pres_key).fetch1('roi_mask')

        roi_mask_pres_keys = []

        for pres_key, roi_mask in data_pairs:
            if roi_mask is not None:
                as_field_mask, (shift_dx, shift_dy) = compare_roi_masks(
                    roi_mask, main_roi_mask, max_shift=self._max_shift if max_shift is None else max_shift)
                roi_mask_pres_keys.append(
                    {**pres_key, "roi_mask": roi_mask, "as_field_mask": as_field_mask,
                     "shift_dx": shift_dx, "shift_dy": shift_dy})
            elif auto_fill_pres_keys:
                roi_mask_pres_keys.append(
                    {**pres_key, "roi_mask": main_roi_mask, "as_field_mask": 'same',
                     "shift_dx": 0, "shift_dy": 0})
            else:
                raise ValueError(f'No ROI mask found for key={pres_key} but `auto_fill_pres_keys` is False')

        main_key = {**field_key, **main_pres_key, "roi_mask": main_roi_mask}

        if add_primary_keys is not None:
            main_key = {**main_key, **add_primary_keys}
        self.insert1(main_key, skip_duplicates=True)

        for roi_mask_pres_key in roi_mask_pres_keys:
            if add_primary_keys is not None:
                main_key = {**main_key, **add_primary_keys}
                roi_mask_pres_key = {**roi_mask_pres_key, **add_primary_keys}

            self.RoiMaskPresentation().insert1(roi_mask_pres_key, skip_duplicates=True)

    def _load_presentation_roi_mask(
            self,
            key: dict,
            roi_mask_dir: str = 'ROIs',
            old_prefix: str = None,
            new_prefix: str = None,
    ) -> np.ndarray | None:
        """Load the ROI mask for a single presentation, reconciling filesystem and database state.

        Args:
            key: Primary key dict for the presentation.
            roi_mask_dir: Sub-directory name containing ROI mask pickle files.
            old_prefix: Path prefix to replace when resolving file paths.
            new_prefix: Replacement path prefix.

        Returns:
            The ROI mask array, or None if no mask file was found.

        Raises:
            ValueError: If the filesystem mask differs from the database mask.
        """
        igor_roi_masks, from_raw_data = (self.raw_params_table & key).fetch1('igor_roi_masks', 'from_raw_data')
        input_file = (self.presentation_table & key).fetch1("pres_data_file")

        roimask_file = to_roi_mask_file(
            input_file, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)

        if igor_roi_masks == 'yes':
            assert not from_raw_data, 'Inconsistent parameters'
            filesystem_roi_mask = read_h5_utils.load_roi_mask(filepath=input_file, ignore_not_found=True)
        else:
            if os.path.isfile(roimask_file):
                with open(roimask_file, 'rb') as f:
                    filesystem_roi_mask = pickle.load(f).copy().astype(np.int32)
                filesystem_roi_mask = to_igor_format(filesystem_roi_mask)
            else:
                filesystem_roi_mask = None

        if len((self.RoiMaskPresentation & key).proj()) > 0:
            database_roi_mask = (self.RoiMaskPresentation & key).fetch1('roi_mask')

            if filesystem_roi_mask is None:
                warnings.warn(
                    f'ROI mask for key=\n{key}\nhas been deleted on the filesystem but not in the database.\n'
                    f'Saving ROI masks to file now: {roimask_file}'
                )
                with open(roimask_file, 'wb') as f:
                    pickle.dump(to_python_format(database_roi_mask), f)

            elif not np.all(filesystem_roi_mask == database_roi_mask):
                raise ValueError(f'ROI mask for key=\n{key}\nhas been changed on filesystem but not in database.')
            else:
                filesystem_roi_mask = database_roi_mask.copy()

        return filesystem_roi_mask

    def plot1(self, key: dict = None, gamma: float = 0.5) -> None:
        """Plot the stack average with the ROI mask overlay for one presentation.

        Args:
            key: Primary key identifying the presentation to plot. If None,
                the first available key is used.
            gamma: Gamma correction value for display. Default is 0.5.
        """
        key = get_primary_key(table=self.proj() * self.presentation_table.proj(), key=key)
        npixartifact, scan_type = (self.field_table & key).fetch1('npixartifact', 'scan_type')
        data_name, alt_name = (self.userinfo_table & key).fetch1('data_stack_name', 'alt_stack_name')
        main_ch_average = (self.presentation_table.StackAverages & key & f'ch_name="{data_name}"').fetch1('ch_average')
        try:
            alt_ch_average = (self.presentation_table.StackAverages & key & f'ch_name="{alt_name}"').fetch1(
                'ch_average')
        except dj.DataJointError:
            alt_ch_average = np.full_like(main_ch_average, np.nan)

        roi_mask = (self.RoiMaskPresentation & key).fetch1('roi_mask')
        plot_field(main_ch_average, alt_ch_average, scan_type=scan_type,
                   roi_mask=roi_mask, title=key, npixartifact=npixartifact, gamma=gamma)

    def load_high_res_bg_dict(self, key: dict, ch_names: list, verbose: bool = True) -> dict:
        """Load high-resolution channel averages as a background dict for the GUI.

        Args:
            key: Primary key dict for the field.
            ch_names: List of channel name strings to load.
            verbose: If True, warn when loading fails.

        Returns:
            A dict mapping display names (e.g. 'HR-wDataCh0') to 2-D
            np.ndarray channel-average images. Returns an empty dict on
            failure.
        """
        try:
            ch_names, ch_averages, *conds = (
                    self.highres_table().StackAverages & key & [f"ch_name='{cn}'" for cn in ch_names]).fetch(
                'ch_name', 'ch_average', *self.highres_table().new_primary_keys)
            bg_dict = dict()

            for name, ch_average, *cond_123 in zip(ch_names, ch_averages, *conds):
                dict_name = f'HR-{name}'
                for cond in cond_123:
                    dict_name += f'-{cond}'
                bg_dict[dict_name] = ch_average

        except Exception as e:
            if verbose:
                warnings.warn(f'Failed to load high resolution data for key\n{key}\nbecause of error:\n{e}')
            return dict()

        return bg_dict


def load_stack_data(
        files: list,
        data_name: str,
        alt_name: str,
        from_raw_data: bool,
        roi_mask_dir: str = 'ROIs',
        old_prefix: str = None,
        new_prefix: str = None,
) -> tuple[list, list, list]:
    """Load image stacks for a list of presentation files.

    Args:
        files: List of file paths to load stacks from.
        data_name: Name of the primary data channel (e.g. 'wDataCh0').
        alt_name: Name of the alternative channel (e.g. 'wDataCh1').
        from_raw_data: If True, read from raw ScanM files instead of h5.
        roi_mask_dir: Sub-directory name for ROI mask output files.
        old_prefix: Path prefix to replace in output file paths.
        new_prefix: Replacement path prefix.

    Returns:
        A 3-tuple of (ch0_stacks, ch1_stacks, output_files) where each is a
        list aligned with ``files``. ch0_stacks and ch1_stacks contain 3-D
        np.ndarray stacks; output_files contains the corresponding ROI mask
        output file paths.
    """
    ch0_stacks, ch1_stacks, output_files = [], [], []
    for data_file in files:
        ch_stacks, wparams = read_utils.load_stacks(
            data_file, from_raw_data=from_raw_data, ch_names=(data_name, alt_name))
        ch0_stacks.append(ch_stacks[data_name])
        ch1_stacks.append(ch_stacks[alt_name])
        output_files.append(to_roi_mask_file(
            data_file, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix))

    return ch0_stacks, ch1_stacks, output_files


def load_default_autorois_models(kind: str = 'default_rgc') -> dict | None:
    """Load the default set of auto-ROI models for a given cell type.

    Args:
        kind: Model variant to load. Supported values are 'default_rgc',
            'default_bc', and 'default_ac'.

    Returns:
        A dict mapping model names to model objects, or None if the dict is
        empty (e.g. if all models failed to load).
    """
    autorois_models = dict()
    if kind == 'default_rgc':
        _add_autorois_unet(autorois_models)
    _add_autorois_corr(autorois_models, kind=kind)
    return autorois_models if len(autorois_models) > 0 else None


def _add_autorois_unet(autorois_models: dict) -> None:
    """Attempt to load a UNet auto-ROI model and add it to the model dict.

    Args:
        autorois_models: Dict to add the loaded model to under the key 'UNet'.
    """
    try:
        from djimaging.autorois.unet import UNet
        config_path = "/gpfs01/euler/data/Resources/AutoROIs/models/UNET_v0.1.0/sd_images.yaml"
        checkpoint_path = "/gpfs01/euler/data/Resources/AutoROIs/models/UNET_v0.1.0/dropout_and_aug_regul.ckpt"
        unet_model = UNet.from_checkpoint(config_path, checkpoint_path)

        autorois_models['UNet'] = unet_model

    except Exception as e:
        warnings.warn(f'Failed to load default AutoROIs models because of error:\n{e}')


def _add_autorois_corr(autorois_models: dict, kind: str) -> None:
    """Load a correlation-based auto-ROI model and add it to the model dict.

    Args:
        autorois_models: Dict to add the loaded model to under the key
            'CorrRoiMask'.
        kind: Model variant controlling hyperparameters. Supported values are
            'default_rgc', 'default_bc', and 'default_ac'.
    """
    from djimaging.autorois.corr_roi_mask_utils import CorrRoiMask

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


def _add_cellpose(autorois_models: dict, kind: str = 'default_rgc') -> None:
    """Load a Cellpose auto-ROI model and add it to the model dict.

    Args:
        autorois_models: Dict to add the loaded model to under the key
            'Cellpose'.
        kind: Model variant. Currently only 'default_rgc' is supported.
    """
    from djimaging.autorois.cellpose_wrapper import CellposeWrapper

    if kind == 'default_rgc':
        init_params = dict(
            model_type='cyto',
            gpu=False,
        )

        eval_params = dict(
            min_size=4,
            diameter=15,
            channels=[0, 0],
        )
    else:
        raise NotImplementedError(kind)

    model_cellpose = CellposeWrapper(init_kwargs=init_params, eval_kwargs=eval_params)
    autorois_models['Cellpose'] = model_cellpose
