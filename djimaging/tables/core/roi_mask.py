from __future__ import annotations

import os
import warnings
from abc import abstractmethod
from pathlib import Path

import datajoint as dj
import numpy as np

from djimaging.utils.scanm import read_h5_utils, read_utils

from djimaging.autorois import default_models
from djimaging.autorois.autoshift_utils import compute_corr_map, correlation_on_centered, compute_corr_map_match_indexes, extract_best_shift
from djimaging.autorois.roi_canvas import InteractiveRoiCanvas

from djimaging.utils.filesystem_utils import as_pre_filepath
from djimaging.utils.dj_storage import file_store_path, load_array, local_file_path, relative_store_path
from djimaging.utils.dj_utils import get_primary_key, check_unique_one
from djimaging.utils.mask_utils import (
    to_roi_mask_file, sort_roi_mask_files,
    load_preferred_roi_mask_igor, load_preferred_roi_mask_file, compare_roi_masks,
    save_roi_mask_file, load_roi_mask_file, relabel_mask, clean_rois, shift_array, add_rois,
    generate_roi_suggestions, estimate_vessel_mask, remove_vessel_rois, select_field_roi_mask,
    to_field_roi_mask_file, mask_fit_score, shrink_rois_to_fit, _ROI_FILE_FORMAT_HELP,
)
from djimaging.utils.mask_format_utils import to_igor_format, to_python_format
from djimaging.utils.plot_utils import plot_field, field_extent_and_labels


class RoiMaskTemplate(dj.Manual):
    database = ""
    _max_shift = 5
    _roi_file_format: str = 'numpy'
    _filepath_store = "reference"
    # Per-presentation masks in the part table RoiMaskPresentation (same / shifted / different from the field mask).
    # False = motion mode: only the field mask (and the presentation it belongs to) is stored here; the presentation
    # masks are computed by djimaging.tables.motion_correction (MotionRoiMask). The concrete table then defines
    # `RoiMaskPresentation = None`.
    _presentation_masks: bool = True
    # Choose the presentation of the field mask from the data instead of only by `mask_alias`, and align the mask to
    # its mean image (see `select_field_roi_mask`). None: on in motion mode (`_presentation_masks=False`), where the
    # presentation of the field mask is the motion reference.
    _select_field_mask: bool | None = None
    _field_mask_max_offset: int = 2  # maximum alignment offset [px] of the field mask
    # Motion mode: a field mask file (`<stem>_FieldROIs.npy` in roi_mask_dir, see `to_field_roi_mask_file`) holds the
    # one mask of a field and names the presentation it belongs to. It is preferred over presentation mask files and
    # Igor masks, which are then only (copied) input for choosing and aligning the field mask.

    @property
    def _selects_field_mask(self) -> bool:
        return (not self._presentation_masks) if self._select_field_mask is None else bool(self._select_field_mask)

    @property
    def definition(self):
        definition = f"""
        # ROI mask
        -> self.field_table
        -> self.raw_params_table
        ---
        -> self.presentation_table
        roi_mask     : <blob>                   # ROI mask for recording field
        roi_mask_file   : <filepath@{self._filepath_store}>  # file the ROI mask was loaded from (relative to store)
        roi_mask_source : enum('igor_h5', 'numpy', 'pickle', 'other')  # format of roi_mask_file
        """
        if self._selects_field_mask:
            definition += """
        align_dx : int32  # roi_mask = shift_array(mask in roi_mask_file, (align_dx, align_dy), cval=1)
        align_dy : int32
        mask_fit=null : float32  # fit of roi_mask to the mean image of its presentation (mask_fit_score)
        mask_selection : enum('fit', 'alias', 'file')  # presentation chosen by fit / mask_alias (no fit) / field mask file
        """
        return definition

    class RoiMaskPresentation(dj.Part):
        _filepath_store = "reference"  # keep in sync with the master table

        @property
        def definition(self):
            definition = f"""
            # ROI Mask
            -> master
            -> self.presentation_table
            ---
            roi_mask      : <blob>       # ROI mask for presentation field
            as_field_mask : enum('same', 'different', 'shifted')  # relationship to field mask
            shift_dx=0    : int32  # Shift in x
            shift_dy=0    : int32  # Shift in y
            roi_mask_file   : <filepath@{self._filepath_store}>  # file the ROI mask was loaded from (relative to store)
            roi_mask_source : enum('igor_h5', 'numpy', 'pickle', 'other')  # format of roi_mask_file
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

    def _get_stack_names(self, field_key: dict, roi_channel: str | None = None) -> tuple[list[str], str]:
        """Names of the stacks that are loaded into the ROI canvas and name of the target channel (listed first).

        The stacks are defined by the user info, see `UserInfoTemplate.get_ch_names`.
        """
        ch_names = self.userinfo_table().get_ch_names(field_key)
        roi_channel = ch_names[0] if roi_channel is None else roi_channel

        if roi_channel not in ch_names:
            raise ValueError(f"roi_channel={roi_channel!r} is not one of the channels {ch_names}")
        return [roi_channel] + [ch_name for ch_name in ch_names if ch_name != roi_channel], roi_channel

    def _get_roi_file_format(self) -> str:
        """Return the configured ROI file format, or raise a descriptive error if unset."""
        if self._roi_file_format is None:
            raise ValueError(
                f"_roi_file_format is not set on {type(self).__name__}. "
                f"{_ROI_FILE_FORMAT_HELP} "
                f"Add `_roi_file_format = 'numpy'` (recommended) or "
                f"`_roi_file_format = 'pickle'` for existing pickle masks as a class attribute."
            )
        return self._roi_file_format

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
        missing_keys = (self.field_table.proj() & (self.presentation_table.proj() - self.proj())).to_dicts()
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
            field_key: dict | None = None,
            pres_key: dict | None = None,
            canvas_width: int = 20,
            autorois_models: str | list | dict = 'default_rgc',
            show_diagnostics: bool = True,
            load_high_res: bool = True,
            max_shift: int | None = None,
            roi_mask_dir: str = 'ROIs',
            old_prefix: str | None = None,
            new_prefix: str | None = None,
            use_stim_onset: bool = True,
            verbose: bool = True,
            roi_channel: str | None = None,
            roi_table: dj.Table | None = None,
            motion_roi_mask_table: dj.Table | None = None,
            motion_params_id: int | None = None,
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
            autorois_models: Either a dict of auto-ROI models, a string key
                ('default_rgc', etc.) to load the default model set (latest UNet), or a list of model
                names (e.g. ['UNet', 'Cellpose3_cyto3', 'MEDIAR']) to load specific models, see
                ``load_default_autorois_models``.
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
            roi_channel: Name of the channel (e.g. 'wDataCh1') that is the initial target for ROI detection
                (cc tool, diagnostics, auto shift, AutoROIs). Can be changed in the GUI. Defaults to the channel
                flagged `roi_default` in the channel table (`data_stack_name` of the user info without channel table).
                All channels are shown as background.
            roi_table: The ROI table (``RoiTemplate``). If the field already has an ROI mask in the database, the GUI
                can replace it ('Deploy to database', only in the dangerzone), see ``deploy_roi_mask``. The ROI table is
                then repopulated for the field. Without it the GUI cannot deploy.
            motion_roi_mask_table: Motion mode only: ``MotionRoiMask``. If it is populated for the field, the other
                presentations show its presentation masks (view only).
            motion_params_id: Motion mode only: parameter set of ``motion_roi_mask_table``. Only needed if the field
                has several.
            **kwargs: Additional keyword arguments forwarded to
                ``InteractiveRoiCanvas``.

        In motion mode (no presentation masks) the mask is drawn and edited on one presentation only: the canvas opens
        on the presentation the initial mask belongs to (the presentation stored in the database, else the one of the
        mask file, else the ``mask_alias`` choice). Shifts between presentations are disabled. The other presentations
        are view only and show the presentation masks of ``MotionRoiMask`` (if populated and the mask was not edited),
        else a preview: the mask moved by the global shift between the mean images (auto shift). 'Save all to file'
        writes one field mask file (``<stem>_FieldROIs.npy``, see ``to_field_roi_mask_file``) instead of one file per
        presentation; ``deploy_roi_mask`` loads it.

        Returns:
            An ``InteractiveRoiCanvas`` instance. Call ``.start_gui()`` on it
            to open the interactive widget.
        """
        if canvas_width <= 0 or canvas_width >= 100:
            raise ValueError(f'canvas_width={canvas_width} must be in (0, 100)%')

        if pres_key is not None:
            field_key = (self.field_table & pres_key).proj().fetch1()
        elif field_key is None:
            field_key = np.random.choice(self.list_missing_field())

        if field_key is None:
            raise ValueError('No field_key provided and no missing field found.')
        field_key = (self.field_table & field_key).proj().fetch1()

        field_data = self._load_field_stacks(
            field_key, roi_channel=roi_channel, roi_mask_dir=roi_mask_dir,
            old_prefix=old_prefix, new_prefix=new_prefix, use_stim_onset=use_stim_onset)
        pres_keys, pres_names, stacks, output_files = (
            field_data[k] for k in ('pres_keys', 'pres_names', 'stacks', 'output_files'))
        ch_names, roi_channel = field_data['ch_names'], field_data['roi_channel']
        n_artifact, pixel_size_d1_d2 = field_data['n_artifact'], field_data['pixel_size_d1_d2']

        # Load high resolution data is possible
        high_res_bg_dict = self.load_high_res_bg_dict(
            field_key, ch_names=ch_names, verbose=verbose) if load_high_res else dict()

        # Load initial ROI masks
        igor_roi_masks = (self.raw_params_table & field_key).fetch1('igor_roi_masks')
        initial_roi_mask, initial_src = self.load_initial_roi_mask(
            field_key=field_key, igor_roi_masks=igor_roi_masks,
            roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix, verbose=verbose)

        if initial_roi_mask is not None:
            initial_roi_mask = to_python_format(initial_roi_mask)

        # Load default AutoROIs models
        if isinstance(autorois_models, str):
            autorois_models = load_default_autorois_models(autorois_models)
        elif isinstance(autorois_models, (list, tuple)):
            autorois_models = load_default_autorois_models(models=list(autorois_models))

        # Load shifts if available
        shifts = self.init_shifts(field_key, pres_keys)

        # Motion mode: draw on the presentation of the mask, no shifts between presentations, one field mask file
        main_stim_idx, view_masks = 0, None
        if not self._presentation_masks:
            filepaths = field_data['filepaths']
            field_files = [to_field_roi_mask_file(f, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix,
                                                  new_prefix=new_prefix) for f in filepaths]
            main_stim_idx = self._main_presentation_idx(
                field_key, pres_keys, filepaths, initial_src, mask_files=[field_files, output_files])
            output_files = [field_files[main_stim_idx]] * len(pres_keys)
            if motion_roi_mask_table is not None:
                view_masks = self._motion_view_masks(field_key, pres_keys, motion_roi_mask_table, motion_params_id)

        roi_canvas = InteractiveRoiCanvas(
            pres_names=pres_names,
            stacks=stacks, primary_channel=roi_channel, n_artifact=n_artifact, bg_dict=high_res_bg_dict,
            main_stim_idx=main_stim_idx, initial_roi_mask=initial_roi_mask, shifts=shifts,
            allow_shifts=self._presentation_masks, view_masks=view_masks,
            canvas_width=canvas_width, autorois_models=autorois_models, output_files=output_files,
            pixel_size_um=pixel_size_d1_d2, show_diagnostics=show_diagnostics,
            max_shift=self._max_shift if max_shift is None else max_shift,
            deploy_callback=None if roi_table is None else (lambda: self.deploy_roi_mask(
                field_key, roi_table=roi_table, roi_mask_dir=roi_mask_dir,
                old_prefix=old_prefix, new_prefix=new_prefix, max_shift=max_shift, verbose=verbose)),
            **kwargs,
        )
        if verbose:
            if not self._presentation_masks:
                print(f"Motion mode: draw on {pres_names[main_stim_idx]} (other presentations are view only); "
                      f"the mask is saved to {output_files[main_stim_idx]}")
            print(f"Returned InteractiveRoiCanvas object. To start GUI, call <enter_object_name>.start_gui().")
        return roi_canvas

    def _main_presentation_idx(self, field_key: dict, pres_keys: list, filepaths: list, initial_src: str,
                               mask_files: list) -> int:
        """Index of the presentation an initial field mask belongs to: the presentation stored in the database,
        else the presentation of the mask file ``initial_src`` (one of the per-presentation file lists in
        ``mask_files``, or the h5 file for Igor masks), else the first one in ``mask_alias`` order."""
        def same_file(a, b):
            return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(os.path.abspath(str(b)))

        pres_primary_key = self.presentation_table.primary_key
        if initial_src == 'database':
            row = (self & field_key).fetch1()
            idxs = [i for i, pk in enumerate(pres_keys) if all(pk[k] == row[k] for k in pres_primary_key)]
            if len(idxs) == 1:
                return idxs[0]
        elif initial_src not in (None, 'none'):
            idxs = []
            for files in mask_files:
                idxs = idxs or [i for i, f in enumerate(files) if same_file(f, initial_src)]
            if len(idxs) == 0:  # Igor mask: stored in the h5 file of the presentation
                raw_data_dir, pre_data_dir = (self.userinfo_table & field_key).fetch1('raw_data_dir', 'pre_data_dir')
                idxs = [i for i, f in enumerate(filepaths) if same_file(
                    as_pre_filepath(f, raw_data_dir=raw_data_dir, pre_data_dir=pre_data_dir), initial_src)]
            if len(idxs) == 1:
                return idxs[0]

        mask_alias, highres_alias = (self.userinfo_table & field_key).fetch1("mask_alias", "highres_alias")
        return int(sort_roi_mask_files(
            files=list(filepaths), mask_alias=mask_alias, highres_alias=highres_alias, as_index=True)[0])

    def _motion_view_masks(self, field_key: dict, pres_keys: list, motion_roi_mask_table,
                           motion_params_id: int | None) -> list | None:
        """Presentation masks of MotionRoiMask (python format, raw presentation coordinates) for the ROI canvas, or
        None if MotionRoiMask is not populated for the field. Missing presentations are None."""
        restr = dict() if motion_params_id is None else dict(motion_params_id=motion_params_id)
        rows = motion_roi_mask_table & field_key & restr
        if len(rows) == 0:
            return None
        param_ids = set(rows.to_arrays('motion_params_id'))
        if len(param_ids) > 1:
            raise ValueError(f'MotionRoiMask has several parameter sets {sorted(param_ids)} for the field, '
                             f'choose one with motion_params_id.')
        view_masks = []
        for pres_key in pres_keys:
            pres_rows = rows & pres_key
            view_masks.append(to_python_format(load_array(pres_rows.fetch1('roi_mask')))
                              if len(pres_rows) == 1 else None)
        return view_masks

    def _field_roi_mask_files(self, pres_keys: list, roi_mask_dir: str, old_prefix: str | None = None,
                              new_prefix: str | None = None) -> list:
        """Field mask file (``<stem>_FieldROIs.npy``) of each presentation, see ``to_field_roi_mask_file``."""
        return [to_field_roi_mask_file(
            local_file_path((self.presentation_table & pres_key).fetch1("pres_data_file"),
                            self.presentation_table._filepath_store),
            roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix) for pres_key in pres_keys]

    def _find_field_roi_mask_file(self, pres_keys: list, roi_mask_dir: str, old_prefix: str | None = None,
                                  new_prefix: str | None = None) -> tuple:
        """The existing field mask file of a field and its presentation: (pres_key, file) or (None, None).

        Raises:
            ValueError: If several presentations of the field have a field mask file.
        """
        files = self._field_roi_mask_files(pres_keys, roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)
        found = [(pres_key, f) for pres_key, f in zip(pres_keys, files) if os.path.isfile(f)]
        if len(found) > 1:
            raise ValueError(f'Several field ROI mask files for one field, delete all but one:\n'
                             f'{[f for _, f in found]}')
        return found[0] if len(found) == 1 else (None, None)

    def _load_field_stacks(
            self,
            field_key: dict,
            roi_channel: str | None = None,
            roi_mask_dir: str = 'ROIs',
            old_prefix: str | None = None,
            new_prefix: str | None = None,
            use_stim_onset: bool = True,
    ) -> dict:
        """Load the stacks of all presentations of a field, as used for ROI detection.

        Returns:
            A dict with the presentation keys (``pres_keys``) and names (``pres_names``), the stacks of each channel
            (``stacks``, channel name -> list with one stack per presentation), the ROI mask file of each presentation
            (``output_files``), the channel names (``ch_names``) and target channel (``roi_channel``), the number of
            artifact rows (``n_artifact``) and the pixel size (``pixel_size_d1_d2``).
        """
        from_raw_data = (self.raw_params_table & field_key).fetch1("from_raw_data")

        field_file_info_df = self.load_field_file_info_df(field_key)
        all_filepaths = field_file_info_df[field_file_info_df['kind'] == 'response']['filepath'].values

        pres_keys = []
        filepaths = []

        for f in all_filepaths:
            store_path = relative_store_path(f, self.presentation_table._filepath_store)
            pres_key_list = (
                (self.presentation_table & field_key) & {"pres_data_file.path": store_path}
            ).proj().to_dicts()
            if len(pres_key_list) == 1:
                pres_keys.append(pres_key_list[0])
                filepaths.append(f)
            elif len(pres_key_list) > 1:
                raise ValueError(f'Found multiple presentation keys for filepath={f}:\n{pres_key_list}')

        pres_names = [{k: v for k, v in pres_key.items() if k not in field_key.keys()}
                      for pres_key in pres_keys]

        # Load stack data
        ch_names, roi_channel = self._get_stack_names(field_key, roi_channel=roi_channel)
        stacks, output_files = load_stack_data(
            files=filepaths, ch_names=ch_names, from_raw_data=from_raw_data,
            roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix,
            file_format=self._get_roi_file_format())

        # Some sanity checks
        assert len(pres_names) == len(pres_keys)
        assert len(filepaths) == len(pres_keys)
        assert all(len(ch_stacks) == len(pres_keys) for ch_stacks in stacks.values())
        assert len(output_files) == len(pres_keys)

        # Load pixel size and scan type
        n_artifact, pixel_size_um, scan_type = (self.presentation_table() & pres_keys).to_arrays(
            'npixartifact', 'pixel_size_um', 'scan_type')
        n_artifact = check_unique_one(n_artifact, name='n_artifact')
        pixel_size_um = check_unique_one(pixel_size_um, name='pixel_size_um')
        scan_type = check_unique_one(scan_type, name='scan_type')

        if scan_type == 'xy':
            pixel_size_d1_d2 = (pixel_size_um, pixel_size_um)
        elif scan_type == 'xz':
            z_step_um = (self.presentation_table() & pres_keys).to_arrays('z_step_um')
            z_step_um = check_unique_one(z_step_um, name='z_step_um')
            pixel_size_d1_d2 = (pixel_size_um, z_step_um)
        else:
            raise NotImplementedError(scan_type)

        # Reduce stacks to after stimulus onset?
        if use_stim_onset:
            triggertimes = (self.presentation_table & pres_keys).to_arrays('triggertimes')
            scan_frequencies = (self.presentation_table.ScanInfo() & pres_keys).to_arrays('scan_frequency')

            stim_onset_idxs = [int(np.floor(tt[0] * fs)) if len(tt) > 0 else 0
                               for tt, fs in zip(triggertimes, scan_frequencies)]
            stacks = {ch_name: [stack[:, :, stim_onset_idx:] for stack, stim_onset_idx in zip(ch_stacks, stim_onset_idxs)]
                      for ch_name, ch_stacks in stacks.items()}

        return dict(
            pres_keys=pres_keys, pres_names=pres_names, filepaths=filepaths, stacks=stacks, output_files=output_files,
            ch_names=ch_names, roi_channel=roi_channel, n_artifact=n_artifact, pixel_size_d1_d2=pixel_size_d1_d2)

    def init_shifts(self, field_key: dict, pres_keys: list) -> list | None:
        """Load saved shift values between the field mask and each presentation mask.

        Args:
            field_key: Primary key dict for the field.
            pres_keys: List of primary key dicts, one per presentation.

        Returns:
            A list of (shift_dx, shift_dy) tuples aligned with pres_keys, or
            None if no ROI mask entry exists yet for the field.
        """
        if len(self & field_key) == 0 or not self._presentation_masks:
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
        In motion mode then try the field mask file (``<stem>_FieldROIs.npy``).
        Second try filesystem file (numpy or pickle), unless ROI masks should be exclusively loaded from Igor.
        Last try to load from Igor file, unless ROI masks should never be loaded from Igor.
        """

        roi_mask = self.load_field_roi_mask_database(field_key=field_key)

        if roi_mask is not None:
            return roi_mask, 'database'

        if not self._presentation_masks:  # Motion mode: field mask file
            pres_keys = (self.presentation_table & field_key).proj().to_dicts()
            _, field_file = self._find_field_roi_mask_file(
                pres_keys, roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)
            if field_file is not None:
                if verbose:
                    print(f'Loaded field ROI mask from file={field_file}')
                return to_igor_format(load_roi_mask_file(field_file).astype(np.int32)), field_file

        if igor_roi_masks != 'yes':
            roi_mask, src_file = self.load_field_roi_mask_file(
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
        database_roi_masks = (self & field_key).to_arrays("roi_mask")

        if len(database_roi_masks) == 1:
            database_roi_mask = database_roi_masks[0].copy()
        elif len(database_roi_masks) == 0:
            database_roi_mask = None
        else:
            raise ValueError(f'Found multiple ROI masks for key=\n{field_key}\n{(self & field_key).keys()}')

        return database_roi_mask

    def load_field_roi_mask_file(
            self,
            field_key: dict,
            roi_mask_dir: str = 'ROIs',
            old_prefix: str = None,
            new_prefix: str = None,
            verbose: bool = True,
    ) -> tuple[np.ndarray | None, str | None]:
        """Load an ROI mask file in the configured NumPy or pickle format.

        Args:
            field_key: Primary key dict for the field.
            roi_mask_dir: Sub-directory name containing ROI mask files.
            old_prefix: Path prefix to replace when resolving file paths.
            new_prefix: Replacement path prefix.
            verbose: If True, print a message when a mask is loaded.

        Returns:
            A 2-tuple of (roi_mask, src_file) where roi_mask is the loaded
            array (or None if not found) and src_file is the source file path
            (or None if not found).
        """
        mask_alias, highres_alias = (self.userinfo_table() & field_key).fetch1("mask_alias", "highres_alias")
        files = [
            local_file_path(value, self.presentation_table._filepath_store)
            for value in (self.presentation_table() & field_key).to_arrays("pres_data_file")
        ]

        roi_mask, src_file = load_preferred_roi_mask_file(
            files, mask_alias=mask_alias, highres_alias=highres_alias,
            roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix,
            file_format=self._get_roi_file_format())
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
        files = [
            local_file_path(value, self.presentation_table._filepath_store)
            for value in (self.presentation_table() & field_key).to_arrays("pres_data_file")
        ]

        files = [as_pre_filepath(f, raw_data_dir=raw_data_dir, pre_data_dir=pre_data_dir) for f in files]

        roi_mask, src_file = load_preferred_roi_mask_igor(files, mask_alias=mask_alias, highres_alias=highres_alias)
        if roi_mask is not None and verbose:
            print(f'Loaded ROI mask from file={src_file} for files=\n{files}\nfor mask_alias={mask_alias}')

        return roi_mask, src_file

    def create_missing_roi_masks(
            self,
            model: str | None = None,
            kind: str = 'default_rgc',
            autorois_models: dict | None = None,
            restrictions: dict | None = None,
            roi_mask_dir: str = 'AutoROIs',
            old_prefix: str | None = None,
            new_prefix: str | None = None,
            roi_channel: str | None = None,
            use_stim_onset: bool = True,
            use_all_stacks: bool = False,
            clean: bool = True,
            min_size: int = 3,
            connectivity: int = 2,
            vessel_threshold: float | None = 50.,
            max_vessel_overlap: float = 0.2,
            auto_shift: bool = True,
            max_shift: int | None = None,
            overwrite_files: bool = False,
            suppress_errors: bool = False,
            verbose: bool = True,
    ) -> dict:
        """Generate ROI mask files with an AutoROIs model for all fields that have no ROI mask yet (no GUI).

        For each field without an entry in this table, the model proposes a mask on the stacks of the first
        presentation (the reference). The mask is optionally cleaned, shifted to every other presentation
        (auto shift, as the 'Auto shift' button of the ROI canvas), and written to one file per presentation in
        ``roi_mask_dir``, exactly as 'Save all to file' in the ROI canvas does. Nothing is written to the database;
        load the files afterwards with ``rescan_filesystem(roi_mask_dir=...)``.

        Args:
            model: Name of the AutoROIs model, see ``load_default_autorois_models``: 'UNet', 'UNet_v1',
                'Cellpose3_cyto2', 'Cellpose3_cyto3', 'MEDIAR' or 'CorrRoiMask'. Defaults to 'UNet' for
                ``kind='default_rgc'`` and to 'CorrRoiMask' for all other kinds.
            kind: Cell type variant of the model ('default_rgc', 'default_bc', 'default_ac', 'default_pr' or
                'default_pr_watershed'). Only 'CorrRoiMask' is available for kinds other than 'default_rgc'.
            autorois_models: Already loaded models (dict name -> model), e.g. from ``load_default_autorois_models``,
                from which ``model`` is picked. If None, only the requested model is loaded and released again
                at the end.
            restrictions: Restrictions for the fields, e.g. ``dict(experimenter='Ran')``.
            roi_mask_dir: Sub-directory (next to the recording files) the mask files are written to.
            old_prefix: Path prefix to replace when resolving file paths.
            new_prefix: Replacement path prefix.
            roi_channel: Name of the channel the model is applied to. Defaults to the channel flagged as ROI
                channel (see ``UserInfoTemplate.get_ch_names``). The first other channel is passed as second input,
                which only ``UNet_v1`` uses.
            use_stim_onset: If True, only use the recording after the stimulus onset (first trigger).
            use_all_stacks: If True, the model is applied to the stacks of all presentations of a field
                (concatenated in time) instead of only the first one. Only useful if presentations are aligned.
            clean: If True, clean the mask (remove the light artifact rows, split disconnected ROIs and remove
                ROIs smaller than ``min_size`` pixels), see ``clean_rois``.
            min_size: Minimum ROI size in pixels for cleaning.
            connectivity: 1 (only NSEW neighbours) or 2 (also diagonal neighbours) connect pixels of a ROI.
            vessel_threshold: If not None, remove ROIs on blood vessels, which the model sometimes mistakes for
                cells. Vessels are found by comparing the channels: the alternative channel (the first channel other than
                ``roi_channel``, e.g. SR101) is predicted from the ROI channel (cells can be bright in both), and
                pixels that stand out more than this many robust standard deviations from that prediction count as
                vessel. Fields without vessels yield no vessel pixels. Default 50; lower values find more vessel pixels. None switches the check off. It is skipped
                (with a warning) if the recording has no second channel.
                See ``mask_utils.estimate_vessel_mask``.
            max_vessel_overlap: A ROI is removed if more than this fraction of its pixels are on vessels.
            auto_shift: If True, estimate the shift of every other presentation relative to the first one from
                the correlation maps. Otherwise all presentations get the same mask without shift.
            max_shift: Largest shift in pixels that auto shift searches for. Defaults to the table-level value.
            overwrite_files: If False, fields for which the mask files of all presentations already exist are
                skipped (load them with ``rescan_filesystem`` instead). If True, existing files are overwritten.
            suppress_errors: If True, continue with the next field if a field fails and report the error.
            verbose: If True, print progress.

        Returns:
            A dict with the lists ``created`` (field keys with new mask files), ``skipped`` (field keys whose files
            already exist) and ``errors`` (tuples of field key and exception).
        """
        if connectivity not in (1, 2):
            raise ValueError(f'connectivity={connectivity} must be 1 or 2')
        max_shift = self._max_shift if max_shift is None else max_shift

        restrictions = dict() if restrictions is None else restrictions
        field_keys = (self.field_table.proj() & (self.presentation_table.proj() - self.proj()) & restrictions
                      ).to_dicts()
        result = dict(created=[], skipped=[], errors=[])
        if len(field_keys) == 0:
            if verbose:
                print('No fields without ROI mask found.')
            return result

        model, autorois_model, loaded_here = _get_autorois_model(model, kind, autorois_models)
        del autorois_models

        try:
            for i, field_key in enumerate(field_keys, start=1):
                if verbose:
                    print(f'[{i}/{len(field_keys)}] {model}: {field_key}')
                try:
                    created = self._create_field_roi_mask_files(
                        field_key, autorois_model, roi_channel=roi_channel, roi_mask_dir=roi_mask_dir,
                        old_prefix=old_prefix, new_prefix=new_prefix, use_stim_onset=use_stim_onset,
                        use_all_stacks=use_all_stacks, clean=clean, min_size=min_size, connectivity=connectivity,
                        vessel_threshold=vessel_threshold, max_vessel_overlap=max_vessel_overlap,
                        auto_shift=auto_shift, max_shift=max_shift, overwrite_files=overwrite_files)
                    result['created' if created else 'skipped'].append(field_key)
                    if verbose and not created:
                        print('  Mask files already exist, skipped. Load them with `rescan_filesystem`.')
                except Exception as e:
                    if not suppress_errors:
                        raise
                    warnings.warn(f'Error for field_key={field_key}:\n{e}')
                    result['errors'].append((field_key, e))
        finally:
            del autorois_model
            if loaded_here:
                _free_memory()

        if verbose:
            print(f"Created mask files for {len(result['created'])} fields, skipped {len(result['skipped'])}, "
                  f"{len(result['errors'])} errors. Load the new files with `rescan_filesystem(roi_mask_dir=...)`.")
        return result

    def _create_field_roi_mask_files(
            self, field_key: dict, autorois_model, roi_channel, roi_mask_dir, old_prefix, new_prefix,
            use_stim_onset, use_all_stacks, clean, min_size, connectivity, auto_shift, max_shift, overwrite_files,
            vessel_threshold=None, max_vessel_overlap=0.2,
    ) -> bool:
        """Create the mask files of all presentations of one field. Returns False if skipped because files exist."""
        data = self._load_field_stacks(
            field_key, roi_channel=roi_channel, roi_mask_dir=roi_mask_dir,
            old_prefix=old_prefix, new_prefix=new_prefix, use_stim_onset=use_stim_onset)
        output_files, stacks, n_artifact = data['output_files'], data['stacks'], data['n_artifact']

        if not overwrite_files and all(os.path.isfile(f) for f in output_files):
            return False

        ch0_stacks = stacks[data['roi_channel']]
        roi_mask = _predict_roi_mask(autorois_model, data, ref_idx=0, use_all_stacks=use_all_stacks)

        roi_mask = relabel_mask(np.asarray(roi_mask), connectivity=connectivity)
        if clean:
            roi_mask = clean_rois(roi_mask, n_artifact, min_size=min_size, connectivity=connectivity, verbose=False)
        roi_mask = _remove_vessel_rois(roi_mask, data, 0, vessel_threshold, max_vessel_overlap, connectivity)

        # Shifts are defined as in the ROI canvas: the mask is in the frame of the first presentation, and the
        # mask file of a presentation is the mask shifted by the inverse of the presentation's shift.
        roi_masks = [roi_mask.copy()]
        if auto_shift and len(ch0_stacks) > 1:
            ref_corr_map = compute_corr_map(ch0_stacks[0])
        for stack in ch0_stacks[1:]:
            shift = (0, 0)
            if auto_shift:
                match_indexes = compute_corr_map_match_indexes(
                    compute_corr_map(stack), ref_corr_map, shift_max=max_shift)
                shift_x, shift_y = extract_best_shift(match_indexes)
                shift = (-shift_x, -shift_y)
            roi_masks.append(shift_array(roi_mask, shift=(-shift[0], -shift[1]), cval=0))

        self._save_roi_mask_files(output_files, roi_masks)
        return True

    def _save_roi_mask_files(self, output_files: list, roi_masks: list) -> None:
        """Save one mask per output file. The root directory has to exist, the sub-directory is created."""
        for output_file in output_files:
            f_root, f_dir = os.path.split(os.path.dirname(output_file))
            if not os.path.isdir(f_root):
                raise FileNotFoundError(f'Could not find directory {f_root}')
            os.makedirs(os.path.join(f_root, f_dir), exist_ok=True)

        for output_file, roi_mask in zip(output_files, roi_masks):
            save_roi_mask_file(output_file, roi_mask, self._get_roi_file_format())

    def correct_roi_masks(
            self,
            model: str | None = None,
            kind: str = 'default_rgc',
            autorois_models: dict | None = None,
            restrictions: dict | None = None,
            roi_mask_dir: str = 'CorrectedROIs',
            old_prefix: str | None = None,
            new_prefix: str | None = None,
            roi_channel: str | None = None,
            use_stim_onset: bool = True,
            clean: bool = True,
            min_size: int = 3,
            connectivity: int = 2,
            vessel_threshold: float | None = 50.,
            max_vessel_overlap: float = 0.2,
            vessel_filter_existing: bool = False,
            add_missing: bool = True,
            iou_threshold: float = 0.1,
            shrink: bool = False,
            shrink_kws: dict | None = None,
            n_artifact: int | None = None,
            overwrite_files: bool = False,
            deploy: bool = False,
            roi_table: dj.Table | None = None,
            suppress_errors: bool = False,
            verbose: bool = True,
    ) -> dict:
        """Propose corrections for existing ROI masks and save them to files (no GUI).

        For each field with a ROI mask in the database, the mask is cleaned (light artifact rows, disconnected ROIs
        and tiny ROIs are removed) and ROIs that an AutoROIs model finds but that are missing in the mask (e.g. cells
        without activity that were not marked, 'dead cells') are added. Added ROIs never overlap or touch existing
        ones. If the mask changed, the corrected mask is saved to one file per presentation in ``roi_mask_dir``,
        shifted like the stored presentation masks, exactly as 'Save all to file' in the ROI canvas does.
        Unchanged masks are not saved.

        By default only files are written and the database is not touched. Use ``plot_roi_mask_correction`` to
        review a correction, and ``deploy_roi_mask`` (or ``deploy=True``) to replace the database entry.

        Args:
            model: Name of the AutoROIs model, see ``create_missing_roi_masks``. Only used if ``add_missing``.
            kind: Cell type variant of the model, see ``create_missing_roi_masks``.
            autorois_models: Already loaded models (dict name -> model). If None, only ``model`` is loaded and
                released again at the end.
            restrictions: Restrictions for the fields, e.g. ``dict(experimenter='Ran')``.
            roi_mask_dir: Sub-directory the corrected mask files are written to. Use a directory different from the
                one the current masks were loaded from, to keep the original files.
            old_prefix: Path prefix to replace when resolving file paths.
            new_prefix: Replacement path prefix.
            roi_channel: Name of the channel the model is applied to. Default: the ROI channel of the user info.
            use_stim_onset: If True, only use the recording after the stimulus onset (first trigger).
            clean: If True, clean the existing mask, see ``clean_rois``.
            min_size: Minimum ROI size in pixels, for cleaning and for added ROIs.
            connectivity: 1 (only NSEW neighbours) or 2 (also diagonal neighbours) connect pixels of a ROI.
            vessel_threshold: If not None, ROIs on blood vessels are not added, see ``create_missing_roi_masks``.
            max_vessel_overlap: A ROI is considered on a vessel if more than this fraction of its pixels are on vessels.
            vessel_filter_existing: If True, ROIs of the existing mask that are on vessels are removed too. By
                default only new ROIs are filtered, since the existing mask may be curated.
            add_missing: If True, add ROIs that the model finds but that are missing in the mask.
            iou_threshold: A ROI of the model counts as missing if its intersection over union with every existing
                ROI is below this value. Lower values add fewer ROIs.
            shrink: If True, remove border pixels that do not fit the activity of their ROI ('shrink to fit', e.g.
                an ROI drawn one pixel too large into a neighbouring cell or a dark gap), on the recording of the
                presentation of the field mask, see ``mask_utils.shrink_rois_to_fit``. Strict: most ROIs and most
                fields stay unchanged.
            shrink_kws: Parameters of ``shrink_rois_to_fit``.
            n_artifact: Number of rows at the top of the field that are treated as light artifact, for all fields.
                If None, the value of the setup is used (``npixartifact`` of the presentation). Use this if the
                artifact is larger than expected; ``plot_light_artifact`` helps to choose the value.
            overwrite_files: If False, fields for which the corrected files exist are skipped. If True, they are
                overwritten.
            deploy: If True, replace the database entry of every corrected field with the corrected masks right
                away (``deploy_roi_mask``). **This deletes everything downstream of the ROI mask of that field**
                and only repopulates ``roi_table``. Requires ``roi_table``.
            roi_table: ROI table (``RoiTemplate``), populated for the deployed fields.
            suppress_errors: If True, continue with the next field if a field fails and report the error.
            verbose: If True, print progress.

        Returns:
            A dict with the lists ``corrected`` (a dict per field with the field key and the number of ROIs
            ``n_before``, ``n_after``, ``n_added`` and ``n_shrunk``), ``unchanged``, ``skipped`` (dicts with field key
            and reason),
            ``deployed`` (field keys) and ``errors`` (tuples of field key and exception).
        """
        if connectivity not in (1, 2):
            raise ValueError(f'connectivity={connectivity} must be 1 or 2')
        if deploy and roi_table is None:
            raise ValueError('deploy=True requires roi_table, which is repopulated for the deployed fields.')

        restrictions = dict() if restrictions is None else restrictions
        field_keys = (self & restrictions).proj().to_dicts()
        result = dict(corrected=[], unchanged=[], skipped=[], deployed=[], errors=[])
        if len(field_keys) == 0:
            if verbose:
                print('No ROI masks found.')
            return result

        autorois_model, loaded_here = None, False
        if add_missing:
            model, autorois_model, loaded_here = _get_autorois_model(model, kind, autorois_models)
        del autorois_models

        try:
            for i, field_key in enumerate(field_keys, start=1):
                if verbose:
                    print(f'[{i}/{len(field_keys)}] {field_key}')
                try:
                    info = self._correct_field_roi_mask(
                        field_key, autorois_model, roi_channel=roi_channel, roi_mask_dir=roi_mask_dir,
                        old_prefix=old_prefix, new_prefix=new_prefix, use_stim_onset=use_stim_onset, clean=clean,
                        min_size=min_size, connectivity=connectivity, iou_threshold=iou_threshold,
                        overwrite_files=overwrite_files, vessel_threshold=vessel_threshold,
                        max_vessel_overlap=max_vessel_overlap, vessel_filter_existing=vessel_filter_existing,
                        shrink=shrink, shrink_kws=shrink_kws, n_artifact_override=n_artifact)
                    status = info.pop('status')
                    result[status].append(info if status in ('corrected', 'skipped') else info['field_key'])
                    if verbose:
                        print(f'  {status}: ' + ', '.join(f'{k}={v}' for k, v in info.items() if k != 'field_key'))

                    if deploy and status == 'corrected':
                        self.deploy_roi_mask(
                            info['field_key'], roi_table=roi_table, roi_mask_dir=roi_mask_dir,
                            old_prefix=old_prefix, new_prefix=new_prefix, verbose=False)
                        result['deployed'].append(info['field_key'])
                except Exception as e:
                    if not suppress_errors:
                        raise
                    warnings.warn(f'Error for field_key={field_key}:\n{e}')
                    result['errors'].append((field_key, e))
        finally:
            del autorois_model
            if loaded_here:
                _free_memory()

        if verbose:
            print(f"Corrected {len(result['corrected'])} masks ({len(result['deployed'])} deployed), "
                  f"{len(result['unchanged'])} unchanged, {len(result['skipped'])} skipped, "
                  f"{len(result['errors'])} errors.")
        return result

    def _expected_corrected_files(self, field_key: dict, main_row: dict, roi_mask_dir: str,
                                  old_prefix: str | None, new_prefix: str | None) -> list:
        """Corrected mask files of a field, from the database only (without loading any stacks)."""
        if self._presentation_masks:
            pres_files = [local_file_path(f, self.presentation_table._filepath_store)
                          for f in (self.presentation_table & field_key).to_arrays('pres_data_file')]
            return [to_roi_mask_file(
                f, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix,
                file_format=self._get_roi_file_format()) for f in pres_files]
        ref_pres_key = {k: main_row[k] for k in self.presentation_table.primary_key}
        return self._field_roi_mask_files([ref_pres_key], roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)

    def _correct_field_roi_mask(
            self, field_key: dict, autorois_model, roi_channel, roi_mask_dir, old_prefix, new_prefix,
            use_stim_onset, clean, min_size, connectivity, iou_threshold, overwrite_files,
            vessel_threshold=None, max_vessel_overlap=0.2, vessel_filter_existing=False, shrink=False,
            shrink_kws=None, n_artifact_override=None,
    ) -> dict:
        """Correct the mask of one field and save it. Returns a dict with ``status`` and information."""
        field_key = (self.field_table & field_key).proj().fetch1()

        if self._presentation_masks and len(self.RoiMaskPresentation & field_key & "as_field_mask='different'") > 0:
            return dict(status='skipped', field_key=field_key, reason="has presentations with 'different' masks")

        main_row = (self & field_key).fetch1()

        # Cheap early skip, before loading any stacks: if all corrected files exist there is nothing to do.
        if not overwrite_files:
            early_files = self._expected_corrected_files(
                field_key, main_row, roi_mask_dir, old_prefix, new_prefix)
            if len(early_files) > 0 and all(os.path.isfile(f) for f in early_files):
                return dict(status='skipped', field_key=field_key, reason=f'corrected files exist in {roi_mask_dir}')

        data = self._load_field_stacks(
            field_key, roi_channel=roi_channel, roi_mask_dir=roi_mask_dir,
            old_prefix=old_prefix, new_prefix=new_prefix, use_stim_onset=use_stim_onset)
        if n_artifact_override is not None:
            data['n_artifact'] = int(n_artifact_override)
        pres_keys, output_files, n_artifact = data['pres_keys'], data['output_files'], data['n_artifact']

        # The field mask is the mask of the presentation stored with it, which is the reference for the shifts.
        ref_idxs = [i for i, pres_key in enumerate(pres_keys)
                    if all(pres_key[k] == main_row[k] for k in self.presentation_table.primary_key)]
        if len(ref_idxs) != 1:
            raise ValueError(f'Could not find the presentation of the ROI mask of field_key={field_key}')
        ref_idx = ref_idxs[0]
        shifts = self.init_shifts(field_key, pres_keys) or [(0, 0)] * len(pres_keys)

        roi_mask_old = to_python_format(load_array(main_row['roi_mask']))
        roi_mask = clean_rois(roi_mask_old, n_artifact, min_size=min_size, connectivity=connectivity,
                              verbose=False) if clean else roi_mask_old.copy()
        if vessel_filter_existing:
            roi_mask = _remove_vessel_rois(roi_mask, data, ref_idx, vessel_threshold, max_vessel_overlap, connectivity)

        n_added = 0
        if autorois_model is not None:
            roi_mask_pred = _predict_roi_mask(autorois_model, data, ref_idx=ref_idx)
            roi_mask_pred = _remove_vessel_rois(
                roi_mask_pred, data, ref_idx, vessel_threshold, max_vessel_overlap, connectivity)
            if np.any(roi_mask > 0):
                rois_to_add = generate_roi_suggestions(
                    mask_pred=roi_mask_pred, mask_true=roi_mask, n_artifact=n_artifact, threshold=iou_threshold)
            else:
                rois_to_add = clean_rois(roi_mask_pred, n_artifact, min_size=min_size, verbose=False)

            if np.any(rois_to_add > 0):
                roi_mask, rois_added = add_rois(rois_to_add, roi_mask, connectivity=connectivity)
                n_added = len(np.unique(rois_added[rois_added > 0]))

        n_shrunk = 0
        if shrink:
            roi_mask, shrink_info = shrink_rois_to_fit(
                roi_mask, data['stacks'][data['roi_channel']][ref_idx], n_artifact=n_artifact, **(shrink_kws or dict()))
            n_shrunk = sum(d['status'] == 'shrunk' for d in shrink_info.values())

        n_before, n_after = len(np.unique(roi_mask_old[roi_mask_old > 0])), len(np.unique(roi_mask[roi_mask > 0]))
        info = dict(field_key=field_key, n_before=n_before, n_after=n_after, n_added=n_added, n_shrunk=n_shrunk)

        if np.array_equal(roi_mask_old > 0, roi_mask > 0) and n_before == n_after:
            return dict(info, status='unchanged')

        if not self._presentation_masks:  # Motion mode: one field mask file of the presentation of the mask
            output_files = self._field_roi_mask_files(
                [pres_keys[ref_idx]], roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)
            shifts = [(0, 0)]

        if not overwrite_files and all(os.path.isfile(f) for f in output_files):
            return dict(status='skipped', field_key=field_key, reason=f'corrected files exist in {roi_mask_dir}')

        self._save_roi_mask_files(output_files, [
            shift_array(roi_mask, shift=(-shift[0], -shift[1]), cval=0) for shift in shifts])
        return dict(info, status='corrected')

    def _field_stack(self, field_key: dict, pres_key: dict, data_name: str):
        """Stack of a presentation after the stimulus onset (x, y, t). None if no file is found."""
        from_raw_data = (self.raw_params_table & field_key).fetch1("from_raw_data")
        pres_path = (self.presentation_table & pres_key).fetch1("pres_data_file")
        store = self.presentation_table._filepath_store
        pres_store_path = relative_store_path(local_file_path(pres_path, store), store)
        info_df = self.load_field_file_info_df(field_key)
        stack_files = [f for f in info_df[info_df['kind'] == 'response']['filepath'].values
                       if relative_store_path(f, store) == pres_store_path]
        if len(stack_files) != 1:
            warnings.warn(f'Could not find the recording file of the field mask, no correlation map: {stack_files}')
            return None
        ch_stacks, _ = read_utils.load_stacks(stack_files[0], from_raw_data=from_raw_data, ch_names=(data_name,))
        triggertimes = (self.presentation_table & pres_key).fetch1('triggertimes')
        scan_frequency = (self.presentation_table.ScanInfo() & pres_key).fetch1('scan_frequency')
        onset_idx = int(np.floor(triggertimes[0] * scan_frequency)) if len(triggertimes) > 0 else 0
        return ch_stacks[data_name][:, :, onset_idx:]

    def _field_corr_map(self, field_key: dict, pres_key: dict, data_name: str):
        """Correlation map of a presentation, from the response after the stimulus onset. None if no file is found."""
        stack = self._field_stack(field_key, pres_key, data_name)
        return None if stack is None else compute_corr_map(stack)

    def plot_light_artifact(
            self,
            field_key: dict | None = None,
            restrictions: dict | None = None,
            n_max: int = 15,
            n_artifact: int | None = None,
            gamma: float = 0.5,
            max_fields: int = 20,
    ) -> None:
        """Plot row profiles to choose ``n_artifact``, the number of light artifact rows at the top of a field.

        The light artifact is at the top rows of a field (first image axis). Its rows are driven by the stimulus light
        and are far stronger and more correlated than the tissue below: whole rows move together. It shows up as a
        step at the artifact edge in three row profiles (mean over the other image axis), computed on the response
        after the stimulus onset: the mean of the stack average (relative to the median of the rows below ``n_max``),
        the temporal standard deviation (relative to the same median, log scale; usually the clearest step), and the
        correlation over time between a row and the next row (close to 1 inside the artifact). Pick ``n_artifact`` as
        the first row after the step; a partial row has an intermediate value.

        Panels: the stack average of the first field with the edge lines, and the three profiles of all fields
        (thin) with their median (thick). Only the field mask presentation of each field is used.

        Args:
            field_key: Primary key identifying the field. If None, the first available key is used. Ignored if
                ``restrictions`` is given.
            restrictions: Restrictions for the fields, e.g. ``dict(experimenter='Ran')``, to overlay the profiles of
                several fields and choose one value for all of them.
            n_max: Number of rows from the top that are shown in the profiles.
            n_artifact: Candidate value, drawn as red line (rows 0 to ``n_artifact - 1`` would be removed). The
                gray dashed line is the setup value ``npixartifact``.
            gamma: Gamma correction value for the stack average.
            max_fields: Maximum number of fields if ``restrictions`` is given.
        """
        from matplotlib import pyplot as plt

        if restrictions is None:
            field_keys = [get_primary_key(table=self.proj(), key=field_key)]
        else:
            field_keys = (self & restrictions).proj().to_dicts()[:max_fields]
            if len(field_keys) == 0:
                raise ValueError('No ROI masks found.')

        rows = np.arange(n_max)
        mean_profiles, std_profiles, corr_profiles, npixs, first = [], [], [], set(), None

        def _pad(profile):
            return np.pad(np.asarray(profile, dtype=float)[:n_max], (0, max(n_max - len(profile), 0)),
                          constant_values=np.nan)

        for fk in field_keys:
            main_row = (self & fk).fetch1()
            pres_key = {k: main_row[k] for k in self.presentation_table.primary_key}
            data_name, _ = self.userinfo_table().get_stack_names(fk)
            average = load_array((self.presentation_table.StackAverages & pres_key & dict(
                ch_name=data_name)).fetch1('ch_average')).astype(float)
            npixs.add(int((self.presentation_table & pres_key).fetch1('npixartifact')))
            if first is None:
                first = (fk, average, (self.field_table & fk).fetch1('scan_type'))

            stack = self._field_stack(fk, pres_key, data_name)
            if stack is None:
                mean_profiles.append(_pad([]))
                std_profiles.append(_pad([]))
                corr_profiles.append(_pad([]))
                continue
            stack = stack.astype(float)
            profiles = (np.nanmean(stack.mean(axis=2), axis=1), np.nanmean(stack.std(axis=2), axis=1))
            refs = [np.nanmedian(p[n_max:]) if len(p) > n_max else np.nanmedian(p) for p in profiles]
            mean_profiles.append(_pad(profiles[0] / refs[0] if refs[0] else profiles[0]))
            std_profiles.append(_pad(profiles[1] / refs[1] if refs[1] else profiles[1]))
            centered = stack - stack.mean(axis=2, keepdims=True)
            n_pairs = min(n_max, centered.shape[0] - 1)
            corr_profiles.append(_pad(np.nanmean(
                correlation_on_centered(centered[1:n_pairs + 1], centered[:n_pairs], axis=2), axis=1)))

        def _edges(ax, to_x=lambda n: n - 0.5):
            for n in sorted(npixs):
                ax.axvline(to_x(n), color='gray', ls='--', lw=1)
            if n_artifact is not None:
                ax.axvline(to_x(n_artifact), color='red', lw=1.5)

        fk, average, scan_type = first
        norm = (average - np.nanmin(average)) / max(np.nanmax(average) - np.nanmin(average), 1e-12)
        extent, xlabel, ylabel = field_extent_and_labels(average.shape, scan_type)
        fig, axs = plt.subplots(1, 4, figsize=(22, 4), gridspec_kw=dict(width_ratios=[1, 1.2, 1.2, 1.2]))
        fig.suptitle(str(fk) if len(field_keys) == 1 else f'{len(field_keys)} fields, image of {fk}')
        axs[0].imshow((norm ** gamma).T, cmap='gray', origin='lower', extent=extent)
        _edges(axs[0], to_x=lambda n: extent[0] - n)  # the first axis is flipped in the image
        axs[0].set(xlabel=xlabel, ylabel=ylabel, title='Stack average')

        for ax, profiles, title, x in (
                (axs[1], mean_profiles, 'Mean (rel. to median)', rows),
                (axs[2], std_profiles, 'Temporal std (rel. to median)', rows),
                (axs[3], corr_profiles, 'Correlation of row i with row i+1', rows + 0.5)):
            for profile in profiles:
                ax.plot(x, profile, '.-', lw=0.8, alpha=0.4 if len(profiles) > 1 else 1., color='C0')
            if len(profiles) > 1:
                ax.plot(x, np.nanmedian(np.array(profiles), axis=0), 'k.-', lw=2, label='median')
                ax.legend()
            _edges(ax)
            ax.set(xlabel='row (top = 0)', title=title)
            ax.set_xticks(rows)
        axs[2].set_yscale('log')
        plt.tight_layout()
        plt.show()

    def plot_roi_mask_correction(
            self,
            field_key: dict | None = None,
            roi_mask_dir: str = 'CorrectedROIs',
            old_prefix: str | None = None,
            new_prefix: str | None = None,
            gamma: float = 0.5,
            n_artifact: int | None = None,
    ) -> None:
        """Plot the ROI mask in the database against its correction from ``correct_roi_masks``.

        Shows the current mask, the corrected mask, the pixels that were added (green) or removed (red), the
        stack average with these differences, the stack average with the outlines of the ROIs that changed, and the
        correlation map (response after the stimulus onset) with the outlines of all corrected ROIs.
        Green: corrected ROIs that are new (do not overlap any current ROI). Blue: corrected ROIs that differ from
        the current mask in any pixel. Red: current ROIs that were removed, i.e. that do not overlap any corrected ROI.

        Args:
            field_key: Primary key identifying the field. If None, the first available key is used.
            roi_mask_dir: Sub-directory the corrected mask files were written to.
            old_prefix: Path prefix to replace when resolving file paths.
            new_prefix: Replacement path prefix.
            gamma: Gamma correction value for the stack average.
            n_artifact: Number of artifact rows used in ``correct_roi_masks``, drawn as red line (the edge of the
                light artifact). If None, the setup value ``npixartifact`` of the presentation is drawn.
        """
        from matplotlib import pyplot as plt
        from matplotlib.colors import ListedColormap

        field_key = get_primary_key(table=self.proj(), key=field_key)
        main_row = (self & field_key).fetch1()
        pres_key = {k: main_row[k] for k in self.presentation_table.primary_key}

        input_file = local_file_path(
            (self.presentation_table & pres_key).fetch1("pres_data_file"), self.presentation_table._filepath_store)
        if self._presentation_masks:
            roi_mask_file = to_roi_mask_file(
                input_file, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix,
                file_format=self._get_roi_file_format())
        else:  # Motion mode: field mask file
            roi_mask_file = to_field_roi_mask_file(
                input_file, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)
        if not os.path.isfile(roi_mask_file):
            raise FileNotFoundError(f'No corrected ROI mask file {roi_mask_file}')

        roi_mask_old = to_python_format(load_array(main_row['roi_mask']))
        roi_mask_new = load_roi_mask_file(roi_mask_file).astype(int)

        scan_type = (self.field_table & field_key).fetch1('scan_type')
        data_name, _ = self.userinfo_table().get_stack_names(field_key)
        ch_average = load_array(
            (self.presentation_table.StackAverages & pres_key & dict(ch_name=data_name)).fetch1('ch_average'))

        corr_map = self._field_corr_map(field_key, pres_key, data_name)

        diff = (roi_mask_new > 0).astype(float) - (roi_mask_old > 0).astype(float)
        diff[diff == 0] = np.nan
        diff_cmap = ListedColormap(['red', 'lime'])  # -1 (removed) -> red, +1 (added) -> green
        average = ch_average.astype(float)
        average = (average - np.nanmin(average)) / max(np.nanmax(average) - np.nanmin(average), 1e-12)

        # Same layout as `plot_field`, which is used by `plot1`
        print(field_key)
        
        extent, xlabel, ylabel = field_extent_and_labels(roi_mask_old.shape, scan_type)
        fig, axs = plt.subplots(1, 6, figsize=(22, 4), sharex='all', sharey='all')
        fig.suptitle(str(field_key))
        for ax in axs:
            ax.set(xlabel=xlabel)
        axs[0].set(ylabel=ylabel)

        for ax, img, title in zip(axs[:2], (roi_mask_old, roi_mask_new), ('Current', 'Corrected')):
            ax.imshow(np.where(img > 0, img, np.nan).T.astype(float), cmap='jet', origin='lower', extent=extent)
            ax.set(title=f'{title} ({len(np.unique(img[img > 0]))} ROIs)')
        axs[2].imshow(diff.T, vmin=-1, vmax=1, cmap=diff_cmap, origin='lower', extent=extent)
        axs[2].set(title='add (G) / rm (R) pixels')
        axs[3].imshow((average ** gamma).T, cmap='gray', origin='lower', extent=extent)
        axs[3].imshow(diff.T, vmin=-1, vmax=1, cmap=diff_cmap, alpha=0.5, origin='lower', extent=extent)
        axs[3].set(title='Stack average + difference')

        def _differs(i):
            old_ids = np.unique(roi_mask_old[(roi_mask_new == i) & (roi_mask_old > 0)])
            return len(old_ids) != 1 or not np.array_equal(roi_mask_old == old_ids[0], roi_mask_new == i)

        new_ids, changed_ids = [], []
        for i in np.unique(roi_mask_new[roi_mask_new > 0]):
            if not np.any(roi_mask_old[roi_mask_new == i] > 0):
                new_ids.append(i)
            elif _differs(i):
                changed_ids.append(i)
        removed_ids = [i for i in np.unique(roi_mask_old[roi_mask_old > 0])
                       if not np.any(roi_mask_new[roi_mask_old == i] > 0)]

        def _outlines(ax):
            for mask, ids, color in ((roi_mask_new, new_ids, 'lime'), (roi_mask_new, changed_ids, 'blue'),
                                     (roi_mask_old, removed_ids, 'red')):
                for roi_id in ids:
                    ax.contour((mask == roi_id).T.astype(float), levels=[0.5], colors=color, linewidths=1.5,
                               origin='lower', extent=extent)

        axs[4].imshow((average ** gamma).T, cmap='gray', origin='lower', extent=extent)
        _outlines(axs[4])
        axs[4].set(title=f'add ({len(new_ids)}, G) / changed ({len(changed_ids)}, B) / '
                         f'rm ({len(removed_ids)}, R)')

        if corr_map is not None:
            axs[5].imshow(corr_map.T, cmap='gray', origin='lower', extent=extent)
        for roi_id in np.unique(roi_mask_new[roi_mask_new > 0]):
            axs[5].contour((roi_mask_new == roi_id).T.astype(float), levels=[0.5], colors='yellow', linewidths=1,
                           origin='lower', extent=extent)
        axs[5].set(title='Correlation + corr. ROIs')
        if n_artifact is None:
            n_artifact = int((self.presentation_table & pres_key).fetch1('npixartifact'))
        for ax in axs:
            if n_artifact > 0:
                ax.axvline(extent[0] - n_artifact, color='red', ls='--', lw=1)  # edge of the light artifact
        plt.tight_layout()
        plt.show()

    def export_field_roi_masks(
            self,
            restrictions: dict | None = None,
            roi_mask_dir: str = 'PreparedROIs',
            old_prefix: str | None = None,
            new_prefix: str | None = None,
            overwrite_files: bool = False,
            update_database: bool = True,
            suppress_errors: bool = False,
            verbose: bool = True,
    ) -> dict:
        """Motion mode: save the prepared field ROI masks (presentation chosen by fit, aligned) to field mask files.

        In motion mode the field mask in the database can differ from its file: it is aligned to the mean image of
        its presentation (``align_dx``, ``align_dy``), while the file keeps the mask as drawn, and with copied masks
        the file does not tell which presentation it belongs to. This writes the field mask of the database to one
        field mask file (``<stem>_FieldROIs.npy`` of its presentation, see ``to_field_roi_mask_file``) in
        ``roi_mask_dir``. The original files are kept, so use a directory other than the one the masks were loaded
        from. Presentation masks only come from ``MotionRoiMask``; no per-presentation copies are written.

        With ``update_database``, the database entry then references the new file (``roi_mask_file``,
        ``roi_mask_source``) and ``align_dx = align_dy = 0``. The mask itself is unchanged, so nothing downstream is
        deleted. Review or edit the files (``draw_roi_mask(roi_mask_dir=...)``) and load edited files with
        ``deploy_roi_mask(roi_mask_dir=...)``, which uses the presentation named by the file.

        Args:
            restrictions: Restrictions for the fields, e.g. ``dict(experimenter='Ran')``.
            roi_mask_dir: Sub-directory (next to the recording files) the mask files are written to.
            old_prefix: Path prefix to replace when resolving file paths.
            new_prefix: Replacement path prefix.
            overwrite_files: If False, fields whose file exists with a different mask are skipped. A file with the
                same mask counts as exported. Fields with a field mask file of another presentation are always
                skipped (delete it first).
            update_database: If True, let the database entry reference the new files (see above).
            suppress_errors: If True, continue with the next field if a field fails and report the error.
            verbose: If True, print progress.

        Returns:
            A dict with the lists ``exported`` (field keys whose file was written), ``linked`` (field keys whose
            database entry now references the file), ``skipped`` (dicts with field key and reason) and ``errors``
            (tuples of field key and exception).
        """
        if self._presentation_masks:
            raise NotImplementedError(
                'export_field_roi_masks is for motion mode (no presentation masks). With presentation masks, the '
                'database masks are the files they were loaded from; use correct_roi_masks or draw_roi_mask.')

        restrictions = dict() if restrictions is None else restrictions
        field_keys = (self & restrictions).proj().to_dicts()
        result = dict(exported=[], linked=[], skipped=[], errors=[])
        for i, field_key in enumerate(field_keys, start=1):
            try:
                info = self._export_field_roi_mask(
                    field_key, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix,
                    overwrite_files=overwrite_files, update_database=update_database)
            except Exception as e:
                if not suppress_errors:
                    raise
                warnings.warn(f'Error for field_key={field_key}:\n{e}')
                result['errors'].append((field_key, e))
                continue
            if info['status'] == 'skipped':
                result['skipped'].append(dict(field_key=field_key, reason=info['reason']))
            elif info['status'] == 'exported':
                result['exported'].append(field_key)
            if info['linked']:
                result['linked'].append(field_key)
            if verbose:
                print(f"[{i}/{len(field_keys)}] {info['status']}{' + linked' if info['linked'] else ''}: {field_key}"
                      + (f" ({info['reason']})" if 'reason' in info else ''))

        if verbose:
            print(f"Exported {len(result['exported'])} field masks to {roi_mask_dir} ({len(result['linked'])} linked "
                  f"in the database), {len(result['skipped'])} skipped, {len(result['errors'])} errors.")
        return result

    def _export_field_roi_mask(self, field_key: dict, roi_mask_dir: str, old_prefix: str | None,
                               new_prefix: str | None, overwrite_files: bool, update_database: bool) -> dict:
        """Write the field mask of one field to its field mask file and optionally link it. Returns a dict with
        ``status`` ('exported', 'exists' or 'skipped') and ``linked``."""
        row = (self & field_key).fetch1()
        pres_primary_key = self.presentation_table.primary_key
        ref_pres_key = {k: row[k] for k in pres_primary_key}
        pres_keys = (self.presentation_table & field_key).proj().to_dicts()
        ref_idxs = [i for i, pk in enumerate(pres_keys) if all(pk[k] == ref_pres_key[k] for k in pres_primary_key)]
        if len(ref_idxs) != 1:
            raise ValueError(f'Could not find the presentation of the ROI mask of field_key={field_key}')
        field_files = self._field_roi_mask_files(pres_keys, roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)
        ref_file = field_files[ref_idxs[0]]

        others = [f for f in field_files if f != ref_file and os.path.isfile(f)]
        if len(others) > 0:
            return dict(status='skipped', linked=False,
                        reason=f'field mask file of another presentation exists: {others}')

        roi_mask = to_python_format(load_array(row['roi_mask']))
        status = 'exported'
        if os.path.isfile(ref_file):
            if np.array_equal(load_roi_mask_file(ref_file).astype(int), roi_mask):
                status = 'exists'
            elif not overwrite_files:
                return dict(status='skipped', linked=False, reason=f'file with a different mask exists: {ref_file}')
        if status == 'exported':
            self._save_roi_mask_files([ref_file], [roi_mask])

        new_values = dict(roi_mask_file=relative_store_path(ref_file, self._filepath_store),
                          roi_mask_source=_roi_mask_source_from_path(ref_file))
        if self._selects_field_mask:
            new_values.update(align_dx=0, align_dy=0)
        if update_database:  # idempotent; the mask itself is unchanged, so nothing downstream is affected
            self.update1({**{k: row[k] for k in self.primary_key}, **new_values})
        return dict(status=status, linked=update_database)

    def rescan_filesystem(
            self,
            restrictions: dict | None = None,
            source = 'infer',  # 'infer' | 'igor' | 'autorois'
            verboselvl: int = 0,
            suppress_errors: bool = False,
            only_new_fields: bool = True,
            roi_mask_dir='ROIs',
            old_prefix=None,
            new_prefix=None,
            max_shift=None,
            auto_fill_pres_keys: bool = False,
            add_primary_keys=None
        ):
        """Scan filesystem for new ROI masks and add them to the database.
        :param restrictions: Restrictions for field_table
        :param source: Source of ROI masks to scan for. Can be 'infer' (to automatically infer from parameters), 'igor' (to only scan for Igor files) or 'autorois' (to only scan for AutoROIs files).
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
            keys_to_scan = ((self.key_source - self) & restrictions).proj().to_dicts()
        else:
            keys_to_scan = (self.key_source & restrictions).proj().to_dicts()

        if verboselvl > 0:
            print(f'Scanning {len(keys_to_scan)} keys for new ROI masks...')

        err_list = []

        for key in keys_to_scan:
            if verboselvl > 2:
                print('#' * 60)
                print('Scanning key:', key)
            try:
                self._add_field_roi_masks(
                    key, source=source, auto_fill_pres_keys=auto_fill_pres_keys,
                    roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix,
                    max_shift=max_shift, add_primary_keys=add_primary_keys, verboselvl=verboselvl)
            except Exception as e:
                if suppress_errors:
                    warnings.warn(f'Error for key={key}:\n{e}')
                    err_list.append((key, e))
                else:
                    raise e

        return err_list

    def deploy_roi_mask(
            self,
            field_key: dict,
            roi_table: dj.Table | None = None,
            roi_mask_dir: str = 'ROIs',
            old_prefix: str | None = None,
            new_prefix: str | None = None,
            max_shift: int | None = None,
            verbose: bool = True,
    ) -> None:
        """Replace the ROI mask of a field in the database by the mask files in ``roi_mask_dir``.

        The existing entry is deleted, which cascades to ALL downstream tables (ROI, traces, ...). The mask files
        of all presentations of the field (e.g. written by the ROI canvas, 'Save all to file') are then inserted as
        for ``rescan_filesystem(source='autorois')``. Delete and insert are one transaction, so the old entry is kept
        if anything fails. Afterwards only ``roi_table`` is repopulated for the field; further downstream tables
        have to be populated by the caller.

        Args:
            field_key: Primary key dict identifying the field.
            roi_table: ROI table (``RoiTemplate``) that is populated for the field afterwards. If None, nothing is
                populated.
            roi_mask_dir: Sub-directory name containing ROI mask files.
            old_prefix: Path prefix to replace when resolving file paths.
            new_prefix: Replacement path prefix.
            max_shift: Maximum allowed pixel shift between presentations. Defaults to the table-level value.
            verbose: If True, print a summary.

        In motion mode a field mask file (``<stem>_FieldROIs.npy``, written by the ROI canvas or
        ``export_field_roi_masks``) is used if there is one; then no presentation mask files are needed.

        Raises:
            FileNotFoundError: If a presentation of the field has no mask file. Nothing is deleted then.
        """
        field_key = (self.field_table & field_key).proj().fetch1()
        pres_keys = (self.presentation_table & field_key).proj().to_dicts()

        field_file = None
        if not self._presentation_masks:
            _, field_file = self._find_field_roi_mask_file(
                pres_keys, roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)

        missing_files = []
        for pres_key in (pres_keys if field_file is None else []):
            input_file = local_file_path(
                (self.presentation_table & pres_key).fetch1("pres_data_file"), self.presentation_table._filepath_store)
            roimask_file = to_roi_mask_file(
                input_file, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix,
                file_format=self._get_roi_file_format())
            if not os.path.isfile(roimask_file):
                missing_files.append(roimask_file)
        if len(missing_files) > 0:
            raise FileNotFoundError(f'Missing ROI mask files, nothing was changed:\n{missing_files}')

        with self.connection.transaction:
            (self & field_key).delete(prompt=False)
            self._add_field_roi_masks(
                field_key, source='autorois', roi_mask_dir=roi_mask_dir,
                old_prefix=old_prefix, new_prefix=new_prefix, max_shift=max_shift)
            n_new = len(self.RoiMaskPresentation & field_key) if self._presentation_masks else len(pres_keys)
            if len(self & field_key) != 1 or n_new != len(pres_keys):
                raise ValueError(f'Expected 1 ROI mask for {len(pres_keys)} presentations, '
                                 f'got {len(self & field_key)} for {n_new}; rolled back.')

        if roi_table is not None:
            roi_table.populate(field_key)

        if verbose:
            print(f'Deployed ROI mask for field_key=\n{field_key}\n'
                  f'({n_new} presentations; populated ROI table: {roi_table is not None})')

    def check_filesystem_consistency(
            self,
            restrictions: dict | None = None,
            verbose: bool = True,
    ) -> list:
        """Check whether database ROI masks are consistent with the files they were loaded from.

        Iterates over every presentation already stored in RoiMaskPresentation
        and compares its mask against the file recorded in ``roi_mask_file``
        (format given by ``roi_mask_source``). Returns a list of problem
        records so the caller can inspect or correct them.

        Args:
            restrictions: Optional DataJoint restriction applied to
                RoiMaskPresentation before iterating (e.g. a field key dict).
            verbose: If True, print progress and a summary at the end.

        Returns:
            A list of dicts, one per problematic presentation, each with:

            - ``'key'``: the presentation primary key dict
            - ``'issue'``: one of

              * ``'mismatch'`` — file exists but its content differs from DB
              * ``'missing_on_filesystem'`` — DB has a mask but the file
                is gone
              * ``'igor_missing'`` — DB has a mask but the H5 file contains
                no ROI mask

            - ``'filesystem_file'``: the file recorded in the database

        Example::

            problems = roi_mask_table.check_filesystem_consistency()
            for p in problems:
                print(p['issue'], p['filesystem_file'])
                print(p['key'])
        """
        if restrictions is None:
            restrictions = dict()

        problems = []
        # In motion mode (no presentation masks) only the field masks are stored, with the file they came from.
        mask_tab = self.RoiMaskPresentation if self._presentation_masks else self
        pres_keys = (mask_tab & restrictions).proj().to_dicts()

        if verbose:
            print(f'Checking filesystem consistency for {len(pres_keys)} presentation(s)...')

        for i, key in enumerate(pres_keys):
            roi_mask_file, roi_mask_source, database_roi_mask = (mask_tab & key).fetch1(
                'roi_mask_file', 'roi_mask_source', 'roi_mask')
            filesystem_file = local_file_path(roi_mask_file, self._filepath_store)
            database_roi_mask = load_array(database_roi_mask)
            # The field mask may be aligned to its presentation; the file has the mask as drawn
            align = (mask_tab & key).fetch1('align_dx', 'align_dy') \
                if mask_tab is self and self._selects_field_mask else (0, 0)

            if verbose:
                print(f'  [{i + 1}/{len(pres_keys)}] Checking key={key}')

            if roi_mask_source == 'igor_h5':
                filesystem_roi_mask = read_h5_utils.load_roi_mask(
                    filepath=filesystem_file, ignore_not_found=True)
                if filesystem_roi_mask is None:
                    if verbose:
                        print(f'    -> PROBLEM: igor_missing  ({filesystem_file})')
                    problems.append(
                        {'key': key, 'issue': 'igor_missing', 'filesystem_file': filesystem_file})
                    continue
            else:
                if not os.path.isfile(filesystem_file):
                    if verbose:
                        print(f'    -> PROBLEM: missing_on_filesystem  ({filesystem_file})')
                    problems.append(
                        {'key': key, 'issue': 'missing_on_filesystem', 'filesystem_file': filesystem_file})
                    continue
                filesystem_roi_mask = load_roi_mask_file(str(filesystem_file)).astype(np.int32)
                filesystem_roi_mask = to_igor_format(filesystem_roi_mask)

            if tuple(align) != (0, 0):
                filesystem_roi_mask = shift_array(filesystem_roi_mask, shift=(int(align[0]), int(align[1])), cval=1)

            if not np.all(filesystem_roi_mask == database_roi_mask):
                if verbose:
                    print(f'    -> PROBLEM: mismatch  ({filesystem_file})')
                problems.append(
                    {'key': key, 'issue': 'mismatch', 'filesystem_file': filesystem_file})
            else:
                if verbose:
                    print(f'    -> OK')

        if verbose:
            if problems:
                print(f'Done. Found {len(problems)} problem(s) out of {len(pres_keys)} presentation(s).')
            else:
                print(f'Done. All {len(pres_keys)} presentation(s) are consistent.')

        return problems

    def _add_field_roi_masks(
            self,
            field_key: dict,
            source: str = 'infer',
            auto_fill_pres_keys: bool = False,
            add_primary_keys: dict | None = None,
            roi_mask_dir: str = 'ROIs',
            old_prefix: str | None = None,
            new_prefix: str | None = None,
            max_shift: int | None = None,
            verboselvl: int = 0,
    ) -> None:
        """Find and insert ROI mask entries for all presentations of a field.

        Args:
            field_key: Primary key dict for the field.
            source: Source of the ROI masks ('infer', 'igor', or 'autorois').
            auto_fill_pres_keys: If True, presentations without a mask file
                are filled with the main field mask (zero shift).
            add_primary_keys: Optional extra primary key columns to inject into
                every inserted row.
            roi_mask_dir: Sub-directory name containing ROI mask files.
            old_prefix: Path prefix to replace when resolving file paths.
            new_prefix: Replacement path prefix.
            max_shift: Maximum allowed pixel shift between presentations.
                Defaults to the table-level ``_max_shift`` value.
            verboselvl: Verbosity level controlling diagnostic output.
        """
        pres_keys = (self.presentation_table & field_key).keys()

        if add_primary_keys:
            pres_keys = [{**pk, **add_primary_keys} for pk in pres_keys]
            field_key = {**field_key, **add_primary_keys}

        if verboselvl > 2:
            print('\nfield_key:', field_key, '\npres_keys:', pres_keys)

        # Filter to new presentation keys BEFORE loading masks, so that existing
        # entries are not needlessly read from disk and their filesystem-vs-database
        # consistency is not checked here (which would cause rescan to fail if any
        # pre-existing mask file was edited outside the database).
        if self._presentation_masks:
            new_pres_keys = [pk for pk in pres_keys
                             if len(self.RoiMaskPresentation & pk) == 0]
        else:
            # Motion mode: only the field mask is stored; presentation masks come from the motion tables.
            new_pres_keys = pres_keys if len(self & field_key) == 0 else []

        if len(new_pres_keys) == 0:
            if verboselvl > 1:
                print('Nothing new to add for field:', field_key)
            if verboselvl > 2:
                print('pres_keys:', [k for k in pres_keys])
            return

        if not self._presentation_masks:
            # Motion mode: a field mask file names its presentation, no choice by fit and no alignment needed.
            field_pres_key, field_file = self._find_field_roi_mask_file(
                pres_keys, roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)
            if field_file is not None:
                self._insert_field_roi_mask_file(field_key, field_pres_key, field_file, verboselvl=verboselvl)
                return

        if source == 'infer' and (self.raw_params_table & field_key).fetch1('igor_roi_masks') == 'either':
            source = self._resolve_field_source(
                field_key, pres_keys, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix)
            if verboselvl > 1:
                print(f"igor_roi_masks='either': using source={source!r} for field:", field_key)

        loaded = [self._load_presentation_roi_mask(
            key, roi_mask_dir, source=source, old_prefix=old_prefix, new_prefix=new_prefix)
                  for key in new_pres_keys]

        if not auto_fill_pres_keys:
            # Filter out keys without ROI mask
            data_pairs = [(pres_key, mask_info) for pres_key, mask_info in zip(new_pres_keys, loaded)
                          if mask_info[0] is not None]
        else:
            data_pairs = list(zip(new_pres_keys, loaded))

        if len(data_pairs) == 0:
            if verboselvl > 1:
                print('No ROI masks found for field:', field_key)
            return

        if len(self & field_key) == 0:
            # Find preferred file that should be used as main key.
            mask_alias, highres_alias = (self.userinfo_table & field_key).fetch1("mask_alias", "highres_alias")
            keys_masks_files = [
                (pres_key, mask_info, local_file_path(
                    (self.presentation_table & pres_key).fetch1('pres_data_file'),
                    self.presentation_table._filepath_store))
                for pres_key, mask_info in data_pairs
                if mask_info[0] is not None]

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

            main_idx, selection_info = int(sort_idxs[0]), dict()
            if self._selects_field_mask:
                main_idx, main_roi_mask_aligned, selection_info = self._select_field_mask_presentation(
                    field_key, keys_masks_files, preferred_idx=main_idx, verboselvl=verboselvl)

            main_pres_key, (main_roi_mask, main_file, main_source) = keys_masks_files[main_idx][:2]
            if self._selects_field_mask:
                main_roi_mask = main_roi_mask_aligned
        else:
            main_pres_key = (self.RoiMaskPresentation().proj() & (self & field_key)).fetch1()
            main_roi_mask = load_array((self.RoiMaskPresentation & main_pres_key).fetch1('roi_mask'))
            main_file, main_source = (self.RoiMaskPresentation & main_pres_key).fetch1(
                'roi_mask_file', 'roi_mask_source')

        roi_mask_pres_keys = []

        # Motion mode: no presentation masks, only the field mask is inserted below.
        for pres_key, (roi_mask, roi_mask_file, roi_mask_source) in (data_pairs if self._presentation_masks else []):
            if roi_mask is not None:
                as_field_mask, (shift_dx, shift_dy) = compare_roi_masks(
                    roi_mask, main_roi_mask, max_shift=self._max_shift if max_shift is None else max_shift)
                roi_mask_pres_keys.append(
                    {**pres_key, "roi_mask": roi_mask, "as_field_mask": as_field_mask,
                     "shift_dx": shift_dx, "shift_dy": shift_dy,
                     "roi_mask_file": roi_mask_file, "roi_mask_source": roi_mask_source})
            elif auto_fill_pres_keys:
                # No file of its own: reference the file of the main mask that was copied
                roi_mask_pres_keys.append(
                    {**pres_key, "roi_mask": main_roi_mask, "as_field_mask": 'same',
                     "shift_dx": 0, "shift_dy": 0,
                     "roi_mask_file": main_file, "roi_mask_source": main_source})
            else:
                raise ValueError(f'No ROI mask found for key={pres_key} but `auto_fill_pres_keys` is False')

        main_key = {**field_key, **main_pres_key, "roi_mask": main_roi_mask,
                    "roi_mask_file": main_file, "roi_mask_source": main_source, **selection_info}

        if add_primary_keys is not None:
            main_key = {**main_key, **add_primary_keys}
        self.insert1(main_key, skip_duplicates=True)

        for roi_mask_pres_key in roi_mask_pres_keys:
            if add_primary_keys is not None:
                roi_mask_pres_key = {**roi_mask_pres_key, **add_primary_keys}

            self.RoiMaskPresentation().insert1(roi_mask_pres_key, skip_duplicates=True)

    def _insert_field_roi_mask_file(self, field_key: dict, pres_key: dict, field_file: str,
                                    verboselvl: int = 0) -> None:
        """Motion mode: insert the field mask of a field mask file for the presentation it belongs to. ``mask_fit`` is
        stored for QC; the mask is not aligned (it was drawn or prepared on this presentation)."""
        roi_mask = to_igor_format(load_roi_mask_file(field_file).astype(np.int32))
        main_key = {**field_key, **{k: pres_key[k] for k in self.presentation_table.primary_key}, "roi_mask": roi_mask,
                    "roi_mask_file": relative_store_path(field_file, self._filepath_store),
                    "roi_mask_source": _roi_mask_source_from_path(field_file)}
        if self._selects_field_mask:
            _, roi_channel = self._get_stack_names(field_key)
            data_file = local_file_path((self.presentation_table & pres_key).fetch1('pres_data_file'),
                                        self.presentation_table._filepath_store)
            img, _, n_artifact = self._presentation_mean_image(pres_key, data_file, roi_channel)
            score = mask_fit_score(roi_mask, img, n_artifact=n_artifact)
            main_key.update(align_dx=0, align_dy=0, mask_fit=float(score) if np.isfinite(score) else None,
                            mask_selection='file')
        if verboselvl > 0:
            print(f'Adding field ROI mask from file={field_file} for field:', field_key)
        self.insert1(main_key, skip_duplicates=True)

    def _select_field_mask_presentation(
            self,
            field_key: dict,
            keys_masks_files: list,
            preferred_idx: int,
            verboselvl: int = 0,
    ) -> tuple[int, np.ndarray, dict]:
        """Choose the presentation of the field mask and align the mask to it, see ``select_field_roi_mask``.

        Args:
            field_key: Primary key of the field.
            keys_masks_files: List of (pres_key, (roi_mask, roi_mask_file, roi_mask_source), data_file).
            preferred_idx: Index of the presentation preferred by ``mask_alias``.
            verboselvl: Verbosity level.

        Returns:
            Index of the chosen presentation, its aligned mask and the selection columns of the table.
        """
        _, roi_channel = self._get_stack_names(field_key)
        images, has_stack_ave, n_artifact = [], [], 0
        for pres_key, _, data_file in keys_masks_files:
            img, stack_ave, n_artifact = self._presentation_mean_image(pres_key, data_file, roi_channel)
            images.append(img)
            has_stack_ave.append(stack_ave)

        res = select_field_roi_mask(
            [mask_info[0] for _, mask_info, _ in keys_masks_files], images, preferred_idx=preferred_idx,
            has_stack_ave=has_stack_ave, n_artifact=n_artifact, max_offset=self._field_mask_max_offset,
            max_copy_shift=self._max_shift)

        pres_names = [os.path.basename(f) for _, _, f in keys_masks_files]
        msg = (f"field mask of {field_key}: {pres_names[res['idx']]} (mask_alias: {pres_names[preferred_idx]}, "
               f"selection={res['selection']}, offset={res['offset']}, fit={res['score']:.3f})")
        if res['n_different'] > 0:
            warnings.warn(f"{res['n_different']} presentation(s) have a different (not copied) ROI mask, they are "
                          f"ignored for the {msg}")
        if res['selection'] == 'alias':
            warnings.warn(f"The ROI mask does not fit the mean image of any presentation, check the {msg}")
        elif res['offset'] != (0, 0) or verboselvl > 0:
            print(f"Aligned {msg}" if res['offset'] != (0, 0) else f"Selected {msg}")

        return res['idx'], res['roi_mask'], dict(
            align_dx=int(res['offset'][0]), align_dy=int(res['offset'][1]),
            mask_fit=float(res['score']) if np.isfinite(res['score']) else None, mask_selection=res['selection'])

    def _presentation_mean_image(self, pres_key: dict, data_file: str, roi_channel: str) -> tuple:
        """Mean image of the ROI channel after stimulus onset, whether the file has a ``Stack_ave`` (written by the
        ROI drawing of Igor) and the number of light artifact rows."""
        from_raw_data = (self.raw_params_table & pres_key).fetch1('from_raw_data')
        stacks, _ = read_utils.load_stacks(data_file, from_raw_data=from_raw_data, ch_names=(roi_channel,))
        stack = np.asarray(stacks[roi_channel], dtype=np.float32)

        triggertimes = load_array((self.presentation_table & pres_key).fetch1('triggertimes'))
        fs = (self.presentation_table.ScanInfo & pres_key).fetch1('scan_frequency')
        onset = int(np.floor(triggertimes[0] * fs)) if len(triggertimes) > 0 else 0
        if onset >= stack.shape[2] - 1:
            onset = 0
        img = stack[:, :, onset:].mean(axis=2)

        has_stack_ave = False
        if str(data_file).endswith('.h5') and os.path.isfile(data_file):
            import h5py
            with h5py.File(data_file, 'r') as f:
                has_stack_ave = 'Stack_ave' in f
        n_artifact = int((self.presentation_table & pres_key).fetch1('npixartifact'))
        return img, has_stack_ave, n_artifact

    def _resolve_field_source(
            self,
            field_key: dict,
            pres_keys: list,
            roi_mask_dir: str = 'ROIs',
            old_prefix: str | None = None,
            new_prefix: str | None = None,
    ) -> str:
        """Choose one mask source ('igor' or 'autorois') for a whole field, for ``igor_roi_masks='either'``.

        If the field already has a mask in the database, its source is kept. Otherwise a mask file
        (``roi_mask_dir``) of any presentation is preferred over an Igor mask.
        """
        if len(self & field_key) > 0:
            return 'igor' if (self & field_key).fetch1('roi_mask_source') == 'igor_h5' else 'autorois'

        for pres_key in pres_keys:
            input_file = local_file_path(
                (self.presentation_table & pres_key).fetch1("pres_data_file"), self.presentation_table._filepath_store)
            roimask_file = to_roi_mask_file(
                input_file, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix,
                file_format=self._get_roi_file_format())
            if os.path.isfile(roimask_file):
                return 'autorois'
        return 'igor'

    def _load_presentation_roi_mask(
            self,
            key: dict,
            roi_mask_dir: str = 'ROIs',
            source: str = 'infer',
            old_prefix: str | None = None,
            new_prefix: str | None = None,
    ) -> tuple[np.ndarray | None, str | None, str | None]:
        """Load the ROI mask for a single presentation, reconciling filesystem and database state.

        Args:
            key: Primary key dict for the presentation.
            roi_mask_dir: Sub-directory name containing ROI mask files.
            old_prefix: Path prefix to replace when resolving file paths.
            new_prefix: Replacement path prefix.

        Returns:
            A 3-tuple (roi_mask, roi_mask_file, roi_mask_source). roi_mask_file is the path of the file the mask
            was read from, relative to the filepath store, and roi_mask_source is its format ('igor_h5', 'numpy',
            'pickle' or 'other'). All three are None if no mask file was found.

        Raises:
            ValueError: If the filesystem mask differs from the database mask.
        """
        input_file = local_file_path(
            (self.presentation_table & key).fetch1("pres_data_file"),
            self.presentation_table._filepath_store)

        if source == 'infer':
            igor_roi_masks = (self.raw_params_table & key).fetch1('igor_roi_masks')
            if igor_roi_masks == 'either':
                raise ValueError("igor_roi_masks='either' has to be resolved per field, see `_resolve_field_source`.")
        elif source == 'igor':
            igor_roi_masks = 'yes'
        elif source == 'autorois':
            igor_roi_masks = 'no'
        else:
            raise ValueError(f"source={source!r} must be 'infer', 'igor' or 'autorois'")

        file_format = self._get_roi_file_format()
        roimask_file = to_roi_mask_file(
            input_file, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix,
            file_format=file_format)

        if igor_roi_masks == 'yes':
            # Igor masks are stored in the h5 file, also if the presentation was loaded from the raw file.
            userinfo = (self.userinfo_table & key).fetch1()
            igor_file = as_pre_filepath(
                input_file, raw_data_dir=userinfo['raw_data_dir'], pre_data_dir=userinfo['pre_data_dir'])
            filesystem_roi_mask = read_h5_utils.load_roi_mask(filepath=igor_file, ignore_not_found=True)
        else:
            if os.path.isfile(roimask_file):
                filesystem_roi_mask = load_roi_mask_file(roimask_file).astype(np.int32)
                filesystem_roi_mask = to_igor_format(filesystem_roi_mask)
            else:
                filesystem_roi_mask = None

        # Database entry of this presentation: its presentation mask, or in motion mode the field mask if the field
        # mask belongs to this presentation.
        db_tab = self.RoiMaskPresentation if self._presentation_masks else self
        if len((db_tab & key).proj()) > 0:
            database_roi_mask = load_array((db_tab & key).fetch1('roi_mask'))
            # The field mask may have been aligned to its presentation (the file has the mask as drawn)
            align = (db_tab & key).fetch1('align_dx', 'align_dy') \
                if db_tab is self and self._selects_field_mask else (0, 0)
            align = (int(align[0]), int(align[1]))

            if filesystem_roi_mask is None:
                if igor_roi_masks == 'yes':
                    # H5 files have no write-back API; writing a pickle would be silently
                    # ignored on the next scan (which still reads from H5), causing an
                    # infinite repeated warning.
                    warnings.warn(
                        f'ROI mask for key=\n{key}\nwas not found in the H5 file but exists in the database. '
                        f'Cannot write back to H5 format; database entry retained.'
                    )
                else:
                    warnings.warn(
                        f'ROI mask for key=\n{key}\nhas been deleted on the filesystem but not in the database.\n'
                        f'Saving ROI masks to file now: {roimask_file}'
                    )
                    save_roi_mask_file(roimask_file, to_python_format(database_roi_mask), file_format)

            elif not np.all((filesystem_roi_mask if align == (0, 0) else
                             shift_array(filesystem_roi_mask, shift=align, cval=1)) == database_roi_mask):
                raise ValueError(f'ROI mask for key=\n{key}\nhas been changed on filesystem but not in database.')
            elif align == (0, 0):
                filesystem_roi_mask = database_roi_mask.copy()

        if filesystem_roi_mask is None:
            return None, None, None

        if igor_roi_masks == 'yes':
            src_file, src_format = igor_file, 'igor_h5'
        else:
            src_file, src_format = roimask_file, _roi_mask_source_from_path(roimask_file)

        return filesystem_roi_mask, relative_store_path(src_file, self._filepath_store), src_format

    def plot1(self, key: dict | None = None, gamma: float = 0.5) -> None:
        """Plot the stack average with the ROI mask overlay for one presentation.

        Args:
            key: Primary key identifying the presentation to plot. If None,
                the first available key is used.
            gamma: Gamma correction value for display. Default is 0.5.
        """
        key = get_primary_key(table=self.proj() * self.presentation_table.proj(), key=key)
        npixartifact, scan_type = (self.field_table & key).fetch1('npixartifact', 'scan_type')
        data_name, alt_name = self.userinfo_table().get_stack_names(key)
        main_ch_average = load_array(
            (self.presentation_table.StackAverages & key & dict(ch_name=data_name)).fetch1('ch_average'))
        try:
            alt_ch_average = load_array(
                (self.presentation_table.StackAverages & key & dict(ch_name=alt_name)).fetch1('ch_average'))
        except dj.DataJointError:
            alt_ch_average = np.full_like(main_ch_average, np.nan)

        if self._presentation_masks:
            roi_mask = load_array((self.RoiMaskPresentation & key).fetch1('roi_mask'))
        else:  # Motion mode: field mask (presentation masks are in MotionRoiMask)
            roi_mask = load_array((self & {k: key[k] for k in self.primary_key}).fetch1('roi_mask'))
        plot_field(main_ch_average, alt_ch_average, scan_type=scan_type,
                   roi_mask=roi_mask, title=key, npixartifact=npixartifact, gamma=gamma)

    def load_high_res_bg_dict(self, key: dict | None, ch_names: list, verbose: bool = True) -> dict:
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
                    self.highres_table().StackAverages & key & [f"ch_name='{cn}'" for cn in ch_names]).to_arrays(
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


def _get_autorois_model(model: str | None, kind: str, autorois_models: dict | None) -> tuple[str, object, bool]:
    """Pick (or load) one AutoROIs model. Returns its name, the model and whether it was loaded here."""
    if model is None:
        model = 'UNet' if kind == 'default_rgc' else 'CorrRoiMask'

    loaded_here = autorois_models is None
    if loaded_here:
        autorois_models = load_default_autorois_models(kind=kind, models=[model])
    if autorois_models is None or model not in autorois_models:
        raise ValueError(f"Model {model!r} is not available, got {list(autorois_models or [])}")
    return model, autorois_models[model], loaded_here


def _predict_roi_mask(autorois_model, data: dict, ref_idx: int = 0, use_all_stacks: bool = False) -> np.ndarray:
    """Let an AutoROIs model propose a ROI mask for the stacks loaded by `RoiMaskTemplate._load_field_stacks`.

    The model is applied to the stack of the reference presentation `ref_idx`, or to all stacks (concatenated in
    time) if `use_all_stacks`. The first channel other than the ROI channel is passed as second input.
    """
    stacks, n_artifact = data['stacks'], data['n_artifact']
    ch0_stacks = stacks[data['roi_channel']]
    other_chs = [ch_name for ch_name in data['ch_names'] if ch_name != data['roi_channel']]
    ch1_stacks = stacks[other_chs[0]] if len(other_chs) > 0 else None

    if use_all_stacks:
        return np.asarray(autorois_model.create_mask_from_data(
            ch0_stack=ch0_stacks, ch1_stack=ch1_stacks, n_artifact=n_artifact,
            pixel_size_um=data['pixel_size_d1_d2'], multiple_stacks=True))
    return np.asarray(autorois_model.create_mask_from_data(
        ch0_stack=ch0_stacks[ref_idx], ch1_stack=None if ch1_stacks is None else ch1_stacks[ref_idx],
        n_artifact=n_artifact, pixel_size_um=data['pixel_size_d1_d2']))


def _remove_vessel_rois(roi_mask: np.ndarray, data: dict, ref_idx: int, vessel_threshold: float | None,
                        max_vessel_overlap: float, connectivity: int) -> np.ndarray:
    """Remove ROIs on blood vessels, found by comparing the first other channel to the ROI channel (no-op if None)."""
    if vessel_threshold is None:
        return roi_mask
    other_chs = [ch_name for ch_name in data['ch_names'] if ch_name != data['roi_channel']]
    if len(other_chs) == 0:
        warnings.warn('vessel_threshold is ignored, because the recording has no second channel.')
        return roi_mask
    alt_img = np.mean(data['stacks'][other_chs[0]][ref_idx], axis=-1)
    main_img = np.mean(data['stacks'][data['roi_channel']][ref_idx], axis=-1)
    vessel_mask = estimate_vessel_mask(alt_img, main_img, n_artifact=data['n_artifact'], alt_z=vessel_threshold)
    return remove_vessel_rois(roi_mask, vessel_mask, max_overlap=max_vessel_overlap, connectivity=connectivity,
                              verbose=False)


def _free_memory() -> None:
    """Release memory of unreferenced objects, also on the GPU."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _roi_mask_source_from_path(path: str | Path) -> str:
    """Return the ``roi_mask_source`` enum value for a ROI mask file, inferred from its extension."""
    suffix = Path(path).suffix.lower()
    if suffix == '.npy':
        return 'numpy'
    if suffix in ('.pkl', '.pickle'):
        return 'pickle'
    if suffix in ('.h5', '.hdf5'):
        return 'igor_h5'
    return 'other'


def load_stack_data(
        files: list,
        ch_names: list[str],
        from_raw_data: bool,
        roi_mask_dir: str = 'ROIs',
        old_prefix: str | None = None,
        new_prefix: str | None = None,
        file_format: str | None = None,
) -> tuple[dict[str, list], list]:
    """Load image stacks for a list of presentation files.

    Args:
        files: List of file paths to load stacks from.
        ch_names: Names of the channels to load, e.g. ['wDataCh0', 'wDataCh1']. The first channel is the
            primary channel, which has to be loadable from raw files (other channels may be replaced by zeros).
        from_raw_data: If True, read from raw ScanM files instead of h5.
        roi_mask_dir: Sub-directory name for ROI mask output files.
        old_prefix: Path prefix to replace in output file paths.
        new_prefix: Replacement path prefix.
        file_format: ``'numpy'`` or ``'pickle'``; controls the suffix of the
            output file paths passed to the GUI for saving.

    Returns:
        A 2-tuple of (stacks, output_files). stacks maps each channel name to a list of 3-D np.ndarray stacks,
        aligned with ``files``. output_files contains the corresponding ROI mask output file paths.
    """
    stacks = {ch_name: [] for ch_name in ch_names}
    output_files = []
    for data_file in files:
        ch_stacks, wparams = read_utils.load_stacks(
            data_file, from_raw_data=from_raw_data, ch_names=tuple(ch_names))
        for ch_name in ch_names:
            stacks[ch_name].append(ch_stacks[ch_name])
        output_files.append(to_roi_mask_file(
            data_file, roi_mask_dir=roi_mask_dir, old_prefix=old_prefix, new_prefix=new_prefix,
            file_format=file_format))

    return stacks, output_files


def load_default_autorois_models(kind: str = 'default_rgc', models: list[str] | None = None) -> dict | None:
    """Load auto-ROI models for a given cell type.

    By default only the latest UNet (and the correlation-based model) is loaded. Other models have to be
    requested explicitly via ``models``, e.g. ``models=['UNet', 'Cellpose3_cyto3', 'MEDIAR']``.

    Args:
        kind: Model variant to load. Supported values are 'default_rgc',
            'default_bc', 'default_ac', 'default_pr' (photoreceptors) and 'default_pr_watershed'
            (photoreceptors, watershed instead of region growing).
        models: Names of the models to load. Available for 'default_rgc': 'UNet' (latest, v0.2.0),
            'UNet_v1' (v0.1.0), 'Cellpose3_cyto2' and 'Cellpose3_cyto3' (finetuned), 'MEDIAR' (finetuned) and 'CorrRoiMask'.
            If None, the default set is loaded and models that fail to load are skipped with a warning.
            Explicitly requested models raise an error if they cannot be loaded.

    Returns:
        A dict mapping model names to model objects, or None if the dict is
        empty (e.g. if all models failed to load).
    """
    autorois_models = dict()

    if models is None:
        if kind == 'default_rgc':
            try:
                _add_autorois_unet(autorois_models)
            except Exception as e:
                warnings.warn(f'Failed to load default AutoROIs model UNet because of error:\n{e}')
        _add_autorois_corr(autorois_models, kind=kind)
    else:
        if isinstance(models, str):
            models = [models]
        loaders = {
            'UNet': _add_autorois_unet,
            'UNet_v1': _add_autorois_unet_v1,
            'Cellpose3_cyto2': lambda d: _add_cellpose(d, base='cyto2'),
            'Cellpose3_cyto3': lambda d: _add_cellpose(d, base='cyto3'),
            'MEDIAR': _add_mediar,
            'CorrRoiMask': lambda d: _add_autorois_corr(d, kind=kind),
        }
        unknown = [m for m in models if m not in loaders]
        if unknown:
            raise ValueError(f'Unknown AutoROIs models {unknown}. Available: {list(loaders)}')
        if kind != 'default_rgc' and any(m not in ('CorrRoiMask',) for m in models):
            raise NotImplementedError(f"Only 'CorrRoiMask' is available for {kind}.")
        for m in models:
            loaders[m](autorois_models)

    return autorois_models if len(autorois_models) > 0 else None


def _models_dir() -> Path:
    return file_store_path("Resources/AutoROIs/models", "reference")


def _add_autorois_unet_v1(autorois_models: dict) -> None:
    """Load the first UNet (v0.1.0, two input channels) and add it to the model dict.

    Args:
        autorois_models: Dict to add the loaded model to under the key 'UNet_v1'.
    """
    autorois_models['UNet_v1'] = default_models.load_unet_v1(_models_dir())


def _add_autorois_unet(autorois_models: dict) -> None:
    """Load the latest, retrained UNet (v0.2.0, one input channel) and add it to the model dict.

    Args:
        autorois_models: Dict to add the loaded model to under the key 'UNet'.
    """
    autorois_models['UNet'] = default_models.load_unet(_models_dir())


def _add_autorois_corr(autorois_models: dict, kind: str) -> None:
    """Load a correlation-based auto-ROI model and add it to the model dict.

    Args:
        autorois_models: Dict to add the loaded model to under the key
            'CorrRoiMask'.
        kind: Model variant controlling hyperparameters. Supported values are
            'default_rgc', 'default_bc', 'default_ac', 'default_pr' and 'default_pr_watershed'.
    """
    autorois_models['CorrRoiMask'] = default_models.get_corr_roi_mask(kind=kind)


def _add_cellpose(autorois_models: dict, kind: str = 'default_rgc', base: str = 'cyto3') -> None:
    """Load a finetuned Cellpose3 auto-ROI model and add it to the model dict.

    Requires the ``cellpose`` package (not a djimaging dependency, see README).

    Args:
        autorois_models: Dict to add the loaded model to under the key 'Cellpose3_<base>'.
        kind: Model variant. Currently only 'default_rgc' is supported.
        base: Cellpose model the weights were finetuned from, 'cyto2' or 'cyto3'.
    """
    if kind != 'default_rgc':
        raise NotImplementedError(kind)
    autorois_models[f'Cellpose3_{base}'] = default_models.load_cellpose3(_models_dir(), base=base)


def _add_mediar(autorois_models: dict, kind: str = 'default_rgc') -> None:
    """Load the finetuned MEDIAR auto-ROI model and add it to the model dict.

    Requires the ``mediar`` package (not a djimaging dependency, see README).

    Args:
        autorois_models: Dict to add the loaded model to under the key 'MEDIAR'.
        kind: Model variant. Currently only 'default_rgc' is supported.
    """
    if kind != 'default_rgc':
        raise NotImplementedError(kind)
    autorois_models['MEDIAR'] = default_models.load_mediar(_models_dir(), device='cpu')
