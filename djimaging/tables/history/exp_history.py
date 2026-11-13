"""Experiment history: every ScanM recording of every experiment, independent of stimuli and fields.

- `ExpHistoryTemplate`: one entry per experiment (scanned folders) with one `Recording` per recording
  (.smp + .smh and/or .h5 file), with the scan parameters needed to reconstruct the laser exposure of the tissue.
- `ExpHistoryFieldTemplate`: which recordings belong to which field.
- `ExpHistoryStimLogTemplate`: which QDSpy stimulus (log entry) was shown during which recording. Covers also
  stimuli that were shown while nothing was recorded (e.g. during focus scans).

Recordings of stimuli that are not in the Stimulus table ("unlisted", e.g. unpublished stimuli) are registered with
their names, but never become Presentations, so they never get traces. See `djimaging.utils.stimulus_listing`.
Laser power and stimulus intensity are not stored digitally, so the exposure history is in time, position, scan
area and pixel dwell time.
"""
from __future__ import annotations

import os
import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
import pandas as pd

from djimaging.utils import exp_history_utils
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.scanm import read_header_utils
from djimaging.utils.stimulus_listing import listed_stimuli


class ExpHistoryTemplate(dj.Computed):
    database = ""
    _filepath_store = "reference"
    # Folders (relative to the experiment folder) that are scanned in addition to `pre_data_dir` and `raw_data_dir`
    # of UserInfo, e.g. ('Stacks',). Can also be passed to populate: make_kwargs=dict(extra_data_dirs=[...]).
    extra_data_dirs = ()
    # Count the triggers of .smp files without .h5 file? Loads the whole .smp file, so it is slow.
    _count_smp_triggers = False
    # Optional: ExperimentImagingSetup (stimulator block). If bound, the pixel size is computed from its dated scan
    # width (as in Field, Presentation and HighRes), and only experiments with an imaging setup are populated.
    # If None, the setupid constants of `setup_utils` (setupid of the recording header) are used.
    experiment_imaging_setup_table = None

    @property
    def definition(self):
        definition = """
        # All ScanM recordings of an experiment, independent of stimuli and fields
        -> self.experiment_table
        ---
        data_dirs : <blob>  # scanned folders, relative to the experiment folder
        n_recordings : int32  # number of recordings
        unscanned_files : <blob>  # ScanM files (.smp, SMP_*.h5) in other folders of the experiment, not registered
        """
        return definition

    @property
    @abstractmethod
    def experiment_table(self):
        pass

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    @property
    def key_source(self):
        try:
            if self.experiment_imaging_setup_table is not None:
                return self.experiment_table.proj() & self.experiment_imaging_setup_table.proj()
            return self.experiment_table.proj()
        except (AttributeError, TypeError):
            pass

    def _scan_width_um(self, key: dict) -> float | None:
        """Scan width at zoom 1 of the experiment from `experiment_imaging_setup_table`, None if not bound."""
        if self.experiment_imaging_setup_table is None:
            return None
        return self.experiment_imaging_setup_table().get_recording_params(key)['scan_width_um']

    class Recording(dj.Part):
        _filepath_store = "reference"

        @property
        def definition(self):
            definition = f"""
            # One ScanM recording: .smp (+ .smh) and/or .h5 file
            -> master
            rec_name : varchar(191)  # .smp file name (else .h5 file name without `SMP_`) without extension
            ---
            source : enum('smp+h5', 'smp', 'h5')  # files of this recording
            link_method : enum('orig_file_name', 'file_name', 'time_stamp', 'none')  # how .smp and .h5 were linked
            smp_file = NULL : <filepath@{self._filepath_store}>  # raw ScanM file
            smh_file = NULL : <filepath@{self._filepath_store}>  # ScanM header of the .smp file
            h5_file = NULL : <filepath@{self._filepath_store}>  # .h5 file exported from the .smp file
            file_size_bytes : int64  # size of the .smp file, else of the .h5 file
            file_mtime : datetime  # modification time of that file
            kind : enum('response', 'hr', 'od', 'outline', 'unknown')  # from file name and UserInfo aliases
            field_token = NULL : varchar(63)  # field part of the file name (.h5 name if available)
            stim_token = NULL : varchar(63)  # stimulus part of the file name (.h5 name if available)
            name_info : <blob>  # all parts of the file names: dict(h5=..., smp=...)
            rec_start = NULL : datetime  # recording start (clock of the ScanM computer)
            n_frames = NULL : int64  # number of recorded frames
            n_frames_set = NULL : int64  # number of frames set in ScanM (only known from .smh)
            stopped_early = NULL : bool  # fewer frames than set: stopped by the user or aborted
            frame_dt_s = NULL : float32  # laser scanning time per frame in s
            rec_dur_s = NULL : float32  # laser scanning time of the recording in s (n_frames * frame_dt_s)
            scan_type : enum('xy', 'xz', 'xyz', 'line', 'unknown')  # from the number of pixels in y and z
            scan_type_id = NULL : int32  # ScanM ScanType (e.g. 11 for z-stacks)
            nxpix = NULL : int64  # number of pixels in x, without retrace and offset
            nypix = NULL : int64  # number of pixels in y
            nzpix = NULL : int64  # number of pixels in z
            nxpix_full = NULL : int64  # number of pixels in x, with retrace and offset
            n_fr_per_step = NULL : int32  # frames per z-step
            real_pix_dur_us = NULL : float32  # pixel dwell time in us
            zoom = NULL : float32  # zoom of the scan
            angle_deg = NULL : float32  # scan angle
            scan_width_um = NULL : float32  # scan width at zoom 1 used for pixel_size_um (imaging setup or setupid)
            pixel_size_um = NULL : float32  # scan_width_um / nxpix / zoom; NULL if setup or zoom are not supported
            absx = NULL : float32  # stage position in x in um
            absy = NULL : float32  # stage position in y in um
            absz = NULL : float32  # stage position in z in um
            z_step_um = NULL : float32  # z-step in um
            setupid = NULL : int32  # setup id in the recording header
            laser_wavelength_nm = NULL : float32  # as set in ScanM, often 0 (not set)
            objective : varchar(191)  # as set in ScanM, often 'n/a'
            user_comment : varchar(1000)  # comment set in ScanM
            computer_name : varchar(191)  # ScanM computer
            scanm_version : varchar(191)  # ScanM version
            input_ch_mask = NULL : int32  # bit mask of the recorded input channels
            stim_ch_mask = NULL : int32  # bit mask of the ScanM output channels
            n_triggers = NULL : int32  # stimulus triggers ('Triggertimes' of the .h5 or counted); NULL: unknown
            header : <blob>  # all header values: dict(smh=..., h5=..., h5_ch_names=...)
            header_issue = NULL : varchar(1000)  # problems while reading the headers
            """
            return definition

    def make(self, key: dict, extra_data_dirs: list | None = None, verbose: int = 0) -> None:
        """Register all recordings of one experiment.

        Parameters
        ----------
        key : dict
            Primary key of the experiment.
        extra_data_dirs : list | None, optional
            Folders (relative to the experiment folder) to scan in addition to `pre_data_dir`, `raw_data_dir` and
            `extra_data_dirs` of the table.
        verbose : int, optional
            Print the registered recordings if > 0.
        """
        header_path = (self.experiment_table & key).fetch1('header_path')
        user_dict = (self.userinfo_table & key).fetch1()

        data_dirs = [user_dict['pre_data_dir'], user_dict['raw_data_dir'], *self.extra_data_dirs,
                     *(extra_data_dirs or [])]
        folders = exp_history_utils.resolve_data_dirs(header_path, data_dirs)
        recordings = exp_history_utils.find_recordings(folders)
        unscanned = exp_history_utils.find_unscanned_recording_files(header_path, folders)
        if len(unscanned) > 0:
            warnings.warn(f"{len(unscanned)} ScanM file(s) in {header_path} are not in the scanned folders "
                          f"{data_dirs} and are not registered, e.g. {unscanned[:3]}. "
                          f"Add their folders to `extra_data_dirs` to register them.")

        scan_width_um = self._scan_width_um(key)
        rows = [self._recording_row(key, rec, user_dict, scan_width_um=scan_width_um) for rec in recordings]

        self.insert1(dict(
            **key, n_recordings=len(rows), data_dirs=[os.path.relpath(f, header_path) for f in folders],
            unscanned_files=[os.path.relpath(f, header_path) for f in unscanned]))
        self.Recording().insert(rows)

        if verbose > 0:
            print(f"{key}: {len(rows)} recordings in {folders}")

    def _recording_row(self, key: dict, rec: exp_history_utils.RecordingFiles, user_dict: dict,
                       scan_width_um: float | None = None) -> dict:
        from djimaging.utils.dj_storage import relative_store_path

        entry = exp_history_utils.read_recording(rec, count_smp_triggers=self._count_smp_triggers,
                                                 scan_width_um=scan_width_um)
        name_info = dict(
            h5=exp_history_utils.parse_recording_name(rec.h5.path, user_dict) if rec.h5 is not None else None,
            smp=exp_history_utils.parse_recording_name(rec.smp.path, user_dict) if rec.smp is not None else None)
        main_info = name_info['h5'] if name_info['h5'] is not None else name_info['smp']
        size, mtime = read_header_utils.file_size_and_mtime((rec.smp or rec.h5).path)

        def store_path(f):
            return relative_store_path(f, self._filepath_store) if f is not None else None

        return dict(
            **key, rec_name=rec.rec_name, source=rec.source, link_method=rec.link_method,
            smp_file=store_path(rec.smp.path if rec.smp else None),
            smh_file=store_path(rec.smp.smh if rec.smp else None),
            h5_file=store_path(rec.h5.path if rec.h5 else None),
            file_size_bytes=size, file_mtime=mtime,
            kind=main_info['kind'],
            field_token=_token(main_info.get('field')),
            stim_token=_token(main_info.get('stimulus')),
            name_info=name_info,
            **entry)

    def listed(self, stimulus_table, restriction=None, from_raw_data: bool = False) -> pd.DataFrame:
        """All recordings with their listed status (see `djimaging.utils.stimulus_listing`).

        Parameters
        ----------
        stimulus_table : dj.Table
            The Stimulus table of the schema.
        restriction : optional
            Restriction applied to the recordings.
        from_raw_data : bool, optional
            Use the parts of the .smp file name instead of the .h5 file name, as Presentation does with
            `from_raw_data=1`. Falls back to the other file if one is missing. Default: False.

        Returns
        -------
        pd.DataFrame
            One row per recording with key, `rec_name`, `kind`, `stim_token`, `listed` and `stim_names`
            (the stimuli it is listed for).
        """
        aliases = dict(zip(*stimulus_table().to_arrays('stim_name', 'alias')))
        recordings = self.Recording() & (restriction if restriction is not None else {})
        rows = recordings.proj('kind', 'name_info').to_dicts()
        for row in rows:
            name_info = row.pop('name_info')
            info = _preferred_name_info(name_info, from_raw_data)
            row['stim_token'] = info.get('stimulus') if info else None
            row['kind'] = info.get('kind', row['kind']) if info else row['kind']
            row['stim_names'] = listed_stimuli(row['stim_token'], aliases) if row['kind'] == 'response' else []
            row['listed'] = len(row['stim_names']) > 0
        return pd.DataFrame(rows)

    def listed_keys(self, stimulus_table, listed: bool = True, restriction=None, from_raw_data: bool = False) -> list:
        """Keys of the recordings that are listed (`listed=True`) or unlisted (`listed=False`). Use as restriction,
        e.g. `ExpHistory.Recording & ExpHistory().listed_keys(Stimulus, listed=False)`."""
        df = self.listed(stimulus_table, restriction=restriction, from_raw_data=from_raw_data)
        if len(df) == 0:
            return []
        key_names = self.Recording.primary_key
        return df[df['listed'] == listed][key_names].to_dict('records')

    def plot1(self, key: dict | None = None) -> None:
        """Timeline (laser scanning) and stage positions of all recordings of one experiment."""
        import matplotlib.pyplot as plt

        key = get_primary_key(table=self, key=key)
        df = (self.Recording & key).proj(
            'rec_start', 'rec_dur_s', 'kind', 'stim_token', 'absx', 'absy').to_pandas().reset_index()
        df = df[df['rec_start'].notnull()].sort_values('rec_start')
        if len(df) == 0:
            print(f"No recording with start time for {key}")
            return

        t0 = df['rec_start'].min()
        df['t_start_min'] = (df['rec_start'] - t0).dt.total_seconds() / 60
        df['dur_min'] = df['rec_dur_s'].fillna(0) / 60
        kinds = sorted(df['kind'].unique())
        colors = {k: f"C{i}" for i, k in enumerate(kinds)}

        fig, axs = plt.subplots(1, 2, figsize=(14, 4), width_ratios=(2, 1))
        for i, row in enumerate(df.itertuples()):
            axs[0].barh(i, row.dur_min, left=row.t_start_min, color=colors[row.kind])
        axs[0].set_yticks(np.arange(len(df)), [f"{r} ({s})" for r, s in zip(df['rec_name'], df['stim_token'])],
                          fontsize=6)
        axs[0].set_xlabel(f"Time since first recording ({t0}) [min]")
        axs[0].invert_yaxis()
        for k in kinds:
            axs[0].plot([], [], c=colors[k], lw=5, label=k)
        axs[0].legend(loc='lower left')

        sc = axs[1].scatter(df['absx'], df['absy'], c=df['t_start_min'], cmap='viridis')
        axs[1].set(xlabel='absx [um]', ylabel='absy [um]', aspect='equal', title='Stage position')
        plt.colorbar(sc, ax=axs[1], label='Time [min]')
        fig.suptitle(key)
        plt.tight_layout()
        plt.show()


def _token(value) -> str | None:
    return None if value is None else str(value)[:63]


def _preferred_name_info(name_info: dict, from_raw_data: bool) -> dict | None:
    first, second = ('smp', 'h5') if from_raw_data else ('h5', 'smp')
    return name_info.get(first) or name_info.get(second)


class ExpHistoryFieldTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        # Recordings of a field (all kinds, also high resolution stacks and unlisted stimuli)
        -> self.field_table
        ---
        n_recordings : int32  # number of recordings of this field
        """
        return definition

    @property
    @abstractmethod
    def field_table(self):
        pass

    @property
    @abstractmethod
    def exp_history_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.field_table.proj() & self.exp_history_table.proj()
        except (AttributeError, TypeError):
            pass

    class Recording(dj.Part):
        @property
        def definition(self):
            definition = """
            # Recording of this field
            -> master
            -> self.recording_table
            """
            return definition

        @property
        @abstractmethod
        def recording_table(self):
            pass

    def make(self, key: dict) -> None:
        """Link a field to all recordings whose file name parts (field, region, conditions) match its key."""
        from_raw_data = (self.field_table().raw_params_table & key).fetch1('from_raw_data')
        exp_key = (self.exp_history_table & key).fetch1('KEY')
        rec_names, name_infos = (self.exp_history_table.Recording & exp_key).to_arrays('rec_name', 'name_info')
        infos = {r: _preferred_name_info(n, from_raw_data) for r, n in zip(rec_names, name_infos)}

        matches = exp_history_utils.match_field_recordings(infos, key, self.field_table().new_primary_keys)

        self.insert1(dict(**key, n_recordings=len(matches)))
        self.Recording().insert([dict(**key, rec_name=rec_name) for rec_name in matches])


class ExpHistoryStimLogTemplate(dj.Computed):
    database = ""
    _max_clock_offset_s = 900.  # Maximum clock difference between the QDSpy and the ScanM computer
    _tolerance_s = 2.  # A stimulus may start this much before the recording and is still assigned to it

    @property
    def definition(self):
        definition = """
        # QDSpy stimuli (log entries) of an experiment and the recordings during which they were shown
        -> self.exp_history_table
        ---
        clock_offset_s = NULL : float32  # estimate: add to QDSpy log times to get ScanM times; NULL: no match
        n_stims : int32  # number of logged stimuli
        n_matched : int32  # number of stimuli shown during a recording
        """
        return definition

    @property
    @abstractmethod
    def exp_history_table(self):
        pass

    @property
    @abstractmethod
    def log_table(self):
        """QdsPyLog (master): the experiments of which the log was parsed."""
        pass

    @property
    def key_source(self):
        try:
            return self.exp_history_table.proj() & self.log_table.proj()
        except (AttributeError, TypeError):
            pass

    class Stim(dj.Part):
        @property
        def definition(self):
            definition = """
            # Logged stimulus and the recording during which it started
            -> master
            -> self.stim_log_table
            ---
            rec_name = NULL : varchar(191)  # recording during which the stimulus started; NULL: no recording
            lag_s = NULL : float32  # stimulus start minus recording start in s
            """
            return definition

        @property
        @abstractmethod
        def stim_log_table(self):
            pass

    def make(self, key: dict) -> None:
        """Estimate the clock offset between the QDSpy and the ScanM computer and assign stimuli to recordings."""
        recs = (self.exp_history_table.Recording & key & 'rec_start IS NOT NULL').proj(
            'rec_start', 'rec_dur_s').to_dicts(order_by='rec_start')
        stim_log_table = self.Stim.stim_log_table
        stims = (stim_log_table & key).proj('t_start').to_dicts()

        rec_starts = [exp_history_utils.seconds_of_day(r['rec_start']) for r in recs]
        rec_ends = [s + (r['rec_dur_s'] or 0.) for s, r in zip(rec_starts, recs)]
        stim_starts = [exp_history_utils.seconds_of_day(s['t_start']) for s in stims]

        offset, matches = exp_history_utils.match_stimuli_to_recordings(
            rec_starts, rec_ends, stim_starts, max_offset_s=self._max_clock_offset_s, tolerance_s=self._tolerance_s)

        self.insert1(dict(**key, clock_offset_s=offset, n_stims=len(stims),
                          n_matched=sum(m[0] is not None for m in matches)))
        stim_keys = stim_log_table.primary_key
        self.Stim().insert([
            {**key, **{k: s[k] for k in stim_keys},
             'rec_name': recs[rec_idx]['rec_name'] if rec_idx is not None else None, 'lag_s': lag}
            for s, (rec_idx, lag) in zip(stims, matches)])
