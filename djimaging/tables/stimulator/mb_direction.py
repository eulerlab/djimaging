"""
Moving bar directions on the retina.

The direction of a moving bar trial is defined by the script angle (`trial_info` of the Stimulus). How it moves on the
QDSpy screen depends on the moving bar version (QDSpy md5), how the screen is projected into the chamber on the
stimulator (ExperimentStimulator), and how the retina lies in the chamber (eye and prepwmorient of the experiment).
See djimaging.utils.stimulator for the conventions. Every row has a confidence ('high', 'medium', 'low') and a reason.

Example usage:

@schema
class MovingBarVersion(stimulator.MovingBarVersionTemplate):
    pass


@schema
class MbDirectionMap(stimulator.MbDirectionMapTemplate):
    presentation_table = Presentation
    stimulus_table = Stimulus
    experiment_stimulator_table = ExperimentStimulator
    mb_version_table = MovingBarVersion
    expinfo_table = Experiment.ExpInfo
    presentation_log_table = PresentationLog  # optional: None if there are no QDSpy logs
    stim_log_table = QdsPyLog.StimLog  # optional: None if there are no QDSpy logs


@schema
class MbRetinalPrefDir(stimulator.MbRetinalPrefDirTemplate):
    osds_table = OsDsIndexes
    mb_direction_map_table = MbDirectionMap
    retinal_field_location_table = RetinalFieldLocation  # optional, for the angle relative to the optic disc

MovingBarVersion().add_defaults()
"""
from abc import abstractmethod

import datajoint as dj
import numpy as np

from djimaging.utils.stimulator import (
    DEFAULT_MB_VERSIONS, combine_confidence, join_reasons, infer_mb_convention, map_mb_directions, retina_status,
    map_pref_dir, cardinal, angle_rel_optic_disc)


def fetch_stim_log(presentation_log_table, stim_log_table, key: dict) -> tuple[dict | None, str]:
    """QDSpy log entry (stim_md5, stim_file_name, params) of a presentation, or None and the reason."""
    if presentation_log_table is None or stim_log_table is None:
        return None, 'no QDSpy log tables in this schema'
    pres_log = (presentation_log_table & key).to_dicts()
    if len(pres_log) != 1:
        return None, f'{len(pres_log)} QDSpy log matches for this presentation'
    restriction = {k: v for k, v in key.items() if k in stim_log_table.primary_key}
    restriction.update(log_idx=pres_log[0]['log_idx'], stim_idx=pres_log[0]['stim_idx'])
    stim_logs = (stim_log_table & restriction).to_dicts()
    if len(stim_logs) != 1:
        return None, f'{len(stim_logs)} QDSpy log entries for log_idx/stim_idx of this presentation'
    return stim_logs[0], ''


def _none_if_nan(value):
    return None if value is None or not np.isfinite(value) else float(value)


class MovingBarVersionTemplate(dj.Manual):
    database = ""

    @property
    def definition(self):
        definition = """
        # Moving bar versions (QDSpy stimulus md5) and their direction convention
        stim_md5         : char(32)       # md5 of the QDSpy stimulus, as in the QDSpy log
        ---
        stim_file_name   : varchar(255)   # QDSpy script name
        dir_convention   : enum('rgc_movingbar', 'ds')  # see djimaging.utils.stimulator.MB_CONVENTIONS
        confidence       : enum('high', 'medium', 'low')
        notes            : varchar(1023)  # source and example users
        """
        return definition

    def add_defaults(self, skip_duplicates: bool = True) -> None:
        """Add the documented versions (QDSpy-stimuli-Documentation/stimuli/MovingBar)."""
        self.insert(DEFAULT_MB_VERSIONS, skip_duplicates=skip_duplicates)


class MbDirectionMapTemplate(dj.Computed):
    database = ""
    _stim_family = 'movingbar'

    # Optional QDSpy log tables (PresentationLog and QdsPyLog.StimLog) to identify the moving bar version
    presentation_log_table = None
    stim_log_table = None

    @property
    def definition(self):
        definition = """
        # Directions of the moving bar trials in the field and on the retina
        -> self.presentation_table
        -> self.experiment_stimulator_table
        ---
        stim_md5 = null          : char(32)       # md5 of the QDSpy stimulus (NULL: no log)
        stim_file_name = null    : varchar(255)   # QDSpy script name (NULL: no log)
        dir_convention = null    : varchar(32)    # moving bar convention (NULL: unknown)
        script_dirs_deg          : <blob>         # script angles of the trial_info directions
        field_dirs_deg           : <blob>         # angle in the field: 0 = +XCoord (to curtain), 90 = +YCoord; NaN unknown
        retinal_dirs_deg         : <blob>         # angle on the retina: 0 = nasal, 90 = dorsal; NaN unknown
        dorsal_components        : <blob>         # cosine of the angle to dorsal; NaN unknown
        script_to_retina = null  : <blob>         # 2x2 map from (cos, sin) of script angles to (nasal, dorsal)
        confidence               : enum('high', 'medium', 'low')  # confidence in retinal_dirs_deg
        reason                   : varchar(1023)  # why the confidence is not high
        """
        return definition

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def experiment_stimulator_table(self):
        pass

    @property
    @abstractmethod
    def mb_version_table(self):
        pass

    @property
    @abstractmethod
    def expinfo_table(self):
        pass

    @property
    def key_source(self):
        try:
            return ((self.presentation_table & (self.stimulus_table & dict(stim_family=self._stim_family))).proj()
                    * self.experiment_stimulator_table.proj())
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        trial_info = (self.stimulus_table & key).fetch1('trial_info')
        script_dirs_deg = np.array([float(trial['name']) for trial in trial_info])
        geometry = self.experiment_stimulator_table().get_geometry(key)
        eye, prepwmorient = (self.expinfo_table & key).fetch1('eye', 'prepwmorient')
        stim_log, log_reason = fetch_stim_log(self.presentation_log_table, self.stim_log_table, key)
        stim_md5 = stim_log['stim_md5'] if stim_log else None
        stim_file_name = stim_log['stim_file_name'] if stim_log else None

        row = compute_mb_direction_map(
            script_dirs_deg=script_dirs_deg, stim_md5=stim_md5, stim_file_name=stim_file_name,
            log_reason=log_reason, versions=self.mb_version_table().to_dicts(), geometry=geometry,
            eye=eye, prepwmorient=prepwmorient)
        self.insert1(dict(**key, **row))

    def report(self):
        """Number of presentations per confidence, convention and reason."""
        df = self.proj('confidence', 'reason', 'dir_convention', 'stim_md5').to_pandas().reset_index()
        if len(df) == 0:
            return df
        df = df.fillna({'dir_convention': 'unknown', 'stim_md5': 'none'})
        return df.groupby(['confidence', 'dir_convention', 'stim_md5', 'reason']).size().rename(
            'n_presentations').reset_index()

    def plot1(self, key=None):
        """Polar plot of the trial directions on the retina (nasal right, dorsal up)."""
        from matplotlib import pyplot as plt
        key = self.keys()[0] if key is None else (self & key).keys()[0]
        script, retinal, conf = (self & key).fetch1('script_dirs_deg', 'retinal_dirs_deg', 'confidence')
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(projection='polar')
        for s, r in zip(script, retinal):
            if np.isfinite(r):
                ax.annotate('', xy=(np.deg2rad(r), 1), xytext=(0, 0), arrowprops=dict(width=1))
                ax.text(np.deg2rad(r), 1.15, f'{s:.0f}', ha='center', va='center')
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]), ['N', 'D', 'T', 'V'])
        ax.set_yticks([])
        ax.set_title(f'script angle on retina ({conf})')
        return fig, ax


def compute_mb_direction_map(script_dirs_deg, stim_md5, stim_file_name, log_reason, versions, geometry, eye,
                             prepwmorient) -> dict:
    """Row of MbDirectionMap (without key) from fetched data."""
    convention, conv_conf, conv_reason = infer_mb_convention(stim_md5, stim_file_name, versions)
    if stim_md5 is None and log_reason:
        conv_reason = join_reasons(log_reason, conv_reason)
    _, _, retina_conf, retina_reason = retina_status(eye, prepwmorient)

    res = map_mb_directions(script_dirs_deg, convention, geometry['screen_x_axis'], geometry['screen_y_axis'],
                            prepwmorient=prepwmorient, eye=eye)
    confidence = combine_confidence(geometry['confidence'], conv_conf, retina_conf)
    reason = join_reasons(geometry['reason'], conv_reason, retina_reason)
    return dict(
        stim_md5=stim_md5, stim_file_name=stim_file_name, dir_convention=convention,
        script_dirs_deg=np.asarray(script_dirs_deg, dtype=float),
        field_dirs_deg=res['field_dirs_deg'], retinal_dirs_deg=res['retinal_dirs_deg'],
        dorsal_components=res['dorsal_components'], script_to_retina=res['script_to_retina'],
        confidence=confidence, reason=reason[:1023])


class MbRetinalPrefDirTemplate(dj.Computed):
    database = ""

    # Optional RetinalFieldLocation table for the angle relative to the optic disc
    retinal_field_location_table = None
    _min_od_dist_um = 100.  # No angle relative to the optic disc closer than this

    @property
    def definition(self):
        definition = """
        # Preferred direction of the moving bar response on the retina
        -> self.osds_table
        -> self.mb_direction_map_table
        ---
        pref_dir_retinal_deg = null : float32   # 0 = nasal, 90 = dorsal, 180 = temporal, 270 = ventral
        pref_dir_cardinal = null    : enum('nasal', 'dorsal', 'temporal', 'ventral')  # nearest cardinal direction
        pref_dir_dorsal = null      : float32   # cosine of the angle to dorsal (known without eye)
        pref_dir_rel_od_deg = null  : float32   # 0 = away from the optic disc, +-180 = towards it
        confidence                  : enum('high', 'medium', 'low')  # confidence in pref_dir_retinal_deg
        reason                      : varchar(1023)
        """
        return definition

    @property
    @abstractmethod
    def osds_table(self):
        pass

    @property
    @abstractmethod
    def mb_direction_map_table(self):
        pass

    @property
    def key_source(self):
        try:
            return self.osds_table.proj() * self.mb_direction_map_table.proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        pref_dir = (self.osds_table & key).fetch1('pref_dir')
        script_to_retina, confidence, reason = (self.mb_direction_map_table & key).fetch1(
            'script_to_retina', 'confidence', 'reason')

        od_pos = None
        if self.retinal_field_location_table is not None:
            rows = (self.retinal_field_location_table & key).to_dicts()
            if len(rows) == 1:
                od_pos = rows[0]['temporal_nasal_pos_um'], rows[0]['ventral_dorsal_pos_um']

        self.insert1(dict(**key, **compute_retinal_pref_dir(
            pref_dir, script_to_retina, confidence, reason, od_pos, min_od_dist_um=self._min_od_dist_um)))

    def plot(self, restriction=None, ds_pvalue_thresh: float = 0.05):
        """Histogram of retinal preferred directions of direction selective ROIs."""
        from matplotlib import pyplot as plt
        restriction = {} if restriction is None else restriction
        rel = (self * self.osds_table.proj('ds_pvalue') & restriction & f'ds_pvalue<{ds_pvalue_thresh}'
               & 'pref_dir_retinal_deg IS NOT NULL')
        angles = rel.to_arrays('pref_dir_retinal_deg')
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(projection='polar')
        ax.hist(np.deg2rad(np.asarray(angles, dtype=float)), bins=np.linspace(0, 2 * np.pi, 25))
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]), ['N', 'D', 'T', 'V'])
        ax.set_title(f'n={len(angles)}, p<{ds_pvalue_thresh}')
        return fig, ax


def compute_retinal_pref_dir(pref_dir_rad, script_to_retina, confidence, reason, od_pos=None,
                             min_od_dist_um=100.) -> dict:
    """Row of MbRetinalPrefDir (without key). `od_pos` is (nasal, dorsal) position of the field or None."""
    angle, dorsal = map_pref_dir(pref_dir_rad, script_to_retina)
    rel_od = np.nan
    if od_pos is not None:
        rel_od = angle_rel_optic_disc(angle, od_pos[0], od_pos[1], min_dist_um=min_od_dist_um)
    if script_to_retina is not None and not (pref_dir_rad is not None and np.isfinite(pref_dir_rad)):
        confidence, reason = 'low', join_reasons(reason, 'no preferred direction')
    return dict(pref_dir_retinal_deg=_none_if_nan(angle), pref_dir_cardinal=cardinal(angle),
                pref_dir_dorsal=_none_if_nan(dorsal), pref_dir_rel_od_deg=_none_if_nan(rel_od),
                confidence=confidence, reason=reason[:1023])
