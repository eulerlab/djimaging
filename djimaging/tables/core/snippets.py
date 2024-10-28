import warnings
from abc import abstractmethod

import datajoint as dj
import numpy as np
from matplotlib import pyplot as plt

from djimaging.tables.core.stimulus import reformat_numerical_trial_info
from djimaging.utils import plot_utils
from djimaging.utils.dj_utils import get_primary_key
from djimaging.utils.snippet_utils import split_trace_by_reps, split_trace_by_group_reps


def get_aligned_snippets_times(snippets_times, raise_error=True, tol=1e-4):
    snippets_times = snippets_times - snippets_times[0, :]

    is_inconsistent = np.any(np.std(snippets_times, axis=1) > tol)
    if is_inconsistent:
        if raise_error:
            raise ValueError(f'Failed to snippet times: max_std={np.max(np.std(snippets_times, axis=1))}')
        else:
            warnings.warn(f'Snippet times are inconsistent: max_std={np.max(np.std(snippets_times, axis=1))}')

    aligned_times = np.mean(snippets_times, axis=1)
    return aligned_times


class SnippetsTemplate(dj.Computed):
    database = ""
    _pad_trace = False  # If True, chose snippet times always contain the trigger times
    _dt_base_line_dict = None  # dict of baseline time for each stimulus with stimulus based baseline correction

    """
    Examples for _dt_base_line_dict:
    This is deprecated and should be replaced by the snippet_base_dt in the stimulus table.
    
    Baden 16 / Franke 17:
    _dt_base_line_dict = {
        'gChirp': 8*0.128,
        'movingbar':  5*0.128,
    }
    """

    @property
    def definition(self):
        definition = """
        # Snippets created from slicing traces using the triggertimes. 
        -> self.preprocesstraces_table
        ---
        snippets               :longblob          # array of snippets (time x repetitions)
        snippets_t0            :blob              # array of snippet start times (repetitions, ) 
        snippets_dt            :float
        triggertimes_snippets  :longblob          # snippeted triggertimes (ntrigger_rep x repetitions)
        droppedlastrep_flag    :tinyint unsigned  # Was the last repetition incomplete and therefore dropped?
        """
        return definition

    @property
    @abstractmethod
    def preprocesstraces_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def traces_table(self):
        pass

    @property
    def key_source(self):
        try:
            return (self.preprocesstraces_table() & (self.stimulus_table() & "isrepeated=1")).proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key):
        stim_name = (self.stimulus_table() & key).fetch1('stim_name')
        ntrigger_rep = (self.stimulus_table() & key).fetch1('ntrigger_rep')
        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')
        pp_trace_t0, pp_trace_dt, pp_trace = (self.preprocesstraces_table() & key).fetch1(
            'pp_trace_t0', 'pp_trace_dt', 'pp_trace')

        pp_trace_times = np.arange(len(pp_trace)) * pp_trace_dt + pp_trace_t0

        snippets, snippets_times, triggertimes_snippets, droppedlastrep_flag = split_trace_by_reps(
            pp_trace, pp_trace_times, triggertimes, ntrigger_rep, allow_drop_last=True, pad_trace=self._pad_trace)

        dt_baseline = self.get_snippet_base_dt(stim_name)
        if dt_baseline is not None:
            n_baseline = int(np.round(dt_baseline / np.mean(np.diff(snippets_times, axis=0))))
            snippets = snippets - np.median(snippets[:n_baseline, :], axis=0)

        self.insert1(dict(
            **key,
            snippets=snippets.astype(np.float32),
            snippets_t0=snippets_times[0, :],
            snippets_dt=pp_trace_dt,
            triggertimes_snippets=triggertimes_snippets.astype(np.float32),
            droppedlastrep_flag=int(droppedlastrep_flag),
        ))

    def get_snippet_base_dt(self, stim_name):
        try:
            dt_baseline = (self.stimulus_table & dict(stim_name=stim_name)).fetch1('snippet_base_dt')
            if not np.isfinite(dt_baseline):
                dt_baseline = None
        except dj.DataJointError:
            dt_baseline = None

        dt_baseline_alt = None if self._dt_base_line_dict is None else self._dt_base_line_dict.get(stim_name, None)

        if dt_baseline is not None and dt_baseline_alt is not None:
            if dt_baseline != dt_baseline_alt:
                raise ValueError(
                    f"dt_baseline[Stimulus]={dt_baseline} and dt_baseline[Snippets]={dt_baseline_alt} are not equal. "
                    f"Please set only one of them, ideally in the stimulus table."
                )
        elif dt_baseline is None and dt_baseline_alt is not None:
            dt_baseline = dt_baseline_alt

        return dt_baseline

    def plot1(self, key=None, xlim=None, xlim_aligned=None):
        key = get_primary_key(table=self, key=key)
        snippets_t0, snippets_dt, snippets, triggertimes_snippets = (self & key).fetch1(
            "snippets_t0", "snippets_dt", "snippets", "triggertimes_snippets")

        snippets_times = (np.tile(np.arange(snippets.shape[0]) * snippets_dt, (len(snippets_t0), 1)).T
                          + snippets_t0)

        fig, axs = plt.subplots(3, 1, figsize=(10, 6))

        plot_utils.plot_trace_and_trigger(
            ax=axs[0], time=snippets_times, trace=snippets, triggertimes=triggertimes_snippets, title=str(key))
        axs[0].set(xlim=xlim)

        axs[1].plot(snippets_times - triggertimes_snippets[0], snippets, alpha=0.5)
        axs[1].set(ylabel='trace', xlabel='rel. to trigger', xlim=xlim_aligned)

        aligned_times = get_aligned_snippets_times(snippets_times=snippets_times)
        plot_utils.plot_traces(
            ax=axs[2], time=aligned_times, traces=snippets.T)
        axs[2].set(ylabel='trace', xlabel='aligned time', xlim=xlim_aligned)

        plt.tight_layout()


class GroupSnippetsTemplate(dj.Computed):
    """
    Example trial info for Center-Ring-Surround flicker (CRS) stimulus with test sequences:
    crs_trial_info = [
        dict(name='UV', ntrigger=120),
        dict(name='_', ntrigger=1),
        dict(name='UV-test', ntrigger=2),
        dict(name='green-test', ntrigger=2),
        dict(name='_', ntrigger=1),
        dict(name='green', ntrigger=120),
        dict(name='_', ntrigger=1),
        dict(name='UV-test', ntrigger=2),
        dict(name='green-test', ntrigger=2),
        dict(name='_', ntrigger=1),
        dict(name='green', ntrigger=120),
        dict(name='_', ntrigger=1),
        dict(name='UV-test', ntrigger=2),
        dict(name='green-test', ntrigger=2),
        dict(name='_', ntrigger=1),
        dict(name='UV', ntrigger=120),
        dict(name='_', ntrigger=1),
        dict(name='UV-test', ntrigger=2),
        dict(name='green-test', ntrigger=2),
        dict(name='_', ntrigger=1),
    ]

    Example for moving bar (MB):
    mb_trial_info = [
        {'name': 0, 'ntrigger': 1},
        {'name': 180, 'ntrigger': 1},
        {'name': 45, 'ntrigger': 1},
        {'name': 225, 'ntrigger': 1},
        {'name': 90, 'ntrigger': 1},
        {'name': 270, 'ntrigger': 1},
        {'name': 135, 'ntrigger': 1},
        {'name': 315, 'ntrigger': 1},
    ]
    """

    database = ""
    _pad_trace = False  # If True, chose snippet times always contain the trigger times

    @property
    def definition(self):
        definition = """
        # Snippets created from slicing traces using the triggertimes. 
        -> self.preprocesstraces_table
        ---
        snippets               :longblob          # dict of array of snippets (group: time [x repetitions])
        snippets_t0           :blob              # dict of array of snippet start times (group: repetitions) 
        snippets_dt            :float
        triggertimes_snippets  :longblob          # dict of array of triggertimes (group: time [x repetitions])
        droppedlastrep_flag    :tinyint unsigned  # Was the last repetition incomplete and therefore dropped?
        """
        return definition

    @property
    @abstractmethod
    def preprocesstraces_table(self):
        pass

    @property
    @abstractmethod
    def stimulus_table(self):
        pass

    @property
    @abstractmethod
    def presentation_table(self):
        pass

    @property
    @abstractmethod
    def traces_table(self):
        pass

    @property
    def key_source(self):
        try:
            return (self.preprocesstraces_table() & (self.stimulus_table() & "trial_info!='None'")).proj()
        except (AttributeError, TypeError):
            pass

    def make(self, key, allow_incomplete=True):
        trial_info, stim_dict = (self.stimulus_table() & key).fetch1('trial_info', 'stim_dict')
        triggertimes = (self.presentation_table() & key).fetch1('triggertimes')
        pp_trace_t0, pp_trace_dt, pp_trace = (self.preprocesstraces_table() & key).fetch1(
            'pp_trace_t0', 'pp_trace_dt', 'pp_trace')

        pp_trace_times = np.arange(len(pp_trace)) * pp_trace_dt + pp_trace_t0

        if not isinstance(trial_info[0], dict):
            trial_info = reformat_numerical_trial_info(trial_info)

        delay = stim_dict.get('trigger_delay', 0.) if stim_dict is not None else 0
        
        snippets, snippets_times, triggertimes_snippets, droppedlastrep_flag = split_trace_by_group_reps(
            pp_trace, pp_trace_times, triggertimes, trial_info=trial_info, delay=delay,
            allow_incomplete=allow_incomplete, pad_trace=self._pad_trace, stack_kind='pad')

        snippets_t0 = {k: v[0, :] for k, v in snippets_times.items()}

        self.insert1(dict(
            **key,
            snippets=snippets,
            snippets_t0=snippets_t0,
            snippets_dt=pp_trace_dt,
            triggertimes_snippets=triggertimes_snippets,
            droppedlastrep_flag=droppedlastrep_flag,
        ))

    def plot1(self, key=None, xlim=None):
        key = get_primary_key(table=self, key=key)
        snippets_t0, snippets_dt, snippets, triggertimes_snippets = (self & key).fetch1(
            "snippets_t0", "snippets_dt", "snippets", "triggertimes_snippets")

        import matplotlib as mpl

        names = list(snippets.keys())

        colors = mpl.colormaps['jet'](np.linspace(0, 1, len(names)))
        name2color = {name: colors[i] for i, name in enumerate(names)}

        fig, axs = plt.subplot_mosaic([["all"]] + [[name] for name in names], figsize=(8, 2 * (len(names) + 1)))

        tt_min = np.min([np.nanmin(snippets[name]) for name in names])
        tt_max = np.max([np.nanmax(snippets[name]) for name in names])

        for i, name in enumerate(names):
            snippets_times = (
                    np.tile(np.arange(snippets[name].shape[0]) * snippets_dt, (len(snippets_t0[name]), 1)).T
                    + snippets_t0[name])

            axs['all'].vlines(triggertimes_snippets[name], tt_min, tt_max, color='k', zorder=-100, lw=0.5, alpha=0.5,
                              label='trigger' if name == names[0] else '_')
            axs['all'].plot(snippets_times, snippets[name], color=name2color[name],
                            label=[name] + ['_'] * (snippets[name].shape[1] - 1), alpha=0.8)

            axs[name].set(title='trace', xlabel='absolute time')
            axs[name].plot(snippets_times - triggertimes_snippets[name][0], snippets[name], lw=1)
            axs[name].set(title=name, xlabel='relative time')
            axs[name].title.set_color(name2color[name])
            axs[name].vlines(triggertimes_snippets[name] - triggertimes_snippets[name][0],
                             tt_min, tt_max, color='k', zorder=-100, lw=0.5, alpha=0.5,
                             label='trigger' if name == names[0] else '_')

        axs['all'].set(xlim=xlim)
        plt.tight_layout()
        axs['all'].legend(bbox_to_anchor=(1, 1), loc='upper left')

        return fig, axs
