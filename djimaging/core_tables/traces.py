import numpy as np
import datajoint as dj
from copy import deepcopy
import h5py
import tqdm
from scipy import signal


class Traces(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Raw Traces for each roi under a specific presentation
    
        -> self.field_table
        -> self.presentation_table
        -> self.roi_table
    
        ---
        traces                  :longblob              # array of raw traces
        traces_times            :longblob              # numerical array of trace times
        traces_flag             :tinyint unsigned      # flag if values in traces are correct(1) or not(0)
        trigger_flag = 1        :tinyint unsigned      # flag if triggertimes aren't outside tracetimes
        """
        return definition

    presentation_table = None
    field_table = None
    roi_table = None

    @property
    def key_source(self):
        return self.presentation_table & (self.field_table & 'loc_flag=0 & od_flag=0')

    def make(self, key):

        # get all params we need for creating traces
        filepath = (self.presentation_table & key).fetch1("h5_header")
        triggertimes, rois = (self.presentation_table * self.roi_table & key).fetch("triggertimes", "roi_id")

        with h5py.File(filepath, "r", driver="stdio") as h5_file:

            # read all traces and their times from file
            if "Traces0_raw" in h5_file.keys() and "Tracetimes0" in h5_file.keys():
                traces = h5_file["Traces0_raw"][()]
                traces_times = h5_file["Tracetimes0"][()]
                traces_dim = len(traces.shape)
            else:
                # set dim = -1 if traces couldn't be read
                traces_dim = -1

            for roi_id in tqdm.tqdm(rois, desc=f"{filepath}"):
                idx = roi_id - 1

                # Defautl values
                out_of_bounds = False  # if idx not valid skip file without of bound error
                trigger_flag = 1
                trace_flag = 0
                trace = np.zeros(0)
                trace_times = np.zeros(0)

                # copy traces and their times to key if their dimension is valid...
                # TODO: Check, why can this be out of range?!
                if traces_dim == 3:
                    if len(traces[0, 0, :]) <= idx:
                        out_of_bounds = True  # if idx not valid skip file without of bound error
                        trace_flag = 0
                    else:
                        trace = traces[:, :, idx]
                        trace_times = traces_times[:, :, idx]
                        trace_flag = 1
                elif traces_dim == 2:
                    if len(traces[0, :]) <= idx:
                        out_of_bounds = True  # if idx not valid skip file without of bound error
                        trace_flag = 0
                    else:
                        trace = traces[:, idx]
                        trace_times = traces_times[:, idx]
                        trace_flag = 1

                if not out_of_bounds:
                    if triggertimes[0] < trace_times[0]:
                        trigger_flag = 0
                        print(f'WARNING: Stimulus onset is before trace onset!\n{key}')

                    if triggertimes[-1] > trace_times[-1]:
                        trigger_flag = 0
                        print(f'WARNING: Stimulus offset is after trace offset!\n{key}')

                self.insert1(dict(
                    **key, trigger_flag=trigger_flag, traces_flag=trace_flag,
                    traces=trace, traces_times=trace_times, roi_id=roi_id))


class DetrendParams(dj.Lookup):
    # TODO: Add default set

    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        detrend_param_set_id:           int         #unique param set id
        ---
        window_length=60:               int       #window length for SavGol filter in seconds
        poly_order=3:                   int         # order of polynomial for savgol filter
        non_negative=0:                 tinyint unsigned
        subtract_baseline=0:            tinyint unsigned
        standardize=1:                  tinyint unsigned  # whether to standardize (divide by sd)
        """
        return definition


class DetrendTraces(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # performs basic preprocessing on raw traces
        -> self.detrendparams_table
        -> self.traces_table
        ---
        detrend_traces:         longblob        #detrended traces
        smoothed_traces:        longblob        #output of savgol filter which is
                                                #subtracted from the raw traces to
                                                #yield detrended traces
        """
        return definition

    presentation_table = None
    detrendparams_table = None
    traces_table = None

    def make(self, key):

        window_len_seconds = (self.detrendparams_table & key).fetch1('window_length')
        poly_order = (self.detrendparams_table & key).fetch1('poly_order')
        subtract_baseline = (self.detrendparams_table & key).fetch1('subtract_baseline')
        non_negative = (self.detrendparams_table & key).fetch1('non_negative')
        standardize = (self.detrendparams_table & key).fetch1('standardize')
        fs = (self.presentation_table & key).fetch1('scan_frequency')

        assert not (non_negative and subtract_baseline), \
            "You are trying to populate DetrendTraces with an invalid parameter set"
        assert (np.logical_or(standardize == non_negative, standardize == subtract_baseline)), \
            "You are trying to populate DetrendTraces with an invalid parameter set"

        raw_traces = (self.traces_table & key).fetch1('traces')
        temp = deepcopy(raw_traces)  # TODO: what is happening here?
        temp[0] = temp[1]
        raw_traces = temp
        traces_times = (self.traces_table & key).fetch1('traces_times')
        window_len_frames = np.ceil(window_len_seconds * fs)
        if window_len_frames % 2 == 0:
            window_len_frames -= 1
        window_len_frames = int(window_len_frames)
        smoothed_traces = \
            signal.savgol_filter(raw_traces, window_length=window_len_frames, polyorder=poly_order)
        detrend_traces = raw_traces - smoothed_traces

        stim_start = None
        if standardize or subtract_baseline:
            stim_start = (self.presentation_table & key).fetch1('triggertimes')[0]
            # heuristic to find out whether triggers are in time base or in frame base
            if stim_start > 1000:
                print("Converting triggers from frame base to time base")
                stim_start /= 500

            assert np.any(traces_times < stim_start), \
                f"stim_start={stim_start:.1g}, traces_starts at {traces_times.min():.1g}: key={key}"

        if non_negative:
            clip_value = np.percentile(detrend_traces, q=2.5)
            detrend_traces[detrend_traces < clip_value] = clip_value
            detrend_traces = detrend_traces - clip_value
            if standardize:
                # find last frame recorded before stimulus started
                baseline_end = np.nonzero(traces_times[traces_times < stim_start])[0][-1]
                baseline = detrend_traces[:baseline_end]
                detrend_traces = detrend_traces / np.std(baseline)
        elif subtract_baseline:
            # find last frame recorded before stimulus started
            baseline_end = np.nonzero(traces_times[traces_times < stim_start])[0][-1]
            baseline = detrend_traces[:baseline_end]
            detrend_traces = detrend_traces - np.median(baseline)

            if standardize:
                detrend_traces = detrend_traces / np.std(baseline)

        self.insert1(dict(key, detrend_traces=detrend_traces, smoothed_traces=smoothed_traces))


class DetrendSnippets(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Snippets created from slicing filtered traces using the triggertimes. 
    
        -> self.stimulus_table
        -> self.presentation_table
        -> self.traces_table
        -> self.detrendtraces_table
        ---
        detrend_snippets             :longblob     # array of snippets (time x repetitions)
        detrend_snippets_times       :longblob     # array of snippet times (time x repetitions)
        smoothed_snippets            :longblob     # snippeted, smoothed signal (time x repetitions)
        triggertimes_snippets        :longblob     # snippeted triggertimes (ntrigger_rep x repetitions)
        droppedlastrep_flag          :tinyint unsigned 
        """
        return definition

    stimulus_table = None
    presentation_table = None
    traces_table = None
    detrendtraces_table = None

    @property
    def key_source(self):
        return super().key_source * (self.stimulus_table & "isrepeated=1")

    def make(self, key):
        ntrigger_rep = (self.stimulus_table & key).fetch1('ntrigger_rep')
        triggertimes = (self.presentation_table & key).fetch1('triggertimes')
        traces_times = (self.traces_table & key).fetch1('traces_times')
        detrend_traces = (self.detrendtraces_table & key).fetch1('detrend_traces')
        smoothed_traces = (self.detrendtraces_table & key).fetch1('smoothed_traces')

        if triggertimes[-1] > 2 * traces_times[-1]:
            triggertimes = triggertimes / 500.

        t_idxs = [np.argwhere(np.isclose(traces_times, t, atol=1e-01))[0][0] for t in triggertimes[::ntrigger_rep]]

        if len(t_idxs) < 2:
            print("Failed to populate Snippets for ", key)
            return

        n_frames_per_rep = int(np.round(np.mean(np.diff(t_idxs))))

        if detrend_traces[t_idxs[-1]:].size < n_frames_per_rep:
            # if there are not enough data points after the last trigger,
            # remove the last trigger (e.g. if a chirp was cancelled)
            droppedlastrep_flag = 1
            t_idxs.pop(-1)
        else:
            droppedlastrep_flag = 0

        detrend_snippets = np.zeros((n_frames_per_rep, len(t_idxs)))
        smoothed_snippets = np.zeros((n_frames_per_rep, len(t_idxs)))
        detrend_snippets_times = np.zeros((n_frames_per_rep, len(t_idxs)))
        triggertimes_snippets = np.zeros((ntrigger_rep, len(t_idxs)))

        for i, idx in enumerate(t_idxs):
            # Frames may be reused, this is not a standard reshaping
            detrend_snippets[:, i] = detrend_traces[idx:idx + n_frames_per_rep]
            detrend_snippets_times[:, i] = traces_times[idx:idx + n_frames_per_rep]
            smoothed_snippets[:, i] = smoothed_traces[idx:idx + n_frames_per_rep]
            triggertimes_snippets[:, i] = triggertimes[i * ntrigger_rep:(i + 1) * ntrigger_rep]

        self.insert1(dict(
            **key,
            detrend_snippets=detrend_snippets,
            smoothed_snippets=smoothed_snippets,
            detrend_snippets_times=detrend_snippets_times,
            triggertimes_snippets=triggertimes_snippets,
            droppedlastrep_flag=droppedlastrep_flag,
        ))


class Averages(dj.Computed):
    database = ""  # hack to suppress DJ error

    @property
    def definition(self):
        definition = """
        # Snippets created from slicing filtered traces using the triggertimes. 
    
        -> self.detrendsnippets_table
        ---
        average             :longblob  # array of snippet average (time)
        average_norm        :longblob  # normalized array of snippet average (time)
        average_times       :longblob  # array of average time, starting at t=0 (time)
        triggertimes_rel    :longblob  # array of relative triggertimes 
        """
        return definition

    detrendsnippets_table = None

    def make(self, key):
        snippets, times = (self.detrendsnippets_table & key).fetch1('detrend_snippets', 'detrend_snippets_times')
        triggertimes_snippets = (self.detrendsnippets_table & key).fetch1('triggertimes_snippets').copy()

        times = times - times[0, :]

        if np.any(np.std(times, axis=1) > 1e-4):
            print(f'failed to compute average for {key}, tracetimes cannot be aligned without interpolation')
            return

        average_times = np.mean(times, axis=1)
        average = np.mean(snippets, axis=1)
        average_norm = (average - np.mean(average)) / np.std(average)
        triggertimes_rel = np.mean(triggertimes_snippets - triggertimes_snippets[0, :], axis=1)

        self.insert1(dict(
            **key,
            average=average,
            average_norm=average_norm,
            average_times=average_times,
            triggertimes_rel=triggertimes_rel,
        ))
