import datajoint as dj

from djimaging.utils.exp_history_utils import ref_path, seconds_of_day
from djimaging.utils.qdspy.read_logs import parse_stimulus_log


def _parse_scan_start_time(scan_params_dict: dict):
    """Extract recording start time from a ScanM scan_params_dict.

    Returns a datetime.time on success, or None if the dict lacks the required
    keys or none of the known formats match.
    """
    from datetime import datetime as _dt

    def _decode(v):
        return v.decode('utf-8') if hasattr(v, 'decode') else str(v)

    date_str = None
    for k in ('datestamp_d_m_y', 'datestamp_y_m_d', 'date', 'datestamp'):
        if scan_params_dict.get(k) is not None:
            date_str = _decode(scan_params_dict[k])
            break

    time_str = None
    for k in ('timestamp_h_m_s_ms', 'timestamp', 'time'):
        if scan_params_dict.get(k) is not None:
            time_str = _decode(scan_params_dict[k])
            break

    if date_str is None or time_str is None:
        return None

    for fmt in (
        '%Y-%m-%d %H-%M-%S-%f',
        '%Y-%m-%d %H:%M:%S.%f',
        '%Y-%m-%d %H:%M:%S',
        '%d-%m-%Y %H-%M-%S-%f',
        '%d-%m-%Y %H:%M:%S.%f',
        '%d-%m-%Y %H:%M:%S',
    ):
        try:
            return _dt.strptime(f'{date_str} {time_str}', fmt).time()
        except ValueError:
            pass
    return None


class QdsPyLogFileTemplate(dj.Manual):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.exp_table
        log_idx : int32
        ---
        log_path        : varchar(1024)
        """
        return definition

    @property
    def exp_table(self):
        """Override this property to specify the experiment table."""
        raise NotImplementedError("Subclasses must implement the exp_table property.")

    def add1(self, key, log_path: str, **kwargs):
        """Add a new log file entry.

        Args:
            key: Key identifying the experiment.
            log_path: Path to the log file.
        """
        log_idxs, log_paths = (self & key).to_arrays('log_idx', 'log_path')
        if len(log_idxs) == 0:
            log_idx = 0
        else:
            if log_path in log_paths:
                raise ValueError(f'Log path {log_path} already exists for this experiment.')
            log_idx = max(log_idxs) + 1

        row = key.copy()
        row.update({
            'log_idx': log_idx,
            'log_path': log_path
        })

        self.insert1(row, **kwargs)


class QdsPyLogTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.log_file_table
        ---
        n_lines_total : int32
        n_lines_data  : int32
        n_lines_err   : int32
        n_err         : int32
        """
        return definition

    class StimLog(dj.Part):
        @property
        def definition(self):
            definition = """
            -> master
            stim_idx            : int32           # stimulus index
            ---
            stim_file_name   : varchar(255)
            stim_path        : varchar(1024)
            stim_md5         : char(32)

            t_abs_s          : float32           # absolute time (s)
            t_since_last_s   : float32           # time since previous stim (s)

            t_start          : time
            t_end            : time
            aborted          : bool
            t_dur_s          : float32
            t_dur_s_calc     : float32

            n_dropped_frames : int32

            params           : <blob>
            sequence_used    = null : int32
            other_info       : <blob>          # additional info with new keys
            """
            return definition

    @property
    def log_file_table(self):
        """Override this property to specify the experiment table."""
        raise NotImplementedError("Subclasses must implement the experiment_table property.")

    def make(self, key, root_dir=None, verbose: bool = False):
        log_path = (self.log_file_table & key).fetch1('log_path')
        if root_dir is not None:
            import os
            log_path = os.path.join(root_dir, log_path)

        stims, stats = parse_stimulus_log(log_path=log_path, verbose=verbose)

        row = key.copy()
        row.update({
            'n_lines_total': stats.get('nLinesTotal', -1),
            'n_lines_data': stats.get('nLinesData', -1),
            'n_lines_err': stats.get('nLinesErr', -1),
            'n_err': stats.get('nErr', -1),
        })

        _KNOWN_STIM_KEYS = {
            'index', 'stimFileName', 'stimPath', 'stimMD5',
            't_abs_s', 't_since_last_s', 't_start', 't_end',
            'aborted', 't_dur_s', 't_dur_s_calc', 'nDroppedFrames',
            'params', 'sequenceUsed',
        }

        stim_rows = []
        for stim in stims:
            stim_row = key.copy()
            stim_row.update({
                'stim_idx': stim['index'],
                'stim_file_name': stim['stimFileName'],
                'stim_path': stim['stimPath'],
                'stim_md5': stim['stimMD5'],
                't_abs_s': stim['t_abs_s'],
                't_since_last_s': stim['t_since_last_s'],
                't_start': stim['t_start'],
                't_end': stim['t_end'],
                'aborted': stim['aborted'],
                't_dur_s': stim.get('t_dur_s', -1),
                't_dur_s_calc': stim.get('t_dur_s_calc', -1),
                'n_dropped_frames': stim.get('nDroppedFrames', -1),
                'params': stim.get('params', dict()),
                'sequence_used': stim.get('sequenceUsed', None),
                'other_info': {k: v for k, v in stim.items() if k not in _KNOWN_STIM_KEYS},
            })
            stim_rows.append(stim_row)

        self.insert1(row)
        self.StimLog().insert(stim_rows)


def _exp_history_info(self, key: dict, exp_key: dict) -> dict | None:
    """Recording of a presentation and clock offset from `self.exp_history_stim_log_table` of PresentationLog.

    None if the table is not bound or not populated for the experiment.

    Returns dict(clock_offset_s, rec_name, rec_start_s, in_recording), `in_recording` the `(log_idx, stim_idx)`
    of the log entries that started during the recording.
    """
    table = getattr(self, 'exp_history_stim_log_table', None)
    if table is None or len(table & exp_key) == 0:
        return None
    clock_offset_s = (table & exp_key).fetch1('clock_offset_s')
    out = dict(clock_offset_s=clock_offset_s, rec_name=None, rec_start_s=None, in_recording=set())
    pres_path = ref_path((self.presentation_table & key).fetch1('pres_data_file'))
    recs = (table.exp_history_table.Recording & exp_key).proj('h5_file', 'smp_file', 'rec_start').to_dicts()
    rec = next((r for r in recs if pres_path is not None
                and pres_path in (ref_path(r['h5_file']), ref_path(r['smp_file']))), None)
    if rec is None:
        return out
    out.update(rec_name=rec['rec_name'], rec_start_s=seconds_of_day(rec['rec_start']))
    stims = (table.Stim & exp_key & dict(rec_name=rec['rec_name'], status='recording')).to_dicts()
    out['in_recording'] = {(s['log_idx'], s['stim_idx']) for s in stims}
    return out


class PresentationLogTemplate(dj.Computed):
    """The QDSpy log entry of each Presentation.

    Log entries are filtered by stimulus (md5, else file name; aborted entries only if no other matches). If several
    remain, the entry is selected by, in this order:

    1. `+recording`: the entry that started during the recording of the presentation, from `ExpHistoryStimLog`
       (if `exp_history_stim_log_table` is bound and populated), which estimated the clock offset between the QDSpy
       and the ScanM computer from all recordings and stimuli of the experiment. Several: not aborted first, then
       the last one (earlier ones are usually test runs).
    2. `+time`: the entry that started closest to the recording start (from `scan_params_table`, else from
       `ExpHistory.Recording`), after shifting the log times by the clock offset of `ExpHistoryStimLog`
       (`+time+offset`; without it the clocks are assumed to agree).
    3. ordinal: the n-th presentation of the stimulus (sorted by key) gets the n-th entry.

    If `exp_history_stim_log_table` is bound but not populated for the experiment, `make` raises when several entries
    match, unless `allow_missing_exp_history=True` is passed (then 2. and 3. apply without the clock offset).
    """
    database = ""
    # Optional: ExpHistoryStimLog (history block), see the class docstring. Populate it before this table.
    exp_history_stim_log_table = None

    @property
    def definition(self):
        definition = """
        -> self.presentation_table
        ---
        log_idx      : int8
        stim_idx     : int32
        match_method : varchar(32)
        t_start      : time
        t_end        : time
        aborted      : bool
        """
        return definition

    @property
    def exp_table(self):
        """Override to specify the experiment table (used in make() to scope log queries)."""
        raise NotImplementedError("Subclasses must implement the exp_table property.")

    @property
    def presentation_table(self):
        """Override to specify the presentation table."""
        raise NotImplementedError("Subclasses must implement the presentation_table property.")

    @property
    def log_table(self):
        """Override to specify the log table (a QdsPyLogTemplate.StimLog instance)."""
        raise NotImplementedError("Subclasses must implement the log_table property.")

    @property
    def stimulus_table(self):
        """Override to specify the stimulus table (must expose stim_hash and stim_dict)."""
        raise NotImplementedError("Subclasses must implement the stimulus_table property.")

    @property
    def scan_params_table(self):
        """Override to enable time-based matching.

        Should return a table (e.g. Presentation.ScanInfo) that has a
        scan_params_dict longblob field fetchable by the presentation key.
        Return None (default) to skip time matching and use ordinal matching only.
        """
        return None

    def make(self, key, verbose: int = 0, skip_if_no_log: bool = False, allow_missing_exp_history: bool = False):
        """
        :param skip_if_no_log: Skip presentations without any matching log entry instead of raising.
        :param allow_missing_exp_history: If `exp_history_stim_log_table` is bound but not populated for the
            experiment, and several log entries match, raise unless True (then time or ordinal matching is used).
        """
        if verbose >= 1:
            print(f"[PresentationLog] Processing key: {key}")

        exp_key = (self.exp_table & key).fetch1('KEY')

        stim_hash, stim_dict = (self.stimulus_table & key).fetch1('stim_hash', 'stim_dict')
        hashes = {stim_hash} | set(stim_dict.get('stim_hash_list', []))
        hashes.discard('')
        names = set(stim_dict.get('stim_name_list', []))
        if 'stim_name' in key:
            names.add(key['stim_name'])

        if verbose >= 2:
            print(f"  stim_hash={stim_hash!r}")
            print(f"  candidate hashes : {hashes}")
            print(f"  candidate names  : {names}")

        def _fetch_log_rows(include_aborted: bool):
            q = self.log_table & exp_key
            if not include_aborted:
                q = q & 'aborted = 0'
            return q.to_dicts(order_by=('log_idx', 'stim_idx'))

        def _filter_rows(rows):
            names_lower = {n.lower() for n in names}
            by_md5 = [r for r in rows if r['stim_md5'] in hashes]
            if by_md5:
                return by_md5, 'md5'
            by_name = [
                r for r in rows
                if any(r['stim_file_name'].lower().startswith(n) for n in names_lower)
            ]
            return by_name, 'filename'

        log_rows = _fetch_log_rows(include_aborted=False)

        if verbose >= 2:
            print(f"  non-aborted log rows for experiment: {len(log_rows)}")
        if verbose >= 3:
            for r in log_rows:
                print(f"    log_idx={r['log_idx']} stim_idx={r['stim_idx']} "
                      f"name={r['stim_file_name']!r} md5={r['stim_md5']!r}")

        matches, method = _filter_rows(log_rows)

        if not matches:
            if verbose >= 2:
                print("  no match in non-aborted rows — retrying with aborted entries included")
            log_rows = _fetch_log_rows(include_aborted=True)
            if not log_rows:
                if skip_if_no_log:
                    if verbose >= 1:
                        print("  no log entries for this experiment — skipping")
                    return
                raise ValueError(f"No log entries at all for experiment key={exp_key}.")
            if verbose >= 3:
                aborted_rows = [r for r in log_rows if r['aborted']]
                for r in aborted_rows:
                    print(f"    [aborted] log_idx={r['log_idx']} stim_idx={r['stim_idx']} "
                          f"name={r['stim_file_name']!r} md5={r['stim_md5']!r}")
            matches, method = _filter_rows(log_rows)

        if verbose >= 2:
            print(f"  match_method={method!r}, {len(matches)} candidate(s) after filtering "
                  f"({sum(r['aborted'] for r in matches)} aborted)")

        if not matches:
            raise ValueError(
                f"No log entry found for key={key}. "
                f"Checked hashes={hashes}, names={names}."
            )

        # --- selection: time-based (preferred) or ordinal (fallback) ---

        matched = None

        if len(matches) == 1:
            matched = matches[0]

        hist = _exp_history_info(self, key, exp_key) if matched is None else None
        if matched is None and hist is None and not allow_missing_exp_history                 and getattr(self, 'exp_history_stim_log_table', None) is not None:
            raise ValueError(
                f"{self.exp_history_stim_log_table.__class__.__name__} is not populated for {exp_key} but is needed "
                f"to select among {len(matches)} matching log entries of {key}. Populate it first, or pass "
                f"make_kwargs=dict(allow_missing_exp_history=True) to fall back to time/ordinal matching.")
        if verbose >= 2 and hist is not None:
            print(f"  ExpHistoryStimLog: recording={hist['rec_name']!r}, clock_offset_s={hist['clock_offset_s']}, "
                  f"{len(hist['in_recording'])} log entr(ies) started during the recording")

        if matched is None and hist is not None:
            in_rec = [r for r in matches if (r['log_idx'], r['stim_idx']) in hist['in_recording']]
            if in_rec:
                # Not aborted first, then the last one (earlier ones are usually test runs)
                matched = max(in_rec, key=lambda r: (not r['aborted'], seconds_of_day(r['t_start'])))
                method = method + '+recording'

        if matched is None:
            rec_s = None
            scan_params_table = self.scan_params_table
            if scan_params_table is not None:
                try:
                    rec_time = _parse_scan_start_time((scan_params_table & key).fetch1('scan_params_dict'))
                    rec_s = seconds_of_day(rec_time)
                    if rec_s is None and verbose >= 2:
                        print("  time-matching: could not parse time from scan_params_dict")
                except Exception as e:
                    if verbose >= 2:
                        print(f"  time-matching: no recording start from scan_params_table ({e})")
            if rec_s is None and hist is not None:
                rec_s = hist['rec_start_s']
            if rec_s is not None:
                offset = hist['clock_offset_s'] if hist is not None else None

                def _delta_s(row):
                    # QDSpy time + offset = ScanM time; MySQL TIME columns are fetched as timedelta
                    return abs(rec_s - (seconds_of_day(row['t_start']) + (offset or 0.)))

                matched = min(matches, key=_delta_s)
                method = method + '+time' + ('+offset' if offset is not None else '')
                if verbose >= 2:
                    print(f"  time-matching: recording start={rec_s:.0f} s of day, clock offset={offset}, "
                          f"best match t_start={matched['t_start']} (delta={_delta_s(matched):.0f}s)")
                if verbose >= 3:
                    for r in matches:
                        marker = "  <-- selected" if r is matched else ""
                        print(f"    stim_idx={r['stim_idx']} t_start={r['t_start']} "
                              f"delta={_delta_s(r):.0f}s{marker}")

        if matched is None:
            stim_pk = self.stimulus_table.primary_key
            stim_filter = {k: key[k] for k in stim_pk if k in key}
            all_pres_keys = sorted(
                (self.presentation_table & exp_key & stim_filter).proj().to_dicts(),
                key=lambda k: tuple(v for _, v in sorted(k.items()))
            )
            pres_idx = next(
                (i for i, pk in enumerate(all_pres_keys)
                 if all(key.get(k) == pk.get(k) for k in pk)),
                None
            )

            if verbose >= 2:
                print(f"  ordinal matching: {len(all_pres_keys)} sibling presentation(s), "
                      f"this ordinal={pres_idx}")
            if verbose >= 3:
                for i, pk in enumerate(all_pres_keys):
                    marker = "  <-- this" if i == pres_idx else ""
                    print(f"    [{i}] {pk}{marker}")

            if pres_idx is None:
                raise ValueError(f"Could not locate key={key} among sibling presentations.")
            if len(matches) <= pres_idx:
                raise ValueError(
                    f"Only {len(matches)} log entr(ies) found for "
                    f"hashes={hashes} / names={names}, but presentation ordinal is {pres_idx}."
                )
            matched = matches[pres_idx]

        if verbose >= 1:
            aborted_flag = " [ABORTED]" if matched['aborted'] else ""
            print(f"  -> matched log_idx={matched['log_idx']} stim_idx={matched['stim_idx']} "
                  f"method={method!r} t_start={matched['t_start']} t_end={matched['t_end']}"
                  f"{aborted_flag}")

        self.insert1({
            **key,
            'log_idx'     : matched['log_idx'],
            'stim_idx'    : matched['stim_idx'],
            'match_method': method,
            't_start'     : matched['t_start'],
            't_end'       : matched['t_end'],
            'aborted'     : matched['aborted'],
        })
