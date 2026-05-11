import datajoint as dj

from djimaging.utils.qdspy.read_logs import parse_stimulus_log


class QdsPyLogFileTemplate(dj.Manual):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.exp_table
        log_idx : tinyint
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
        log_idxs, log_paths = (self & key).fetch('log_idx', 'log_path')
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
        n_lines_total : int
        n_lines_data  : int
        n_lines_err   : int
        n_err         : int
        """
        return definition

    class StimLog(dj.Part):
        @property
        def definition(self):
            definition = """
            -> master
            stim_idx            : int
            ---
            stim_file_name   : varchar(255)
            stim_path        : varchar(1024)
            stim_md5         : char(32)
            t_abs_s          : float
            t_since_last_s   : float
            t_start          : time
            t_end            : time
            aborted          : bool
            t_dur_s          : float
            t_dur_s_calc     : float
            n_dropped_frames : int
            params           : blob
            sequence_used    = null : int
            other_info       : blob
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


class PresentationLogTemplate(dj.Computed):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.presentation_table
        ---
        log_idx      : tinyint
        stim_idx     : int
        match_method : varchar(32)
        t_start      : time
        t_end        : time
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

    def make(self, key, verbose: int = 0):
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

        log_rows = (self.log_table & exp_key & 'aborted = 0').fetch(
            as_dict=True, order_by='log_idx, stim_idx'
        )

        if verbose >= 2:
            print(f"  non-aborted log rows for experiment: {len(log_rows)}")
        if verbose >= 3:
            for r in log_rows:
                print(f"    log_idx={r['log_idx']} stim_idx={r['stim_idx']} "
                      f"name={r['stim_file_name']!r} md5={r['stim_md5']!r}")

        matches = [r for r in log_rows if r['stim_md5'] in hashes]
        method = 'md5'

        if not matches:
            if verbose >= 2:
                print("  no MD5 match — falling back to filename matching")
            matches = [r for r in log_rows if r['stim_file_name'] in names]
            method = 'filename'

        if verbose >= 2:
            print(f"  match_method={method!r}, {len(matches)} candidate(s) after filtering")

        if not matches:
            raise ValueError(
                f"No log entry found for key={key}. "
                f"Checked hashes={hashes}, names={names}."
            )

        stim_pk = self.stimulus_table.primary_key
        stim_filter = {k: key[k] for k in stim_pk if k in key}
        all_pres_keys = sorted(
            (self.presentation_table & exp_key & stim_filter).fetch('KEY'),
            key=lambda k: tuple(v for _, v in sorted(k.items()))
        )
        pres_idx = next(
            (i for i, pk in enumerate(all_pres_keys)
             if all(key.get(k) == pk.get(k) for k in pk)),
            None
        )

        if verbose >= 2:
            print(f"  sibling presentations for this stimulus: {len(all_pres_keys)}, "
                  f"this presentation ordinal: {pres_idx}")
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
            print(f"  -> matched log_idx={matched['log_idx']} stim_idx={matched['stim_idx']} "
                  f"method={method!r} t_start={matched['t_start']} t_end={matched['t_end']}")

        self.insert1({
            **key,
            'log_idx'     : matched['log_idx'],
            'stim_idx'    : matched['stim_idx'],
            'match_method': method,
            't_start'     : matched['t_start'],
            't_end'       : matched['t_end'],
        })