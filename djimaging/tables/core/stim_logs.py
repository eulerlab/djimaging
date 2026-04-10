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
            # ROI Mask
            -> master
            stim_idx            : int           # stimulus index
            ---
            stim_file_name   : varchar(255)
            stim_path        : varchar(1024)
            stim_md5         : char(32)
        
            t_abs_s          : float           # absolute time (s)
            t_since_last_s   : float           # time since previous stim (s)
        
            t_start          : time
            t_end            : time
        
            aborted          : bool
            t_dur_s          : float
            t_dur_s_calc     : float
        
            n_dropped_frames : int
            
            params           : blob
            other_info      : blob          # additional info with new keys
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
                'other_info': {k: v for k, v in stim.items() if k not in [
                    'stimFileName', 'stimPath', 'stimMD5', 't_abs_s', 't_since_last_s',
                    't_start', 't_end', 'aborted', 't_dur_s', 't_dur_s_calc',
                    'nDroppedFrames', 'params'
                ]}
            })
            stim_rows.append(stim_row)

        self.insert1(row)
        self.StimLog().insert(stim_rows)
