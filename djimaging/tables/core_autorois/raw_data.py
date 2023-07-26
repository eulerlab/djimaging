from abc import abstractmethod

import datajoint as dj


class RawDataParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.userinfo_table
        raw_id:       int       # unique param set id
        ---
        from_raw_data: tinyint unsigned  # Load raw smp data (1) or h5 data (0)
        compute_from_stack: tinyint unsigned  # Compute traces from stack. Otherwise try to import Igor traces.
        include_artifacts: tinyint unsigned  # automatically exclude all ROIs with artifacts?
        trace_precision: enum("line", "pixel")  # Compute traces with either line precision or pixel precision?
        trigger_precision: enum("line", "pixel")   # Compute triggers with either line precision or pixel precision?
        igor_roi_masks: enum("yes", "init", "no")  # Either load or ignore existing ROI masks, e.g. from Igor 
        """
        return definition

    @property
    @abstractmethod
    def userinfo_table(self):
        pass

    def add_default(self, raw_id=1, experimenter_list=None,
                    from_raw_data=False, compute_from_stack=True, include_artifacts=False,
                    trace_precision='line', trigger_precision='line', igor_roi_masks="init", skip_duplicates=True):
        """Add default preprocess parameter to table"""
        if experimenter_list is None:
            experimenter_list = self.userinfo_table.fetch('experimenter')

        for experimenter in experimenter_list:
            key = dict(
                experimenter=experimenter,
                raw_id=raw_id,
                from_raw_data=int(from_raw_data),
                include_artifacts=int(include_artifacts),
                compute_from_stack=int(compute_from_stack),
                trace_precision=trace_precision,
                trigger_precision=trigger_precision,
                igor_roi_masks=igor_roi_masks,
            )
            self.insert1(key, skip_duplicates=skip_duplicates)
