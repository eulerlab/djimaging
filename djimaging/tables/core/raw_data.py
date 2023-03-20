import datajoint as dj


class RawDataParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        traces_params_id:       int       # unique param set id
        ---
        include_artifacts: tinyint unsigned  # automatically exclude all ROIs with artifacts?
        compute_from_stack: tinyint unsigned  # Compute traces from stack. Otherwise try to import Igor traces.
        trace_precision: enum("line", "pixel")  # Compute traces with either line precision or pixel precision?
        trigger_precision: enum("line", "pixel")   # Compute triggers with either line precision or pixel precision?
        """
        return definition

    def add_default(self, traces_params_id=1, include_artifacts=True, compute_from_stack=True,
                    trace_precision='line', trigger_precision='line', skip_duplicates=True):
        """Add default preprocess parameter to table"""
        key = dict(
            traces_params_id=traces_params_id,
            include_artifacts=int(include_artifacts),
            compute_from_stack=int(compute_from_stack),
            trace_precision=trace_precision,
            trigger_precision=trigger_precision,
        )
        self.insert1(key, skip_duplicates=skip_duplicates)
