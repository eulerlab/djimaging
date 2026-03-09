from __future__ import annotations

from abc import abstractmethod

import datajoint as dj


class RawDataParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.userinfo_table
        raw_id: tinyint unsigned  # unique param set id
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
    def userinfo_table(self) -> dj.Table:
        """Return the user info table."""
        pass

    def add_default(
            self,
            raw_id: int = 1,
            experimenter_list: list = None,
            from_raw_data: bool = False,
            compute_from_stack: bool = True,
            include_artifacts: bool = False,
            trace_precision: str = 'line',
            trigger_precision: str = 'line',
            igor_roi_masks: str = "yes",
            skip_duplicates: bool = True,
    ) -> None:
        """Add default raw-data parameters for all (or specified) experimenters.

        Args:
            raw_id: Unique identifier for this parameter set. Default is 1.
            experimenter_list: List of experimenter names to add parameters
                for. If None, all experimenters in the userinfo table are used.
            from_raw_data: If True, load raw ScanM data instead of h5 files.
                Default is False.
            compute_from_stack: If True, compute traces from the imaging stack.
                Otherwise, try to import traces from Igor. Default is True.
            include_artifacts: If True, include ROIs that overlap with the
                light-artifact region. Default is False.
            trace_precision: Precision mode for trace computation, either
                'line' or 'pixel'. Default is 'line'.
            trigger_precision: Precision mode for trigger computation, either
                'line' or 'pixel'. Default is 'line'.
            igor_roi_masks: How to handle Igor ROI masks: 'yes' loads them,
                'init' uses them only for initialisation, 'no' ignores them.
                Default is 'yes'.
            skip_duplicates: If True, silently skip duplicate entries.
                Default is True.
        """
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
