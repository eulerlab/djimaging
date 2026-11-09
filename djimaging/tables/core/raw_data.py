from __future__ import annotations

import warnings
from abc import abstractmethod

import datajoint as dj


class RawDataParamsTemplate(dj.Lookup):
    database = ""

    @property
    def definition(self):
        definition = """
        -> self.userinfo_table
        raw_id: int32  # unique param set id
        ---
        from_raw_data: bool  # Load raw smp data (1) or h5 data (0)
        compute_from_stack: bool  # Compute traces from stack. Otherwise try to import Igor traces.
        include_artifacts: bool  # automatically exclude all ROIs with artifacts?
        trace_precision: enum('line', 'pixel')  # Compute traces with either line precision or pixel precision?
        trigger_precision: enum('line', 'pixel')   # Compute triggers with either line precision or pixel precision?
        igor_roi_masks: enum('yes', 'init', 'no', 'either')  # Load or ignore existing ROI masks, e.g. from Igor; 'either': per field, mask file if available, else Igor
        """
        return definition

    @property
    @abstractmethod
    def userinfo_table(self) -> dj.Table:
        """Return the user info table."""
        pass

    def add_params(
            self,
            from_raw_data: bool,
            igor_roi_masks: str,
            raw_id: int = 1,
            experimenter_list: list = None,
            compute_from_stack: bool = True,
            include_artifacts: bool = False,
            trace_precision: str = 'line',
            trigger_precision: str = 'line',
            skip_duplicates: bool = True,
    ) -> None:
        """Add raw-data parameters for all (or specified) experimenters.

        `from_raw_data` and `igor_roi_masks` depend on how your data was recorded and processed,
        there is no sensible default, so they have to be set explicitly.

        Args:
            from_raw_data: If True, load raw ScanM data (.smp files) instead of h5 files.
            igor_roi_masks: How to handle ROI masks drawn in Igor and stored in the h5 files: 'yes' loads them,
                'no' ignores them (masks come from mask files only), 'init' uses them only as starting mask in the
                ROI canvas, 'either' decides per field: ROI mask file if one exists, else the Igor mask.
            raw_id: Unique identifier for this parameter set. Default is 1.
            experimenter_list: List of experimenter names to add parameters
                for. If None, all experimenters in the userinfo table are used.
            compute_from_stack: If True, compute traces from the imaging stack.
                Otherwise, try to import traces from Igor. Default is True.
            include_artifacts: If True, include ROIs that overlap with the
                light-artifact region. Default is False.
            trace_precision: Precision mode for trace computation, either
                'line' or 'pixel'. Default is 'line'.
            trigger_precision: Precision mode for trigger computation, either
                'line' or 'pixel'. Default is 'line'.
            skip_duplicates: If True, silently skip duplicate entries.
                Default is True.
        """
        if igor_roi_masks not in ('yes', 'init', 'no', 'either'):
            raise ValueError(f"igor_roi_masks must be 'yes', 'init', 'no' or 'either', got {igor_roi_masks!r}.")

        if experimenter_list is None:
            experimenter_list = self.userinfo_table.to_arrays('experimenter')

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
        """Deprecated, use `add_params` and set `from_raw_data` and `igor_roi_masks` explicitly."""
        warnings.warn(
            "RawDataParams.add_default() is deprecated, the defaults (from_raw_data=False, igor_roi_masks='yes') "
            "often do not fit the data. Use add_params(from_raw_data=..., igor_roi_masks=...) instead.",
            DeprecationWarning, stacklevel=2)
        self.add_params(
            from_raw_data=from_raw_data, igor_roi_masks=igor_roi_masks, raw_id=raw_id,
            experimenter_list=experimenter_list, compute_from_stack=compute_from_stack,
            include_artifacts=include_artifacts, trace_precision=trace_precision,
            trigger_precision=trigger_precision, skip_duplicates=skip_duplicates)
