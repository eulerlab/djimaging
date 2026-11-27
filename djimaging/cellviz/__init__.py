"""CellViz: read-only viewer to play back a presentation with its ROI mask and synchronised traces.

In the pipeline, once ``RoiMask`` and ``Traces`` (and optionally ``PreprocessTraces``, ``ChirpQI``, ``OsDsIndexes``,
``HighRes``) are populated::

    from djimaging.cellviz import CellViz
    CellViz.list_fields(my_schema)                          # fields with traces
    CellViz.from_schema(my_schema, field_key=dict(field='GCL0', date='2026-09-16'))

- ``CellViz``: the notebook viewer (needs ``ipywidgets``; clicking ROIs needs ``ipympl``); ``CellViz.from_schema`` /
  ``open_cellviz`` open it on a field of an activated schema (``djimaging.cellviz.dj_adapter``).
- ``LocalFieldSource``: local ScanM files + saved ROI masks, no database (``djimaging.cellviz.offline``).
- ``DataJointFieldSource`` / ``CellVizTables``: stored pipeline results, read-only.
- ``export_mp4``: video export (needs ``imageio-ffmpeg`` or ffmpeg on the PATH), by default into
  ``<processed store>/cellviz/``.

Install the optional dependencies with ``uv pip install -e ".[cellviz]"``. Tutorial:
``djimaging/notebooks/tutorial_cellviz.ipynb``.

Names are imported lazily so that ``djimaging.cellviz.offline`` works without the optional widget packages.
"""
_EXPORTS = {
    'CellViz': 'djimaging.cellviz.viewer',
    'open_cellviz': 'djimaging.cellviz.viewer',
    'LocalFieldSource': 'djimaging.cellviz.offline',
    'DataJointFieldSource': 'djimaging.cellviz.dj_adapter',
    'CellVizTables': 'djimaging.cellviz.dj_adapter',
    'export_mp4': 'djimaging.cellviz.export',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
