"""Read-only recording inspector: play back a presentation with its ROI mask and synchronised traces.

- ``LocalFieldSource``: local ScanM files + saved ROI masks, no database (``djimaging.inspector.offline``).
- ``DataJointFieldSource`` / ``open_recording_inspector``: stored pipeline results, read-only
  (``djimaging.inspector.dj_adapter``).
- ``RecordingInspector``: the notebook viewer (needs ``ipywidgets``; clicking ROIs needs ``ipympl``).
- ``export_mp4``: video export (needs ``imageio-ffmpeg`` or ffmpeg on the PATH).

Names are imported lazily so that ``djimaging.inspector.offline`` works without the optional widget packages.
"""
_EXPORTS = {
    'LocalFieldSource': 'djimaging.inspector.offline',
    'DataJointFieldSource': 'djimaging.inspector.dj_adapter',
    'InspectorTables': 'djimaging.inspector.dj_adapter',
    'RecordingInspector': 'djimaging.inspector.viewer',
    'open_recording_inspector': 'djimaging.inspector.viewer',
    'export_mp4': 'djimaging.inspector.export',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
