"""The one rule that decides whether a recording belongs to a *listed* stimulus.

A recording is **listed for stimulus S** if the stimulus part of its file name (position `stimulus_loc` of the
UserInfo, compared case-insensitively) equals one of the `_`-separated aliases of S in the Stimulus table of the
schema. A recording is **listed** if it is listed for at least one stimulus. Only `kind='response'` recordings can be
listed: high resolution stacks, optic disk and outline recordings have no stimulus.

Only listed recordings become Presentations, and everything with stimulus-specific data (Traces, PreprocessTraces,
Snippets, Averages and all analyses) depends on Presentation. Recordings of unlisted (e.g. unpublished) stimuli are
registered in ExpHistory, but never get traces.
"""
from __future__ import annotations

from collections.abc import Mapping


def split_alias(alias: str | None) -> set[str]:
    """Aliases of one stimulus: the `_`-separated, lower-cased parts of its `alias` string, without empty parts."""
    if alias is None:
        return set()
    return {a for a in str(alias).lower().split('_') if a}


def listed_stimuli(stim_token: str | None, aliases: Mapping[str, str]) -> list[str]:
    """Names of all stimuli a recording with stimulus part `stim_token` is listed for.

    Parameters
    ----------
    stim_token : str | None
        Stimulus part of the file name. None or empty: not listed.
    aliases : Mapping[str, str]
        `{stim_name: alias}` of the Stimulus table, e.g. `dict(zip(*Stimulus().to_arrays('stim_name', 'alias')))`.

    Returns
    -------
    list[str]
        Sorted stimulus names. Empty if the recording is not listed. More than one name means that the stimulus part
        is an alias of several stimuli (possible with `add_stimulus(unique_alias=False)`).
    """
    if stim_token is None:
        return []
    token = str(stim_token).strip().lower()
    if not token:
        return []
    return sorted(stim_name for stim_name, alias in aliases.items() if token in split_alias(alias))


def is_listed(stim_token: str | None, aliases: Mapping[str, str], kind: str = 'response') -> bool:
    """True if a recording with stimulus part `stim_token` and file kind `kind` is listed, see module docstring."""
    return kind == 'response' and len(listed_stimuli(stim_token, aliases)) > 0


def is_listed_for(stim_token: str | None, stim_name: str, alias: str, kind: str = 'response') -> bool:
    """True if a recording with stimulus part `stim_token` and file kind `kind` is listed for stimulus `stim_name`."""
    return is_listed(stim_token, {stim_name: alias}, kind=kind)
