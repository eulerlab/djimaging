from __future__ import annotations

import datajoint as dj


class ChannelTemplate(dj.Lookup):
    """Imaging channels that are extracted when multi-channel traces are enabled.

    Only needed when ``TracesTemplate._use_channels = True``. Maps the channel id stored in the
    primary key of the downstream tables to the name of the data stack in the recording file.
    """
    database = ""

    @property
    def definition(self):
        definition = """
        # Imaging channel
        ch_id : int32  # channel id, used as primary key downstream
        ---
        ch_name : varchar(63)  # name of the data stack in the recording file, e.g. wDataCh0
        """
        return definition

    def add_channels(self, channels: list[tuple[str, int]], skip_duplicates: bool = False) -> None:
        """Add channels.

        Args:
            channels: List of (ch_name, ch_id) tuples, e.g. [('wDataCh0', 0), ('wDataCh1', 1)].
            skip_duplicates: Skip channels that already exist.
        """
        rows = [dict(ch_id=ch_id, ch_name=ch_name) for ch_name, ch_id in channels]
        self.insert(rows, skip_duplicates=skip_duplicates)
