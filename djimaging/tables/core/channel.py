from __future__ import annotations

import datajoint as dj


class ChannelTemplate(dj.Lookup):
    """Imaging channels of the recordings, e.g. a functional channel (Ch0) and a marker channel (Ch1, e.g. SR-101).

    Maps the channel id stored in the primary key of downstream tables to the name of the data stack in the
    recording file. Used when ``UserInfoTemplate._use_channels = True``; the channels then replace
    ``data_stack_name`` and ``alt_stack_name`` of the user info.

    - ``is_functional``: Traces are only extracted for functional channels. Other channels (e.g. markers for blood
      vessels) are background channels, which are still shown in the ROI canvas and field plots.
    - ``roi_default``: Exactly one channel is the default target for ROI detection. It is preselected in the ROI canvas
      (cc tool, diagnostics, AutoROIs) and can be changed there.
    """
    database = ""

    @property
    def definition(self):
        definition = """
        # Imaging channel
        ch_id : int32  # channel id, used as primary key downstream
        ---
        ch_name : varchar(63)  # name of the data stack in the recording file, e.g. wDataCh0
        is_functional = 1 : bool  # traces are extracted for functional channels, others are background (e.g. SR-101)
        roi_default = 0 : bool  # default target channel for ROI detection in the ROI canvas, exactly one channel
        """
        return definition

    def add_channels(
            self,
            channels: list[tuple[str, int]],
            functional: list[str] | None = None,
            roi_channel: str | None = None,
            skip_duplicates: bool = False,
    ) -> None:
        """Add channels.

        Args:
            channels: List of (ch_name, ch_id) tuples, e.g. [('wDataCh0', 0), ('wDataCh1', 1)].
            functional: Names of the functional channels, e.g. ['wDataCh0']. All others are background channels.
                Defaults to all channels.
            roi_channel: Name of the channel that is the default target for ROI detection, e.g. 'wDataCh1'.
                Defaults to the first functional channel.
            skip_duplicates: Skip channels that already exist.
        """
        if len(channels) == 0:
            raise ValueError("No channels given")

        ch_names = [ch_name for ch_name, _ in channels]
        functional = ch_names if functional is None else list(functional)

        if len(functional) == 0:
            raise ValueError("At least one channel has to be functional")
        if any(ch_name not in ch_names for ch_name in functional):
            raise ValueError(f"functional={functional} are not all in the channels {ch_names}")

        if roi_channel is None:
            roi_channel = next(ch_name for ch_name in ch_names if ch_name in functional)
        elif roi_channel not in ch_names:
            raise ValueError(f"roi_channel={roi_channel!r} is not one of the channels {ch_names}")

        rows = [dict(ch_id=ch_id, ch_name=ch_name, is_functional=int(ch_name in functional),
                     roi_default=int(ch_name == roi_channel))
                for ch_name, ch_id in channels]
        self.insert(rows, skip_duplicates=skip_duplicates)

    def get_functional_channels(self) -> list[tuple[str, int]]:
        """(ch_name, ch_id) of the functional channels, ordered by channel id."""
        rows = sorted(self.to_dicts(), key=lambda row: row['ch_id'])
        channels = [(row['ch_name'], row['ch_id']) for row in rows if row['is_functional']]
        if len(channels) == 0:
            raise ValueError("channel_table has no functional channel")
        return channels

    def get_roi_channel(self) -> str:
        """Name of the default target channel for ROI detection.

        Falls back to the first functional channel if no channel is flagged as `roi_default`.
        """
        rows = sorted(self.to_dicts(), key=lambda row: row['ch_id'])
        if len(rows) == 0:
            raise ValueError("channel_table is empty")

        defaults = [row['ch_name'] for row in rows if row['roi_default']]
        if len(defaults) > 1:
            raise ValueError(f"Multiple default ROI channels: {defaults}. Set roi_default for exactly one channel.")
        return defaults[0] if len(defaults) == 1 else self.get_functional_channels()[0][0]

    def get_ch_names(self) -> list[str]:
        """Names of all channels with the default ROI channel first, then ordered by channel id."""
        ch_names = [row['ch_name'] for row in sorted(self.to_dicts(), key=lambda row: row['ch_id'])]
        roi_channel = self.get_roi_channel()
        return [roi_channel] + [ch_name for ch_name in ch_names if ch_name != roi_channel]
