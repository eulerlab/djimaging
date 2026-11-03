from __future__ import annotations

import datajoint as dj


class ChannelTemplate(dj.Lookup):
    """Imaging channels that are extracted when multi-channel traces are enabled.

    Only needed when ``TracesTemplate._use_channels = True``. Maps the channel id stored in the
    primary key of the downstream tables to the name of the data stack in the recording file.

    Exactly one channel is the default target for ROI detection (``roi_default``). It is preselected in the
    ROI canvas (cc tool, diagnostics, AutoROIs) and can be changed there. All channels can be viewed as background.
    """
    database = ""

    @property
    def definition(self):
        definition = """
        # Imaging channel
        ch_id : int32  # channel id, used as primary key downstream
        ---
        ch_name : varchar(63)  # name of the data stack in the recording file, e.g. wDataCh0
        roi_default = 0 : bool  # default target channel for ROI detection in the ROI canvas, exactly one channel
        """
        return definition

    def add_channels(
            self,
            channels: list[tuple[str, int]],
            roi_channel: str | None = None,
            skip_duplicates: bool = False,
    ) -> None:
        """Add channels.

        Args:
            channels: List of (ch_name, ch_id) tuples, e.g. [('wDataCh0', 0), ('wDataCh1', 1)].
            roi_channel: Name of the channel that is the default target for ROI detection, e.g. 'wDataCh1'.
                Defaults to the first channel in `channels`.
            skip_duplicates: Skip channels that already exist.
        """
        if len(channels) == 0:
            raise ValueError("No channels given")

        ch_names = [ch_name for ch_name, _ in channels]
        if roi_channel is None:
            roi_channel = ch_names[0]
        elif roi_channel not in ch_names:
            raise ValueError(f"roi_channel={roi_channel!r} is not one of the channels {ch_names}")

        rows = [dict(ch_id=ch_id, ch_name=ch_name, roi_default=int(ch_name == roi_channel))
                for ch_name, ch_id in channels]
        self.insert(rows, skip_duplicates=skip_duplicates)

    def get_roi_channel(self) -> str:
        """Name of the default target channel for ROI detection.

        Falls back to the channel with the lowest id if no channel is flagged as `roi_default`.
        """
        rows = sorted(self.to_dicts(), key=lambda row: row['ch_id'])
        if len(rows) == 0:
            raise ValueError("channel_table is empty")

        defaults = [row['ch_name'] for row in rows if row['roi_default']]
        if len(defaults) > 1:
            raise ValueError(f"Multiple default ROI channels: {defaults}. Set roi_default for exactly one channel.")
        return defaults[0] if len(defaults) == 1 else rows[0]['ch_name']

    def get_ch_names(self) -> list[str]:
        """Names of all channels with the default ROI channel first, then ordered by channel id."""
        ch_names = [row['ch_name'] for row in sorted(self.to_dicts(), key=lambda row: row['ch_id'])]
        roi_channel = self.get_roi_channel()
        return [roi_channel] + [ch_name for ch_name in ch_names if ch_name != roi_channel]
