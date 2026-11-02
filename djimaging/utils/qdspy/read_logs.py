import ast
from datetime import datetime
from typing import List, Dict, Tuple

_STAGE_INFO_KEYS = frozenset({"scaling_x", "scaling_y", "offset_x", "offset_y", "rotation"})


def parse_stimulus_log(log_path: str, verbose: bool = True) -> Tuple[List[Dict], Dict]:
    """
    Parse a stimulus log file and extract stimulus presentation information.

    Args:
        log_path: Path to the log file to parse
        verbose: If True, print parsing errors and summary statistics

    Returns:
        A tuple containing:
        - stims: List of dictionaries containing stimulus information
        - stats: Dictionary with parsing statistics (nLinesTotal, nLinesData,
                 nLinesErr, nErr)

    Each stimulus entry contains:
        - index: Stimulus index
        - stimFileName: Name of the stimulus file
        - stimPath: Path to the stimulus file (forward-slash normalised)
        - stimMD5: MD5 hash of the stimulus
        - t_abs_s: Absolute time from log start (seconds)
        - t_since_last_s: Time since last stimulus ended (seconds)
        - t_start: Start time
        - t_end: End time
        - t_dur_s: Duration in seconds
        - aborted: Whether stimulus was aborted
        - t_dur_s_calc: Calculated duration based on frames
        - nDroppedFrames: Number of dropped frames
        - params: Additional parameters emitted by the stimulus script
        - sequenceUsed: Video-sequence index used (MouseCam stimuli only)
        - stageInfo: Stage calibration dict emitted by newer QDSpy versions
                     (scaling_x/y, offset_x/y, rotation); absent in older logs
    """
    nLinesTotal = 0
    nLinesData = 0
    nLinesErr = 0
    nErr = 0
    isStimStarted = False
    stims = []
    nStims = 0
    dt_log_start = None
    dt_last_end = None
    dt_start = None
    stimInfo = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as fLog:
        for line in fLog:
            nLinesTotal += 1
            sDateTime = line[:15]
            sInfoType = line[15:23].strip().upper()
            sMsg = line[23:len(line) - 1].strip()

            try:
                dt = datetime.strptime(sDateTime, "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            if nLinesTotal == 1:
                dt_log_start = dt
                dt_last_end = dt_log_start

            if sInfoType not in ["DATA"]:
                continue

            try:
                data = ast.literal_eval(sMsg)
            except (ValueError, SyntaxError):
                if verbose:
                    print(f"ERROR: parsing line {nLinesTotal - 1} failed:")
                    print(f"'{sMsg}'")
                nLinesErr += 1
                continue

            if not isinstance(data, dict):
                nLinesErr += 1
                continue

            stimState = data.get("stimState", "").upper()

            if stimState == "STARTED":
                if isStimStarted:
                    if verbose:
                        print("ERROR: Two consecutive stimulus starts")
                    nErr += 1

                isStimStarted = True
                dt_start = dt
                t_diff = (dt - dt_log_start).total_seconds()
                t_diff_last = (dt - dt_last_end).total_seconds()
                # Normalise to forward slashes so paths are consistent
                # regardless of the OS that generated or reads the log.
                norm_fn = data.get("stimFileName", "").replace("\\", "/")
                stimInfo = {
                    "index": nStims,
                    "stimFileName": norm_fn.rsplit("/", 1)[-1],
                    "stimPath": norm_fn.rsplit("/", 1)[0] if "/" in norm_fn else "",
                    "stimMD5": data.get("stimMD5", ""),
                    "t_abs_s": t_diff,
                    "t_since_last_s": t_diff_last,
                    "t_start": dt.time(),
                }

            elif stimState in ["ABORTED", "FINISHED"]:
                if isStimStarted:
                    norm_fn = data.get("stimFileName", "").replace("\\", "/")
                    fn_start = (
                        stimInfo["stimPath"] + "/" + stimInfo["stimFileName"]
                        if stimInfo["stimPath"]
                        else stimInfo["stimFileName"]
                    )
                    if fn_start != norm_fn:
                        if verbose:
                            print("ERROR: File paths for stimulus start and end differ")
                        nErr += 1

                    dt_last_end = dt
                    t_diff = (dt - dt_start).total_seconds()
                    stimInfo.update({
                        "aborted": stimState == "ABORTED",
                        "t_end": dt.time(),
                        "t_dur_s": t_diff,
                    })
                    stims.append(stimInfo)
                    nStims += 1
                    isStimStarted = False
                else:
                    if verbose:
                        print("ERROR: Stimulus end w/o start?")
                    nErr += 1

            elif "nFrames" in data:
                # Frame statistics arrive after FINISHED, so nStims-1 is correct here.
                if nStims > 0:
                    stims[nStims - 1].update({
                        "t_dur_s_calc": data["nFrames"] / data["avgFreq_Hz"],
                        "nDroppedFrames": data["nDroppedFrames"],
                    })

            elif isStimStarted:
                # Extra DATA lines emitted while a stimulus is running.
                # Classify by content rather than position to support all QDSpy versions.
                if "SequenceUsed" in data:
                    stimInfo["sequenceUsed"] = data["SequenceUsed"]
                elif set(data.keys()) == _STAGE_INFO_KEYS:
                    # Stage calibration line added in newer QDSpy versions.
                    stimInfo["stageInfo"] = data
                else:
                    # Treat as stimulus parameters; last assignment wins if emitted
                    # multiple times (older QDSpy versions emitted params after stage info).
                    stimInfo["params"] = data

            else:
                if verbose:
                    print("ERROR: Data w/o start??")
                nErr += 1

            nLinesData += 1

    if verbose:
        print(f"{nLinesData} of {nLinesTotal} line(s) extracted.")
        print(f"{nLinesErr} line(s) failed parsing, {nErr} error(s) occurred post-processing.")

    stats = {
        "nLinesTotal": nLinesTotal,
        "nLinesData": nLinesData,
        "nLinesErr": nLinesErr,
        "nErr": nErr,
    }

    return stims, stats
