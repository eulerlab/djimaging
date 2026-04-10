import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple


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
        - stimPath: Path to the stimulus file
        - stimMD5: MD5 hash of the stimulus
        - t_abs_s: Absolute time from log start (seconds)
        - t_since_last_s: Time since last stimulus ended (seconds)
        - t_start: Start time
        - t_end: End time
        - t_dur_s: Duration in seconds
        - aborted: Whether stimulus was aborted
        - t_dur_s_calc: Calculated duration based on frames
        - nDroppedFrames: Number of dropped frames
        - params: Additional parameters
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
    iLineLastStart = 0
    dt_start = None
    stimInfo = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as fLog:
        for line in fLog:
            # Extract elements of each line
            nLinesTotal += 1
            sDateTime = line[:15]
            sInfoType = line[15:23].strip().upper()
            sMsg = line[23:len(line) - 1].strip()

            # Convert time stamp into a datetime
            dt = datetime.strptime(sDateTime, "%Y%m%d_%H%M%S")
            if nLinesTotal == 1:
                # First line; take as start time
                dt_log_start = dt
                dt_last_end = dt_log_start

            # Filter for relevant information
            if sInfoType not in ["DATA"]:
                # Ignore field
                continue

            # Convert data line into dictionary
            sMsg = sMsg.replace("'", "\"")
            sMsg = sMsg.replace("\\\\", "/")
            sMsg = sMsg.replace("(", "[")
            sMsg = sMsg.replace(")", "]")

            try:
                data = json.loads(sMsg)
            except json.JSONDecodeError as e:
                # JSON parsing failed
                if verbose:
                    print(f"ERROR: parsing line {nLinesTotal - 1} failed:")
                    print(f"'{sMsg}'")
                nLinesErr += 1
                data = None
                continue

            # Get stimulus start/stop pairs
            try:
                stimState = data["stimState"].upper()
            except KeyError:
                stimState = None

            if stimState:
                # Data contains stimulus information
                if stimState == "STARTED":
                    if isStimStarted:
                        if verbose:
                            print("ERROR: Two consecutive stimulus starts")
                        nErr += 1

                    isStimStarted = True
                    iLineLastStart = nLinesData
                    dt_start = dt
                    t_diff = (dt - dt_log_start).total_seconds()
                    t_diff_last = (dt - dt_last_end).total_seconds()
                    stimInfo = {
                        "index": nStims,
                        "stimFileName": Path(data["stimFileName"]).name,
                        "stimPath": str(Path(data["stimFileName"]).parent),
                        "stimMD5": data["stimMD5"],
                        "t_abs_s": t_diff,
                        "t_since_last_s": t_diff_last,
                        "t_start": dt.time()
                    }

                elif stimState in ["ABORTED", "FINISHED"]:
                    if isStimStarted:
                        # Check if stimulus end belongs to stimulus start
                        fn = str(Path(stimInfo["stimPath"], stimInfo["stimFileName"])).replace("\\", "/")
                        if not data["stimFileName"] == fn:
                            if verbose:
                                print("ERROR: File paths for stimulus start and end differ")
                            nErr += 1

                        # Append stimulus list entry
                        dt_last_end = dt
                        t_diff = (dt - dt_start).total_seconds()
                        stimInfo.update({
                            "aborted": stimState == "ABORTED",
                            "t_end": dt.time(),
                            "t_dur_s": t_diff
                        })
                        stims.append(stimInfo)
                        nStims += 1
                        isStimStarted = False
                    else:
                        if verbose:
                            print("ERROR: Stimulus end w/o start?")
                        nErr += 1
            else:
                # Other information
                try:
                    _ = data["nFrames"]
                    isFrameInfo = True
                except KeyError:
                    isFrameInfo = False

                if isFrameInfo:
                    # Information about stimulus presentation statistics
                    if nStims > 0:
                        stims[nStims - 1].update({
                            "t_dur_s_calc": data["nFrames"] / data["avgFreq_Hz"],
                            "nDroppedFrames": data["nDroppedFrames"]
                        })
                else:
                    if nLinesData > iLineLastStart and nLinesData < iLineLastStart + 3:
                        # Last start was only up to 2 lines before
                        if nStims > 0:
                            stims[nStims - 1].update({"params": data})
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
        "nErr": nErr
    }

    return stims, stats
