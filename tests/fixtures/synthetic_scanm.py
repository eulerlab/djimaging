from pathlib import Path

import h5py
import numpy as np


NX = 8
NY = 8
FRAME_DT = 0.1
PIXEL_DURATION_US = FRAME_DT / (NX * NY) * 1e6
STIMULATOR_DELAY_MS = 2.0
TRIGGER_X_INDEX = 1
EXPERIMENT_DAY = "20990101"
EXPERIMENT_NUMBER = "1"
RECORDING_PREFIX = "SYN_SUBJECT_REGION_FIELD0"

RECORDING_SPECS = {
    "chirp": {
        "n_frames": 900,
        "trigger_times": np.array(
            [time for start in 7.0 + 16.0 * np.arange(5) for time in (start, start + 4.0)]
        ),
    },
    "MB": {
        "n_frames": 1200,
        "trigger_times": 7.0 + 4.5 * np.arange(24),
    },
    "DN": {
        "n_frames": 3100,
        "trigger_times": 7.0 + 0.2 * np.arange(1500),
    },
}


def generate_tutorial_dataset(root: Path, seed: int = 42) -> Path:
    """Create a small xy-RGC dataset that follows the tutorial's file layout."""
    root = Path(root)
    experiment_dir = root / EXPERIMENT_DAY / EXPERIMENT_NUMBER
    pre_dir = experiment_dir / "Pre"
    pre_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "Raw").mkdir(exist_ok=True)

    (experiment_dir / "synthetic__left.ini").write_text(_experiment_ini(), encoding="utf-8")

    rng = np.random.default_rng(seed)
    roi_mask = _roi_mask()
    for stimulus, spec in RECORDING_SPECS.items():
        filepath = pre_dir / f"{RECORDING_PREFIX}_{stimulus}_TEST.h5"
        _write_recording(
            filepath=filepath,
            stimulus=stimulus,
            n_frames=spec["n_frames"],
            requested_trigger_times=spec["trigger_times"],
            roi_mask=roi_mask,
            rng=rng,
        )

    resources_dir = root / "resources"
    resources_dir.mkdir(exist_ok=True)
    noise = rng.choice(np.array([-1, 1], dtype=np.int8), size=(1500, 15, 20))
    with h5py.File(resources_dir / "noise.h5", "w") as h5_file:
        h5_file.create_dataset("stimulusarray", data=noise, compression="gzip", compression_opts=1)

    return root


def expected_trace_times(
        roi_mask: np.ndarray, n_frames: int, precision: str = "line") -> np.ndarray:
    """Calculate synthetic ROI timestamps directly from the scan geometry."""
    if precision not in {"line", "pixel"}:
        raise ValueError(f"Unknown precision: {precision}")

    pixel_dt = PIXEL_DURATION_US * 1e-6
    frame_times = np.arange(n_frames) * FRAME_DT
    roi_ids = np.unique(roi_mask)
    roi_ids = roi_ids[roi_ids < 0]
    roi_ids = roi_ids[np.argsort(np.abs(roi_ids))]
    trace_times = np.empty((n_frames, len(roi_ids)), dtype=float)

    x_indexes, y_indexes = np.indices(roi_mask.shape)
    for column, roi_id in enumerate(roi_ids):
        in_roi = roi_mask == roi_id
        if precision == "line":
            offsets = y_indexes[in_roi] * NX * pixel_dt
        else:
            offsets = (y_indexes[in_roi] * NX + x_indexes[in_roi]) * pixel_dt
        trace_times[:, column] = frame_times + np.median(offsets)

    return trace_times


def _write_recording(
        filepath: Path,
        stimulus: str,
        n_frames: int,
        requested_trigger_times: np.ndarray,
        roi_mask: np.ndarray,
        rng: np.random.Generator,
) -> None:
    frame_times = np.arange(n_frames) * FRAME_DT
    trigger_stack, trigger_times = _trigger_stack(n_frames, requested_trigger_times)

    data_stack = rng.normal(10_000, 4, size=(NX, NY, n_frames))
    for roi_id in (1, 2):
        response = _response(stimulus, frame_times, trigger_times, roi_id, rng)
        roi_pixels = roi_mask == -roi_id
        data_stack[roi_pixels, :] += response

    data_stack = _as_uint16(data_stack)

    alt_stack = rng.normal(11_000, 3, size=(NX, NY, n_frames))
    alt_stack[roi_mask == -1, :] += 250
    alt_stack[roi_mask == -2, :] += 500

    alt_stack = _as_uint16(alt_stack)

    roi_ids = np.unique(roi_mask)
    roi_ids = roi_ids[roi_ids < 0]
    roi_ids = roi_ids[np.argsort(np.abs(roi_ids))]
    traces = np.column_stack([
        np.mean(data_stack[roi_mask == roi_id], axis=0) for roi_id in roi_ids
    ])
    trace_times = expected_trace_times(roi_mask, n_frames, precision="line")

    with h5py.File(filepath, "w") as h5_file:
        _write_stack(h5_file, "wDataCh0", data_stack)
        _write_stack(h5_file, "wDataCh1", alt_stack)
        _write_stack(h5_file, "wDataCh2", trigger_stack)
        h5_file.create_dataset("ROIs", data=roi_mask.astype(np.int16))
        h5_file.create_dataset("Traces0_raw", data=traces)
        h5_file.create_dataset(
            "Tracetimes0", data=trace_times + STIMULATOR_DELAY_MS / 1000
        )
        h5_file.create_dataset("Triggertimes", data=trigger_times)
        h5_file.create_dataset(
            "Triggervalues", data=np.full(trigger_times.size, 60_000, dtype=np.float32)
        )
        _write_wparams_num(h5_file)
        _write_os_params(h5_file)


def _write_stack(h5_file: h5py.File, name: str, values: np.ndarray) -> None:
    h5_file.create_dataset(
        name,
        data=values,
        compression="gzip",
        compression_opts=1,
        shuffle=True,
    )


def _as_uint16(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def _write_wparams_num(h5_file: h5py.File) -> None:
    params = {
        "User_dxPix": NX,
        "User_dyPix": NY,
        "User_dzPix": 0,
        "User_nPixRetrace": 0,
        "User_nXPixLineOffs": 0,
        "RealPixDur": PIXEL_DURATION_US,
        "Zoom": 1,
        "Angle_deg": 0,
        "XCoord_um": 123,
        "YCoord_um": 245,
        "ZCoord_um": 9,
        "ZStep_um": 1,
        "User_ScanType": 10,
    }
    dataset = h5_file.create_dataset("wParamsNum", data=np.asarray(list(params.values()), dtype=np.float32))

    labels = np.full((len(params) + 1, 1), "", dtype=h5py.string_dtype(encoding="utf-8"))
    labels[1:, 0] = list(params)
    dataset.attrs["IGORWaveDimensionLabels"] = labels


def _write_os_params(h5_file: h5py.File) -> None:
    dataset = h5_file.create_dataset(
        "OS_Parameters", data=np.asarray([STIMULATOR_DELAY_MS], dtype=np.float32)
    )
    labels = np.full((2, 1), "", dtype=h5py.string_dtype(encoding="utf-8"))
    labels[1, 0] = "StimulatorDelay"
    dataset.attrs["IGORWaveDimensionLabels"] = labels


def _trigger_stack(n_frames: int, requested_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stack = np.full((NX, NY, n_frames), 10_000, dtype=np.uint16)
    line_dt = FRAME_DT / NY
    line_indices = np.rint(np.asarray(requested_times) / line_dt).astype(int)

    for line_index in line_indices:
        frame_index, y_index = divmod(line_index, NY)
        if frame_index >= n_frames:
            raise ValueError(f"Trigger at line {line_index} is outside a {n_frames}-frame recording")
        stack[TRIGGER_X_INDEX:, y_index, frame_index] = 60_000

    pixel_dt = PIXEL_DURATION_US * 1e-6
    return stack, line_indices * line_dt + TRIGGER_X_INDEX * pixel_dt


def _response(
        stimulus: str,
        frame_times: np.ndarray,
        trigger_times: np.ndarray,
        roi_id: int,
        rng: np.random.Generator,
) -> np.ndarray:
    response = rng.normal(0, 2, size=frame_times.size)

    if stimulus == "chirp":
        scale = 1.0 if roi_id == 1 else 0.7
        for start in trigger_times[::2]:
            rel_time = frame_times - start
            active = (rel_time >= 0) & (rel_time < 14)
            pattern = (
                100 * np.exp(-((rel_time - 2) / 0.8) ** 2)
                - 70 * np.exp(-((rel_time - 7) / 1.2) ** 2)
                + 45 * np.sin(2 * np.pi * rel_time / 4)
                + 25 * np.sin(2 * np.pi * rel_time ** 2 / 90)
            )
            response[active] += scale * pattern[active]
    elif stimulus == "MB":
        directions = np.array([0, 180, 45, 225, 90, 270, 135, 315])
        preferred_direction = 0 if roi_id == 1 else 90
        for index, start in enumerate(trigger_times):
            rel_time = frame_times - start
            active = (rel_time >= 0) & (rel_time < 3.8)
            direction = directions[index % directions.size]
            direction_gain = 1 + 0.75 * np.cos(np.deg2rad(direction - preferred_direction))
            time_kernel = (
                80 * np.exp(-((rel_time - 1.7) / 0.45) ** 2)
                + 35 * np.exp(-((rel_time - 3.0) / 0.35) ** 2)
            )
            response[active] += direction_gain * time_kernel[active]
    elif stimulus == "DN":
        response += rng.normal(0, 15 if roi_id == 1 else 10, size=frame_times.size)
    else:
        raise ValueError(f"Unknown stimulus: {stimulus}")

    return response


def _roi_mask() -> np.ndarray:
    roi_mask = np.ones((NX, NY), dtype=np.int16)
    roi_mask[3:5, 1:3] = -1
    roi_mask[5:7, 5:7] = -2
    return roi_mask


def _experiment_ini() -> str:
    return """[Project]
string_projName=Synthetic tutorial data
string_date=2099-01-01
string_setupID=3

[Animal]
string_animSpecies=synthetic
string_animGender=unknown
string_eye=left

[Preparation]
string_prep=wholemount
string_prepWMOpticDiscPos=111;222;3
uint8_prepWMOrient=0

[Pharmacology]
string_pharmDrug=none
string_pharmDrugConc_um=
string_preTime=
string_pharmRem=
"""
