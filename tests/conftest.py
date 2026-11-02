from pathlib import Path
from typing import Callable

import pytest

from tests.fixtures.synthetic_scanm import (
    EXPERIMENT_DAY,
    EXPERIMENT_NUMBER,
    RECORDING_PREFIX,
    generate_tutorial_dataset,
)


@pytest.fixture(scope="session")
def tutorial_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_tutorial_dataset(tmp_path_factory.mktemp("synthetic_scanm_data"))


@pytest.fixture(scope="session")
def synthetic_recording_path(tutorial_data_dir: Path) -> Callable[[str], Path]:
    pre_dir = tutorial_data_dir / EXPERIMENT_DAY / EXPERIMENT_NUMBER / "Pre"

    def get_recording(stimulus: str) -> Path:
        return pre_dir / f"{RECORDING_PREFIX}_{stimulus}_TEST.h5"

    return get_recording
