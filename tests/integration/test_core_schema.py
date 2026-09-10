import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("DJ_TEST_MYSQL") != "1",
    reason="requires the MySQL integration-test service",
)


def test_all_tutorial_tables_initialize(tutorial_schema):
    tables = tutorial_schema.schema.list_tables()
    assert tables
    for name in ("ClassifierV2", "Baden16TracesV2", "CelltypeAssignmentV2"):
        assert getattr(tutorial_schema, name).table_name in tables


def test_user_info_round_trip(tutorial_schema, tmp_path):
    user_info = tutorial_schema.UserInfo()
    user_info.upload_user(
        {
            "experimenter": "ci_user",
            "data_dir": str(tmp_path),
            "field_loc": 0,
            "stimulus_loc": 1,
        },
        verbose=0,
    )

    row = (user_info & {"experimenter": "ci_user"}).fetch1()
    assert row["data_dir"] == f"{tmp_path}/"


def test_add_default_raw_data_params(tutorial_schema, tmp_path):
    tutorial_schema.UserInfo().upload_user(
        {
            "experimenter": "ci_params",
            "data_dir": str(tmp_path),
            "field_loc": 0,
            "stimulus_loc": 1,
        },
        verbose=0,
    )

    raw_data_params = tutorial_schema.RawDataParams()
    raw_data_params.add_default(experimenter_list=["ci_params"])

    row = (raw_data_params & {"experimenter": "ci_params"}).fetch1()
    assert row["raw_id"] == 1
    assert row["compute_from_stack"] == 1
