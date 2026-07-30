import os

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("DJ_TEST_MYSQL") != "1",
    reason="requires the MySQL integration-test service",
)


def test_core_tables_are_declared(core_schema):
    assert "experimenter" in core_schema.UserInfo().heading.names
    assert "raw_id" in core_schema.RawDataParams().heading.names


def test_user_info_round_trip(core_schema, tmp_path):
    user_info = core_schema.UserInfo()
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


def test_add_default_raw_data_params(core_schema, tmp_path):
    core_schema.UserInfo().upload_user(
        {
            "experimenter": "ci_params",
            "data_dir": str(tmp_path),
            "field_loc": 0,
            "stimulus_loc": 1,
        },
        verbose=0,
    )

    raw_data_params = core_schema.RawDataParams()
    raw_data_params.add_default(experimenter_list=["ci_params"])

    row = (raw_data_params & {"experimenter": "ci_params"}).fetch1()
    assert row["raw_id"] == 1
    assert row["compute_from_stack"] == 1
