"""Folder with the real test data (not in the repository).

Set the environment variable `DJIMAGING_TEST_DATA` to the folder, e.g. `X:/Data/DataJointTestData` on the server.
Default: `test_data` in the repository root. Tests that need a missing file are skipped.
"""
import os

TEST_DATA_PATH = os.path.normpath(os.environ.get(
    'DJIMAGING_TEST_DATA', os.path.join(os.path.dirname(__file__), '..', '..', 'test_data')))
