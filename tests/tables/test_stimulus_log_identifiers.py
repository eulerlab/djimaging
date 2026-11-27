import pytest

from djimaging.tables.core.stimulus import log_identifiers


def test_log_identifiers_only_given_lists():
    assert log_identifiers() == {}
    assert log_identifiers(["abc"], None) == {'stim_hash_list': ["abc"]}
    assert log_identifiers([], ["RGC_Chirp"]) == {'stim_hash_list': [], 'stim_name_list': ["RGC_Chirp"]}


def test_log_identifiers_rejects_single_string():
    # A string would be iterated character by character and never match a log entry
    with pytest.raises(TypeError):
        log_identifiers("f124bb886d58ff93a1e1d4448d76e979")
