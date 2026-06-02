"""Tests for deterministic serialization used by evaluation cache keys."""

import sys
from pathlib import Path
from typing import cast

# pytest.ini adds `src` and `examples`, so add repo root for `performance_eval` imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from performance_eval.utility import _stable_serialize, cache_file_from_payload, check_git_clean, get_git_commit_hash


def test_stable_serialize_set_order_is_deterministic():
    value = {"z", "a", "m"}

    first = _stable_serialize(value)
    second = _stable_serialize(value)

    assert first == second
    assert first == ["a", "m", "z"]


def test_stable_serialize_nested_structure_is_deterministic():
    value = {
        "allowed_labels": {"EMAIL", "PERSON", "ORG"},
        "comparable_golds": [
            (
                "sample text",
                {
                    (10, 20, "PERSON"),
                    (0, 5, "ORG"),
                },
            )
        ],
        "max_errors": 3,
    }

    first = _stable_serialize(value)
    second = _stable_serialize(value)

    assert first == second

    expected = {
        "allowed_labels": ["EMAIL", "ORG", "PERSON"],
        "comparable_golds": [
            [
                "sample text",
                [
                    [0, 5, "ORG"],
                    [10, 20, "PERSON"],
                ],
            ]
        ],
        "max_errors": 3,
    }
    assert first == expected


def test_stable_serialize_dict_keys_are_sorted_as_strings():
    value = {10: "ten", "2": "two", 1: "one"}

    serialized = cast(dict[str, object], _stable_serialize(value))

    assert list(serialized.keys()) == ["1", "10", "2"]


def test_cache_file_from_payload_is_deterministic_for_equivalent_sets(tmp_path):
    module_file = tmp_path / "dummy_module.py"
    module_file.write_text("# test module\n", encoding="utf-8")

    payload_a = {"allowed_labels": {"PERSON", "EMAIL"}, "max_errors": 1}
    payload_b = {"allowed_labels": {"EMAIL", "PERSON"}, "max_errors": 1}

    cache_file_a = cache_file_from_payload(module_file, payload_a)
    cache_file_b = cache_file_from_payload(module_file, payload_b)

    assert cache_file_a == cache_file_b
    assert cache_file_a.parent.name == ".cache"


def test_get_git_commit_hash_returns_valid_hash():
    """Test that get_git_commit_hash returns a 40-char hex string (SHA-1)."""
    commit_hash = get_git_commit_hash(".")
    assert isinstance(commit_hash, str)
    assert len(commit_hash) == 40
    assert all(c in "0123456789abcdef" for c in commit_hash)


def test_check_git_clean_raises_on_uncommitted_changes():
    """Test that check_git_clean raises ValueError when there are uncommitted changes."""
    import pytest
    with pytest.raises(ValueError, match="Git working directory is not clean"):
        check_git_clean(".")
