"""Tests for the privacy policy loading and consent bookkeeping."""

from oekg.privacy import (
    build_consent_record,
    consent_is_current,
    load_policy,
    policy_version,
)


def _write(tmp_path, text: str):
    path = tmp_path / "PRIVACY.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_policy_returns_text(tmp_path):
    path = _write(tmp_path, "  # Policy\n\nSome text.\n")
    assert load_policy(path) == "# Policy\n\nSome text."


def test_load_policy_missing_file_returns_empty_string(tmp_path):
    assert load_policy(tmp_path / "does_not_exist.md") == ""


def test_load_policy_on_a_directory_returns_empty_string(tmp_path):
    assert load_policy(tmp_path) == ""


def test_policy_version_is_stable_and_short():
    version = policy_version("# Policy\n\nSome text.")
    assert version == policy_version("# Policy\n\nSome text.")
    assert len(version) == 12


def test_policy_version_ignores_whitespace_and_line_endings():
    assert policy_version("a b\nc") == policy_version("a  b\r\n  c")


def test_policy_version_changes_with_wording():
    assert policy_version("We store nothing.") != policy_version("We store everything.")


def test_consent_is_current_only_for_the_matching_version():
    record = build_consent_record("abc123")
    assert consent_is_current(record, "abc123")
    assert not consent_is_current(record, "different")


def test_consent_is_not_current_without_a_record():
    assert not consent_is_current(None, "abc123")
    assert not consent_is_current({}, "abc123")


def test_consent_is_not_current_when_not_accepted():
    record = {**build_consent_record("abc123"), "accepted": False}
    assert not consent_is_current(record, "abc123")


def test_consent_is_never_current_without_a_version():
    assert not consent_is_current(build_consent_record(""), "")


def test_build_consent_record_carries_version_and_timestamp():
    record = build_consent_record("abc123")
    assert record["accepted"] is True
    assert record["version"] == "abc123"
    assert record["accepted_at"].endswith("+00:00")


def test_shipped_policy_is_loadable_and_versioned():
    from oekg.config import get_config

    policy = load_policy(get_config().privacy_policy_path)
    assert policy.startswith("# Privacy Policy")
    assert len(policy_version(policy)) == 12
