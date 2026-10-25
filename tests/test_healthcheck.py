"""Tests for the health check: alert state machine, mail, secret hygiene."""

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from oekg import healthcheck as hc
from oekg.config import get_config

NOW = datetime(2026, 10, 25, 12, 0, tzinfo=timezone.utc)
OK = hc.CheckResult("chat", True, "answered", 0.4)
BAD = hc.CheckResult("embeddings", False, "RuntimeError: boom", 1.2)


def _config(tmp_path, **overrides):
    values = {
        "log_dir": tmp_path,
        "health_mail_to": "ops@example.org",
        "health_smtp_host": "smtp.example.org",
        "health_smtp_port": 587,
        "health_smtp_starttls": True,
        "health_smtp_user": "",
        "health_smtp_password": None,
        "health_failure_threshold": 2,
        "llm_api_key": "kiwi-secret-key",
        "embedding_api_key": "gateway-secret-key",
        "oep_token": "oep-secret-token",
    }
    values.update(overrides)
    return replace(get_config(), **values)


# -- state machine ------------------------------------------------------------


def _run(state, *results, threshold=2):
    return hc.advance(state, list(results), NOW, threshold)


def test_a_single_failure_does_not_alert():
    state, event = _run(hc._empty_state(), OK, BAD)
    assert event is None
    assert state["failures"] == 1 and state["failed"] == ["embeddings"]


def test_the_threshold_alerts_exactly_once_per_outage():
    state = hc._empty_state()
    events = []
    for _ in range(5):
        state, event = _run(state, BAD)
        events.append(event)
    assert events == [None, "down", None, None, None]


def test_recovery_alerts_only_after_a_down_alert():
    state, _ = _run(hc._empty_state(), BAD)  # one blip, never alerted
    state, event = _run(state, OK)
    assert event is None and state == hc._empty_state()

    down, _ = _run(hc._empty_state(), BAD)
    down, event = _run(down, BAD)
    assert event == "down" and down["alerted"]
    _, event = _run(down, OK)
    assert event == "recovered"


def test_a_blip_between_clean_runs_resets_the_streak():
    state, _ = _run(hc._empty_state(), BAD)
    state, _ = _run(state, OK)
    state, event = _run(state, BAD)
    assert event is None and state["failures"] == 1


def test_the_outage_start_survives_later_failing_runs():
    state, _ = _run(hc._empty_state(), BAD)
    started = state["since"]
    state, _ = _run(state, BAD)
    assert state["since"] == started


def test_corrupt_state_file_means_all_was_well(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("{not json", encoding="utf-8")
    assert hc.load_state(path) == hc._empty_state()
    assert hc.load_state(tmp_path / "missing.json") == hc._empty_state()


# -- secrets ------------------------------------------------------------------


def test_redaction_removes_configured_secrets_and_key_shaped_text(tmp_path):
    cfg = _config(tmp_path)
    text = "401 for kiwi-secret-key via oep-secret-token; echoed sk-proj-AbC123****xYz9"
    cleaned = hc._redact(text, cfg)
    for leaked in ("kiwi-secret-key", "oep-secret-token", "AbC123"):
        assert leaked not in cleaned
    assert "***" in cleaned


def test_a_failing_check_is_reported_without_its_secret(tmp_path):
    cfg = _config(tmp_path)

    def broken(_config):
        raise RuntimeError("rejected key kiwi-secret-key")

    (result,) = hc.run_checks(cfg, [("chat", broken)])
    assert not result.ok
    assert "kiwi-secret-key" not in result.detail
    assert result.detail.startswith("RuntimeError")


def test_one_failing_check_does_not_stop_the_others(tmp_path):
    cfg = _config(tmp_path)

    def broken(_config):
        raise RuntimeError("down")

    results = hc.run_checks(cfg, [("a", broken), ("b", lambda _c: "fine")])
    assert [r.ok for r in results] == [False, True]


# -- checks -------------------------------------------------------------------


def test_embedding_dimension_must_match_the_vector_store(tmp_path, monkeypatch):
    cfg = _config(tmp_path)

    class FakeClient:
        def __init__(self, **_kwargs):
            self.embeddings = SimpleNamespace(
                create=lambda **_k: SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * 3072)])
            )

        def close(self):
            pass

    monkeypatch.setattr(hc, "OpenAI", FakeClient)
    monkeypatch.setattr(hc.faiss, "read_index", lambda _path: SimpleNamespace(d=1024))
    with pytest.raises(RuntimeError, match="rebuilding the index"):
        hc._check_embeddings(cfg)


def test_checks_refuse_to_run_without_their_key(tmp_path):
    # With api_key=None the OpenAI SDK would silently use OPENAI_API_KEY.
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        hc._check_chat(_config(tmp_path, llm_api_key=None))
    with pytest.raises(RuntimeError, match="EMBEDDING_API_KEY"):
        hc._check_embeddings(_config(tmp_path, embedding_api_key=None))


# -- mail ---------------------------------------------------------------------


def test_down_mail_names_the_failed_checks_and_nothing_secret(tmp_path):
    cfg = _config(tmp_path)
    state, _ = _run(hc._empty_state(), OK, BAD)
    state, event = _run(state, OK, BAD)
    message = hc.build_message(cfg, event, [OK, BAD], state)
    assert message["Subject"] == "[Ask OEKG] DOWN: embeddings"
    assert message["To"] == "ops@example.org"
    body = message.get_content()
    assert "FAIL  embeddings" in body and "ok    chat" in body
    for secret in ("kiwi-secret-key", "gateway-secret-key", "oep-secret-token"):
        assert secret not in body and secret not in str(message)


def test_recovery_mail_says_when_the_outage_began(tmp_path):
    cfg = _config(tmp_path)
    outage = {"failures": 3, "since": "2026-10-25T11:50:00+00:00", "alerted": True, "failed": []}
    message = hc.build_message(cfg, "recovered", [OK], outage)
    assert message["Subject"] == "[Ask OEKG] recovered"
    assert "2026-10-25T11:50:00" in message.get_content()


class FakeSMTP:
    instances: list = []

    def __init__(self, host, port, timeout):
        self.host, self.port, self.calls, self.sent = host, port, [], []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def ehlo(self):
        self.calls.append("ehlo")

    def starttls(self):
        self.calls.append("starttls")

    def login(self, user, password):
        self.calls.append(("login", user, password))

    def send_message(self, message):
        self.sent.append(message)


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.instances = []
    monkeypatch.setattr(hc.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(hc.smtplib, "SMTP_SSL", FakeSMTP)
    return FakeSMTP


def test_mail_upgrades_to_tls_before_logging_in(tmp_path, smtp):
    cfg = _config(tmp_path, health_smtp_user="bot", health_smtp_password="pw-1234567")
    hc.send_mail(cfg, hc.build_message(cfg, "recovered", [OK], hc._empty_state()))
    (server,) = smtp.instances
    assert server.calls.index("starttls") < server.calls.index(("login", "bot", "pw-1234567"))
    assert len(server.sent) == 1


def test_mail_refuses_credentials_without_tls(tmp_path, smtp):
    cfg = _config(tmp_path, health_smtp_user="bot", health_smtp_starttls=False)
    with pytest.raises(RuntimeError, match="without TLS"):
        hc.send_mail(cfg, hc.build_message(cfg, "recovered", [OK], hc._empty_state()))
    assert smtp.instances == []


def test_unconfigured_mail_fails_loudly(tmp_path, smtp):
    cfg = _config(tmp_path, health_smtp_host="")
    with pytest.raises(RuntimeError, match="not configured"):
        hc.send_mail(cfg, hc.build_message(cfg, "recovered", [OK], hc._empty_state()))


# -- end to end through main() ------------------------------------------------


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """main() with checks, mail and .env loading replaced by fakes."""
    cfg = _config(tmp_path)
    monkeypatch.setattr(hc, "get_config", lambda: cfg)
    monkeypatch.setattr(hc, "load_dotenv", lambda *_a, **_k: None)
    box = SimpleNamespace(results=[OK], mails=[], mail_fails=False, cfg=cfg)

    monkeypatch.setattr(hc, "run_checks", lambda _cfg: box.results)

    def fake_send(_cfg, message):
        if box.mail_fails:
            raise RuntimeError("relay unreachable")
        box.mails.append(message["Subject"])

    monkeypatch.setattr(hc, "send_mail", fake_send)
    return box


def test_an_outage_costs_two_mails(harness):
    harness.results = [BAD]
    codes = [hc.main([]) for _ in range(4)]
    assert codes == [1, 1, 1, 1]
    assert harness.mails == ["[Ask OEKG] DOWN: embeddings"]

    harness.results = [OK]
    assert hc.main([]) == 0
    assert hc.main([]) == 0
    assert harness.mails == ["[Ask OEKG] DOWN: embeddings", "[Ask OEKG] recovered"]


def test_an_unsent_alert_is_retried_on_the_next_run(harness):
    harness.results = [BAD]
    harness.mail_fails = True
    assert hc.main([]) == 1  # first failure, below the threshold
    assert hc.main([]) == 2  # alert due, relay down
    assert hc.main([]) == 2  # still due
    assert harness.mails == []

    harness.mail_fails = False
    assert hc.main([]) == 1
    assert harness.mails == ["[Ask OEKG] DOWN: embeddings"]


def test_an_unsent_recovery_is_retried_too(harness):
    harness.results = [BAD]
    hc.main([])
    hc.main([])
    harness.mails.clear()

    harness.results = [OK]
    harness.mail_fails = True
    assert hc.main([]) == 2
    harness.mail_fails = False
    assert hc.main([]) == 0
    assert harness.mails == ["[Ask OEKG] recovered"]


def test_test_mail_reports_a_broken_relay_with_exit_2(harness, capsys):
    harness.mail_fails = True
    assert hc.main(["--test-mail"]) == 2
    assert "relay unreachable" in capsys.readouterr().err
    harness.mail_fails = False
    assert hc.main(["--test-mail"]) == 0
    assert harness.mails == ["[Ask OEKG] test mail"]
