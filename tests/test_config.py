from oekg.config import AppConfig, _env


def test_env_helper_handles_missing_and_empty(monkeypatch):
    monkeypatch.setenv("OEKG_TEST_VAR", "value")
    assert _env("OEKG_TEST_VAR", "default") == "value"
    monkeypatch.setenv("OEKG_TEST_VAR", "")
    assert _env("OEKG_TEST_VAR", "default") == "default"
    monkeypatch.delenv("OEKG_TEST_VAR", raising=False)
    assert _env("OEKG_TEST_VAR", "default") == "default"


def test_model_env_override(monkeypatch):
    # Model fields use default_factory, so they re-read the environment per instance.
    monkeypatch.setenv("OEKG_SPARQL_MODEL", "custom-sparql-model")
    cfg = AppConfig()
    assert cfg.sparql_model == "custom-sparql-model"


def test_sane_defaults():
    cfg = AppConfig()
    assert cfg.retrieval_top_k == 10
    assert cfg.sparql_repair_rounds == 2
    assert cfg.uri_validation_enabled is True
    assert cfg.request_log_path.name == "requests.log"
    assert cfg.error_log_path.name == "errors.log"
