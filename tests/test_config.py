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


def test_chat_and_embeddings_default_to_separate_university_services():
    # PRIVACY.md section 3.1 names two distinct services and claims both are
    # operated by the university; that holds only while both hosts point there.
    cfg = AppConfig()
    assert cfg.llm_base_url == "https://kiwi-secure.uni-osnabrueck.de/api/agents/v1"
    assert cfg.embedding_base_url == "https://litellm.uni-osnabrueck.de/v1"
    for url in (cfg.llm_base_url, cfg.embedding_base_url):
        assert url.split("/")[2].endswith("uni-osnabrueck.de")
    assert cfg.embedding_model == "bge-m3"


def test_sparql_and_summaries_use_different_agents():
    # KiWi binds reasoning to the agent, not the request, so "reason only for
    # SPARQL generation" is expressed by this split and nothing else.
    cfg = AppConfig()
    assert cfg.sparql_model == "agent_NpAmRwLMlX5QBrUiqcwjM"
    assert cfg.summary_model == "agent_7qE8YPFNPQ9KQQInK9FNc"
    assert cfg.sparql_model != cfg.summary_model
    # The reasoning agent was measured above two minutes for one query, so the
    # old 120 s timeout would have turned slow answers into failures.
    assert cfg.llm_timeout >= 300


def test_embedding_key_does_not_fall_back_to_the_chat_key(monkeypatch):
    # A fallback would send KiWi's key to the gateway -- a different service.
    monkeypatch.setenv("LLM_API_KEY", "kiwi-test-key")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    cfg = AppConfig()
    assert cfg.llm_api_key == "kiwi-test-key"
    assert cfg.embedding_api_key is None

    monkeypatch.setenv("EMBEDDING_API_KEY", "gateway-test-key")
    assert AppConfig().embedding_api_key == "gateway-test-key"
