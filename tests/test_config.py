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


def test_chat_and_embeddings_default_to_the_university_gateway():
    # PRIVACY.md section 3.1 names the gateway as the only service and claims it
    # is operated by the university; that holds only while the hosts point there.
    cfg = AppConfig()
    assert cfg.llm_base_url == cfg.embedding_base_url == "https://litellm.uni-osnabrueck.de/v1"
    assert cfg.llm_base_url.split("/")[2].endswith("uni-osnabrueck.de")
    assert cfg.embedding_model == "bge-m3"


def test_one_chat_model_serves_every_task():
    # Reasoning is a per-request switch on the gateway, so "think only for SPARQL
    # generation" needs no second model -- but it does need the switch on.
    cfg = AppConfig()
    assert cfg.sparql_model == cfg.summary_model == "Qwen/Qwen3.5-122B-A10B-FP8"
    assert cfg.no_think_enabled is True
    # A reasoning pass over the large SPARQL prompt can take minutes, so a short
    # timeout would turn slow answers into failures.
    assert cfg.llm_timeout >= 300


def test_embedding_key_does_not_fall_back_to_the_chat_key(monkeypatch):
    # Chat and embeddings are configured separately; a missing embedding key must
    # be an error, not a silent reuse of the chat credential.
    monkeypatch.setenv("LLM_API_KEY", "chat-test-key")
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    cfg = AppConfig()
    assert cfg.llm_api_key == "chat-test-key"
    assert cfg.embedding_api_key is None

    monkeypatch.setenv("EMBEDDING_API_KEY", "gateway-test-key")
    assert AppConfig().embedding_api_key == "gateway-test-key"
