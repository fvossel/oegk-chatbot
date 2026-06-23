# OEKG-Chatbot

A GPT-powered chatbot for exploring and querying the [Open Energy Knowledge Graph (OEKG)](https://openenergyplatform.org/) from the Open Energy Platform (OEP).

Ask natural-language questions about OEKG data and get answers backed by SPARQL queries that are generated, validated, executed and summarised for you — with the underlying query and an interactive, downloadable result table always one click away.

---

## ✨ Features

- **Natural language → SPARQL** over the OEKG, using retrieval-augmented generation against the ontology.
- **Self-correcting queries** — if the endpoint rejects a query, the error is fed back to the model for a repair attempt; generated identifiers are also checked against the known vocabulary before execution.
- **Interactive results** — answers come with a real, sortable table plus **CSV/Excel download** and clickable links to the underlying OEP resources.
- **Inspectable SPARQL** — the generated query lives in a collapsible panel with copy-to-clipboard and an on-demand plain-language explanation.
- **Saved questions** — re-run previously answered questions instantly from a shared, file-backed cache; add or curate entries manually.

---

## 🏗️ Architecture

The Streamlit entry point [`streamlit_app.py`](streamlit_app.py) only wires things together and renders the UI. All domain logic lives in the `oekg/` package:

| Module | Responsibility |
| --- | --- |
| `oekg/config.py` | Central, env-overridable configuration (`AppConfig`). |
| `oekg/resources.py` | Loads the FAISS index, ontology docs and system prompts. |
| `oekg/retrieval.py` | Embeds a query and returns the top-k ontology documents. |
| `oekg/llm.py` | OpenAI-backed SPARQL generation, summarisation and explanation. |
| `oekg/validation.py` | Offline check of generated URIs against the OEKG vocabulary. |
| `oekg/oep.py` | OEP SPARQL execution, URI resolution and scenario bundling. |
| `oekg/pipeline.py` | End-to-end orchestration (`RagPipeline`). |
| `oekg/cache.py` | Persistent question → SPARQL cache. |
| `oekg/logging_setup.py` | Rotating request/error logs. |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/fvossel/oegk-chatbot.git
cd oegk-chatbot
```

### 2. Install requirements

Use a **Python 3.10+ virtual environment**:

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# or, for an editable install with dev tooling:
pip install -e ".[dev]"
```

### 3. Configure credentials

Copy [`.env.example`](.env.example) to `.env` and fill in your keys (`.env` is gitignored):

```dotenv
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
OEP_TOKEN="YOUR_OEP_API_TOKEN"
```

All other settings have sensible defaults and can be overridden via `OEKG_*` environment variables (see the table below or `.env.example`).

### 4. Resources

The RAG resources ship with the repository and need no manual setup:
`vector_store.faiss`, `classes.json`, `relations_final.json`, `ids.json`,
`sparql_system_prompt.txt`, `summary_system_prompt.txt`, `logo.svg`. The FAISS
index was built with `text-embedding-3-large`; if you change
`OEKG_EMBEDDING_MODEL`, the index must be regenerated to match.

### 5. Run the app

```bash
streamlit run streamlit_app.py
```

The app is served at `http://localhost:8501`. Because `.streamlit/config.toml`
sets `headless = true`, the browser is **not** opened automatically — open the
printed URL yourself. To serve on the privileged port 80 (production), add
`--server.port 80`.

---

## ⚙️ Configuration

All variables are optional except the two credentials. Defaults are shown.

| Variable | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key. |
| `OEP_TOKEN` | — | **Required.** OEP API token. |
| `OEKG_SPARQL_MODEL` | `gpt-5.4-mini` | Model for SPARQL generation. |
| `OEKG_SUMMARY_MODEL` | `gpt-5.4-nano` | Model for summaries/explanations. |
| `OEKG_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model (must match the FAISS index). |
| `OEKG_RETRIEVAL_TOP_K` | `10` | Number of ontology docs retrieved per query. |
| `OEKG_MAX_USER_HISTORY` | `5` | User turns kept as conversation context. |
| `OEKG_MAX_SUMMARY_CHARS` | `10000` | Max result size (chars) that is summarised. |
| `OEKG_MAX_INPUT_CHARS` | `2000` | Max length of a user question. |
| `OEKG_URI_VALIDATION` | `1` | Validate generated URIs against the vocabulary. |
| `OEKG_SPARQL_ENDPOINT` | OEP OEKG SPARQL URL | SPARQL endpoint. |
| `OEKG_SPARQL_RETRIES` | `3` | Network retries per query. |
| `OEKG_SPARQL_RETRY_DELAY` | `2` | Seconds between retries. |
| `OEKG_SPARQL_TIMEOUT` | `30` | Per-request timeout (seconds). |
| `OEKG_SPARQL_REPAIR_ROUNDS` | `2` | LLM self-correction rounds on endpoint errors. |
| `OEKG_CACHE_ENABLED` | `1` | Enable the question→SPARQL cache. |
| `OEKG_CACHE_PATH` | `query_cache.json` | Cache file location. |
| `OEKG_LOG_DIR` | `logs` | Directory for request/error logs. |
| `OEKG_LOG_LEVEL` | `INFO` | Log level. |

---

## 🧪 Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

The test suite covers the deterministic, network-free logic (transforms,
cache, config, URI validation) and the pipeline orchestration with injected
fakes. CI runs lint + tests on every push (see `.github/workflows/ci.yml`).

---

## 🛡️ Privacy

- Your questions, relevant context, and parts of the knowledge graph are sent to the OpenAI API (US/EU servers).
- **Do not submit personal, confidential, or sensitive information.**

---

## 📄 License

[Apache License 2.0](LICENSE) – see LICENSE file for details.
