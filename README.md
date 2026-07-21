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
- **Live OEP data** — when a result references an OEP dataset, load (on demand) a preview of the **actual table rows** plus an OEMetadata card (units, time/region coverage, license & attribution, sources); OEO ontology terms link to their authoritative definitions.

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
| `oekg/privacy.py` | Privacy policy loading and consent bookkeeping. |
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
| `OEKG_PROMPT_CACHE_KEY` | _(empty)_ | Optional OpenAI prompt-cache routing key. |
| `OEKG_URI_VALIDATION` | `1` | Validate generated URIs against the vocabulary. |
| `OEKG_NORMALISE_OUTPUT` | `1` | Strip stray markdown fences/labels from model output. |
| `OEKG_SYNTAX_VALIDATION` | `1` | Local rdflib syntax pre-flight (advisory). |
| `OEKG_SCHEMA_VALIDATION` | `1` | Flag predicate objects outside their declared range. |
| `OEKG_ENTITY_GROUNDING` | `1` | Resolve entity mentions against real graph labels. |
| `OEKG_GROUNDING_MAX_MENTIONS` | `6` | Max entity mentions grounded per question. |
| `OEKG_GROUNDING_MIN_LEN` | `3` | Min length of a grounded mention. |
| `OEKG_GROUNDING_RESULTS_PER_MENTION` | `5` | Max resolved labels per mention. |
| `OEKG_NEAR_MISS` | `1` | Enrich empty-result retries with real near-miss labels. |
| `OEKG_NEAR_MISS_LIMIT` | `8` | Max near-miss labels fetched. |
| `OEKG_NEAR_MISS_MIN_TOKEN` | `4` | Min mention-token length for near-miss search. |
| `OEKG_SPARQL_ENDPOINT` | OEP OEKG SPARQL URL | SPARQL endpoint. |
| `OEKG_SPARQL_RETRIES` | `3` | Network retries per query. |
| `OEKG_SPARQL_RETRY_DELAY` | `2` | Seconds between retries. |
| `OEKG_SPARQL_TIMEOUT` | `30` | Per-request timeout (seconds). |
| `OEKG_SPARQL_REPAIR_ROUNDS` | `2` | LLM self-correction rounds on endpoint errors. |
| `OEKG_OEP_API_BASE` | OEP REST API base | Base URL for dataset preview/metadata. |
| `OEKG_DATASET_PREVIEW_ROWS` | `50` | Rows fetched for an inline dataset preview. |
| `OEKG_CACHE_ENABLED` | `1` | Enable the question→SPARQL cache. |
| `OEKG_CACHE_PATH` | `query_cache.json` | Cache file location. |
| `OEKG_SEMANTIC_CACHE` | `1` | Offer the closest cached paraphrase (with confirmation). |
| `OEKG_SEMANTIC_CACHE_THRESHOLD` | `0.92` | Min cosine similarity for a semantic match. |
| `OEKG_PRIVACY_POLICY_PATH` | `PRIVACY.md` | Policy shown in the consent gate. |
| `OEKG_REQUIRE_CONSENT` | `1` | Require explicit consent before the chatbot can be used. |
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

The app ships with its own privacy policy, [`PRIVACY.md`](PRIVACY.md), and
**gates itself behind an explicit consent checkbox**: the policy is shown on
first use and neither the language model, the shared question cache nor the
saved-questions list is reachable until the user accepts it. Consent lives in
the Streamlit session only (no cookie, no client-side storage) and can be
withdrawn at any time from the sidebar.

Consent is bound to a fingerprint of the policy text, so **editing `PRIVACY.md`
automatically re-prompts** everyone for renewed consent.

In short:

- Your questions, relevant context, and parts of the knowledge graph are sent to the OpenAI API (US/EU servers).
- `logs/requests.log` records the **text of every question** (no IP address or user identifier).
- `query_cache.json` is **shared between all users**; entries are only written on an explicit user action.
- **Do not submit personal, confidential, or sensitive information.**

> **Operators:** `PRIVACY.md` names the controller of the reference deployment
> and describes *this* code's behaviour. If you run a modified or independently
> hosted instance, review it, adjust the controller and retention details, and
> point `OEKG_PRIVACY_POLICY_PATH` at your version. It is a starting point, not
> legal advice.

---

## 📄 License

[Apache License 2.0](LICENSE) – see LICENSE file for details.
