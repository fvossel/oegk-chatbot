"""Streamlit entry point for the OEKG chatbot.

This module is intentionally thin: it wires together the components from the
:mod:`oekg` package and renders the UI. All domain logic lives in that package.
"""

from dotenv import load_dotenv

load_dotenv(override=True)

import base64
import io
import logging
import uuid

import pandas as pd
import streamlit as st

from oekg.cache import QueryCache
from oekg.config import get_config
from oekg.logging_setup import configure_logging, get_request_logger
from oekg.pipeline import RagPipeline
from oekg.privacy import (
    CONSENT_STATE_KEY,
    build_consent_record,
    consent_is_current,
    load_policy,
    policy_version,
)
from oekg.resources import load_resources

# ==============================
#         CONFIG SECTION
# ==============================

CONFIG = get_config()
configure_logging(CONFIG)
logger = logging.getLogger(__name__)
# Consent events go to the request log so the controller can demonstrate that
# the gate was in effect (Art. 7 (1) GDPR). They carry no user identifier.
consent_logger = get_request_logger()

APP_TITLE = "Ask OEKG"
_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_CORRECTION_NOTE = "ℹ️ The query was automatically refined before returning these results."


# ==============================
#       CACHED RESOURCES
# ==============================


@st.cache_resource(show_spinner=False)
def get_query_cache() -> QueryCache:
    return QueryCache(CONFIG.cache_path, enabled=CONFIG.cache_enabled)


@st.cache_resource(show_spinner="Setting up Language Model and document retrieval...")
def get_pipeline() -> RagPipeline:
    resources = load_resources(CONFIG)
    return RagPipeline(resources, get_query_cache(), CONFIG)


# ==============================
#            HELPERS
# ==============================


def trim_history() -> None:
    """Keep only the last ``max_user_history`` user turns (plus responses)."""
    user_count = 0
    trimmed: list[dict] = []
    for turn in reversed(st.session_state.chat_history):
        if turn["role"] == "user":
            user_count += 1
        trimmed.insert(0, turn)
        if user_count == CONFIG.max_user_history:
            break
    st.session_state.chat_history = trimmed


def _link_column_config(df: pd.DataFrame) -> dict:
    """Render columns that hold OEP/HTTP URLs as clickable links."""
    config = {}
    for col in df.columns:
        values = df[col].dropna().astype(str)
        if len(values) and (values.str.startswith("http").mean() > 0.5):
            config[col] = st.column_config.LinkColumn(col)
    return config


def render_results_table(df: pd.DataFrame, turn_id: str) -> None:
    """Show the result DataFrame interactively, with CSV/Excel download buttons."""
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=_link_column_config(df) or None,
    )
    csv_col, xlsx_col = st.columns(2)
    csv_col.download_button(
        "⬇️ Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="oekg_results.csv",
        mime="text/csv",
        key=f"csv_{turn_id}",
        use_container_width=True,
    )
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Results")
        xlsx_col.download_button(
            "⬇️ Download Excel",
            buffer.getvalue(),
            file_name="oekg_results.xlsx",
            mime=_XLSX_MIME,
            key=f"xlsx_{turn_id}",
            use_container_width=True,
        )
    except Exception:  # noqa: BLE001 - Excel export is a nice-to-have
        logger.warning("Excel export unavailable", exc_info=True)


def render_sparql_panel(full_query: str, turn_id: str, pipeline: RagPipeline) -> None:
    """Show the generated SPARQL in a collapsible, copyable panel with an explainer."""
    with st.expander("🔍 Show the SPARQL query"):
        st.code(full_query, language="sparql")
        explanation_key = f"explanation_{turn_id}"
        if st.button("🧠 Explain this query in plain language", key=f"explain_{turn_id}"):
            with st.spinner("Explaining query..."):
                st.session_state[explanation_key] = (
                    pipeline.explain_sparql(full_query) or "_No explanation available._"
                )
        if st.session_state.get(explanation_key):
            st.markdown(st.session_state[explanation_key])


def _render_metadata_card(card: dict, url: str) -> None:
    """Render the OEMetadata info card for one dataset."""
    if not card:
        st.info("No metadata available for this dataset.")
        return
    if card.get("description"):
        st.markdown(card["description"])
    bullets = []
    if card.get("region"):
        region = card["region"]
        if card.get("resolution"):
            region += f" ({card['resolution']})"
        bullets.append(f"**Region:** {region}")
    if card.get("temporal"):
        bullets.append(f"**Time coverage:** {card['temporal']}")
    lic = card.get("license") or {}
    if lic.get("name"):
        text = f"[{lic['name']}]({lic['path']})" if lic.get("path") else lic["name"]
        if lic.get("attribution"):
            text += f" — {lic['attribution']}"
        bullets.append(f"**License:** {text}")
    if card.get("badge"):
        bullets.append(f"**Review badge:** {card['badge']}")
    if card.get("n_fields"):
        bullets.append(f"**Columns:** {card['n_fields']}")
    if bullets:
        st.markdown("\n".join(f"- {b}" for b in bullets))
    sources = card.get("sources") or []
    if sources:
        rendered = " · ".join(
            f"[{s['title']}]({s['path']})" if s.get("path") else s["title"] for s in sources
        )
        st.caption("Sources: " + rendered)
    st.markdown(f"[↗ Open this dataset on OEP]({url})")


def _render_data_preview(rows: pd.DataFrame, table: str, key: str) -> None:
    """Render the actual dataset rows plus a quick chart and download."""
    if rows is None or rows.empty:
        st.info("No data rows available (the table may be empty or restricted).")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(f"Showing the first {len(rows)} rows of `{table}`.")
    st.download_button(
        "⬇️ Download preview (CSV)",
        rows.to_csv(index=False).encode("utf-8"),
        file_name=f"{table}.csv",
        mime="text/csv",
        key=f"dsdl_{key}",
        use_container_width=True,
    )
    numeric_cols = [c for c in rows.columns if pd.api.types.is_numeric_dtype(rows[c])]
    if numeric_cols:
        chosen = st.selectbox("📈 Quick chart — numeric column", numeric_cols, key=f"dschart_{key}")
        st.bar_chart(rows[chosen])


def render_dataset_section(df: pd.DataFrame, turn_id: str, pipeline: RagPipeline) -> None:
    """List OEP datasets referenced in a result; load each one's data on demand."""
    references = pipeline.dataset_references(df)
    if not references:
        return
    st.caption("📊 **Referenced OEP datasets** — open one to load its metadata and a data preview.")
    for index, (url, table) in enumerate(references):
        with st.expander(f"📂 {table}"):
            state_key = f"ds_{turn_id}_{index}"
            if st.button("Load metadata & data preview", key=f"dsload_{state_key}"):
                with st.spinner(f"Loading {table} from OEP..."):
                    st.session_state[state_key] = {
                        "card": pipeline.dataset_metadata(table),
                        "rows": pipeline.dataset_preview(table),
                        "url": url,
                    }
            loaded = st.session_state.get(state_key)
            if loaded:
                _render_metadata_card(loaded["card"], loaded["url"])
                _render_data_preview(loaded["rows"], table, state_key)


def render_ontology_terms(df: pd.DataFrame, pipeline: RagPipeline) -> None:
    """Render a glossary line linking OEO terms in a result to their definitions."""
    terms = pipeline.ontology_links(df)
    if not terms:
        return
    links = " · ".join(f"[{t['label']}]({t['url']})" for t in terms)
    st.markdown(f"🧬 **Ontology terms:** {links}")


def render_assistant_extras(turn: dict, pipeline: RagPipeline) -> None:
    """Render the correction note, SPARQL panel, result table and OEP extras."""
    if turn.get("corrected"):
        st.caption(_CORRECTION_NOTE)
    if turn.get("full_query"):
        render_sparql_panel(turn["full_query"], turn["turn_id"], pipeline)
    df = turn.get("results_df")
    if df is not None:
        render_results_table(df, turn["turn_id"])
        render_dataset_section(df, turn["turn_id"], pipeline)
        render_ontology_terms(df, pipeline)


def render_assistant_turn(turn: dict, pipeline: RagPipeline) -> None:
    with st.chat_message("assistant"):
        st.markdown(turn["content"])
        render_assistant_extras(turn, pipeline)


def handle_question(
    question: str,
    pipeline: RagPipeline,
    prefer_cache: bool = True,
    forced_sparql: str | None = None,
) -> None:
    """Run a question through the pipeline and update the chat history/UI."""
    # Drop any stale "cache this question" offer from a previous turn.
    st.session_state.pop("last_cacheable", None)

    history = [(t["role"], t["content"]) for t in st.session_state.chat_history]
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    result = pipeline.answer(
        question,
        history=history,
        status=st.spinner,
        prefer_cache=prefer_cache,
        stream=True,
        forced_sparql=forced_sparql,
    )

    turn_id = uuid.uuid4().hex[:12]
    with st.chat_message("assistant"):
        if result.summary_stream is not None:
            content = st.write_stream(result.summary_stream) or "The results are shown below."
        else:
            content = result.answer
            st.markdown(content)
        turn = {
            "role": "assistant",
            "content": content,
            "full_query": result.full_query,
            "sparql": result.sparql,
            "results_df": result.results_df,
            "corrected": result.corrected,
            "turn_id": turn_id,
        }
        render_assistant_extras(turn, pipeline)

    st.session_state.chat_history.append(turn)

    # Remember the latest query-backed turn so the user can choose to cache it.
    # (Bot-information replies and cache hits have nothing new to save.)
    if result.sparql and not result.is_bot_message and not result.cache_hit:
        st.session_state.last_cacheable = {"question": question, "sparql": result.sparql}

    trim_history()


def render_consent_gate() -> bool:
    """Show the privacy policy and block the app until the user accepts it.

    Returns ``True`` when the session may proceed. Consent lives in the session
    only: nothing is stored on the client, and closing the tab revokes it.
    """
    if not CONFIG.require_consent:
        return True

    policy = load_policy(CONFIG.privacy_policy_path)
    if not policy:
        st.error(
            f"The privacy policy (`{CONFIG.privacy_policy_path}`) could not be read, so "
            "consent cannot be obtained and the chatbot stays disabled. Restore the file "
            "or point `OEKG_PRIVACY_POLICY_PATH` at it."
        )
        logger.error("privacy policy unreadable | path=%s", CONFIG.privacy_policy_path)
        return False

    version = policy_version(policy)
    if consent_is_current(st.session_state.get(CONSENT_STATE_KEY), version):
        return True

    st.subheader("Before you start")
    st.markdown(
        "This chatbot sends what you type to an external language-model API "
        "(OpenAI, processing in the USA). Please read the privacy policy below and "
        "confirm that you agree — the chatbot stays disabled until you do.\n\n"
        "**Do not enter personal, confidential or sensitive information.**"
    )
    with st.expander("📄 Privacy Policy", expanded=True):
        st.markdown(policy)
    st.download_button(
        "⬇️ Download the privacy policy",
        policy.encode("utf-8"),
        file_name="privacy_policy.md",
        mime="text/markdown",
    )
    accepted = st.checkbox(
        "I have read the privacy policy and consent to my input being processed as described in it.",
        key="privacy_consent_checkbox",
    )
    if st.button("Continue to the chatbot", disabled=not accepted, type="primary"):
        st.session_state[CONSENT_STATE_KEY] = build_consent_record(version)
        consent_logger.info("consent | action=given | policy_version=%s", version)
        st.rerun()
    return False


def render_privacy_sidebar() -> None:
    """Let the user re-read the policy and withdraw consent at any time."""
    st.header("🔐 Privacy")
    policy = load_policy(CONFIG.privacy_policy_path)
    if policy:
        with st.expander("📄 Privacy Policy"):
            st.markdown(policy)
    if not CONFIG.require_consent:
        return
    st.caption("Withdrawal takes effect immediately and clears this conversation.")
    if st.button("Withdraw consent", use_container_width=True):
        record = st.session_state.get(CONSENT_STATE_KEY) or {}
        st.session_state.pop(CONSENT_STATE_KEY, None)
        st.session_state.chat_history = []
        st.session_state.pop("semantic_pending", None)
        st.session_state.pop("last_cacheable", None)
        consent_logger.info(
            "consent | action=withdrawn | policy_version=%s", record.get("version")
        )
        st.rerun()


def save_to_cache(question: str, sparql: str, query_cache: QueryCache, pipeline: RagPipeline) -> None:
    """Store a question/query pair plus its embedding (for semantic matching)."""
    try:
        embedding = pipeline.embed(question)
    except Exception:  # noqa: BLE001 - embedding is only needed for semantic match
        logger.warning("Could not compute embedding for cache entry", exc_info=True)
        embedding = None
    query_cache.put(question, sparql, embedding=embedding)


# ==============================
#       STREAMLIT FRONTEND
# ==============================

st.set_page_config(page_title=APP_TITLE, page_icon=str(CONFIG.logo_path))

st.warning(
    "⚠️ **Privacy Notice:**\n\n"
    "- Your questions, selected context, and graph snippets will be sent to an external Language Model API (OpenAI) "
    "(potentially processed in the USA). Do not submit personal, confidential, or sensitive information.\n"
    "- The full privacy policy is shown before you start and stays available in the sidebar."
)

with open(CONFIG.logo_path, "rt") as f:
    svg_logo = f.read()
svgb64 = base64.b64encode(svg_logo.encode("utf-8")).decode()

st.markdown(
    f"""
    <div style='display: flex; align-items: center; gap: 1em; margin-bottom:2em;'>
        <a href='https://openenergyplatform.org/'><img src="data:image/svg+xml;base64,{svgb64}" height="48"/></a>
        <span style='font-size:2em; font-weight: bold;'>Chat with OEKG</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    """
**Notice:**
- The bot's answers may be incomplete or not fully up-to-date.
- Complex queries may not be understood as intended.
- For authoritative and most recent data visit [openenergyplatform.org](https://openenergyplatform.org/).
"""
)

st.markdown(
    """
    <style>
    .oekg-footer {
        position: fixed;
        left: 0; bottom: 10px; width: 100vw;
        text-align: center;
        color: #444;
        font-size: 0.92em;
        padding: 0.7em 0;
        z-index: 9999999;
        background-color:white;
    }
    .oekg-footer a { color: #2473c8; text-decoration: none; font-weight: 600; }
    .oekg-footer a:hover { text-decoration: underline; }
    .block-container { padding-bottom: 70px !important; }
    .streamlit-expanderHeader { overflow-x: auto; white-space: nowrap; }
    .streamlit-chat-message { overflow-x: auto; white-space: nowrap; }
    </style>
    <div class="oekg-footer">
        Powered by <a href="https://openenergyplatform.org/" target="_blank">Open Energy Platform</a> |
        <a href="https://github.com/fvossel/oegk-chatbot" target="_blank">Source</a> |
        <a href="https://openenergyplatform.org/contact/" target="_blank">Contact</a> |
        <a href="https://openenergyplatform.org/legal/privacy_policy/" target="_blank">Privacy Policy</a> |
        <a href="https://openenergyplatform.org/legal/tou/" target="_blank">Terms of Use</a> |
        <a href="https://openenergyplatform.org/legal/privacy_policy/" target="_blank">Imprint</a>
    </div>
    """,
    unsafe_allow_html=True,
)

if not CONFIG.openai_api_key or not CONFIG.oep_token:
    st.error(
        "Missing credentials: set `OPENAI_API_KEY` and `OEP_TOKEN` in your `.env` "
        "(see `.env.example`) and restart the app."
    )
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Nothing below this line may run before the user has accepted the policy --
# the gate sits in front of the model, the cache and the saved questions alike.
if not render_consent_gate():
    st.stop()

pipeline = get_pipeline()
query_cache = get_query_cache()

# --- Sidebar: re-run previously cached questions ------------------------
with st.sidebar:
    st.header("💾 Saved questions")
    st.caption(
        "Pick a previously answered question to re-run it instantly using its "
        "cached SPARQL query. This list is shared by everyone using this instance."
    )
    cached_entries = query_cache.list_questions()
    if cached_entries:
        selected = st.selectbox(
            "Previous questions",
            options=[e.question for e in cached_entries],
            index=None,
            placeholder="Choose a question...",
        )
        if st.button("Run cached query", disabled=selected is None, use_container_width=True):
            st.session_state.pending_question = selected
        if st.button("Clear cache", use_container_width=True):
            query_cache.clear()
            st.rerun()
    else:
        st.caption("No questions cached yet.")

    # Manually add (or overwrite) a question/SPARQL pair in the shared cache.
    with st.expander("➕ Cache a query manually"):
        with st.form("manual_cache_form", clear_on_submit=True):
            manual_question = st.text_input("Question")
            manual_sparql = st.text_area(
                "SPARQL query",
                height=160,
                help="Enter the query without the standard PREFIX block — it is added automatically on execution.",
            )
            saved = st.form_submit_button("Save to cache", use_container_width=True)
        if saved:
            if manual_question.strip() and manual_sparql.strip():
                save_to_cache(manual_question, manual_sparql, query_cache, pipeline)
                logger.info("manual cache entry added | question=%r", manual_question.strip())
                st.success("Saved to cache.")
                st.rerun()
            else:
                st.warning("Please provide both a question and a SPARQL query.")

    st.divider()
    render_privacy_sidebar()

# Show chat history (user & assistant), in order.
for past_turn in st.session_state.chat_history:
    if past_turn["role"] == "user":
        with st.chat_message("user"):
            st.markdown(past_turn["content"])
    else:
        render_assistant_turn(past_turn, pipeline)

try:
    # A pending semantic-cache suggestion takes priority: ask the user to confirm.
    semantic = st.session_state.get("semantic_pending")
    if semantic:
        with st.chat_message("user"):
            st.markdown(semantic["question"])
        with st.chat_message("assistant"):
            st.info(
                f"I don't have an exact saved answer for this, but a similar saved "
                f"question exists:\n\n> **{semantic['match_question']}**\n\n"
                f"(similarity {semantic['similarity']:.0%}). Use its saved query, or "
                "generate a fresh answer?"
            )
            use_col, fresh_col = st.columns(2)
            use_saved = use_col.button("Use the saved query", use_container_width=True)
            generate_fresh = fresh_col.button("Generate a fresh answer", use_container_width=True)
        if use_saved:
            st.session_state.pop("semantic_pending", None)
            handle_question(semantic["question"], pipeline, forced_sparql=semantic["sparql"])
        elif generate_fresh:
            st.session_state.pop("semantic_pending", None)
            handle_question(semantic["question"], pipeline, prefer_cache=False)

    # A cached question selected in the sidebar takes priority this run.
    pending = st.session_state.pop("pending_question", None)
    if pending:
        handle_question(pending, pipeline, prefer_cache=True)

    if prompt := st.chat_input(
        "Ask me something about OEKG...", max_chars=CONFIG.max_input_chars
    ):
        # Offer the closest paraphrase from the cache (if any) before generating.
        match = None
        if query_cache.get(prompt) is None:
            match = pipeline.find_semantic_match(prompt)
        if match is not None:
            entry, similarity = match
            st.session_state.semantic_pending = {
                "question": prompt,
                "match_question": entry.question,
                "sparql": entry.sparql,
                "similarity": similarity,
            }
            st.rerun()
        else:
            handle_question(prompt, pipeline, prefer_cache=True)

    # Explicit caching of the most recent query-backed turn.
    last = st.session_state.get("last_cacheable")
    if last:
        st.caption(
            "Saving stores your question and its query in the **shared** cache, where "
            "everyone using this instance can see and re-run it."
        )
        if st.button(f"💾 Cache this question + query: “{last['question']}”"):
            save_to_cache(last["question"], last["sparql"], query_cache, pipeline)
            logger.info("cached chat turn | question=%r", last["question"])
            st.session_state.pop("last_cacheable", None)
            st.success("Question and query saved to the shared cache.")
            st.rerun()

except Exception:
    error_id = uuid.uuid4().hex[:8]
    st.error(
        f"An unexpected error occurred (reference `{error_id}`). "
        "Please try again or contact the administrators."
    )
    logger.exception("Unhandled exception occurred | error_id=%s", error_id)
