import streamlit as st
import llm_pipeline
import base64
from utils.load_rag_data import load_data

# ==============================
#         CONFIG SECTION
# ==============================

# Logo and UI config
LOGO_PATH = "logo.svg"
APP_TITLE = "Ask OEKG"
APP_ICON = LOGO_PATH
HEADER_HEIGHT = 48

# Chat history limits
MAX_USER_HISTORY = 5


# ==============================
#       STREAMLIT FRONTEND
# ==============================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON
)

# Privacy notice (place this PROMINENTLY)
st.warning(
    "⚠️ **Privacy Notice:**\n\n"
    "- Your questions, selected context, and graph snippets will be sent to an external Language Model API (OpenAI) "
    "(potentially processed in the USA). Do not submit personal, confidential, or sensitive information."
)

# Logo for the header
with open(LOGO_PATH, "rt") as f:
    svg_logo = f.read()
svgb64 = base64.b64encode(svg_logo.encode('utf-8')).decode()

st.markdown(
    f"""
    <div style='display: flex; align-items: center; gap: 1em; margin-bottom:2em;'>
        <a href='https://openenergyplatform.org/'><img src="data:image/svg+xml;base64,{svgb64}" height="{HEADER_HEIGHT}"/></a>
        <span style='font-size:2em; font-weight: bold;'>Chat with OEKG</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    """
**Notice:**  
- The bot's answers may be incomplete or not fully up-to-date.  
- Complex queries may not be understood as intended.  
- For authoritative and most recent data visit [openenergyplatform.org](https://openenergyplatform.org/).
""")

st.markdown(
    """
    <style>
    .oekg-footer {
        position: fixed;
        left: 0; bottom: 0; width: 100vw;
        text-align: center;
        color: #444;
        font-size: 0.92em;
        padding: 0.7em 0;
        z-index: 9999;
    }
    .oekg-footer a {
        color: #2473c8;
        text-decoration: none;
        font-weight: 600;
    }
    .oekg-footer a:hover {
        text-decoration: underline;
    }
    .block-container { padding-bottom: 70px !important; }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class="oekg-footer">
        Powered by <a href="https://openenergyplatform.org/" target="_blank">Open Energy Platform</a> | 
        <a href="https://github.com/fvossel/oegk-chatbot" target="_blank">Source</a>
    </div>
    """,
    unsafe_allow_html=True
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "openai_api_key" not in st.session_state or "faiss_index" not in st.session_state or "documents_dict" not in st.session_state or "ids" not in st.session_state or "sparql_system_prompt" not in st.session_state or "summary_system_prompt" not in st.session_state:
    with st.spinner("Setting up Language Model and document retrieval..."):
        openai_api_key, faiss_index, documents_dict, ids, sparql_system_prompt, summary_system_prompt = load_data()
        st.session_state.aopenai_api_key = openai_api_key
        st.session_state.faiss_index = faiss_index
        st.session_state.documents_dict = documents_dict
        st.session_state.ids = ids
        st.session_state.sparql_system_prompt = sparql_system_prompt
        st.session_state.summary_system_prompt = summary_system_prompt



# Show chat history (user & assistant), in order
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)
try:
    # Message input field / core interaction
    if prompt := st.chat_input("Ask me something about OEKG..."):
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        answer = llm_pipeline.call_rag_pipeline(nl_query=prompt, streamlit_module=st, openai_api_key=st.session_state.aopenai_api_key, faiss_index=st.session_state.faiss_index, documents_dict=st.session_state.documents_dict, ids=st.session_state.ids, sparql_system_prompt=st.session_state.sparql_system_prompt, summary_system_prompt=st.session_state.summary_system_prompt, oep_api_token=st.secrets.get("oep_token", ""))

        st.session_state.chat_history.append(("assistant", answer))
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Limit chat history: only store last MAX_USER_HISTORY user turns (plus responses)
        user_count = 0
        new_history = []
        for role, msg in reversed(st.session_state.chat_history):
            if role == "user":
                user_count += 1
            new_history.insert(0, (role, msg))
            if user_count == MAX_USER_HISTORY:
                break
        st.session_state.chat_history = new_history

except Exception:
    st.error("Some major errors occured. Please contact the administrators.")