import streamlit as st
import llm_pipeline
from rdflib import Graph
import base64
from bs4 import BeautifulSoup
import requests

# ==============================
#         CONFIG SECTION
# ==============================

# Logo and UI config
LOGO_PATH = "logo.svg"
APP_TITLE = "Ask OEKG"
APP_ICON = LOGO_PATH
HEADER_HEIGHT = 48

# OEKG data sources
OEKG_LOGIN_URL = "https://openenergyplatform.org/accounts/login/"
OEKG_TURTLE_URL = "https://openenergyplatform.org/scenario-bundles/all_in_turtle"
OEKG_FALLBACK_URL = (
    "https://github.com/OpenEnergyPlatform/oekg/raw/3449824246e39c10d0fd66028159b7b10040ce67/"
    "oekg/oekg_rework/output_rework_oekg_final.ttl"
)

# Knowledge Graph format
OEKG_FORMAT = "turtle"

# Chat history limits
MAX_USER_HISTORY = 5

# ==============================

def login_oep(username: str, password: str) -> requests.Session | None:
    """
    Log in to the Open Energy Platform. Returns authenticated Session on success, None on fail.
    """
    try:
        s = requests.Session()
        response = s.get(OEKG_LOGIN_URL)
        soup = BeautifulSoup(response.content, "lxml")
        form = soup.find("form")
        token_input = (
            form.find("input", attrs={"name": "csrfmiddlewaretoken"}) if form else None
        )
        if not token_input:
            return None

        csrf_token = token_input.get("value")
        data = {
            "login": username,
            "password": password,
            "csrfmiddlewaretoken": csrf_token,
        }
        s.post(OEKG_LOGIN_URL, data=data, headers={"Referer": OEKG_LOGIN_URL})
        return s
    except Exception:
        return None

def get_oekg_data() -> str:
    """
    Download the OEKG (Open Energy Knowledge Graph) data.
    Tries login for live data; falls back to public GitHub snapshot if needed.
    """
    username = st.secrets.get("oep_username", "")
    password = st.secrets.get("oep_password", "")
    session = None

    if username and password:
        session = login_oep(username, password)

    url = OEKG_TURTLE_URL if session else OEKG_FALLBACK_URL
    if session is None:
        st.warning(
            "Could not log in to Open Energy Platform. "
            "Loading public OEKG data from GitHub, which might be outdated."
        )
        session = requests.Session()

    try:
        response = session.get(url)
        response.raise_for_status()
        return response.content
    except Exception:
        st.error("Failed to load OEKG data.")
        return ""

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
        Powered by <a href="https://openenergyplatform.org/" target="_blank">
        Open Energy Platform</a>
    </div>
    """,
    unsafe_allow_html=True
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "graph" not in st.session_state:
    with st.spinner("Loading OEKG data..."):
        oekg_data = get_oekg_data()
        g = Graph()
        g.parse(data=oekg_data, format=OEKG_FORMAT)
        st.session_state.graph = g

g = st.session_state.graph

# Show chat history (user & assistant), in order
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# Message input field / core interaction
if prompt := st.chat_input("Ask me something about OEKG..."):
    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner():
        answer = llm_pipeline.call_rag_pipeline(nl_query=prompt, streamlit_module=st, graph=g)

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