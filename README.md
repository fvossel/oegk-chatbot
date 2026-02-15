# OEKG-Chatbot

A GPT-powered chatbot for exploring and querying the [Open Energy Knowledge Graph (OEKG)](https://openenergyplatform.org/) from the Open Energy Platform (OEP).

This chatbot enables users to ask natural language questions about OEKG data and receive answers backed by SPARQL queries, using modern language models as reasoning engines.

---

## 🔗 Open Energy Platform

For authoritative data and more information, visit the [Open Energy Platform (OEP)](https://openenergyplatform.org/).

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/oekg-chatbot.git
cd oekg-chatbot
```

### 2. Install Requirements

We recommend using a **Python 3.10+ virtual environment**:

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure OpenAI and OEP Credentials

Modify the .env file at `root` level with:

```.env
OEP_TOKEN = "YOUR_OEP_API_TOKEN"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
```

### 4. Prepare Resources

Ensure these files are present (or update their paths in the configuration as needed):

- `logo.svg`
- Knowledge graph index/resources

### 5. Run the Chatbot App

```bash
streamlit run streamlit_app.py --server.port 80
```

The app should open in your browser at `http://localhost:80`.

---

## 🛡️ Privacy

- Your questions, relevant context, and parts of the knowledge graph are sent to the OpenAI API (US/EU servers).
- **Do not submit personal, confidential, or sensitive information.**

---

## ❓ Support

For questions about the knowledge graph, visit [openenergyplatform.org](https://openenergyplatform.org/).

For issues with this chatbot, please open an issue in this GitHub repository.

---

## 📄 License

[Apache License 2.0](LICENSE) – see LICENSE file for details.
