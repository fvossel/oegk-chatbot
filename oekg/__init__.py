"""OEKG chatbot core package.

Houses the application's domain logic, split into clear layers:

- :mod:`oekg.config`     - central configuration.
- :mod:`oekg.logging_setup` - logging configuration.
- :mod:`oekg.cache`      - persistent question -> SPARQL cache.
- :mod:`oekg.resources`  - loading of static resources (index, prompts, ...).
- :mod:`oekg.retrieval`  - semantic document retrieval.
- :mod:`oekg.llm`        - OpenAI-backed SPARQL generation and summarisation.
- :mod:`oekg.oep`        - Open Energy Platform SPARQL execution.
- :mod:`oekg.pipeline`   - end-to-end orchestration.

The Streamlit entry point (``streamlit_app.py``) only wires these together
and renders the UI.
"""

from oekg.config import AppConfig, get_config
from oekg.pipeline import PipelineResult, RagPipeline

__all__ = [
    "AppConfig",
    "get_config",
    "RagPipeline",
    "PipelineResult",
]
