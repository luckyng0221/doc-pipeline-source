"""Entry point for `streamlit run streamlit_app.py` and Streamlit Community Cloud."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Lets the app run from a fresh checkout even before `pip install -e .`.
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from doc_pipeline.app.ui import main  # noqa: E402

main(ROOT / "config.yaml")
