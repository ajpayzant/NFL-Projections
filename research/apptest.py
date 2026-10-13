"""Headless smoke test of every app page: python -m research.apptest"""
import os
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
for page in ("slate", "game", "depth", "player", "edges", "fantasy", "results", "model"):
    os.environ["APP_PAGE"] = page
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    print(f"{page:8s} exceptions: {[e.value for e in at.exception][:2]} | tables: {len(at.dataframe)} "
          f"| errors: {[e.value for e in at.error]} | title: {[t.value for t in at.title]}", flush=True)
