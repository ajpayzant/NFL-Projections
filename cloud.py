"""Keep a Streamlit Cloud container's data in step with the weekly refresh.

The refresh (live.py, then publish.py) runs where the projection engine lives and uploads its outputs to
this repo's release tagged `data`. The deployed app downloads whatever is newer than its own copy; locally
(where live.py writes the files itself) nothing happens.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import requests

import config as C

DEFAULT_REPO = "ajpayzant/NFL-Projections"
TAG = "data"
ASSETS = {"meta.json": C.LIVE / "meta.json",
          "players.parquet": C.LIVE / "players.parquet",
          "players_backup_qb.parquet": C.LIVE / "players_backup_qb.parquet",
          "games.parquet": C.LIVE / "games.parquet",
          "candidates.parquet": C.LIVE / "candidates.parquet",
          "history.parquet": C.LIVE / "history.parquet",
          "dist.pkl": C.MODELS / "dist.pkl",
          "gamelines_oos.parquet": C.MODELS / "gamelines_oos.parquet",
          "results_live.parquet": C.DATA / "tracking" / "results_live.parquet"}
OPTIONAL = {"results_live.parquet"}
MANIFEST = "manifest.json"
LOCAL_MANIFEST = C.LIVE / "manifest.json"


def _secret(name: str) -> str | None:
    v = os.environ.get(name.upper())
    if v:
        return v
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def enabled() -> bool:
    """On Streamlit Cloud (code under /mount/src) or when a data_repo secret is set."""
    return any("/mount/src" in p for p in sys.path + [os.getcwd()]) or bool(_secret("data_repo"))


def _url(name: str) -> str:
    return f"https://github.com/{_secret('data_repo') or DEFAULT_REPO}/releases/download/{TAG}/{name}"


def _download(name: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(_url(name), stream=True, timeout=120) as r:
        r.raise_for_status()
        fd, tmp = tempfile.mkstemp(dir=dest.parent)
        with os.fdopen(fd, "wb") as fh:
            shutil.copyfileobj(r.raw, fh, length=1 << 20)
    os.replace(tmp, dest)


def sync() -> str | None:
    if not enabled():
        return None
    try:
        remote = requests.get(_url(MANIFEST), timeout=20).json()
    except Exception:
        return None
    local = json.loads(LOCAL_MANIFEST.read_text()) if LOCAL_MANIFEST.exists() else {}
    missing = [n for n, p in ASSETS.items() if not p.exists() and n not in OPTIONAL]
    if remote.get("built_at") != local.get("built_at") or missing:
        try:
            for name, path in ASSETS.items():
                if name not in OPTIONAL:
                    _download(name, path)
        except Exception:
            return local.get("built_at")
        for name in OPTIONAL:
            try:
                _download(name, ASSETS[name])
            except Exception:
                pass
        LOCAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        LOCAL_MANIFEST.write_text(json.dumps(remote))
    return remote.get("built_at")
