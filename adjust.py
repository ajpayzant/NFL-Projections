"""The Depth charts tab's edits: where they are stored, and how they move the projections at view time.

An edit per team-week: {"qb": a quarterback to start instead of the model's, "out": players ruled out,
"in": players added}. live.py applies them when it rebuilds the slate (the engine re-projects the team);
between rebuilds -- and always on the cloud, which has no engine -- they are applied here:

    quarterback swap   the team is replaced by its rows from the backup-QB run (live.backup_lineup: the
                       engine re-projected every team under its likeliest backup, once)
    a player ruled out his targets and carries go to the teammates who are playing, in proportion to their
                       own, and their yards and touchdowns follow at their own per-target / per-carry rates
                       (the arithmetic of the engine's Phase 20 overrides)
    a player added     carries no projection until the next rebuild

Applying is idempotent: a swap already in the rows, or a player already absent, is left alone, so the
same edits can sit in the file after a rebuild that has used them.

Storage: a secret GitHub gist on the cloud (secrets gist_id + github_token; this tool's own file,
nfl_lineups.json, so it can share the NHL / NBA tools' gist), else data/overrides/overrides.json.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import polars as pl
import requests

import config as C

FILE = C.OVERRIDES / "overrides.json"
GIST_FILE = "nfl_lineups.json"
_cache: dict = {"at": 0.0, "data": None}
CACHE_SECS = 10
RECEIVING = ("receptions", "receiving_yards", "receiving_tds")
RUSHING = ("rushing_yards", "rushing_tds")


def _secret(name: str) -> str | None:
    v = os.environ.get(name.upper())
    if v:
        return v
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def backend() -> str:
    return "gist" if _secret("gist_id") and _secret("github_token") else "file"


def _headers() -> dict:
    return {"Authorization": f"Bearer {_secret('github_token')}", "Accept": "application/vnd.github+json"}


def load() -> dict:
    if backend() == "file":
        return json.loads(FILE.read_text()) if FILE.exists() else {}
    if _cache["data"] is not None and time.time() - _cache["at"] < CACHE_SECS:
        return dict(_cache["data"])
    try:
        r = requests.get(f"https://api.github.com/gists/{_secret('gist_id')}", headers=_headers(), timeout=20)
        r.raise_for_status()
        f = r.json().get("files", {}).get(GIST_FILE) or {}
        data = json.loads(f.get("content") or "{}")
    except Exception:
        data = _cache["data"] or {}
    _cache.update(at=time.time(), data=data)
    return dict(data)


def key(meta: dict, team: str) -> str:
    return f"{meta['season']}:{meta['week']}:{team}"


def save(meta: dict, team: str, edit: dict | None) -> None:
    data = load()
    k = key(meta, team)
    if edit is None or not (edit.get("qb") or edit.get("out") or edit.get("in")):
        data.pop(k, None)
    else:
        data[k] = {"qb": edit.get("qb"), "out": list(edit.get("out") or []), "in": list(edit.get("in") or [])}
    data = {k2: v for k2, v in data.items() if int(k2.split(":")[0]) >= meta["season"]}
    if backend() == "file":
        C.OVERRIDES.mkdir(parents=True, exist_ok=True)
        FILE.write_text(json.dumps(data, indent=1))
        return
    body = {"files": {GIST_FILE: {"content": json.dumps(data, indent=1)}}}
    requests.patch(f"https://api.github.com/gists/{_secret('gist_id')}", headers=_headers(), json=body, timeout=20).raise_for_status()
    _cache.update(at=time.time(), data=data)


def _fantasy(df: pl.DataFrame) -> pl.DataFrame:
    c = lambda n: pl.col(n).fill_null(0.0) if n in df.columns else pl.lit(0.0)
    base = (0.04 * c("passing_yards") + 4 * c("passing_tds") - 2 * c("interceptions") + 0.1 * c("rushing_yards")
            + 6 * c("rushing_tds") + 0.1 * c("receiving_yards") + 6 * c("receiving_tds"))
    out = df
    for name, rp in (("fantasy_ppr", 1.0), ("fantasy_half", 0.5), ("fantasy_std", 0.0)):
        new = base + rp * c("receptions")
        ratio = (new / pl.col(name).clip(lower_bound=1e-6)).fill_nan(1.0)
        out = out.with_columns(*[(pl.col(f"{name}_{q}") * ratio).alias(f"{name}_{q}") for q in ("p10", "p50", "p90")
                                 if f"{name}_{q}" in out.columns], new.alias(name))
    tds = [n for n in ("rushing_tds", "receiving_tds") if n in out.columns]
    return out.with_columns(pl.sum_horizontal([pl.col(n).fill_null(0) for n in tds]).alias("anytime_td"))


def _redistribute(team_rows: pl.DataFrame, out_ids: list[str]) -> pl.DataFrame:
    """Rule players out: their targets and carries to the teammates still playing, in proportion to their own."""
    gone = team_rows.filter(pl.col("player_id").is_in(out_ids))
    keep = team_rows.filter(~pl.col("player_id").is_in(out_ids))
    if gone.is_empty():
        return team_rows
    skill = keep["group"] != "QB"
    for pool, rates, eligible in (("targets", RECEIVING, skill), ("carries", RUSHING, keep["group"] == "RB")):
        if pool not in keep.columns:
            continue
        freed = float(gone[pool].fill_null(0).sum()) if pool in gone.columns else 0.0
        mine = keep[pool].fill_null(0).to_numpy() * eligible.to_numpy()
        if freed <= 0 or mine.sum() <= 0:
            continue
        add = freed * mine / mine.sum()
        factor = np.where(mine > 0, (mine + add) / np.where(mine > 0, mine, 1), 1.0)
        keep = keep.with_columns(pl.Series(pool, keep[pool].fill_null(0).to_numpy() + add),
                                 *[(pl.col(r).fill_null(0) * pl.Series(factor)).alias(r) for r in rates if r in keep.columns])
    return keep


def apply(players: pl.DataFrame, backup: pl.DataFrame, overrides: dict, meta: dict) -> pl.DataFrame:
    out = players
    for k, edit in overrides.items():
        s, w, team = k.split(":")
        if int(s) != meta["season"] or int(w) != meta["week"]:
            continue
        rows = out.filter(pl.col("team") == team)
        qb = edit.get("qb")
        if qb and not rows.filter((pl.col("group") == "QB") & (pl.col("player_id") == qb)).height:
            alt = backup.filter(pl.col("team") == team)
            if alt.filter((pl.col("group") == "QB") & (pl.col("player_id") == qb)).height:
                rows = alt.select([c for c in out.columns if c in alt.columns])
        if edit.get("out"):
            rows = _fantasy(_redistribute(rows, edit["out"]))
        rows = rows.with_columns(pl.lit("manual override").alias("source"))
        out = pl.concat([out.filter(pl.col("team") != team), rows], how="diagonal_relaxed")
    return out
