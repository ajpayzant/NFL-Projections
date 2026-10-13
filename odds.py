"""Sportsbook odds from The Odds API (the-odds-api.com), for the Edges tab.

    python odds.py            # fetch game lines + today's player props -> data/odds/odds.parquet

Odds are fetched only on demand: the Edges tab's "Refresh odds" button (or this script), never
by the scheduled refresh, to save API credits. Needs an API key: odds_api_key in the Streamlit
app's secrets or in .streamlit/secrets.toml locally (or ODDS_API_KEY in the environment).

Credits per fetch: 3 for the game lines of every NFL game (3 markets x 1 region), plus up to 12
per game for player props (one per market a book has posted) -- about 170 for a full week (fetched one game at a time, only for games starting in the next
PROPS_HOURS hours, since books post props on game day).

Rows (one per book, market and outcome):
    event_id, commence_time, home, away, book, market, name, player, point, price, last_update, fetched_at
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

import config as C

API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"
ODDS_DIR = C.DATA / "odds"
ODDS_FILE = ODDS_DIR / "odds.parquet"
# The books the Edges tab compares (The Odds API keys -> titles). Naming books instead of a region
# costs the same (up to 10 books = 1 region) and keeps the fetch and the comparison small.
BOOKS = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
         "fanatics": "Fanatics"}
GAME_MARKETS = ["h2h", "spreads", "totals"]
PROP_MARKETS = ["player_pass_yds", "player_pass_tds", "player_pass_completions", "player_pass_attempts",
                "player_pass_interceptions", "player_rush_yds", "player_rush_attempts", "player_receptions",
                "player_reception_yds", "player_rush_reception_yds", "player_pass_rush_yds", "player_anytime_td"]
PROPS_HOURS = 24 * 7   # NFL books post props days ahead: the whole week's games


def api_key() -> str | None:
    k = os.environ.get("ODDS_API_KEY")
    if k:
        return k
    try:
        import streamlit as st
        return st.secrets.get("odds_api_key")
    except Exception:
        pass
    p = C.ROOT / ".streamlit" / "secrets.toml"   # local runs outside streamlit
    if p.exists():
        import tomllib
        return tomllib.loads(p.read_text()).get("odds_api_key")
    return None


def _get(path: str, key: str, **params) -> tuple[object, dict]:
    r = requests.get(f"{API}{path}", params={"apiKey": key, **params}, timeout=C.REQUEST_TIMEOUT)
    r.raise_for_status()
    usage = {h: r.headers.get(f"x-requests-{h}") for h in ("remaining", "used", "last")}
    return r.json(), usage


def _rows(ev: dict, fetched_at: str) -> list[dict]:
    out = []
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for o in mk.get("outcomes", []):
                out.append({"event_id": ev["id"], "commence_time": ev["commence_time"],
                            "home": ev["home_team"], "away": ev["away_team"],
                            "book": bk.get("title") or bk["key"], "market": mk["key"],
                            "name": o.get("name"), "player": o.get("description"),
                            "point": o.get("point"), "price": o.get("price"),
                            "last_update": mk.get("last_update") or bk.get("last_update"),
                            "fetched_at": fetched_at})
    return out


def fetch(key: str | None = None, verbose: bool = True) -> pd.DataFrame | None:
    """Game lines for every listed game, props for games starting soon. Writes ODDS_FILE."""
    key = key or api_key()
    if not key:
        if verbose:
            print("odds: no ODDS_API_KEY, skipped")
        return None
    now = datetime.now(timezone.utc)
    fetched_at = now.isoformat(timespec="seconds")
    rows = []
    games, usage = _get("/odds", key, bookmakers=",".join(BOOKS), markets=",".join(GAME_MARKETS), oddsFormat="american")
    for ev in games:
        rows += _rows(ev, fetched_at)
    soon = [ev for ev in games
            if datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00")) < now + timedelta(hours=PROPS_HOURS)]
    for ev in soon:
        try:
            one, usage = _get(f"/events/{ev['id']}/odds", key, bookmakers=",".join(BOOKS),
                              markets=",".join(PROP_MARKETS), oddsFormat="american")
            rows += _rows(one, fetched_at)
        except requests.HTTPError as e:  # a game with no props yet: keep the rest
            if verbose:
                print(f"odds: props for {ev['away_team']} @ {ev['home_team']} failed ({e})")
    df = pd.DataFrame(rows)
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ODDS_FILE, index=False)
    if verbose:
        print(f"odds: {len(games)} games, props for {len(soon)}, {len(df):,} prices from "
              f"{df.book.nunique() if len(df) else 0} books; credits left {usage.get('remaining')}")
    return df


def load() -> pd.DataFrame | None:
    """The saved odds, kept to BOOKS (a file fetched before the list was narrowed has more)."""
    if not ODDS_FILE.exists():
        return None
    o = pd.read_parquet(ODDS_FILE)
    return o[o.book.isin(BOOKS.values())].reset_index(drop=True)


if __name__ == "__main__":
    try:
        fetch()
    except Exception as e:  # never fail the refresh over odds
        print(f"odds: fetch failed: {e!r}", file=sys.stderr)
