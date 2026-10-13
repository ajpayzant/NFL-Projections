"""Results tracking for the Results tab: the model's final projections vs the betting lines.

    python tracking.py snapshot     # store this week's final projections (no API credits)
    python tracking.py grade        # grade every finished game of a snapshot week: opening + closing lines

Snapshot keeps the last projection of each game before it kicks off (a later refresh replaces games not
yet started). Grade pulls, from The Odds API's historical data, the opening (10 AM ET on game day) and
closing (10 minutes before kickoff) prices of DraftKings, FanDuel, BetMGM, Caesars and Fanatics for every
game line and prop, prices each exactly as the Edges tab does, and grades it with the box score. About
2 x 12 x 10 = 240 credits per game for props plus the game lines.

Only aggregates are kept (bets, wins, profit and log-loss sums by week, opening / closing price, market,
position, side and edge bucket): no individual prices, since the file is published with the app's data.
Profit is per 1-unit bet at the best of the five books' prices.
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl
import requests

import config as C
import dist
import edges
import odds

TRACK = C.DATA / "tracking"
RESULTS_LIVE = TRACK / "results_live.parquet"
HIST = "https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl"
OPEN_ET = time(10, 0)
CLOSE_BEFORE = timedelta(minutes=10)
BUCKETS = [(-1.0, 0.0, "< 0"), (0.0, 0.02, "0-2%"), (0.02, 0.05, "2-5%"), (0.05, 0.08, "5-8%"),
           (0.08, 0.12, "8-12%"), (0.12, 1.0, "12%+")]
PRIMARY = {"Over", "Yes", "Favourite", "Favourite -"}
ET = ZoneInfo("America/New_York")
STAT = {"player_pass_yds": "passing_yards", "player_pass_tds": "passing_tds", "player_pass_completions": "completions",
        "player_pass_attempts": "attempts", "player_pass_interceptions": "interceptions", "player_rush_yds": "rushing_yards",
        "player_rush_attempts": "carries", "player_receptions": "receptions", "player_reception_yds": "receiving_yards",
        "player_rush_reception_yds": "rush_rec_yards", "player_pass_rush_yds": "pass_rush_yards"}


def _snap(season: int, week: int) -> dict:
    return {k: TRACK / f"snap_{season}w{week:02d}_{k}.parquet" for k in ("players", "games")} | \
        {"done": TRACK / f"graded_{season}w{week:02d}.json"}


def snapshot(verbose: bool = True) -> None:
    meta = json.loads((C.LIVE / "meta.json").read_text())
    f = _snap(meta["season"], meta["week"])
    p, g = pl.read_parquet(C.LIVE / "players.parquet"), pl.read_parquet(C.LIVE / "games.parquet")
    TRACK.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    kick = {r["game_id"]: pd.Timestamp(r["kickoff_et"]).tz_localize(ET) if pd.Timestamp(r["kickoff_et"]).tzinfo is None
            else pd.Timestamp(r["kickoff_et"]) for r in g.select("game_id", "kickoff_et").to_dicts()}
    started = {k for k, v in kick.items() if v <= now}
    for k, new in (("players", p), ("games", g)):
        if f[k].exists() and started:
            old = pl.read_parquet(f[k])
            new = pl.concat([old.filter(pl.col("game_id").is_in(list(started))),
                             new.filter(~pl.col("game_id").is_in(list(started)))], how="diagonal_relaxed")
        new.write_parquet(f[k])
    if verbose:
        print(f"tracking: snapshot {meta['season']} week {meta['week']}: {g.height} games ({len(started)} started, kept)")


CREDITS = {"used": 0}


def _hist(path: str, key: str, **params) -> dict:
    r = requests.get(HIST + path, params={"apiKey": key, "bookmakers": ",".join(odds.BOOKS), "oddsFormat": "american",
                                          **params}, timeout=60)
    r.raise_for_status()
    CREDITS["used"] += int(r.headers.get("x-requests-last") or 0)
    return r.json()


def _odds_for(games: pd.DataFrame, key: str) -> dict[str, pd.DataFrame]:
    out = {"open": [], "close": []}
    want = {edges.nick(h) + "|" + edges.nick(a) for h, a in zip(games.home_name, games.away_name)}
    for g in games.itertuples():
        ko = pd.Timestamp(g.kickoff_et)
        ko = ko.tz_localize(ET) if ko.tzinfo is None else ko
        stamps = {"open": datetime.combine(ko.date(), OPEN_ET, ET).astimezone(timezone.utc),
                  "close": (ko - CLOSE_BEFORE).astimezone(timezone.utc)}
        for tag, when in stamps.items():
            if tag == "open" and when >= stamps["close"]:
                continue
            ds = when.strftime("%Y-%m-%dT%H:%M:%SZ")
            evs = [e for e in _hist("/events", key, date=ds).get("data", [])
                   if edges.nick(e["home_team"]) + "|" + edges.nick(e["away_team"]) in want
                   and edges.nick(e["home_team"]) == edges.nick(g.home_name)]
            for e in evs:
                try:
                    ev = _hist(f"/events/{e['id']}/odds", key, date=ds,
                               markets=",".join(odds.GAME_MARKETS + odds.PROP_MARKETS)).get("data")
                except requests.HTTPError:
                    continue
                if ev:
                    out[tag] += odds._rows(ev, ds)
    return {k: pd.DataFrame(v) for k, v in out.items()}


def _outcomes(b: pd.DataFrame, season: int, week: int) -> pd.Series:
    C.engine_on_path()
    from src.data import storage
    pg = storage.load("player_games", layer="processed", seasons=[season]).filter(pl.col("week") == week).to_pandas()
    pg["rush_rec_yards"] = pg.rushing_yards.fillna(0) + pg.receiving_yards.fillna(0)
    pg["pass_rush_yards"] = pg.passing_yards.fillna(0) + pg.rushing_yards.fillna(0)
    pg["td"] = pg.rushing_tds.fillna(0) + pg.receiving_tds.fillna(0)
    pg = pg.set_index(["game_id", "player_id"])
    gm = storage.load("games", layer="processed", seasons=[season]).to_pandas().set_index("game_id")
    out = []
    for r in b.itertuples(index=False):
        if r.market in edges.GAME:
            g = gm.loc[r.game_id]
            if pd.isna(g.home_score):
                out.append(np.nan)
                continue
            if r.market == "totals":
                t = g.home_score + g.away_score
                out.append(np.nan if t == r.point else float((t > r.point) == (r.name == "Over")))
                continue
            home = edges.nick(r.name) == edges.nick(r.home_name)
            margin = (g.home_score - g.away_score) * (1 if home else -1)
            v = margin if r.market == "h2h" else margin + r.point
            out.append(np.nan if v == 0 else float(v > 0))
            continue
        k = (r.game_id, r.pid)
        if r.pid is None or k not in pg.index:
            out.append(np.nan)          # did not play: books void the prop
            continue
        if r.market == "player_anytime_td":
            scored = pg.at[k, "td"] >= 1
            out.append(float(scored) if r.name == "Yes" else float(not scored))
            continue
        v = pg.at[k, STAT[r.market]]
        out.append(np.nan if v == r.point else float((v > r.point) == (r.name == "Over")))
    return pd.Series(out, index=b.index)


def aggregate(b: pd.DataFrame, source: str) -> pd.DataFrame:
    b = b.dropna(subset=["won", "p_model", "p_mkt", "dec"]).copy()
    b["edge"] = b.p_model - b.p_mkt
    b["bucket"] = pd.cut(b.edge, [x[0] for x in BUCKETS] + [1.0], labels=[x[2] for x in BUCKETS], right=False)
    b["pnl"] = np.where(b.won == 1, b.dec - 1, -1.0)
    pm, pk = np.clip(b.p_model, 1e-6, 1 - 1e-6), np.clip(b.p_mkt, 1e-6, 1 - 1e-6)
    b["ll_model"] = -(b.won * np.log(pm) + (1 - b.won) * np.log(1 - pm))
    b["ll_mkt"] = -(b.won * np.log(pk) + (1 - b.won) * np.log(1 - pk))
    b["primary"] = b.side.isin(PRIMARY)
    a = b.groupby(["week", "snap", "kind", "position", "side", "bucket", "primary"], observed=True).agg(
        bets=("won", "size"), wins=("won", "sum"), pnl=("pnl", "sum"), ll_model=("ll_model", "sum"),
        ll_mkt=("ll_mkt", "sum")).reset_index()
    a["bucket"] = a.bucket.astype(str)
    return a.assign(source=source)


def grade(verbose: bool = True) -> None:
    key = odds.api_key()
    if not key:
        print("tracking: no ODDS_API_KEY, grading skipped")
        return
    tables = pickle.load(open(C.MODELS / "dist.pkl", "rb"))
    oos = pl.read_parquet(C.MODELS / "gamelines_oos.parquet")
    md = dist.GameDist(oos["pred_margin"].to_numpy(), oos["margin"].to_numpy())
    td = dist.GameDist(oos["pred_total"].to_numpy(), oos["total"].to_numpy(), bandwidth=2.5)
    for f in sorted(TRACK.glob("snap_*_games.parquet")):
        tag = f.name[5:12]                      # 2026w06
        season, week = int(tag[:4]), int(tag[5:])
        files = _snap(season, week)
        games = pl.read_parquet(files["games"]).to_pandas()
        done = json.loads(files["done"].read_text()) if files["done"].exists() else []
        C.engine_on_path()
        from src.data import storage
        scores = storage.load("games", layer="processed", seasons=[season]).to_pandas().set_index("game_id")
        games["home_score"] = games.game_id.map(scores.home_score)
        finished = games[games.home_score.notna() & ~games.game_id.isin(done)]
        if finished.empty:
            continue
        players = pl.read_parquet(files["players"]).to_pandas()
        parts = []
        for snap, rows in _odds_for(finished, key).items():
            if rows.empty:
                continue
            df, _ = edges.compare(rows, finished, players, tables, md, td, now=pd.Timestamp("1970-01-01", tz="UTC"))
            if df.empty:
                continue
            df = df[df.book_p.notna()]
            k = ["game_id", "market", "name", "player", "point"]
            b = df.groupby(k, dropna=False).agg(p_model=("p_model", "first"), p_mkt=("book_p", "mean"), dec=("dec", "max"),
                                                kind=("kind", "first"), position=("position", "first"), pid=("pid", "first"),
                                                home_name=("game_id", lambda x: finished.set_index("game_id").home_name.get(x.iloc[0]))).reset_index()
            b["side"] = np.where(b.market.isin(["totals"]) | b.market.str.startswith("player_"), b.name,
                                 np.where(b.p_mkt >= 0.5, "Favourite", "Underdog"))
            sp = b.market == "spreads"
            b.loc[sp, "side"] = np.where(b.point[sp] < 0, "Favourite -", "Underdog +")
            b["won"] = _outcomes(b, season, week)
            parts.append(b.assign(snap=snap, week=week))
        if parts:
            agg = aggregate(pd.concat(parts, ignore_index=True), "live")
            old = pd.read_parquet(RESULTS_LIVE) if RESULTS_LIVE.exists() else pd.DataFrame()
            pd.concat([old, agg], ignore_index=True).to_parquet(RESULTS_LIVE, index=False)
            if verbose:
                print(f"tracking: graded {season} week {week}: {len(finished)} games, {int(agg.bets.sum()):,} bet sides, "
                      f"profit {agg.pnl.sum():+.1f} u ({CREDITS['used']:,} credits)")
        files["done"].write_text(json.dumps(done + finished.game_id.tolist()))


if __name__ == "__main__":
    try:
        {"snapshot": snapshot, "grade": grade}[sys.argv[1]]()
    except Exception as e:   # never fail the refresh over tracking
        print(f"tracking: {sys.argv[1:]} failed: {e!r}", file=sys.stderr)
