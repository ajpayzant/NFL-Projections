"""The weekly slate: who plays, every player's projection, fantasy points and the game lines.

    python live.py                 # refresh the engine's data, project the next unplayed week
    python live.py --week 7        # a given week of the current season
    python live.py --no-refresh    # skip the data refresh (re-project from what is on disk)

Steps:
  1. refresh     the engine refetches this season's data and rebuilds its processed tables
  2. who plays   availability.py: P(plays) for every rostered QB/RB/WR/TE and the likeliest QB starter,
                 with the Depth charts tab's edits on top (overrides.json)
  3. features    the engine's feature tables, the scheduled week built on that expected lineup
                 (src.features.expected), so its competition features see who will actually play
  4. project     the engine's forward pass on the lineup, then the correction layer (correct.py)
  5. price       fantasy points (PPR / half / standard) with ranges; game lines from gamelines.py
  6. save        data/live/: players.parquet, games.parquet, candidates.parquet, meta.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

import adjust
import availability as A
import config as C
import correct
import dist
import gamelines as G

C.engine_on_path()
from src.data import storage  # noqa: E402
from src.features import expected  # noqa: E402
from src.projections import week as engine_week  # noqa: E402

ET = ZoneInfo("America/New_York")
STAT_GROUP = {"QB": "QB", "RB": "RB", "WR": "WR+TE", "TE": "WR+TE"}


def _engine(script: list[str]) -> None:
    r = subprocess.run([sys.executable, *script], cwd=C.ENGINE, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    if r.returncode != 0:
        raise RuntimeError(f"engine {' '.join(script)} failed:\n{(r.stdout + r.stderr)[-2000:]}")


def refresh(season: int, wk: int) -> None:
    _engine(["scripts/run_week.py", "--season", str(season), "--week", str(wk), "--only", "update"])


def next_week(season: int = C.CURRENT_SEASON) -> int | None:
    g = storage.load("games", layer="processed", seasons=[season]).filter(
        (pl.col("season_type") == "REG") & ~pl.col("is_played"))
    return int(g["week"].min()) if g.height else None


# ---------------------------------------------------------------- who plays

def who_plays(season: int, wk: int) -> pl.DataFrame:
    """Every candidate with p_play / p_start, the model's call, and the user's edits applied."""
    models = pickle.load(open(A.MODEL_FILE, "rb"))
    pred = A.predict(models, A.frame([season]).filter(pl.col("week") == wk))
    lu = A.lineup(pred).with_columns(pl.lit(True).alias("in_lineup"))
    pred = pred.join(lu, on=["game_id", "team", "player_id"], how="left").with_columns(
        pl.col("in_lineup").fill_null(False))
    pred = pred.with_columns((pl.col("in_lineup") & (pl.col("pos") == "QB")).alias("starter"),
                             pl.lit("model").alias("source"))
    for k, v in adjust.load().items():          # "season:week:team" -> {"out": [...], "in": [...], "qb": id}
        s, w, team = k.split(":")
        if int(s) != season or int(w) != wk:
            continue
        t = pl.col("team") == team
        if v.get("out"):
            pred = pred.with_columns(pl.when(t & pl.col("player_id").is_in(v["out"])).then(False).otherwise(pl.col("in_lineup")).alias("in_lineup"))
        if v.get("in"):
            pred = pred.with_columns(pl.when(t & pl.col("player_id").is_in(v["in"])).then(True).otherwise(pl.col("in_lineup")).alias("in_lineup"))
        if v.get("qb"):
            q = t & (pl.col("pos") == "QB")
            pred = pred.with_columns(
                pl.when(q).then(pl.col("player_id") == v["qb"]).otherwise(pl.col("starter")).alias("starter"))
            pred = pred.with_columns(pl.when(q).then(pl.col("starter")).otherwise(pl.col("in_lineup")).alias("in_lineup"))
        pred = pred.with_columns(pl.when(t).then(pl.lit("manual override")).otherwise(pl.col("source")).alias("source"))
    return pred


# ---------------------------------------------------------------- the slate

def backup_lineup(cand: pl.DataFrame) -> pl.DataFrame:
    """Every team's lineup with its likeliest backup QB starting instead (for the Depth charts tab's QB
    swap: one engine run re-projects every team under its backup, since teams do not interact)."""
    q = cand.filter(pl.col("pos") == "QB").sort("p_start", "p_play", descending=True)
    backup = q.filter(~pl.col("starter")).group_by("game_id", "team").first().select("game_id", "team", "player_id")
    skill = cand.filter(pl.col("in_lineup") & (pl.col("pos") != "QB"))
    return pl.concat([skill, cand.join(backup, on=["game_id", "team", "player_id"], how="semi")], how="diagonal_relaxed")         .with_columns(pl.lit(True).alias("in_lineup"))


def project(season: int, wk: int, cand: pl.DataFrame) -> pl.DataFrame:
    lineup = cand.filter(pl.col("in_lineup"))
    expected.write(lineup.select("season", "week", "team", "player_id", pl.col("pos").alias("position")))
    _engine(["scripts/build_features.py", "--seasons", str(season), "--include-scheduled"])
    fc = engine_week.project(season, wk, lineup=lineup.select("team", "player_id"))
    model = correct.load()
    p = correct.apply(model, fc.players, season, wk)
    p = p.join(cand.select("game_id", "player_id", "p_play", "p_start", "starter", "source", "depth", "inj_out",
                           "inj_doubtful", "inj_questionable", "years_exp"), on=["game_id", "player_id"], how="left")
    # fantasy points and their ranges
    for name, rp in correct.SCORING.items():
        p = p.with_columns(correct.fantasy(p, rp).alias(name))
    parts = []
    for group, g in p.group_by("group"):
        group = group[0]
        cols = {}
        for name in ("fantasy_ppr", "fantasy_half", "fantasy_std"):
            emp = model["dist"][f"{group}|{name}"]["emp"]
            mu = g[name].to_numpy()
            for q in (0.1, 0.5, 0.9):
                cols[f"{name}_p{int(q * 100)}"] = emp.quantile(mu, q)
        tds = [c for c in ("rushing_tds", "receiving_tds") if c in g.columns]
        anytime = g.select(pl.sum_horizontal([pl.col(c).fill_null(0) for c in tds])).to_series().to_numpy()
        cols["anytime_td"] = anytime
        d_any = model["dist"][f"{group}|anytime_td"]
        cols["p_anytime_td"] = dist.td_probability(d_any["cal"], anytime) if "cal" in d_any else d_any["emp"].p_at_least_one(anytime)
        parts.append(g.with_columns([pl.Series(k, v) for k, v in cols.items()]))
    p = pl.concat(parts, how="diagonal")
    games = storage.load("games", layer="processed", seasons=[season]).filter(pl.col("week") == wk)
    opp = pl.concat([games.select("game_id", pl.col("home_team").alias("team"), pl.col("away_team").alias("opp"), pl.lit(True).alias("is_home")),
                     games.select("game_id", pl.col("away_team").alias("team"), pl.col("home_team").alias("opp"), pl.lit(False).alias("is_home"))])
    return p.join(opp, on=["game_id", "team"], how="left").with_columns(pl.lit(season).alias("season"), pl.lit(wk).alias("week"))


def game_lines(season: int, wk: int) -> pl.DataFrame:
    model = pickle.load(open(G.FILE, "rb"))
    rows = G.game_rows([season]).filter(pl.col("week") == wk)
    out = G.predict(model, rows)
    g = storage.load("games", layer="processed", seasons=[season]).select(
        "game_id", "kickoff_et", "home_score", "away_score", "roof", "temp", "wind", "spread_line", "total_line")
    names = pl.read_parquet(C.ENGINE / "data/raw/teams.parquet").select("team_abbr", "team_name").unique("team_abbr")
    out = out.join(names.rename({"team_abbr": "home", "team_name": "home_name"}), on="home", how="left")         .join(names.rename({"team_abbr": "away", "team_name": "away_name"}), on="away", how="left")
    return out.join(g, on="game_id", how="left").with_columns(
        ((pl.col("total_hat") + pl.col("margin_hat")) / 2).alias("home_pts"),
        ((pl.col("total_hat") - pl.col("margin_hat")) / 2).alias("away_pts")).sort("kickoff_et")


def tie_receivers_to_qb(base: pl.DataFrame, alt: pl.DataFrame) -> pl.DataFrame:
    """In the backup-QB run, scale each team's receiving lines by how much its passing changed.

    The engine's receiver chain sizes targets off the quarterbacks' attempts but prices them with the
    receiver's own efficiency, so a backup who throws for 175 yards instead of 263 left his receivers
    on 266 (Dallas, 2026 week 6). Receiving yards, receptions and receiving TDs follow the ratio of the
    backup's passing yards, completions and passing TDs to the starter's, as the team's receiving and
    passing lines agree in the main run.
    """
    q = lambda f, c: f.filter(pl.col("group") == "QB").group_by("team").agg(pl.col(c).sum())
    ratio = None
    for qc, rc in (("passing_yards", "receiving_yards"), ("completions", "receptions"), ("passing_tds", "receiving_tds")):
        r = q(alt, qc).join(q(base, qc).rename({qc: "_b"}), on="team").with_columns(
            (pl.col(qc) / pl.col("_b").clip(lower_bound=1e-6)).clip(0.3, 1.5).alias(f"_r_{rc}")).select("team", f"_r_{rc}")
        ratio = r if ratio is None else ratio.join(r, on="team", how="full", coalesce=True)
    out = alt.join(ratio, on="team", how="left")
    skill = pl.col("group") != "QB"
    out = out.with_columns([pl.when(skill).then(pl.col(rc) * pl.col(f"_r_{rc}").fill_null(1.0)).otherwise(pl.col(rc)).alias(rc)
                            for rc in ("receiving_yards", "receptions", "receiving_tds") if rc in out.columns])
    out = out.drop([c for c in out.columns if c.startswith("_r_")])
    return adjust._fantasy(out)


HISTORY_COLS = ["season", "week", "game_id", "team", "opponent", "player_id", "player_name", "position", "completions",
                "attempts", "passing_yards", "passing_tds", "interceptions", "carries", "rushing_yards", "rushing_tds",
                "targets", "receptions", "receiving_yards", "receiving_tds", "fantasy_points_ppr"]


def history(seasons: int = 3) -> pl.DataFrame:
    """Recent box scores for the Player page (the cloud app has no engine lake)."""
    pg = storage.load("player_games", layer="processed", seasons=list(range(C.CURRENT_SEASON - seasons + 1, C.CURRENT_SEASON + 1)))
    pg = pg.filter(pl.col("position").is_in(list(C.POSITIONS)))
    snaps = storage.load("player_usage", layer="processed", seasons=list(range(C.CURRENT_SEASON - seasons + 1, C.CURRENT_SEASON + 1)))         .select("game_id", "player_id", "snap_share")
    g = storage.load("games", layer="processed").select("game_id", "home_team", "kickoff_et")
    return pg.select([c for c in HISTORY_COLS if c in pg.columns]).join(snaps, on=["game_id", "player_id"], how="left")         .join(g, on="game_id", how="left").with_columns((pl.col("team") == pl.col("home_team")).alias("is_home")).drop("home_team")


def build(season: int | None = None, wk: int | None = None, fetch: bool = True, verbose: bool = True) -> dict:
    season = season or C.CURRENT_SEASON
    wk = wk or next_week(season)
    if wk is None:
        raise SystemExit(f"no unplayed regular-season week left in {season}")
    if fetch:
        refresh(season, wk)
        wk = next_week(season) if wk is None else wk
    cand = who_plays(season, wk)
    players = project(season, wk, cand)
    games = game_lines(season, wk)
    alt = backup_lineup(cand)
    players_alt = project(season, wk, alt.with_columns(pl.lit(True).alias("starter")).with_columns(
        (pl.col("starter") & (pl.col("pos") == "QB")).alias("starter")))
    players_alt = tie_receivers_to_qb(players, players_alt)
    expected_lineup_restore = cand.filter(pl.col("in_lineup"))
    expected.write(expected_lineup_restore.select("season", "week", "team", "player_id", pl.col("pos").alias("position")))
    C.LIVE.mkdir(parents=True, exist_ok=True)
    players_alt.write_parquet(C.LIVE / "players_backup_qb.parquet")
    history().write_parquet(C.LIVE / "history.parquet")
    players.write_parquet(C.LIVE / "players.parquet")
    games.write_parquet(C.LIVE / "games.parquet")
    cand.write_parquet(C.LIVE / "candidates.parquet")
    meta = {"season": season, "week": wk, "built_at": datetime.now(ET).isoformat(timespec="minutes"),
            "players": players.height, "games": games.height}
    (C.LIVE / "meta.json").write_text(json.dumps(meta))
    if verbose:
        print(f"{season} week {wk}: {games.height} games, {players.height} players projected "
              f"({int(cand['in_lineup'].sum())} of {cand.height} rostered expected to play)")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int)
    ap.add_argument("--season", type=int, default=C.CURRENT_SEASON)
    ap.add_argument("--no-refresh", action="store_true")
    a = ap.parse_args()
    build(a.season, a.week, fetch=not a.no_refresh)
    p = pl.read_parquet(C.LIVE / "players.parquet")
    show = ["player_name", "team", "opp", "projected_position", "p_play", "fantasy_ppr", "fantasy_ppr_p10", "fantasy_ppr_p90"]
    print(p.sort("fantasy_ppr", descending=True).select(show).head(15).to_pandas().round(2).to_string(index=False))
    g = pl.read_parquet(C.LIVE / "games.parquet")
    print(g.select("away", "home", "away_pts", "home_pts", "margin_hat", "total_hat", "p_home", "spread_line", "total_line")
          .to_pandas().round(2).to_string(index=False))


if __name__ == "__main__":
    main()
