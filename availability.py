"""Who plays: the chance each rostered QB / RB / WR / TE records an offensive snap, and which QB starts.

The engine's chain was fitted and replayed on players who played, and it shares team volume out
within the team, so the forward pass must be told who is going to be on the field (week.project's
`lineup`). Projecting the whole roster gave week 6 of 2026's starting quarterbacks 0.715 of their
team's dropbacks against 0.966 in the replay.

Candidates are every QB / RB / WR / TE on the team's weekly roster as active or game-day inactive
(the inactive list is only known 90 minutes before kickoff, so it is a candidate, never a feature).
Labels: played = an offensive snap or a touch; started (QBs) = the team's dropback leader.
Features are pregame only:
    his recent participation      played last game, snap share last game and weighted, the share of
                                  his last four rostered games he played, games since he played
    the depth chart               his rank at his position in the last snapshot before kickoff
    the injury report             final game status and practice participation for the week
    quarterbacks                  started last game, weighted dropback share
    context                       position, experience, a new team, week of season
LightGBM, walk-forward by season.

    python availability.py            # fit on 2016-2025, score walk-forward 2019-2026, save the models
"""
from __future__ import annotations

import json
import pickle
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

import config as C

C.engine_on_path()
from src.data import storage  # noqa: E402

# Two feature sets. The week's injury report is published Wednesday to Friday; before it is, a player
# with no entry is not healthy, he is unreported -- and a model trained on weeks that all had their report
# read Lamar Jackson (inactive in week 5, nothing reported yet for week 6) as an 86% starter. So the early
# model sees last week's report and whether he was inactive, and the report model adds this week's.
EARLY = ["pos_QB", "pos_RB", "pos_WR", "pos_TE", "years_exp", "week", "n_prior", "team_change",
         "prev_played", "prev_snap", "ew_snap", "played_rate_4", "season_share", "games_since",
         "depth", "depth_missing", "qb_prev_start", "qb_ew_share", "prev_inactive", "prev_inj_out",
         "prev_inj_doubtful", "prev_inj_questionable"]
REPORT = EARLY + ["inj_out", "inj_doubtful", "inj_questionable", "prac_dnp", "prac_limited", "prac_full"]
FEATS = REPORT
PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1}
ROUNDS = 400
OFFENSE_SLOTS = {"QB", "RB", "FB", "HB", "WR", "LWR", "RWR", "SWR", "SLOT", "TE", "TE1", "TE2"}
MODEL_FILE = C.MODELS / "availability.pkl"


# ---------------------------------------------------------------- inputs

def schedule(seasons: list[int]) -> pl.DataFrame:
    """One row per team-game (played or scheduled) with its kickoff."""
    g = storage.load("games", layer="processed", seasons=seasons).filter(pl.col("season_type") == "REG")
    sides = [g.select("game_id", "season", "week", "kickoff_et", "is_played", pl.col(a).alias("team"))
             for a in ("home_team", "away_team")]
    return pl.concat(sides)


def candidates(seasons: list[int]) -> pl.DataFrame:
    """Rostered QB/RB/WR/TE per team-game. A week whose roster is not published yet reads the team's
    latest one (as the engine's feature build does)."""
    wr = storage.load("weekly_rosters", layer="processed", seasons=seasons).filter(
        pl.col("status").is_in(list(C.CANDIDATE_STATUSES)) & pl.col("projected_position").is_in(list(C.POSITIONS))
    ).select("season", pl.col("week").alias("roster_week"), "team", "player_id", "player_name",
             pl.col("projected_position").alias("pos"), "years_exp", "status")
    published = wr.select("season", "team", "roster_week").unique().sort("roster_week")
    sch = schedule(seasons).sort("week")
    sch = sch.join_asof(published, left_on="week", right_on="roster_week", by=["season", "team"], strategy="backward")
    return sch.join(wr, on=["season", "team", "roster_week"], how="inner").drop("roster_week")


def outcomes(seasons: list[int]) -> pl.DataFrame:
    u = storage.load("player_usage", layer="processed", seasons=seasons).select(
        "game_id", "team", "player_id", pl.col("offense_snaps").fill_null(0), pl.col("snap_share").fill_null(0),
        (pl.col("carries").fill_null(0) + pl.col("targets").fill_null(0)).alias("touches"))
    q = storage.load("passer_games", layer="processed", seasons=seasons).select("game_id", "team", "player_id", "dropbacks")
    q = q.with_columns((pl.col("dropbacks") / pl.col("dropbacks").sum().over("game_id", "team")).alias("db_share"),
                       (pl.col("dropbacks") == pl.col("dropbacks").max().over("game_id", "team")).alias("started"))
    return u.join(q, on=["game_id", "team", "player_id"], how="full", coalesce=True).with_columns(
        ((pl.col("offense_snaps").fill_null(0) > 0) | (pl.col("touches").fill_null(0) > 0)
         | (pl.col("dropbacks").fill_null(0) > 0)).alias("played"),
        pl.col("started").fill_null(False), pl.col("db_share").fill_null(0.0), pl.col("snap_share").fill_null(0.0))


def injuries(seasons: list[int]) -> pl.DataFrame:
    i = storage.load("injuries", seasons=seasons)
    i = i.filter(pl.col("game_type") == "REG") if "game_type" in i.columns else i
    st = pl.col("report_status").fill_null("")
    pr = pl.col("practice_status").fill_null("")
    return i.select(
        pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32), "team", pl.col("gsis_id").alias("player_id"),
        (st == "Out").cast(pl.Float64).alias("inj_out"), (st == "Doubtful").cast(pl.Float64).alias("inj_doubtful"),
        (st == "Questionable").cast(pl.Float64).alias("inj_questionable"),
        pr.str.contains("Did Not").cast(pl.Float64).alias("prac_dnp"),
        pr.str.contains("Limited").cast(pl.Float64).alias("prac_limited"),
        pr.str.contains("Full").cast(pl.Float64).alias("prac_full"),
    ).unique(["season", "week", "team", "player_id"], keep="last")


def depth(seasons: list[int], sched: pl.DataFrame) -> pl.DataFrame:
    """His best rank at an offensive skill slot on the depth chart in force before kickoff."""
    out = []
    old = [s for s in seasons if s <= 2024]
    if old:
        d = storage.load("depth_charts", seasons=old).filter(
            (pl.col("formation") == "Offense") & pl.col("position").is_in(["QB", "RB", "FB", "WR", "TE"]))
        out.append(d.select(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32), pl.col("club_code").alias("team"), pl.col("gsis_id").alias("player_id"),
                            pl.col("depth_team").cast(pl.Float64, strict=False).alias("depth"))
                   .group_by("season", "week", "team", "player_id").agg(pl.col("depth").min()))
    new = [s for s in seasons if s >= 2025]
    for s in new:
        try:
            d = storage.load("depth_charts", seasons=[s])
        except FileNotFoundError:
            continue
        d = d.filter(pl.col("pos_abb").is_in(list(OFFENSE_SLOTS))).select(
            pl.col("dt").str.to_datetime(time_zone="UTC", strict=False).alias("dt"), "team",
            pl.col("gsis_id").alias("player_id"), pl.col("pos_rank").cast(pl.Float64).alias("depth"))
        d = d.group_by("dt", "team", "player_id").agg(pl.col("depth").min()).sort("dt")
        games = sched.filter(pl.col("season") == s).select(
            "season", "week", "team", pl.col("kickoff_et").dt.convert_time_zone("UTC").alias("ko")
            if sched.schema["kickoff_et"] == pl.Datetime(time_zone="America/New_York")
            else pl.col("kickoff_et").dt.replace_time_zone("America/New_York", ambiguous="earliest").dt.convert_time_zone("UTC").alias("ko")).sort("ko")
        snaps = d.select("team", "dt").unique().sort("dt")
        pick = games.join_asof(snaps, left_on="ko", right_on="dt", by="team", strategy="backward")
        out.append(pick.join(d, on=["team", "dt"], how="inner").select(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32), "team", "player_id", "depth"))
    return pl.concat(out) if out else pl.DataFrame(schema={"season": pl.Int32, "week": pl.Int32, "team": pl.Utf8,
                                                           "player_id": pl.Utf8, "depth": pl.Float64})


# ---------------------------------------------------------------- the frame

def frame(seasons: list[int]) -> pl.DataFrame:
    """Candidates with pregame features and (for played games) labels."""
    hist = list(range(min(seasons) - 1, max(seasons) + 1))
    hist = [s for s in hist if s >= C.FIRST_SEASON]
    cand = candidates(hist)
    out = outcomes(hist)
    f = cand.join(out.select("game_id", "team", "player_id", "played", "started", "snap_share", "db_share"),
                  on=["game_id", "team", "player_id"], how="left")
    f = f.with_columns(
        pl.when(pl.col("is_played")).then(pl.col("played").fill_null(False)).alias("played"),
        pl.when(pl.col("is_played")).then(pl.col("started").fill_null(False)).alias("started"),
        pl.col("snap_share").fill_null(0.0), pl.col("db_share").fill_null(0.0),
    ).sort("player_id", "kickoff_et")
    # history: a player's own sequence of rostered team-games (a game he missed counts as zero)
    p = pl.col("played").cast(pl.Float64)
    f = f.with_columns(
        p.shift(1).over("player_id").alias("prev_played"),
        pl.col("snap_share").shift(1).over("player_id").alias("prev_snap"),
        pl.col("snap_share").ewm_mean(alpha=0.4, adjust=False).shift(1).over("player_id").alias("ew_snap"),
        p.rolling_mean(4, min_samples=1).shift(1).over("player_id").alias("played_rate_4"),
        pl.int_range(pl.len()).over("player_id").alias("n_prior"),
        (pl.col("team") != pl.col("team").shift(1).over("player_id")).cast(pl.Float64).alias("team_change"),
        pl.col("started").cast(pl.Float64).shift(1).over("player_id").alias("qb_prev_start"),
        pl.col("db_share").ewm_mean(alpha=0.5, adjust=False).shift(1).over("player_id").alias("qb_ew_share"),
    )
    f = f.with_columns(
        (p.cum_sum().over("player_id", "season") - p.fill_null(0)).alias("_gp_before"),
        (pl.int_range(pl.len()).over("player_id", "season")).alias("_games_before"),
    ).with_columns(
        pl.when(pl.col("_games_before") > 0).then(pl.col("_gp_before") / pl.col("_games_before")).alias("season_share"),
    )
    # games since he last played (capped)
    last = pl.when(pl.col("played")).then(pl.col("n_prior"))
    f = f.with_columns(last.forward_fill().shift(1).over("player_id").alias("_last_played_idx")).with_columns(
        (pl.col("n_prior") - pl.col("_last_played_idx")).clip(0, 20).fill_null(20).cast(pl.Float64).alias("games_since"))
    sched = schedule(hist)
    f = f.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))
    f = f.join(depth(hist, sched), on=["season", "week", "team", "player_id"], how="left").with_columns(
        pl.col("depth").is_null().cast(pl.Float64).alias("depth_missing"))
    inj = injuries(hist)
    f = f.join(inj, on=["season", "week", "team", "player_id"], how="left")
    reported = inj.select("season", "week", "team").unique().with_columns(pl.lit(True).alias("has_report"))
    f = f.join(reported, on=["season", "week", "team"], how="left").with_columns(pl.col("has_report").fill_null(False))
    f = f.with_columns(*[pl.col(c).fill_null(0.0) for c in ("inj_out", "inj_doubtful", "inj_questionable", "prac_dnp",
                                                           "prac_limited", "prac_full")],
                       *[(pl.col("pos") == q).cast(pl.Float64).alias(f"pos_{q}") for q in C.POSITIONS],
                       pl.col("years_exp").cast(pl.Float64), pl.col("week").cast(pl.Float64).alias("week_f"))
    f = f.with_columns(pl.col("week_f").alias("week")).drop("week_f", "_gp_before", "_games_before", "_last_played_idx")
    f = f.sort("player_id", "kickoff_et").with_columns(
        (pl.col("status") == "INA").cast(pl.Float64).shift(1).over("player_id").alias("prev_inactive"),
        *[pl.col(c).shift(1).over("player_id").alias(f"prev_{c}") for c in ("inj_out", "inj_doubtful", "inj_questionable")])
    return f.filter(pl.col("season").is_in(seasons))


def _X(f: pl.DataFrame, feats: list[str]) -> np.ndarray:
    return f.select([pl.col(c).cast(pl.Float64) for c in feats]).to_numpy()


def fit(f: pl.DataFrame) -> dict:
    tr = f.filter(pl.col("played").is_not_null())
    q = tr.filter(pl.col("pos") == "QB")
    out = {"features": {"early": EARLY, "report": REPORT}}
    for name, feats in (("early", EARLY), ("report", REPORT)):
        out[f"play_{name}"] = lgb.train(PARAMS, lgb.Dataset(_X(tr, feats), tr["played"].cast(pl.Int8).to_numpy()), ROUNDS)
        out[f"start_{name}"] = lgb.train(PARAMS, lgb.Dataset(_X(q, feats), q["started"].cast(pl.Int8).to_numpy()), ROUNDS)
    return out


def predict(models: dict, f: pl.DataFrame, force: str | None = None) -> pl.DataFrame:
    """p_play for everyone, p_start for QBs normalised to one starter per team-game. Each row uses the report
    model once its team's injury report for the week is out, else the early model (or `force`)."""
    use = np.full(f.height, force == "report") if force else f["has_report"].to_numpy()
    pe = models["play_early"].predict(_X(f, EARLY))
    pr = models["play_report"].predict(_X(f, REPORT))
    se = models["start_early"].predict(_X(f, EARLY))
    sr = models["start_report"].predict(_X(f, REPORT))
    out = f.with_columns(pl.Series("p_play", np.where(use, pr, pe)), pl.Series("model", np.where(use, "report", "early")))
    raw = np.where(use, sr, se)
    out = out.with_columns(pl.Series("_s", np.where(f["pos"].to_numpy() == "QB", raw, 0.0)))
    out = out.with_columns((pl.col("_s") / pl.col("_s").sum().over("game_id", "team")).fill_nan(0.0).alias("p_start"))
    return out.drop("_s")


def lineup(pred: pl.DataFrame, threshold: float = 0.5) -> pl.DataFrame:
    """Who the forward pass projects: skill players likely to play, and each team's likeliest starter."""
    skill = pred.filter((pl.col("pos") != "QB") & (pl.col("p_play") >= threshold))
    qb = pred.filter(pl.col("pos") == "QB").sort("p_start", descending=True).group_by("game_id", "team").first()
    return pl.concat([skill.select("game_id", "team", "player_id"), qb.select("game_id", "team", "player_id")])


# ---------------------------------------------------------------- evaluation

def _ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate(test_seasons=(2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)) -> pl.DataFrame:
    full = frame(list(range(C.FIRST_SEASON, C.CURRENT_SEASON + 1)))
    rows = []
    for s in test_seasons:
        m = fit(full.filter(pl.col("season") < s))
        held = full.filter((pl.col("season") == s) & pl.col("played").is_not_null())
        early = predict(m, held, force="early")
        te = predict(m, held, force="report")
        y = te["played"].cast(pl.Float64).to_numpy()
        base = te["prev_played"].fill_null(te["played"].cast(pl.Float64).mean()).to_numpy() * 0.9 + 0.05
        sel = lineup(te)
        act = te.filter(pl.col("played")).select("game_id", "team", "player_id")
        tp = sel.join(act, on=["game_id", "team", "player_id"], how="semi").height
        q = te.filter(pl.col("pos") == "QB")
        top = q.sort("p_start", descending=True).group_by("game_id", "team").first()
        rows.append({"season": s, "n": te.height, "played_rate": y.mean(), "ll_model": _ll(te["p_play"].to_numpy(), y),
                     "ll_early": _ll(early["p_play"].to_numpy(), y),
                     "qb_starter_acc_early": float(early.filter(pl.col("pos") == "QB").sort("p_start", descending=True)
                                                   .group_by("game_id", "team").first()["started"].mean()),
                     "ll_last_game": _ll(base, y), "lineup_precision": tp / max(sel.height, 1),
                     "lineup_recall": tp / max(act.height, 1),
                     "qb_starter_acc": float(top["started"].mean()),
                     "qb_starter_last_game": float(q.filter(pl.col("qb_prev_start") == 1).group_by("game_id", "team").first()["started"].mean())})
        print(rows[-1], flush=True)
    return pl.DataFrame(rows)


def main() -> None:
    R = evaluate()
    pl.Config.set_tbl_width_chars(200)
    print(R.with_columns(pl.col(pl.Float64).round(4)))
    full = frame(list(range(C.FIRST_SEASON, C.CURRENT_SEASON + 1)))
    models = fit(full.filter(pl.col("played").is_not_null()))
    C.MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as fh:
        pickle.dump(models, fh)
    (C.MODELS / "availability_eval.json").write_text(json.dumps(R.to_dicts(), indent=1, default=float))
    print(f"saved {MODEL_FILE}")


if __name__ == "__main__":
    main()
    sys.exit(0)
