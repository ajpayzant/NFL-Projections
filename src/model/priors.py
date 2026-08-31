"""What to believe about a player before he has played: group priors, shrinkage, rookie curves.

Three fitted artifacts, all measured out-of-sample on 2016-2025 and all inspectable:

1. **Group priors** -- the value of every share and rate by position and depth slot. This is what
   makes a full roster projectable. A team's third tight end has no meaningful history; what he has
   is a job, and the average of that job over ten seasons is a defensible starting point.
2. **Shrinkage constants** -- one `k` per metric, the number of opportunities at which a player's own
   history and his group prior carry equal weight. Fitted by grid search against held-out seasons,
   never assumed. This is the single largest accuracy change over the workbook, which took a
   40-carry sample of yards per carry at face value.
3. **Rookie draft-slot curves** -- the empirical average of each metric against draft pick, so the
   first pick in a class starts somewhere other than the position median.

**The objective is opportunity error, not rate error.** Every candidate `k` is scored on
`(predicted_rate - actual_rate) x actual_denominator`, which lands in units a reader can hold: targets,
carries, yards. That choice does real work -- it weights a starter's target-share miss far above a
fifth receiver's without any hand-set weighting, and it is the quantity that composes into fantasy
points downstream. A player who never played contributes his full predicted volume as error, so
predicting targets for someone who got none is punished rather than ignored.

**No season sees itself.** History is filtered to seasons strictly before the target, priors are
rebuilt from that same window, and depth slots come from the chart published before week 1. The
leakage guard lives in `blend.season_weights` and in the `preseason` snapshot rule, not in comments.

    python -m src.model.priors --fit          # fit everything, write data/fitted/, print the report
    python -m src.model.priors --report       # re-print the last report without refitting
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache

import polars as pl

from src.config import (
    FITTED,
    HISTORY_FROM,
    LAST_COMPLETE_SEASON,
    PARTICIPATION_FROM,
    PROJ_SEASON,
    Settings,
    ensure_dirs,
)
from src.data import depth, history, lake
from src.model.blend import blend_counts, ratio, shrink

# --------------------------------------------------------------------------- #
# what gets fitted
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Metric:
    """A projectable quantity, defined as a ratio of two countable things."""

    name: str
    num: str
    den: str
    table: str          # "skill" | "qb"
    kind: str           # "share" (of a team pool) | "rate" (per own opportunity)
    since: int = HISTORY_FROM
    monotone_in_pick: bool = True   # does more draft capital mean more of this, for a rookie?


SKILL_METRICS = (
    Metric("target_share", "targets", "team_targets", "skill", "share"),
    Metric("carry_share", "carries", "team_carries", "skill", "share"),
    Metric("air_yards_share", "receiving_air_yards", "team_air_yards", "skill", "share"),
    Metric("rz_target_share", "rz_targets", "team_rz_targets", "skill", "share"),
    Metric("rz_carry_share", "rz_carries", "team_rz_carries", "skill", "share"),
    Metric("inside_5_carry_share", "inside_5_carries", "team_inside_5_carries", "skill", "share"),
    Metric("short_yardage_carry_share", "short_yardage_carries", "team_short_yardage_carries", "skill", "share"),
    Metric("late_down_target_share", "late_down_targets", "team_late_down_targets", "skill", "share"),
    Metric("rec_td_share", "receiving_tds", "team_receiving_tds", "skill", "share"),
    Metric("rush_td_share", "rushing_tds", "team_rushing_tds", "skill", "share"),
    Metric("snap_share", "offense_snaps", "team_offense_snaps", "skill", "share", since=PARTICIPATION_FROM),
    Metric("route_participation", "routes", "team_dropbacks", "skill", "share", since=PARTICIPATION_FROM),
    Metric("rush_participation", "rush_plays", "team_designed_rushes", "skill", "share", since=PARTICIPATION_FROM),
    Metric("catch_rate", "receptions", "targets", "skill", "rate", monotone_in_pick=False),
    Metric("yards_per_target", "receiving_yards", "targets", "skill", "rate", monotone_in_pick=False),
    Metric("yards_per_carry", "rushing_yards", "carries", "skill", "rate", monotone_in_pick=False),
    Metric("adot", "receiving_air_yards", "targets", "skill", "rate", monotone_in_pick=False),
    Metric("tprr", "targets", "routes", "skill", "rate", since=PARTICIPATION_FROM),
    Metric("rush_success_rate", "rush_successes", "carries", "skill", "rate", monotone_in_pick=False),
    # -2 a time and almost pure noise at the player level, which is the argument for fitting it
    # rather than typing it: the fit will shrink nearly all of it to the position mean, and that is
    # the right answer instead of crediting a 500-touch career's two fumbles as a trait.
    Metric("fumble_rate", "fumbles_lost", "touches", "skill", "rate", monotone_in_pick=False),
)

QB_METRICS = (
    Metric("dropback_share", "dropbacks", "team_dropbacks", "qb", "share"),
    Metric("attempt_rate", "attempts", "dropbacks", "qb", "rate", monotone_in_pick=False),
    Metric("completion_pct", "completions", "attempts", "qb", "rate", monotone_in_pick=False),
    Metric("yards_per_attempt", "passing_yards", "attempts", "qb", "rate", monotone_in_pick=False),
    Metric("pass_td_rate", "passing_tds", "attempts", "qb", "rate", monotone_in_pick=False),
    Metric("int_rate", "interceptions", "attempts", "qb", "rate", monotone_in_pick=False),
    Metric("sack_rate", "sacks_suffered", "dropbacks", "qb", "rate", monotone_in_pick=False),
    Metric("scramble_rate", "scrambles", "dropbacks", "qb", "rate", monotone_in_pick=False),
    Metric("air_yards_per_attempt", "passing_air_yards", "attempts", "qb", "rate", monotone_in_pick=False),
    Metric("designed_rush_share", "designed_qb_rushes", "team_designed_rushes", "qb", "share"),
    Metric("clean_rush_share", "rush_attempts_clean", "team_carries", "qb", "share"),
    Metric("yards_per_clean_rush", "rush_yards_clean", "rush_attempts_clean", "qb", "rate", monotone_in_pick=False),
    Metric("pass_td_share", "passing_tds", "team_pass_tds", "qb", "share"),
    # A scrambler and a designed-run quarterback are not the same player and one "rush attempt share"
    # cannot express both, which is what the workbook tried. Yards per carry differs by ~2 between
    # the two, so they are projected apart and added back together.
    Metric("scramble_ypc", "scramble_yards", "scrambles", "qb", "rate", monotone_in_pick=False),
    Metric("designed_rush_ypc", "designed_qb_rush_yards", "designed_qb_rushes", "qb", "rate", monotone_in_pick=False),
    # the same pool the skill players' `rush_td_share` divides, so the two normalize together
    Metric("qb_rush_td_share", "rushing_tds", "team_rushing_tds", "qb", "share"),
    Metric("qb_fumble_rate", "fumbles_lost", "dropbacks", "qb", "rate", monotone_in_pick=False),
)

METRICS = SKILL_METRICS + QB_METRICS
BY_NAME = {m.name: m for m in METRICS}

# Depth slots deeper than this behave alike, and splitting them further only thins the sample.
SLOT_CAP = {"QB": 3, "RB": 4, "WR": 5, "TE": 3}

# A (position, slot) prior is itself shrunk toward the position prior. The group must hold this
# fraction of the position's total opportunity before it is taken at face value.
GROUP_SHRINK_FRACTION = 0.05

# Log-spaced, spanning "trust the player entirely" to "trust the group entirely". Units are the
# metric's own denominator on a one-season scale, so the range has to be wide.
K_GRID = (0.0, 2.0, 5.0, 10.0, 20.0, 35.0, 60.0, 100.0, 160.0, 250.0, 400.0, 650.0,
          1000.0, 1600.0, 2500.0, 4000.0, 1e12)

PICK_BINS = ((1, 10), (11, 20), (21, 32), (33, 50), (51, 75), (76, 100),
             (101, 150), (151, 200), (201, 400), (401, 9999))  # 401+ = undrafted

FIT_FIRST_TARGET = 2019   # leaves three seasons of history behind the earliest fitted season

# How far a rookie's prior moves from the job he is listed for toward what his draft position says.
# `w = 0` is the depth-slot prior alone under either form in `rookie_prior`, so the grid nests all
# three estimators and the two endpoints stay directly comparable.
ROOKIE_W_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
MIN_NORM_ROOKIES = 5        # below this a slot's typical draft capital is one player, not a norm
MIN_ROOKIES_PER_SEASON = 20  # below this a season contributes noise to the grid, not signal
RATIO_CLIP = (0.4, 2.5)     # a thin pick bin must not be allowed to triple a depth-slot prior

ROOKIE_BLEND_PATH = FITTED / "rookie_blend.parquet"
SLOT_NORM_PATH = FITTED / "rookie_slot_norm.parquet"


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def _hist(table: str) -> pl.DataFrame:
    seasons = lake.history_seasons()
    return history.skill_seasons(seasons) if table == "skill" else history.qb_seasons(seasons)


def team_pools(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """Every team-pool denominator per team-season, including for players who never played.

    Taken from `player_usage`, where the `team_*` columns repeat on every player row: one row per
    game holds the team's totals, so de-duplicating on the game and summing gives the season pool.
    A share prediction for a player who never took the field still has to be scored against a real
    pool, which is why this cannot come from the players' own rows.
    """
    seasons = seasons or lake.history_seasons()
    us = lake.regular_season(lake.read("player_usage", seasons=seasons))
    team_cols = [c for c in us.columns if c.startswith("team_")]
    per_game = us.unique(subset=["game_id", "team"], keep="first").select(
        "season", "team", "game_id", *team_cols,
        pl.col("plays").alias("team_plays"),
        pl.col("designed_rushes").alias("team_designed_rushes"),
    )
    return (
        per_game.group_by(["season", "team"])
        .agg(pl.col("^team_.*$").sum(), pl.len().alias("team_games"))
        # every receiving touchdown is somebody's passing touchdown, so the QB pool is the same pool
        .with_columns(pl.col("team_receiving_tds").alias("team_pass_tds"))
    )


def slots(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Depth slot for every charted offensive player, in every season, capped into buckets.

    The projection season uses the newest chart; past seasons use the last chart published before
    their week 1, so a fitted number never rests on a slot that was only knowable in December.
    """
    frames = []
    for s in seasons:
        when = "latest" if s >= PROJ_SEASON else "preseason"
        try:
            frames.append(depth.depth_chart(s, when))
        except (FileNotFoundError, ValueError):
            continue
    ch = pl.concat(frames, how="diagonal_relaxed")
    cap = pl.col("position").replace_strict(SLOT_CAP, default=4, return_dtype=pl.Int32)
    return ch.with_columns(
        pl.min_horizontal(pl.col("depth_slot"), cap).alias("slot_bucket")
    ).select("season", "team", "player_id", "position", "depth_tier", "depth_slot", "slot_bucket",
             "draft_pick", "draft_season", "snapshot")


# --------------------------------------------------------------------------- #
# group priors
# --------------------------------------------------------------------------- #
def group_prior(metric: Metric, hist_slots: pl.DataFrame, before: int) -> pl.DataFrame:
    """(position, slot_bucket) -> prior value, from seasons strictly before `before`.

    Two levels: the slot's own pooled ratio, shrunk toward the position's pooled ratio by how much
    of the position's volume the slot actually holds. A thin bucket therefore reports something near
    its position mean instead of a number built on fifty carries.
    """
    h = hist_slots.filter((pl.col("season") < before) & (pl.col("season") >= metric.since))
    if h.is_empty():
        return pl.DataFrame(schema={"position": pl.String, "slot_bucket": pl.Int32, "prior": pl.Float64})
    grp = h.group_by(["position", "slot_bucket"]).agg(
        pl.col(metric.num).sum().alias("num"), pl.col(metric.den).sum().alias("den")
    )
    pos = h.group_by("position").agg(
        pl.col(metric.num).sum().alias("pnum"), pl.col(metric.den).sum().alias("pden")
    )
    return (
        grp.join(pos, on="position", how="left")
        .with_columns(
            ratio("num", "den", "grp_rate"),
            ratio("pnum", "pden", "pos_rate"),
        )
        .with_columns(
            (
                (pl.col("num") + GROUP_SHRINK_FRACTION * pl.col("pden") * pl.col("pos_rate"))
                / (pl.col("den") + GROUP_SHRINK_FRACTION * pl.col("pden"))
            ).alias("prior"),
        )
        .select("position", "slot_bucket", "prior", pl.col("pos_rate").alias("position_prior"),
                pl.col("den").alias("prior_den"))
    )


# --------------------------------------------------------------------------- #
# panels: one row per projectable player-season, with obs, prior and outcome
# --------------------------------------------------------------------------- #
def panel(
    metric: Metric,
    target: int,
    hist: pl.DataFrame,
    slot_frame: pl.DataFrame,
    pools: pl.DataFrame,
    settings: Settings,
) -> pl.DataFrame:
    """Everything needed to score a prediction of `metric` for `target`, without seeing `target`."""
    charted = slot_frame.filter(pl.col("season") == target)
    if metric.table == "qb":
        charted = charted.filter(pl.col("position") == "QB")
    else:
        charted = charted.filter(pl.col("position").is_in(["RB", "WR", "TE"]))
    if charted.is_empty():
        return pl.DataFrame()

    past = hist.filter(pl.col("season") >= metric.since)
    blended = blend_counts(
        past.filter(pl.col("season") < target), target, [metric.num, metric.den],
        by=("player_id",), weights=settings.recency,
    ).rename({metric.num: "_bnum", metric.den: "n"})

    hist_slots = past.join(
        slot_frame.select("season", "player_id", "position", "slot_bucket"),
        on=["season", "player_id"], how="inner", suffix="_chart",
    ).with_columns(pl.col("position_chart").alias("position")).drop("position_chart")
    priors = group_prior(metric, hist_slots, before=target)

    actual = hist.filter(pl.col("season") == target).select(
        "player_id", pl.col(metric.num).alias("_anum"), pl.col(metric.den).alias("_aden")
    )

    out = (
        charted.join(blended, on="player_id", how="left")
        .join(priors, on=["position", "slot_bucket"], how="left")
        .join(actual, on="player_id", how="left")
    )
    # The denominator the error is measured in: a team pool for a share (known even for a player who
    # never dressed), the player's own opportunities for a rate (zero means no evidence, not zero).
    if metric.kind == "share":
        pool = pools.filter(pl.col("season") == target)
        pool_col = metric.den if metric.den in pool.columns else None
        if pool_col is None:
            return pl.DataFrame()
        out = out.join(
            pool.select("season", "team", pl.col(pool_col).alias("den_t")),
            on=["season", "team"], how="left",
        )
    else:
        out = out.with_columns(pl.col("_aden").fill_null(0.0).alias("den_t"))

    return out.with_columns(
        ratio("_bnum", "n", "obs"),
        ratio("_anum", "_aden", "actual"),
        pl.lit(target).alias("target_season"),
    ).filter(pl.col("den_t") > 0)


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
def _score(p: pl.DataFrame, k: float) -> tuple[float, float]:
    """Mean absolute and root-mean-square error, in units of the metric's denominator."""
    err = (
        (shrink("obs", "prior", "n", k) - pl.col("actual").fill_null(0.0)) * pl.col("den_t")
    ).alias("err")
    e = p.select(err)["err"]
    return float(e.abs().mean()), float((e**2).mean() ** 0.5)


def fit_metric(metric: Metric, panels: pl.DataFrame) -> dict:
    """Grid-search `k` for one metric on the pooled held-out seasons.

    Reported twice on purpose. Over the whole projectable population the gain from shrinkage looks
    small, because a player with no history is predicted by his prior at every `k` and so contributes
    the same error to each candidate -- a large, constant term that dilutes the percentage.
    `gain_vs_own_pct_played` restricts to players who do have history, which is the population the
    constant actually governs, and is the number that says whether shrinkage earns its place.
    """
    usable = panels.filter(pl.col("prior").is_not_null())
    if usable.height < 200:
        return {"metric": metric.name, "k": None, "n_obs": usable.height, "note": "too few observations"}
    scored = [(k, *_score(usable, k)) for k in K_GRID]
    best_k, best_mae, best_rmse = min(scored, key=lambda t: t[1])
    own_mae = next(m for k, m, _ in scored if k == 0.0)
    prior_mae = next(m for k, m, _ in scored if k == 1e12)

    with_hist = usable.filter(pl.col("obs").is_not_null() & (pl.col("n") > 0))
    if with_hist.height >= 100:
        h_best = min((_score(with_hist, k)[0], k) for k in K_GRID)
        h_own = _score(with_hist, 0.0)[0]
        h_gain = 100.0 * (h_own - h_best[0]) / h_own if h_own else 0.0
        k_played = h_best[1]
    else:
        h_gain, k_played = 0.0, best_k

    denom = usable["den_t"].mean()
    return {
        "metric": metric.name,
        "kind": metric.kind,
        "table": metric.table,
        "k": best_k if best_k < 1e11 else float("inf"),
        "mae": best_mae,
        "rmse": best_rmse,
        "mae_own_history": own_mae,
        "mae_prior_only": prior_mae,
        "gain_vs_own_pct": 100.0 * (own_mae - best_mae) / own_mae if own_mae else 0.0,
        "gain_vs_prior_pct": 100.0 * (prior_mae - best_mae) / prior_mae if prior_mae else 0.0,
        "gain_vs_own_pct_played": h_gain,
        "k_played_only": k_played if k_played < 1e11 else float("inf"),
        "pct_no_history": 100.0 * (1.0 - with_hist.height / usable.height),
        "n_obs": usable.height,
        "mean_denominator": float(denom) if denom is not None else None,
        "units": metric.den,
    }


def build_panels(
    metric: Metric, slot_frame: pl.DataFrame, pools: pl.DataFrame, settings: Settings,
    targets: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    hist = _hist(metric.table)
    first = max(FIT_FIRST_TARGET, metric.since + 1)
    targets = targets or tuple(range(first, LAST_COMPLETE_SEASON + 1))
    frames = [
        p for t in targets
        if not (p := panel(metric, t, hist, slot_frame, pools, settings)).is_empty()
    ]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


# --------------------------------------------------------------------------- #
# rookies
# --------------------------------------------------------------------------- #
def _pick_bin() -> pl.Expr:
    expr = pl.when(pl.col("draft_pick").is_null()).then(pl.lit(len(PICK_BINS) - 1))
    for i, (lo, hi) in enumerate(PICK_BINS):
        expr = expr.when((pl.col("draft_pick") >= lo) & (pl.col("draft_pick") <= hi)).then(pl.lit(i))
    return expr.otherwise(pl.lit(len(PICK_BINS) - 1)).cast(pl.Int32).alias("pick_bin")


def _isotonic_decreasing(values: list[float | None], weights: list[float]) -> list[float | None]:
    """Pool adjacent violators, so a later pick never projects above an earlier one.

    Draft capital buys opportunity monotonically in aggregate; a bin ordering that says otherwise is
    sample noise, and pooling the offending neighbours is the least-assumption way to remove it.
    """
    keep = [i for i, v in enumerate(values) if v is not None and weights[i] > 0]
    # blocks of (weighted mean, total weight, member indices)
    blocks: list[tuple[float, float, list[int]]] = []
    for i in keep:
        blocks.append((float(values[i]), float(weights[i]), [i]))
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2, i2 = blocks.pop()
            v1, w1, i1 = blocks.pop()
            blocks.append(((v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, i1 + i2))
    out = list(values)
    for value, _w, members in blocks:
        for i in members:
            out[i] = value
    return out


def _rookie_seasons(table: str, slot_frame: pl.DataFrame) -> pl.DataFrame:
    """Every player's rookie season -- his draft season -- with draft pick and charted position."""
    draft = (
        lake.read("draft_picks", layer="raw")
        .filter(pl.col("gsis_id").is_not_null())
        .select(
            pl.col("gsis_id").alias("player_id"),
            pl.col("season").alias("drafted_in"),
            pl.col("pick").alias("draft_pick"),
        )
    )
    return (
        _hist(table)
        .join(draft, on="player_id", how="inner")
        .filter(pl.col("season") == pl.col("drafted_in"))
        .join(
            slot_frame.select("season", "player_id", pl.col("position").alias("chart_position")),
            on=["season", "player_id"], how="left",
        )
        .with_columns(pl.coalesce("chart_position", "position").alias("position"))
        .with_columns(_pick_bin())
    )


def _curve(metric: Metric, rookies: pl.DataFrame) -> pl.DataFrame:
    """One metric's opportunity-weighted rookie average per (position, pick bin), smoothed."""
    if metric.num not in rookies.columns or metric.den not in rookies.columns:
        return pl.DataFrame()
    agg = (
        rookies.filter(pl.col("season") >= metric.since)
        .group_by(["position", "pick_bin"])
        .agg(
            pl.col(metric.num).sum().alias("num"),
            pl.col(metric.den).sum().alias("den"),
            pl.len().alias("rookies"),
        )
        .with_columns(ratio("num", "den", "value_raw"))
        .sort(["position", "pick_bin"])
    )
    rows = []
    for pos in agg["position"].unique().sort():
        sub = agg.filter(pl.col("position") == pos)
        vals = sub["value_raw"].to_list()
        wts = [float(w or 0.0) for w in sub["den"].to_list()]
        smooth = _isotonic_decreasing(vals, wts) if metric.monotone_in_pick else vals
        for b, raw, sm, n, den in zip(
            sub["pick_bin"].to_list(), vals, smooth, sub["rookies"].to_list(), wts, strict=True
        ):
            rows.append({
                "metric": metric.name, "position": pos, "pick_bin": int(b),
                "pick_lo": PICK_BINS[int(b)][0], "pick_hi": PICK_BINS[int(b)][1],
                "value_raw": raw, "value": sm, "rookies": int(n), "denominator": den,
            })
    return pl.DataFrame(rows)


def rookie_curves(slot_frame: pl.DataFrame, before: int | None = None) -> pl.DataFrame:
    """Every metric's rookie curve. `before` restricts the fit to earlier seasons, for held-out use."""
    frames = []
    for table in ("skill", "qb"):
        rookies = _rookie_seasons(table, slot_frame)
        if before is not None:
            rookies = rookies.filter(pl.col("season") < before)
        for metric in (m for m in METRICS if m.table == table):
            c = _curve(metric, rookies)
            if not c.is_empty():
                frames.append(c)
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def rookie_pool(slot_frame: pl.DataFrame) -> pl.DataFrame:
    """Every drafted rookie by the depth slot he was actually charted at, one row per player-season.

    The draft table is already joined into the chart as a tie-breaker, so a rookie is simply somebody
    whose draft season is the season in front of us.
    """
    return (
        slot_frame.filter(pl.col("draft_season") == pl.col("season"))
        .select("season", "position", "slot_bucket", "draft_pick")
        .with_columns(_pick_bin())
    )


def slot_capital_norm(
    pool: pl.DataFrame, curve: pl.DataFrame, before: int, min_n: int = MIN_NORM_ROOKIES
) -> pl.DataFrame:
    """What the draft curve says about the *typical* rookie charted at each depth slot.

    The reason a third rookie estimator exists. The depth-slot prior and the draft curve are two
    marginals of one thing: the slot averages over draft capital, the curve averages over slots.
    Interpolating between them gives a second-round rookie listed third the same prior as a
    second-round rookie listed first, which is how a rookie ends up projected over the veteran in
    front of him. Dividing his curve value by this norm turns draft capital into a *relative*
    statement -- more was invested in him than in the usual holder of this job -- which can then scale
    the slot's prior instead of replacing it.

    Bins with fewer than `min_n` historical rookies are dropped rather than trusted; a missing norm
    leaves the ratio at 1.0, which is the depth-slot prior untouched.
    """
    h = pool.filter(pl.col("season") < before)
    if h.is_empty() or curve.is_empty():
        return pl.DataFrame(schema={"position": pl.String, "slot_bucket": pl.Int32,
                                    "norm": pl.Float64})
    return (
        h.join(curve, on=["position", "pick_bin"], how="inner")
        .group_by(["position", "slot_bucket"])
        .agg(pl.col("curve").mean().alias("norm"), pl.len().alias("n_rookies"))
        .filter((pl.col("n_rookies") >= min_n) & (pl.col("norm") > 0))
        .select("position", "slot_bucket", "norm")
    )


def rookie_prior(slot: pl.Expr, form: str, w: float, bounded: bool = True) -> pl.Expr:
    """The three rookie estimators, in one place so a fit and a projection cannot diverge.

    - `slot` (`w = 0`, either form): his job, draft position ignored.
    - `additive`: `w` of the way from his job toward what his draft slot bought outright.
    - `capital_ratio`: his job, scaled by how his draft capital compares with the typical rookie
      holding that job.

    `bounded` clips to [0, 1] for shares, which every one of them is: scaling a first-round tight
    end's TE1 prior by his draft capital otherwise puts him on 101% of his team's routes.
    """
    if form == "additive":
        out = w * pl.coalesce("curve", slot) + (1.0 - w) * slot
    else:
        ratio_ = (pl.col("curve") / pl.col("norm")).clip(*RATIO_CLIP)
        out = slot * pl.coalesce(ratio_, pl.lit(1.0)) ** w
    return out.clip(0.0, 1.0) if bounded else out.clip(lower_bound=0.0)


def fit_rookie_blend(
    metric: Metric, panels: pl.DataFrame, curves_by_target: dict[int, pl.DataFrame],
    pool: pl.DataFrame,
) -> dict | None:
    """Grid-search how a rookie's prior combines his depth slot with his draft position.

    Scored on the panels already built for `metric`, held out by season -- the curve and the slot norm
    are both refitted on seasons strictly before each target -- and in the same opportunity units as
    every other fit here. The two endpoints are what the literature and the workbook do instead, so
    the report shows what is won over each rather than only over the worse one.
    """
    bounded = metric.kind == "share"
    frames = []
    for (target,), sub in panels.group_by("target_season"):
        rook = sub.filter(pl.col("draft_season") == target)
        if rook.height < MIN_ROOKIES_PER_SEASON or target not in curves_by_target:
            continue
        cv = curves_by_target[target].filter(pl.col("metric") == metric.name).select(
            "position", "pick_bin", pl.col("value").alias("curve")
        )
        if cv.is_empty():
            continue
        frames.append(
            rook.with_columns(_pick_bin())
            .join(cv, on=["position", "pick_bin"], how="left")
            .join(slot_capital_norm(pool, cv, before=target),
                  on=["position", "slot_bucket"], how="left")
            .select("prior", "curve", "norm", "actual", "den_t")
        )
    if not frames:
        return None
    r = pl.concat(frames, how="diagonal_relaxed").filter(pl.col("prior").is_not_null())
    if r.height < MIN_ROOKIES_PER_SEASON:
        return None

    def mae(form: str, w: float) -> float:
        e = r.select(
            ((rookie_prior(pl.col("prior"), form, w, bounded) - pl.col("actual").fill_null(0.0))
             * pl.col("den_t")).abs().alias("e")
        )["e"]
        return float(e.mean())

    grid = [(f, w, mae(f, w)) for f in ("additive", "capital_ratio") for w in ROOKIE_W_GRID]
    best_form, best_w, best = min(grid, key=lambda t: t[2])
    slot_only, curve_only = mae("additive", 0.0), mae("additive", 1.0)
    return {
        "metric": metric.name, "kind": metric.kind, "rookies": int(r.height),
        "form": best_form, "w": best_w, "mae": best,
        "mae_slot_prior": slot_only, "mae_draft_curve": curve_only,
        "gain_vs_slot_pct": 100.0 * (slot_only - best) / slot_only if slot_only else 0.0,
        "gain_vs_curve_pct": 100.0 * (curve_only - best) / curve_only if curve_only else 0.0,
        "units": metric.den,
    }


# --------------------------------------------------------------------------- #
# fit everything, save, report
# --------------------------------------------------------------------------- #
def fit_all(settings: Settings | None = None) -> dict[str, pl.DataFrame]:
    settings = settings or Settings()
    ensure_dirs()
    seasons = tuple(range(HISTORY_FROM, PROJ_SEASON + 1))
    slot_frame = slots(seasons)
    pools = team_pools()

    # One curve set per target season, shared by every metric's rookie fit: `rookie_curves` returns
    # all metrics at once, so memoising here turns 30 x 7 refits into 7.
    targets = tuple(range(FIT_FIRST_TARGET, LAST_COMPLETE_SEASON + 1))
    curves_by_target = {t: rookie_curves(slot_frame, before=t) for t in targets}
    pool = rookie_pool(slot_frame)

    fits, prior_rows, blend_rows = [], [], []
    for metric in METRICS:
        panels = build_panels(metric, slot_frame, pools, settings)
        if panels.is_empty():
            fits.append({"metric": metric.name, "k": None, "note": "no panel"})
            continue
        fits.append(fit_metric(metric, panels))
        if (rb := fit_rookie_blend(metric, panels, curves_by_target, pool)) is not None:
            blend_rows.append(rb)
        # the prior a projection will actually use: everything through the last complete season
        hist_slots = _hist(metric.table).join(
            slot_frame.select("season", "player_id", "position", "slot_bucket"),
            on=["season", "player_id"], how="inner", suffix="_chart",
        ).with_columns(pl.col("position_chart").alias("position")).drop("position_chart")
        gp = group_prior(metric, hist_slots, before=PROJ_SEASON)
        prior_rows.append(gp.with_columns(pl.lit(metric.name).alias("metric")))

    fit_df = pl.DataFrame(fits, infer_schema_length=None)
    priors_df = pl.concat(prior_rows, how="diagonal_relaxed") if prior_rows else pl.DataFrame()
    curves = rookie_curves(slot_frame)
    blend = pl.DataFrame(blend_rows, infer_schema_length=None)

    # The norms a projection will use: fitted on everything through the last complete season.
    norms = [
        slot_capital_norm(
            pool,
            curves.filter(pl.col("metric") == m.name).select(
                "position", "pick_bin", pl.col("value").alias("curve")
            ),
            before=PROJ_SEASON,
        ).with_columns(pl.lit(m.name).alias("metric"))
        for m in METRICS
    ]
    norms_df = pl.concat([n for n in norms if not n.is_empty()], how="diagonal_relaxed") \
        if any(not n.is_empty() for n in norms) else pl.DataFrame()

    fit_df.write_parquet(FITTED / "shrinkage.parquet")
    priors_df.write_parquet(FITTED / "group_priors.parquet")
    curves.write_parquet(FITTED / "rookie_curves.parquet")
    if not blend.is_empty():
        blend.write_parquet(ROOKIE_BLEND_PATH)
    if not norms_df.is_empty():
        norms_df.write_parquet(SLOT_NORM_PATH)
    (FITTED / "shrinkage.json").write_text(
        json.dumps(
            {
                "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "history": [HISTORY_FROM, LAST_COMPLETE_SEASON],
                "targets_from": FIT_FIRST_TARGET,
                "recency": list(settings.recency),
                "k": {r["metric"]: r.get("k") for r in fits},
            },
            indent=2,
        )
    )
    load_rookie_blend.cache_clear()
    load_slot_norm.cache_clear()
    return {"fit": fit_df, "priors": priors_df, "curves": curves,
            "rookie_blend": blend, "slot_norm": norms_df}


@dataclass(frozen=True, eq=False)
class Fitted:
    """Every fitted artifact the estimator reads, as one value that can be passed around.

    It exists for the backtest. The projection reads these off disk, where they were fitted on
    everything through the last complete season -- correct for projecting 2026, and leakage for
    projecting 2022, because a 2022 receiver's depth-slot prior would then be built partly from
    what happened in 2023. `fitted_as_of` rebuilds the frames from seasons strictly before a target
    so a held-out season can be projected the way it would have been projected at the time.

    `slot_games`, `games_k` and `games_tau` belong to `roster.py` and are filled by it; they travel
    here so that one argument carries the whole fitted state rather than five.

    The four scalars -- `k`, `blends`, `games_k` and the team-level fits in `team.py` -- are *not*
    refitted per target season. They are two parameters per metric grid-searched over seven target
    seasons, and refitting them per fold would multiply the cost of a backtest by the cost of the
    whole fit for a change smaller than the noise in the fold. `backtest.py` says so in its report
    rather than leaving the reader to assume otherwise, and `priors.fit_metric`'s own held-out
    numbers are the leak-free answer for those scalars specifically.
    """

    k: dict[str, float]
    priors: pl.DataFrame
    curves: pl.DataFrame
    norms: pl.DataFrame
    blends: dict[str, tuple[str, float]]
    slot_games: pl.DataFrame = field(default_factory=pl.DataFrame)
    games_k: float | dict[str, float] | None = None
    games_tau: float | dict[str, float] | None = None
    as_of: int | None = None          # None means "the saved fit", i.e. everything on disk

    @property
    def label(self) -> str:
        return "saved" if self.as_of is None else f"before {self.as_of}"


def fitted_saved() -> Fitted:
    """The artifacts on disk -- what the projection uses."""
    return Fitted(
        k=load_k(), priors=load_priors(), curves=load_curves(), norms=load_slot_norm(),
        blends=load_rookie_blend(),
    )


def fitted_as_of(before: int, settings: Settings | None = None) -> Fitted:
    """Group priors, rookie curves and slot norms rebuilt from seasons strictly before `before`.

    The depth slots behind them come from the chart published before each season's week 1, so a
    prior never rests on a job a player only held in December. Scalars are carried over from the
    saved fit -- see `Fitted` for why.
    """
    settings = settings or Settings()
    slot_frame = slots(tuple(range(HISTORY_FROM, before)))
    keyed = slot_frame.select("season", "player_id", "position", "slot_bucket")

    hist_slots = {
        table: _hist(table)
        .join(keyed, on=["season", "player_id"], how="inner", suffix="_chart")
        .with_columns(pl.col("position_chart").alias("position"))
        .drop("position_chart")
        for table in ("skill", "qb")
    }
    prior_rows = [
        group_prior(m, hist_slots[m.table], before=before).with_columns(
            pl.lit(m.name).alias("metric")
        )
        for m in METRICS
    ]
    priors_df = pl.concat([p for p in prior_rows if not p.is_empty()], how="diagonal_relaxed")

    curves = rookie_curves(slot_frame, before=before)
    pool = rookie_pool(slot_frame)
    norm_rows = []
    for m in METRICS:
        if curves.is_empty():
            break
        cv = curves.filter(pl.col("metric") == m.name).select(
            "position", "pick_bin", pl.col("value").alias("curve")
        )
        n = slot_capital_norm(pool, cv, before=before)
        if not n.is_empty():
            norm_rows.append(n.with_columns(pl.lit(m.name).alias("metric")))
    norms = pl.concat(norm_rows, how="diagonal_relaxed") if norm_rows else pl.DataFrame()

    return Fitted(
        k=load_k(), priors=priors_df, curves=curves, norms=norms, blends=load_rookie_blend(),
        as_of=before,
    )


def load_k() -> dict[str, float]:
    """Fitted `k` per metric, falling back to the Settings defaults when nothing is fitted yet."""
    path = FITTED / "shrinkage.json"
    settings = Settings()
    out = {
        m.name: (settings.default_share_k if m.kind == "share" else settings.default_rate_k)
        for m in METRICS
    }
    if path.is_file():
        saved = json.loads(path.read_text()).get("k", {})
        for name, k in saved.items():
            if k is not None:
                out[name] = float(k)
    return out


def load_priors() -> pl.DataFrame:
    path = FITTED / "group_priors.parquet"
    return pl.read_parquet(path) if path.is_file() else pl.DataFrame()


def load_curves() -> pl.DataFrame:
    path = FITTED / "rookie_curves.parquet"
    return pl.read_parquet(path) if path.is_file() else pl.DataFrame()


def load_rookie_check() -> pl.DataFrame:
    """The whole rookie fit, including what each form scored -- the accuracy table, not just the pick."""
    return pl.read_parquet(ROOKIE_BLEND_PATH) if ROOKIE_BLEND_PATH.is_file() else pl.DataFrame()


@lru_cache(maxsize=1)
def load_rookie_blend() -> dict[str, tuple[str, float]]:
    """Fitted (form, weight) per metric. A metric absent here keeps the depth-slot prior."""
    b = load_rookie_check()
    if b.is_empty():
        return {}
    return {r["metric"]: (str(r["form"]), float(r["w"])) for r in b.iter_rows(named=True)}


@lru_cache(maxsize=1)
def load_slot_norm() -> pl.DataFrame:
    return pl.read_parquet(SLOT_NORM_PATH) if SLOT_NORM_PATH.is_file() else pl.DataFrame()


def _report(art: dict[str, pl.DataFrame]) -> None:
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_fmt_float("mixed")
    fit = art["fit"]
    print("\nSHRINKAGE  (MAE in denominator units; k = opportunities that outweigh the prior)")
    print(fit.select(
        "metric", "kind", "k",
        pl.col("mae").round(2),
        pl.col("mae_own_history").round(2).alias("mae_own"),
        pl.col("mae_prior_only").round(2).alias("mae_prior"),
        pl.col("gain_vs_own_pct_played").round(1).alias("gain_vs_own_%"),
        pl.col("gain_vs_prior_pct").round(1).alias("gain_vs_prior_%"),
        pl.col("pct_no_history").round(0).alias("no_hist_%"),
        "n_obs", "units",
    ).sort("gain_vs_own_%", descending=True))

    blend = art.get("rookie_blend", pl.DataFrame())
    if not blend.is_empty():
        print("\nROOKIE PRIORS  depth slot vs draft slot vs both, held out by season")
        print(blend.select(
            "metric", "kind", "rookies", "form", "w",
            pl.col("mae").round(2),
            pl.col("mae_slot_prior").round(2).alias("mae_slot"),
            pl.col("mae_draft_curve").round(2).alias("mae_curve"),
            pl.col("gain_vs_slot_pct").round(1).alias("gain_vs_slot_%"),
            pl.col("gain_vs_curve_pct").round(1).alias("gain_vs_curve_%"),
            "units",
        ).sort("gain_vs_slot_%", descending=True))

    curves = art["curves"]
    if not curves.is_empty():
        print("\nROOKIE TARGET SHARE BY DRAFT SLOT (WR)")
        print(curves.filter((pl.col("metric") == "target_share") & (pl.col("position") == "WR"))
              .select("pick_lo", "pick_hi", pl.col("value_raw").round(4), pl.col("value").round(4), "rookies")
              .sort("pick_lo"))

    priors = art["priors"]
    if not priors.is_empty():
        print("\nGROUP PRIORS, selected")
        print(priors.filter(pl.col("metric").is_in(["target_share", "carry_share", "route_participation", "yards_per_carry"]))
              .select("metric", "position", "slot_bucket", pl.col("prior").round(4), pl.col("position_prior").round(4))
              .sort(["metric", "position", "slot_bucket"]))
    print(f"\nwritten to {FITTED}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", action="store_true", help="refit and write artifacts")
    p.add_argument("--report", action="store_true", help="print the saved artifacts")
    args = p.parse_args(argv)
    if args.report and not args.fit:
        art = {
            "fit": pl.read_parquet(FITTED / "shrinkage.parquet"),
            "priors": load_priors(),
            "curves": load_curves(),
            "rookie_blend": load_rookie_check(),
        }
    else:
        art = fit_all()
    _report(art)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
