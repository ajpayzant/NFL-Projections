"""The projectable roster: who is on the team, where he sits, how often he plays, and how much of
the field he is on when he does.

The workbook this replaces only carried players who already had a season of history -- 376 of them.
This module carries every offensive player on a 2026 roster, because a projection that cannot name
a team's fourth receiver cannot answer what happens when its second one is hurt.

**Two sources, two jobs.** The roster says who is on the team; the depth chart says in what order.
Neither can do the other's work: the chart lists 84 players who are no longer rostered, and the
roster lists 61 who are not on the chart. So membership comes from the roster, order comes from the
chart, and a rostered player the chart omits is slotted *behind* everyone it lists, ranked among his
fellow omissions by prior opportunity per game and then draft pick -- the same two tie-breakers
`depth._resolve_slots` already uses inside a tier, and both knowable in August.

**Participation is a product, not a share.** Every rate in `history` is opportunity-weighted over the
games a player actually played, so a snap share means "when he was out there, this much of the
offence". What a season needs is that number times how many games he is out there for:

    weekly contribution = expected_games / 17  x  participation when active

Keeping the two apart is what makes the deep roster behave. A ninth receiver's charted history says
he ran routes on 62% of dropbacks -- true, and useless on its own, because the only ninth receivers
who ever recorded a route were the ones promoted after an injury. He played 0.3 games. The product,
0.011 of a week, is the honest answer, and neither factor alone gets close to it.

**Expected games is three questions, not one.** They are kept apart because only the first can be
fitted on a population that matches the one it is applied to:

- `games_if_available` -- his own durability record blended toward the average games played from his
  depth slot, the constant grid-searched on 2019-2025 exactly as the shrinkage constants are. This is
  conditional on turning up in the season at all, because the population it is fitted on is everyone
  charted in August or seen in a game: the historical `rosters` table carries weekly changes, about
  four players per team, and no August roster exists before 2026.
- `presence` -- how often a job that deep exists at all. A 2026 August roster is 90 men, 28.6 of them
  offensive skill players per team; the population that shows up in a season is 21.9. Somebody has to
  be the difference and depth slot is the only thing that says who, so this is measured directly:
  RB4 0.98, RB6 0.49, TE5 0.45, QB4 0.14. Without it the deep tail of a 90-man roster is projected as
  though every camp body survives the cut, which put team snap sums 13% over the identity.
- `status_factor` -- the one stated assumption in the module. The historical week-1 status column is
  missing entirely for 2017-2018 and inconsistent elsewhere, so there is nothing honest to fit against;
  the multipliers live in `Settings.status_availability` where a user can see and move them.

  This factor was near-inert in August and is not any more, which is worth knowing before reading a
  number that depends on it. A camp snapshot is 90-man and almost entirely ACT -- 20 of 915 offensive
  players were anything else. A settled September snapshot is the post-cuts population, and it carries
  everyone the team let go: 502 ACT, 209 CUT, 182 practice-squad DEV, 60 RES. That is not a worse file,
  it is the *same* population the availability fit was built on -- the 2023-2025 week-1 snapshots run
  441/172/171/82, 450/219/187/60 and 451/190/180/65 -- so the fit and the roster it is applied to agree
  now in a way they did not in August. What changed is how much of the answer these multipliers carry.

**Rookies get draft capital as a relative statement, not an absolute one.** A rookie has no history,
so his prior is the whole projection, and the two obvious answers are both marginals of the same
thing: the depth-slot prior averages over draft capital, the draft curve averages over depth slots.
Interpolating between them gives a second-round rookie listed third the same prior as one listed
first, which is how a rookie ends up over the veteran in front of him. So a third estimator is fitted
alongside them -- his slot's prior scaled by how his draft capital compares with the typical rookie
holding that job -- and it wins on all three metrics it can be fitted for, by 15-20% over the slot
prior and 4-12% over the curve. `dropback_share` has too few rookies a season to fit and keeps the
slot prior, the same auto-revert discipline the team chain uses.

**The diagnostic that matters.** Five skill players are on the field for every snap, so summing
`games/17 x snap share` over everyone who took one has to come back to five. Measured over the full
2021-2025 population it is 4.94, the shortfall being only that each player's share is measured
against his own games rather than the season. That is the number this module reports itself against --
not the 4.31 that the August-charted subset comes to, because the 2026 roster is the whole population
and not the subset. It currently lands within 1.2% on all four metrics, with nothing tuned to make it
do so: the sum is an output of the two fitted factors, which is the only reason it is worth reading.

    python -m src.model.roster --fit       # fit availability, write data/fitted/, print the report
    python -m src.model.roster --report    # re-print from the saved artifacts
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Container
from datetime import UTC, date, datetime
from functools import lru_cache

import polars as pl

from src.config import (
    FITTED,
    HISTORY_FROM,
    LAST_COMPLETE_SEASON,
    OFFENSE_POSITIONS,
    PROJ_SEASON,
    REG_WEEKS,
    TEAM_FIXUP,
    Settings,
    ensure_dirs,
)
from src.data import depth, history, lake
from src.model import estimate, priors
from src.model.blend import season_weights, shrink, shrink_weight

# The field-presence rates plus the quarterback's equivalent. `rush_participation` was here and is gone:
# the play-by-play credits a designed run to the man who carried it, so it was never field presence, and
# what it measured was `carry_share` spelled differently (see `opportunity.POOLS`).
PARTICIPATION_METRICS = ("snap_share", "route_participation", "dropback_share")

# Games played collapses with depth in a way participation does not: a fifth receiver plays 10.5
# games, an eighth half of one. Deep enough to reach the bottom of a real roster; the isotonic
# smoothing in `slot_games_prior` is what keeps the thin bins down there from misbehaving.
AVAIL_SLOT_CAP = 12

# Seasons of evidence, so the units are directly readable: k = 1.0 means one full season of a
# player's own attendance record weighs the same as his slot's average.
GAMES_K_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 1e9)

# Which quantile of a slot's attendance record the prior aims at. The mean is the wrong target here
# and it is worth being explicit about why: attendance is hard left-tailed -- a starting quarterback
# either plays every week or misses a block of them -- so the mean of a slot sits well below its
# median and prices a torn ACL into every healthy player. The metric this fit is scored on is MAE,
# which is minimised by the conditional *median*, so a median-quantile prior is not a preference for
# optimism, it is the internally consistent choice for the loss already in use. Fitted, not assumed.
GAMES_TAU_GRID = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
GAMES_TAU = 0.5

# Both constants are fitted twice, because first-string jobs and bench jobs are not the same
# population and one pair of constants has to compromise between them:
#
# - At slot 1-2 the job exists for certain (`presence` is 1.0), so attendance is *only* injury, and
#   that distribution is hard left-tailed -- 17, 17, 17, 6. Its mean sits far below its median, which
#   is why a mean-targeting prior projected a starting quarterback at thirteen games.
# - At slot 5+ attendance is mostly whether he holds a job at all. No such skew, and lifting the
#   prior there is not a harmless error: with proportional pool normalisation an over-projected
#   backup takes his share off the starter in front of him.
#
# Four parameters on 4,806 held-out rows, and MAE is additive over the segments, so fitting them
# apart *is* fitting them jointly -- it cannot score worse than one shared pair, and it measures
# 3.697 against 3.702.
AVAIL_FRONT_MAX = 2
AVAIL_SEGMENTS = ("front", "back")

AVAIL_PATH = FITTED / "availability.json"
SLOT_GAMES_PATH = FITTED / "availability_slots.parquet"


# --------------------------------------------------------------------------- #
# the roster
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _draft() -> pl.DataFrame:
    d = lake.read("draft_picks", layer="raw").filter(pl.col("gsis_id").is_not_null())
    return d.group_by(pl.col("gsis_id").alias("player_id")).agg(
        pl.col("pick").min().alias("draft_pick_dr"), pl.col("season").min().alias("draft_season_dr")
    )


@lru_cache(maxsize=1)
def _draft_numbers() -> pl.DataFrame:
    """player_id -> draft pick, from every roster snapshot in the lake that still carries the column.

    A player's draft position never changes, so any season's snapshot answers for him and the newest
    one is not privileged. Worth the extra read because `draft_picks` only reaches back to 2016, and the
    roster column is what prices the veterans drafted before that -- twenty of the 2026 offence, who are
    otherwise indistinguishable from undrafted the moment upstream stops shipping it, as it just did.

    Returns an empty frame rather than raising: a missing draft pick costs one prior, and by the time
    this is reached the roster itself has already been read successfully.
    """
    frames = []
    for dataset in ("rosters", "rosters_weekly"):
        try:
            df = lake.read(dataset, layer="raw")
        except FileNotFoundError:
            continue
        if {"gsis_id", "draft_number"} <= set(df.columns):
            frames.append(df.select("gsis_id", "draft_number"))
    if not frames:
        return pl.DataFrame(schema={"player_id": pl.String, "draft_number_ros": pl.Int32})
    return (
        pl.concat(frames)
        .filter(pl.col("gsis_id").is_not_null() & pl.col("draft_number").is_not_null())
        .group_by(pl.col("gsis_id").alias("player_id"))
        .agg(pl.col("draft_number").cast(pl.Int32, strict=False).min().alias("draft_number_ros"))
    )


@lru_cache(maxsize=8)
def _games_by_team(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Games a player actually played, per season *and team*.

    Per team rather than per season because a mid-season trade splits a player between two pools, and
    the pool arithmetic downstream is per team. `history.skill_seasons` assigns him his modal team,
    which is the wrong answer for exactly this purpose.
    """
    frames = []
    for weeks in (history.skill_weeks(seasons), history.qb_weeks(seasons)):
        frames.append(
            weeks.group_by(["season", "team", "player_id"]).agg(
                pl.col("game_id").n_unique().cast(pl.Float64).alias("games")
            )
        )
    return (
        pl.concat(frames)
        .group_by(["season", "team", "player_id"])
        .agg(pl.col("games").max())     # a QB who also caught a pass appears in both frames
    )


def _by_segment(value: float | dict | None, default: float) -> dict[str, float]:
    """One constant per availability segment, from a scalar, a mapping or nothing.

    A scalar applies to both -- which is what `backtest`'s `own_games` ablation passes when it sets the
    blend to zero, and what an artifact written before the split carries.
    """
    if isinstance(value, dict):
        return {s: float(value.get(s, default)) for s in AVAIL_SEGMENTS}
    v = default if value is None else float(value)
    return dict.fromkeys(AVAIL_SEGMENTS, v)


def _segment_expr(per_segment: dict[str, float]) -> pl.Expr:
    """The segment's constant, per row, keyed off `avail_slot`."""
    return (
        pl.when(pl.col("avail_slot") <= AVAIL_FRONT_MAX)
        .then(pl.lit(per_segment["front"], pl.Float64))
        .otherwise(pl.lit(per_segment["back"], pl.Float64))
    )


@lru_cache(maxsize=8)
def _team_games(seasons: tuple[int, ...]) -> pl.DataFrame:
    """How many regular-season games each team actually played, per season.

    Sixteen before 2021 and seventeen after, and that is not a detail: availability has to be a *rate*
    before it can be pooled across a window that straddles the change. A prior built from raw counts
    reads half of 2016-2020 as a player who missed a game, which biases every slot's prior low and
    every projection with it. Measured rather than assumed, because a cancelled game is real too.
    """
    return (
        history.team_weeks(seasons)
        .group_by(["season", "team"])
        .agg(pl.col("game_id").n_unique().cast(pl.Float64).alias("team_games"))
    )


def _age(season: int) -> pl.Expr:
    """Age on 1 September of the season, in years."""
    ref = date(season, 9, 1)
    return (
        (pl.lit(ref) - pl.col("birth_date").cast(pl.Date, strict=False)).dt.total_days() / 365.25
    ).alias("age")


# A real league-wide snapshot is 32 rosters of about 53. Well under that is a partial file rather
# than a small league, and taking it at face value would project a season off forty players.
MIN_SNAPSHOT_ROWS = 1000

# The four columns without which a roster row means nothing: no identifier, no position, no team or no
# name and there is no player to project. Everything else upstream ships is optional, and treated that
# way on purpose -- the 2026 snapshot silently stopped carrying `draft_number` mid-season, and a hard
# `pl.col` reference to it turned a missing column into a ColumnNotFoundError that took the whole app
# down on load. A column that disappears should cost the estimate it feeds, not the session.
REQUIRED_ROSTER_COLUMNS = ("gsis_id", "position", "team", "full_name")


def _opt(columns: Container[str], name: str, dtype: pl.DataType = pl.String) -> pl.Expr:
    """`name` if the snapshot carries it, an all-null column of `dtype` if it does not."""
    return pl.col(name) if name in columns else pl.lit(None, dtype).alias(name)


def _snapshot_rows(season: int, when: str = "latest") -> pl.DataFrame:
    """One full league-wide roster snapshot of `season` -- the newest one, or week 1 for a backtest.

    Two datasets carry rosters and only one of them is a snapshot in every season. `rosters_weekly` is
    a genuine week-by-week capture for 2016-2025 but stops at the last complete season, while the
    `rosters` table we refresh ourselves is a *live* snapshot of the projection season -- upstream
    rewrites the same file as transactions happen -- and, for earlier seasons in the shared lake, a
    sparse file of a few hundred rows that is not a roster at all.

    So both the table preference and the week depend on what is being asked for, and getting either
    backwards is the same silent failure: a roster frozen at the start of the season.

    - `when="preseason"`, which is what a backtest asks for: the weekly capture, at its lowest week,
      so the run sees the roster a user would have seen in August and nothing it learned later.
    - anything else, which is the projection season: our own refreshed snapshot first, and the highest
      week rather than the lowest. Preferring the weekly capture here would trade a file we refresh
      ourselves every morning for one that arrives on the other repo's schedule, and taking its lowest
      week would pin the population to week 1 for the rest of the season -- so a receiver traded in
      October would still be projected on the team that traded him, with a share of its targets, for
      every remaining week. That is the failure this ordering exists to prevent, and it costs nothing
      today only because 2026 has no weekly capture yet.

    The size check catches the case where neither table is whole, which is why the preference is a
    fallback chain rather than a single choice.
    """
    preseason = when == "preseason"
    tried = []
    for dataset in (("rosters_weekly", "rosters") if preseason else ("rosters", "rosters_weekly")):
        try:
            df = lake.read(dataset, layer="raw", seasons=(season,))
        except FileNotFoundError:
            continue
        if "game_type" in df.columns:
            df = df.filter(pl.col("game_type") == "REG")
        if "week" in df.columns:
            edge = pl.col("week").min() if preseason else pl.col("week").max()
            df = df.filter(pl.col("week") == edge)
        missing = [c for c in REQUIRED_ROSTER_COLUMNS if c not in df.columns]
        tried.append(f"{dataset}={df.height}" + (f" (no {', '.join(missing)})" if missing else ""))
        if df.height >= MIN_SNAPSHOT_ROWS and not missing:
            return df
    edge_name = "week-1" if preseason else "current"
    raise FileNotFoundError(
        f"no league-wide {edge_name} roster snapshot for {season} "
        f"(found {', '.join(tried) or 'nothing'})"
    )


@lru_cache(maxsize=8)
def roster(season: int = PROJ_SEASON, when: str = "latest") -> pl.DataFrame:
    """Every offensive player on a `season` roster, with his depth slot resolved.

    One row per player. `charted` says whether the depth chart listed him at this position, and
    `chart_team` says where -- a disagreement is resolved to the roster, because the roster is the
    fresher of the two and is the one that decides who is actually on the team.

    `when` picks the snapshot for both sources, so the chart and the roster are read as of the same
    moment: `"latest"` for the season being projected, `"preseason"` for a backtest that must not see
    anything published after week 1.
    """
    ros = _snapshot_rows(season, when)
    have = set(ros.columns)
    ros = (
        ros.filter(pl.col("gsis_id").is_not_null() & pl.col("position").is_in(OFFENSE_POSITIONS))
        .select(
            pl.lit(season, pl.Int32).alias("season"),
            pl.col("team").replace(TEAM_FIXUP).alias("team"),
            pl.col("gsis_id").alias("player_id"),
            pl.col("full_name").alias("player"),
            pl.col("position").replace({"FB": "RB"}).alias("position"),
            pl.col("position").alias("roster_position"),
            _opt(have, "status"),
            _opt(have, "years_exp").cast(pl.Int32, strict=False).alias("years_exp"),
            _opt(have, "rookie_year").cast(pl.Int32, strict=False).alias("rookie_year"),
            _opt(have, "draft_number").cast(pl.Int32, strict=False).alias("draft_number"),
            _opt(have, "jersey_number").cast(pl.Int32, strict=False).alias("jersey"),
            _age(season) if "birth_date" in have else pl.lit(None, pl.Float64).alias("age"),
            _opt(have, "height").cast(pl.Float64, strict=False).alias("height"),
            _opt(have, "weight").cast(pl.Float64, strict=False).alias("weight"),
        )
        .sort(["player_id", "team"])
        .unique(subset=["player_id"], keep="first")
    )
    # Neither `unique` nor `join` promises to preserve row order, so the final sort is what makes this
    # frame reproducible. It is not cosmetic: the opportunity stage divides pools in the order it is
    # handed the players, so an unstable order here moved league carries by ~0.2% between two runs of
    # the *same* scenario -- enough to hide a real edit inside run-to-run noise.
    ros = ros.join(_draft_numbers(), on="player_id", how="left").with_columns(
        pl.coalesce("draft_number", "draft_number_ros").alias("draft_number")
    ).drop("draft_number_ros").sort("player_id")

    chart = depth.depth_chart(season, when)
    listed = (
        chart.select(
            "player_id", "position",
            pl.col("team").alias("chart_team"),
            pl.col("depth_tier").alias("chart_tier"),
            "alignment", "snapshot",
        )
        .sort(["player_id", "chart_tier"])
        .unique(subset=["player_id", "position"], keep="first")
    )
    anywhere = set(chart["player_id"].to_list())

    ros = ros.join(listed, on=["player_id", "position"], how="left").with_columns(
        pl.col("chart_tier").is_not_null().alias("charted"),
        pl.col("player_id").is_in(anywhere).alias("charted_anywhere"),
    )

    # An omitted player sits one tier below the deepest the chart names at his position, so he ranks
    # behind every listed player and among the other omissions by the usual tie-breakers.
    behind = pl.col("chart_tier").max().over(["team", "position"]).fill_null(0) + 1
    ros = ros.with_columns(pl.coalesce("chart_tier", behind).cast(pl.Int32).alias("depth_tier"))

    cap = pl.col("position").replace_strict(priors.SLOT_CAP, default=4, return_dtype=pl.Int32)
    out = depth._resolve_slots(ros, season).with_columns(
        pl.min_horizontal("depth_slot", cap).alias("slot_bucket"),
        pl.min_horizontal("depth_slot", pl.lit(AVAIL_SLOT_CAP, pl.Int32)).alias("avail_slot"),
        pl.coalesce("draft_pick", "draft_number").alias("draft_pick"),
        (
            (pl.col("years_exp") == 0) | (pl.col("rookie_year") == season)
        ).fill_null(False).alias("is_rookie"),
        (pl.col("chart_team") != pl.col("team")).fill_null(False).alias("team_disagreement"),
    )
    return out.with_columns(priors._pick_bin()).sort(["team", "position", "depth_slot"])


# --------------------------------------------------------------------------- #
# availability: how many of the 17 does he play
# --------------------------------------------------------------------------- #
def _history_games(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Attendance per player-season as a rate of his team's games, summed across teams.

    Across teams because durability is a property of the player: a back who played nine games for one
    team and six for another was available for fifteen, and that is what predicts next season. The
    *target* the fit is scored against stays per team, because that is what a team's pool needs
    filling. Predicting availability and predicting where he does it are different questions.

    `rate` is the number that travels between seasons; `games` is kept beside it so the report and the
    app can print an attendance record in the units a reader thinks in.
    """
    return (
        _avail_panel(seasons)
        .group_by(["season", "player_id"])
        .agg(
            pl.col("team_games").max().alias("team_games"),
            pl.min_horizontal(pl.col("games").sum(), pl.col("team_games").max()).alias("games"),
        )
        .with_columns((pl.col("games") / pl.col("team_games")).clip(0.0, 1.0).alias("rate"))
    )


def attendance_history(seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """The attendance record itself, per player-season: games, his team's games, and the rate.

    The public form of what the fit reads, for the app to put beside the knob. `expected_games` is the
    most argued-with number in the projection and the argument is only honest with the record in front
    of it: four straight seventeens and two nines either side of a fifteen can shrink to the same
    estimate, and only one of them is a player anybody should be talked out of.
    """
    return _history_games(tuple(seasons or lake.history_seasons()))


@lru_cache(maxsize=4)
def _positions(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Position by player-season from what he actually did, for players no chart listed."""
    sk = history.skill_seasons(seasons).select("season", "player_id", "position")
    qb = history.qb_seasons(seasons).select("season", "player_id", pl.lit("QB").alias("position"))
    return pl.concat([sk, qb]).unique(subset=["season", "player_id"], keep="first")


@lru_cache(maxsize=4)
def _avail_panel(seasons: tuple[int, ...]) -> pl.DataFrame:
    """One row per player-season-team, over everyone charted in August *or* seen in a game.

    The population is the compromise the data forces. Charted-only would exclude the September
    signing who played twelve games, and understate how much of a roster's playing time goes to
    players nobody listed. Played-only would exclude the charted starter who tore an ACL in camp,
    and a durability record built without him is not a durability record. The union has both, and it
    is defined identically in every season, which is what a fitted constant needs.

    Games are counted *for that team*: a player traded in October stops filling this pool the day he
    leaves, and the pool is what the projection has to fill.
    """
    played = _games_by_team(seasons)
    pos = _positions(seasons)
    frames = []
    for season in seasons:
        try:
            ch = depth.depth_chart(season, "latest" if season >= PROJ_SEASON else "preseason")
        except (FileNotFoundError, ValueError):
            continue
        ch = ch.select("season", "team", "player_id", "position",
                       pl.col("depth_tier").alias("chart_tier"))
        extra = (
            played.filter(pl.col("season") == season)
            .join(ch.select("season", "team", "player_id"), on=["season", "team", "player_id"],
                  how="anti")
            .join(pos, on=["season", "player_id"], how="inner")
            .select("season", "team", "player_id", "position",
                    pl.lit(None, pl.Int32).alias("chart_tier"))
        )
        u = pl.concat([ch, extra], how="diagonal_relaxed")
        behind = pl.col("chart_tier").max().over(["team", "position"]).fill_null(0) + 1
        u = u.with_columns(pl.coalesce("chart_tier", behind).cast(pl.Int32).alias("depth_tier"))
        frames.append(depth._resolve_slots(u, season))
    panel = pl.concat(frames, how="diagonal_relaxed")
    return (
        panel.join(played, on=["season", "team", "player_id"], how="left")
        .join(_team_games(seasons), on=["season", "team"], how="left")
        .with_columns(
            pl.col("games").fill_null(0.0),
            pl.col("team_games").fill_null(float(REG_WEEKS - 1)),
            pl.min_horizontal("depth_slot", pl.lit(AVAIL_SLOT_CAP, pl.Int32)).alias("avail_slot"),
            pl.col("chart_tier").is_not_null().alias("charted"),
        )
        .with_columns(
            (pl.col("games") / pl.col("team_games")).clip(0.0, 1.0).alias("rate")
        )
        .select("season", "team", "player_id", "position", "depth_slot", "avail_slot", "charted",
                "games", "team_games", "rate")
    )


def slot_games_prior(
    panel: pl.DataFrame, before: int, tau: float | dict[str, float] | None = None
) -> pl.DataFrame:
    """The `tau` quantile of the attendance *rate* by (position, slot), before `before`.

    Two deliberate choices, both of which move a starter's projection materially:

    - **A rate, not a count.** The window straddles the 2021 move from sixteen games to seventeen, so
      a count pools two different denominators; see `_team_games`.
    - **A quantile, not a mean.** Attendance is not symmetric. A first-string quarterback plays every
      week or misses a block of them, and averaging the two produces a number no quarterback's season
      ever looks like -- 13.1 games, which is neither the 17 of the majority nor the 6 of the injured.
      Since the fit is scored on MAE, and MAE is minimised by the median, the median is the estimate
      the loss actually asks for. `tau` is grid-searched in `fit_availability` all the same.

    Smoothed to be non-increasing in slot. Deeper means less playing time in aggregate; a bin that
    says otherwise is a handful of promoted backups, and pooling the offending neighbours removes it
    without inventing a functional form. `prior_games` is `prior_rate` in seventeenths, carried for
    readability only -- nothing downstream computes with it.
    """
    taus = _by_segment(tau, GAMES_TAU)
    h = panel.filter(pl.col("season") < before)
    agg = (
        h.group_by(["position", "avail_slot"])
        .agg(
            *[pl.col("rate").quantile(t, interpolation="linear").alias(f"q_{s}")
              for s, t in taus.items()],
            pl.col("rate").mean().alias("mean_rate"),
            pl.len().alias("n"),
        )
        .with_columns(
            pl.when(pl.col("avail_slot") <= AVAIL_FRONT_MAX)
            .then(pl.col("q_front")).otherwise(pl.col("q_back")).alias("raw_rate"),
            pl.when(pl.col("avail_slot") <= AVAIL_FRONT_MAX)
            .then(pl.lit(taus["front"])).otherwise(pl.lit(taus["back"])).alias("tau"),
        )
        .drop([f"q_{s}" for s in taus])
        .sort(["position", "avail_slot"])
    )
    rows = []
    for pos in agg["position"].unique().sort():
        sub = agg.filter(pl.col("position") == pos)
        w = [float(n) for n in sub["n"].to_list()]
        smooth = priors._isotonic_decreasing(sub["raw_rate"].to_list(), w)
        # The mean is smoothed the same way, so `mae_mean_prior` compares the two targets and not
        # one target against an unsmoothed version of the other.
        means = priors._isotonic_decreasing(sub["mean_rate"].to_list(), w)
        for slot, raw, t, mean, sm, n in zip(
            sub["avail_slot"].to_list(), sub["raw_rate"].to_list(), sub["tau"].to_list(), means,
            smooth, sub["n"].to_list(), strict=True,
        ):
            rows.append({
                "position": pos, "avail_slot": int(slot), "tau": float(t), "raw_rate": float(raw),
                "mean_rate": float(mean), "prior_rate": float(sm),
                "prior_games": float(sm) * float(REG_WEEKS - 1), "n": int(n),
            })
    return pl.DataFrame(rows)


def slot_presence(panel: pl.DataFrame, before: int) -> pl.DataFrame:
    """The share of team-seasons in which a player this deep turns up in the season at all.

    This is the one thing the depth-slot games prior cannot say on its own, and the reason a naive
    version over-projects the bottom of a roster. The prior is conditional -- it averages over players
    who were charted in August or who played, so an eleventh receiver in it is an eleventh receiver
    who got promoted, and he played five games. A 90-man August roster in 2026 has twelve receivers
    on it and most of them will be cut before week 1.

    Measured as the count of panel rows at a slot over 32 team-seasons, so a slot every team fills
    reads 1.0 and one that turns up eight times in five years reads 0.05, and smoothed to be
    non-increasing because a deeper job cannot be more likely to exist than a shallower one. It is
    applied as a separate named factor rather than folded into the prior, so the number a user reads
    as "games when he is on the field" stays the number that was actually fitted.
    """
    h = panel.filter(pl.col("season") < before)
    n = h["season"].n_unique()
    if not n:
        return pl.DataFrame(schema={"position": pl.String, "avail_slot": pl.Int32,
                                    "presence": pl.Float64})
    agg = (
        h.group_by(["position", "avail_slot"])
        .agg((pl.len() / (32.0 * n)).clip(upper_bound=1.0).alias("raw"))
        .sort(["position", "avail_slot"])
    )
    rows = []
    for pos in agg["position"].unique().sort():
        sub = agg.filter(pl.col("position") == pos)
        smooth = priors._isotonic_decreasing(sub["raw"].to_list(), [1.0] * sub.height)
        for slot, p in zip(sub["avail_slot"].to_list(), smooth, strict=True):
            rows.append({"position": pos, "avail_slot": int(slot), "presence": float(p)})
    return pl.DataFrame(rows)


def _join_slot_prior(
    df: pl.DataFrame,
    prior: pl.DataFrame,
    want: tuple[str, ...] = ("prior_rate", "presence"),
) -> pl.DataFrame:
    """Attach the slot's prior columns, falling back to the deepest slot seen at that position."""
    if "prior_rate" not in prior.columns and "prior_games" in prior.columns:
        # An artifact written before availability became a rate. Readable rather than fatal.
        prior = prior.with_columns(
            (pl.col("prior_games") / float(REG_WEEKS - 1)).alias("prior_rate")
        )
    cols = [c for c in want if c in prior.columns]
    deepest = (
        prior.sort(["position", "avail_slot"])
        .group_by("position")
        .agg(*[pl.col(c).last().alias(f"_deep_{c}") for c in cols])
    )
    out = (
        df.join(prior.select("position", "avail_slot", *cols), on=["position", "avail_slot"],
                how="left")
        .join(deepest, on="position", how="left")
        .with_columns(*[pl.coalesce(c, f"_deep_{c}", pl.lit(0.0)).alias(c) for c in cols])
        .drop([f"_deep_{c}" for c in cols])
    )
    if "presence" not in out.columns:
        out = out.with_columns(pl.lit(1.0).alias("presence"))
    return out


def _own_games(hist: pl.DataFrame, target: int, settings: Settings) -> pl.DataFrame:
    """A player's recency-weighted attendance record from before `target`.

    `n_seasons` is the weight actually available divided by the full weight vector, so one season of
    history reads as 0.5 of the evidence three do -- which is what a fitted `k` in season units
    trades against.
    """
    total = float(sum(settings.recency))
    h = hist.with_columns(season_weights("season", target, settings.recency)).filter(
        pl.col("recency_weight") > 0
    )
    if h.is_empty():
        return pl.DataFrame(schema={"player_id": pl.String, "obs_rate": pl.Float64,
                                    "obs_games": pl.Float64, "n_seasons": pl.Float64,
                                    "seasons_seen": pl.UInt32})
    return h.group_by("player_id").agg(
        (
            (pl.col("rate") * pl.col("recency_weight")).sum() / pl.col("recency_weight").sum()
        ).alias("obs_rate"),
        (pl.col("recency_weight").sum() / total).alias("n_seasons"),
        pl.col("season").n_unique().alias("seasons_seen"),
    ).with_columns((pl.col("obs_rate") * float(REG_WEEKS - 1)).alias("obs_games"))


def fit_availability(settings: Settings | None = None) -> dict:
    """Grid-search the slot prior's quantile and the blending constant, per segment, on held-out
    seasons.

    Scored in games -- so the number is readable and comparable across refits -- against the two
    things the blend is made of: the player's own record alone (`k = 0`) and his slot's prior alone
    (`k = inf`). If neither endpoint is beaten there is no case for the blend, and the report says so
    rather than burying it. `mae_mean_prior` is the old mean-of-the-slot prior at the same `k`, kept as
    a standing check that the quantile is earning its place rather than being assumed into the model.

    `(tau, k)` is fitted separately for slots 1-2 and for everything behind them; see
    `AVAIL_FRONT_MAX` for why. MAE is a mean of per-row absolute errors, so the segments are additive
    and minimising each is minimising the whole -- the reported `mae` is the pooled number and is
    directly comparable with a single-pair fit.

    Everything is searched on the same held-out targets, which is why the prior is rebuilt inside the
    target loop: a 2022 slot prior must not know 2024.
    """
    settings = settings or Settings()
    seasons = tuple(range(HISTORY_FROM, LAST_COMPLETE_SEASON + 1))
    panel = _avail_panel(seasons)
    hist = _history_games(seasons)
    targets = [t for t in range(priors.FIT_FIRST_TARGET, LAST_COMPLETE_SEASON + 1)
               if not panel.filter(pl.col("season") == t).is_empty()]

    def build(tau: float) -> pl.DataFrame:
        """The whole held-out panel with a prior built at one flat `tau`, for scoring one segment."""
        frames = []
        for target in targets:
            cur = panel.filter(pl.col("season") == target)
            frames.append(
                _join_slot_prior(cur, slot_games_prior(panel, before=target, tau=tau),
                                 want=("prior_rate", "mean_rate", "presence"))
                .join(_own_games(hist, target, settings), on="player_id", how="left")
            )
        return pl.concat(frames, how="diagonal_relaxed")

    built = {tau: build(tau) for tau in GAMES_TAU_GRID}

    def part(scored: pl.DataFrame, segment: str) -> pl.DataFrame:
        front = pl.col("avail_slot") <= AVAIL_FRONT_MAX
        return scored.filter(front if segment == "front" else ~front)

    def errs(scored: pl.DataFrame, k: float, prior_col: str = "prior_rate") -> pl.Series:
        return scored.select(
            (
                shrink("obs_rate", prior_col, "n_seasons", k) * pl.col("team_games")
                - pl.col("games")
            ).abs().alias("e")
        )["e"]

    def mae(scored: pl.DataFrame, k: float, prior_col: str = "prior_rate") -> float:
        return float(errs(scored, k, prior_col).mean())

    out: dict = {
        "fitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seasons": [priors.FIT_FIRST_TARGET, LAST_COMPLETE_SEASON],
        "front_max_slot": AVAIL_FRONT_MAX,
    }
    k_by, tau_by, detail = {}, {}, {}
    pooled_err, pooled_n = 0.0, 0
    for segment in AVAIL_SEGMENTS:
        grid = {(tau, k): mae(part(built[tau], segment), k)
                for tau in GAMES_TAU_GRID for k in GAMES_K_GRID}
        (tau, k), best = min(grid.items(), key=lambda kv: kv[1])
        seg = part(built[tau], segment)
        e = errs(seg, k)
        pooled_err += float(e.sum())
        pooled_n += int(e.len())
        k_by[segment], tau_by[segment] = (k if k < 1e8 else float("inf")), tau
        detail[segment] = {
            "k": k_by[segment], "tau": tau, "mae": best, "n": int(seg.height),
            "mae_own_record": mae(seg, 0.0), "mae_slot_prior": mae(seg, 1e9),
            "mae_mean_prior": mae(seg, k, "mean_rate"),
            "projected_games": float(seg.select(
                (
                    shrink("obs_rate", "prior_rate", "n_seasons", k) * pl.lit(float(REG_WEEKS - 1))
                ).alias("g")
            )["g"].mean()),
            "actual_games": float(seg["games"].mean()),
            "grid": {str(kk): grid[(tau, kk)] for kk in GAMES_K_GRID},
            "tau_grid": {str(tt): grid[(tt, k)] for tt in GAMES_TAU_GRID},
        }

    whole = pl.concat([part(built[tau_by[s]], s) for s in AVAIL_SEGMENTS], how="diagonal_relaxed")
    own = sum(float(errs(part(built[tau_by[s]], s), 0.0).sum()) for s in AVAIL_SEGMENTS) / pooled_n
    prior_only = sum(
        float(errs(part(built[tau_by[s]], s), 1e9).sum()) for s in AVAIL_SEGMENTS
    ) / pooled_n
    best = pooled_err / pooled_n
    out |= {
        "k": k_by, "tau": tau_by, "segments": detail,
        "mae": best, "mae_own_record": own, "mae_slot_prior": prior_only,
        "mae_mean_prior": sum(
            float(errs(part(built[tau_by[s]], s), k_by[s], "mean_rate").sum())
            for s in AVAIL_SEGMENTS
        ) / pooled_n,
        "gain_vs_own_pct": 100.0 * (own - best) / own if own else 0.0,
        "gain_vs_prior_pct": 100.0 * (prior_only - best) / prior_only if prior_only else 0.0,
        "n": int(whole.height),
    }
    return out


def load_availability() -> dict:
    return json.loads(AVAIL_PATH.read_text()) if AVAIL_PATH.is_file() else {}


@lru_cache(maxsize=1)
def load_slot_games() -> pl.DataFrame:
    return pl.read_parquet(SLOT_GAMES_PATH) if SLOT_GAMES_PATH.is_file() else pl.DataFrame()


def slot_games_as_of(before: int, tau: float | None = None) -> pl.DataFrame:
    """The depth-slot attendance prior and presence rate, from seasons strictly before `before`.

    The saved parquet is the same thing fitted through the last complete season. This rebuilds it for
    a held-out target so that a 2022 twelfth receiver's survival odds are not informed by 2024.
    """
    panel = _avail_panel(tuple(range(HISTORY_FROM, before)))
    return slot_games_prior(panel, before=before, tau=tau).join(
        slot_presence(panel, before=before), on=["position", "avail_slot"], how="left"
    )


def availability(
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
    ros: pl.DataFrame | None = None,
    fitted: priors.Fitted | None = None,
) -> pl.DataFrame:
    """Expected games for every rostered player, before and after roster status and roster survival.

    Three numbers, kept apart on purpose, because they answer three questions and only the first is
    fitted on a population that matches its use:

    - `games_if_available` -- the blend of his own attendance record and his slot's prior, both as a
      rate of the team's games and then put back into seventeenths. This is the fitted quantity, and
      it is *conditional on being in the season at all*. It is deliberately not discounted for an
      injury that has not happened: the prior is the slot's median attendance, so a starter who has
      played when healthy projects close to a full season, and the only things that mark him down are
      a roster status that says he is hurt now and the range layer's own injury distribution.
    - `presence` -- how often a job that deep exists on an in-season roster. A 90-man August roster
      carries about 28.6 offensive players per team; the population that appears in a season is about
      21.9. Somebody has to be the difference, and depth slot is the only thing that says who.
    - `status_factor` -- the stated multiplier for anyone not ACT.

    `expected_games` is their product. Reading them separately is what lets a user see that a twelfth
    receiver's low projection is a roster-survival judgement rather than an injury one.
    """
    settings = settings or Settings()
    ros = roster(season) if ros is None else ros
    saved = load_availability()
    raw_k = fitted.games_k if fitted is not None and fitted.games_k is not None else saved.get("k")
    k = _by_segment(raw_k, settings.default_share_k)
    raw_tau = fitted.games_tau if fitted is not None and fitted.games_tau is not None else \
        saved.get("tau")
    tau = _by_segment(raw_tau, GAMES_TAU)

    prior = fitted.slot_games if fitted is not None else load_slot_games()
    if prior.is_empty():
        panel = _avail_panel(tuple(range(HISTORY_FROM, season)))
        prior = slot_games_prior(panel, before=season, tau=tau).join(
            slot_presence(panel, before=season), on=["position", "avail_slot"], how="left"
        )
    hist = _history_games(tuple(range(HISTORY_FROM, season)))

    games = float(REG_WEEKS - 1)
    k_expr = _segment_expr(k)
    out = (
        _join_slot_prior(ros, prior)
        .join(_own_games(hist, season, settings), on="player_id", how="left")
        .with_columns(
            shrink("obs_rate", "prior_rate", "n_seasons", k_expr).alias("available"),
            shrink_weight("n_seasons", k_expr).alias("games_own_weight"),
        )
        .with_columns(
            pl.col("status")
            .replace_strict(settings.status_availability, default=1.0, return_dtype=pl.Float64)
            .alias("status_factor"),
            # Back into games, which is the unit every reader and every override thinks in.
            (pl.col("available").clip(0.0, 1.0) * games).alias("games_if_available"),
            (pl.col("prior_rate") * games).alias("prior_games"),
        )
    )
    return out.with_columns(
        pl.min_horizontal(
            pl.col("games_if_available") * pl.col("presence") * pl.col("status_factor"),
            pl.lit(games),
        ).clip(lower_bound=0.0).alias("expected_games")
    ).with_columns((pl.col("expected_games") / games).alias("active_weeks"))


# --------------------------------------------------------------------------- #
# participation: how much of the offence when he is out there
# --------------------------------------------------------------------------- #
def participation_detail(
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
    ros: pl.DataFrame | None = None,
    fitted: priors.Fitted | None = None,
) -> pl.DataFrame:
    """One row per player per participation metric, showing every input to the answer.

    `obs` is the player, `prior` is his job, `used` is the blend, and `own_weight` is how much of
    `used` is him -- the four numbers a user needs to decide whether to override it. The arithmetic is
    `estimate.estimate`, which every other share and rate in the projection also goes through.
    """
    ros = roster(season) if ros is None else ros
    return estimate.estimate(PARTICIPATION_METRICS, ros, season, settings, fitted)


def participation(
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
    ros: pl.DataFrame | None = None,
    fitted: priors.Fitted | None = None,
) -> pl.DataFrame:
    """One row per rostered player: his depth, his expected games, and his participation rates.

    This is what `opportunity.py` consumes. `weekly_*` columns are the availability-weighted
    contribution -- participation times the fraction of the season he is there for -- which is the
    quantity that has to add up across a team.
    """
    settings = settings or Settings()
    ros = roster(season) if ros is None else ros
    avail = availability(season, settings, ros=ros, fitted=fitted)
    detail = participation_detail(season, settings, ros=ros, fitted=fitted)

    wide = detail.pivot(on="metric", index="player_id", values="used")
    have = [m for m in PARTICIPATION_METRICS if m in wide.columns]
    out = avail.join(wide, on="player_id", how="left")
    return out.with_columns(
        *[(pl.col(m).fill_null(0.0) * pl.col("active_weeks")).alias(f"weekly_{m}") for m in have]
    ).select(
        "season", "team", "player_id", "player", "position", "roster_position", "status",
        "depth_tier", "depth_slot", "slot_bucket", "avail_slot", "charted", "charted_anywhere",
        "team_disagreement", "alignment", "is_rookie", "draft_pick", "pick_bin", "years_exp", "age",
        "height", "weight", "prior_opp_per_game",
        "prior_games", "obs_games", "n_seasons", "games_own_weight", "games_if_available",
        "presence", "status_factor", "expected_games", "active_weeks",
        *have, *[f"weekly_{m}" for m in have],
    ).sort(["team", "position", "depth_slot"])


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def measure_benchmark(seasons: tuple[int, ...] | None = None) -> dict[str, float]:
    """What the availability-weighted participation sum actually comes to, per team-week.

    Over everyone who recorded anything, which is the population the 2026 roster corresponds to, and
    with exactly the arithmetic the projection uses. Five skill players are on the field for every
    snap, so this is the 5.0 identity net of the fact that each player's share is measured against
    his own games rather than the season -- which is why it reads 4.95 and not 5.00.
    """
    seasons = seasons or tuple(range(2021, LAST_COMPLETE_SEASON + 1))
    out: dict[str, float] = {}
    for name in PARTICIPATION_METRICS:
        metric = priors.BY_NAME[name]
        use = tuple(s for s in seasons if s >= metric.since)
        hist = priors._hist(metric.table)
        if not use or name not in hist.columns:
            continue
        per_team = (
            hist.filter(pl.col("season").is_in(use))
            .with_columns(
                (pl.col(name).fill_null(0.0) * pl.col("games") / float(REG_WEEKS - 1)).alias("w")
            )
            .group_by(["season", "team"])
            .agg(pl.col("w").sum())
        )
        out[name] = float(per_team["w"].mean())
    return out


# --------------------------------------------------------------------------- #
# rookies: how much of the prior is draft capital
def team_diagnostic(part: pl.DataFrame, benchmark: dict[str, float]) -> pl.DataFrame:
    """Per-team weekly participation sums against the measured benchmark."""
    cols = [c for c in part.columns if c.startswith("weekly_")]
    per_team = part.group_by("team").agg(
        pl.len().alias("players"),
        pl.col("expected_games").sum().alias("player_games"),
        *[pl.col(c).sum() for c in cols],
    )
    rows = []
    for c in cols:
        name = c.removeprefix("weekly_")
        bm = benchmark.get(name)
        rows.append({
            "metric": name,
            "projected_mean": float(per_team[c].mean()),
            "projected_min": float(per_team[c].min()),
            "projected_max": float(per_team[c].max()),
            "benchmark": bm,
            "gap_pct": 100.0 * (float(per_team[c].mean()) - bm) / bm if bm else None,
        })
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# fit, save, report
# --------------------------------------------------------------------------- #
def fit_all(settings: Settings | None = None) -> dict:
    settings = settings or Settings()
    ensure_dirs()
    seasons = tuple(range(HISTORY_FROM, LAST_COMPLETE_SEASON + 1))

    fit = fit_availability(settings)
    panel = _avail_panel(seasons)
    slot_prior = slot_games_prior(panel, before=PROJ_SEASON, tau=fit["tau"]).join(
        slot_presence(panel, before=PROJ_SEASON), on=["position", "avail_slot"], how="left"
    )

    fit["benchmark"] = measure_benchmark()
    fit["population_per_team"] = float(
        panel.filter(pl.col("season") >= 2021).group_by(["season", "team"]).len()["len"].mean()
    )

    slot_prior.write_parquet(SLOT_GAMES_PATH)
    load_slot_games.cache_clear()
    AVAIL_PATH.write_text(json.dumps(fit, indent=2))

    part = participation(PROJ_SEASON, settings)
    return {
        "fit": fit, "slot_prior": slot_prior, "participation": part,
        "rookie_blend": priors.load_rookie_blend(),
        "diagnostic": team_diagnostic(part, fit["benchmark"]),
    }


def clear_cache() -> None:
    roster.cache_clear()
    _games_by_team.cache_clear()
    _team_games.cache_clear()
    _draft.cache_clear()
    load_slot_games.cache_clear()


def _report(art: dict) -> None:
    pl.Config.set_tbl_width_chars(210)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_fmt_float("mixed")
    fit, part = art["fit"], art["participation"]

    print(f"\nROSTER {PROJ_SEASON}   {part.height} offensive players, {part['team'].n_unique()} teams")
    print(part.group_by("position").agg(
        pl.len().alias("players"),
        pl.col("charted").sum().alias("charted"),
        (~pl.col("charted")).sum().alias("unlisted"),
        pl.col("team_disagreement").sum().alias("team_moved"),
        pl.col("is_rookie").sum().alias("rookies"),
        pl.col("draft_pick").is_not_null().sum().alias("drafted"),
        pl.col("expected_games").mean().round(2).alias("exp_games"),
    ).sort("position"))
    bad = part.filter(pl.col("status") != "ACT")
    if not bad.is_empty():
        print(part.group_by("status").agg(
            pl.len(), pl.col("status_factor").first().round(2),
            pl.col("expected_games").mean().round(2).alias("exp_games"),
        ).sort("len", descending=True))

    print("\nAVAILABILITY  expected games, blend of own record and depth slot  (MAE in games)")
    seg = fit.get("segments") or {}
    print(pl.DataFrame([
        {
            "segment": name if name != "front" else f"front (slots 1-{fit.get('front_max_slot', 2)})",
            "slot_tau": d["tau"], "k_seasons": d["k"], "mae": round(d["mae"], 3),
            "mae_own_record": round(d["mae_own_record"], 3),
            "mae_slot_prior": round(d["mae_slot_prior"], 3),
            "mae_mean_prior": round(d["mae_mean_prior"], 3),
            "proj_games": round(d["projected_games"], 2),
            "actual_games": round(d["actual_games"], 2), "n": d["n"],
        }
        for name, d in seg.items()
    ] or [{"segment": "pooled", "mae": round(fit["mae"], 3)}]))
    print(pl.DataFrame([{
        "pooled_mae": round(fit["mae"], 3),
        "mae_own_record": round(fit["mae_own_record"], 3),
        "mae_slot_prior": round(fit["mae_slot_prior"], 3),
        "mae_mean_prior": round(fit["mae_mean_prior"], 3) if "mae_mean_prior" in fit else None,
        "gain_vs_own_%": round(fit["gain_vs_own_pct"], 1),
        "gain_vs_prior_%": round(fit["gain_vs_prior_pct"], 1),
        "n": fit["n"],
    }]))
    sp = art["slot_prior"]
    print("  games when he is in the season (left) x how often a job that deep exists (right)")
    print(sp.pivot(on="position", index="avail_slot", values="prior_games")
          .join(sp.pivot(on="position", index="avail_slot", values="presence"),
                on="avail_slot", suffix="_present")
          .sort("avail_slot").select(pl.all().round(2)))

    # The one number a reader checks this module against: a first-string player's projected season.
    print("\nSTARTERS  what slot 1 is projected to play, by position")
    print(part.filter(pl.col("depth_slot") == 1).group_by("position").agg(
        pl.len().alias("n"),
        pl.col("obs_games").mean().round(2).alias("own_record"),
        pl.col("games_if_available").mean().round(2).alias("if_available"),
        pl.col("expected_games").mean().round(2).alias("expected"),
        pl.col("expected_games").min().round(1).alias("lowest"),
        (pl.col("expected_games") >= 16.0).mean().round(2).alias("share_16_plus"),
    ).sort("position"))

    blend = art.get("rookie_blend") or {}
    rook = [{"metric": m, "form": blend[m][0], "w": blend[m][1]}
            for m in PARTICIPATION_METRICS if m in blend]
    if rook:
        print("\nROOKIE PRIORS  fitted in priors.py -- see python -m src.model.priors --report")
        print(pl.DataFrame(rook))

    detail = participation_detail(PROJ_SEASON)
    print("\nPARTICIPATION  how much of each answer is the player rather than his job")
    print(detail.group_by("metric").agg(
        pl.col("k").first(),
        pl.col("used").mean().round(3).alias("mean_used"),
        pl.col("own_weight").mean().round(2).alias("mean_own_weight"),
        (pl.col("source") == "blend").sum().alias("from_blend"),
        (pl.col("source") == "slot_prior").sum().alias("from_slot"),
        (pl.col("source") == "draft_blend").sum().alias("from_draft"),
    ).sort("metric"))

    print(f"\nTEAM SUMS  availability-weighted participation per week, vs the identity measured over"
          f" the whole population\n           ({fit.get('population_per_team', 0):.1f} players per"
          f" team in the fitted population, {part.height / 32:.1f} on a 2026 roster)")
    print(art["diagnostic"].with_columns(pl.col("^projected.*$").round(3),
                                         pl.col("benchmark").round(3),
                                         pl.col("gap_pct").round(1)))

    print("\nPHI, as an example")
    print(part.filter(pl.col("team") == "PHI").select(
        "position", "depth_slot", "player", "status", "is_rookie", "draft_pick",
        pl.col("expected_games").round(1),
        *[pl.col(c).round(3) for c in PARTICIPATION_METRICS if c in part.columns],
    ).head(22))
    print(f"\nwritten to {FITTED}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", action="store_true", help="refit availability and write artifacts")
    p.add_argument("--report", action="store_true", help="print from the saved artifacts")
    args = p.parse_args(argv)
    if args.report and not args.fit:
        fit = load_availability()
        part = participation(PROJ_SEASON)
        art = {
            "fit": fit, "slot_prior": load_slot_games(), "participation": part,
            "rookie_blend": priors.load_rookie_blend(),
            "diagnostic": team_diagnostic(part, fit.get("benchmark", {})),
        }
    else:
        art = fit_all()
    _report(art)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
