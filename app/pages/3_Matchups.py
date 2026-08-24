"""The schedule as an input: who each team plays, when, and what that does to a projection.

Two different things get called a matchup and they are worth separating.

**The game's scoring level** — is this a shootout or a slog. That comes from the posted line where one
exists and from the team estimate where it does not, and it is the part of a matchup with real
predictive weight: it moves plays, red-zone trips and touchdowns.

**The opponent's defence against a position** — the fantasy convention. It is shown here as last
season's measured rate allowed, and it is shown as *history* rather than as a projection, because a
single season of defence-versus-position is a small and noisy sample. The engine's own defensive term
is the league-normalised season rating on the Team page, not this table. Read this to explain a
number, not to build one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl                                                          # noqa: E402
import streamlit as st                                                       # noqa: E402

import ui                                                                    # noqa: E402
from src.config import LAST_COMPLETE_SEASON, POSITIONS                       # noqa: E402

view = ui.controls("Matchups")
ui.scenario_banner(view)

env = ui.environment(view)
shape = ui.shape(view)

# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #
st.subheader("The schedule")
metric = st.radio(
    "Cell", ["opponent", "implied_points", "total", "spread", "plays", "pass_attempts", "carries"],
    horizontal=True,
)
if metric == "opponent":
    cell = pl.when(pl.col("is_home")).then(pl.col("opponent")) \
        .otherwise(pl.lit("@ ") + pl.col("opponent"))
    grid = env.select("team", "week", cell.alias("v")).pivot(on="week", index="team", values="v")
    ui.table(grid.sort("team"), height=640)
    ui.note("`@` is away. A missing week is the bye.")
else:
    grid = env.select("team", "week", metric).pivot(on="week", index="team", values=metric)
    weeks = [c for c in grid.columns if c != "team"]
    ui.table(grid.sort("team"), config=ui.fixed(1, *weeks), height=640)
    ui.note(
        "Blank is the bye week. `implied_points` is the blended scoring level for that team in that "
        "game — the number every count in the projection is scaled by."
    )

# --------------------------------------------------------------------------- #
# leaderboards
# --------------------------------------------------------------------------- #
st.subheader("What kind of games each team plays")
lead = (
    env.group_by("team").agg(
        pl.col("implied_points").mean().alias("implied_pg"),
        pl.col("total").mean().alias("total_pg"),
        pl.col("spread").mean().alias("spread_pg"),
        pl.col("plays").mean().alias("plays_pg"),
        pl.col("dropbacks").mean().alias("dropbacks_pg"),
        pl.col("carries").mean().alias("carries_pg"),
        pl.col("seconds_per_play").mean().alias("seconds_per_play"),
        pl.col("has_market").sum().alias("games_with_a_line"),
    )
    .with_columns(
        (pl.col("dropbacks_pg") / pl.col("plays_pg")).alias("pass_rate"),
        pl.col("implied_pg").rank("min", descending=True).cast(pl.Int32).alias("implied_rank"),
        pl.col("plays_pg").rank("min", descending=True).cast(pl.Int32).alias("volume_rank"),
    )
    .sort("implied_pg", descending=True)
)
ui.table(
    lead,
    config={**ui.fixed(2, "implied_pg", "total_pg", "spread_pg", "plays_pg", "dropbacks_pg",
                       "carries_pg", "seconds_per_play"),
            **ui.percent("pass_rate")},
    height=620,
)
ui.note(
    "`seconds_per_play` is pace: a low number is more snaps for the same number of drives. Volume and "
    "scoring level are separate things and a team can be high in one and low in the other."
)

# --------------------------------------------------------------------------- #
# defence against a position
# --------------------------------------------------------------------------- #
st.subheader(f"Defence against each position, {LAST_COMPLETE_SEASON}")
dbp = ui.defence_by_position()
position = st.radio("Position", list(POSITIONS), horizontal=True, key="dpos")
faced = dbp.filter(pl.col("position") == position).select(
    pl.col("defense").alias("team"), "games", "targets_pg", "carries_pg", "fantasy_points_pg",
    "yards_per_target_allowed", "yards_per_carry_allowed", "catch_rate_allowed", "tds_allowed_pg",
    pl.col("fantasy_points_pg__factor").alias("points_factor"),
    pl.col("targets_pg__factor").alias("targets_factor"),
).sort("fantasy_points_pg", descending=True)
ui.table(
    faced,
    config={**ui.fixed(2, "targets_pg", "carries_pg", "fantasy_points_pg", "tds_allowed_pg",
                       "yards_per_target_allowed", "yards_per_carry_allowed"),
            **ui.percent("catch_rate_allowed"),
            **ui.fixed(3, "points_factor", "targets_factor")},
    height=620,
)
ui.note(
    f"Measured over {LAST_COMPLETE_SEASON} only, so a `points_factor` of 1.20 is one season of "
    "17 games against that position — informative, not decisive. It is not what the engine uses; the "
    "chain reads the league-normalised season ratings instead."
)

# --------------------------------------------------------------------------- #
# best and worst weeks
# --------------------------------------------------------------------------- #
st.subheader("Softest and hardest weeks")
ui.note(
    f"Every team-week joined to what its opponent allowed to {position}s in {LAST_COMPLETE_SEASON}. "
    "Sorted by the opponent's fantasy points factor, with that game's own scoring level beside it — "
    "the two reasons a week can be a good one, kept apart so you can see which is doing the work."
)
weeks = env.select("team", "week", "opponent", "is_home", "implied_points", "total", "spread").join(
    faced.select(pl.col("team").alias("opponent"), "points_factor", "targets_factor",
                 pl.col("fantasy_points_pg").alias("opp_points_allowed_pg")),
    on="opponent", how="left",
)
picked_team = st.selectbox("Team", ["— every team —"] + sorted(env["team"].unique().to_list()))
scope = weeks if picked_team.startswith("—") else weeks.filter(pl.col("team") == picked_team)
soft, hard = st.columns(2)
cfg = {**ui.fixed(3, "points_factor", "targets_factor"),
       **ui.fixed(1, "implied_points", "total", "spread", "opp_points_allowed_pg")}
with soft:
    st.caption("Softest")
    ui.table(scope.sort("points_factor", descending=True).head(20), config=cfg)
with hard:
    st.caption("Hardest")
    ui.table(scope.sort("points_factor").head(20), config=cfg)

# --------------------------------------------------------------------------- #
# one week
# --------------------------------------------------------------------------- #
st.subheader("One week")
week = st.slider("Week", int(env["week"].min()), int(env["week"].max()), 1)
games = env.filter(pl.col("week") == week).select(
    "game_id", "team", "opponent", "is_home", "roof", "temp", "wind", "rest_days", "div_game",
    "spread", "total", "implied_points", "has_market", "plays", "dropbacks", "carries", "targets",
    "pass_tds", "rush_tds",
).sort(["game_id", "is_home"], descending=[False, True])
ui.table(
    games,
    config={**ui.fixed(1, "spread", "total", "implied_points", "plays", "dropbacks", "carries",
                       "targets", "temp", "wind"),
            **ui.fixed(2, "pass_tds", "rush_tds")},
    height=620,
)

byes = sorted(set(shape["team"].to_list()) - set(games["team"].to_list()))
if byes:
    st.caption(f"On bye: {', '.join(byes)}")

top = ui.weekly(view).filter(pl.col("week") == week).sort("fantasy_points", descending=True).select(
    "player", "position", "team", "opponent", "is_home", "p_play", "targets", "carries",
    "receptions", "receiving_yards", "rushing_yards", "passing_yards", "fantasy_points",
).head(40)
st.caption(f"Highest projected players in week {week}")
ui.table(
    top,
    config={**ui.fixed(2, "fantasy_points", "targets", "carries", "receptions"),
            **ui.fixed(1, "receiving_yards", "rushing_yards", "passing_yards"),
            **ui.percent("p_play")},
    height=520,
)
ui.note(
    "A weekly line is an expectation including availability, so a player at 60% to play shows 60% of "
    "a line. Per-game context is the one part of the engine the held-out backtest could not prove at "
    "season level — it is a dead heat there — so treat the week-to-week spread as informative about "
    "the schedule and unproven as a weekly ranking."
)
