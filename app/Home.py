"""The board: every projected player, ranked, with the comparisons a draft is decided on.

Rank alone is not a decision. Two receivers ranked 14th and 15th are the same player; a running back
ranked 24th with 40 points of daylight to 25th is a cliff. So the default columns are the ones that
answer *how much*: points, per game, points above the average starter at the position, and the drop
to the next man at that position.

Ranges are the same argument one level down. A projection is a mean, and two players with the same mean
are not the same pick, so floor / median / ceiling and a volatility score sit behind the **Ranges**
switch. Behind a switch because they cost a Monte Carlo run of the current frame -- ten thousand
seasons, shocked at the game, the team and the player -- and because a range nobody asked for is a
number nobody checked.

`streamlit run app/Home.py`
"""

from __future__ import annotations

import polars as pl
import streamlit as st

import ui

view = ui.controls("Board")
ui.scenario_banner(view)

board = ui.board(view)

DEFAULT = ["overall_rank", "position_rank", "tier", "player", "team", "position", "games",
           "fantasy_points", "points_per_game", "vs_starter", "drop_next", "delta_points"]

# With a scenario live, the board carries what the edits did to each player -- `vs_baseline` points and
# the position ranks he gained -- because a ranking is only worth as much as the reason it moved.
if not view.scenario.is_baseline:
    d = ui.board_diff(view).select(
        "player_id",
        pl.col("d_fantasy_points").alias("vs_baseline"),
        (pl.col("position_rank") - pl.col("new_position_rank")).alias("ranks_gained"),
    )
    board = board.join(d, on="player_id", how="left")
    DEFAULT = DEFAULT[:-1] + ["vs_baseline", "ranks_gained", "delta_points"]

OPTIONAL = ["targets", "receptions", "receiving_yards", "receiving_tds", "carries", "rushing_yards",
            "rushing_tds", "attempts", "completions", "passing_yards", "passing_tds",
            "interceptions", "offense_snaps", "routes", "depth_slot", "slot_bucket", "status",
            "draft_pick", "last_team", "last_games", "last_points"]

# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #
left, mid, right = st.columns([2, 2, 3])
with left:
    positions = ui.position_filter()
with mid:
    teams = ui.team_filter(view)
with right:
    search = st.text_input("Search", placeholder="part of a name")

row = st.columns([3, 2, 2])
with row[0]:
    extra = st.multiselect("Extra columns", OPTIONAL, default=[])
with row[1]:
    only_startable = st.toggle(
        "Startable only", value=False,
        help=f"Inside the position's starter count: {view.settings.starters}",
    )
with row[2]:
    limit = st.number_input("Rows", min_value=25, max_value=1000, value=200, step=25)

# --------------------------------------------------------------------------- #
# ranges, on demand
# --------------------------------------------------------------------------- #
RANGE = ["p50", "floor", "ceiling", "range", "volatility", "boom_rate", "bust_rate"]
sim = ui.sim_controls(view)
if sim is not None:
    board = board.join(
        sim.season.select("player_id", *RANGE, "sim_mean", "games_p5", "median_rank", "ceiling_rank",
                          "floor_rank"),
        on="player_id", how="left",
    )
    # once a board has ranges on it, the median is the honest thing to rank by: the projection is a
    # mean, and a mean is pulled by a ceiling the player reaches one season in twenty
    DEFAULT = DEFAULT[:1] + ["median_rank", "ceiling_rank"] + DEFAULT[1:-1] + RANGE + [DEFAULT[-1]]

shown = ui.apply_filters(board, positions, teams, search)
if only_startable:
    shown = shown.filter(pl.col("startable"))

# --------------------------------------------------------------------------- #
# headline
# --------------------------------------------------------------------------- #
cols = st.columns(4)
cols[0].metric("Players projected", f"{board.height:,}")
cols[1].metric("Shown", f"{shown.height:,}")
cols[2].metric("Points on the board", f"{board['fantasy_points'].sum():,.0f}")
moved = board.filter(pl.col("changed_team")) if "changed_team" in board.columns else board.head(0)
cols[3].metric("Changed team", f"{moved.height:,}")

order = [c for c in DEFAULT if c in shown.columns] + [c for c in extra if c in shown.columns]
config = {
    **ui.fixed(1, "fantasy_points", "vs_starter", "drop_next", "delta_points", "last_points",
               "vs_baseline"),
    **ui.fixed(2, "points_per_game"),
    **ui.fixed(1, "games", "last_games"),
    **ui.range_config(),
}
ui.table(shown.head(int(limit)), config=config, order=order, height=620)
ui.note(
    "`vs_starter` is points above the mean of the startable players at that position; `drop_next` is "
    "the gap to the next man at the same position; `delta_points` is against what he actually scored "
    "last season under the same scoring. Blank `delta_points` means he did not play then."
    + ("" if view.scenario.is_baseline else
       " `vs_baseline` and `ranks_gained` are this scenario against the same season with no edits.")
    + ("" if sim is None else
       " `floor` and `ceiling` are the 5th and 95th percentiles of the simulated season, `range` the "
       "distance between them, and `volatility` the spread as a fraction of the mean — the number that "
       "separates the safe 250-point back from the volatile one. `boom_rate` and `bust_rate` are the "
       "share of weeks over and under the measured thresholds for the position.")
)

if sim is not None:
    st.subheader("Who the range disagrees about")
    ui.note(
        "Players whose median rank is furthest from where the mean projection puts them. A player above "
        "the line is one the mean flatters — his projection is carried by a ceiling he rarely reaches; "
        "one below is the opposite, and is usually the safer pick at the same cost."
    )
    gap = shown.with_columns(
        (pl.col("overall_rank") - pl.col("median_rank")).alias("rank_gap")
    ).filter(pl.col("median_rank").is_not_null())
    disagree = pl.concat([
        gap.sort("rank_gap").head(12), gap.sort("rank_gap", descending=True).head(12),
    ]).unique(subset="player_id", keep="first").sort("rank_gap")
    ui.table(
        disagree.select("player", "position", "team", "overall_rank", "median_rank", "rank_gap",
                        "fantasy_points", "p50", "floor", "ceiling", "volatility"),
        config={**ui.range_config(), **ui.fixed(1, "fantasy_points")}, height=420,
    )

    left, right = st.columns(2)
    with left:
        st.caption("Safest startable seasons — lowest volatility")
        safe = shown.filter(pl.col("startable") & pl.col("volatility").is_not_null())
        ui.table(safe.sort("volatility").head(15).select(
            "player", "position", "team", "p50", "floor", "ceiling", "volatility"),
            config=ui.range_config(), height=560)
    with right:
        st.caption("Highest ceilings for the projection — most volatile")
        ui.table(safe.sort("volatility", descending=True).head(15).select(
            "player", "position", "team", "p50", "floor", "ceiling", "volatility"),
            config=ui.range_config(), height=560)

# --------------------------------------------------------------------------- #
# position shape
# --------------------------------------------------------------------------- #
st.subheader("Where each position runs out")
ui.note(
    "Projected points by position rank. The knee in a curve is the round the position stops being "
    "worth reaching for."
)
curve = (
    board.filter(pl.col("position_rank") <= 48)
    .select("position_rank", "position", "fantasy_points")
    .pivot(on="position", index="position_rank", values="fantasy_points")
    .sort("position_rank")
)
st.line_chart(curve, x="position_rank", height=300)

tabs = st.tabs(["By position", "Tiers", "Movers"])

with tabs[0]:
    by_pos = board.group_by("position").agg(
        pl.len().alias("players"),
        pl.col("fantasy_points").sum().alias("points"),
        pl.col("fantasy_points").mean().alias("mean_points"),
        pl.when(pl.col("startable")).then(pl.col("fantasy_points")).otherwise(None)
        .mean().alias("mean_starter"),
        pl.col("fantasy_points").max().alias("best"),
    ).sort("points", descending=True)
    ui.table(by_pos, config=ui.fixed(1, "points", "mean_points", "mean_starter", "best"))

with tabs[1]:
    tiers = (
        ui.apply_filters(board, positions, teams, "")
        .filter(pl.col("tier") <= 8)
        .group_by(["position", "tier"]).agg(
            pl.len().alias("n"),
            pl.col("fantasy_points").max().alias("top"),
            pl.col("fantasy_points").min().alias("bottom"),
            pl.col("player").sort_by("fantasy_points", descending=True).str.join(", ").alias("players"),
        ).sort(["position", "tier"])
    )
    ui.table(tiers, config=ui.fixed(1, "top", "bottom"), height=520)
    ui.note(f"Tiers are blocks of {view.settings.tier_size} within a position, in projected order.")

with tabs[2]:
    if "delta_points" in board.columns:
        movers = board.filter(pl.col("delta_points").is_not_null()).select(
            "player", "position", "team", "last_team", "changed_team", "last_games", "games",
            "last_points", "fantasy_points", "delta_points",
        )
        up, down = st.columns(2)
        cfg = ui.fixed(1, "last_points", "fantasy_points", "delta_points", "games", "last_games")
        with up:
            st.caption("Biggest projected gains on last season")
            ui.table(movers.sort("delta_points", descending=True).head(25), config=cfg)
        with down:
            st.caption("Biggest projected falls")
            ui.table(movers.sort("delta_points").head(25), config=cfg)
        ui.note(
            "Read the falls with the games column beside them: most of the largest are players who "
            "played 17 games last season and are projected for fewer, or who were last season's "
            "outliers. A projection regresses an outlier by construction."
        )
    else:
        st.info("No prior season available to compare against.")
