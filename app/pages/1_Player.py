"""One player, and every number the projection is made of.

The point of this page is that a projection is not an opinion to be accepted or rejected whole. It is
a product of four things -- will he play, how much of the offence goes through him, how good is he at
converting it, and what kind of games does his team play -- and each of those is separately arguable.
So each is shown separately, with the sample size behind it.

The decomposition table is the important one. For every share and rate:

- `obs` is what the player himself did, over `n` opportunities
- `prior` is what his job does -- his position and depth slot, plus a draft-slot curve if he is a rookie
- `used` is the blend, `own_weight` is how much of it is him, and `source` says which won

A number with `own_weight` near 1 is a measurement. A number with `own_weight` near 0 is an assumption
about a role, and that is the kind worth arguing with. Every one of them is editable -- on the team
page in the grid beside his teammates, where the consequence of taking a share off one of them is
visible, or on the Adjustments page as a list. Here they are shown with the sample size behind them,
which is what the argument should turn on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl                                                          # noqa: E402
import streamlit as st                                                       # noqa: E402

import ui                                                                    # noqa: E402

view = ui.controls("Player")
ui.scenario_banner(view)

me = ui.pick_player(view)
if me is None:
    st.stop()

pid, team, position = me["player_id"], me["team"], me["position"]
weekly = ui.weekly(view).filter(pl.col("player_id") == pid).sort("week")
part = ui.participation(view).filter(pl.col("player_id") == pid)
prow = part.row(0, named=True) if not part.is_empty() else {}

st.header(f"{me['player']} · {position} {team}")
ui.flag_row(
    rookie=bool(me.get("is_rookie")),
    changed_team=bool(me.get("changed_team")),
    startable=bool(me.get("startable")),
    **{f"status {me.get('status')}": me.get("status") not in (None, "ACT")},
)
ui.edited_badge(view, "player", pid)

# --------------------------------------------------------------------------- #
# the season line
# --------------------------------------------------------------------------- #
head = st.columns(6)
head[0].metric("Fantasy points", f"{me['fantasy_points']:.1f}")
head[1].metric("Per game", f"{me['points_per_game']:.2f}")
head[2].metric(f"{position} rank", f"{int(me['position_rank'])}", f"{me['overall_rank']:.0f} overall",
               delta_color="off")
head[3].metric("Games", f"{me['games']:.1f}", f"tier {int(me['tier'])}", delta_color="off")
head[4].metric("vs avg starter", f"{me['vs_starter']:+.1f}")
head[5].metric(
    "vs last season",
    f"{me['delta_points']:+.1f}" if me.get("delta_points") is not None else "—",
    f"{me.get('last_points') or 0:.0f} in {me.get('last_games') or 0:.0f} g",
    delta_color="off",
)

if not view.scenario.is_baseline:
    mine = ui.board_diff(view).filter(pl.col("player_id") == pid)
    if mine.is_empty():
        ui.note("This scenario does not move him: the same points as the baseline, to rounding.")
    else:
        d = mine.row(0, named=True)
        ranks = (f"{position} rank {int(d['position_rank'])} → {int(d['new_position_rank'])}"
                 if d["position_rank"] is not None and d["new_position_rank"] is not None else "")
        ui.note(
            f"Against the baseline: {d['d_fantasy_points']:+.1f} points"
            + (f", {d['d_targets']:+.1f} targets" if d.get("d_targets") is not None else "")
            + (f", {d['d_carries']:+.1f} carries" if d.get("d_carries") is not None else "")
            + (f" · {ranks}" if ranks else "")
        )

STAT_GROUPS = {
    "QB": ("attempts", "completions", "passing_yards", "passing_tds", "interceptions",
           "designed_rushes", "scrambles", "rushing_yards", "rushing_tds"),
    "RB": ("carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards",
           "receiving_tds", "offense_snaps", "routes"),
    "WR": ("targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
           "carries", "rushing_yards", "routes", "offense_snaps"),
    "TE": ("targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
           "routes", "offense_snaps"),
}
stats = [c for c in STAT_GROUPS.get(position, STAT_GROUPS["WR"]) if c in weekly.columns]

st.subheader("Projected season")
season_line = pl.DataFrame({
    "stat": stats,
    "season": [float(weekly[c].sum()) for c in stats],
    "per_game": [float(weekly[c].sum()) / max(float(weekly["p_play"].sum()), 1e-9) for c in stats],
})
ui.table(season_line, config=ui.fixed(2, "season", "per_game"))
ui.note(
    "`per_game` is over projected games played, not over 17: a player expected to miss three weeks is "
    "scored over the 14 he is there for, which is the number a lineup decision needs."
)

# --------------------------------------------------------------------------- #
# the range around it
# --------------------------------------------------------------------------- #
st.subheader("The range around it")
ui.note(
    "Ten thousand seasons of the frame above, shocked at three levels: the game both teams are in, his "
    "team's volume, then his own share and his own efficiency. Every dispersion is measured from "
    "2016–2025 residuals rather than assumed, and the shocks are correlated, so his ceiling arrives "
    "with his quarterback's."
)
sim = ui.sim_controls(view, detail=(pid,), key="player_sim")

if sim is None:
    st.info("Turn on **Ranges** to simulate the season and see his floor, ceiling and volatility.")
else:
    mine = sim.season.filter(pl.col("player_id") == pid)
    if mine.is_empty():
        st.info("He is not in the simulated frame — no projected volume to shock.")
    else:
        r = mine.row(0, named=True)
        row = st.columns(6)
        row[0].metric("Median", f"{r['p50']:.0f}", f"mean {r['sim_mean']:.0f}", delta_color="off")
        row[1].metric("Floor (P5)", f"{r['p5']:.0f}", f"P25 {r['p25']:.0f}", delta_color="off")
        row[2].metric("Ceiling (P95)", f"{r['p95']:.0f}", f"P75 {r['p75']:.0f}", delta_color="off")
        row[3].metric("Volatility", f"{r['volatility']:.0%}",
                      f"{position} rank by median {int(r['median_rank'])}", delta_color="off")
        row[4].metric("Boom weeks", f"{r['boom_rate']:.0%}",
                      f"over {sim.dispersion.thresholds(position)[0]:.0f}", delta_color="off")
        row[5].metric("Bust weeks", f"{r['bust_rate']:.0%}",
                      f"under {sim.dispersion.thresholds(position)[1]:.0f}", delta_color="off")

        st.caption("The simulated season, plotted")
        st.bar_chart(sim.histogram(pid), x="points", y="share", height=260)
        ui.note(
            "Each bar is the share of seasons landing in that band. The bar at zero is the seasons he "
            "never took the field in — for a fragile player it is the most important feature of the "
            "range, so it gets its own bin rather than being smoothed into the first one."
        )

        over, under = st.columns([2, 5])
        with over:
            line = st.number_input("P(over) points", min_value=0.0, value=float(round(r["p50"], 0)),
                                   step=5.0, help="Odds of clearing this many points over the season.")
        with under:
            st.metric(f"P(over {line:.0f})", f"{sim.p_over(pid, float(line)):.1%}")
            st.caption(
                f"Read against a games range of {r['games_p5']:.0f}–{r['games_p95']:.0f} "
                f"(median {r['games_p50']:.0f}) against {r['projected_games']:.1f} projected."
            )

        st.caption("Ranks by median, floor and ceiling")
        board_rank = sim.season.filter(pl.col("position") == position).select(
            "player", "team", "projected", "p50", "floor", "ceiling", "volatility",
            "median_rank", "floor_rank", "ceiling_rank",
        ).sort("median_rank")
        near = board_rank.with_columns(
            (pl.col("median_rank") - int(r["median_rank"])).abs().alias("near")
        ).sort("near").head(11).sort("median_rank").drop("near")
        ui.table(near.with_columns((pl.col("player") == me["player"]).alias("this_player")),
                 config=ui.range_config())
        ui.note(
            "The ten players nearest him at the position by median. Where his `floor_rank` beats his "
            "`ceiling_rank` he is the safe pick of the group; where it is the other way round he is the "
            "one you take when you need the upside."
        )

        if not sim.stats.is_empty():
            st.caption("Each stat, with its own range")
            ui.table(sim.stats.filter(pl.col("player_id") == pid).drop("player_id"),
                     config=ui.range_config(), height=380)

# --------------------------------------------------------------------------- #
# the seventeen games
# --------------------------------------------------------------------------- #
st.subheader("Week by week")
env = ui.environment(view).filter(pl.col("team") == team).select(
    "week", "opponent", "is_home", "spread", "total", "implied_points", "has_market", "plays",
    "dropbacks", "carries", "targets", "pass_tds", "rush_tds",
).rename({"carries": "team_carries", "targets": "team_targets", "plays": "team_plays",
          "dropbacks": "team_dropbacks", "pass_tds": "team_pass_tds", "rush_tds": "team_rush_tds"})

games = weekly.select(
    "week", "opponent", "is_home", "p_play", *stats, "fantasy_points"
).join(env.select("week", "spread", "total", "implied_points", "has_market"), on="week", how="left")

st.bar_chart(games.select("week", "fantasy_points"), x="week", y="fantasy_points", height=240)
ui.note(
    "Each bar is an expectation, availability included: a bye week is absent and a week he is only "
    "60% likely to play is 60% of a line, not a full one."
)

# with the sim run, each week carries its own range -- which is the number a start/sit decision wants,
# because a 12-point expectation that is 4-to-26 is a different call from one that is 10-to-14
if sim is not None and not sim.weekly.is_empty():
    wr = sim.weekly.filter(pl.col("player_id") == pid).select(
        "week", pl.col("p5").alias("week_p5"), pl.col("p50").alias("week_p50"),
        pl.col("p95").alias("week_p95"), pl.col("boom_rate").alias("week_boom"),
        pl.col("bust_rate").alias("week_bust"),
    )
    games = games.join(wr, on="week", how="left")
    st.line_chart(games.select("week", "week_p5", "week_p50", "week_p95"), x="week", height=240)
    ui.note(
        "The same weeks as percentiles of the simulated distribution rather than as expectations. The "
        "floor line is where availability shows up: a week he might miss has a floor of nothing however "
        "good the matchup is."
    )

ui.table(
    games,
    config={
        **ui.fixed(2, "fantasy_points", "p_play", *stats),
        **ui.fixed(1, "spread", "total", "implied_points"),
        **ui.range_config(per_game=True),
        **ui.fixed(2, "week_p5", "week_p50", "week_p95"),
        **ui.percent("week_boom", "week_bust"),
    },
    height=560,
)

# --------------------------------------------------------------------------- #
# what the projection is made of
# --------------------------------------------------------------------------- #
st.subheader("What the number is made of")
DETAIL = ["metric", "kind", "units", "obs", "n", "prior_slot", "curve", "prior", "used",
          "own_weight", "k", "seasons_used", "source"]


def detail_for(frame: pl.DataFrame) -> pl.DataFrame:
    got = frame.filter(pl.col("player_id") == pid)
    return got.select([c for c in DETAIL if c in got.columns]).sort("metric")


cfg = {
    **ui.fixed(4, "obs", "prior_slot", "curve", "prior", "used"),
    **ui.percent("own_weight"),
    **ui.fixed(1, "n", "k"),
}
d1, d2, d3 = st.tabs(["Shares — who gets the ball", "Rates — what he does with it", "Availability"])
with d1:
    ui.table(detail_for(ui.share_detail(view)), config=cfg, height=520)
with d2:
    ui.table(detail_for(ui.rate_detail(view)), config=cfg, height=520)
with d3:
    ui.table(detail_for(ui.participation_detail(view)), config=cfg)
    avail = pl.DataFrame([{
        "prior_games": prow.get("prior_games"), "obs_games": prow.get("obs_games"),
        "games_own_weight": prow.get("games_own_weight"),
        "games_if_available": prow.get("games_if_available"),
        "presence": prow.get("presence"), "status": prow.get("status"),
        "status_factor": prow.get("status_factor"),
        "expected_games": prow.get("expected_games"), "active_weeks": prow.get("active_weeks"),
    }]) if prow else pl.DataFrame()
    if not avail.is_empty():
        ui.table(avail, config={**ui.fixed(2, "prior_games", "obs_games", "games_if_available",
                                           "expected_games"),
                                **ui.percent("games_own_weight", "presence", "status_factor",
                                             "active_weeks")})
        ui.note(
            "`expected_games` is his own games history blended with his slot's, times `presence` "
            "(is a man in this slot on the field at all) times a stated factor for his roster status. "
            "`active_weeks` is that as a fraction of the season, and it multiplies every share before "
            "the team's pool is divided."
        )

ui.note(
    "`own_weight` is n / (n + k): how much of `used` is the player rather than his job. Low weight is "
    "not a defect — it is the correct answer for a rookie or a backup — but it is the number to argue "
    "with, because it is a claim about a role rather than a measurement of a person."
)

# --------------------------------------------------------------------------- #
# history
# --------------------------------------------------------------------------- #
st.subheader("What he has actually done")
hist = ui.actuals(ui.HISTORY_VIEW, view.scoring).filter(pl.col("player_id") == pid).sort("season")
if hist.is_empty():
    st.info("No games in the processed tables — a rookie, or a player who has not taken a snap.")
else:
    hcols = [c for c in ("games", "fantasy_points", *stats) if c in hist.columns]
    past = hist.select("season", "team", *hcols).with_columns(
        (pl.col("fantasy_points") / pl.col("games")).alias("points_per_game")
    )
    projected = pl.DataFrame([{
        "season": view.season, "team": team,
        **{c: (float(weekly[c].sum()) if c in weekly.columns
               else float(me.get(c) or 0.0)) for c in hcols},
        "points_per_game": float(me["points_per_game"]),
    }]).with_columns(pl.col("games").cast(pl.Float64))
    both = pl.concat([past, projected.select(past.columns)], how="vertical_relaxed")
    ui.table(both, config={**ui.fixed(1, "fantasy_points", "games", *stats),
                           **ui.fixed(2, "points_per_game")})
    st.bar_chart(both.select("season", "points_per_game"), x="season", y="points_per_game", height=220)
    ui.note(
        "The last row is the projection, on the same scoring as the rows above it. Points per game "
        "rather than totals, so a season cut short by injury is not read as a decline in ability."
    )

# --------------------------------------------------------------------------- #
# competition
# --------------------------------------------------------------------------- #
st.subheader("Who he is competing with")
POOL_OF = {"QB": ("dropbacks", "carries"), "RB": ("carries", "targets"),
           "WR": ("targets", "air_yards"), "TE": ("targets", "air_yards")}
opp = ui.opportunity_frame(view).filter(pl.col("team") == team)
pools = [p for p in POOL_OF.get(position, ("targets",)) if p in opp.columns]
mates = (
    opp.group_by(["player_id", "player", "position", "depth_slot", "slot_bucket"])
    .agg(pl.col("p_play").mean().alias("p_play"),
         *[pl.col(f"share_{p}").mean().alias(f"share_{p}") for p in pools],
         *[pl.col(p).sum().alias(p) for p in pools])
    .sort(pools[0], descending=True)
)
ui.table(
    mates.with_columns((pl.col("player_id") == pid).alias("this_player")),
    config={**ui.percent("p_play", *[f"share_{p}" for p in pools]), **ui.fixed(1, *pools)},
    height=420,
)
ui.note(
    f"Season totals for the {team} {position} room and everyone else drawing on the same pools "
    f"({', '.join(pools)}). The shares here are availability-weighted and post-normalisation, so they "
    "are what actually divided the team's total."
)

with st.expander("The depth chart this came from"):
    chart = ui.roster_frame(view).filter(pl.col("team") == team).select(
        [c for c in ("position", "depth_tier", "depth_slot", "slot_bucket", "player", "status",
                     "is_rookie", "draft_pick", "years_exp", "age", "charted", "team_disagreement",
                     "alignment") if c in ui.roster_frame(view).columns]
    ).sort(["position", "depth_slot"])
    ui.table(chart, height=520)

# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
st.subheader("Against somebody else")
other = ui.pick_player(view, label="Compare with", key="compare", index=1)
if other is not None and other["player_id"] != pid:
    fields = ["player", "team", "position", "position_rank", "tier", "games", "fantasy_points",
              "points_per_game", "vs_starter", "drop_next", "delta_points"]
    board = ui.board(view)
    pair = board.filter(pl.col("player_id").is_in([pid, other["player_id"]])).select(
        [c for c in fields if c in board.columns]
    )
    ui.table(pair, config={**ui.fixed(1, "fantasy_points", "vs_starter", "drop_next",
                                      "delta_points", "games"),
                           **ui.fixed(2, "points_per_game")})
    a = ui.weekly(view).filter(pl.col("player_id") == pid).select(
        "week", pl.col("fantasy_points").alias(me["player"]))
    b = ui.weekly(view).filter(pl.col("player_id") == other["player_id"]).select(
        "week", pl.col("fantasy_points").alias(other["player"]))
    st.line_chart(a.join(b, on="week", how="full", coalesce=True).sort("week"), x="week", height=260)
    ui.note("A gap in a line is a bye week, not a zero.")
