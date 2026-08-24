"""One team: the offence it is projected to run, and the roster that has to add up to it.

A player projection is a share of a team, so the team is the thing that has to be right first. This
page is laid out in that order -- season shape, then the seventeen games, then the roster dividing
them -- and it ends with the two audits that say whether the division was coherent: what each pool's
shares summed to before scaling, and whether the players sum back to the team they came from.

Inputs on the left, projections on the right. The left column is what the estimator believes about a
player and is editable in place; the right is what that belief produces once it is multiplied by the
team. Type over a number and the projection on the right is the next thing you see, along with what
the edit did to everybody else on the roster -- because a share taken is a share taken from a
teammate, and that consequence is the point of editing here rather than in a spreadsheet.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl                                                          # noqa: E402
import streamlit as st                                                       # noqa: E402

import ui                                                                    # noqa: E402
from src.model import overrides                                              # noqa: E402

view = ui.controls("Team")
ui.scenario_banner(view)

shape = ui.shape(view)
teams = sorted(shape["team"].to_list())
picked = st.selectbox("Team", teams, index=0)

env = ui.environment(view).filter(pl.col("team") == picked).sort("week")
board = ui.board(view).filter(pl.col("team") == picked)
part = ui.participation(view).filter(pl.col("team") == picked)

st.header(f"{picked} · {view.season}")
head = st.columns(6)
head[0].metric("Points / game", f"{env['points'].mean():.2f}")
head[1].metric("Plays / game", f"{env['plays'].mean():.1f}")
head[2].metric("Pass attempts / game", f"{env['pass_attempts'].mean():.1f}")
head[3].metric("Carries / game", f"{env['carries'].mean():.1f}")
head[4].metric("Offensive TDs / game", f"{env['offensive_tds'].mean():.2f}")
head[5].metric("Games with a posted line", f"{int(env['has_market'].sum())} / {env.height}")

# --------------------------------------------------------------------------- #
# stage 1: the season shape
# --------------------------------------------------------------------------- #
st.subheader("Season shape")
ui.note(
    "Stage one: what kind of offence and defence this is over a season, before any particular game. "
    "Every per-game number below it is this, times a context factor."
)
side = st.radio("Side", ["offense", "defense"], horizontal=True, label_visibility="collapsed")
prefix = "off_" if side == "offense" else "def_"
metrics = [c for c in shape.columns if c.startswith(prefix)]
long = (
    shape.select("team", *metrics)
    .unpivot(index="team", variable_name="metric", value_name="value")
    .with_columns(pl.col("metric").str.strip_prefix(prefix))
    .with_columns(
        pl.col("value").rank("min", descending=True).over("metric").cast(pl.Int32).alias("rank_high"),
        pl.col("value").mean().over("metric").alias("league_mean"),
    )
    .filter(pl.col("team") == picked)
    .with_columns((pl.col("value") / pl.col("league_mean")).alias("vs_league"))
    .select("metric", "value", "league_mean", "vs_league", "rank_high")
    .sort("metric")
)
ui.table(long, config={**ui.fixed(3, "value", "league_mean"), **ui.fixed(2, "vs_league")},
         height=460)
ui.note(
    "`rank_high` is 1 for the largest value in the league, whether or not large is good — a defence "
    "ranked 1 in `points_allowed_pg` is the worst one. `vs_league` is the ratio the per-game chain "
    "actually multiplies by when this team is the opponent."
)

# --------------------------------------------------------------------------- #
# stage 2: the seventeen games
# --------------------------------------------------------------------------- #
st.subheader("The seventeen games")
cols = [c for c in ("week", "opponent", "is_home", "rest_days", "div_game", "roof", "temp", "wind",
                    "spread", "total", "implied_points", "has_market", "points", "plays",
                    "dropbacks", "pass_attempts", "carries", "targets", "air_yards", "pass_tds",
                    "rush_tds", "red_zone_trips", "yards_per_attempt", "yards_per_carry")
        if c in env.columns]
st.bar_chart(env.select("week", "implied_points"), x="week", y="implied_points", height=220)
ui.table(
    env.select(cols),
    config={**ui.fixed(1, "spread", "total", "implied_points", "points", "plays", "dropbacks",
                       "pass_attempts", "carries", "targets", "air_yards", "temp", "wind"),
            **ui.fixed(2, "pass_tds", "rush_tds", "red_zone_trips", "yards_per_attempt",
                       "yards_per_carry")},
    height=640,
)
ui.note(
    "`implied_points` is the posted line where there is one and the model's own estimate where there "
    "is not, put on one scale by a shrunk per-team offset. `has_market` false means the number is "
    "entirely ours."
)

st.markdown("**Edit these games**")
ui.note(
    "Any week, individually. `set` a number here and the players divide the edited pool rather than "
    "the projected one — edit `dropbacks` and the designed-run count follows it, so pace and mix "
    "cannot be left contradicting each other."
)
ENV_EDIT = tuple(c for c in overrides.TEAM_FIELDS if c in env.columns)
ui.grid(
    env.select("team", "week", "opponent", *ENV_EDIT),
    level="team", key_col="team", week_col="week", fields=ENV_EDIT,
    key=f"env:{picked}", digits=2, height=640, hide=("team",),
)

with st.expander("The context factors behind those games"):
    fcols = [c for c in env.columns if c.startswith("f_")]
    bcols = [c for c in env.columns if c.startswith("b_")]
    ui.table(env.select("week", "opponent", *bcols), height=420)
    ui.table(env.select("week", "opponent", *fcols), digits=3, height=420)
    ui.note(
        "`b_*` are the bucket a game falls in for each split; `f_*` is the product of that game's "
        "factors for one metric. A factor at 0.80 or 1.25 is at the clip, which means the game "
        "environment is being trusted further than the split behind it supports."
    )

# --------------------------------------------------------------------------- #
# the roster
# --------------------------------------------------------------------------- #
st.subheader("The roster dividing it")
shares = ui.shares_wide(view)
rates = ui.rates_wide(view)

INPUTS = {
    "QB": ("dropback_share", "pass_td_share", "designed_rush_share", "qb_rush_td_share",
           "attempt_rate", "completion_pct", "yards_per_attempt", "pass_td_rate", "int_rate",
           "sack_rate", "scramble_rate", "yards_per_clean_rush"),
    "RB": ("carry_share", "clean_rush_share", "rz_carry_share", "inside_5_carry_share",
           "short_yardage_carry_share", "target_share", "rush_td_share", "rec_td_share",
           "yards_per_carry", "rush_success_rate", "catch_rate", "yards_per_target"),
    "WR": ("target_share", "air_yards_share", "rz_target_share", "late_down_target_share",
           "rec_td_share", "catch_rate", "yards_per_target", "adot", "tprr"),
    "TE": ("target_share", "air_yards_share", "rz_target_share", "rec_td_share", "catch_rate",
           "yards_per_target", "adot", "tprr"),
}

# `active_weeks` is shown but not editable: it is `expected_games` over 17, and the two are kept in
# agreement by deriving one from the other rather than by trusting a user to edit both.
OUTPUTS = {
    "QB": ("attempts", "completions", "passing_yards", "passing_tds", "interceptions",
           "rushing_yards", "rushing_tds"),
    "RB": ("carries", "rushing_yards", "rushing_tds", "targets", "receptions", "receiving_yards",
           "receiving_tds"),
    "WR": ("targets", "receptions", "receiving_yards", "receiving_tds", "carries"),
    "TE": ("targets", "receptions", "receiving_yards", "receiving_tds", "routes"),
}

tabs = st.tabs(list(INPUTS))
for tab, pos in zip(tabs, INPUTS, strict=True):
    with tab:
        men = part.filter(pl.col("position") == pos).select(
            "player_id", "player", "depth_slot", "slot_bucket", "status", "is_rookie",
            "expected_games", "active_weeks", "snap_share", "route_participation",
        )
        if men.is_empty():
            st.info(f"No {pos} on the {picked} roster.")
            continue
        want = [c for c in INPUTS[pos] if c in shares.columns or c in rates.columns]
        inp = men.join(
            shares.select("player_id", *[c for c in want if c in shares.columns]),
            on="player_id", how="left",
        ).join(
            rates.select("player_id", *[c for c in want if c in rates.columns]),
            on="player_id", how="left",
        ).sort("depth_slot")

        got = board.filter(pl.col("position") == pos).select(
            "player_id", "player", "position_rank", "games",
            *[c for c in OUTPUTS[pos] if c in board.columns], "fantasy_points", "points_per_game",
        ).sort("fantasy_points", descending=True)

        left, right = st.columns(2)
        with left:
            st.caption("Inputs — editable: what the estimator believes about the player")
            ui.grid(
                inp, level="player", key_col="player_id",
                fields=tuple(c for c in inp.columns if c in overrides.PLAYER_FIELDS),
                key=f"inputs:{picked}:{pos}", digits=3, height=420, hide=("player_id",),
                config=ui.fixed(2, "expected_games"),
            )
        with right:
            st.caption("Projections — what those beliefs produce against this team's games")
            ui.table(
                got.drop("player_id"),
                config={**ui.fixed(1, *[c for c in OUTPUTS[pos] if c in got.columns],
                                   "fantasy_points", "games"),
                        **ui.fixed(2, "points_per_game")},
                height=420,
            )

# --------------------------------------------------------------------------- #
# what the edits did
# --------------------------------------------------------------------------- #
mine_ids = set(part["player_id"].to_list())
touching = [o for o in ui.live().items
            if (o.level == "team" and o.key == picked) or (o.level == "player" and o.key in mine_ids)]
if touching:
    st.subheader("What the edits did")
    cols = st.columns([5, 1])
    cols[0].markdown(" · ".join(f"`{o.label}`" for o in touching))
    if cols[1].button("Reset this team"):
        for o in touching:
            ui.reset(level=o.level, key=o.key, field_name=o.field, week=o.week)
        st.rerun()

    moved = ui.board_diff(view).filter(
        pl.col("team").eq(picked) | pl.col("new_team").eq(picked)
    )
    if moved.is_empty():
        st.info("Every edit applied, and none of them moved a projected point on this roster.")
    else:
        ui.table(
            moved.select("player", "position", "fantasy_points", "new_fantasy_points",
                         "d_fantasy_points", "d_targets", "d_carries", "d_receiving_yards",
                         "d_rushing_yards", "position_rank", "new_position_rank"),
            config={**ui.fixed(1, "fantasy_points", "new_fantasy_points", "d_fantasy_points",
                               "d_receiving_yards", "d_rushing_yards"),
                    **ui.fixed(2, "d_targets", "d_carries")},
            height=380,
        )
        ui.note(
            "Against the same season with no edits. The teammates are in here on purpose: with pool "
            "normalisation on, a share given to one player is taken from the others, so the column "
            "that matters is whether the losses look like the ones you meant to cause."
        )
    with st.expander("Where each edit landed"):
        ui.table(
            ui.provenance(view).filter(
                (pl.col("level") == "team") & (pl.col("key") == picked)
                | pl.col("key").is_in(list(mine_ids))
            ).select("stage", "level", "key", "field", "week", "mode", "value", "base_recorded",
                     "base_now", "used_now", "rows", "applied", "reason"),
            digits=4, height=320,
        )
        ui.note(
            "`base_now` is what the engine had before the edit and `used_now` what it used after. "
            "`applied` false with a reason is an edit that found nothing to change — a player who has "
            "left, or a column that no longer exists — and it is reported rather than dropped."
        )

# --------------------------------------------------------------------------- #
# does it add up
# --------------------------------------------------------------------------- #
st.subheader("Does it add up")
mine = ui.team_pools(view).filter((pl.col("team") == picked) & pl.col("exclusive")).drop("team")
ui.table(
    mine, digits=3,
    config={**ui.fixed(3, "measured_target", "raw_sum", "factor"), **ui.fixed(1, "gap_pct")},
)
ui.note(
    "What this roster's shares summed to *before* scaling, against what the pool was measured to sum "
    "to. A negative `gap_pct` is a team with work nobody on the depth chart has claimed — usually a "
    "vacated role — and `factor` is what normalisation multiplied every share by to close it. With "
    "the sidebar toggle off, the gap is left in place and this team projects that much short of its "
    "own offence."
)

worst = ui.team_pools(view).filter(pl.col("exclusive"))
with st.expander("Every team, ranked by how far its shares were off"):
    pool_pick = st.selectbox("Pool", sorted(worst["pool"].unique().to_list()),
                             index=sorted(worst["pool"].unique().to_list()).index("targets"))
    ui.table(
        worst.filter(pl.col("pool") == pool_pick)
        .select("team", "raw_sum", "factor", "gap_pct")
        .sort("gap_pct"),
        config={**ui.fixed(3, "raw_sum", "factor"), **ui.fixed(1, "gap_pct")},
        height=460,
    )

with st.expander("League-wide pool audit"):
    ui.table(
        ui.pool_report(view).drop("team_col"), digits=3,
        config={**ui.fixed(3, "measured_target", "raw_mean", "raw_min", "raw_max", "after_mean"),
                **ui.fixed(2, "count_per_game"), **ui.fixed(1, "gap_pct")},
        height=560,
    )
    ui.note(
        "`exclusive` false means the shares are not dividing the pool and are never scaled: five "
        "players are on the field for one snap. `measured_target` is what the pool was observed to "
        "sum to over 2016-2025 rather than an assumed 1.0."
    )

with st.expander("Players summed back against the team they came from"):
    ui.table(
        ui.reconciliation(view).sort("from_pool", descending=True),
        config={**ui.fixed(2, "players_per_game", "team_per_game", "gap_pct"),
                **ui.fixed(3, "worst_abs_gap")},
    )
    ui.note(
        "`from_pool` true means the stat is drawn straight from a normalised pool and must match to "
        "rounding. Yards and completions are a count times a rate and are deliberately not forced: a "
        "roster more efficient than the team's recent offence should project above it, and scaling "
        "that away would hide the disagreement rather than show it."
    )
