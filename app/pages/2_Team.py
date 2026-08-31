"""One team: the offence it is projected to run, and the roster that has to add up to it.

The page is laid out in the order the engine multiplies, because that is also the order somebody
disagrees in:

1. **Depth chart** — who is ahead of whom. A slot is not a number applied to a projection, it is the row
   every prior is read off, so it is the first thing to be right and the first thing to edit.
2. **Availability** — how much of the season each of them is there for.
3. **The rooms** — one man at a time: pick him out of his room and every estimate behind him is drawn as
   a meter with the ✎ that changes it, with the room's sheet still there for typing over a dozen at once.
4. **Team volume** — how much there is to divide, per game: one week at a time, or a multiplier across
   all seventeen.
5. **What if** — the same numbers, moved one at a time with the consequence on screen: who is dividing
   each of the team's pools, then one candidate edit run through the whole engine before it is written
   down, reported as what it did to him, to his teammates and to the offence's totals.
6. **Does it add up** — the audits that say whether the division stayed coherent, and what the edits on
   this roster moved.

Every team knob there is lives on this page, and the Edits page is the ledger of what has been turned
rather than a second set of controls. What the edits did to this roster is under *does it add up*,
because "is the division still coherent" and "did it do what I meant" are the same question twice.

The rooms tab is the one that changed shape. It used to be a 915-row sheet of thirty-nine editable
columns per position, which is the fast way to work and an impossible way to *read*: nothing on screen
said which of a man's numbers was a measurement and which was an assumption about his slot. Now the room
is a short list, the man is a panel of meters beside it, and the sheet is one click away for when a dozen
cells need typing at once. Every explanation on the page is behind the ⓘ next to the thing it explains.
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
# Keyed, so the league page can hand this page a team: a record that looks wrong there is one click from
# the roster that produced it. A stale key from a scenario with different teams would raise, so it is
# checked against the options rather than trusted.
if st.session_state.get(ui.FOCUS_TEAM) not in teams:
    st.session_state.pop(ui.FOCUS_TEAM, None)
picked = st.selectbox("Team", teams, index=0, key=ui.FOCUS_TEAM)
# Going through the league is thirty-two of these sittings, so the way to the next one is a button rather
# than the dropdown, and one of the buttons knows which teams are still untouched.
ui.team_nav(view, teams, picked)

env = ui.environment(view).filter(pl.col("team") == picked).sort("week")
board = ui.board(view).filter(pl.col("team") == picked)
part = ui.participation(view).filter(pl.col("team") == picked)
chart = ui.depth_frame(view, picked)

st.header(f"{picked} · {view.season}")
ui.tiles([
    {"name": "points / game", "value": env["points"].mean(), "digits": 2, "highlight": True,
     "sub": f"{env['points'].sum():,.0f} over the season"},
    {"name": "plays / game", "value": env["plays"].mean()},
    {"name": "pass attempts / game", "value": env["pass_attempts"].mean()},
    {"name": "carries / game", "value": env["carries"].mean()},
    {"name": "offensive TDs / game", "value": env["offensive_tds"].mean(), "digits": 2},
    {"name": "games with a line", "value": int(env["has_market"].sum()), "digits": 0,
     "sub": f"of {env.height}"},
])

# `mine_ids` is needed in two places, so it is computed once out here. The released men have to be added
# back into it from the baseline: they are this team's edits and they are the only ones with no row in the
# run, so read off `part` alone they would be missing from "your edits here" and would survive a reset
# that says it drops them.
mine_ids = set(part["player_id"].to_list()) | set(ui.released_frame(view, picked)["player_id"].to_list())
touching = [o for o in ui.live().items
            if (o.level == "team" and o.key == picked) or (o.level == "player" and o.key in mine_ids)]

# What has already been done to this roster, above the tabs rather than inside one of them: it is the
# frame the six tabs are read in — a share that looks wrong is a different thing when you are the one who
# typed it — and it used to be a seventh tab nobody opened before disagreeing with a number.
if touching:
    with st.container(border=True):
        row = st.columns([6, 2, 2])
        row[0].markdown("**Your edits here** · "
                        + " · ".join(f"`{o.label}`" for o in touching))
        if row[1].button("↺ Reset this team", width="stretch",
                         help="drops these edits only; the rest of the scenario is kept"):
            for o in touching:
                ui.drop_edit(o.level, o.key, o.field, o.week)
            st.rerun()
        with row[2]:
            ui.page_link("pages/7_Edits.py", "Every edit →", "✏️")

# The tabs are the engine's own order, and the labels are questions rather than nouns so the reader
# knows which one to open.
TABS = ["🧭 Depth chart", "🚑 Availability", "🎛️ The rooms", "📈 Team volume", "🔬 What if",
        "🧮 Does it add up"]
depth_tab, avail_tab, rooms_tab, volume_tab, whatif_tab, audit_tab = st.tabs(TABS)

# --------------------------------------------------------------------------- #
# 1. the depth chart
# --------------------------------------------------------------------------- #
with depth_tab:
    ui.section(
        "Roster order",
        "The bar is each man's share of the pool his position is judged on — dropbacks for a "
        "quarterback, carries for a back, targets for a receiver — and the line under it is his "
        "projected season. ✏️ marks a player you have overridden.",
    )
    deep = st.slider("Players shown per position", 3, 12, 6,
                     help="How far down each room to draw. The editor below has all of them.")
    ui.depth_board(chart, limit=int(deep))

    st.divider()
    ui.section(
        "Move somebody",
        "The chart is an input, not a fact: it is the published August order with the estimator's own "
        "guess for anybody it left out, and it is the row every prior is read off. Move a man and he is "
        "**re-priced as what he was moved to** — his slot's expected games, his slot's survival rate, "
        "his place in the quarterback queue — rather than being handed a starter's share while still "
        "priced as a backup.",
    )
    ui.depth_editor(view, chart, picked, key="depth")

    st.divider()
    ui.section(
        "Take somebody off this roster",
        "Our sources are a published depth chart and a cumulative roster file, and in the days after "
        "cutdown day both still list men who are not on this team any more. Removing one here is not the "
        "same as saying he will play no games: he is taken off the roster **before** anything is "
        "estimated from it, so the room closes up behind him, the men who moved up are re-priced as what "
        "they moved up to, and his share of every pool goes to the players who are left. Nothing after "
        "that — board, weeks, exports, simulations — has a row to count him in.",
    )
    ui.release_editor(view, chart, picked, key="release")

    DEPTH_LIST = ["position", "depth_slot", "player", "status", "pool_share", "expected_games",
                  "position_rank", "fantasy_points", "edited"]
    marked = ui.with_edits(chart, "player", "player_id")
    ui.focus_table(
        marked.select([c for c in (
            "position", "depth_slot", "slot_bucket", "player", "status", "is_rookie", "draft_pick",
            "years_exp", "age", "charted", "team_disagreement", "alignment", "pool_share", "snap_share",
            "route_participation", "expected_games", "games", "position_rank", "fantasy_points",
            "points_per_game", "edited", "edits") if c in marked.columns]),
        DEPTH_LIST, key=f"chart:{picked}", height=560,
        config={**ui.percent("pool_share", "snap_share", "route_participation"),
                **ui.fixed(1, "expected_games", "games", "fantasy_points"),
                **ui.fixed(2, "points_per_game")},
        label_text="the roster facts behind it",
        note_text=(
            "`slot_bucket` is the slot the *priors* are keyed on, capped per position — a fourth "
            "quarterback and a third are priced off the same row, because there is not enough history "
            "of a QB4 to price him separately. `charted` false is a player the published chart does not "
            "list and the estimator has placed itself; `team_disagreement` is the two sources putting "
            "him in different slots. Both are reasons to check a slot rather than defects — but they "
            "are the slots worth checking."
        ),
    )

# --------------------------------------------------------------------------- #
# 2. availability
# --------------------------------------------------------------------------- #
with avail_tab:
    ui.section(
        "How much of the season each of them is there for",
        "`expected_games` is the one availability knob and the most argued-with number in the "
        "projection. It is the fastest way to run a holdout or a long injury: set a starter to 8 and his "
        "share of every pool is halved before normalisation hands the rest to the room behind him. "
        "Type in `expected_games`; every other column is the record behind it — `engine` is what the "
        "model said, and the season columns are the games he actually played.",
    )
    ui.rank_bars(chart.sort("expected_games", descending=True).head(16), "expected_games",
                 height=420, digits=1)

    keep = [c for c in ("player_id", "position", "depth_slot", "player", "status", "expected_games",
                        "games", "fantasy_points") if c in chart.columns]
    avail_rows = ui.with_evidence(view, chart.select(keep), ui.GAMES_METRIC)
    with st.expander("Type over them — the whole roster as a sheet, with the record beside the knob"):
        ui.grid(
            avail_rows, level="player", key_col="player_id", fields=("expected_games",),
            key=f"avail:{picked}", digits=2, height=560, hide=("player_id",),
            config={**ui.fixed(1, "games", "fantasy_points"),
                    **ui.evidence_config(ui.GAMES_METRIC, digits=1)},
        )
    ui.note("This is one team's roster. For the whole league on one surface — everybody the projection "
            "is not treating as a full season, his injury record beside the knob, and a *back in week N* "
            "control that converts a return date into the games it is worth — use the **Availability** "
            "page.")

    st.divider()
    ui.section(
        "Set a whole room at once",
        "The healthy-starter case, in one click per man: unless somebody is hurt or holding out, a "
        "starter plays about every week, and the estimator is deliberately conservative about that "
        "because it is shrinking him towards the average attendance of everyone who has held his slot.",
    )
    who = st.selectbox("Position", ["QB", "RB", "WR", "TE"], key=f"avail:room:{picked}")
    room_ids = tuple(chart.filter((pl.col("position") == who) & (pl.col("depth_slot") <= 3))
                     ["player_id"].to_list())
    ui.room_panel(view, ui.GAMES_METRIC, room_ids, key=f"avail:{picked}:{who}", position=who)

# --------------------------------------------------------------------------- #
# 3. the rooms — one man at a time, with the sheet behind him
# --------------------------------------------------------------------------- #
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

# The one number a room is really argued about, and therefore the evidence the sheet opens with.
HEADLINE = {"QB": "dropback_share", "RB": "carry_share", "WR": "target_share", "TE": "target_share"}

# The room list: enough to choose a man by, and no more. His numbers are the panel beside it.
ROOM_LIST = ["depth_slot", "player", "status", "expected_games", "pool_share", "position_rank",
             "fantasy_points", "edited"]

with rooms_tab:
    ui.section(
        "One man at a time",
        "Pick a name on the left and every estimate behind him is drawn on the right: the fill is what "
        "the projection ran on, the ticks are his own record and what his slot is worth to everybody who "
        "has held it, and the footer says how much of his own play is carrying it. A fill on the left "
        "tick is a measurement; a fill on the right is an assumption, and the assumptions sort to the "
        "top because they are the ones worth arguing with. ✎ changes any of them where it stands.",
    )
    pos = st.segmented_control("Room", list(INPUTS), default="WR", key=f"room:pos:{picked}",
                              format_func=lambda p: f"{p} · "
                              f"{part.filter(pl.col('position') == p).height}") or "WR"

    shares = ui.shares_wide(view)
    rates = ui.rates_wide(view)
    men = part.filter(pl.col("position") == pos)
    room = board.filter(pl.col("position") == pos)

    if men.is_empty() or room.is_empty():
        st.info(f"No {pos} on the {picked} roster.")
    else:
        # the list is the board's own numbers for the room, with the slot and the pool share it is judged
        # on joined on, so choosing a man and reading him are the same frame
        listing = room.join(
            chart.select("player_id", "depth_slot", "slot_bucket", "pool_share", "expected_games"),
            on="player_id", how="left",
        ).sort("fantasy_points", descending=True)
        listing = ui.with_edits(listing, "player", "player_id")

        left, right = st.columns([4, 8], gap="medium")
        with left:
            his = ui.pick_from(
                listing, key=f"room:list:{picked}:{pos}", columns=ROOM_LIST, height=560,
                config={**ui.percent("pool_share"), **ui.fixed(1, "fantasy_points", "expected_games"),
                        "edited": st.column_config.CheckboxColumn("✏️", width="small"),
                        "depth_slot": st.column_config.NumberColumn("slot", format="%d",
                                                                    width="small"),
                        "position_rank": st.column_config.NumberColumn("pos #", format="%d",
                                                                       width="small")},
            )
        with right:
            if his is None:
                st.info("Pick a name.")
            else:
                with st.container(border=True):
                    ui.player_panel(view, his, key=f"room:panel:{picked}:{pos}", history=False)

        st.divider()
        want = [c for c in INPUTS[pos] if c in shares.columns or c in rates.columns]
        inp = men.select(
            "player_id", "player", "depth_slot", "slot_bucket", "status", "is_rookie",
            "expected_games", "active_weeks", "snap_share", "route_participation",
        ).join(
            shares.select("player_id", *[c for c in want if c in shares.columns]),
            on="player_id", how="left",
        ).join(
            rates.select("player_id", *[c for c in want if c in rates.columns]),
            on="player_id", how="left",
        ).sort("depth_slot")

        with st.expander(f"The whole {pos} room as a sheet — a dozen numbers, typed straight in"):
            ui.note(
                "The fast surface, kept because a meter at a time is the wrong tool for retyping a room: "
                "left is what the estimator believes about the player and is editable in place. "
                "`active_weeks` is shown but locked — it is `expected_games` over 17, and the two are "
                "kept in agreement by deriving one from the other rather than by trusting anybody to "
                "edit both."
            )
            marked = ui.with_edits(inp, "player", "player_id")
            # The sheet is where a number gets typed, so the record behind one of them belongs *in* it
            # rather than a scroll away: `engine` is what the estimator said before anybody touched it,
            # `obs`/`n` are his own record and how much of it there is, `prior` is what the job is worth,
            # and the year columns are the five seasons as a line. One metric at a time, because five
            # evidence columns for a dozen metrics is a sheet nobody can read.
            choices = ["nothing", *(c for c in want if c in overrides.PLAYER_FIELDS)]
            first = HEADLINE.get(pos, "")
            beside = st.selectbox(
                "Show the evidence for", choices,
                index=choices.index(first) if first in choices else 0,
                format_func=lambda c: "nothing" if c == "nothing" else ui.label(c),
                key=f"inputs:evidence:{picked}:{pos}",
                help="joins one metric's record onto the sheet: the estimator's own answer, the size of "
                     "the player's sample, what his slot is worth, and the last five seasons as a trend",
            )
            shown = marked if beside == "nothing" else ui.with_evidence(view, marked, beside)
            ui.grid(
                shown, level="player", key_col="player_id",
                fields=tuple(c for c in marked.columns if c in overrides.PLAYER_FIELDS),
                key=f"inputs:{picked}:{pos}", digits=3, height=420,
                hide=("player_id",) + (() if marked["edited"].any() else ("edited", "edits")),
                config={**ui.fixed(2, "expected_games"),
                        **({} if beside == "nothing" else ui.evidence_config(beside))},
            )
            if beside != "nothing":
                ui.note(
                    f"`{beside}` is the editable column and `engine` beside it is what the estimator "
                    "said, so the two disagree exactly where somebody has typed. A low `n` with a high "
                    "`own_weight` is thin evidence being trusted; a flat trend across five seasons is a "
                    "number that has been true for years, and worth more argument to move."
                )

        # The sums belong under the sheet, because a share is a claim on a pool and a dozen defensible
        # claims can add up to something impossible. What the engine does about it depends on the pool.
        sums = ui.pool_sums(view, picked, want, tuple(inp["player_id"].to_list()))
        if not sums.is_empty():
            with st.expander(f"What the {pos} sheet adds up to — every pool those shares claim"):
                ui.table(
                    sums, digits=3, height=min(80 + 35 * sums.height, 420), auto=False,
                    config={"room": st.column_config.NumberColumn(f"this {pos} room", format="%.3f"),
                            "roster": st.column_config.NumberColumn("whole roster", format="%.3f"),
                            "as run": st.column_config.NumberColumn("as run", format="%.3f"),
                            "there to divide": st.column_config.NumberColumn("there to divide",
                                                                             format="%.3f"),
                            "over %": st.column_config.NumberColumn("over by", format="%+.1f%%")},
                )
                ui.note(
                    "`this room` and `whole roster` are the shares exactly as typed, and neither is "
                    "supposed to come to 1.000: a share is what a man takes **while he is playing**, and "
                    "a roster of sixty is not all playing. **`as run`** is the same roster with each "
                    "man's availability in it, which is the only one of the three comparable with "
                    "`there to divide` — and it is the number the engine sees. Over the target and every "
                    "claimant is scaled down proportionally, so raising one man here quietly lowers "
                    "everybody, which is what the *what if* tab is for. Well under it is a real finding "
                    "about the depth chart rather than a rounding problem: a role nobody has inherited. "
                    "`there to divide` is measured from played seasons, not assumed to be 1.000."
                )

        line = [c for c in ui.stat_columns(pos) if c in board.columns]
        with st.expander("The football line those beliefs produce, for the whole room"):
            ui.table(
                room.select("player", "position_rank", "games", *line, "fantasy_points",
                            "points_per_game").sort("fantasy_points", descending=True),
                config={**ui.stat_config(line), **ui.fixed(1, "fantasy_points", "games"),
                        **ui.fixed(2, "points_per_game")},
                height=420,
            )

        # the check on a number across the whole room: one metric, everybody on it, with the record and
        # the league behind each -- the comparison a per-player panel cannot make
        editable = [c for c in want if c in overrides.PLAYER_FIELDS]
        if editable:
            ui.section("The same number across the room",
                       "A share is zero-sum: raising his lowers theirs, and the man to check is whoever "
                       "the estimator is least sure about. This is one metric for everybody in the room "
                       "with his own record, his sample and his slot's average beside it.")
            focus = st.selectbox("Metric", editable, key=f"focus:{picked}:{pos}",
                                 format_func=ui.label)
            ui.room_panel(view, focus, tuple(inp["player_id"].to_list()),
                          key=f"room:{picked}:{pos}", position=pos)

# --------------------------------------------------------------------------- #
# 4. team volume
# --------------------------------------------------------------------------- #
# Both scopes of a team edit are here, because they are the same edit at two sizes and having them on two
# pages is how one gets typed twice. A `set` on one week is the schedule kept and one game moved; a
# multiplier over all seventeen is a claim about the offence itself.
ENV_EDIT = tuple(c for c in overrides.TEAM_FIELDS if c in env.columns)
SCOPES = ["one week at a time", "every week at once"]

with volume_tab:
    ui.section(
        "How much there is to divide",
        "What this offence has run per game, by season, then what it is projected to run and what the "
        "league is projected to run. Nobody knows from memory whether 34 dropbacks a game is a lot; "
        "these three rows are the other half of knowing. A team edit that puts an offence outside "
        "everything it has ever run is a claim about a new coordinator, not a fix.",
    )
    ui.table(ui.team_history(view, picked, ENV_EDIT), digits=2, height=260)

    st.divider()
    ui.section(
        "The seventeen games",
        "`implied_points` is the posted line where there is one and the model's own estimate where "
        "there is not, put on one scale by a shrunk per-team offset.",
    )
    ui.week_bars(env, y="implied_points", height=240)

    scope = st.radio("Change it", SCOPES, horizontal=True, key=f"env:scope:{picked}")
    if scope == SCOPES[0]:
        ui.note(
            "Type in any cell to set that week alone — edit `dropbacks` and the designed-run count "
            "follows it, so pace and mix cannot be left contradicting each other. The other sixteen "
            "games stay where the model had them."
        )
        ui.grid(
            env.select("team", "week", "opponent", *ENV_EDIT),
            level="team", key_col="team", week_col="week", fields=ENV_EDIT,
            key=f"env:{picked}", digits=2, height=560, hide=("team",),
        )
    else:
        ui.note(
            "A multiplier applied to all seventeen games at once, with `base` this team's season mean. "
            "`set` is in the other scope only, because setting the same absolute number in every game "
            "would flatten the schedule into a season average and throw away the per-game chain."
        )
        for f in ENV_EDIT:
            ui.knob("team", picked, f, base=float(env[f].mean()), fmt="%.3f",
                    widget_key=f"adj:allweeks:{picked}:{f}", modes=("multiply",))

    with st.expander("Season shape: what kind of offence and defence this is, before any one game"):
        ui.note("Every per-game number above is this, times a context factor.")
        side = st.radio("Side", ["offense", "defense"], horizontal=True)
        prefix = "off_" if side == "offense" else "def_"
        metrics = [c for c in shape.columns if c.startswith(prefix)]
        long = (
            shape.select("team", *metrics)
            .unpivot(index="team", variable_name="metric", value_name="value")
            .with_columns(pl.col("metric").str.strip_prefix(prefix))
            .with_columns(
                pl.col("value").rank("min", descending=True).over("metric").cast(pl.Int32)
                .alias("rank_high"),
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
            "`rank_high` is 1 for the largest value in the league, whether or not large is good — a "
            "defence ranked 1 in `points_allowed_pg` is the worst one. `vs_league` is the ratio the "
            "per-game chain actually multiplies by when this team is the opponent."
        )

    with st.expander("The per-game detail behind those seventeen"):
        cols = [c for c in ("week", "opponent", "is_home", "rest_days", "div_game", "roof", "temp",
                            "wind", "spread", "total", "implied_points", "has_market", "points",
                            "plays", "dropbacks", "pass_attempts", "carries", "targets", "air_yards",
                            "pass_tds", "rush_tds", "red_zone_trips", "yards_per_attempt",
                            "yards_per_carry") if c in env.columns]
        ui.focus_table(
            env.select(cols),
            ["week", "opponent", "is_home", "implied_points", "spread", "total", "plays", "dropbacks",
             "carries"], key=f"env:detail:{picked}", height=560,
            config={**ui.fixed(1, "spread", "total", "implied_points", "points", "plays", "dropbacks",
                               "pass_attempts", "carries", "targets", "air_yards", "temp", "wind"),
                    **ui.fixed(2, "pass_tds", "rush_tds", "red_zone_trips", "yards_per_attempt",
                               "yards_per_carry")},
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
# 5. what if
# --------------------------------------------------------------------------- #
# The tab the other five needed: a grid of shares can be typed into but it cannot show what typing did,
# and what typing did is the whole question. Every count in the projection is a slice of a team pool, so
# the surface reads top to bottom as who is dividing this offence, which of his numbers is worth moving,
# and then one candidate edit run through the engine before it is written down.
CHART_LIMIT = 14                       # bars a reader can actually compare

with whatif_tab:
    ui.section(
        "Who is producing this offence",
        "Pick one of the team's own numbers and this is the roster dividing it: what each man is "
        "projected for, what fraction of the team that is, and what it is worth to him. Every count in "
        "the projection is a slice of one of these pools, which is why moving one man moves the men "
        "around him.",
    )
    pool = st.selectbox(
        "Team stat", [p for p, _ in ui.CONTRIBUTION_POOLS],
        format_func=lambda p: ui.POOL_NAME.get(p, ui.label(p)), key=f"whatif:pool:{picked}",
    )
    facts = ui.pool_facts(view, picked, pool)
    con = ui.contributions(view, picked, pool)
    pool_name = ui.POOL_NAME.get(pool, ui.label(pool))

    if not facts or con.is_empty():
        st.info(f"Nobody on the {picked} roster is claiming any {pool_name.lower()}.")
    else:
        ui.tiles([
            {"name": f"{pool_name.lower()} / game", "value": facts["team_per_game"], "highlight": True,
             "sub": f"{facts['team_season']:,.0f} over the season"},
            {"name": "men claiming it", "value": facts["players"], "digits": 0},
            {"name": "claimed before scaling", "value": facts.get("raw_sum"), "digits": 3,
             "percent": True,
             "sub": "over 100% is a room where somebody is high"},
            {"name": "scaled by", "value": facts.get("factor"), "digits": 3,
             "sub": "to make the roster sum to the pool"},
        ])
        if not facts["exclusive"]:
            ui.note("Several men are on the field for the same snap, so these shares are **not** "
                    "dividing anything and are never scaled: they can sum to four and be right.")
        elif facts["queue"]:
            ui.note("This pool goes to one man at a time, so it is filled **in depth order** rather "
                    "than scaled: the starter is filled to his claim and the man behind him gets what "
                    "is left over.")

        ui.rank_bars(con.head(CHART_LIMIT), "projected", height=max(220, 26 * min(con.height,
                                                                                 CHART_LIMIT) + 40))
        if con.height > CHART_LIMIT:
            st.caption(f"The {CHART_LIMIT} biggest claims; the table has all {con.height}.")
        ui.focus_table(
            con.select("player", "position", "depth_slot", "status", "expected_games", "share",
                       "of_team", "projected", "fantasy_points", "points_per_game", "position_rank",
                       "edited", "edits"),
            ["player", "depth_slot", "share", "of_team", "projected", "fantasy_points", "edited"],
            key=f"whatif:con:{picked}:{pool}",
            config={"projected": st.column_config.NumberColumn(f"projected {pool_name.lower()}",
                                                               format="%.1f"),
                    **ui.bar("of_team", maximum=float(con["of_team"].max() or 1.0))},
            height=min(80 + 35 * con.height, 420),
            note_text=(
                "`share of the team` is his projected count over the team's, which is the number worth "
                "arguing about — it already has his availability in it, so a starter projected for nine "
                "games is not shown producing a starter's season. ✏️ marks a man you have already edited."
            ),
        )

    # ---- one candidate edit, run through the engine ------------------------ #
    st.divider()
    ui.section(
        "Move one number and watch what it does",
        "One edit, run through the whole season **before it is written down**: his line, then everybody "
        "on this roster whose line moved because of it, then the offence's own totals. Nothing is "
        "committed until you apply it, and moving the control back to 100% throws it away.",
    )

    seats = chart.select("player_id", "player", "position", "depth_slot").sort(
        ["position", "depth_slot"])
    seat_ids = seats["player_id"].to_list()
    seat_names = {r["player_id"]: f"{r['player']} · {r['position']}{r['depth_slot']}"
                  for r in seats.rows(named=True)}
    lead = None if con.is_empty() else con["player_id"][0]
    start = seat_ids.index(lead) if lead in seat_ids else 0

    if not seat_ids:
        st.info(f"No roster for {picked}.")
    else:
        top = st.columns([3, 3])
        pid = top[0].selectbox("Player", seat_ids, index=start,
                               format_func=lambda i: seat_names.get(i, i),
                               key=f"whatif:who:{picked}")
        his_position = seats.filter(pl.col("player_id") == pid)["position"][0]
        options = [f for f in ui.knobs_for(his_position) if f in overrides.PLAYER_FIELDS]
        pool_metrics = [m for m in facts.get("shares", ()) if m in options]
        field = top[1].selectbox("Number", options, index=options.index(pool_metrics[0])
                                 if pool_metrics else 0, format_func=ui.knob_option,
                                 key=f"whatif:field:{picked}:{his_position}")
        kind = ui.knob_kind(field)
        st.caption(f"**{ui.label(field)}** — {ui.KNOB_HELP.get(field, '')}  \n"
                   f"{ui.KIND_ICON.get(kind, '·')} {ui.KIND_NOTE[kind]}")

        engine, running, _ = ui.candidate_numbers(view, pid, field, "set", 0.0)
        digits = ui.digits_for(field, 3)
        if running is None:
            st.info(f"The engine has no {ui.label(field)} for {seat_names.get(pid, pid)} — pick "
                    "another number, or check that he is in the room this one belongs to.")
        else:
            row = st.columns([4, 4])
            with row[0]:
                ui.tiles([
                    {"name": "the engine says", "value": engine, "digits": digits},
                    {"name": "the projection is running on", "value": running, "digits": digits,
                     "highlight": True,
                     "sub": ("" if engine is None or abs(running - engine) < 1e-9
                             else f"{running - engine:+.{digits}f} from your edits")},
                ])
            how = row[1].radio("Try it", ["as a percentage of what he has", "as an exact number"],
                               horizontal=True, key=ui._keyed(f"whatif:how:{picked}"))
            if how.startswith("as a percentage"):
                pct = st.slider("Percentage of the number he is running on now", 50, 150, 100, 5,
                                format="%d%%", key=ui._keyed(f"whatif:pct:{picked}:{field}"))
                mode, value = "multiply", pct / 100.0
                trying = running * value
                live_edit = pct != 100
            else:
                trying = st.number_input(
                    f"{ui.label(field)} for {seat_names.get(pid, pid)}", value=float(running),
                    step=None, format=f"%.{digits}f",
                    key=ui._keyed(f"whatif:set:{picked}:{field}"),
                )
                mode, value = "set", float(trying)
                live_edit = abs(float(trying) - running) > 10.0 ** -(digits + 2)

            if not live_edit:
                st.info("Move the control and the three panels fill in: what it does to him, what it "
                        "does to his teammates, and what it does to the offence's totals.")
            else:
                r = ui.ripple(view, pid, field, mode, float(value))
                st.markdown(f"**{ui.label(field)}**  ·  `{running:,.{digits}f}` → "
                            f"`{trying:,.{digits}f}`")
                ui.tiles([
                    {"name": r.player, "value": r.points_after, "sub": f"{r.his_points:+.1f} points",
                     "highlight": True},
                    {"name": f"his teammates ({r.movers.height})", "value": r.room_points,
                     "signed": True, "sub": "every other man, added up"},
                    {"name": "the whole offence", "value": r.net_points, "signed": True,
                     "sub": "near zero is production moved, not created"},
                    {"name": f"{r.player.split()[-1]} at {his_position}", "value": r.rank_after,
                     "digits": 0,
                     "sub": ("" if r.rank_after is None or r.rank_before is None
                             else f"{r.rank_before - r.rank_after:+.0f} places")},
                ])
                st.info(r.reading)

                his, theirs = st.columns([2, 3])
                with his:
                    st.caption("**Him**, line by line")
                    ui.table(
                        r.me,
                        config={"stat": st.column_config.TextColumn("stat"),
                                "before": st.column_config.NumberColumn("now", format="%.1f"),
                                "after": st.column_config.NumberColumn("with your edit",
                                                                       format="%.1f"),
                                "change": st.column_config.NumberColumn("change", format="%+.1f")},
                        height=min(80 + 35 * r.me.height, 420),
                    )
                with theirs:
                    st.caption("**Everybody else who moved** — biggest first")
                    if r.movers.is_empty():
                        st.success("Nobody else on the roster moved: this number is his alone.")
                    else:
                        cols = ["player", "position", "fantasy_points", "new_fantasy_points",
                                "d_fantasy_points", "position_rank", "new_position_rank"]
                        extra = [c for c in (f"d_{pool}", "d_targets", "d_carries", "d_receptions",
                                             "d_receiving_yards", "d_rushing_yards")
                                 if c in r.movers.columns]
                        keep = [c for c in cols if c in r.movers.columns] + list(dict.fromkeys(extra))
                        ui.rank_bars(r.movers.sort("d_fantasy_points"), "d_fantasy_points",
                                     height=max(200, 26 * min(r.movers.height, 12)), colour="#e45756")
                        ui.table(r.movers.select(keep), height=min(80 + 35 * r.movers.height, 380))

                if facts and facts.get("exclusive") and not con.is_empty():
                    st.caption(f"**The {pool_name.lower()}, before and after** — the same pool, "
                               "divided two ways")
                    shift = ui.contribution_shift(view, r.candidate, picked, pool)
                    if not shift.is_empty():
                        bars = shift.head(CHART_LIMIT).select("player", "now", "after").rename(
                            {"after": "with your edit"})
                        st.bar_chart(bars, x="player", y=["now", "with your edit"],
                                     horizontal=True, stack=False,
                                     x_label=f"projected {pool_name.lower()}", y_label="",
                                     height=max(240, 40 * bars.height + 40))
                        with st.expander("The same thing as numbers"):
                            ui.table(
                                shift.select("player", "position", "depth_slot", "now", "after",
                                             "change"),
                                config={**ui.fixed(1, "now", "after"),
                                        "change": st.column_config.NumberColumn("change",
                                                                                format="%+.1f")},
                                height=min(80 + 35 * shift.height, 420),
                            )

                with st.expander("What the offence's own totals did"):
                    ui.table(
                        r.team_totals.filter(pl.col("before").is_not_null()),
                        config={"stat": st.column_config.TextColumn("team total"),
                                "before": st.column_config.NumberColumn("now", format="%.1f"),
                                "after": st.column_config.NumberColumn("with your edit",
                                                                       format="%.1f"),
                                "change": st.column_config.NumberColumn("change", format="%+.1f")},
                        height=420,
                    )
                    ui.note(
                        "With pool normalisation on, a share edit leaves the counts drawn straight from "
                        "a pool exactly where they were — the targets and the carries do not move, only "
                        "who has them. Yards and receptions are a count times a rate, so they move by "
                        "the difference in efficiency between the man who gained the work and the men "
                        "who lost it. And **touchdowns are their own pool**: giving him targets does "
                        "not give him touchdowns, `rec_td_share` does."
                    )
                    if not r.elsewhere.is_empty():
                        st.warning(f"{r.elsewhere.height} player(s) off this roster moved too, which a "
                                   "player edit should not cause. History's ⚖️ Share sums tab is the place to "
                                   "take that.")

                st.divider()
                apply_row = st.columns([2, 5])
                if apply_row[0].button(f"✔ Apply this to {r.player}", type="primary",
                                       key=ui._keyed(f"whatif:apply:{picked}")):
                    ui.edit(overrides.Override("player", pid, field, mode, float(value),
                                               base=engine, note=f"what if · {picked}"))
                    st.rerun()
                apply_row[1].caption(
                    "One override, with the engine's own number recorded as the base — so it shows as a "
                    "diff everywhere downstream and a reset puts him back on the estimator rather than "
                    "on this number."
                )

            with st.expander(f"Check {ui.label(field)} before you type over it: the room, the record "
                             "and the league"):
                room_ids = tuple(seats.filter(pl.col("position") == his_position)["player_id"]
                                 .to_list())
                ui.room_panel(view, field, room_ids, key=f"whatif:room:{picked}:{field}",
                              position=his_position)


# --------------------------------------------------------------------------- #
# 6. does it add up
# --------------------------------------------------------------------------- #
with audit_tab:
    ui.section(
        "This team's pools",
        "What this roster's shares summed to *before* scaling, against what the pool was measured to "
        "sum to. A negative `gap_pct` is a team with work nobody on the depth chart has claimed — "
        "usually a vacated role — and `factor` is what normalisation multiplied every share by to close "
        "it. With the sidebar toggle off, the gap is left in place and this team projects that much "
        "short of its own offence.",
    )
    mine = ui.team_pools(view).filter((pl.col("team") == picked) & pl.col("exclusive")).drop("team")
    ui.table(
        mine, digits=3,
        config={**ui.fixed(3, "measured_target", "raw_sum", "factor"), **ui.fixed(1, "gap_pct")},
    )

    worst = ui.team_pools(view).filter(pl.col("exclusive"))
    with st.expander("Every team, ranked by how far its shares were off"):
        pool_pick = st.selectbox("Pool", sorted(worst["pool"].unique().to_list()),
                                 index=sorted(worst["pool"].unique().to_list()).index("targets"))
        ranked = (worst.filter(pl.col("pool") == pool_pick)
                  .select("team", "raw_sum", "factor", "gap_pct").sort("gap_pct"))
        ui.rank_bars(ranked, "gap_pct", name="team", height=440, top=32, digits=1)
        ui.table(ranked, config={**ui.fixed(3, "raw_sum", "factor"), **ui.fixed(1, "gap_pct")},
                 height=460)

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
            "rounding. Yards and completions are a count times a rate and are deliberately not forced: "
            "a roster more efficient than the team's recent offence should project above it, and "
            "scaling that away would hide the disagreement rather than show it."
        )

    # The other half of "does it add up", once anything has been typed: whether what moved is what was
    # meant to move. It is here rather than on the Edits page because the teammates are the point — a
    # share given is a share taken, and the ledger cannot show that without being about one team.
    if touching:
        st.divider()
        moved = ui.board_diff(view).filter(
            pl.col("team").eq(picked) | pl.col("new_team").eq(picked)
        )
        if moved.is_empty():
            ui.section("What your edits moved here",
                       "Every edit applied, and none of them moved a projected point on this roster.")
        else:
            ui.section(
                "What your edits moved here",
                "Against the same season with no edits. The teammates are in here on purpose: with "
                "pool normalisation on, a share given to one player is taken from the others, so the "
                "column that matters is whether the losses look like the ones you meant to cause.",
            )
            ui.rank_bars(moved.sort("d_fantasy_points", descending=True), "d_fantasy_points",
                         height=max(220, 26 * min(moved.height, 14)))
            ui.table(
                moved.select("player", "position", "fantasy_points", "new_fantasy_points",
                             "d_fantasy_points", "d_targets", "d_carries", "d_receiving_yards",
                             "d_rushing_yards", "position_rank", "new_position_rank"),
                config={**ui.fixed(1, "fantasy_points", "new_fantasy_points", "d_fantasy_points",
                                   "d_receiving_yards", "d_rushing_yards"),
                        **ui.fixed(2, "d_targets", "d_carries")},
                height=380,
            )
        ui.note("Where each of these edits landed, and what it was recorded against, is the **Edits** "
                "page's own list — one row per edit per frame it was offered to.")
