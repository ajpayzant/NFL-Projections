"""Who is going to be out there, and for how long — the whole league on one surface.

Availability is the term the projection *states* rather than fits. Every other input is estimated from
something a player did; `expected_games` is his own attendance blended with his slot's, then multiplied
by a factor attached to his roster status, and that factor is a judgement. So this page puts the
judgement, the population it decides and the injury record behind it in one place, and makes the edit
where the evidence is.

This is the league-wide surface: one pass over everybody whose attendance is in question, in one sitting.
A single team's games are on the **Team** page's 🚑 Availability tab, beside the depth chart that decides
who picks up what he drops.

Two tabs, which are the two shapes the question comes in:

1. **Who needs a decision** — everybody the projection is not treating as a full season, with the record
   beside the knob. A short list, because a page of nine hundred rows is a page nobody reviews; the whole
   board is under it as a sheet for the case where you already know who you are after, and the assumptions
   and the league's injury report are under that, because they are what put men on the list.
2. **Back in week N** — a return date, translated into the games it is worth off that team's own
   schedule. Stated plainly: the engine spreads availability evenly over the season rather than holding
   specific weeks out, so a date is a count here and not a set of weeks.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl                                                          # noqa: E402
import streamlit as st                                                       # noqa: E402

import ui                                                                    # noqa: E402

view = ui.controls("Availability", icon="🚑")
ui.scenario_banner(view)

men = ui.availability_board(view)
flagged = men.filter(pl.col("review") != "")
edits = [o for o in ui.live().items if o.field == ui.GAMES_METRIC]
starters = men.filter(pl.col("depth_slot") == 1)

ui.section(
    "Who is going to be out there, and for how long",
    f"`expected_games` is out of {ui.FULL_SEASON:.0f} — the games a team plays in an "
    f"{ui.REG_WEEKS}-week calendar. It is inside every count in the projection rather than applied to "
    "the total afterwards, so halving it halves his share of every pool before the team's work is "
    "divided, and the men behind him pick the rest up. It is also the one term the projection *states* "
    "rather than fits, which is why the record behind it is on this page beside the knob.",
    sub="the availability term, its population, and the record behind it",
    level=2,
)
ui.tiles([
    {"name": "worth a look", "value": flagged.height, "digits": 0, "highlight": True,
     "sub": "not being treated as a whole season"},
    {"name": "not active", "value": men.filter(pl.col("status") != "ACT").height, "digits": 0,
     "sub": "a roster status carrying a stated factor"},
    {"name": "starters' projected games", "value": starters["expected_games"].mean(),
     "sub": f"mean over every slot 1 · of {ui.FULL_SEASON:.0f}"},
    {"name": "your availability edits", "value": len(edits), "digits": 0,
     "sub": "games counts you have typed"},
    {"name": "players", "value": men.height, "digits": 0, "sub": "on the board"},
])

TABS = [f"🚑 Who needs a decision ({flagged.height})", "📆 Back in week N"]
review_tab, return_tab = st.tabs(TABS)

# The columns the argument is made on: what ran, what the engine said, his own record and its size, the
# job's average, then what the league's injury report says about him.
GRID = ["player_id", "player", "position", "team", "depth_slot", "status", "review", "expected_games",
        "engine", "obs", "n", "own_weight", "prior", "weeks_out", "worst_season_out", "main_injury",
        "report", "games", "fantasy_points", "edited", "edits",
        *[str(s) for s in ui.HISTORY_VIEW], "record"]


# the eight columns worth reading in a list, with the argument itself -- record against job -- in the
# panel beside it rather than in eleven more columns nobody scrolls to
AVAIL_LIST = ["player", "position", "team", "depth_slot", "status", "review", "expected_games",
              "engine", "edited"]


def avail_panel(row: dict, key: str) -> None:
    """One man's availability argument: what ran, what it was made of, and the ✎ on it.

    The whole point of the redesign on this page. The old surface was a nineteen-column grid where the
    number you were meant to change sat between the number it came from and the record it came out of,
    and reading across a row was the work. Here the four numbers are cards, the blend is one meter with
    his record and his job's average marked on it, and his attendance is drawn.
    """
    pid = row["player_id"]
    head, act = st.columns([8, 3], vertical_alignment="center")
    head.markdown(f"### {row['player']} · {row['position']} {row['team']}")
    with act:
        ui.edit_button(pid, key=f"{key}:bench", width="stretch")
    ui.chips(
        f"slot {int(row['depth_slot'])}" if row.get("depth_slot") is not None else "",
        str(row["status"]) if row.get("status") else "",
        str(row["review"]) if row.get("review") else "",
        "carries an override" if row.get("edited") else "",
    )
    ui.tiles([
        {"name": "projected games", "value": row.get("expected_games"), "highlight": True,
         "sub": f"of {ui.FULL_SEASON:.0f}"},
        {"name": "the engine's own", "value": row.get("engine"),
         "sub": "before anything you typed"},
        {"name": "his record", "value": row.get("obs"),
         "sub": (f"over {float(row['n_seasons']):.0f} seasons"
                 if row.get("n_seasons") is not None else "no seasons on file")},
        {"name": "his job's average", "value": row.get("prior"),
         "sub": "everyone who has held this slot"},
        {"name": "roster status", "value": row.get("status_factor"), "digits": 2,
         "sub": f"{row.get('status')} · a stated factor"},
    ])
    knob = st.columns([5, 2])
    with knob[0]:
        ui.meters([ui.meter_html(
            "projected games", row.get("expected_games"), maximum=ui.FULL_SEASON, digits=1,
            marks=(("the engine", row.get("engine")), ("his record", row.get("obs")),
                   ("his job", row.get("prior"))),
            foot=((f"{float(row['own_weight']):.0%} of it is his own record"
                   if row.get("own_weight") is not None else ""),
                  (f"the league ruled him out {float(row['weeks_out']):.0f} weeks"
                   if row.get("weeks_out") else "")),
        )])
    with knob[1]:
        st.caption("Change his games")
        ui.knob_popover(view, pid, ui.GAMES_METRIC,
                        base=None if row.get("engine") is None else float(row["engine"]),
                        key=key, digits=1)
    ui.edit_list("player", pid, heading="**Your edits on him**", v=view, key=f"{key}:edits")

    att = ui.attendance().filter(pl.col("player_id") == pid).sort("season")
    if att.is_empty():
        st.caption("No played season in the tables on view — a rookie has no attendance record, which is "
                   "exactly why his number is mostly his job's average.")
    else:
        st.caption("Games played, season by season, against what is projected for him now")
        drawn = pl.concat([
            att.select("season", pl.col("games").cast(pl.Float64)),
            pl.DataFrame({"season": [view.season],
                          "games": [float(row.get("expected_games") or 0.0)]},
                         schema={"season": att.schema["season"], "games": pl.Float64}),
        ])
        ui.season_bars(drawn, "season", "games", mark=view.season, height=200, digits=1)

    his = ui.injuries().filter(pl.col("player_id") == pid).sort("season", descending=True)
    if not his.is_empty():
        with st.expander("What the league has said about him"):
            ui.table(his.select([c for c in ("season", "team", "weeks_listed", "weeks_out",
                                             "weeks_doubtful", "weeks_questionable", "weeks_dnp",
                                             "weeks_limited", "main_injury", "distinct_injuries")
                                 if c in his.columns]), height=220)
            ui.note("`weeks_out` is the league itself ruling him out, which is the closest a report gets "
                    "to observed missed time. `weeks_questionable` is far softer and is here to be read "
                    "beside it rather than added to it — most Questionable designations play.")


def games_grid(frame: pl.DataFrame, key: str, height: int = 520) -> None:
    """The one editable surface on this page: type in `expected_games`, everything else is the record."""
    shown = frame.select([c for c in GRID if c in frame.columns])
    ui.grid(
        shown, level="player", key_col="player_id", fields=(ui.GAMES_METRIC,), key=key,
        digits=1, height=height, hide=("player_id",),
        config={**ui.evidence_config(ui.GAMES_METRIC, digits=1),
                **ui.fixed(1, "games", "fantasy_points", "engine", "obs", "prior"),
                **ui.sparkline("report", "weeks out"),
                **ui.sparkline("record", "games played")},
    )


# --------------------------------------------------------------------------- #
# 1. who needs a decision
# --------------------------------------------------------------------------- #
with review_tab:
    ui.section(
        "Everybody the projection is not treating as a whole season",
        "`review` says why he is on this list: a roster status that is not active, a starter the model "
        "has missing more than two games, a man projected well under his own attendance record, or one "
        "you have already edited. 17 for a starter nobody has reported hurt is a perfectly ordinary "
        "edit, and the estimator is deliberately conservative there because it shrinks him towards the "
        "average attendance of everyone who has held his slot.",
        sub="click a name; his whole argument is on the right",
    )
    if flagged.is_empty():
        st.info("Nobody: every player is active and projected in line with his own record.")
    else:
        filters = st.columns([3, 3])
        with filters[0]:
            why = st.multiselect("Reason", sorted(flagged["review"].unique().to_list()),
                                 default=sorted(flagged["review"].unique().to_list()))
        with filters[1]:
            pos = ui.position_filter("avail:pos")
        picked = flagged.filter(pl.col("review").is_in(why)) if why else flagged
        picked = picked.filter(pl.col("position").is_in(pos)) if pos else picked
        st.caption(f"{picked.height} players, the least available first")

        left, right = st.columns([5, 7], gap="medium")
        with left:
            man = ui.pick_from(
                picked, key="avail:review:list", columns=AVAIL_LIST, height=560,
                config={**ui.fixed(1, "expected_games", "engine"),
                        "depth_slot": st.column_config.NumberColumn("slot", format="%d",
                                                                    width="small"),
                        "edited": st.column_config.CheckboxColumn("✏️", width="small")},
            )
        with right:
            if man is None:
                st.info("Pick a name.")
            else:
                with st.container(border=True):
                    avail_panel(man, key="avail:review:knob")

        with st.expander("The same men as a sheet — for a session that is really about typing"):
            games_grid(picked, key="avail:review", height=560)

    # The whole board, under the short list rather than beside it on a tab of its own: the list is what a
    # review pass reads, and this is the surface for the other case -- you already know the name, and the
    # reason he is not on the list is exactly what you are about to disagree with.
    with st.expander("Every player on the board, and the chain behind the number"):
        cols = st.columns([2, 2, 2, 1])
        with cols[0]:
            pos_all = ui.position_filter("avail:all:pos")
        with cols[1]:
            teams = ui.team_filter(view, "avail:all:team")
        search = cols[2].text_input("Search", key="avail:all:search")
        only = cols[3].toggle("Starters only", value=False, key="avail:all:starters")
        shown = ui.apply_filters(men, pos_all, teams, search)
        if only:
            shown = shown.filter(pl.col("depth_slot") <= 2)
        st.caption(f"{shown.height} players, the least available first. `expected_games` is the only "
                   "editable column — everything else is the record it was made from.")
        games_grid(shown, key="avail:all", height=620)

        st.caption("The full availability chain for these players")
        ui.table(
            shown.select([c for c in ("player", "position", "team", "depth_slot", "avail_slot",
                                      "status", "obs_games", "prior_games", "games_if_available",
                                      "presence", "status_factor", "expected_games", "active_weeks",
                                      "charted") if c in shown.columns]),
            height=520,
        )
        ui.note(
            "Left to right is the multiplication: his own attendance blended with his slot's gives "
            "`games if fully available`, times `chance the job exists` (is anybody in this slot on the "
            "field), times the factor for his roster status, gives `projected games`. "
            "`share of the season active` is that over "
            f"{ui.FULL_SEASON:.0f}, and it is what multiplies every share of every pool."
        )

    st.divider()
    ui.section("Check a room before you type over it",
               "The same metric across a position, with each man's attendance record and where the "
               "position's slots sit. A share is zero-sum and so is a snap: giving a backup eight games "
               "is a claim about the starter in front of him.",
               sub="the room, because availability is zero-sum too")
    room_cols = st.columns([1, 1, 3])
    team = room_cols[0].selectbox("Team", sorted(men["team"].unique().to_list()), key="avail:room:team")
    who = room_cols[1].selectbox("Position", list(ui.POSITIONS), key="avail:room:pos")
    room_ids = tuple(men.filter((pl.col("team") == team) & (pl.col("position") == who))
                     ["player_id"].to_list())
    ui.room_panel(view, ui.GAMES_METRIC, room_ids, key=f"avail:room:{team}:{who}", position=who)

    st.divider()
    # What decides who is on the list above, and the record an argument against it is made from. Both were
    # a tab of their own, which is the wrong shape for reference material: nobody opens it before making a
    # judgement, and it belongs under the judgement it explains.
    # The assumption is editable here rather than displayed here, and that is the point of the section.
    # Asserted numbers used to be read-only on the grounds that a stated assumption is not a knob, which
    # left a reader who disagreed with one of them exactly one way to say so: type `expected_games = 0`
    # on every man holding the status. Two numbers restate the assumption; two hundred edits restate it
    # one player at a time and bury the argument in the override log.
    with st.expander("The stated assumptions — what each roster status is worth"):
        ui.note(
            "The factor each roster status is worth, and how many players it is currently deciding. This "
            "is the one place in the engine where a number is asserted rather than measured, so it is "
            "shown with its population: a factor on a status nobody holds is a footnote. Move one and "
            "every player holding that status moves with it — which is the cheap way to say *the "
            "practice squad does not play here*, instead of zeroing them one at a time."
        )
        ui.status_editor(view, key="avail:status")

    with st.expander(f"The league's injury report, {ui.HISTORY_VIEW[0]}–{ui.LAST_COMPLETE_SEASON}"):
        ui.note(
            "One row per player-season, regular season only, from the reports the clubs file. This is the "
            "record an availability argument is made from — not an input to the projection, which uses "
            "games actually played, but the reason a games number is or is not believable."
        )
        rep = ui.injuries()
        if rep.is_empty():
            st.info("No injury table in the lake. `python -m src.data.refresh` fetches it.")
        else:
            pick = st.columns([1, 2, 2])
            season = pick[0].selectbox("Season", [*sorted(rep["season"].unique().to_list(),
                                                          reverse=True), "every season"], index=0)
            with pick[1]:
                rpos = st.multiselect("Position",
                                      sorted(rep["position"].drop_nulls().unique().to_list()),
                                      default=["QB", "RB", "WR", "TE"], key="avail:rep:pos")
            sort_by = pick[2].radio("Ranked by", ["weeks_out", "weeks_listed", "weeks_dnp"],
                                    horizontal=True, format_func=ui.label)
            reported = rep if season == "every season" else rep.filter(pl.col("season") == season)
            if rpos:
                reported = reported.filter(pl.col("position").is_in(rpos))
            ranked = (reported.select([c for c in ("player", "position", "team", "season",
                                                  "weeks_listed", "weeks_out", "weeks_doubtful",
                                                  "weeks_questionable", "weeks_dnp", "weeks_limited",
                                                  "main_injury", "distinct_injuries")
                                       if c in reported.columns])
                      .sort(sort_by, descending=True).head(60))
            ui.rank_bars(ranked.head(20), sort_by, "player", height=420, digits=1)
            ui.focus_table(
                ranked, ["player", "position", "team", "season", "weeks_listed", "weeks_out",
                         "weeks_dnp", "main_injury"],
                key=f"avail:rep:{season}:{sort_by}", height=480, digits=1,
                label_text="every designation, not only these",
            )

            by_pos = reported.group_by("position").agg(
                pl.len().alias("player_seasons"),
                pl.col("weeks_out").mean().alias("mean_weeks_out"),
                pl.col("weeks_listed").mean().alias("mean_weeks_listed"),
                (pl.col("weeks_out") >= 4).mean().alias("share_missing_a_month"),
            ).sort("mean_weeks_out", descending=True)
            st.caption("**Weeks ruled out, by position.** Read as a base rate rather than as a "
                       "forecast: it is the average over players who appeared on a report at all, so it "
                       "is the population of men who got hurt, not of everybody.")
            ui.meters([
                ui.meter_html(f"{r['position']} · {int(r['player_seasons'])} player-seasons",
                              float(r["mean_weeks_out"]),
                              maximum=float(by_pos["mean_weeks_out"].max()) * 1.2, digits=1,
                              foot=(f"{float(r['share_missing_a_month']):.0%} of them missed a month "
                                    "or more",
                                    f"listed {float(r['mean_weeks_listed']):.1f} weeks on average"))
                for r in by_pos.rows(named=True)
            ])

# --------------------------------------------------------------------------- #
# 2. back in week N
# --------------------------------------------------------------------------- #
with return_tab:
    ui.section(
        "A return date, as the games it is worth",
        "The engine holds availability as a season count spread evenly across the weeks — there is no "
        "per-week availability knob, because `expected_games` is applied to a frame that has no weeks in "
        "it. So a return date is converted here into the number of games it leaves, off this team's own "
        "schedule, and what gets written is that count. His projected line is thinned evenly rather "
        "than zeroed for named weeks: right for a season total and for a draft board, and not a "
        "substitute for a start/sit call in the weeks concerned. For one week only, the ✎ on `chance he "
        "plays` on the Week page is the right knob.",
        sub="pick a man, then the week he is back",
    )
    labels = (men.with_columns(
        (pl.col("player") + "  ·  " + pl.col("position") + " " + pl.col("team")
         + pl.when(pl.col("review") != "")
         .then(pl.lit("  ·  ") + pl.col("review")).otherwise(pl.lit(""))).alias("label")
    ).sort(["review", "fantasy_points"], descending=[True, True], nulls_last=True))
    chosen = st.selectbox("Player", labels["label"].to_list(), index=0, key="avail:return:player")
    row = labels.filter(pl.col("label") == chosen).row(0, named=True)
    schedule = ui.team_weeks(view, row["team"])
    bye = [w for w in range(1, ui.REG_WEEKS + 1) if w not in schedule]

    ui.tiles([
        {"name": "projected games", "value": row.get("expected_games"), "highlight": True,
         "sub": (f"the engine had {row['engine']:.1f}" if row.get("engine") is not None
                 else f"of {ui.FULL_SEASON:.0f}")},
        {"name": "his own record", "value": row.get("obs"),
         "sub": f"over {row.get('n_seasons') or 0:.0f} seasons"},
        {"name": "roster status", "value": row.get("status_factor"), "digits": 2,
         "sub": f"{row['status']} · a stated factor"},
        {"name": f"{row['team']} bye", "value": bye[0] if bye else None, "digits": 0,
         "sub": "the week they are away"},
    ])

    pick = st.columns([3, 2, 2, 2])
    back = pick[0].select_slider("Back in week", schedule, value=schedule[0],
                                 key=ui._keyed(f"avail:back:{row['player_id']}"))
    left = ui.games_from_week(view, row["team"], int(back))
    with pick[1]:
        ui.tiles([{"name": "games left", "value": left, "digits": 0,
                   "sub": f"of {len(schedule)} on the schedule"}])
    base = None if row.get("engine") is None else float(row["engine"])
    if pick[2].button(f"Set to {left} games", width="stretch", type="primary",
                      key=ui._keyed(f"avail:setback:{row['player_id']}"),
                      help=f"back for week {int(back)} on this team's schedule"):
        ui.set_games(row["player_id"], float(left), base, f"back in week {int(back)}")
        st.rerun()
    if pick[3].button("Out for the season", width="stretch",
                      key=ui._keyed(f"avail:out:{row['player_id']}")):
        ui.set_games(row["player_id"], 0.0, base, "out for the season")
        st.rerun()

    st.divider()
    with st.container(border=True):
        avail_panel(row, key="avail:return:knob")

