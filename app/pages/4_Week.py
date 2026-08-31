"""The season one week at a time, which is how it is actually played — and one game at a time under it.

The engine is per-week the whole way through: `volume(week) = participation x share x team volume(week)`,
seventeen times, and the season total is their sum. A page that only ever shows the sum hides half of
what was computed — and the half it hides is the half a lineup is set from. Two receivers with the same
total are a different problem if one of them is away in week 12 and the other in week 6.

Nothing here is a new number. It is the same weekly frame the totals are made of, read across rather
than down: the matchup, the chance he plays, the line the market has posted, and where he ranks at his
position in that week alone.

One week picker and one fixture picker at the top, and then four tabs, widest first:

1. **The week's board** — every man in the chosen week, the slate he is playing in, the same seventeen
   weeks as a grid, and the calendar of byes behind all of it.
2. **One game** — the fixture picked above at the scale the engine actually builds at. `compose.weekly`
   is one row per player per game, so a game is not a slice of the answer, it *is* the answer, twice
   over: two offences with a projected volume each, divided among the men on their rosters, added back
   up into two stat lines and a score.
3. **Adjust this game** — the knobs for one afternoon. Most disagreements about a projection are about
   one afternoon: he is banged up this week, their left tackle is out, the starter is being rested with
   the division wrapped up. None of those is a season-long claim, and typing one in as a season-long
   claim is how a projection ends up wrong in sixteen games in order to be right in one. So a per-game
   edit lands where the engine divides the work — inside `opportunity()`, before the team's pool is
   split — and the season follows on its own, because the season is the sum of the seventeen and one of
   them has moved.
4. **Opponent & market** — the schedule as an *input*. Two different things get called a matchup and
   they are worth separating. **The game's scoring level** — shootout or slog — comes from the posted
   line where one exists and the team estimate where it does not, and it is the part with real
   predictive weight: it moves plays, red-zone trips and touchdowns. **The opponent's defence against a
   position** is the fantasy convention, shown here as last season's measured rate allowed and shown as
   *history* rather than as a projection, because one season of defence-versus-position is 17 games and
   it is noisy. The engine's own defensive term is the league-normalised season rating on the Team page,
   not that table. Read it to explain a number, not to build one.

Three things the editing tab will not do:

- **Type over a projected total.** The stat line is the product of the inputs beside it. Change an input.
- **Move a season number for you.** `depth_slot` and the games played are one number for the whole
  season, so they are shown here and edited on the Team page or the roster sheet.
- **Pretend the per-game context is proven.** The held-out backtest is a dead heat at season level on the
  per-game chain. Week-to-week spread is informative about the schedule and unproven as a weekly ranking,
  which is an argument for editing a game you know something about — not for trusting the seventeen.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl                                                          # noqa: E402
import streamlit as st                                                       # noqa: E402

import ui                                                                    # noqa: E402
from src.config import LAST_COMPLETE_SEASON                                  # noqa: E402

view = ui.controls("Week", icon="🗓️")
ui.scenario_banner(view)

# --------------------------------------------------------------------------- #
# which week, and which game inside it: one pair of pickers for the whole page
# --------------------------------------------------------------------------- #
# These were two pages with a week picker each, which is how a session came to have week 5 open on one
# and week 12 on the other. The week is the frame of everything below; the fixture is the frame of the
# two middle tabs, and it re-lists itself when the week moves.
every = ui.weeks(view)
pick = st.columns([3, 3, 3])
week = pick[0].select_slider("Week", every, value=every[0], key="weekly:week")
slate = ui.game_index(view, int(week))
fixtures = slate["game"].to_list()
choice = pick[1].selectbox("Game", fixtures, key="games:game") if fixtures else None
pick[2].caption("The week is the frame of the whole page. The fixture is what 🏈 **One game** and "
                "🎛️ **Adjust this game** are about — the season follows the games rather than the other "
                "way round.")

out = ui.byes(view, int(week))
board = ui.week_board(view, int(week))
env_all = ui.environment(view)
env = env_all.filter(pl.col("week") == int(week))

if choice is None:
    st.warning(f"No games in week {int(week)}, so the two game tabs have nothing to open.")
    row: dict = {}
    game_id = away = home = None
else:
    row = slate.filter(pl.col("game") == choice).row(0, named=True)
    game_id, away, home = row["game_id"], row["away"], row["home"]

ui.section(
    f"Week {int(week)}",
    "Nothing on this page is a new number: it is the same weekly frame the season totals are the sum "
    "of, read across rather than down. A man expected for fourteen of seventeen games carries 0.82 in "
    "every week rather than three zeroes, because the engine holds a games count — so `chance he plays` "
    "is availability spread evenly, and the place to say *this* week is the ✎ beside it.",
    sub=f"{board['team'].n_unique()} teams playing · {len(out)} on a bye",
    level=2,
)
ui.tiles([
    {"name": "players projected", "value": board.height, "digits": 0,
     "sub": f"in week {int(week)} alone"},
    {"name": "top projection", "value": board["fantasy_points"].max(), "highlight": True,
     "sub": board.row(0, named=True)["player"]},
    {"name": "mean implied points", "value": None if env.is_empty() else env["implied_points"].mean(),
     "sub": ("" if env.is_empty()
             else f"{int(env['has_market'].sum())} of {env.height} games with a line")},
    {"name": "teams playing", "value": board["team"].n_unique(), "digits": 0,
     "sub": f"{len(out)} away"},
])
if out:
    ui.chips(f"on a bye in week {int(week)}: {', '.join(out)}")
    st.caption("Their players are absent from the list below rather than sitting in it on zero.")

TABS = ["📋 The week's board", "🏈 One game", "🎛️ Adjust this game", "🛡️ Opponent & market"]
board_tab, game_tab, adjust_tab, market_tab = st.tabs(TABS)

# --------------------------------------------------------------------------- #
# 1. the week's board: the men, the slate they are in, the season as a grid, the calendar
# --------------------------------------------------------------------------- #
with board_tab:
    ui.section(f"Week {int(week)}, a man at a time",
               "`rank at position that week` is ranked on this week alone, which is the number a "
               "start/sit argument is actually about — a season rank cannot say that a WR2 has the best "
               "matchup on the board this Sunday.",
               sub="click a name; his week is on the right")
    filters = st.columns([2, 2, 2, 2])
    with filters[0]:
        positions = ui.position_filter("weekly:pos")
    with filters[1]:
        teams = ui.team_filter(view, "weekly:team")
    search = filters[2].text_input("Search", key="weekly:search")
    mode = filters[3].radio("Show", ["Fantasy", "Stat line"], horizontal=True, key="weekly:mode")

    shown = ui.apply_filters(board, positions, teams, search)
    if shown.is_empty():
        st.info("Nothing matches those filters in this week.")
    else:
        # `week_rank` stays in the list on purpose: it is the one column this page has that the season
        # board cannot, and the reason to open the page at all
        WEEK_LIST = ["player", "position", "team", "opponent", "week_rank", "p_play",
                     "fantasy_points", "edited"]
        STAT_LIST = ["player", "position", "team", "opponent", "week_rank", "p_play"]
        listing = shown.sort("fantasy_points", descending=True).head(300)
        if mode == "Fantasy":
            cols = [c for c in WEEK_LIST if c in listing.columns]
        else:
            stats = [c for c in ui.stat_columns(positions or list(ui.POSITIONS))
                     if c in listing.columns]
            cols = [c for c in STAT_LIST if c in listing.columns] + stats + ["fantasy_points"]

        left, right = st.columns([5, 7], gap="medium")
        with left:
            man = ui.pick_from(
                listing, key=f"weekly:list:{int(week)}:{mode}", columns=cols, height=560,
                config={**ui.stat_config([c for c in cols if c in ui.ALL_STATS]),
                        **ui.fixed(1, "fantasy_points"), **ui.percent("p_play"),
                        "week_rank": st.column_config.NumberColumn("rank", format="%d",
                                                                   width="small"),
                        "edited": st.column_config.CheckboxColumn("✏️", width="small")},
            )
        with right:
            if man is None:
                st.info("Pick a name.")
            else:
                with st.container(border=True):
                    head, act = st.columns([8, 3], vertical_alignment="center")
                    head.markdown(f"### {man['player']} · {man['position']} {man['team']} — "
                                  f"week {int(week)}")
                    with act:
                        ui.edit_button(man["player_id"], key=f"weekly:bench:{int(week)}",
                                       week=int(week), label_text=f"✎ Adjust week {int(week)}",
                                       width="stretch")
                    ui.chips(
                        (f"{'at' if not man.get('is_home') else 'hosting'} {man.get('opponent')}"
                         if man.get("opponent") else ""),
                        "line posted" if man.get("has_market") else "model only",
                        str(man["status"]) if man.get("status") not in (None, "ACT") else "",
                        "carries an override" if man.get("edited") else "",
                    )
                    ui.tiles([
                        {"name": "fantasy points", "value": man.get("fantasy_points"),
                         "highlight": True, "sub": f"week {int(week)}"},
                        {"name": f"{man['position']} rank this week",
                         "value": man.get("week_rank"), "digits": 0,
                         "sub": "in this week alone"},
                        {"name": "his game", "value": man.get("implied_points"),
                         "sub": (f"total {man['total']:.1f} · spread {man['spread']:+.1f}"
                                 if man.get("total") is not None
                                 and man.get("spread") is not None else "his team's implied points")},
                    ])
                    play = st.columns([3, 2])
                    with play[0]:
                        ui.tiles([{"name": "chance he plays", "value": man.get("p_play"),
                                   "digits": 3, "percent": True,
                                   "sub": "availability, spread over the season"}])
                    with play[1]:
                        st.caption(f"Change it for week {int(week)} only")
                        ui.knob_popover(view, man["player_id"], "p_play",
                                        base=man.get("p_play"), week=int(week),
                                        key=f"weekly:play:{int(week)}", digits=3)
                    ui.stat_blocks(man, man.get("position"))
                    st.caption("His seventeen games — this week is one bar of it")
                    his = ui.weekly(view).filter(pl.col("player_id") == man["player_id"]).sort("week")
                    ui.week_bars(his.select("week", "fantasy_points"), "fantasy_points", height=200)

        ui.section("The top of the board this week", sub=f"week {int(week)}, best fifteen")
        ui.rank_bars(listing.head(15), "fantasy_points", "player", height=320, digits=1)

        with st.expander("Everybody in this week as a table"):
            ui.table(listing.select(cols), digits=2, height=560,
                     config={**ui.stat_config([c for c in cols if c in ui.ALL_STATS]),
                             **ui.fixed(1, "fantasy_points")})

    st.divider()
    # the slate itself, which was the last tab of a Matchups page. It belongs under the week's board
    # rather than beside it: the first question about a man's week is who he is playing, and the second
    # is what kind of afternoon that game is projected to be.
    ui.section("The games in this week",
               "The slate with its own context — the roof, the rest, the line — for every team playing. "
               "`implied points` is the posted line where there is one and the model's own estimate "
               "where there is not, and it is the number every player's volume in this week is scaled "
               "by. Per-game context is the one part of the engine the held-out backtest could not "
               "prove at season level, so treat the week-to-week spread as informative about the "
               "schedule and unproven as a weekly ranking.",
               sub=f"{env['team'].n_unique() // 2 if not env.is_empty() else 0} fixtures · "
                   f"{len(out)} teams away")
    if env.is_empty():
        st.info(f"No team-games in week {int(week)}.")
    else:
        ui.tiles([
            {"name": "teams playing", "value": env["team"].n_unique(), "digits": 0,
             "sub": f"week {int(week)}"},
            {"name": "highest implied", "value": env["implied_points"].max(), "highlight": True,
             "sub": env.sort("implied_points", descending=True).row(0, named=True)["team"]},
            {"name": "mean total", "value": env["total"].mean(), "sub": "points on the board"},
            {"name": "teams away", "value": len(out), "digits": 0,
             "sub": ", ".join(out) if out else "nobody on a bye"},
        ])
        ui.focus_table(
            env.select([c for c in ("team", "opponent", "is_home", "rest_days", "div_game", "roof",
                                    "temp", "wind", "spread", "total", "implied_points", "has_market",
                                    "plays", "dropbacks", "carries", "targets", "pass_tds", "rush_tds")
                        if c in env.columns]).sort("implied_points", descending=True),
            ["team", "opponent", "is_home", "spread", "total", "implied_points", "plays", "dropbacks",
             "carries", "has_market"],
            key=f"weekly:slate:{int(week)}", height=520, digits=1,
            config={**ui.fixed(1, "spread", "total", "implied_points", "plays", "dropbacks", "carries",
                               "targets", "temp", "wind"),
                    **ui.fixed(2, "pass_tds", "rush_tds")},
            label_text="the roof, the temperature and the rest too",
        )

    st.divider()
    # this one stays a grid on purpose: seventeen weeks across is the shape of a season, and there is no
    # panel that says "away in week 12" more plainly than an empty cell in the twelfth column
    ui.section(
        "A row per player, a column per week",
        "The shape of a season rather than its total. An empty cell is a bye — left empty on purpose, "
        "because a bye is not a zero-point game — and the line at the end is the same row drawn, so a "
        "player whose season is flat is visibly different from one carried by a soft December.",
        sub="a grid, on purpose",
    )
    cols = st.columns([2, 2, 2, 2])
    with cols[0]:
        gpos = ui.position_filter("weekly:grid:pos")
    with cols[1]:
        gteams = ui.team_filter(view, "weekly:grid:team")
    stat = cols[2].selectbox("Value", ["fantasy_points", *ui.ALL_STATS], index=0,
                             format_func=ui.label, key="weekly:grid:stat")
    how_many = cols[3].number_input("Players shown", 10, 300, 60, 10, key="weekly:grid:n")

    people = ui.apply_filters(ui.board(view), gpos, gteams, "")
    ids = tuple(people.sort("fantasy_points", descending=True).head(int(how_many))["player_id"]
                .to_list())
    matrix = ui.weekly_matrix(view, ids, value=stat)
    if matrix.is_empty():
        st.info("Nobody is projected for that stat.")
    else:
        weeks = [str(w) for w in every if str(w) in matrix.columns]
        ui.table(
            matrix.drop("player_id"), digits=1, height=620,
            order=["player", "position", "team", "season_points", "path", *weeks],
            config={**ui.fixed(1, *weeks, "season_points"),
                    "path": st.column_config.LineChartColumn("week by week"),
                    "season_points": st.column_config.NumberColumn(
                        f"season {ui.label(stat)}", format="%.1f")},
        )
        by_week = pl.DataFrame({
            "week": [int(w) for w in weeks],
            "total": [float(matrix[w].fill_null(0.0).sum()) for w in weeks],
            "players": [int(matrix[w].is_not_null().sum()) for w in weeks],
        })
        ui.section("The same rows, summed by week",
                   "The dips are the byes of the teams these players are on, which is the reason a "
                   "positional total is not flat across the season even before a matchup is considered.",
                   sub=f"when these {matrix.height} players actually score")
        ui.week_bars(by_week.select("week", "total"), "total", height=240)
        with st.expander("The same weeks as a table"):
            ui.table(by_week, digits=1, height=200,
                     config={"total": st.column_config.NumberColumn(f"total {ui.label(stat)}",
                                                                   format="%.1f")})

    st.divider()
    ui.section(
        "Who is away, and when",
        "A team with no bye week listed is a team the schedule table has playing every week, which is a "
        "data problem rather than a football one. The weeks with six teams away are the ones a bench "
        "cannot cover, and they are worth knowing before a draft rather than in October. One team's own "
        "seventeen games, with its pace and its scoring level drawn against the league, are on the "
        "🛡️ **Opponent & market** tab.",
        sub="the calendar behind every weekly number on this page",
    )
    all_teams = sorted(ui.board(view)["team"].unique().to_list())
    away_in = {t: None for t in all_teams}
    for w in every:
        for t in ui.byes(view, w):
            away_in[t] = w
    byes = pl.DataFrame({"team": list(away_in), "bye_week": [away_in[t] for t in away_in]},
                        schema={"team": pl.String, "bye_week": pl.Int32})
    counted = byes.drop_nulls("bye_week").group_by("bye_week").agg(
        pl.len().alias("teams"), pl.col("team").sort().str.join(", ").alias("who"),
    ).sort("bye_week")
    ui.tiles([
        {"name": "weeks with a bye in them", "value": counted.height, "digits": 0,
         "sub": f"of {len(every)} in the season"},
        {"name": "worst week", "value": None if counted.is_empty() else counted["teams"].max(),
         "digits": 0, "highlight": True,
         "sub": ("" if counted.is_empty()
                 else f"week {int(counted.sort('teams', descending=True).row(0, named=True)['bye_week'])}"
                      " — that many offences away at once")},
        {"name": "teams with no bye listed", "value": int(byes["bye_week"].is_null().sum()),
         "digits": 0, "sub": "a schedule problem, not a football one"},
    ])
    ui.week_bars(counted.select(pl.col("bye_week").alias("week"), "teams"), "teams", height=220)
    left, right = st.columns([2, 3])
    with left:
        ui.table(byes.sort(["bye_week", "team"], nulls_last=True), height=420)
    with right:
        ui.table(counted, height=420)

# --------------------------------------------------------------------------- #
# 2. one game: the projected score and both box scores
# --------------------------------------------------------------------------- #
with game_tab:
    if game_id is None:
        st.info("Pick a week with games in it and this tab opens on one of them.")
    else:
        ui.section(
            f"{away} at {home}" if not row.get("neutral") else f"{away} vs {home}",
            "The two implied points, the spread and the total are each blended against the market "
            "separately, so they will not reconcile to the last tenth — three numbers rather than one "
            "invented fourth. Implied points is the one that matters here: it is what every count in "
            "this game is scaled by.",
            sub=f"week {int(week)} · {row.get('gameday') or game_id.replace('_', ' ')}",
        )
        ui.tiles([
            {"name": f"{away} · away", "value": row["away_points"], "highlight": True,
             "sub": "implied points"},
            {"name": f"{home} · home", "value": row["home_points"], "highlight": True,
             "sub": "implied points"},
            {"name": "total", "value": row["total"], "sub": f"{row['favourite']} favoured"},
            {"name": f"{home} spread", "value": row["spread"], "signed": True,
             "sub": "line posted" if row["has_market"] else "the model on its own"},
        ])
        ui.chips(*[c for c in ("line posted" if row["has_market"] else "model only",
                               "neutral site" if row.get("neutral") else "",
                               f"on a bye: {', '.join(out)}" if out else "") if c])

        st.divider()
        ui.section(
            "Every game of the season, and how it is projected to end",
            "The schedule with the projection in it. `spread` is the home side's, so a negative number "
            "is the home team favoured, and `line posted` says whether the market or the model is doing "
            "the talking. Pick any row's week and fixture in the controls above to open it.",
        )
        scope = st.segmented_control("Show", ["This week", "The whole season"], default="This week",
                                     key="games:scope") or "This week"
        shown_games = slate if scope == "This week" else ui.game_index(view)
        ui.focus_table(
            shown_games.drop("game_id"),
            ["week", "game", "away", "away_points", "home", "home_points", "total", "spread",
             "has_market"],
            key=f"games:slate:{scope}", height=620,
            config={**ui.fixed(1, "away_points", "home_points", "total", "spread")},
        )

        st.divider()
        ui.section(
            "Scoring level by week",
            "A week with fewer games is a bye week, and a week with no lines posted is a week the model "
            "is projecting on its own — which is most of them until the season is close.",
        )
        weekly_games = (
            ui.game_index(view).group_by("week").agg(
                pl.len().alias("games"),
                pl.col("total").mean().alias("mean_total"),
                pl.col("has_market").sum().alias("lines_posted"),
            ).sort("week")
        )
        # week order, not rank order: this is a season's shape rather than a ranking, so `rank_bars`
        # would sort the schedule away
        st.bar_chart(weekly_games.select("week", "mean_total"), x="week", y="mean_total", height=240,
                     x_label="week", y_label="mean total")
        ui.table(weekly_games, height=240, config=ui.fixed(1, "mean_total"))

        st.divider()
        ui.section(
            "The two offences, as their rosters project them",
            "Every man's projected line for this game, added up per side. A sum over the roster rather "
            "than a second projection — which is exactly why it is worth putting next to implied "
            "points. If the two halves of this table stop looking like the same afternoon, something "
            "above it is the reason.",
        )
        tb = ui.team_box(view, game_id)
        stat_cols = [c for c in tb.columns if c in ui.ALL_STATS]
        ui.table(
            tb, height=140,
            config={**ui.stat_config(stat_cols),
                    **ui.fixed(1, "implied_points", "spread", "total", "fantasy_points")},
        )

        st.divider()
        # A game's box score is read a man at a time, so the roster is the list and his own afternoon is
        # the panel: the stat line as cards, and the ✎ on `chance he plays` for this game and no other,
        # which is the edit this surface exists for and which used to mean scrolling to another tab and
        # finding his row.
        ui.section(
            "One man's afternoon",
            "A weekly line is an expectation including availability, so a man at 60% to play is showing "
            "60% of a line. That is `chance he plays`, and it is why a bench player is not on zero — "
            "and it is the number to change when somebody is out: his work goes back into his own "
            "team's pool for this week, and the room absorbs it.",
        )
        box_filters = st.columns([2, 3, 2])
        side_pick = box_filters[0].segmented_control("Side", ui.game_teams(view, game_id),
                                                     default=away, key="games:box:side") or away
        with box_filters[1]:
            box_positions = ui.position_filter("games:box:pos")
        everyone = box_filters[2].toggle("Everybody", value=False, key="games:box:all",
                                         help="off, only men with a projected line in this game")

        BOX_LIST = ["player", "position", "depth_slot", "p_play", "fantasy_points", "edited"]
        pb = ui.player_box(view, game_id, side_pick, box_positions)
        if not everyone and "fantasy_points" in pb.columns:
            pb = pb.filter(pl.col("fantasy_points") >= 0.5)
        if pb.is_empty():
            st.info("Nobody on this roster is projected for anything in this game.")
        else:
            box_left, box_right = st.columns([4, 8], gap="medium")
            with box_left:
                box_man = ui.pick_from(
                    pb, key=f"games:box:list:{game_id}:{side_pick}", columns=BOX_LIST, height=520,
                    config={**ui.percent("p_play"), **ui.fixed(1, "fantasy_points"),
                            "edited": st.column_config.CheckboxColumn("✏️", width="small"),
                            "depth_slot": st.column_config.NumberColumn("slot", format="%d",
                                                                        width="small")},
                )
            with box_right:
                if box_man is None:
                    st.info("Pick a name.")
                else:
                    with st.container(border=True):
                        head, act = st.columns([8, 3], vertical_alignment="center")
                        head.markdown(f"### {box_man['player']} · {box_man['position']} {side_pick} — "
                                      f"week {int(week)}")
                        with act:
                            # the whole editor, opened on this week: `p_play` beside it is the one knob
                            # this surface exists for, and the rest of them are one click away rather
                            # than on another page
                            ui.edit_button(box_man["player_id"], key=f"games:bench:{game_id}",
                                           week=int(week), label_text="✎ Adjust this game",
                                           width="stretch")
                        ui.chips(
                            *[c for c in (box_man.get("status")
                                          if box_man.get("status") not in (None, "ACT") else "",
                                          f"slot {int(box_man['depth_slot'])}"
                                          if box_man.get("depth_slot") is not None else "",
                                          "carries an override" if box_man.get("edited") else "") if c]
                        )
                        knob = st.columns([3, 2])
                        with knob[0]:
                            ui.tiles([
                                {"name": "fantasy points", "value": box_man.get("fantasy_points"),
                                 "highlight": True, "sub": "this game"},
                                {"name": "chance he plays", "value": box_man.get("p_play"),
                                 "digits": 3, "percent": True, "sub": "0 is out, 1 is certain"},
                            ])
                        with knob[1]:
                            st.caption("Change his availability for this game alone")
                            ui.knob_popover(view, box_man["player_id"], "p_play",
                                            base=box_man.get("p_play"), week=int(week),
                                            key=f"games:box:play:{game_id}", digits=3)
                        ui.stat_blocks(box_man, box_man.get("position"))

        with st.expander("Both rosters as a table — every man, every column"):
            for team in ui.game_teams(view, game_id):
                where = f"at {home}" if team == away else f"hosting {away}"
                st.caption(f"**{team}** — {where}")
                full = ui.player_box(view, game_id, team, box_positions)
                if full.is_empty():
                    st.info("Nobody on this roster is projected for anything in this game.")
                    continue
                cols = [c for c in full.columns if c not in ("player_id", "edits")]
                ui.table(
                    full.select(cols), height=min(90 + 35 * full.height, 460),
                    config={**ui.stat_config([c for c in cols if c in ui.ALL_STATS]),
                            **ui.fixed(1, "fantasy_points"), **ui.percent("p_play")},
                )

        with st.expander("Do the box scores still add up to the two offences?"):
            ui.note(
                "The season-wide version of this is History's ⚖️ Share sums tab; this is the same "
                "question about the game on screen, and it is here because per-game editing is the one "
                "thing that could break it. A counted pool — targets, carries, dropbacks — must match "
                "to rounding whatever has been typed on the 🎛️ **Adjust this game** tab, because an "
                "edit lands *before* the division rather than after it. Yards and completions are a "
                "count times a rate and are deliberately not forced to match: a room of unusually "
                "efficient receivers really should project above the team's own yardage, and scaling "
                "that away would hide the disagreement instead of showing it."
            )
            rec = ui.game_reconcile(view, game_id)
            if rec.is_empty():
                st.info("No reconcilable columns in this game.")
            else:
                ui.table(rec, height=440,
                         config=ui.fixed(2, "from_the_players", "from_the_offence", "gap"))

# --------------------------------------------------------------------------- #
# 3. adjust this game
# --------------------------------------------------------------------------- #
with adjust_tab:
    if game_id is None:
        st.info("Pick a week with games in it and this tab opens on one of them.")
    else:
        # no scenario bar here on purpose: naming a scenario, opening a saved one and starting a new one
        # are in the sidebar of every page, every edit is autosaved as it is made, and the ✏️ Edits page
        # is the ledger of them. A second copy of those controls beside an editor is how a session came
        # to have two names for one set of edits.
        ui.section(
            "How much there is to divide",
            "Reach for this first. Moving an offence's dropbacks for one week scales every share-"
            "holder's count in proportion and leaves the division alone; moving one man's share moves "
            "work between teammates and leaves the team where it was. Edit `dropbacks` and the "
            "designed-run count follows it, so pace and mix cannot be left contradicting each other. "
            "The other sixteen games stay where the model had them.",
            sub=f"week {int(week)} · {away} and {home} · seventeen games, one of them",
        )
        env_fields = ui.game_env_fields(view)
        ui.grid(
            ui.game_env_sheet(view, game_id, env_fields),
            level="team", key_col="team", week_col="week", fields=env_fields,
            key=f"games:env:{game_id}", digits=2, height=140, hide=("week",),
        )

        st.divider()
        # Deliberately still a sheet. The box-score section is where one man is read and edited one
        # number at a time; this is the surface for a session that is really about typing -- a row of
        # men, a set of columns, tab across. Both exist because they are different jobs, not two
        # attempts at one.
        ui.section(
            "Who takes it",
            "Every man on one side of this game, as editable cells. The ✎ beside a number on the "
            "🏈 **One game** tab is the way to change one man with his evidence in front of you; this "
            "is the way to change twelve.",
            sub="a sheet, on purpose",
        )
        controls = st.columns([2, 3, 3])
        side = controls[0].radio("Side", ui.game_teams(view, game_id), horizontal=True,
                                 key="games:sheet:side")
        column_set = controls[1].selectbox("Columns", ui.SHEET_COLUMN_SETS, index=0,
                                           key="games:sheet:cols")
        with controls[2]:
            sheet_positions = ui.position_filter("games:sheet:pos")

        fields = ui.game_fields(column_set, sheet_positions)
        sheet = ui.game_sheet(view, game_id, side, fields, sheet_positions)
        if sheet.is_empty():
            st.info("Nobody on this roster matches those positions.")
        else:
            ui.note(
                f"**Week {int(week)} only.** Type in any editable cell and it applies to this game and "
                "no other. `chance he plays` is the one to reach for when a man is out: set it to 0 and "
                f"his work goes back into the {side} pool for this week — the room absorbs the carries, "
                "the quarterback queue promotes the next man for this week alone, and the team still "
                "runs what it was projected to run. The slot and the games played are one number for "
                "the whole season, so they are shown locked and edited on the Team page."
            )
            ui.grid(
                sheet, level="player", key_col="player_id", fields=fields, week_col="week",
                key=f"games:sheet:{game_id}:{side}", digits=3, height=620,
                hide=("player_id", "week"),
                base_frame=ui.game_sheet_baseline(view, game_id, side, fields, sheet_positions),
                require_base=True,
                config={**ui.fixed(1, "fantasy_points"),
                        **ui.fixed(2, "targets", "carries", "attempts"),
                        "edited": st.column_config.CheckboxColumn("✏️", help="he carries an override",
                                                                  width="small"),
                        "depth_slot": st.column_config.NumberColumn("slot", format="%d"),
                        "p_play": st.column_config.NumberColumn(
                            "chance he plays", min_value=0.0, max_value=1.0, step=0.05,
                            format="%.3f",
                            help="his availability in this game. 0 is out, 1 is certain to play. Only "
                                 "this game moves; his projected games for the season come from the "
                                 "Availability page.")},
            )

        with st.expander("What has been typed into this game"):
            prov = ui.provenance(view)
            mine = (prov.filter(pl.col("stage").is_in(["game", "game rates", "team", "season grain"]))
                    if not prov.is_empty() else prov)
            if mine.is_empty():
                st.info("Nothing yet. Every number in this game is the engine's own.")
            else:
                ui.table(
                    mine.select([c for c in ("stage", "level", "key", "field", "week", "mode", "value",
                                             "base_recorded", "base_now", "used_now", "rows",
                                             "applied", "reason") if c in mine.columns]),
                    height=360, digits=4,
                )
                ui.note(
                    "`stage` is where in the engine the edit landed: `game` is before the pool was "
                    "divided, `game rates` is between the counts and the stat line, `team` is the "
                    "offence's own volume. A row at `season grain` is an edit that named a week on a "
                    "field that only has one value for the whole season — reported rather than "
                    "silently obeyed, because a week cannot be singled out on it."
                )

        st.divider()
        # and what it did to the season, which was a tab of its own on a page that no longer exists. It
        # belongs beside the editor rather than one click from it: the whole argument for editing a game
        # is that the season follows, and that is a claim worth reading immediately after typing.
        ui.section(
            "The season, after the games you have edited",
            "The point of editing a game rather than a season is that the season follows on its own: "
            "it is the sum of the seventeen, and one of them has moved. This is the same board diff "
            "every other page reads, narrowed to the men in this game — so an edit worth a tenth of a "
            "point over a year reads as one, and an edit worth forty is visible as the claim about the "
            "whole season it actually is.",
            sub=f"{away} at {home}, week {int(week)}, seen from the season",
        )
        moved = ui.game_season_effect(view, game_id)
        if moved.is_empty():
            st.info("Nothing in this game has been edited, so no season total has moved.")
        else:
            ui.tiles([
                {"name": "players moved", "value": moved.height, "digits": 0, "highlight": True,
                 "sub": "season totals that are no longer the engine's"},
                {"name": "points moved", "value": float(moved["d_fantasy_points"].abs().sum())
                 if "d_fantasy_points" in moved.columns else None,
                 "sub": "added up without the signs"},
                {"name": "net points", "value": float(moved["d_fantasy_points"].sum())
                 if "d_fantasy_points" in moved.columns else None, "signed": True,
                 "sub": "what the offence gained or lost overall"},
                {"name": "biggest gain", "value": float(moved["d_fantasy_points"].max())
                 if "d_fantasy_points" in moved.columns else None, "signed": True, "tone": "up"},
                {"name": "biggest loss", "value": float(moved["d_fantasy_points"].min())
                 if "d_fantasy_points" in moved.columns else None, "signed": True, "tone": "down"},
            ])
            if "d_fantasy_points" in moved.columns:
                ui.rank_bars(moved.sort(pl.col("d_fantasy_points").abs(), descending=True).head(16),
                             "d_fantasy_points", "player", height=300, digits=1)
            cols = [c for c in ("player", "position", "team", "fantasy_points", "new_fantasy_points",
                                "d_fantasy_points", "games", "new_games", "d_games")
                    if c in moved.columns]
            ui.focus_table(
                moved.select(cols), cols, key=f"games:season:{game_id}", height=460,
                config=ui.fixed(1, *[c for c in cols if "points" in c or "games" in c]),
                label_text="Every man this game moved",
            )

        summary = ui.board_diff_summary(view)
        ui.section(
            "The whole board, not only this game",
            "`net` against `abs` is the reading worth making. A team edit that adds volume to a game "
            "moves both together; a share edit inside one receiving room shows a large `abs` and a "
            "`net` near zero, because the division took from the teammates what it gave to the player. "
            "The ✏️ Edits page has every edit in one list, and 📋 **The week's board** shows which week "
            "moved.",
            sub="every edit in the scenario, by position",
        )
        if summary.is_empty():
            st.info("The scenario is the engine's own answer.")
        else:
            ui.table(summary, height=240)

# --------------------------------------------------------------------------- #
# 4. the opponent and the market: the schedule as an input rather than a frame
# --------------------------------------------------------------------------- #
with market_tab:
    # every team's season in one row, which three of the four sections below are read off
    lead = (
        env_all.group_by("team").agg(
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

    ui.section(
        "The schedule, as the projection reads it",
        "A game's scoring level and an opponent's history against a position are two different claims. "
        "The first one moves the projection — it scales plays, red-zone trips and touchdowns. The "
        "second is here to explain a number rather than to build one: one season of "
        "defence-versus-position is 17 games and it is noisy.",
        sub=f"{env_all['team'].n_unique()} teams · {env_all.height} team-games · "
            f"{int(env_all['has_market'].sum())} with a posted line",
    )
    ui.tiles([
        {"name": "mean implied points", "value": env_all["implied_points"].mean(),
         "sub": "the league's own scoring level"},
        {"name": "highest scoring offence", "value": lead.row(0, named=True)["implied_pg"],
         "highlight": True, "sub": f"{lead.row(0, named=True)['team']} a game"},
        {"name": "most plays",
         "value": lead.sort("plays_pg", descending=True).row(0, named=True)["plays_pg"],
         "sub": f"{lead.sort('plays_pg', descending=True).row(0, named=True)['team']} a game"},
        {"name": "games with a line", "value": int(env_all["has_market"].sum()), "digits": 0,
         "sub": f"of {env_all.height} — the rest are the model's own"},
    ])

    st.divider()
    # a matrix on purpose: thirty-two rows by seventeen columns is the shape of a season, and no panel
    # says "three of their first five are away" as plainly as reading across the row
    ui.section("Every team, every week",
               "`@` is away and a missing week is the bye. `implied points` is the blended scoring level "
               "for that team in that game — the number every count in the projection is scaled by.",
               sub="a matrix, on purpose")
    metric = st.segmented_control(
        "Cell", ["opponent", "implied_points", "total", "spread", "plays", "pass_attempts", "carries"],
        default="opponent", key="match:cell", format_func=ui.label) or "opponent"
    if metric == "opponent":
        cell = pl.when(pl.col("is_home")).then(pl.col("opponent")) \
            .otherwise(pl.lit("@ ") + pl.col("opponent"))
        grid = env_all.select("team", "week", cell.alias("v")).pivot(on="week", index="team",
                                                                    values="v")
        ui.table(grid.sort("team"), height=640)
    else:
        grid = env_all.select("team", "week", metric).pivot(on="week", index="team", values=metric)
        grid_weeks = [c for c in grid.columns if c != "team"]
        ui.table(grid.sort("team"), config=ui.fixed(1, *grid_weeks), height=640)

    st.divider()
    ui.section("What kind of games each team plays",
               "`seconds per play` is pace: a low number is more snaps for the same number of drives. "
               "Volume and scoring level are separate things and a team can be high in one and low in "
               "the other, which is why they are two meters rather than one ranking.",
               sub="click a team; its seventeen games are on the right")
    TEAM_LIST = ["team", "implied_rank", "implied_pg", "plays_pg", "pass_rate", "games_with_a_line"]
    team_left, team_right = st.columns([5, 7], gap="medium")
    with team_left:
        team_row = ui.pick_from(
            lead, key="match:team:list", columns=TEAM_LIST, height=600,
            config={**ui.fixed(1, "implied_pg", "plays_pg"), **ui.percent("pass_rate"),
                    "implied_rank": st.column_config.NumberColumn("rank", format="%d",
                                                                  width="small"),
                    "games_with_a_line": st.column_config.NumberColumn("lines", format="%d",
                                                                       width="small")},
        )
    with team_right:
        if team_row is None:
            st.info("Pick a team.")
        else:
            with st.container(border=True):
                t = team_row["team"]
                st.markdown(f"### {t} — the season they are projected to play")
                ui.chips(f"{int(team_row['implied_rank'])} in implied points",
                         f"{int(team_row['volume_rank'])} in plays",
                         f"{int(team_row['games_with_a_line'])} games with a posted line")
                ui.tiles([
                    {"name": "implied points a game", "value": team_row["implied_pg"],
                     "highlight": True, "sub": f"league mean {lead['implied_pg'].mean():.1f}"},
                    {"name": "plays a game", "value": team_row["plays_pg"],
                     "sub": f"league mean {lead['plays_pg'].mean():.1f}"},
                    {"name": "pass rate", "value": team_row["pass_rate"], "digits": 3,
                     "percent": True, "sub": "dropbacks over plays"},
                    {"name": "seconds a play", "value": team_row["seconds_per_play"],
                     "sub": "pace — lower is more snaps"},
                ])
                ui.meters([
                    ui.meter_html("implied points a game", team_row["implied_pg"],
                                  maximum=float(lead["implied_pg"].max()) * 1.1, digits=1,
                                  marks=(("league mean", lead["implied_pg"].mean()),),
                                  foot=(f"ranked {int(team_row['implied_rank'])} of {lead.height}",)),
                    ui.meter_html("plays a game", team_row["plays_pg"],
                                  maximum=float(lead["plays_pg"].max()) * 1.1, digits=1,
                                  marks=(("league mean", lead["plays_pg"].mean()),),
                                  foot=(f"ranked {int(team_row['volume_rank'])} of {lead.height}",)),
                    ui.meter_html("pass rate", team_row["pass_rate"], maximum=1.0, digits=3,
                                  percent=True,
                                  marks=(("league mean", lead["pass_rate"].mean()),),
                                  foot=("how the volume splits before anybody's share of it",)),
                ])
                his_games = env_all.filter(pl.col("team") == t).sort("week")
                ui.week_bars(his_games.select("week", "implied_points"), "implied_points", height=200)
                ui.focus_table(
                    his_games.select([c for c in ("week", "opponent", "is_home", "rest_days",
                                                 "div_game", "roof", "spread", "total",
                                                 "implied_points", "has_market", "plays", "dropbacks",
                                                 "carries", "targets") if c in his_games.columns]),
                    ["week", "opponent", "is_home", "spread", "total", "implied_points", "plays",
                     "has_market"],
                    key=f"match:team:games:{t}", height=420, digits=1,
                    note_text=f"Weeks missing from this table are the bye: {t} plays "
                              f"{his_games.height} games.",
                    label_text="the roof, the rest and the volume too",
                )

    with st.expander("Every team as a table"):
        ui.table(
            lead,
            config={**ui.fixed(2, "implied_pg", "total_pg", "spread_pg", "plays_pg", "dropbacks_pg",
                               "carries_pg", "seconds_per_play"),
                    **ui.percent("pass_rate")},
            height=620,
        )

    st.divider()
    ui.section(
        f"Defence against each position, {LAST_COMPLETE_SEASON}",
        f"Measured over {LAST_COMPLETE_SEASON} only, so a points factor of 1.20 is one season of 17 "
        "games against that position — informative, not decisive. It is not what the engine uses; the "
        "chain reads the league-normalised season ratings instead. The tick on every bar is 1.00, which "
        "is the league itself.",
        sub="history, not a projection",
    )
    dbp = ui.defence_by_position()
    position = st.segmented_control("Position", list(ui.POSITIONS), default="WR",
                                    key="match:dpos") or "WR"
    faced = dbp.filter(pl.col("position") == position).select(
        pl.col("defense").alias("team"), "games", "targets_pg", "carries_pg", "fantasy_points_pg",
        "yards_per_target_allowed", "yards_per_carry_allowed", "catch_rate_allowed", "tds_allowed_pg",
        pl.col("fantasy_points_pg__factor").alias("points_factor"),
        pl.col("targets_pg__factor").alias("targets_factor"),
    ).sort("fantasy_points_pg", descending=True)
    if faced.is_empty():
        st.info(f"No {LAST_COMPLETE_SEASON} defensive table for {position}s in the lake.")
    else:
        softest, hardest = faced.head(5), faced.tail(5).reverse()
        ui.tiles([
            {"name": f"softest against {position}s",
             "value": softest.row(0, named=True)["points_factor"],
             "digits": 2, "highlight": True, "tone": "up",
             "sub": f"{softest.row(0, named=True)['team']} · "
                    f"{softest.row(0, named=True)['fantasy_points_pg']:.1f} points a game allowed"},
            {"name": f"hardest against {position}s",
             "value": hardest.row(0, named=True)["points_factor"],
             "digits": 2, "tone": "down",
             "sub": f"{hardest.row(0, named=True)['team']} · "
                    f"{hardest.row(0, named=True)['fantasy_points_pg']:.1f} points a game allowed"},
            {"name": "league mean allowed", "value": faced["fantasy_points_pg"].mean(),
             "sub": f"points a game to {position}s"},
        ])
        ui.section("Points allowed a game, ranked",
                   sub=f"{LAST_COMPLETE_SEASON} · thirty-two defences, softest first")
        ui.rank_bars(faced, "fantasy_points_pg", "team", height=560, digits=1, top=32)
        ui.focus_table(
            faced,
            ["team", "games", "fantasy_points_pg", "points_factor", "targets_pg", "targets_factor",
             "tds_allowed_pg", "catch_rate_allowed"],
            key=f"match:def:{position}", height=520, digits=2,
            config={**ui.fixed(2, "targets_pg", "carries_pg", "fantasy_points_pg", "tds_allowed_pg",
                               "yards_per_target_allowed", "yards_per_carry_allowed"),
                    **ui.percent("catch_rate_allowed"),
                    **ui.fixed(3, "points_factor", "targets_factor")},
            label_text="the yardage rates too",
        )

    st.divider()
    ui.section(
        "Softest and hardest weeks",
        f"Every team-week joined to what its opponent allowed to {position}s in {LAST_COMPLETE_SEASON}. "
        "Sorted by the opponent's fantasy points factor, with that game's own scoring level beside it — "
        "the two reasons a week can be a good one, kept apart so you can see which is doing the work. "
        "The position is the one picked just above.",
        sub=f"against {position}s · pick a team to narrow it",
    )
    allowed = dbp.filter(pl.col("position") == position).select(
        pl.col("defense").alias("opponent"),
        pl.col("fantasy_points_pg__factor").alias("points_factor"),
        pl.col("targets_pg__factor").alias("targets_factor"),
        pl.col("fantasy_points_pg").alias("opp_points_allowed_pg"),
    )
    team_weeks = env_all.select("team", "week", "opponent", "is_home", "implied_points", "total",
                                "spread").join(allowed, on="opponent", how="left")
    picked_team = st.selectbox("Team", ["— every team —"] + sorted(env_all["team"].unique().to_list()),
                              key="match:weeks:team")
    scope_weeks = (team_weeks if picked_team.startswith("—")
                   else team_weeks.filter(pl.col("team") == picked_team))
    cfg = {**ui.fixed(3, "points_factor", "targets_factor"),
           **ui.fixed(1, "implied_points", "total", "spread", "opp_points_allowed_pg")}
    COLS = ["team", "week", "opponent", "is_home", "points_factor", "implied_points", "total",
            "opp_points_allowed_pg"]
    soft_col, hard_col = st.columns(2, gap="medium")
    with soft_col:
        st.caption("**Softest** — the opponent allowed the most to this position last season")
        ui.focus_table(scope_weeks.sort("points_factor", descending=True, nulls_last=True).head(20),
                       COLS, key=f"match:soft:{picked_team}:{position}", height=520, digits=2,
                       config=cfg)
    with hard_col:
        st.caption("**Hardest** — and these are the weeks to look at a bench")
        ui.focus_table(scope_weeks.sort("points_factor", nulls_last=True).head(20), COLS,
                       key=f"match:hard:{picked_team}:{position}", height=520, digits=2, config=cfg)
