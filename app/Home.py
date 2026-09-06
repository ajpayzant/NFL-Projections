"""The board: every projected player, ranked, with the comparisons a draft is decided on.

The list leads with the football rather than with the scoring, because that is the claim being reviewed:
a projection says a receiver catches 68 of 98 for 912 and six, and the points are an arithmetic
consequence of it. Filtered to one position the line is sortable columns; across the whole board it is
one string per row, since no set of stat columns is honest for a quarterback and a slot receiver at once.
The scoring is still there at the end -- points, per game, points above the average starter, and the drop
to the next man, which is what answers *how much* once the football has been read.

That is what the panel beside it is for. Click a row and everything about that player arrives next to
the list: his line, the estimates the projection is made of drawn as meters, his weeks, his own record,
and a ✎ on every number that can be argued with. The board used to be a forty-four-column table whose
right-hand half nobody had ever scrolled to; the columns are all still reachable -- **All columns** on
the second tab, or the Exports page at full precision -- but nothing is the default that cannot be read.

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
ui.page_map()

board = ui.board(view)

# The list leads with the football. A projection is a claim about carries, targets, yards and touchdowns
# before it is a claim about points, and reviewing one means reading that claim -- so the scoring is the
# columns at the end rather than the thing the page opens on.
#
# What the football looks like depends on the filter, because there is no set of stat columns that is
# honest for a quarterback and a slot receiver at the same time. One position selected gets its own box
# score as sortable numbers; the whole board gets each man's line as a string, which is position-correct
# for every row and has no blank cells in it.
# Six columns at most, and they are the six a football argument is actually had over. The rest of the box
# score is one tab across, on **All columns**, and all of it is in the panel beside the list.
HEADLINES = {
    "QB": ("games", "attempts", "passing_yards", "passing_tds", "interceptions", "rushing_yards"),
    "RB": ("games", "carries", "rushing_yards", "rushing_tds", "targets", "receiving_yards"),
    "WR": ("games", "targets", "receptions", "receiving_yards", "receiving_tds"),
    "TE": ("games", "targets", "receptions", "receiving_yards", "receiving_tds"),
}
SCORING = ["fantasy_points", "points_per_game", "if_healthy", "vs_starter", "drop_next"]

FANTASY = ["overall_rank", "position_rank", "tier", "player", "team", "position", "games",
           "fantasy_points", "points_per_game", "if_healthy", "vs_starter", "drop_next",
           "delta_points"]

# With a scenario live, the board carries what the edits did to each player -- `vs_baseline` points and
# the position ranks he gained -- because a ranking is only worth as much as the reason it moved.
if not view.scenario.is_baseline:
    d = ui.board_diff(view).select(
        "player_id",
        pl.col("d_fantasy_points").alias("vs_baseline"),
        (pl.col("position_rank") - pl.col("new_position_rank")).alias("ranks_gained"),
    )
    board = board.join(d, on="player_id", how="left")
    FANTASY = FANTASY[:-1] + ["vs_baseline", "ranks_gained", "delta_points"]
    SCORING = SCORING + ["vs_baseline"]

board = ui.with_edits(board, "player", "player_id")

# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #
left, mid, right, far = st.columns([2, 2, 2, 2])
with left:
    positions = ui.position_filter()
with mid:
    teams = ui.team_filter(view)
with right:
    search = st.text_input("Search", placeholder="part of a name")
with far:
    only_startable = st.toggle(
        "Startable only", value=False,
        help=f"Inside the position's starter count: {view.settings.starters}",
    )
    hide_gone = ui.roster_filter("board:gone")

# now the filter is known, so the list can be the football first: one position's own box score, or every
# position's line as a string
one = positions[0] if len(positions) == 1 else ""
line = [c for c in HEADLINES.get(one, ()) if c in board.columns] or ["games", "line"]
LIST = ["overall_rank", "position_rank", "player", *([] if one else ["position"]), "team",
        *line, *SCORING]

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
    FANTASY = FANTASY[:1] + ["median_rank", "ceiling_rank"] + FANTASY[1:-1] + RANGE + [FANTASY[-1]]
    LIST = LIST[:2] + ["median_rank"] + LIST[2:] + ["volatility"]

shown = ui.apply_filters(board, positions, teams, search, hide_gone=hide_gone)
if only_startable:
    shown = shown.filter(pl.col("startable"))
if "line" in LIST:                      # only the rows on screen pay for the string
    shown = ui.with_stat_line(shown)

ui.tiles([
    {"name": "players projected", "value": board.height, "digits": 0},
    {"name": "shown", "value": shown.height, "digits": 0, "sub": "after filters"},
    {"name": "points on the board", "value": board["fantasy_points"].sum(), "digits": 0},
    {"name": "changed team", "value": (board.filter(pl.col("changed_team")).height
                                       if "changed_team" in board.columns else None), "digits": 0},
    {"name": "released", "value": (board.filter(pl.col("status").is_in(list(ui.OFF_ROSTER))).height
                                   if "status" in board.columns else None), "digits": 0,
     "sub": "off the team, worth zero"},
    {"name": "carrying an override", "value": board.filter(pl.col("edited")).height, "digits": 0,
     "highlight": bool(board.filter(pl.col("edited")).height)},
])

TABS = ["🏈 The board", "📋 All columns", "📉 Where each position runs out", "🥇 Stat leaders",
        "🧱 Tiers", "↕️ Movers"]
board_tab, columns_tab, curve_tab, leaders_tab, tiers_tab, movers_tab = st.tabs(TABS)

# --------------------------------------------------------------------------- #
# the board, as a list with a player beside it
# --------------------------------------------------------------------------- #
with board_tab:
    if shown.is_empty():
        st.info("No players match those filters.")
    else:
        listing, panel = st.columns([7, 5] if one else [6, 6], gap="medium")
        with listing:
            sortable = ("Sortable numbers, because one position can be compared on them."
                        if one else
                        "Pick a single position in the filter above and the line becomes sortable "
                        "columns.")
            ui.section(
                "Ranked",
                "The projection first — what he is expected to *do* — and the points it converts to at "
                f"the end. {sortable} Click any row and the panel fills with that player; the ✏️ column "
                "says who carries an override.",
            )
            picked = ui.pick_from(
                shown, key="board:list", columns=[*LIST, "edited"], height=760,
                config={**ui.stat_config(),
                        **ui.fixed(1, "fantasy_points", "vs_starter", "drop_next", "vs_baseline",
                                   "games", "if_healthy"),
                        **ui.fixed(2, "points_per_game"),
                        **ui.range_config(),
                        "line": st.column_config.TextColumn("projected line", width="large"),
                        "edited": st.column_config.CheckboxColumn("✏️", width="small"),
                        "overall_rank": st.column_config.NumberColumn("#", format="%d", width="small"),
                        "position_rank": st.column_config.NumberColumn("pos #", format="%d",
                                                                       width="small")},
            )
        with panel:
            if picked is None:
                st.info("Pick a player.")
            else:
                with st.container(border=True):
                    ui.player_panel(view, picked, key="board:panel", compact=True)
                    try:
                        st.page_link("pages/3_Player.py", label="His whole page →", icon="👤")
                    except Exception:              # noqa: BLE001
                        st.caption("Player page")

# --------------------------------------------------------------------------- #
# the whole frame, for the reader who wants every column
# --------------------------------------------------------------------------- #
with columns_tab:
    VIEWS = ("Stat line", "Fantasy", "Everything")
    row = st.columns([3, 4, 2])
    with row[0]:
        # the football is the default here for the same reason it leads the list: the points are a
        # conversion of it, and the conversion is not the thing being reviewed
        columns_mode = st.segmented_control(
            "Columns", VIEWS, default="Stat line", key="board:mode",
            help="Stat line is the projected football line for the positions on screen — attempts, "
                 "yards, touchdowns — with the points beside it rather than instead of it.",
        ) or "Stat line"
    with row[1]:
        extra = st.multiselect(
            "Extra columns",
            [c for c in (*ui.ALL_STATS, "depth_slot", "slot_bucket", "status", "is_rookie",
                         "draft_pick", "startable", "last_team", "last_games", "last_points")
             if c not in FANTASY], default=[])
    with row[2]:
        limit = st.number_input("Rows", min_value=25, max_value=1000, value=200, step=25)

    if columns_mode == "Stat line":
        want = ["overall_rank", "position_rank", "player", "team", "position", "games",
                *ui.stat_columns(positions), "fantasy_points", "points_per_game"]
    elif columns_mode == "Everything":
        want = list(shown.columns)
    else:
        want = list(FANTASY)
    order = [c for c in dict.fromkeys([*want, *extra]) if c in shown.columns] or None

    ui.table(
        shown.head(int(limit)), height=620, order=order,
        config={**ui.stat_config(),
                **ui.fixed(1, "fantasy_points", "vs_starter", "drop_next", "delta_points",
                           "last_points", "vs_baseline", "if_healthy"),
                **ui.fixed(2, "points_per_game"),
                **ui.fixed(1, "games", "last_games"),
                **ui.range_config()},
    )
    ui.note(
        "`vs_starter` is points above the mean of the startable players at that position; `drop_next` is "
        "the gap to the next man at the same position; `delta_points` is against what he actually scored "
        "last season under the same scoring. Blank `delta_points` means he did not play then. Every count "
        "is already availability-weighted — a receiver expected to miss two games is projected for "
        "fifteen games of targets, not seventeen with a discount applied afterwards. `if_healthy` is the "
        "same projection over a full seventeen, which is the number to hold a career line against: the "
        "distance between it and `fantasy_points` is what availability cost him rather than a lower "
        "opinion of him."
        + ("" if view.scenario.is_baseline else
           " `vs_baseline` and `ranks_gained` are this scenario against the same season with no edits.")
    )

    if sim is not None:
        ui.section("Who the range disagrees about",
                   "Players whose median rank is furthest from where the mean projection puts them. A "
                   "player above the line is one the mean flatters — his projection is carried by a "
                   "ceiling he rarely reaches; one below is the opposite, and is usually the safer pick "
                   "at the same cost.")
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
        safe = shown.filter(pl.col("startable") & pl.col("volatility").is_not_null())
        low, high = st.columns(2)
        with low:
            ui.section("Safest startable seasons", sub="lowest volatility")
            ui.rank_bars(safe.sort("volatility").head(15), "volatility", height=380, digits=3)
        with high:
            ui.section("Highest ceilings", sub="most volatile")
            ui.rank_bars(safe.sort("volatility", descending=True).head(15), "volatility",
                         height=380, digits=3, colour="#e45756")

# --------------------------------------------------------------------------- #
# position shape
# --------------------------------------------------------------------------- #
with curve_tab:
    ui.section("Where each position runs out",
               "Projected points by position rank. The knee in a curve is the round the position stops "
               "being worth reaching for.")
    curve = (
        board.filter(pl.col("position_rank") <= 48)
        .select("position_rank", "position", "fantasy_points")
        .pivot(on="position", index="position_rank", values="fantasy_points")
        .sort("position_rank")
    )
    st.line_chart(curve, x="position_rank", height=320)

    by_pos = board.group_by("position").agg(
        pl.len().alias("players"),
        pl.col("fantasy_points").sum().alias("points"),
        pl.col("fantasy_points").mean().alias("mean_points"),
        pl.when(pl.col("startable")).then(pl.col("fantasy_points")).otherwise(None)
        .mean().alias("mean_starter"),
        pl.col("fantasy_points").max().alias("best"),
    ).sort("points", descending=True)
    ui.table(by_pos, config=ui.fixed(1, "points", "mean_points", "mean_starter", "best"))

# --------------------------------------------------------------------------- #
# stat leaders
# --------------------------------------------------------------------------- #
with leaders_tab:
    # A projection is a claim about football before it is a claim about points, so the board is worth
    # reading one stat at a time: who catches the most passes, who carries it most, who throws for most.
    choices = [c for c in ui.stat_columns(positions) if c in shown.columns]
    if not choices:
        st.info("No stat columns for the positions selected.")
    else:
        lead = st.columns([2, 2, 4])
        stat = lead[0].selectbox("Stat", choices, index=0, format_func=ui.label)
        count = lead[1].number_input("How many", min_value=5, max_value=40, value=20, step=5)
        ui.section(f"Most projected {ui.label(stat)}", sub="filters applied")
        top = (shown.filter(pl.col(stat).is_not_null() & (pl.col(stat) > 0))
               .sort(stat, descending=True).head(int(count)))
        ui.rank_bars(top, stat, height=max(240, 26 * top.height), top=int(count))
        ui.note("The bar is the comparison the number alone does not make: a gap of ten between first "
                "and second is a different board from a flat top ten.")

# --------------------------------------------------------------------------- #
# tiers
# --------------------------------------------------------------------------- #
with tiers_tab:
    ui.section("Tiers", f"Blocks of {view.settings.tier_size} within a position, in projected order. "
                        "Two players in the same tier are the same pick.")
    tiers = (
        ui.apply_filters(board, positions, teams, "", hide_gone=hide_gone)
        .filter(pl.col("tier") <= 8)
        .group_by(["position", "tier"]).agg(
            pl.len().alias("n"),
            pl.col("fantasy_points").max().alias("top"),
            pl.col("fantasy_points").min().alias("bottom"),
            pl.col("player").sort_by("fantasy_points", descending=True).str.join(", ").alias("players"),
        ).sort(["position", "tier"])
    )
    ui.table(tiers, config=ui.fixed(1, "top", "bottom"), height=520)

# --------------------------------------------------------------------------- #
# movers
# --------------------------------------------------------------------------- #
with movers_tab:
    if "delta_points" not in board.columns:
        st.info("No prior season available to compare against.")
    else:
        ui.section("Against last season",
                   "Read the falls with the games column beside them: most of the largest are players "
                   "who played 17 games last season and are projected for fewer, or who were last "
                   "season's outliers. A projection regresses an outlier by construction.")
        movers = board.filter(pl.col("delta_points").is_not_null()).select(
            "player", "position", "team", "last_team", "changed_team", "last_games", "games",
            "last_points", "fantasy_points", "delta_points",
        )
        up, down = st.columns(2)
        with up:
            st.caption("Biggest projected gains")
            ui.rank_bars(movers.sort("delta_points", descending=True).head(15), "delta_points",
                         height=400, digits=1)
        with down:
            st.caption("Biggest projected falls")
            ui.rank_bars(movers.sort("delta_points").head(15), "delta_points", height=400,
                         digits=1, colour="#e45756")
        ui.table(movers.sort("delta_points", descending=True), height=420,
                 config=ui.fixed(1, "last_points", "fantasy_points", "delta_points", "games",
                                 "last_games"))
