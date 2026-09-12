"""The whole league at once: projected records, all thirty-two offences, and every game.

This is the page a projection is checked from. Every other surface in the app starts from a player and
works upward, and that direction cannot answer the first question a football person asks -- *does this
team come out where this roster should?* A five-win projection on a contender's roster is wrong
somewhere, and no per-player table says so; a division table says it in one line.

Nothing here is a second model of the season. A projected record is three steps off the same
`implied_points` the weekly pages read (`src.model.standings`), and the team totals are the same board
the player pages read, summed. So a number on this page and the man behind it can never disagree, and an
edit made anywhere lands here.

The last tab is the only thing in the app about the *user* rather than the season. Going team by team is
thirty-two sittings, and the question between sittings is which ones are done -- read off the edits
themselves, so there is nothing extra to keep in step.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl                                                          # noqa: E402
import streamlit as st                                                       # noqa: E402

import ui                                                                    # noqa: E402
from src.model import standings as model                                     # noqa: E402

view = ui.controls("League", icon="🏆")
ui.scenario_banner(view)

table = ui.standings(view)
teams = ui.team_projections(view)
games = ui.game_grid(view)
fit = ui.standings_accuracy()

ui.section(
    f"NFL {view.season}",
    "A projected record is a sum of seventeen win probabilities, not a count of games won: a team "
    "favoured in eleven games by three points apiece has not won eleven of them. The width of the error "
    f"turning a margin into a probability is measured rather than assumed — {fit['sigma']:.1f} points, "
    f"fitted on {int(fit['games']):,} played games — and it is wide, which is why a 9.4-7.6 should be "
    "read as somewhere between six and twelve.",
    sub=f"{table.height} teams · {games.height // 2} games · σ {fit['sigma']:.1f} points",
    level=2,
)
best = table.sort("expected_wins", descending=True)
ui.tiles([
    {"name": "best record", "value": best["expected_wins"].max(), "digits": 1, "highlight": True,
     "sub": f"{best.row(0, named=True)['team']} — {best.row(0, named=True)['expected_wins']:.1f}-"
            f"{best.row(0, named=True)['expected_losses']:.1f}"},
    {"name": "worst record", "value": best["expected_wins"].min(), "digits": 1,
     "sub": f"{best.row(-1, named=True)['team']} — {best.row(-1, named=True)['expected_wins']:.1f}-"
            f"{best.row(-1, named=True)['expected_losses']:.1f}"},
    {"name": "points / game", "value": table["points_per_game"].mean(), "digits": 1,
     "sub": f"{table['points_per_game'].min():.1f} to {table['points_per_game'].max():.1f}"},
    {"name": "teams over .500", "value": int((table["expected_wins"] > 8.5).sum()), "digits": 0,
     "sub": "of 32 — the rest are under it"},
    {"name": "win model, when scored", "value": fit["straight_up"], "digits": 3, "percent": True,
     "sub": f"picked straight up · Brier {fit['brier']:.3f}"},
])

TABS = ["🏆 Standings", "📊 All 32 offences", "🗓️ Every game", "💰 Against the posted lines",
        "✅ How far through the league"]
standings_tab, teams_tab, games_tab, market_tab, coverage_tab = st.tabs(TABS)

# --------------------------------------------------------------------------- #
# 1. the standings
# --------------------------------------------------------------------------- #
# Eight small tables rather than one of thirty-two rows, because a division is the unit a record is read
# in: nine wins is a division title in one of them and third place in another, and a single sorted list
# of the league cannot show that.
RECORD_COLS = ["team", "record", "expected_wins", "division_wins", "points_per_game",
               "allowed_per_game", "point_margin"]
RECORD_CONFIG = {
    "record": st.column_config.TextColumn("W-L", width="small"),
    "expected_wins": st.column_config.NumberColumn("exp W", format="%.2f", width="small"),
    "division_wins": st.column_config.NumberColumn("div W", format="%.2f", width="small"),
    "points_per_game": st.column_config.NumberColumn("PF/g", format="%.1f", width="small"),
    "allowed_per_game": st.column_config.NumberColumn("PA/g", format="%.1f", width="small"),
    "point_margin": st.column_config.NumberColumn("season margin", format="%+.0f"),
}


def with_record(df: pl.DataFrame) -> pl.DataFrame:
    """The rounded record as a string beside the fractional one, because both are wanted at once."""
    return df.with_columns(
        (pl.col("expected_wins").round(1).cast(pl.String) + "-"
         + pl.col("expected_losses").round(1).cast(pl.String)).alias("record"),
    )


with standings_tab:
    ui.section(
        "By division",
        "Sorted on expected wins, and there are no tiebreakers here on purpose: expected wins are "
        "continuous so two teams are never actually tied, and a projected 9.4-7.6 does not carry the "
        "head-to-head and common-games record a real tiebreaker needs. `div W` is how many of the six "
        "in-division games a team is projected to take, which is the part of a record that decides "
        "anything.",
        sub="the projection, read the way a standings page is read",
    )
    shown = with_record(table)
    for conference in ("AFC", "NFC"):
        st.markdown(f"#### {conference}")
        columns = st.columns(4, gap="medium")
        for i, name in enumerate(d for d in model.DIVISIONS if d.startswith(conference)):
            block = shown.filter(pl.col("division") == name).sort("expected_wins", descending=True)
            with columns[i]:
                st.caption(name)
                ui.table(block.select([c for c in RECORD_COLS if c in block.columns]),
                         height=180, config=RECORD_CONFIG, auto=False)

    st.divider()
    ui.section("The same records as one list",
               "Ranked across the league rather than inside a division, which is the order a wild card "
               "is decided in — and the order that says whether a division is soft or the team is good.",
               sub="click a team to work on it")
    side = st.columns([7, 5], gap="medium")
    with side[0]:
        conf = st.radio("Conference", ["Both", "AFC", "NFC"], horizontal=True, key="league:conf")
        listing = shown if conf == "Both" else shown.filter(pl.col("conference") == conf)
        listing = listing.sort("expected_wins", descending=True)
        picked = ui.pick_from(
            listing, key=f"league:standings:{conf}",
            columns=["team", "division", "record", "expected_wins", "points_per_game",
                     "allowed_per_game", "point_margin"],
            height=520, config=RECORD_CONFIG,
        )
    with side[1]:
        if picked is None:
            st.info("No teams.")
        else:
            code = str(picked["team"])
            with st.container(border=True):
                st.markdown(f"### {code} · {picked['division']}")
                ui.tiles([
                    {"name": "record", "value": picked["expected_wins"], "digits": 2,
                     "highlight": True, "sub": str(picked["record"])},
                    {"name": "points / game", "value": picked["points_per_game"], "digits": 1,
                     "sub": f"allowing {picked['allowed_per_game']:.1f}"},
                    {"name": "in the division", "value": picked["division_wins"], "digits": 2,
                     "sub": "of six games"},
                ])
                ui.work_on_team(code, key="league:standings:open")
                his = games.filter(pl.col("team") == code).sort("week")
                st.caption("His seventeen games — the bar is the chance of winning each")
                ui.week_bars(his.select("week", pl.col("win_prob").alias("chance")), "chance",
                             height=200)
                ui.table(
                    his.select("week", "opponent", "implied_points", "implied_points_opp", "margin",
                               "win_prob"),
                    height=380,
                    config={
                        "implied_points": st.column_config.NumberColumn("them", format="%.1f"),
                        "implied_points_opp": st.column_config.NumberColumn("opp", format="%.1f"),
                        "margin": st.column_config.NumberColumn("margin", format="%+.1f"),
                        "win_prob": st.column_config.ProgressColumn(
                            "chance", format="%.2f", min_value=0.0, max_value=1.0),
                    },
                    auto=False,
                )

    ui.section("Expected wins, ranked", sub="the whole league")
    ui.rank_bars(shown.sort("expected_wins", descending=True), "expected_wins", "team", height=520,
                 digits=2)
    ui.note("Because both sides of a game are one probability between them, these thirty-two numbers "
            f"add to exactly half the games played: {table['expected_wins'].sum():.0f} of "
            f"{int(table['games'].sum())}. An edit that moves one team up has moved others down.")
    st.caption("A record here follows one number: the team's implied points, week by week. So an edit to "
               "how much an offence *does* — its plays, its dropbacks, who takes the targets — changes "
               "every projection on the board and leaves the record where it was. Say a team will score "
               "more, on its Team page's team-volume tab, and the record moves.")

# --------------------------------------------------------------------------- #
# 2. all 32 offences
# --------------------------------------------------------------------------- #
# The team-level version of the board: the same totals the player pages sum to, so this is where a
# team's whole projection is sanity-checked in one row rather than thirty.
TEAM_SETS = {
    "Scoring": ["team", "division", "expected_wins", "points_for", "points_against", "point_margin",
                "points_per_game", "allowed_per_game", "fantasy_points"],
    "Volume": ["team", "division", "team_plays", "team_dropbacks", "team_pass_attempts",
               "team_carries", "team_dropback_rate", "targets", "carries", "players"],
    "Passing": ["team", "division", "attempts", "completions", "passing_yards", "passing_tds",
                "interceptions", "sacks", "team_yards_per_attempt"],
    "Rushing": ["team", "division", "carries", "rushing_yards", "rushing_tds",
                "team_yards_per_carry", "fumbles_lost"],
    "Receiving": ["team", "division", "targets", "receptions", "receiving_yards", "receiving_tds"],
}

with teams_tab:
    ui.section(
        "One row per team",
        "Summed off the same board the player pages read, so a total here and the men behind it cannot "
        "disagree. `points for` is the other way round: scoring is a team-level estimate the pool chain "
        "divides *down* to players rather than something they add up to, which is why the reconciliation "
        "audit on the team page exists at all.",
        sub="the sanity check a thirty-row roster table cannot give",
    )
    row = st.columns([3, 3, 6])
    which = row[0].radio("Show", list(TEAM_SETS), horizontal=True, key="league:teamset")
    conf2 = row[1].radio("Conference", ["Both", "AFC", "NFC"], horizontal=True, key="league:teamconf")
    cols = [c for c in TEAM_SETS[which] if c in teams.columns]
    body = teams if conf2 == "Both" else teams.filter(pl.col("conference") == conf2)
    ui.table(body.select(cols), digits=1, height=620,
             config={"expected_wins": st.column_config.NumberColumn("exp W", format="%.2f"),
                     "point_margin": st.column_config.NumberColumn("margin", format="%+.0f")})

    st.divider()
    ui.section("The league on one number",
               "Where a team sits against the other thirty-one on whichever number is being argued "
               "about. The spread is the useful part: a projection that has every team inside two "
               "carries a game of each other is a projection that has stopped distinguishing them.",
               sub="pick the number")
    metric = st.selectbox("Number", [c for c in teams.columns if teams.schema[c].is_numeric()],
                          index=0, format_func=ui.label, key="league:metric")
    ranked = teams.select("team", metric).drop_nulls(metric).sort(metric, descending=True)
    ui.rank_bars(ranked, metric, "team", height=560, digits=2)
    ui.kv(**{"high": f"{ranked['team'][0]} {ranked[metric][0]:,.2f}",
             "low": f"{ranked['team'][-1]} {ranked[metric][-1]:,.2f}",
             "league mean": f"{ranked[metric].mean():,.2f}",
             "top to bottom": f"{ranked[metric][0] - ranked[metric][-1]:,.2f}"})

# --------------------------------------------------------------------------- #
# 3. every game
# --------------------------------------------------------------------------- #
with games_tab:
    ui.section(
        "The schedule as projected scores",
        "Two rows of the environment frame joined into one game. `margin` is the difference in implied "
        "points and nothing more, so a game with a posted line is mostly the market's opinion and a game "
        "without one is entirely the model's — the `line posted` column says which.",
        sub="one row per game",
    )
    every = ui.weeks(view)
    week = st.select_slider("Week", every, value=every[0], key="league:week")
    this = games.filter(pl.col("week") == int(week))
    home = this.filter(pl.col("is_home")) if "is_home" in this.columns else this
    fixtures = home.join(
        this.select(pl.col("team").alias("opponent"),
                    pl.col("implied_points").alias("away_points"), pl.col("week")),
        on=["week", "opponent"], how="left",
    ).select(
        pl.col("opponent").alias("away"), pl.col("team").alias("home"),
        pl.col("away_points").alias("away score"), pl.col("implied_points").alias("home score"),
        pl.col("margin").alias("home margin"), pl.col("win_prob").alias("home wins"),
        *([pl.col("div_game").alias("division")] if "div_game" in this.columns else []),
        *([pl.col("has_market").alias("line posted")] if "has_market" in this.columns else []),
    ).sort("home margin", descending=True)
    ui.table(fixtures, digits=1, height=460,
             config={"home wins": st.column_config.ProgressColumn("home wins", format="%.2f",
                                                                 min_value=0.0, max_value=1.0),
                     "home margin": st.column_config.NumberColumn("home margin", format="%+.1f"),
                     "away score": st.column_config.NumberColumn("away", format="%.1f"),
                     "home score": st.column_config.NumberColumn("home", format="%.1f")})

    st.divider()
    ui.section("The season a team at a time",
               "Every team's seventeen chances in one grid. A row that is dark all the way across is a "
               "schedule, not a roster — and it is the reason two similar teams project to different "
               "records.",
               sub="a grid, on purpose")
    matrix = games.select("team", "week", "win_prob").pivot("week", index="team", values="win_prob")
    week_cols = [c for c in matrix.columns if c != "team"]
    ui.table(
        matrix.join(table.select("team", "expected_wins"), on="team", how="left")
        .sort("expected_wins", descending=True),
        height=620, auto=False,
        order=["team", "expected_wins", *week_cols],
        config={"expected_wins": st.column_config.NumberColumn("exp W", format="%.2f"),
                **{c: st.column_config.ProgressColumn(c, format="%.2f", min_value=0.0, max_value=1.0)
                   for c in week_cols}},
    )
    ui.note("An empty cell is the bye. The columns are calendar weeks, so week 18 is there and the "
            "record is still seventeen games.")

# --------------------------------------------------------------------------- #
# 4. against the posted lines
# --------------------------------------------------------------------------- #
# A sanity check, and deliberately only that. No player's stat line is projected from a betting line --
# the market reaches this app in two places, both about a team's afternoon, and both are spent in
# `src.model.team` before the player chain starts. What is left for a reader is the useful part: two
# independent estimates of the same game, and the places they disagree most are the places to look.
#
# The disagreement is read against `ui.MARKET_NOISE`. Both numbers miss a team's actual points by about
# seven a game in week 1, and over 2019-2025 the line beats this model's own estimate by 0.7% there, so a
# three-point gap says nothing about either. What it cannot do is tell you which one is wrong.
with market_tab:
    gaps = ui.market_gaps(view)
    by_team = ui.market_by_team(view)
    ui.section(
        "Against the posted lines",
        "Two independent reads on the same game: the number a book has posted, and what this model "
        "makes of the matchup on its own. Neither is projected from the other — a line never touches a "
        f"player's targets or yards — so where they disagree by more than {ui.MARKET_WORTH_A_LOOK:.0f} "
        "points, one of the two is wrong and it is worth knowing which. Under that, it is noise: both "
        f"numbers miss a team's actual score by about {ui.MARKET_NOISE:.0f} points a game in week 1, and "
        "measured over 2019-2025 the line is only 0.7% closer than the model at that horizon.",
        sub=f"{gaps.height} team-games have a number posted, of {games.height}",
        level=2,
    )
    if gaps.is_empty():
        st.info("No lines on file for this season yet. Nothing here is needed for a projection — the "
                "model estimates every game from the two rosters either way.")
    else:
        big = gaps.filter(pl.col("gap").abs() >= ui.MARKET_WORTH_A_LOOK)
        ui.tiles([
            {"name": "average disagreement", "value": float(gaps["gap"].abs().mean()), "digits": 1,
             "highlight": True, "sub": "points a game between the two estimates"},
            {"name": "worth a look", "value": big.height, "digits": 0,
             "sub": f"gaps of {ui.MARKET_WORTH_A_LOOK:.0f}+ points"},
            {"name": "biggest gap", "value": float(gaps["gap"].abs().max()), "digits": 1,
             "sub": f"{gaps.row(0, named=True)['team']} week {int(gaps.row(0, named=True)['week'])}"},
            {"name": "model above the line", "value": float((gaps["gap"] > 0).mean()), "digits": 0,
             "percent": True, "sub": "the rest are below it — 50% is unbiased"},
            {"name": "weight the line carries", "value": ui.market_weight_used(view), "digits": 2,
             "sub": "of a lined game's scoring level · 0 turns it off"},
        ])

        st.divider()
        ui.section("Team by team",
                   "Sorted by disagreement, the model above the line first. `carried offset` is the part "
                   "of a team's gap the projection accepts and applies to its unlined games too — shrunk "
                   "toward zero, because two lined games is not enough to move fifteen others whole.",
                   sub="lined games only, except the last two columns")
        ui.table(
            by_team, digits=1,
            order=["team", "lined", "market_per_game", "model_per_game", "gap", "worst_gap",
                   "carried_offset", "used_per_game", "games"],
            height=560,
            config={"lined": st.column_config.NumberColumn("lined", format="%d", width="small"),
                    "games": st.column_config.NumberColumn("games", format="%d", width="small"),
                    "market_per_game": st.column_config.NumberColumn("line PF/g", format="%.1f"),
                    "model_per_game": st.column_config.NumberColumn("model PF/g", format="%.1f"),
                    "gap": st.column_config.NumberColumn("gap", format="%+.1f"),
                    "worst_gap": st.column_config.NumberColumn("worst", format="%.1f"),
                    "carried_offset": st.column_config.NumberColumn("carried offset", format="%+.1f"),
                    "used_per_game": st.column_config.NumberColumn("used PF/g", format="%.1f")},
        )
        ui.note("`used PF/g` is what the projection actually runs on, over all seventeen games: the "
                "blend where a line exists and the model's own estimate plus the carried offset where "
                "one does not.")

        st.divider()
        ui.section("Game by game",
                   "The same gaps unaggregated, biggest first. A single game this far apart is usually "
                   "one of three things: a depth chart the model has wrong, a line that has news the "
                   "model has not, or a team the model prices wrong all season — the middle column tells "
                   "the three apart, since a team that is off by the same amount every week is the third.",
                   sub="every lined game")
        ui.table(
            gaps, digits=1,
            order=[c for c in ("week", "team", "opponent", "is_home", "market_points", "own_points",
                               "gap", "implied_points", "market_spread", "model_spread")
                   if c in gaps.columns],
            height=560,
            config={"week": st.column_config.NumberColumn("wk", format="%d", width="small"),
                    "is_home": st.column_config.CheckboxColumn("home", width="small"),
                    "market_points": st.column_config.NumberColumn("line", format="%.1f"),
                    "own_points": st.column_config.NumberColumn("model", format="%.1f"),
                    "gap": st.column_config.NumberColumn("gap", format="%+.1f"),
                    "implied_points": st.column_config.NumberColumn("used", format="%.1f"),
                    "market_spread": st.column_config.NumberColumn("line spread", format="%+.1f"),
                    "model_spread": st.column_config.NumberColumn("model spread", format="%+.1f")},
        )
        ui.note("To take the lines out of the projection entirely, turn off **Use posted lines** in the "
                "sidebar. It costs about a point of accuracy on projected margins and roughly five "
                "percentage points of straight-up winners. It barely touches the player board: no "
                "player rating is estimated from a line, and switching the market off moves the median "
                "player by 0.02 projected points across a whole season, the most affected by 0.8.")

# --------------------------------------------------------------------------- #
# 5. how far through the league
# --------------------------------------------------------------------------- #
with coverage_tab:
    done = ui.coverage(view)
    reviewed = int(done["reviewed"].sum())
    ui.section(
        "Which teams you have been through",
        "A pass over the league is thirty-two sittings, and this is the list of which ones are done. "
        "It is read off the edits themselves rather than a checkbox, so nothing has to be kept in step "
        "and nothing is lost when the browser reloads — a team carrying edits has been looked at, a team "
        "carrying none has not. Your work is saved as you go; the scenario panel in the sidebar names "
        "the file.",
        sub=f"{reviewed} of 32 teams carry edits",
        level=2,
    )
    ui.tiles([
        {"name": "teams touched", "value": reviewed, "digits": 0, "highlight": True,
         "sub": f"{32 - reviewed} still untouched"},
        {"name": "edits in total", "value": int(done["edits"].sum()), "digits": 0,
         "sub": f"on {int(done['players_edited'].sum())} players"},
        {"name": "team-level edits", "value": int(done["team_edits"].sum()), "digits": 0,
         "sub": "the offence rather than a man"},
        {"name": "for one week only", "value": int(done["week_edits"].sum()), "digits": 0,
         "sub": "the rest are season-wide"},
    ])
    if not reviewed:
        st.info("Nothing edited yet. Open a team, fix its depth chart and its shares, and it will "
                "appear here as done — the sidebar keeps the work.")

    left, right = st.columns([7, 5], gap="medium")
    with left:
        chosen = ui.pick_from(
            done, key="league:coverage",
            columns=["team", "division", "reviewed", "edits", "players_edited", "team_edits",
                     "week_edits", "last_touched"],
            height=620,
            config={"reviewed": st.column_config.CheckboxColumn("done", width="small"),
                    "edits": st.column_config.NumberColumn("edits", format="%d", width="small"),
                    "players_edited": st.column_config.NumberColumn("players", format="%d"),
                    "team_edits": st.column_config.NumberColumn("team", format="%d"),
                    "week_edits": st.column_config.NumberColumn("one week", format="%d"),
                    "last_touched": st.column_config.TextColumn("last touched")},
        )
    with right:
        by_division = done.group_by("division").agg(
            pl.col("reviewed").sum().alias("done"),
            pl.col("edits").sum().alias("edits"),
        ).sort("division")
        st.caption("By division")
        ui.table(by_division, height=320, auto=False,
                 config={"done": st.column_config.ProgressColumn("done", format="%d of 4",
                                                                min_value=0, max_value=4),
                         "edits": st.column_config.NumberColumn("edits", format="%d")})
        if chosen is not None:
            code = str(chosen["team"])
            with st.container(border=True):
                st.markdown(f"### {code}")
                st.caption("Untouched so far." if not chosen["reviewed"]
                           else f"{int(chosen['edits'])} edits, last at {chosen['last_touched'][11:16]}")
                ui.work_on_team(code, key="league:coverage:open")
                ui.page_link("pages/7_Edits.py", "Every edit in one list →", icon="🎛️")
