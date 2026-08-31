"""What actually happened, and whether to believe the projection: the record, and the model's report card.

Every other page answers "what is projected". This one answers "against what", and it is the page a
disagreement gets settled on. A projection is only ever a claim about a distribution somebody has
already seen: 24 points a game is a lot or a little depending on the league it is in, 34 dropbacks is
aggressive or not depending on what that offence has run, and 240 points is a career year or a
disappointment depending on the four seasons behind it.

Seven tabs. The first four are the record, widest to narrowest; the last three are the model scored
against it, which used to be a page of its own and should not have been — "what happened" and "was the
engine right about what happened" are the same question asked twice, and a reader who wants the second
has already opened the first.

1. **The league** — the scoring environment season by season. Plays, points, pass rate and pace, so the
   level every team number is read against is explicit rather than remembered.
2. **Teams** — a team a row, its own five seasons in the panel beside it, drawn against the league mean
   and against what the projection says it will do next.
3. **Players** — a man's record season by season on the current scoring, with his projection as the last
   bar of it. The same view the Player page opens on, arranged for looking *back*.
4. **Projection against record** — the whole board against what each of them actually did last season,
   the risers and the fallers, and then **this season so far**: empty until the lake carries games from
   the projection season, then the only comparison that finally matters, pro-rated to the week the
   season is actually in.
5. **Backtest** — was the board right on seasons it had not seen? Against two baselines: the player's
   own recency-weighted average, and what the Excel workbook actually did.
6. **Calibration** — is the level tilted, and are the ranges honest? Regressing outcome on projection
   separates the two errors MAE cannot tell apart, and interval coverage says whether the advertised
   floor was really missed one season in twenty.
7. **Share sums** — do the books balance? Every team's targets go to exactly one player, so the shares
   have to sum to the pool; and, at the end, whether the lake those sums were read from is current, plus
   the provenance chain behind every estimate in the live scenario.

Tabs 1–4 are **measured, not modelled.** Nothing in them is scaled, shrunk or normalised — it is the
lake, aggregated and scored the way the live scenario scores a projection, so a comparison is apples to
apples. The one modelled number that appears is labelled as the projection every time. **The backtest is
read, not run:** it is minutes of composition per variant and a command-line job
(`python -m src.model.backtest`) whose output is an artifact, so tab 5 reports what the last run found
and how old that run is.

`streamlit run app/Home.py` and pick History.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl                                                          # noqa: E402
import streamlit as st                                                       # noqa: E402

import ui                                                                    # noqa: E402
from src.config import PROJ_SEASON                                           # noqa: E402
from src.data import history                                                 # noqa: E402

view = ui.controls("History", icon="📚")
ui.scenario_banner(view)

SEASONS = ui.HISTORY_VIEW
LAST = ui.LAST_COMPLETE_SEASON

means = history.league_means(SEASONS).sort("season")
last_mean = means.filter(pl.col("season") == LAST)
first_mean = means.row(0, named=True)
recent = last_mean.row(0, named=True) if not last_mean.is_empty() else first_mean

ui.section(
    "What actually happened",
    "The evidence every projection on every other page is a claim against. It is measured rather than "
    "modelled: no shrinkage, no normalisation, no priors — the lake aggregated and scored exactly the "
    f"way the live scenario scores a projection, so the comparison is fair. Scored as "
    f"**{ui.scoring_label(view.scoring)}**; change the scoring in the sidebar and every point total on "
    "this page changes with it, because the seasons are re-scored rather than looked up.",
    sub=f"{SEASONS[0]}–{SEASONS[-1]} · {len(SEASONS)} seasons · projecting {view.season}",
    level=2,
)
ui.tiles([
    {"name": f"points a game, {LAST}", "value": recent.get("points_per_game"), "highlight": True,
     "sub": f"{recent.get('points_per_game', 0) - first_mean.get('points_per_game', 0):+.1f} "
            f"against {SEASONS[0]}"},
    {"name": f"plays a game, {LAST}", "value": recent.get("plays_per_game"),
     "sub": f"{recent.get('plays_per_game', 0) - first_mean.get('plays_per_game', 0):+.1f} "
            f"against {SEASONS[0]}"},
    {"name": f"pass rate, {LAST}", "value": recent.get("pass_rate"), "digits": 3, "percent": True,
     "sub": "dropbacks over plays, league wide"},
    {"name": f"seconds a play, {LAST}", "value": recent.get("seconds_per_play"),
     "sub": "pace — lower is more snaps for the same drives"},
    {"name": "seasons on file", "value": len(SEASONS), "digits": 0,
     "sub": f"{SEASONS[0]} to {SEASONS[-1]}"},
])

TABS = ["🌎 The league", "🏟️ Teams", "🧑 Players", "🎯 Projection against record", "📉 Backtest",
        "🧭 Calibration", "⚖️ Share sums"]
(league_tab, team_tab, player_tab, versus_tab, backtest_tab, calibration_tab,
 sums_tab) = st.tabs(TABS)

# --------------------------------------------------------------------------- #
# 1. the league
# --------------------------------------------------------------------------- #
with league_tab:
    ui.section(
        "The scoring environment, season by season",
        "A team's 24 points a game means nothing on its own. This is the level it is read against, and "
        "it moves: the pass rate, the pace and the touchdown rate are not constants, and a projection "
        "built on a five-year average is implicitly betting that the drift stops. Every value is per "
        "team-game, averaged over all 32 teams, so it is directly comparable to a single team's row on "
        "the next tab.",
        sub="per team-game, all 32 teams averaged",
    )
    LEAGUE_METRICS = {
        "points_per_game": "points a game",
        "plays_per_game": "plays a game",
        "pass_rate": "pass rate",
        "seconds_per_play": "seconds a play",
        "off_td_per_game": "offensive touchdowns a game",
        "rz_trips_per_game": "red-zone trips a game",
        "targets_per_game": "targets a game",
        "yards_per_attempt": "yards a pass attempt",
        "yards_per_carry": "yards a carry",
        "epa_per_play": "EPA a play",
        "explosive_pass_rate": "explosive pass rate",
        "success_rate": "success rate",
    }
    shown = st.segmented_control(
        "Metric", list(LEAGUE_METRICS), default="points_per_game", key="hist:league:metric",
        format_func=lambda m: LEAGUE_METRICS[m]) or "points_per_game"
    digits = 3 if means[shown].max() is not None and float(means[shown].max()) < 2 else 1
    ui.season_bars(means.select("season", shown), "season", shown, height=260, digits=digits)
    st.caption(f"{LEAGUE_METRICS[shown]}, league wide, one bar a season")
    ui.meters([
        ui.meter_html(str(r["season"]), r[shown],
                      maximum=float(means[shown].max()) * 1.1 if means[shown].max() else None,
                      digits=digits, marks=(("five-season mean", means[shown].mean()),))
        for r in means.rows(named=True)
    ])
    ui.focus_table(
        means,
        ["season", "points_per_game", "plays_per_game", "pass_rate", "seconds_per_play",
         "off_td_per_game", "rz_trips_per_game", "targets_per_game"],
        key="hist:league", height=260, digits=2,
        config=ui.fixed(1, "points_per_game", "plays_per_game", "seconds_per_play",
                        "targets_per_game", "rz_trips_per_game")
        | ui.fixed(2, "off_td_per_game", "yards_per_attempt", "yards_per_carry")
        | ui.percent("pass_rate", "success_rate", "explosive_pass_rate", "red_zone_rate"),
        label_text="all sixty-odd league columns",
    )

    ui.section(
        "Where the points went, by position",
        f"Every scored season in the window, grouped by position: how many men cleared a startable "
        f"total, and what the top of each room actually looked like. This is the shape a projection has "
        f"to reproduce — if the board says twelve tight ends will clear 150 points and history says "
        f"five, the board is wrong about tight ends rather than about a tight end.",
        sub=f"scored as {ui.scoring_label(view.scoring)}",
    )
    scored = ui.actuals(SEASONS, view.scoring)
    by_pos = (
        scored.filter(pl.col("position").is_in(list(ui.POSITIONS)))
        .group_by(["season", "position"]).agg(
            pl.col("fantasy_points").max().alias("best"),
            pl.col("fantasy_points").sort(descending=True).head(12).mean().alias("top_twelve_mean"),
            (pl.col("fantasy_points") > 150).sum().alias("over_150"),
            pl.len().alias("men_who_played"),
        ).sort(["position", "season"])
    )
    pos_pick = st.segmented_control("Position", list(ui.POSITIONS), default="WR",
                                    key="hist:league:pos") or "WR"
    mine = by_pos.filter(pl.col("position") == pos_pick)
    if mine.is_empty():
        st.info(f"No scored {pos_pick} seasons in the window.")
    else:
        ui.tiles([
            {"name": f"best {pos_pick} season, {LAST}",
             "value": mine.filter(pl.col("season") == LAST)["best"].max(), "highlight": True,
             "sub": "one man, one season, on this scoring"},
            {"name": "top twelve, mean", "value": mine.filter(pl.col("season") == LAST)
             ["top_twelve_mean"].max(), "sub": f"the startable {pos_pick}s in {LAST}"},
            {"name": "cleared 150 points", "value": mine.filter(pl.col("season") == LAST)
             ["over_150"].max(), "digits": 0, "sub": f"{pos_pick}s in {LAST}"},
            {"name": "men who played at all", "value": mine.filter(pl.col("season") == LAST)
             ["men_who_played"].max(), "digits": 0, "sub": "a snap or more"},
        ])
        ui.season_bars(mine.select("season", "top_twelve_mean"), "season", "top_twelve_mean",
                       height=240, digits=1)
        st.caption(f"The mean of the top twelve {pos_pick}s, season by season — the level a starter at "
                   "the position has actually had to reach")
        ui.table(by_pos, height=360, digits=1)

# --------------------------------------------------------------------------- #
# 2. teams
# --------------------------------------------------------------------------- #
with team_tab:
    ui.section(
        "What each offence has actually run",
        "Click a team. Its five seasons are on the right, per game so the rows compare, with the league "
        "mean marked on every bar and the projection for this season as the last one. This is the "
        "evidence for a team-level edit, which is the override with the least intuition behind it: "
        "nobody knows from memory whether 34 dropbacks a game is a lot.",
        sub="a team a row; its record and its projection beside it",
    )
    ts = ui.team_seasons(SEASONS)
    latest = ts.filter(pl.col("season") == LAST).select(
        "team", "games", "points_per_game", "plays_per_game", "dropbacks_per_game",
        "rush_att_per_game", "targets_per_game", "pass_rate", "seconds_per_play",
        "points_allowed_per_game", "epa_per_play", "rz_trips_per_game",
    ).sort("points_per_game", descending=True)
    TEAM_LIST = ["team", "points_per_game", "plays_per_game", "pass_rate", "seconds_per_play",
                 "points_allowed_per_game"]
    left, right = st.columns([5, 7], gap="medium")
    with left:
        st.caption(f"{LAST}, sorted by points a game")
        picked = ui.pick_from(
            latest, key="hist:team:list", columns=TEAM_LIST, height=600,
            config=ui.fixed(1, "points_per_game", "plays_per_game", "seconds_per_play",
                            "points_allowed_per_game", "dropbacks_per_game", "rush_att_per_game",
                            "targets_per_game", "rz_trips_per_game")
            | ui.percent("pass_rate") | ui.fixed(3, "epa_per_play"),
        )
    with right:
        if picked is None:
            st.info("Pick a team.")
        else:
            team = picked["team"]
            with st.container(border=True):
                st.markdown(f"### {team} — five seasons, and what is projected next")
                order = latest["team"].to_list()
                ui.chips(f"{order.index(team) + 1} of {len(order)} in points a game, {LAST}",
                         f"{picked['games']:.0f} games in {LAST}")
                ui.tiles([
                    {"name": f"points a game, {LAST}", "value": picked["points_per_game"],
                     "highlight": True,
                     "sub": f"league mean {latest['points_per_game'].mean():.1f}"},
                    {"name": "plays a game", "value": picked["plays_per_game"],
                     "sub": f"league mean {latest['plays_per_game'].mean():.1f}"},
                    {"name": "pass rate", "value": picked["pass_rate"], "digits": 3, "percent": True,
                     "sub": f"league mean {latest['pass_rate'].mean():.1%}"},
                    {"name": "points allowed a game", "value": picked["points_allowed_per_game"],
                     "sub": "the other half of a game's scoring level"},
                ])
                mine = ts.filter(pl.col("team") == team).sort("season")
                metric = st.selectbox(
                    "Draw", ["points_per_game", "plays_per_game", "dropbacks_per_game",
                             "rush_att_per_game", "targets_per_game", "pass_rate",
                             "seconds_per_play", "rz_trips_per_game", "epa_per_play",
                             "points_allowed_per_game"],
                    format_func=ui.label, key="hist:team:metric")
                dig = 3 if metric in ("pass_rate", "epa_per_play") else 1
                ui.season_bars(mine.select("season", metric), "season", metric, height=240, digits=dig)
                st.caption(f"{ui.label(metric)} by season — measured, no projection in this picture")
                ui.meters([
                    ui.meter_html(str(r["season"]), r[metric],
                                  maximum=float(ts[metric].max()) * 1.05 if ts[metric].max() else None,
                                  digits=dig,
                                  marks=((f"league {LAST}", latest[metric].mean()
                                          if metric in latest.columns else None),))
                    for r in mine.rows(named=True)
                ])
                st.caption("The same offence per game, with the projection and the league under it")
                ui.table(
                    ui.team_history(view, team,
                                    ("plays", "dropbacks", "carries", "targets", "points",
                                     "red_zone_trips", "pass_rate", "seconds_per_play"),
                                    SEASONS),
                    digits=2, height=300,
                )
                ui.note("The last two rows are the only modelled ones on this tab: what the engine "
                        f"projects this team to run in {view.season}, and what it projects the average "
                        "team to run. Everything above them happened.")

    ui.section(f"The league on one measure, {LAST}", sub="thirty-two teams, ranked")
    rank_on = st.selectbox("Rank by", [c for c in TEAM_LIST if c != "team"] + ["epa_per_play"],
                          format_func=ui.label, key="hist:team:rank")
    asc = rank_on in ("seconds_per_play", "points_allowed_per_game")
    ui.rank_bars(latest.sort(rank_on, descending=not asc), rank_on, "team", height=560, top=32,
                 digits=3 if rank_on in ("pass_rate", "epa_per_play") else 1)
    with st.expander("Every team, every season, every column"):
        ui.table(ts.sort(["team", "season"]), height=560, digits=2)

# --------------------------------------------------------------------------- #
# 3. players
# --------------------------------------------------------------------------- #
with player_tab:
    ui.section(
        "A man's record, and his projection as the last bar of it",
        "Scored on the live scoring rather than looked up, so these totals move when the sidebar does. "
        "Points per game is the honest comparison across seasons — a season cut short by injury is a "
        "small sample rather than a bad player — and it is why both are drawn.",
        sub="click a name; his five seasons are on the right",
    )
    scored = ui.actuals(SEASONS, view.scoring)
    per_season = scored.with_columns(
        (pl.col("fantasy_points") / pl.col("games").clip(lower_bound=1)).alias("points_per_game")
    )
    board = ui.board(view)

    fl, fm, fr = st.columns([2, 2, 3])
    with fl:
        hpos = ui.position_filter("hist:player:pos")
    with fm:
        season_pick = st.selectbox("Season", list(reversed(SEASONS)), key="hist:player:season")
    search = fr.text_input("Search", placeholder="part of a name", key="hist:player:search")

    ranked = ui.apply_filters(per_season.filter(pl.col("season") == season_pick), hpos, [], search) \
        .sort("fantasy_points", descending=True).head(400)

    if ranked.is_empty():
        st.info("Nobody matches those filters in that season.")
    else:
        PLAYER_LIST = ["player", "position", "team", "games", "fantasy_points", "points_per_game"]
        left, right = st.columns([5, 7], gap="medium")
        with left:
            st.caption(f"{season_pick}, what they actually scored")
            man = ui.pick_from(
                ranked, key=f"hist:player:list:{season_pick}", columns=PLAYER_LIST, height=600,
                config=ui.fixed(1, "fantasy_points", "points_per_game") | ui.fixed(0, "games"),
            )
        with right:
            if man is None:
                st.info("Pick a name.")
            else:
                pid = man["player_id"]
                his = per_season.filter(pl.col("player_id") == pid).sort("season")
                proj = board.filter(pl.col("player_id") == pid)
                with st.container(border=True):
                    st.markdown(f"### {man['player']} · {man['position']} — "
                                f"{his.height} scored season(s) on file")
                    ui.chips(f"{man['position']} {man['team']} in {season_pick}",
                             (f"projected {view.season} on {proj.row(0, named=True)['team']}"
                              if not proj.is_empty() else "not on a projected roster"),
                             ("changed team" if not proj.is_empty()
                              and proj.row(0, named=True).get("changed_team") else ""),
                             ("rookie" if not proj.is_empty()
                              and proj.row(0, named=True).get("is_rookie") else ""))
                    p = None if proj.is_empty() else proj.row(0, named=True)
                    best = his.sort("fantasy_points", descending=True).row(0, named=True)
                    ui.tiles([
                        {"name": f"{season_pick} points", "value": man.get("fantasy_points"),
                         "sub": f"{man.get('games', 0):.0f} games · "
                                f"{man.get('points_per_game', 0):.1f} a game"},
                        {"name": "his best season", "value": best["fantasy_points"], "digits": 1,
                         "sub": f"{best['season']} · {best['points_per_game']:.1f} a game"},
                        {"name": f"projected {view.season}",
                         "value": None if p is None else p.get("fantasy_points"),
                         "highlight": True,
                         "sub": ("not projected" if p is None
                                 else f"{p.get('games', 0):.1f} games · "
                                      f"{p.get('points_per_game', 0):.1f} a game")},
                        {"name": "against last season",
                         "value": None if p is None else p.get("delta_points"), "signed": True,
                         "sub": ("" if p is None else f"{LAST} was {p.get('last_points') or 0:.1f}")},
                    ])
                    drawn = pl.concat([
                        his.select(pl.col("season").cast(pl.String).alias("season"),
                                   pl.col("fantasy_points").alias("points")),
                        pl.DataFrame({"season": [str(view.season)],
                                      "points": [None if p is None else float(p["fantasy_points"])]},
                                     schema={"season": pl.String, "points": pl.Float64}),
                    ], how="vertical")
                    ui.season_bars(drawn, "season", "points", mark=str(view.season), height=240,
                                   digits=1)
                    st.caption("Measured seasons, and the projection picked out at the end")
                    ui.meters([
                        ui.meter_html(f"{r['season']} · {r['games']:.0f} games", r["points_per_game"],
                                      maximum=float(max(his["points_per_game"].max() or 1.0,
                                                        0.0 if p is None
                                                        else float(p.get("points_per_game") or 0.0)))
                                      * 1.15,
                                      digits=1,
                                      marks=((f"projected {view.season}",
                                              None if p is None else p.get("points_per_game")),),
                                      foot=(f"{r['fantasy_points']:.1f} points in all",))
                        for r in his.rows(named=True)
                    ])
                    with st.expander("His seasons as a table — every counted stat"):
                        ui.table(his.drop("player_id"), height=280, digits=1)
                    if p is not None:
                        st.caption("The projection this is being compared against, and the ✎ on every "
                                   "estimate behind it")
                        ui.player_panel(view, p, key=f"hist:panel:{pid}", compact=True,
                                        weeks=False, history=False)

        ui.section(f"The top of {season_pick}", sub="what they actually scored, twenty deep")
        ui.rank_bars(ranked.head(20), "fantasy_points", "player", height=460, digits=1)

# --------------------------------------------------------------------------- #
# 4. projection against record
# --------------------------------------------------------------------------- #
with versus_tab:
    board = ui.board(view)
    ui.section(
        f"What the board says against what they did in {LAST}",
        "Each point is one man. On the diagonal the projection agrees with last season; above it the "
        "engine expects more than he managed and below it less, and the interesting names are the ones "
        f"furthest from the line. A man who played eight games in {LAST} is *supposed* to be well above "
        "it, which is why games are in the table and why per-game is offered as well as the total.",
        sub=f"{board.height} projected men, {int(board['last_points'].is_not_null().sum())} with a "
            f"{LAST} record",
    )
    have = board.filter(pl.col("last_points").is_not_null())
    fl, fm = st.columns([2, 3])
    with fl:
        vpos = ui.position_filter("hist:vs:pos")
    with fm:
        basis = st.segmented_control("Compare on", ["season total", "per game"], default="season total",
                                     key="hist:vs:basis") or "season total"
    scope = have if not vpos else have.filter(pl.col("position").is_in(vpos))
    if scope.is_empty():
        st.info("Nobody with a record matches that filter.")
    else:
        if basis == "per game":
            scope = scope.with_columns(
                (pl.col("last_points") / pl.col("last_games").clip(lower_bound=1))
                .alias(f"{LAST} a game"),
                pl.col("points_per_game").alias(f"{view.season} a game"),
            )
            x, y = f"{LAST} a game", f"{view.season} a game"
        else:
            scope = scope.with_columns(pl.col("last_points").alias(f"{LAST} points"),
                                       pl.col("fantasy_points").alias(f"{view.season} projected"))
            x, y = f"{LAST} points", f"{view.season} projected"
        risers = scope.sort("delta_points", descending=True)
        fallers = scope.sort("delta_points")
        ui.tiles([
            {"name": "biggest riser", "value": risers.row(0, named=True)["delta_points"],
             "signed": True, "highlight": True,
             "sub": f"{risers.row(0, named=True)['player']} · "
                    f"{risers.row(0, named=True)['position']} "
                    f"{risers.row(0, named=True)['team']}"},
            {"name": "biggest faller", "value": fallers.row(0, named=True)["delta_points"],
             "signed": True,
             "sub": f"{fallers.row(0, named=True)['player']} · "
                    f"{fallers.row(0, named=True)['position']} "
                    f"{fallers.row(0, named=True)['team']}"},
            {"name": "mean change", "value": scope["delta_points"].mean(), "signed": True,
             "sub": "over everybody with a record"},
            {"name": "changed team", "value": int(scope["changed_team"].sum()), "digits": 0,
             "sub": "a new offence divides its pools differently"},
        ])
        ui.scatter(scope.select("player", "position", "team", x, y), x, y, height=380, diagonal=True)
        ui.section("The twenty largest moves", sub="up first, then down")
        up, down = st.columns(2, gap="medium")
        with up:
            st.caption("**Projected well above** last season")
            ui.rank_bars(risers.head(12), "delta_points", "player", height=320, digits=1)
        with down:
            st.caption("**Projected well below** it")
            ui.rank_bars(fallers.head(12).with_columns(pl.col("delta_points").abs()
                                                       .alias("points lost")),
                         "points lost", "player", height=320, digits=1, colour="#b06a6a")
        VS_COLS = ["player", "position", "team", "last_team", "last_games", "last_points",
                   "games", "fantasy_points", "delta_points", "changed_team"]
        ui.focus_table(
            scope.sort(pl.col("delta_points").abs(), descending=True), VS_COLS,
            key=f"hist:vs:{basis}", height=520, digits=1,
            config=ui.fixed(1, "last_points", "fantasy_points", "delta_points", "games", "last_games",
                            "points_per_game"),
            note_text="`delta_points` is the projection minus what he actually scored last season. It is "
                      "not an error — the season has not happened — it is the size of the claim the "
                      "engine is making about him.",
            label_text="every projected column too",
        )

    # ----------------------------------------------------------------------- #
    # and the same comparison against the season being played, which is the one that finally counts. It
    # was a tab of its own and is empty for most of the year, which is the wrong shape for a tab: it
    # belongs under the comparison it is the live version of.
    # ----------------------------------------------------------------------- #
    st.divider()
    stale = ui.staleness()
    week = int(stale["current_week"] or 0)
    ui.section(
        f"{view.season}, as far as it has been played",
        "The only comparison that finally matters, and it needs games in the lake to exist. Once "
        "`processed/` carries weeks from this season, this section reads them, scores them the same way, and "
        "sets them against the projection **pro-rated to the same number of games** — because a man on "
        "pace over four games is not ahead of a seventeen-game projection, he is on pace against four "
        "seventeenths of it.",
        sub=(f"week {week} · the usage tables carry week {stale['latest_week']}"
             if stale["in_season"] and stale["latest_week"] else "the season has not started"),
    )
    try:
        so_far = ui.actuals((view.season,), view.scoring)
    except Exception as exc:                          # noqa: BLE001 -- a missing partition is the answer
        so_far = pl.DataFrame()
        why = str(exc)
    else:
        why = ""

    if so_far.is_empty():
        st.info(
            f"**Nothing played yet.** The lake carries no scored {view.season} games, so there is "
            f"nothing to compare the projection against. It fills itself in as the season is "
            f"played and the lake is refreshed — everything else on this page works now.",
            icon="📅",
        )
        if why:
            ui.note(f"What the reader hit: `{why}`. That is a missing partition rather than a failure, "
                    "which is why the rest of the page is unaffected.")
    else:
        board = ui.board(view)
        played = so_far.with_columns(
            (pl.col("fantasy_points") / pl.col("games").clip(lower_bound=1)).alias("actual_pg")
        ).select("player_id", "player", "position", "team", "games", "fantasy_points", "actual_pg")
        joined = played.join(
            board.select("player_id", pl.col("fantasy_points").alias("projected_season"),
                         pl.col("points_per_game").alias("projected_pg"),
                         pl.col("games").alias("projected_games")),
            on="player_id", how="left",
        ).with_columns(
            (pl.col("projected_pg") * pl.col("games")).alias("expected_by_now"),
            (pl.col("fantasy_points") - pl.col("projected_pg") * pl.col("games")).alias("ahead_by"),
            (pl.col("actual_pg") - pl.col("projected_pg")).alias("ahead_a_game"),
        ).sort("ahead_by", descending=True, nulls_last=True)
        ui.tiles([
            {"name": "men with a game played", "value": joined.height, "digits": 0,
             "sub": f"scored {view.season} seasons in the lake"},
            {"name": "furthest ahead of pace", "value": joined.row(0, named=True)["ahead_by"],
             "signed": True, "highlight": True, "sub": joined.row(0, named=True)["player"]},
            {"name": "furthest behind",
             "value": joined.sort("ahead_by", nulls_last=True).row(0, named=True)["ahead_by"],
             "signed": True,
             "sub": joined.sort("ahead_by", nulls_last=True).row(0, named=True)["player"]},
            {"name": "mean gap a game", "value": joined["ahead_a_game"].mean(), "signed": True,
             "sub": "actual minus projected, per game — the board's level so far"},
        ])
        ui.scatter(joined.select("player", "position", "team", "expected_by_now", "fantasy_points"),
                   "expected_by_now", "fantasy_points", height=380, diagonal=True)
        st.caption("Projected-by-now against actually-scored. The diagonal is exactly on pace.")
        ahead, behind = st.columns(2, gap="medium")
        with ahead:
            st.caption("**Ahead of pace**")
            ui.rank_bars(joined.head(12), "ahead_by", "player", height=320, digits=1)
        with behind:
            st.caption("**Behind it**")
            ui.rank_bars(joined.sort("ahead_by", nulls_last=True).head(12)
                         .with_columns(pl.col("ahead_by").abs().alias("points short")),
                         "points short", "player", height=320, digits=1, colour="#b06a6a")
        ui.focus_table(
            joined,
            ["player", "position", "team", "games", "fantasy_points", "expected_by_now", "ahead_by",
             "actual_pg", "projected_pg", "ahead_a_game"],
            key="hist:sofar", height=520, digits=1,
            config=ui.fixed(1, "fantasy_points", "expected_by_now", "ahead_by", "actual_pg",
                            "projected_pg", "ahead_a_game", "projected_season", "projected_games")
            | ui.fixed(0, "games"),
            note_text="A man missing a projected column is one the board does not carry — a practice-squad "
                      "call-up, most often — and he is left in rather than dropped, because the fact that "
                      "the projection never named him is itself the finding.",
        )

# --------------------------------------------------------------------------- #
# 5. was it right on seasons it had not seen
# --------------------------------------------------------------------------- #
with backtest_tab:
    summary = ui.backtest_summary()
    age = ui.backtest_age()
    if summary.is_empty():
        st.info("No backtest on file. Run `python -m src.model.backtest` — it writes "
                "`data/fitted/backtest_summary.parquet` and this tab reads it.", icon="📉")
    else:
        seasons = sorted(s for s in summary["season"].unique().to_list() if s)
        view_cut = st.segmented_control(
            "Population", ["all", "played", "regulars", "starters"], default="all",
            key="hist:pop") or "all"
        st.caption(
            "`all` is every player the projection named — points given to somebody who never played "
            "count as error. `starters` is selected on the outcome, so it says whether the board got "
            "the right people, not whether it was calibrated."
        )
        if age is not None and age > 30:
            st.warning(f"The backtest is {age:.0f} days old. If the engine has changed since, these "
                       f"numbers describe a model that no longer exists.", icon="⚠️")

        pooled = summary.filter(
            (pl.col("season") == 0) & (pl.col("view") == view_cut)
            & (pl.col("stat") == "fantasy_points")
        ).sort("mae")
        ui.section(
            "Season fantasy points, pooled over every held-out season",
            "`full` is the engine. `ewma` is the player's own recency-weighted season average — the "
            "honest floor, and hard to beat for an established starter. `workbook` is what the "
            "spreadsheet this replaces actually did. The rest turn off one node each, so a node that "
            "helps its own metric and hurts the board is visible here. Judge on all three of MAE, RMSE "
            "and rank correlation: this population is four-tenths zeroes and heavily right-skewed, so "
            "shaving the level off everybody lowers MAE while making every startable player worse.",
            sub=(f"held-out {seasons[0]}–{seasons[-1]}"
                 + (f" · run {age:.0f} days ago" if age is not None else " · age unknown")),
        )
        if not pooled.is_empty():
            named = {r["variant"]: r for r in pooled.rows(named=True)}
            engine = named.get("full")
            ewma = named.get("ewma")
            book = named.get("workbook")
            best = pooled.row(0, named=True)

            def against(other: dict | None) -> str:
                """How the engine did against a baseline, when both are on file."""
                if not (engine and other) or engine.get("mae") is None or other.get("mae") is None:
                    return "not in this run"
                return f"the engine is {other['mae'] - engine['mae']:+.2f} against it"

            rho = None if not engine else engine.get("spearman")
            ui.tiles([
                {"name": "the engine · MAE", "value": None if not engine else engine.get("mae"),
                 "digits": 2, "highlight": True,
                 "sub": "" if rho is None else f"rank correlation {float(rho):.3f}"},
                {"name": "his own average · MAE", "value": None if not ewma else ewma.get("mae"),
                 "digits": 2, "sub": against(ewma)},
                {"name": "the workbook · MAE", "value": None if not book else book.get("mae"),
                 "digits": 2, "sub": against(book)},
                {"name": "best variant on file", "value": best.get("mae"), "digits": 2,
                 "sub": f"{best['variant']} · over {int(best['n']):,} player-seasons"},
            ])
            st.caption("Mean absolute error by variant — shorter is better")
            ui.rank_bars(pooled.select(pl.col("variant").alias("player"), "mae").head(16),
                         "mae", "player", height=340, digits=2, colour="#b06a6a")
            ui.focus_table(
                pooled.select("variant", "n", "mae", "rmse", "bias", "spearman"),
                ["variant", "n", "mae", "rmse", "bias", "spearman"],
                key=f"hist:pooled:{view_cut}", height=420, digits=3,
                config=ui.fixed(2, "mae", "rmse", "bias") | ui.fixed(3, "spearman"),
            )

        # both of these are matrices on purpose: a variant against a season, and a variant against a
        # stat, are exactly the comparisons a grid is the right shape for
        ui.section("Per season", sub="mean absolute error, a variant a row")
        per = summary.filter(
            (pl.col("season") != 0) & (pl.col("view") == view_cut)
            & (pl.col("stat") == "fantasy_points")
        )
        wide = per.select("variant", "season", "mae").pivot(
            on="season", index="variant", values="mae"
        ).sort("variant")
        ui.table(wide, digits=2)

        ui.section("By stat", sub="the engine against the two baselines, stat by stat")
        stats = summary.filter(
            (pl.col("season") == 0) & (pl.col("view") == view_cut)
            & pl.col("variant").is_in(["full", "ewma", "workbook"])
        ).select("stat", "variant", "mae").pivot(on="variant", index="stat", values="mae")
        ui.table(stats, digits=2)

# --------------------------------------------------------------------------- #
# 6. is the level tilted, and is the range honest
# --------------------------------------------------------------------------- #
with calibration_tab:
    cal = ui.backtest_calibration()
    if not cal:
        st.info("No backtest on file, so there is nothing to calibrate against. Run "
                "`python -m src.model.backtest`.", icon="📉")
    else:
        ui.section(
            "Is the level right, given what the projection said?",
            "MAE says how close the projections land; this says whether they are **tilted**. "
            "Regressing the outcome on the projection separates the two errors MAE cannot tell apart: "
            "a slope of 1 is the claim, below 1 means the model commits harder than the evidence "
            "supports and the extremes come back toward the middle, above 1 means it hedged.",
            sub="the tick on every bar is 1.00, which is the claim",
        )
        lines = cal["lines"]
        # the whole tab in one picture: a slope bar with 1.00 marked on it, a position a bar. A slope
        # is NaN where the fit had fewer than ten players or no spread in the projection, and those
        # rows are left out of the drawing rather than drawn at zero.
        drawable = lines.filter(pl.col("slope").is_finite()).sort(["population", "position"])
        ui.meters([
            ui.meter_html(
                f"{r['position']} · {r['population']}", r["slope"], maximum=1.6, digits=3,
                marks=(("the claim", 1.0),),
                foot=(f"mean error {float(r['mean_err']):+.1f} points" if r.get("mean_err") is not None
                      else "",
                      f"over {int(r['n']):,} player-seasons" if r.get("n") is not None else ""),
            )
            for r in drawable.rows(named=True)
        ])
        ui.focus_table(
            lines.select("position", "population", "n", "slope", "intercept", "mean_err",
                         "median_err"),
            ["position", "population", "n", "slope", "intercept", "mean_err", "median_err"],
            key="hist:cal:lines", height=420, digits=3,
            config=ui.fixed(3, "slope") | ui.fixed(2, "intercept", "mean_err", "median_err"),
        )
        st.warning(
            "**`median_err` is negative everywhere and that is not a defect.** Season fantasy points "
            "are heavily right-skewed, so a projection correctly aimed at the *mean* sits above the "
            "*median* outcome by construction. `mean_err` is the number to judge — it is within a "
            "couple of points of zero for every position. Shrinking the projections to flatten "
            "`median_err` would break the mean to flatter a statistic that was never supposed to be "
            "zero.", icon="🧭",
        )

        ui.section(
            "By projected-games band",
            "Cut on **projected** games, never on realised games: conditioning on having appeared keeps "
            "only the players who beat the availability the model applied, which shifts every error in "
            "the table. `never_played` states that same fact without selecting on it.",
            sub="the known residual, and it is second-order",
        )
        bands = cal["bands"]
        ui.rank_bars(bands.select(pl.col("band").alias("player"), "mean_err"), "mean_err", "player",
                     height=300, digits=1)
        ui.focus_table(
            bands.select("band", "n", "proj_mean", "act_mean", "mean_err", "median_err",
                         "never_played"),
            ["band", "n", "proj_mean", "act_mean", "mean_err", "median_err", "never_played"],
            key="hist:cal:bands", height=340, digits=2,
            config=ui.fixed(1, "proj_mean", "act_mean", "mean_err", "median_err")
            | ui.percent("never_played"),
        )
        ui.note(
            "Across the 5–11 projected-games band the mean is over-projected by roughly 6–12 points, "
            "offset by a small under-projection across the deep bench. Total volume is conserved by "
            "pool normalisation, so this is a distributional error along the games axis rather than a "
            "level error. It is documented here rather than corrected because every candidate "
            "correction tested either had the wrong sign in aggregate or traded the mid-band for the "
            "tail."
        )

    # ------------------------------------------------------------------- #
    # A mean and a range are two promises, and the second one is scored differently: the level is
    # a regression and the width is coverage. They were two tabs, which read as two subjects --
    # they are one subject, "is the advertised number honest", so they are one tab.
    # ------------------------------------------------------------------- #
    st.divider()
    disp = ui.dispersion()
    st.caption(ui.sim_provenance())
    if not disp.meta.get("fitted"):
        st.warning("The dispersion is not fitted, so the ranges on the other pages are pooled "
                   "defaults rather than measured. Run `python -m src.model.simulate --fit`.",
                   icon="⚠️")
    else:
        cover = disp.meta.get("coverage") or {}
        if cover:
            ui.section(
                "Did the advertised range contain the season?",
                "The scale was fitted on the **interquartile range of the PIT**, which is invariant to "
                "a shift and therefore scores only the width. The alternative — a distance-from-uniform "
                "score — cannot tell a bad width from a bad centre, and will widen an interval to "
                "absorb a level error. Whatever location error is left stays visible as `median bias`, "
                "where it is the projection's to answer for rather than the interval's. The population "
                "is projected-startable seasons (over "
                f"{disp.meta.get('calibration_min_projected', 0):.0f} points), cut on the projection "
                "and never on the outcome.",
                sub="the promise a range makes, scored",
            )
            ui.tiles([
                {"name": "P5–P95 coverage", "value": cover.get("cover_90"), "digits": 3,
                 "highlight": True,
                 "sub": f"{cover.get('cover_90', 0) - 0.90:+.3f} against the 0.90 promised"},
                {"name": "P25–P75 coverage", "value": cover.get("cover_50"), "digits": 3,
                 "sub": f"{cover.get('cover_50', 0) - 0.50:+.3f} against the 0.50 promised"},
                {"name": "PIT interquartile range", "value": cover.get("pit_iqr"), "digits": 3,
                 "sub": f"{cover.get('pit_iqr', 0) - 0.5:+.3f} against 0.500 — this is what was fitted"},
                {"name": "median bias", "value": cover.get("median_bias"), "digits": 1, "signed": True,
                 "sub": "the projection's to answer for, not the interval's"},
            ])
            ui.meters([
                ui.meter_html("P5–P95 coverage", cover.get("cover_90"), maximum=1.0, digits=3,
                              percent=True, marks=(("promised", 0.90),),
                              foot=("a floor missed more often than one season in twenty was never a "
                                    "floor",)),
                ui.meter_html("P25–P75 coverage", cover.get("cover_50"), maximum=1.0, digits=3,
                              percent=True, marks=(("promised", 0.50),)),
            ])
            below, above = cover.get("below"), cover.get("above")
            if below is not None and above is not None:
                st.caption(
                    f"{below:.1%} of held-out seasons finished **below** their P5 and {above:.1%} "
                    f"**above** their P95, against 5% expected at each end."
                )

        ui.section(
            "Fitted width per position",
            "`scale` multiplies every persistent sigma for the position. `indistinguishable` is the set "
            "of grid values whose calibration loss is within tolerance of the best — where it holds "
            "more than one value, the fit is not claiming precision it does not have.",
            sub="what the ranges on the other pages are made of",
        )
        ridge = disp.meta.get("scale_ridge") or {}
        ui.table(pl.DataFrame([
            {"position": pos, "scale": scale,
             "indistinguishable": ", ".join(f"{x:g}" for x in ridge.get(pos, [])),
             "games bias (starter)": (disp.meta.get("games_bias") or {}).get(f"{pos}:starter"),
             "games bias (depth)": (disp.meta.get("games_bias") or {}).get(f"{pos}:depth")}
            for pos, scale in sorted(disp.scale.items())
        ]), config=ui.fixed(3, "games bias (starter)", "games bias (depth)"))
        seen = disp.meta.get("calibration_targets") or []
        if seen:
            ui.note(f"Fitted on {seen[0]}–{seen[-1]}. The projection driving those intervals is ex "
                    f"ante, but the width is in sample: measured by refitting without the last season, "
                    f"the in-sample advantage is worth about 0.007 of coverage.")

# --------------------------------------------------------------------------- #
# 7. do the books balance, and is the lake they were read from current
# --------------------------------------------------------------------------- #
with sums_tab:
    ui.section(
        "Do the players add up to a whole offence?",
        "A team's targets in a game are a fixed quantity and every one goes to exactly one player, so "
        "Σ over players of P(he plays) × his share must be 1 — or rather, must be the total actually "
        "observed over 2016–2025, which for some pools is not 1: `receiving_tds` counts a two-point "
        "conversion the touchdown pool does not. `gap_pct` is how far the estimated shares were from "
        "the pool *before* anything was rescaled, and it is the honest headline. The workbook this "
        "replaces warned when shares exceeded 100% and then left them there.",
        sub="the audit the normalisation toggle stands next to",
    )
    pools = ui.pool_report(view)
    worst_pool = pools.sort(pl.col("gap_pct").abs(), descending=True)
    ui.tiles([
        {"name": "pools checked", "value": pools.height, "digits": 0,
         "sub": "every counted quantity the offence divides"},
        {"name": "worst gap before rescaling", "value": worst_pool.row(0, named=True)["gap_pct"],
         "digits": 2, "highlight": True,
         "sub": f"{worst_pool.row(0, named=True)['pool']} · per cent of the pool"},
        {"name": "normalisation", "value": None,
         "sub": "on" if view.settings.normalize_pools else "off — the counts are the raw shares"},
    ])
    ui.focus_table(
        pools.drop("team_col"),
        [c for c in ("pool", "measured_target", "raw_mean", "after_mean", "gap_pct",
                     "count_per_game") if c in pools.columns],
        key="hist:pools", height=420, digits=3,
        config=ui.fixed(3, "measured_target", "raw_mean", "raw_min", "raw_max", "after_mean")
        | ui.fixed(2, "count_per_game", "gap_pct"),
        label_text="the minimum and maximum too",
    )
    if not view.settings.normalize_pools:
        st.warning("Normalisation is **off** in this scenario, so the counts on every other page are "
                   "the raw shares — the pools do not balance and team totals will not reconcile.",
                   icon="⚠️")

    ui.section(
        "Players against the team they were divided from",
        "Counts drawn from a normalised pool must match to rounding. Yards and completions are a count "
        "times a rate and are deliberately *not* forced to match: a team whose receivers are all more "
        "efficient than its recent offence really should project above the team's own yardage, and "
        "scaling that away would hide the disagreement instead of showing it. `from_pool` says which "
        "rows are held to the standard. `gap_pct` is the league mean and is the number to judge; "
        "`worst_abs_gap` is the single worst team-game, so it is a count of events rather than a "
        "percentage and one queued room can own it by itself.",
        sub="a counted pool must reconcile; a rate is allowed to disagree",
    )
    rec = ui.reconciliation(view)
    strict = rec.filter(pl.col("from_pool"))
    ui.tiles([
        {"name": "counted pools checked", "value": strict.height, "digits": 0,
         "sub": "held to the pool they were divided from"},
        {"name": "worst mean gap", "value": (None if strict.is_empty() else
                                            strict.sort(pl.col("gap_pct").abs(),
                                                        descending=True).row(0, named=True)["gap_pct"]),
         "digits": 2, "highlight": True, "signed": True,
         "sub": ("" if strict.is_empty() else
                 f"{strict.sort(pl.col('gap_pct').abs(), descending=True).row(0, named=True)['stat']}"
                 " · per cent of the pool")},
        {"name": "worst single team-game", "value": (None if strict.is_empty()
                                                     else strict["worst_abs_gap"].max()),
         "digits": 3, "sub": "events, not per cent — one team, one week"},
    ])
    ui.table(
        rec.sort("from_pool", descending=True),
        config=ui.fixed(2, "players_per_game", "team_per_game", "gap_pct")
        | ui.fixed(3, "worst_abs_gap"),
        height=400,
    )
    broken = strict.filter(pl.col("gap_pct").abs() > 5.0)
    if broken.is_empty():
        st.success("Every pool-derived count sums back to within five per cent of the team pool it was "
                   "divided from. The residuals that remain are the queued rooms below, where the "
                   "correction lands on the last men in the depth order rather than on every claimant.",
                   icon="✅")
    else:
        st.error(f"These come from a normalised pool and do not sum back to it: "
                 f"{', '.join(broken['stat'].to_list())}. That is a defect, not a tolerance.",
                 icon="🚨")

    ui.section(
        "Which rosters the estimator struggles with",
        "A team summing short has a job its depth chart does not name — a vacated target share nobody "
        "has inherited — which is a real finding about the roster rather than a defect in the scaling.",
        sub="pick a pool; the teams furthest short are first",
    )
    tp = ui.team_pools(view)
    pool_pick = st.selectbox("Pool", sorted(tp["pool"].unique().to_list()),
                             index=sorted(tp["pool"].unique().to_list()).index("targets")
                             if "targets" in tp["pool"].to_list() else 0, key="hist:pool")
    worst = tp.filter(pl.col("pool") == pool_pick).sort("gap_pct")
    ui.rank_bars(worst.select(pl.col("team").alias("player"), "gap_pct").head(16), "gap_pct",
                 "player", height=320, digits=1)
    ui.focus_table(worst, [c for c in ("team", "measured_target", "raw_sum", "factor", "gap_pct")
                           if c in worst.columns],
                   key=f"hist:teampool:{pool_pick}", height=380, digits=3,
                   config=ui.fixed(3, "measured_target", "raw_sum", "factor") | ui.fixed(1, "gap_pct"))

    # ------------------------------------------------------------------- #
    # The sums are only as good as what they were computed from, so the age of the lake and the
    # chain behind every estimate close the audit rather than sitting in a tab of their own.
    # ------------------------------------------------------------------- #
    st.divider()
    stale = ui.staleness()
    status = ui.freshness()
    oldest = status["age_days"].max()
    missing = status.filter(pl.col("missing_proj_season"))
    ui.section(
        f"Is the lake current for {PROJ_SEASON}?",
        "`own refresh` tables are the light 2026 ones this repo refreshes itself; `shared lake` tables "
        "come from the parquet lake and are not ours to update. The stale check is deliberately about "
        "coverage rather than a fixed age in days — a processed table without the projection season is "
        "normal in August and alarming in November.",
        sub="age is the easy question; coverage is the one that matters",
    )
    ui.tiles([
        {"name": "week of the season", "value": stale["current_week"], "digits": 0,
         "sub": f"{PROJ_SEASON} has not started" if not stale["in_season"] else f"of {PROJ_SEASON}"},
        {"name": "usage tables carry", "value": stale["latest_week"], "digits": 0,
         "highlight": True,
         "sub": ("nothing to be behind on yet" if not stale["in_season"]
                 else "no weeks at all" if stale["latest_week"] is None
                 else f"{stale['weeks_behind']} week(s) behind")},
        {"name": "tables read", "value": status.height, "digits": 0, "sub": "everything the engine opens"},
        {"name": "oldest table", "value": oldest, "digits": 0, "sub": "days since it was written"},
        {"name": f"missing {PROJ_SEASON}", "value": missing.height, "digits": 0,
         "sub": "normal for usage before the season"},
    ])

    if not stale["in_season"]:
        st.success(f"The {PROJ_SEASON} season has not started, so there is no in-season usage to be "
                   f"behind on. Age below is the age of the history the projection is built from.",
                   icon="🗓️")
    elif stale["latest_week"] is None:
        st.error(f"It is week {stale['current_week']} of {PROJ_SEASON} and `processed/` carries **no "
                 f"weeks at all** for this season. Every number in this app is being projected from "
                 f"history alone, with none of the season that has already been played.", icon="🚨")
    elif stale["weeks_behind"] >= 1:
        st.error(f"**`processed/` is {stale['weeks_behind']} week"
                 f"{'s' if stale['weeks_behind'] != 1 else ''} behind.** It is week "
                 f"{stale['current_week']} and the usage tables stop at week {stale['latest_week']}, so "
                 f"the projection has not seen the most recent games. Refresh the lake.", icon="🚨")
    else:
        st.success(f"`processed/` is current: week {stale['current_week']}, and the usage tables carry "
                   f"week {stale['latest_week']}.", icon="✅")

    if not missing.is_empty():
        st.warning(f"No {PROJ_SEASON} partition: {', '.join(missing['table'].to_list())}. Before the "
                   f"season this is normal for usage tables and a problem for rosters, depth charts "
                   f"and schedules.", icon="⚠️")

    both = st.columns([2, 3], gap="medium")
    with both[0]:
        if stale["tables"]:
            st.caption("Per processed table, the latest week it carries")
            ui.table(pl.DataFrame([
                {"table": f"processed/{k}", "latest week": v,
                 "behind": None if v is None else max(stale["current_week"] - v, 0)}
                for k, v in stale["tables"].items()
            ]), height=300)
    with both[1]:
        st.caption("Every table the engine reads")
        ui.table(status, config=ui.fixed(1, "age_days"), height=300)

    ui.section("What each estimate was built from",
               "Per estimate: the prior, the observation behind it, the weight the observation earned, "
               "and the number that was used. This is the table that answers 'why is he projected "
               "there'.",
               sub="the whole chain, one row an estimate")
    prov = ui.provenance(view)
    if prov.is_empty():
        st.info("No provenance for this scenario.")
    else:
        ui.focus_table(
            prov.head(400),
            [c for c in ("stage", "level", "key", "field", "mode", "value", "base_now", "used_now",
                         "applied") if c in prov.columns],
            key="hist:prov", height=380, digits=4,
            note_text=f"First 400 of {prov.height:,} rows — the Exports page will write all of them.",
            label_text="every column of the chain",
        )

