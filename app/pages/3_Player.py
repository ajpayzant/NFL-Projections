"""One player, and every number the projection is made of.

The point of this page is that a projection is not an opinion to be accepted or rejected whole. It is
a product of four things -- will he play, how much of the offence goes through him, how good is he at
converting it, and what kind of games does his team play -- and each of those is separately arguable.

**Ratings** is the tab that matters, and it is drawn rather than tabulated. Each estimate is a meter: the
fill is what the projection ran on, and the two ticks are his own record and what his job is worth to
everyone who has held it. That picture is the whole argument -- a fill on the left tick is a measurement
of the man, a fill on the right is an assumption about a role -- and it used to be eleven numeric columns
with the reader doing the subtraction. The assumptions sort to the top, because they are what is worth
arguing with, and the ✎ beside each one opens the evidence and changes the number where it stands.

**Adjust** is the same edits without the argument, for when it is already known which numbers are wrong:
where he sits on the chart, and then every rating he has in one typed list, for the season or for chosen
weeks, written in one go. Nothing is computed until it is applied, so four changes cost one engine run
rather than four. The slot is on top of that list because it is not a rating -- it is the rank every one
of them is estimated off, so moving him re-prices him and everybody the move renumbers.

The rest of the page is the same argument at other scales: the stat line the points are made of, the
seventeen games, the simulated range, what he has actually done, the men he is competing with for the
same pool, and -- last -- **Side by side**, which is him against one to three other men on every estimate
they have in common, the sample size beside each. That last tab was a page of its own, and it should not
have been: a comparison is a thing done *while looking at a player*, and the layer that settles it is the
`n` under a share rather than the total above it. Every table on the page is capped at the columns worth
reading with the full frame one click away.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import polars as pl                                                          # noqa: E402
import streamlit as st                                                       # noqa: E402

import ui                                                                    # noqa: E402
from src.model import overrides                                              # noqa: E402
from src.model.overrides import Override                                     # noqa: E402

view = ui.controls("Player")
ui.scenario_banner(view)

me = ui.pick_player(view)
if me is None:
    st.stop()

pid, team, position = me["player_id"], me["team"], me["position"]
weekly = ui.weekly(view).filter(pl.col("player_id") == pid).sort("week")
part = ui.participation(view).filter(pl.col("player_id") == pid)
prow = part.row(0, named=True) if not part.is_empty() else {}
stats = [c for c in ui.stat_columns(position) if c in weekly.columns]

# --------------------------------------------------------------------------- #
# who he is, and what the projection says
# --------------------------------------------------------------------------- #
ui.player_panel(view, me, key="player", estimates=False, weeks=False, history=False)

if not view.scenario.is_baseline:
    mine = ui.board_diff(view).filter(pl.col("player_id") == pid)
    if mine.is_empty():
        st.caption("This scenario does not move him: the same points as the baseline, to rounding.")
    else:
        d = mine.row(0, named=True)
        ranks = (f"{position} rank {int(d['position_rank'])} → {int(d['new_position_rank'])}"
                 if d["position_rank"] is not None and d["new_position_rank"] is not None else "")
        st.caption(
            f"Against the baseline: {d['d_fantasy_points']:+.1f} points"
            + (f", {d['d_targets']:+.1f} targets" if d.get("d_targets") is not None else "")
            + (f", {d['d_carries']:+.1f} carries" if d.get("d_carries") is not None else "")
            + (f" · {ranks}" if ranks else "")
        )

TABS = ["🎛️ Ratings", "✎ Adjust", "📊 Stat line", "🗓️ Weeks", "🎲 Range", "📚 His record",
        "👥 Competition", "⚖️ Side by side"]
(ratings_tab, bench_tab, line_tab, weeks_tab, range_tab, record_tab, room_tab,
 vs_tab) = st.tabs(TABS)

# --------------------------------------------------------------------------- #
# 1. the ratings — the reason the page exists
# --------------------------------------------------------------------------- #
with ratings_tab:
    rated = ui.ratings(view, pid)
    if rated.is_empty():
        st.info("No estimates for him — a player with no history and no slot the estimator recognises.")
    else:
        kinds = {"share": "Shares — who gets the ball", "rate": "Rates — what he does with it",
                 "availability": "Availability — is he out there", "all": "All of them"}
        pick = st.segmented_control(
            "Ratings", list(kinds), default="all", key="ratings_kind",
            format_func=lambda k: kinds[k],
        ) or "all"
        part_of = rated if pick == "all" else rated.filter(pl.col("kind") == pick)

        ui.section("What the number is made of",
                   "The fill is what the projection ran on. The left tick is his own record, the right "
                   "tick is what his job is worth to everybody who has held that position and depth "
                   "slot, and the footer says how much sample is behind the first one. Where the two "
                   "ticks are far apart the weight is the whole argument: a wide gap with a low weight "
                   "is the estimator saying it does not believe his sample yet. ✎ changes it.")
        ui.rating_meters(view, pid, part_of, key="player:meters")

        with st.expander("The same estimates as a table"):
            FIRST = ["kind", "metric", "obs", "n", "prior", "used", "applied", "own_weight",
                     "override", "source", "units"]
            ui.table(part_of, config=ui.rating_config(), height=460,
                     order=[c for c in FIRST if c in part_of.columns]
                     + [c for c in part_of.columns if c not in FIRST])

        st.divider()
        ui.section("The full argument about one of them",
                   "The ✎ beside a meter is the fast path. This is the same edit with everything around "
                   "it: his record season by season with the opportunity behind each one, the position's "
                   "quartiles, and the rest of his room on the same metric — because a share is zero-sum "
                   "and raising his lowers theirs.")
        bases = {r["metric"]: float(r["used"]) for r in rated.rows(named=True)
                 if r.get("used") is not None}
        with st.container(border=True):
            ui.metric_knob(view, pid, bases, key="player:knob")
        ui.note(
            f"This sets the number. To see what changing it **does** — his own line, every teammate who "
            f"moves with him and the {team} totals, run through the engine before anything is written "
            "down — the **Team** page's 🔬 What if tab takes one number at a time and shows the ripple."
        )
        try:
            st.page_link("pages/2_Team.py", label="Team → 🔬 What if", icon="🔬")
        except Exception:                 # noqa: BLE001
            st.caption("Team → What if")

# --------------------------------------------------------------------------- #
# 2. every number of his at once
# --------------------------------------------------------------------------- #
with bench_tab:
    # Where he sits comes first, because it prices everything under it. A slot is not one of the ratings
    # in the editor below -- it is a rank, and setting it renumbers his whole room -- so it gets its own
    # control rather than a row that would look like a value and behave like an ordering.
    ui.section(
        "Where he sits on the chart",
        "Moving him is the right edit when the argument is *who is the starter*; overriding a share is "
        "the right edit when the argument is *how the work is split between two men who are both "
        "playing*. The first re-prices him off a starter's priors, the second does not.",
    )
    if not prow:
        st.caption("The estimator has no depth-chart row for him, so there is no slot to move.")
    else:
        room = (ui.participation(view)
                .filter((pl.col("team") == team) & (pl.col("position") == position))
                .sort("depth_slot"))
        # `base` comes off the baseline run, not off `prow`: once he has been moved, `prow` carries the
        # moved slot, and recording that as "what it was" would lose the published chart.
        was = ui.baseline(view).part.filter(pl.col("player_id") == pid)
        base_slot = float(was["depth_slot"][0]) if not was.is_empty() else float(prow["depth_slot"])
        slotted = next((o for o in ui.live().items
                        if o.level == "player" and o.key == pid
                        and o.field == overrides.DEPTH_FIELD), None)

        slot_cols = st.columns([4, 3, 1])
        with slot_cols[0]:
            ui.tiles([
                {"name": f"{position}{int(prow['depth_slot'])}",
                 "value": int(prow["depth_slot"]), "digits": 0, "highlight": True,
                 "sub": f"of {room.height} in the room"
                        + ("" if slotted is None else f" · was {base_slot:.0f}")},
                {"name": "priced off slot", "value": int(prow["slot_bucket"]), "digits": 0,
                 "sub": "the capped slot the priors are keyed on"},
            ])
        want = slot_cols[1].number_input(
            "Move him to", 1, max(room.height, 1), int(prow["depth_slot"]), 1,
            key=ui._keyed(f"player:slot:{pid}"),
            help="the room renumbers around him, and everybody it moves is re-priced",
        )
        if slot_cols[2].button("↺", key=ui._keyed(f"player:slot:{pid}:reset"), disabled=slotted is None,
                               help="follow the published chart again"):
            ui.drop_edit("player", pid, overrides.DEPTH_FIELD)
            st.rerun()
        if int(want) != int(prow["depth_slot"]):
            ui.edit(Override("player", pid, overrides.DEPTH_FIELD, "set", float(want), base=base_slot))
            st.rerun()
        ui.focus_table(
            room.select([c for c in ("depth_slot", "player", "status", "expected_games", "snap_share",
                                     "route_participation", "slot_bucket", "is_rookie", "active_weeks")
                         if c in room.columns]),
            ["depth_slot", "player", "status", "expected_games", "snap_share", "route_participation"],
            key=f"player:room:{pid}",
            config={**ui.fixed(1, "expected_games"),
                    **ui.percent("snap_share", "route_participation")},
            height=min(80 + 35 * room.height, 300),
            label_text="the rest of the room's facts",
        )

    # Off the roster entirely — not a slot and not a number, so it is not in the editor below either.
    # It is offered here because this is the page the disagreement usually starts on: you have opened a
    # man our sources still list on this team to find out what the projection thinks he is worth.
    st.divider()
    slot_now = (f"{position}{int(prow['depth_slot'])}" if prow.get("depth_slot") is not None
                else "his slot")
    off = st.columns([5, 2], vertical_alignment="center")
    off[0].markdown(
        f"**Not on {team} any more?** Take him off the roster and his row is removed before anything is "
        "estimated from it: the room closes up behind him, the men who move up are re-priced as what "
        "they moved up to, and his share of every pool goes to the players who are left. Zeroing his "
        f"games instead leaves him holding {slot_now} with the pool still divided around him."
    )
    if off[1].button("🚫 Off the roster", width="stretch",
                     key=ui._keyed(f"player:release:{pid}"),
                     help=f"remove him from {team} for the whole season"):
        ui.release(pid)
        # He has no row in the next run, so the picker's remembered label is about to stop existing.
        st.session_state.pop("player", None)
        st.rerun()
    ui.note(
        "This page has nothing to show for a man who is off the roster — there is no projection of him "
        "to read — so it opens on somebody else afterwards, and **Team → 🧭 Depth chart** is where the "
        "team's releases are listed and undone. For a single week he misses, set `chance he plays` to 0 "
        "in **Week → 🎛️ Adjust this game** rather than removing him."
    )

    st.divider()
    # the meters next door argue about one estimate at a time, with the evidence beside it. This is the
    # other need: he is wrong in four places and it is already known which four -- so type over four
    # numbers, for the season or for chosen weeks, and pay for one engine run instead of four
    ui.section(
        "Change several of his at once",
        "Type over as many of them as are wrong — for the season, or for the weeks he is hurt in — and "
        "apply them together. Nothing is computed until you do, so six edits cost one run of the season "
        "rather than six, and **Discard** forgets what is typed without touching the scenario. It is the "
        "same editor the ✎ opens from every other page.",
    )
    ui.bench(view, pid, key="player:bench")

# --------------------------------------------------------------------------- #
# 3. the stat line
# --------------------------------------------------------------------------- #
with line_tab:
    # Points first, because that is what a league pays; the football line under it, because that is what
    # the points are made of and the only part of the projection a change of scoring cannot move.
    ui.section("Projected stat line",
               "The small number under each total is per projected game played, not per 17: a player "
               "expected to miss three weeks is scored over the 14 he is there for, which is the number "
               "a lineup decision needs. The totals themselves already carry his availability.",
               sub=ui.stat_line(me, position))
    played = max(float(weekly["p_play"].sum()), 1e-9)
    per_game = {c: float(weekly[c].sum()) / played for c in ui.ALL_STATS if c in weekly.columns}
    ui.stat_blocks(me, position, per_game=per_game)

    with st.expander("The same line as a table"):
        ui.table(
            pl.DataFrame({
                "stat": stats,
                "season": [float(weekly[c].sum()) for c in stats],
                "per_game": [per_game[c] for c in stats],
            }),
            config=ui.fixed(2, "season", "per_game"),
        )

# --------------------------------------------------------------------------- #
# 4. the seventeen games
# --------------------------------------------------------------------------- #
with weeks_tab:
    # The same run the 🎲 tab's switch asked for -- read rather than re-offered, so the band and the
    # season range can never be two different simulations.
    sim_weeks = (ui.simulation(view, draws=ui.sim_draws("player_sim"), detail=(pid,))
                 if ui.sim_on() else None)
    games = weekly.select(
        "week", "opponent", "is_home", "p_play", *stats, "fantasy_points"
    ).join(
        ui.environment(view).filter(pl.col("team") == team)
        .select("week", "spread", "total", "implied_points", "has_market"),
        on="week", how="left",
    )
    band = None
    if sim_weeks is not None and not sim_weeks.weekly.is_empty():
        wr = sim_weeks.weekly.filter(pl.col("player_id") == pid).select(
            "week", pl.col("p5").alias("week_p5"), pl.col("p50").alias("week_p50"),
            pl.col("p95").alias("week_p95"), pl.col("boom_rate").alias("week_boom"),
            pl.col("bust_rate").alias("week_bust"),
        )
        games = games.join(wr, on="week", how="left")
        band = ("week_p5", "week_p95")

    ui.section("Week by week",
               "Each bar is an expectation, availability included: a bye week is absent and a week he is "
               "only 60% likely to play is 60% of a line, not a full one. With ranges on, the shaded "
               "band behind the bars is the 5th to 95th percentile of the simulated week — which is what "
               "a start/sit call actually turns on, because a 12-point expectation that is 4-to-26 is a "
               "different decision from one that is 10-to-14.")
    ui.week_bars(games, band=band)
    ui.focus_table(
        games, ["week", "opponent", "is_home", "p_play", "fantasy_points", "implied_points", "spread",
                *stats[:4]], key="player:weeks", height=520,
        config={**ui.fixed(2, "fantasy_points", "p_play", *stats),
                **ui.fixed(1, "spread", "total", "implied_points"),
                **ui.range_config(per_game=True),
                **ui.fixed(2, "week_p5", "week_p50", "week_p95"),
                **ui.percent("week_boom", "week_bust")},
    )

# --------------------------------------------------------------------------- #
# 5. the range around it
# --------------------------------------------------------------------------- #
with range_tab:
    ui.section("The range around it",
               "Ten thousand seasons of the frame on this page, shocked at three levels: the game both "
               "teams are in, his team's volume, then his own share and his own efficiency. Every "
               "dispersion is measured from 2016–2025 residuals rather than assumed, and the shocks are "
               "correlated, so his ceiling arrives with his quarterback's.")
    sim = ui.sim_controls(view, detail=(pid,), key="player_sim")
    if sim is None:
        st.info("Turn on **Ranges** to simulate the season and see his floor, ceiling and volatility.")
    else:
        mine = sim.season.filter(pl.col("player_id") == pid)
        if mine.is_empty():
            st.info("He is not in the simulated frame — no projected volume to shock.")
        else:
            r = mine.row(0, named=True)
            lo, hi = sim.dispersion.thresholds(position)
            ui.tiles([
                {"name": "median", "value": r["p50"], "digits": 0, "sub": f"mean {r['sim_mean']:.0f}"},
                {"name": "floor · P5", "value": r["p5"], "digits": 0, "sub": f"P25 {r['p25']:.0f}"},
                {"name": "ceiling · P95", "value": r["p95"], "digits": 0, "sub": f"P75 {r['p75']:.0f}",
                 "highlight": True},
                {"name": "volatility", "value": r["volatility"], "digits": 2, "percent": True,
                 "sub": f"rank by median {int(r['median_rank'])}"},
                {"name": "boom weeks", "value": r["boom_rate"], "digits": 2, "percent": True,
                 "sub": f"over {lo:.0f}"},
                {"name": "bust weeks", "value": r["bust_rate"], "digits": 2, "percent": True,
                 "sub": f"under {hi:.0f}"},
            ])
            st.caption("The simulated season — each bar is the share of seasons landing in that band. "
                       "The bar at zero is the seasons he never took the field in.")
            st.bar_chart(sim.histogram(pid), x="points", y="share", height=260)

            over, under = st.columns([2, 5])
            with over:
                line = st.number_input("P(over) points", min_value=0.0,
                                       value=float(round(r["p50"], 0)), step=5.0,
                                       help="Odds of clearing this many points over the season.")
            with under:
                st.metric(f"P(over {line:.0f})", f"{sim.p_over(pid, float(line)):.1%}", border=True)
                st.caption(
                    f"Read against a games range of {r['games_p5']:.0f}–{r['games_p95']:.0f} "
                    f"(median {r['games_p50']:.0f}) against {r['projected_games']:.1f} projected."
                )

            ui.section("The ten nearest him at the position",
                       "Where his `floor_rank` beats his `ceiling_rank` he is the safe pick of the "
                       "group; where it is the other way round he is the one you take for the upside.")
            board_rank = sim.season.filter(pl.col("position") == position).select(
                "player", "team", "projected", "p50", "floor", "ceiling", "volatility",
                "median_rank", "floor_rank", "ceiling_rank",
            ).sort("median_rank")
            near = board_rank.with_columns(
                (pl.col("median_rank") - int(r["median_rank"])).abs().alias("near")
            ).sort("near").head(11).sort("median_rank").drop("near")
            ui.table(near.with_columns((pl.col("player") == me["player"]).alias("this_player")),
                     config=ui.range_config())

            if not sim.stats.is_empty():
                with st.expander("Each stat, with its own range"):
                    ui.table(sim.stats.filter(pl.col("player_id") == pid).drop("player_id"),
                             config=ui.range_config(), height=380)

# --------------------------------------------------------------------------- #
# 6. what he has actually done
# --------------------------------------------------------------------------- #
with record_tab:
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

        ui.section("His record, and the projection on the end of it",
                   "Points per game rather than totals, so a season cut short by injury is not read as "
                   "a decline in ability. The red bar is the projection, on the same scoring as the "
                   "measured seasons beside it.")
        ui.season_bars(both.with_columns(pl.col("season").cast(pl.String)), "season",
                       "points_per_game", mark=str(view.season), digits=2)
        stat_pick = st.selectbox("Draw another stat", ["points_per_game", "fantasy_points", *hcols],
                                 index=0, format_func=ui.label, key="player:hist:stat")
        if stat_pick != "points_per_game":
            ui.season_bars(both.with_columns(pl.col("season").cast(pl.String)), "season", stat_pick,
                           mark=str(view.season), digits=ui.digits_for(stat_pick, 1))
        ui.focus_table(both, ["season", "team", "games", "fantasy_points", "points_per_game"],
                       key="player:record", height=280,
                       config={**ui.fixed(1, "fantasy_points", "games", *stats),
                               **ui.fixed(2, "points_per_game")})

    st.divider()
    ui.section("Availability, in full",
               "`expected_games` is his own games history blended with his slot's, times `presence` (is "
               "a man in this slot on the field at all) times a stated factor for his roster status. "
               "`active_weeks` is that as a fraction of the season, and it multiplies every share before "
               "the team's pool is divided.")
    if prow:
        ui.tiles([
            {"name": "projected games", "value": prow.get("expected_games"),
             "sub": f"of {ui.REG_WEEKS - 1}", "highlight": True},
            {"name": "his own average", "value": prow.get("obs_games"),
             "sub": f"{ui.fmt_num(prow.get('n_seasons'), 0)} seasons"},
            {"name": "his slot's average", "value": prow.get("prior_games")},
            {"name": "weight on his own", "value": prow.get("games_own_weight"), "digits": 2,
             "percent": True},
            {"name": "chance the job exists", "value": prow.get("presence"), "digits": 2,
             "percent": True},
            {"name": "assumed availability", "value": prow.get("status_factor"), "digits": 2,
             "percent": True, "sub": str(prow.get("status") or "")},
        ])
        left, right = st.columns([1, 11])
        with left:
            ui.knob_popover(view, pid, ui.GAMES_METRIC, base=prow.get("expected_games"),
                            key="player:games")
        with right:
            st.caption("✎ sets his projected games — the one player number that is not per-game.")
    else:
        st.info("No participation row for him.")

# --------------------------------------------------------------------------- #
# 7. who he is competing with
# --------------------------------------------------------------------------- #
with room_tab:
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
    ui.section(f"The {team} {pools[0]} pool",
               f"Season totals for everyone drawing on the same pools ({', '.join(pools)}). The shares "
               "are availability-weighted and post-normalisation, so they are what actually divided the "
               "team's total — not the estimator's belief about a man on the field.")
    ui.rank_bars(mates.head(12), f"share_{pools[0]}", height=max(240, 26 * min(mates.height, 12)),
                 digits=3)
    ui.table(
        mates.with_columns((pl.col("player_id") == pid).alias("this_player")).drop("player_id"),
        config={**ui.percent("p_play", *[f"share_{p}" for p in pools]), **ui.fixed(1, *pools)},
        height=420,
    )

    with st.expander("The depth chart this came from"):
        roster = ui.roster_frame(view)
        ui.table(
            roster.filter(pl.col("team") == team).select(
                [c for c in ("position", "depth_tier", "depth_slot", "slot_bucket", "player", "status",
                             "is_rookie", "draft_pick", "years_exp", "age", "charted",
                             "team_disagreement", "alignment") if c in roster.columns]
            ).sort(["position", "depth_slot"]), height=520,
        )

# --------------------------------------------------------------------------- #
# 8. him against one to three others
# --------------------------------------------------------------------------- #
# This was a page of its own, and a page is the wrong place for it: nobody opens a comparison cold, they
# arrive at one from a man they are already reading. So it is a tab of his, he is always one side of it,
# and the layers go narrowest first -- the line, then the estimates the line is made of with the sample
# behind each, then the weeks, the record and the range.
with vs_tab:
    ui.section(
        "Him against one to three others",
        "A board is read as an order, but a decision is made between two or three men at a time — and "
        "what a total cannot settle is which of them the model actually knows something about. Two "
        "receivers at 210 points are not the same bet if one is a measurement over four seasons and the "
        "other is a depth slot's prior wearing a name.",
        sub=f"{me['player']} against the men picked here, in the order picked",
    )
    pool = ui.board(view).filter(pl.col("player_id") != pid)
    vs_labels = {
        r["player_id"]: (f"{r['player']}  ·  {r['position']} {r['team']}  ·  "
                         f"{float(r['fantasy_points']):.0f} pts")
        for r in pool.rows(named=True)
    }
    # the man nearest him at his own position is the comparison he came for nine times in ten, so it is
    # the one already made; the pick sticks once it has been changed, because it carries a key
    near = pool.filter(pl.col("position") == position)
    if not near.is_empty() and me.get("position_rank") is not None:
        near = near.with_columns(
            (pl.col("position_rank") - float(me["position_rank"])).abs().alias("gap")
        ).sort("gap")
    others = st.multiselect(
        "Against", list(vs_labels), default=near["player_id"].to_list()[:1], max_selections=3,
        format_func=lambda p: vs_labels.get(p, p), key="player:vs",
        help="one to three other men; every table below puts them beside him in the order picked",
    )

    if not others:
        st.info(f"Pick at least one other player to put beside {me['player']}.")
    else:
        ids = tuple(p for p in (pid, *others) if p in vs_labels or p == pid)
        names = ui.display_names(view, ids)
        lines = ui.compare_lines(view, ids)
        rows = {r["player_id"]: r for r in ui.board(view)
                .filter(pl.col("player_id").is_in(list(ids))).rows(named=True)}
        ids = tuple(p for p in ids if p in rows)

        # ----------------------------------------------------------------- #
        # the line
        # ----------------------------------------------------------------- #
        best = max(float(r["fantasy_points"]) for r in rows.values())
        cards = st.columns(len(ids))
        for col, who_id in zip(cards, ids, strict=True):
            r = rows[who_id]
            gap = float(r["fantasy_points"]) - best
            with col:
                # one bordered card a man, so four men are four things to read rather than sixteen
                # metrics in a row: the tiles wrap inside the column
                with st.container(border=True):
                    st.markdown(f"### {r['player']}")
                    ui.chips(
                        f"{r['position']} {r['team']}",
                        f"slot {int(r['depth_slot'])}",
                        str(r["status"]) if r.get("status") not in (None, "ACT") else "",
                        rookie=bool(r.get("is_rookie")), changed_team=bool(r.get("changed_team")),
                        startable=bool(r.get("startable")),
                        edited=bool(ui.live().touching("player", who_id)),
                    )
                    ui.tiles([
                        {"name": "fantasy points", "value": r["fantasy_points"], "highlight": True,
                         "sub": "best on this list" if gap == 0 else f"{gap:+.1f} vs best"},
                        {"name": "per game", "value": r["points_per_game"], "digits": 2,
                         "sub": f"over {r['games']:.1f} games"},
                        {"name": f"{r['position']} rank", "value": r["position_rank"], "digits": 0,
                         "sub": f"{int(r['overall_rank'])} overall"},
                        {"name": "vs average starter", "value": r["vs_starter"], "signed": True,
                         "sub": (f"drop to next {r['drop_next']:.1f}"
                                 if r.get("drop_next") is not None else "")},
                    ])
                    st.markdown(f"`{ui.stat_line(r, r['position'])}`")
                    # a comparison is where a disagreement gets noticed, so it is also where one gets
                    # settled
                    ui.edit_button(who_id, key=f"player:vs:bench:{who_id}", width="stretch")

        ui.section("The gap, drawn",
                   "The same totals on one axis. Two men within a point of each other are a coin toss "
                   "whatever the order of the board says, and thirty points apart is a decision — which "
                   "is a picture rather than a subtraction.",
                   sub="fantasy points, all of them against the best on the list")
        ui.meters([
            ui.meter_html(
                rows[p]["player"], float(rows[p]["fantasy_points"]), maximum=best * 1.05, digits=1,
                foot=(f"{rows[p]['points_per_game']:.1f} a game over {rows[p]['games']:.1f} games",
                      f"{rows[p]['position']} {int(rows[p]['position_rank'])}"),
            )
            for p in ids
        ])

        ui.section("The projected line",
                   "The same columns for each of them, so a difference in points can be traced to a "
                   "difference in volume or in efficiency rather than taken on faith.")
        ui.table(lines, height=min(80 + 40 * lines.height, 320),
                 config={**ui.stat_config([c for c in lines.columns if c in ui.ALL_STATS]),
                         **ui.fixed(1, "fantasy_points", "games", "vs_starter", "drop_next",
                                    "delta_points")})

        if len({r["position"] for r in rows.values()}) > 1:
            ui.note("They are not all at the same position: the stat line is the union of both, so a "
                    "column that does not apply to a man is empty rather than zero, and a positional "
                    "rank is inside his own position.")

        # ----------------------------------------------------------------- #
        # where the difference comes from — the layer the tab exists for
        # ----------------------------------------------------------------- #
        ui.section(
            "Where the difference comes from",
            "Every estimate they have in common, with the sample size under each. `n` is the whole "
            "argument: two men on the same share are not the same claim if one of them has held it over "
            "900 routes and the other over 90 — the second is mostly his depth slot's prior, and it is "
            "the one to argue with.",
            sub="the ratings under the totals, not the totals",
        )
        rated_vs = ui.compare_ratings(view, ids)
        if rated_vs.is_empty():
            st.info("No estimates in common — a player the estimator has no metrics for.")
        else:
            vs_kinds = (sorted(rated_vs["kind"].drop_nulls().unique().to_list())
                        if "kind" in rated_vs.columns else [])
            vs_pick = st.radio("Which estimates", ["all", *vs_kinds], horizontal=True,
                               key="player:vs:kind",
                               format_func=lambda k: {"share": "Shares — who gets the ball",
                                                      "rate": "Rates — what he does with it",
                                                      "availability": "Availability — is he out there",
                                                      "all": "All of them"}.get(k, k))
            shown = (rated_vs if vs_pick == "all" or "kind" not in rated_vs.columns
                     else rated_vs.filter(pl.col("kind") == vs_pick))
            who = [names[p] for p in ids if names.get(p) in shown.columns]

            # One meter a metric with both men on the same track. Two receivers on 0.22 and 0.19 of the
            # targets is a two-row table nobody reads twice; the same thing drawn -- with `n` under each
            # bar -- is the argument about which of the two the model knows.
            METER_CAP = 12
            over = shown.height - METER_CAP
            ui.section(
                "The same estimates, drawn",
                "A share is drawn against 1.0, because 1.0 is what a share can be. A rate is drawn "
                "against whichever of them is higher, because there is no natural ceiling on yards a "
                "carry — so read a rate meter as *the gap between these two* and a share meter as *how "
                "much of the pool*. The number under each bar is the sample it was measured over.",
                sub=(f"the first {METER_CAP} of {shown.height} · the rest are in the table below"
                     if over > 0 else f"{shown.height} estimates in common"),
            )
            drawn = shown.head(METER_CAP).rows(named=True)
            for i in range(0, len(drawn), 2):
                pair = st.columns(2, gap="medium")
                for col, m in zip(pair, drawn[i:i + 2], strict=False):
                    share = m.get("kind") == "share"
                    vals = [(w, m.get(w)) for w in who]
                    numbers = [abs(float(v)) for _, v in vals if v is not None]
                    top = 1.0 if share else (max(numbers) * 1.15 if numbers else 1.0)
                    with col:
                        st.caption(f"**{m['metric']}**"
                                   + (f" · {m['units']}" if m.get("units") else "")
                                   + (f" · {m['kind']}" if m.get("kind") else ""))
                        ui.meters([
                            ui.meter_html(
                                w, None if v is None else float(v), maximum=top,
                                digits=3 if share else 2, percent=share,
                                foot=(f"n {float(m[f'n · {w}']):,.0f} {m.get('units') or ''}".strip()
                                      if m.get(f"n · {w}") is not None else "",),
                            )
                            for w, v in vals
                        ])

            with st.expander("The same estimates as a table — every metric, every sample size"):
                ui.table(
                    shown, digits=3, height=min(80 + 35 * shown.height, 560),
                    config={**ui.fixed(3, *who), **ui.fixed(0, *[f"n · {w}" for w in who])},
                )
                ui.note("`units` says what `n` is counted in — routes, dropbacks, carries, seasons — "
                        "because a sample size means nothing without it.")

        # ----------------------------------------------------------------- #
        # the season, and the record
        # ----------------------------------------------------------------- #
        ui.section("Week by week",
                   "A gap in a line is a bye, not a zero. Where two lines cross a bye in different "
                   "weeks, the season totals are comparable and the two Novembers are not.",
                   sub="the seventeen games the season is the sum of")
        paths = ui.compare_weeks(view, ids)
        if paths.is_empty():
            st.info("No weekly rows for these players.")
        else:
            st.line_chart(paths, x="week", height=300)
            with st.expander("The same weeks as a table"):
                ui.table(paths, digits=2, height=520)

        vs_hist = ui.compare_history(view, ids)
        ui.section(
            "What they have actually done",
            f"Points per game, on the scoring in the sidebar, with {view.season} on the end as the "
            "projection. Per game rather than totals, so a season cut short is not read as a decline in "
            "ability — the games themselves are the availability argument, and they are in 🚑 Availability "
            "on the **Team** page and on the **Availability** page.",
            sub="the record beside the projection, season by season",
        )
        if vs_hist.is_empty():
            st.info("Neither of them has a season in the processed tables.")
        else:
            st.line_chart(vs_hist, x="season", height=280)
            with st.expander("The same seasons as a table"):
                ui.table(vs_hist, digits=2, height=280)

        with st.expander("Availability, side by side"):
            avail = ui.availability_board(view).filter(pl.col("player_id").is_in(list(ids)))
            ui.table(
                avail.select([c for c in ("player", "position", "team", "depth_slot", "status",
                                          "expected_games", "engine", "obs", "n_seasons",
                                          "own_weight", "prior", "weeks_out", "worst_season_out",
                                          "main_injury", "report") if c in avail.columns]),
                digits=1, height=220,
                config={**ui.bar("own_weight"), **ui.sparkline("report", "weeks out")},
            )
            ui.note("`weeks out` is what the league's own injury reports said, season by season. It is "
                    "not an input to the projection — games played is — but it is the reason a games "
                    "number is or is not believable.")

        # ----------------------------------------------------------------- #
        # the range, on the switch in the 🎲 tab rather than a second one
        # ----------------------------------------------------------------- #
        ui.section("The range around each of them",
                   "Ten thousand seasons of the same weekly frame. The comparison a total cannot make: "
                   "a man with the higher median and the lower floor is a different pick, not a better "
                   "one.",
                   sub="floors and ceilings, on the **Ranges** switch in the 🎲 Range tab")
        vs_sim = (ui.simulation(view, draws=ui.sim_draws("player_sim"), detail=(pid,))
                  if ui.sim_on() else None)
        if vs_sim is None:
            st.info("Turn on **Ranges** — the switch is in the 🎲 Range tab — to compare floors and "
                    "ceilings.")
        else:
            mine = vs_sim.season.filter(pl.col("player_id").is_in(list(ids)))
            if mine.is_empty():
                st.info("None of them is in the simulated frame — no projected volume to shock.")
            else:
                # the floor, the median and the ceiling on one axis: the three meters *are* the
                # comparison, and the man whose floor bar is shortest is the one the median is lying about
                ranked = {r["player_id"]: r for r in mine.rows(named=True)}
                axis = max(float(r["p95"]) for r in ranked.values()) * 1.05
                for band, column, why in (("floor · P5", "p5", "the season if it goes wrong"),
                                          ("median · P50", "p50", "the middle season"),
                                          ("ceiling · P95", "p95", "the season if it goes right")):
                    st.caption(f"**{band}** — {why}")
                    ui.meters([
                        ui.meter_html(names.get(p, p), float(ranked[p][column]), maximum=axis, digits=0,
                                      marks=(("projected", ranked[p].get("projected")),),
                                      foot=(f"volatility {float(ranked[p]['volatility']):.0%}"
                                            if ranked[p].get("volatility") is not None else "",))
                        for p in ids if p in ranked
                    ])
                for p in ids:
                    if p not in ranked:
                        continue
                    st.caption(f"{names.get(p, p)} — the simulated season. Each bar is the share of "
                               "seasons landing in that band, and the bar at zero is the seasons he "
                               "never took the field in — for a fragile player that bar is the "
                               "comparison.")
                    st.bar_chart(vs_sim.histogram(p), x="points", y="share", height=200)
                with st.expander("The simulated season as a table, with the ranks"):
                    ui.table(mine.select([c for c in ui.RANGE_COLUMNS if c in mine.columns]
                                         + [c for c in ("median_rank", "floor_rank", "ceiling_rank")
                                            if c in mine.columns]),
                             config=ui.range_config(), height=min(80 + 40 * mine.height, 300))

        # ----------------------------------------------------------------- #
        # and the edits on them
        # ----------------------------------------------------------------- #
        touched = [p for p in ids if ui.live().touching("player", p)]
        if touched:
            ui.section("Your overrides on these players",
                       "Every number above includes them. Drop one with ↺ and the comparison re-prices "
                       "on the next rerun.",
                       sub=f"{len(touched)} of the {len(ids)} men on this tab carry one")
            for p in touched:
                st.markdown(f"**{names.get(p, p)}**")
                ui.edit_list("player", p, v=view, key="player:vs")
        else:
            ui.note("No overrides on either of them: every number above is the engine's own answer. The "
                    "knobs are in 🎛️ Ratings and ✎ Adjust, beside the estimate each one changes.")
