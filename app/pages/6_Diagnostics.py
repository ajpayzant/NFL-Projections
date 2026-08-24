"""Whether to believe any of it: what the model got wrong last time, and what it is reading now.

Every other page shows projections. This one shows the evidence for them, and it is arranged so that a
number a user distrusts can be traced to the measurement that justifies it — or to the absence of one.
Four questions, in the order they matter:

1. **Was the board right on seasons it had not seen?** The backtest, against two baselines: the
   player's own recency-weighted average, and what the Excel workbook actually did.
2. **Are the ranges honest?** A projection is a mean and a range is a second promise. Interval
   calibration says whether the advertised floor was really missed one season in twenty.
3. **Do the books balance?** Every team's targets go to exactly one player, so the shares have to sum
   to the pool. This is the audit the normalisation toggle is standing next to.
4. **Is the data current?** The age of every table the engine read, and — the question age alone
   cannot answer — whether `processed/` has kept up with the week the season is actually in.

**The backtest is read, not run.** It is minutes of composition per variant and it is a command-line
job (`python -m src.model.backtest`) whose output is an artifact. This page reports what the last run
found and how old that run is, because a page that recomputed it is a page nobody would wait for.

`streamlit run app/Home.py` and pick Diagnostics.
"""

from __future__ import annotations

import polars as pl
import streamlit as st

import ui
from src.config import PROJ_SEASON

view = ui.controls("Diagnostics", icon="🔬")
ui.scenario_banner(view)

tabs = st.tabs(["Backtest", "Calibration", "Intervals", "Share sums", "Data freshness"])

# --------------------------------------------------------------------------- #
# 1. the backtest
# --------------------------------------------------------------------------- #
with tabs[0]:
    summary = ui.backtest_summary()
    age = ui.backtest_age()
    if summary.is_empty():
        st.info("No backtest on file. Run `python -m src.model.backtest` — it writes "
                "`data/fitted/backtest_summary.parquet` and this tab reads it.", icon="📉")
    else:
        seasons = sorted(s for s in summary["season"].unique().to_list() if s)
        st.caption(f"held-out seasons {seasons[0]}–{seasons[-1]} · "
                   + (f"run {age:.0f} days ago" if age is not None else "age unknown"))
        if age is not None and age > 30:
            st.warning(f"The backtest is {age:.0f} days old. If the engine has changed since, these "
                       f"numbers describe a model that no longer exists.", icon="⚠️")

        view_cut = st.selectbox(
            "Population", ["all", "played", "regulars", "starters"], index=0,
            help="`all` is every player the projection named — points given to somebody who never "
                 "played count as error. `starters` is selected on the outcome, so it says whether the "
                 "board got the right people, not whether it was calibrated.",
        )
        pooled = summary.filter(
            (pl.col("season") == 0) & (pl.col("view") == view_cut)
            & (pl.col("stat") == "fantasy_points")
        ).sort("mae")
        st.subheader("Season fantasy points, pooled over every held-out season")
        ui.table(
            pooled.select("variant", "n", "mae", "rmse", "bias", "spearman"),
            config=ui.fixed(2, "mae", "rmse", "bias") | ui.fixed(3, "spearman"),
        )
        ui.note(
            "`full` is the engine. `ewma` is the player's own recency-weighted season average — the "
            "honest floor, and hard to beat for an established starter. `workbook` is what the "
            "spreadsheet this replaces actually did. The rest turn off one node each, so a node that "
            "helps its own metric and hurts the board is visible here. Judge on all three of MAE, RMSE "
            "and rank correlation: this population is four-tenths zeroes and heavily right-skewed, so "
            "shaving the level off everybody lowers MAE while making every startable player worse."
        )

        st.subheader("Per season")
        per = summary.filter(
            (pl.col("season") != 0) & (pl.col("view") == view_cut)
            & (pl.col("stat") == "fantasy_points")
        )
        wide = per.select("variant", "season", "mae").pivot(
            on="season", index="variant", values="mae"
        ).sort("variant")
        ui.table(wide, digits=2)

        st.subheader("By stat")
        stats = summary.filter(
            (pl.col("season") == 0) & (pl.col("view") == view_cut)
            & pl.col("variant").is_in(["full", "ewma", "workbook"])
        ).select("stat", "variant", "mae").pivot(on="variant", index="stat", values="mae")
        ui.table(stats, digits=2)

# --------------------------------------------------------------------------- #
# 2. calibration
# --------------------------------------------------------------------------- #
with tabs[1]:
    cal = ui.backtest_calibration()
    if not cal:
        st.info("No backtest on file, so there is nothing to calibrate against. Run "
                "`python -m src.model.backtest`.", icon="📉")
    else:
        st.subheader("Is the level right, given what the projection said?")
        ui.note(
            "MAE says how close the projections land; this says whether they are **tilted**. "
            "Regressing the outcome on the projection separates the two errors MAE cannot tell apart: "
            "a slope of 1 is the claim, below 1 means the model commits harder than the evidence "
            "supports and the extremes come back toward the middle, above 1 means it hedged."
        )
        ui.table(
            cal["lines"].select("position", "population", "n", "slope", "intercept",
                               "mean_err", "median_err"),
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

        st.subheader("By projected-games band")
        ui.note(
            "Cut on **projected** games, never on realised games: conditioning on having appeared keeps "
            "only the players who beat the availability the model applied, which shifts every error in "
            "the table. `never_played` states that same fact without selecting on it."
        )
        ui.table(
            cal["bands"].select("band", "n", "proj_mean", "act_mean", "mean_err", "median_err",
                                "never_played"),
            config=ui.fixed(1, "proj_mean", "act_mean", "mean_err", "median_err")
            | ui.percent("never_played"),
        )
        ui.note(
            "The known residual, and it is second-order: across the 5–11 projected-games band the mean "
            "is over-projected by roughly 6–12 points, offset by a small under-projection across the "
            "deep bench. Total volume is conserved by pool normalisation, so this is a distributional "
            "error along the games axis rather than a level error. It is documented here rather than "
            "corrected because every candidate correction tested either had the wrong sign in aggregate "
            "or traded the mid-band for the tail."
        )

# --------------------------------------------------------------------------- #
# 3. intervals
# --------------------------------------------------------------------------- #
with tabs[2]:
    disp = ui.dispersion()
    st.caption(ui.sim_provenance())
    if not disp.meta.get("fitted"):
        st.warning("The dispersion is not fitted, so the ranges on the other pages are pooled "
                   "defaults rather than measured. Run `python -m src.model.simulate --fit`.",
                   icon="⚠️")
    else:
        cover = disp.meta.get("coverage") or {}
        if cover:
            st.subheader("Did the advertised range contain the season?")
            cols = st.columns(4)
            cols[0].metric("P5–P95 coverage", f"{cover.get('cover_90', float('nan')):.3f}",
                           delta=f"{cover.get('cover_90', 0) - 0.90:+.3f} vs 0.90")
            cols[1].metric("P25–P75 coverage", f"{cover.get('cover_50', float('nan')):.3f}",
                           delta=f"{cover.get('cover_50', 0) - 0.50:+.3f} vs 0.50")
            cols[2].metric("PIT interquartile range", f"{cover.get('pit_iqr', float('nan')):.3f}",
                           delta=f"{cover.get('pit_iqr', 0) - 0.5:+.3f} vs 0.500")
            cols[3].metric("Median bias", f"{cover.get('median_bias', float('nan')):+.1f}")
            below, above = cover.get("below"), cover.get("above")
            if below is not None and above is not None:
                ui.note(
                    f"{below:.1%} of held-out seasons finished **below** their P5 and {above:.1%} "
                    f"**above** their P95, against 5% expected at each end. A floor missed more often "
                    f"than that is a floor that was never a floor."
                )
            st.info(
                f"The scale was fitted on the **interquartile range of the PIT**, which is invariant to "
                f"a shift and therefore scores only the width. The alternative — a distance-from-uniform "
                f"score — cannot tell a bad width from a bad centre, and will widen an interval to "
                f"absorb a level error. Whatever location error is left stays visible as `median_bias` "
                f"above, where it is the projection's to answer for rather than the interval's. The "
                f"population is projected-startable seasons (over "
                f"{disp.meta.get('calibration_min_projected', 0):.0f} points), cut on the projection and "
                f"never on the outcome.", icon="📐",
            )

        st.subheader("Fitted width per position")
        ridge = disp.meta.get("scale_ridge") or {}
        ui.table(pl.DataFrame([
            {"position": pos, "scale": scale,
             "indistinguishable": ", ".join(f"{x:g}" for x in ridge.get(pos, [])),
             "games bias (starter)": (disp.meta.get("games_bias") or {}).get(f"{pos}:starter"),
             "games bias (depth)": (disp.meta.get("games_bias") or {}).get(f"{pos}:depth")}
            for pos, scale in sorted(disp.scale.items())
        ]), config=ui.fixed(3, "games bias (starter)", "games bias (depth)"))
        ui.note(
            "`scale` multiplies every persistent sigma for the position. `indistinguishable` is the set "
            "of grid values whose calibration loss is within tolerance of the best — where it holds more "
            "than one value, the fit is not claiming precision it does not have."
        )
        seen = disp.meta.get("calibration_targets") or []
        if seen:
            ui.note(f"Fitted on {seen[0]}–{seen[-1]}. The projection driving those intervals is ex "
                    f"ante, but the width is in sample: measured by refitting without the last season, "
                    f"the in-sample advantage is worth about 0.007 of coverage.")

# --------------------------------------------------------------------------- #
# 4. share sums
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Do the players add up to a whole offence?")
    ui.note(
        "A team's targets in a game are a fixed quantity and every one goes to exactly one player, so "
        "Σ over players of P(he plays) × his share must be 1 — or rather, must be the total actually "
        "observed over 2016–2025, which for some pools is not 1: `receiving_tds` counts a two-point "
        "conversion the touchdown pool does not. `gap_pct` is how far the estimated shares were from "
        "the pool *before* anything was rescaled, and it is the honest headline. The workbook this "
        "replaces warned when shares exceeded 100% and then left them there."
    )
    pools = ui.pool_report(view)
    ui.table(
        pools.drop("team_col"),
        config=ui.fixed(3, "measured_target", "raw_mean", "raw_min", "raw_max", "after_mean")
        | ui.fixed(2, "count_per_game", "gap_pct"),
        height=420,
    )
    if not view.settings.normalize_pools:
        st.warning("Normalisation is **off** in this scenario, so the counts on every other page are "
                   "the raw shares — the pools do not balance and team totals will not reconcile.",
                   icon="⚠️")

    st.subheader("Players against the team they were divided from")
    ui.note(
        "Counts drawn from a normalised pool must match to rounding. Yards and completions are a count "
        "times a rate and are deliberately *not* forced to match: a team whose receivers are all more "
        "efficient than its recent offence really should project above the team's own yardage, and "
        "scaling that away would hide the disagreement instead of showing it. `from_pool` says which "
        "rows are held to the strict standard."
    )
    rec = ui.reconciliation(view)
    ui.table(
        rec.sort("from_pool", descending=True),
        config=ui.fixed(2, "players_per_game", "team_per_game", "gap_pct")
        | ui.fixed(4, "worst_abs_gap"),
        height=400,
    )
    strict = rec.filter(pl.col("from_pool") & (pl.col("worst_abs_gap") > 0.01))
    if strict.is_empty():
        st.success("Every pool-derived count reconciles to within 0.01 of a single event per team-game.",
                   icon="✅")
    else:
        st.error(f"These come from a normalised pool and do not reconcile: "
                 f"{', '.join(strict['stat'].to_list())}. That is a defect, not a tolerance.",
                 icon="🚨")

    st.subheader("Which rosters the estimator struggles with")
    ui.note("A team summing short has a job its depth chart does not name — a vacated target share "
            "nobody has inherited — which is a real finding about the roster rather than a defect in "
            "the scaling.")
    tp = ui.team_pools(view)
    pool_pick = st.selectbox("Pool", sorted(tp["pool"].unique().to_list()),
                             index=sorted(tp["pool"].unique().to_list()).index("targets")
                             if "targets" in tp["pool"].to_list() else 0)
    worst = tp.filter(pl.col("pool") == pool_pick).sort("gap_pct")
    ui.table(worst, config=ui.fixed(3, "measured_target", "raw_sum", "factor")
             | ui.fixed(1, "gap_pct"), height=380)

# --------------------------------------------------------------------------- #
# 5. freshness
# --------------------------------------------------------------------------- #
with tabs[4]:
    stale = ui.staleness()
    st.subheader(f"Is the lake current for {PROJ_SEASON}?")

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

    if stale["tables"]:
        ui.table(pl.DataFrame([
            {"table": f"processed/{k}", "latest week": v,
             "behind": None if v is None else max(stale["current_week"] - v, 0)}
            for k, v in stale["tables"].items()
        ]))

    st.subheader("Every table the engine reads")
    status = ui.freshness()
    ui.table(status, config=ui.fixed(1, "age_days"), height=440)

    oldest = status["age_days"].max()
    missing = status.filter(pl.col("missing_proj_season"))
    cols = st.columns(3)
    cols[0].metric("Tables read", f"{status.height}")
    cols[1].metric("Oldest", f"{oldest:.0f} days" if oldest is not None else "—")
    cols[2].metric(f"Missing {PROJ_SEASON}", f"{missing.height}")
    if not missing.is_empty():
        st.warning(f"No {PROJ_SEASON} partition: {', '.join(missing['table'].to_list())}. Before the "
                   f"season this is normal for usage tables and a problem for rosters, depth charts "
                   f"and schedules.", icon="⚠️")
    ui.note(
        "`own refresh` tables are the light 2026 ones this repo refreshes itself; `shared lake` tables "
        "come from the parquet lake and are not ours to update. The stale check is deliberately about "
        "coverage rather than a fixed age in days — a processed table without the projection season is "
        "normal in August and alarming in November."
    )

    st.subheader("What each estimate was built from")
    ui.note("Per estimate: the prior, the observation behind it, the weight the observation earned, and "
            "the number that was used. This is the table that answers 'why is he projected there'.")
    prov = ui.provenance(view)
    if prov.is_empty():
        st.info("No provenance for this scenario.")
    else:
        ui.table(prov.head(400), height=380)
        ui.note(f"First 400 of {prov.height:,} rows — the Exports page will write all of them.")
