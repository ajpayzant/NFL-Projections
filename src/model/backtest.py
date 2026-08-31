"""Project a season that has already been played, and score it against what happened.

Every other fit in this engine is scored on its own node: `priors` scores a target share against the
targets a player really got, `team` scores a per-game chain against the game. Those numbers say each
piece is better than the thing it replaced. None of them says the *board* is better, and the board is
what a user reads. This module composes a held-out season end to end -- roster, availability, shares,
efficiency, the schedule -- and scores season fantasy points against two baselines:

- **`ewma`** -- the recency-weighted average of the player's own recent season totals. No priors, no
  shares, no schedule; just "he scored 240, 190 and 265, so call it 235". It is the honest floor, and
  it is a real opponent: for an established starter it is hard to beat.
- **`workbook`** -- what the Excel workbook this replaces actually did: a three-season blend taken at
  face value, shares left unnormalised, no per-game context. Its three departures from the engine are
  the three things the rebuild claims to have fixed, so this is the number the rebuild is for.

The rest of the variants are **node ablations**. Each turns off exactly one thing and re-composes, so
the decision the plan asks for -- keep the node or revert it -- is made on composed fantasy points
rather than on the node's own objective. A node that helps its own metric and hurts the board loses.

**What is held out and what is not.** Held out per target season: history (`blend.season_weights`
zeroes the target and everything after), depth-slot group priors, rookie draft curves, slot capital
norms, the availability slot prior and presence rates, the measured pool targets, and the dropback
split. Depth charts come from the last snapshot before week 1 and the roster from week 1, so nobody is
listed where December put him. Weather is the venue's climate from prior seasons, not the weather the
game got. The market is switched off entirely (`market_weight = 0`), because the only lines on file
for a played season are closing lines and a closing line knows the season.

**Not** held out, and this makes the numbers below optimistic: the per-metric shrinkage constants
`k`, the rookie blend forms, the availability blend `k`, and the team layer's fitted `(w, keep)`,
per-game splits and market weight. These are a handful of scalars grid-searched over seven target
seasons; refitting them inside every fold would multiply the cost of this harness by the cost of the
entire fit. Their own held-out numbers are reported where they are fitted -- `priors --fit`,
`team --fit` -- and those are the leak-free answers for the scalars specifically. The `k=0` and
`k=inf` ablations here bound how much they can be worth.

**Calibration** is reported next to MAE, because they answer different questions: MAE says how close
the projections land, the calibration line says whether they are tilted. Regressing outcome on
projection separates a level error from a spread error, and it is also the table that stops a
non-defect being chased -- season points are right-skewed, so a projection correctly aimed at the mean
sits above the median outcome and shows a negative median error in every band. `calibration` says
which of the two numbers to judge.

The run ends with **interval coverage**: the same held-out seasons scored against the simulated range
instead of the point projection, because a range is a second promise and MAE does not check it. A
model can win every table above and still miss the floor it advertised one season in five.

    python -m src.model.backtest                              # 2021-2025, every variant
    python -m src.model.backtest --targets 2024 2025
    python -m src.model.backtest --variants full workbook ewma
    python -m src.model.backtest --no-intervals                # skip the Monte Carlo per season
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, replace

import polars as pl

from src.config import FITTED, LAST_COMPLETE_SEASON, PROJ_SEASON, Settings, ensure_dirs
from src.data import history, lake
from src.model import blend, compose, efficiency, opportunity, priors, roster, team

# The first season with participation history behind it. 2021 is projectable -- fantasy points do not
# depend on snap counts -- but its route and snap columns come back empty, so it is reported apart.
FIRST_TARGET = 2021

# Stats scored against actuals. Every one of them exists in `player_games` under the same name, and the
# list lives with the aggregate that produces them so the two cannot drift apart.
SCORED = history.SEASON_STATS

# Population cuts, and they are not interchangeable -- a node can win one and lose another, so the
# report shows all four rather than picking one.
#
#   all       every player the projection named. The honest denominator: points given to somebody who
#             never took the field are error, not a rounding difference. Four in ten of these players
#             score nothing at all, which makes it the cut most sensitive to level bias.
#   played    at least one game. Isolates the projection from the roster-survival question.
#   regulars  at least eight games -- the population a weekly lineup is actually drawn from.
#   starters  the players who *finished* as startable at their position. Selected on the outcome, so
#             it cannot be read as calibration; it is the cut that says whether the board got the
#             people who mattered roughly right.
VIEWS = ("all", "played", "regulars", "starters")

# What "startable" means, per position, for the `starters` view.
STARTERS = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}

# The three numbers a node is judged on together. MAE alone cannot decide anything here: this
# population is four-tenths zeroes and heavily right-skewed, so a change that shaves the level off
# every projection lowers MAE while making every player worth starting worse.
JUDGED = ("mae", "rmse", "spearman")

PLAYER_PATH = FITTED / "backtest_players.parquet"
SUMMARY_PATH = FITTED / "backtest_summary.parquet"


# --------------------------------------------------------------------------- #
# variants
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variant:
    """One projection to score. `on` transforms the ex-ante settings and artifacts of a target."""

    name: str
    note: str
    on: Callable[[Settings, priors.Fitted], tuple[Settings, priors.Fitted]]
    engine: bool = True          # False means it does not compose (the EWMA baseline)


def _all_k(fitted: priors.Fitted, value: float) -> priors.Fitted:
    """Force every metric's `k`. A metric missing from the dict would fall back to the Settings
    default rather than to `value`, so every name is written out."""
    return replace(fitted, k={m.name: value for m in priors.METRICS})


def _no_curve(fitted: priors.Fitted) -> priors.Fitted:
    """Rookies priced off their depth slot alone: `w = 0` is the slot prior in either blend form."""
    return replace(fitted, blends={m: (form, 0.0) for m, (form, _) in fitted.blends.items()})


VARIANTS = (
    Variant("full", "the engine as built, market off",
            lambda s, f: (s, f)),
    Variant("ewma", "recency-weighted mean of his own recent season totals",
            lambda s, f: (s, f), engine=False),
    Variant("workbook", "flat blend, shares unnormalised, no per-game context",
            lambda s, f: (replace(s, normalize_pools=False, use_context_factors=False),
                          replace(_all_k(f, 0.0), games_k=0.0))),
    Variant("no_shrinkage", "ablate shrinkage: own history at face value (k=0)",
            lambda s, f: (s, _all_k(f, 0.0))),
    Variant("prior_only", "ablate own history: the depth slot alone (k=inf)",
            lambda s, f: (s, _all_k(f, 1e12))),
    Variant("no_normalise", "ablate pool normalisation",
            lambda s, f: (replace(s, normalize_pools=False), f)),
    Variant("flat_efficiency", "ablate the per-game efficiency factors",
            lambda s, f: (replace(s, use_context_factors=False), f)),
    Variant("no_rookie_curve", "ablate rookie draft curves",
            lambda s, f: (s, _no_curve(f))),
    Variant("own_games", "ablate the availability blend: his own attendance alone",
            lambda s, f: (s, replace(f, games_k=0.0))),
    Variant("closing_lines", "NOT ex-ante: the market at the fitted weight",
            lambda s, f: (replace(s, market_weight=None), f)),
)

BY_NAME = {v.name: v for v in VARIANTS}


# --------------------------------------------------------------------------- #
# ex-ante inputs for one target season
# --------------------------------------------------------------------------- #
def base_settings(scoring: str | None = None) -> Settings:
    """The settings a preseason projection of a played season is entitled to.

    `market_weight = 0` is the whole of it: the schedule file's `spread_line` is the closing line, and
    blending a closing line into a preseason projection would score the market's in-season
    information as the engine's. `team.game_environment` also drops its market level correction at
    zero weight, so nothing reads the line by a side door.
    """
    s = Settings(market_weight=0.0)
    return s.with_scoring(scoring) if scoring else s


@dataclass(frozen=True, eq=False)
class Ex:
    """Everything about a target season that every variant shares, computed once.

    `population` is the one that matters for a fair comparison: every variant, baselines included, is
    scored over the same set of players -- the offensive week-1 roster the projection covers. Left to
    itself the EWMA baseline would predict for anybody who ever recorded a stat, which is three times
    as many players and almost all of them zeroes, and its MAE would look a third of the engine's
    purely because its denominator is full of easy nothings.
    """

    season: int
    hist: tuple[int, ...]
    fitted: priors.Fitted
    rows: pl.DataFrame
    pool_targets: dict[str, float]
    split_target: float
    actual: pl.DataFrame
    population: pl.DataFrame


def ex_ante(season: int, settings: Settings) -> Ex:
    """Refit every per-season artifact from seasons strictly before `season`."""
    hist = lake.history_seasons(last=season - 1)
    fitted = priors.fitted_as_of(season, settings)
    saved = roster.load_availability()
    tau = saved.get("tau")
    fitted = replace(
        fitted,
        slot_games=roster.slot_games_as_of(season, tau=tau),
        games_k=saved.get("k"),
        games_tau=tau,
    )
    when = "latest" if season >= PROJ_SEASON else "preseason"
    return Ex(
        season=season,
        hist=hist,
        fitted=fitted,
        rows=team.game_rows(season, expected_weather=season <= LAST_COMPLETE_SEASON),
        pool_targets=opportunity.measure_targets(hist),
        split_target=compose.measure_dropback_split(hist),
        actual=actual_season(season, settings),
        population=roster.roster(season, when).select("player_id", "player", "position"),
    )


def project_weekly(season: int, settings: Settings, ex: Ex) -> pl.DataFrame:
    """Compose `season` from `ex`, stopping at the weekly frame. Same call chain as the projection.

    The roster and the depth chart are taken from before week 1 for a played season and from the
    newest snapshot for 2026, which is the same rule `priors.slots` and `roster._avail_panel` use.

    Split out from `project` because the per-game rows are what the Monte Carlo shocks, so calibrating
    its intervals against a held-out season needs the weekly frame and not the totals.
    """
    when = "latest" if season >= PROJ_SEASON else "preseason"
    ros = roster.roster(season, when)
    f = ex.fitted
    part = roster.participation(season, settings, ros=ros, fitted=f)
    shares = opportunity.player_shares(season, settings, ros=ros, fitted=f)
    rates = efficiency.rates(season, settings, ros=ros, fitted=f)
    opp = opportunity.opportunity(
        season, settings, shares=shares, part=part, fitted=f, rows=ex.rows,
        pool_targets=ex.pool_targets,
    )
    return compose.weekly(season, settings, opp=opp, player_rates=rates,
                          split_target=ex.split_target)


def project(season: int, settings: Settings, ex: Ex) -> pl.DataFrame:
    """The season totals of `project_weekly`, which is what every variant is scored on."""
    return compose.seasonal(project_weekly(season, settings, ex), season, settings)


# --------------------------------------------------------------------------- #
# actuals and the EWMA baseline
# --------------------------------------------------------------------------- #
def actual_season(season: int, settings: Settings) -> pl.DataFrame:
    """What each player actually did in `season`, ready to join onto a projection.

    Every outcome column carries an `a_` prefix, the same convention `team.historical_rows` uses, so
    an actual cannot be mistaken for an input by a join or by a reader.
    """
    got = history.player_seasons((season,), settings.scoring)
    have = [c for c in SCORED if c in got.columns]
    return got.select(
        "player_id",
        pl.col("games").alias("a_games"),
        *[pl.col(c).alias(f"a_{c}") for c in have],
    )


def ewma(season: int, settings: Settings, population: pl.DataFrame) -> pl.DataFrame:
    """The recency-weighted average of a player's own recent season totals, over `population`.

    Counts are blended, so this is `blend.blend_counts` with no shrinkage and no prior -- the same
    recency vector the engine uses, applied to season totals instead of to shares. Because fantasy
    points are linear in the counts, blending them directly agrees with blending the counts and then
    scoring, which is why one call covers every stat in the table.

    A rookie gets zero. That is the baseline's real weakness rather than a handicap imposed on it, and
    it is the reason the `played` view is reported beside the full one -- so the comparison is not
    simply a report on how many rookies were on a roster.
    """
    hist = lake.history_seasons(last=season - 1)
    have = [c for c in SCORED]
    if not hist:
        return population.with_columns([pl.lit(0.0).alias(c) for c in (*have, "games")])
    past = history.player_seasons(hist, settings.scoring)
    have = [c for c in SCORED if c in past.columns]
    b = blend.blend_counts(past, season, [*have, "games"], by=("player_id",),
                           weights=settings.recency)
    return population.join(b.select("player_id", *have, "games"), on="player_id", how="left") \
        .with_columns([pl.col(c).fill_null(0.0) for c in (*have, "games")])


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_frame(predicted: pl.DataFrame, ex: Ex) -> pl.DataFrame:
    """One row per projected player: what was projected, what happened, and the view he falls in.

    A left join from the projection. The population is who the projection named, so points given to a
    player who never took the field are counted as error rather than dropped -- the same convention
    `priors` scores its shares under. `coverage` in the report says how much of the season's actual
    scoring belonged to players the projection never listed.
    """
    have = [c for c in SCORED if c in predicted.columns]
    keep = [c for c in ("player_id", "player", "position", "games") if c in predicted.columns]
    out = predicted.select(*keep, *have).join(ex.actual, on="player_id", how="left")
    return out.with_columns(
        pl.lit(ex.season).alias("season"),
        pl.col("a_games").fill_null(0.0),
        *[pl.col(f"a_{c}").fill_null(0.0) for c in have],
    ).with_columns(
        (pl.col("a_games") >= 1.0).alias("played"),
        (pl.col("a_games") >= 8.0).alias("regulars"),
        pl.lit(True).alias("all"),
        pl.col("a_fantasy_points").rank("min", descending=True).over("position")
        .alias("a_position_rank"),
    ).with_columns(
        (pl.col("a_position_rank")
         <= pl.col("position").replace_strict(STARTERS, default=0, return_dtype=pl.Int32))
        .alias("starters")
    )


def _metrics(df: pl.DataFrame, stat: str) -> dict:
    e = pl.col(stat) - pl.col(f"a_{stat}")
    agg = df.select(
        e.abs().mean().alias("mae"),
        (e**2).mean().sqrt().alias("rmse"),
        e.mean().alias("bias"),
        pl.corr(stat, f"a_{stat}", method="spearman").alias("spearman"),
        pl.len().alias("n"),
    )
    r = agg.row(0, named=True)
    return {k: (float(v) if v is not None and k != "n" else v) for k, v in r.items()}


def summarise(scored: pl.DataFrame, stats: tuple[str, ...] = SCORED) -> pl.DataFrame:
    """MAE, RMSE, bias and rank correlation per variant, view, season and stat.

    Rank correlation is in the table because it is what a ranking board is read for: a projection
    that is ten points low on everybody is useless as a level and perfect as an order, and MAE alone
    cannot tell those apart.
    """
    rows = []
    for view in VIEWS:
        sub = scored.filter(pl.col(view))
        for (variant, season), g in sub.group_by(["variant", "season"], maintain_order=True):
            for stat in stats:
                if stat not in g.columns:
                    continue
                rows.append({"variant": variant, "view": view, "season": int(season),
                             "stat": stat, **_metrics(g, stat)})
        # pooled over seasons, which is the number the decision is made on
        for (variant,), g in sub.group_by("variant", maintain_order=True):
            for stat in stats:
                if stat not in g.columns:
                    continue
                rows.append({"variant": variant, "view": view, "season": 0,
                             "stat": stat, **_metrics(g, stat)})
    return pl.DataFrame(rows, infer_schema_length=None)


def coverage(ex: Ex, predicted: pl.DataFrame) -> dict:
    """How much of the season's real fantasy scoring the projected population did not include."""
    total = float(ex.actual["a_fantasy_points"].sum())
    named = float(
        ex.actual.join(predicted.select("player_id"), on="player_id", how="semi")
        ["a_fantasy_points"].sum()
    )
    return {"season": ex.season, "actual_points": total,
            "covered_pct": 100.0 * named / total if total else 0.0,
            "projected_players": int(predicted.height),
            "actual_players": int(ex.actual.height)}


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def run(
    targets: tuple[int, ...] | None = None,
    names: tuple[str, ...] | None = None,
    scoring: str | None = None,
    verbose: bool = True,
) -> dict[str, pl.DataFrame]:
    """Compose and score every variant on every target season."""
    targets = targets or tuple(range(FIRST_TARGET, LAST_COMPLETE_SEASON + 1))
    chosen = [BY_NAME[n] for n in (names or tuple(v.name for v in VARIANTS))]
    base = base_settings(scoring)

    scored, cover = [], []
    for season in targets:
        ex = ex_ante(season, base)
        first = True
        for v in chosen:
            if v.engine:
                s, f = v.on(base, ex.fitted)
                pred = project(season, s, replace(ex, fitted=f))
            else:
                pred = ewma(season, base, ex.population)
            frame = score_frame(pred, ex).with_columns(pl.lit(v.name).alias("variant"))
            scored.append(frame)
            if first and v.engine:
                cover.append(coverage(ex, pred))
                first = False
            if verbose:
                m = _metrics(frame.filter(pl.col("played")), "fantasy_points")
                print(f"  {season} {v.name:<16} mae {m['mae']:7.2f}  rmse {m['rmse']:7.2f}"
                      f"  rho {m['spearman']:.3f}  n {m['n']}")
    # relaxed: a variant's frame carries only the stats its source has, and the EWMA baseline's
    # `games` is a blended count rather than a sum of weekly availability
    players = pl.concat(scored, how="diagonal_relaxed")
    return {"players": players, "summary": summarise(players),
            "coverage": pl.DataFrame(cover)}


def decisions(summary: pl.DataFrame, view: str = "played") -> pl.DataFrame:
    """Every ablation against the full engine, pooled: does the node earn its place?

    The plan's rule is to revert a node that loses. The judgement is on three numbers, not one, and
    the reason is in `JUDGED`: MAE on this population rewards under-projection, so a node that adds
    volume can lower MAE while raising RMSE and scrambling the order. `beats_full` counts how many of
    the three the ablation wins, and only a clean sweep is a revert.

    `closing_lines` and the two baselines are excluded -- none of them is an ablation of a node.
    """
    pooled = summary.filter(
        (pl.col("view") == view) & (pl.col("season") == 0) & (pl.col("stat") == "fantasy_points")
    )
    full = pooled.filter(pl.col("variant") == "full")
    if full.is_empty():
        return pl.DataFrame()
    ref = {m: float(full[m][0]) for m in JUDGED}
    notes = {v.name: v.note for v in VARIANTS}
    # lower is better for mae and rmse, higher for spearman
    wins = [
        (pl.col("mae") < ref["mae"]).cast(pl.Int32),
        (pl.col("rmse") < ref["rmse"]).cast(pl.Int32),
        (pl.col("spearman") > ref["spearman"]).cast(pl.Int32),
    ]
    return (
        pooled.filter(~pl.col("variant").is_in(["full", "ewma", "closing_lines"]))
        .select(
            "variant", "mae", "rmse", "spearman",
            (pl.col("mae") - ref["mae"]).alias("d_mae"),
            (pl.col("rmse") - ref["rmse"]).alias("d_rmse"),
            (pl.col("spearman") - ref["spearman"]).alias("d_rho"),
            pl.sum_horizontal(wins).alias("beats_full"),
        )
        .with_columns(
            pl.col("variant").replace_strict(notes, default="").alias("note"),
            pl.when(pl.col("beats_full") == 0).then(pl.lit("keep"))
            .when(pl.col("beats_full") == len(JUDGED)).then(pl.lit("REVERT"))
            .otherwise(pl.lit("mixed")).alias("verdict"),
        )
        .sort("d_mae", descending=True)
    )


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _report(art: dict[str, pl.DataFrame], view: str) -> None:
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_rows(60)
    pl.Config.set_fmt_float("mixed")
    summary, players = art["summary"], art["players"]

    seasons = sorted(players["season"].unique().to_list())
    print(f"\nBACKTEST  {seasons[0]}-{seasons[-1]}  season fantasy points, "
          f"{players['variant'].n_unique()} variants, view '{view}'")
    print("\nCOVERAGE  share of each season's real scoring done by players the projection listed")
    print(art["coverage"].with_columns(
        pl.col("actual_points").round(0), pl.col("covered_pct").round(1)
    ))

    fp = summary.filter((pl.col("stat") == "fantasy_points") & (pl.col("view") == view))
    print(f"\nPOOLED  every target season together, players with "
          f"{'>=1 game' if view == 'played' else view}")
    print(fp.filter(pl.col("season") == 0).select(
        "variant", pl.col("n"), pl.col("mae").round(2), pl.col("rmse").round(2),
        pl.col("bias").round(2), pl.col("spearman").round(3),
    ).sort("mae"))

    print("\nPER SEASON  MAE of season fantasy points")
    per = fp.filter(pl.col("season") > 0).select("variant", "season", pl.col("mae").round(2))
    print(per.pivot(on="season", index="variant", values="mae").sort("variant"))

    print("\nBY VIEW  the full engine and its two baselines under each population cut")
    print(summary.filter(
        (pl.col("stat") == "fantasy_points") & (pl.col("season") == 0)
        & pl.col("variant").is_in(["full", "workbook", "ewma"])
    ).select("view", "variant", pl.col("n"), pl.col("mae").round(2), pl.col("rmse").round(2),
             pl.col("bias").round(2), pl.col("spearman").round(3))
     .sort(["view", "mae"]))

    print("\nPER STAT  pooled MAE, full engine against the baselines")
    ps = summary.filter(
        (pl.col("view") == view) & (pl.col("season") == 0)
        & pl.col("variant").is_in(["full", "workbook", "ewma"])
    ).select("stat", "variant", pl.col("mae").round(3))
    wide = ps.pivot(on="variant", index="stat", values="mae")
    if {"full", "workbook"} <= set(wide.columns):
        wide = wide.with_columns(
            (100.0 * (pl.col("workbook") - pl.col("full")) / pl.col("workbook"))
            .round(1).alias("gain_vs_workbook_pct")
        )
    print(wide)

    for cut in dict.fromkeys([view, "starters"]):
        d = decisions(summary, cut)
        if d.is_empty():
            continue
        print(f"\nNODE DECISIONS  '{cut}': each ablation against the full engine. beats_full counts "
              f"how many of {JUDGED} it wins; only a sweep is a revert")
        print(d.select("variant", pl.col("mae").round(2), pl.col("d_mae").round(3),
                       pl.col("d_rmse").round(3), pl.col("d_rho").round(4), "beats_full",
                       "verdict", "note"))

    print("\nBY POSITION  pooled MAE of season fantasy points, full engine vs baselines")
    sub = players.filter(pl.col(view) & pl.col("position").is_not_null()
                         & pl.col("variant").is_in(["full", "workbook", "ewma"]))
    rows = []
    for (variant, pos), g in sub.group_by(["variant", "position"], maintain_order=True):
        rows.append({"variant": variant, "position": pos, **_metrics(g, "fantasy_points")})
    bypos = pl.DataFrame(rows, infer_schema_length=None)
    print(bypos.filter(pl.col("position").is_in(list(STARTERS)))
          .select("position", "variant", pl.col("n"), pl.col("mae").round(2),
                  pl.col("spearman").round(3))
          .sort(["position", "mae"]))


# The projected-games bands the calibration table is cut on. Cut on the projection, so a band is a
# statement about what the model said rather than about what happened.
GAME_BANDS = (3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0)

GAME_BAND_LABELS = ("0-3", "3-5", "5-7", "7-9", "9-11", "11-13", "13-15", "15-17")


def calibration(players: pl.DataFrame, variant: str = "full") -> dict[str, pl.DataFrame]:
    """Is the projection level right, conditional on what it said? Slope, intercept, and by band.

    MAE says how close the projections land; this says whether they are *tilted*. Regressing the
    outcome on the projection is the standard test and it separates the two errors MAE cannot:

    - **slope** below 1 means the projections are spread too wide -- the model commits harder than the
      evidence supports and the extremes come back toward the middle. Above 1 means the opposite, that
      it hedges and the outcomes are more spread than it said. Either way a slope of 1 is the claim.
    - **intercept** is the flat level error left when the slope is accounted for.

    Two traps this table is arranged to avoid. The first is that `bias` in `summarise` and `median_err`
    here answer different questions, and only the first is a defect: season fantasy points are heavily
    right-skewed, so a projection that is *correctly* aimed at the mean sits above the median outcome
    and shows a negative median error in every band. Reading that as over-projection and shrinking it
    away would break the mean to flatter a statistic that was never supposed to be zero. So both are
    printed, `mean_err` is the one to judge, and `median_err` is here to be recognised rather than
    fixed.

    The second is the population. The bands are cut on projected games and never on realised games,
    for the reason `simulate.coverage` sets out at length: conditioning on appearing keeps only the
    players who beat the availability the model applied, which shifts every error in the table.
    `never_played` is reported instead, as the share of each band that finished with no games at all,
    because that is the same fact stated without selecting on it.
    """
    df = players.filter(pl.col("variant") == variant) if "variant" in players.columns else players
    err = pl.col("a_fantasy_points") - pl.col("fantasy_points")

    def fit(sub: pl.DataFrame) -> tuple[float, float]:
        """Least squares of actual on projected, which is the calibration line."""
        import numpy as np

        x = sub["fantasy_points"].to_numpy().astype(float)
        y = sub["a_fantasy_points"].to_numpy().astype(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10 or np.ptp(x[m]) < 1e-9:
            return float("nan"), float("nan")
        slope, intercept = np.polyfit(x[m], y[m], 1)
        return float(slope), float(intercept)

    rows = []
    for pos in (None, *sorted(df["position"].unique())):
        sub = df if pos is None else df.filter(pl.col("position") == pos)
        for cut, name in ((0.0, "board"), (20.0, "proj>20")):
            s = sub.filter(pl.col("fantasy_points") > cut) if cut else sub
            if s.height < 10:
                continue
            slope, intercept = fit(s)
            rows.append({
                "position": pos or "ALL", "population": name, "n": s.height,
                "slope": slope, "intercept": intercept,
                "mean_err": float(s.select(err.mean()).item()),
                "median_err": float(s.select(err.median()).item()),
            })

    banded = df.with_columns(
        pl.col("games").cut(list(GAME_BANDS), labels=list(GAME_BAND_LABELS)).alias("band")
    ).group_by("band").agg(
        pl.len().alias("n"),
        pl.col("fantasy_points").mean().alias("proj_mean"),
        pl.col("a_fantasy_points").mean().alias("act_mean"),
        err.mean().alias("mean_err"),
        err.median().alias("median_err"),
        (pl.col("a_games") == 0).mean().alias("never_played"),
    ).sort("band")
    return {"lines": pl.DataFrame(rows), "bands": banded}


def _calibration_report(players: pl.DataFrame) -> None:
    cal = calibration(players)
    print("\nCALIBRATION  actual = intercept + slope x projected. A slope of 1 is the claim; below 1 "
          "means the\n  projections are spread wider than the outcomes, above 1 that the model hedged.")
    print(cal["lines"].select(
        "position", "population", "n", pl.col("slope").round(3), pl.col("intercept").round(2),
        pl.col("mean_err").round(2), pl.col("median_err").round(2),
    ).sort("population", "position"))
    print("\n  by projected-games band. `mean_err` is the one to judge -- `median_err` is negative "
          "everywhere by\n  construction, because season points are right-skewed and the projection "
          "is aimed at the mean.")
    print(cal["bands"].select(
        "band", "n", pl.col("proj_mean").round(1), pl.col("act_mean").round(1),
        pl.col("mean_err").round(1), pl.col("median_err").round(1),
        pl.col("never_played").round(3),
    ))


def intervals(targets: tuple[int, ...], draws: int = 1500, scoring: str | None = None) -> pl.DataFrame:
    """Interval coverage: did the simulated ranges contain the seasons the model had not seen?

    A point projection is scored by MAE and a range is scored by this, and the two are not substitutes.
    A model can win on MAE and still promise a floor it misses one season in five, and a user reads the
    floor as a promise. `cvm` is the distance of the whole PIT from uniform, which is the number the
    calibration is fitted on; the two coverages are what a reader actually wants to see.
    """
    from src.model import simulate

    base = base_settings(scoring)
    disp = simulate.load()
    rows = []
    for season in targets:
        wk, actual = simulate._ex_ante_weekly(season, base)
        sim = simulate.run(wk, base, disp, draws=draws)
        for cut, name in ((0.0, "board"), (simulate.STARTABLE, "startable")):
            rows.append(simulate.coverage(sim, actual, min_projected=cut)
                        .with_columns(pl.lit(season).alias("season"), pl.lit(name).alias("population")))
    return pl.concat(rows)


def _interval_report(targets: tuple[int, ...], draws: int, scoring: str | None) -> None:
    from src.model import simulate

    disp = simulate.load()
    if not disp.meta.get("fitted"):
        print("\nINTERVALS  skipped: no fitted dispersion. Run `python -m src.model.simulate --fit`.")
        return
    got = intervals(targets, draws, scoring)
    state = "calibrated" if disp.meta.get("calibrated") else "NOT calibrated"
    print(f"\nINTERVALS  simulated coverage, {draws} draws, dispersion {state} (scale {disp.scale})")
    seen = sorted(set(targets) & set(disp.meta.get("calibration_targets") or ()))
    if seen:
        print(f"  the projection is ex ante, but the scale was fitted on {seen} -- in sample for the "
              f"width, out of sample for the mean")
    print(f"  the population is cut on the projection and not the outcome, `board` is everyone and "
          f"`startable` is projected over {simulate.STARTABLE:.0f}")
    print(got.filter(pl.col("position") == "ALL").select(
        "population", "season", "n", pl.col("cover_90").round(3), pl.col("want_90"),
        pl.col("cover_50").round(3), pl.col("want_50"),
        pl.col("below").round(3), pl.col("above").round(3), pl.col("pit_iqr").round(3),
        pl.col("pit_mean").round(3), pl.col("cvm").round(4), pl.col("median_bias").round(1),
    ).sort("population", "season", descending=[True, False]))
    print("\n  startable only, pooled by position")
    print(got.filter((pl.col("position") != "ALL") & (pl.col("population") == "startable"))
          .group_by("position").agg(
              pl.col("n").sum(), pl.col("cover_90").mean().round(3),
              pl.col("cover_50").mean().round(3), pl.col("below").mean().round(3),
              pl.col("above").mean().round(3), pl.col("cvm").mean().round(4),
          ).sort("position"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--targets", type=int, nargs="+", default=None,
                   help=f"target seasons (default {FIRST_TARGET}-{LAST_COMPLETE_SEASON})")
    p.add_argument("--variants", nargs="+", default=None, choices=[v.name for v in VARIANTS])
    p.add_argument("--scoring", default=None, help="scoring preset (default full PPR)")
    p.add_argument("--view", default="played", choices=list(VIEWS))
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-intervals", action="store_true",
                   help="skip the simulated interval coverage, which costs a Monte Carlo per season")
    p.add_argument("--interval-draws", type=int, default=1500)
    p.add_argument("--interval-seasons", type=int, nargs="+", default=None,
                   help="seasons to score intervals on (default the last two target seasons)")
    a = p.parse_args(argv)

    ensure_dirs()
    targets = tuple(a.targets) if a.targets else tuple(range(FIRST_TARGET, LAST_COMPLETE_SEASON + 1))
    art = run(targets, tuple(a.variants) if a.variants else None,
              a.scoring, verbose=not a.quiet)
    art["players"].write_parquet(PLAYER_PATH)
    art["summary"].write_parquet(SUMMARY_PATH)
    _report(art, a.view)
    _calibration_report(art["players"])
    if not a.no_intervals:
        # the last two by default: the dispersion is measured from this very backtest, so scoring its
        # intervals on every target season would take longer than the point projection it belongs to
        seasons = tuple(a.interval_seasons) if a.interval_seasons else targets[-2:]
        _interval_report(seasons, a.interval_draws, a.scoring)
    print(f"\nwrote {PLAYER_PATH.name} and {SUMMARY_PATH.name} to {FITTED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
