"""One player, one metric, one number -- and every input that produced it.

Every share and every rate in the projection is estimated the same way, and this module is that way,
written once. The workbook had a different formula in a different sheet for each quantity, which is
how a 40-carry sample of yards per carry ended up taken at face value in one place and shrunk in
another.

    used = (n x own + k x prior) / (n + k)

`n` is the player's own opportunity count -- targets, carries, routes -- recency-weighted across
seasons, never a rate. Blending counts rather than rates is what makes a four-game season contribute
four games of evidence automatically. `k` is fitted per metric in `priors.py` and is in the same
units, so `k = 60` for target share reads directly as "sixty targets of his own outweigh his job's
average". `prior` is what his job is worth, from the (position, depth-slot) table, with rookies
priced off draft capital by whichever of the fitted forms won out of sample.

**Everything is returned, not just the answer.** `obs`, `n`, `prior`, `used` and `own_weight` all
come back, because a user about to override a number needs to know whether it rests on nine hundred
routes or on nine. That is the same discipline as the workbook's BASE / Adj / USED triplet, which is
the one thing about it that worked.

`season_history` is the other half of the same idea: the record season by season rather than the one
number the blend reads. A career average is what an estimator should use and a trend is what a person
should argue with, so both are available and neither is derived from the other in the app.
"""

from __future__ import annotations

import polars as pl

from src.config import PROJ_SEASON, Settings
from src.model import priors
from src.model.blend import blend_counts, ratio, shrink, shrink_weight

# Which roster rows a metric applies to. A quarterback has no target share and a receiver has no
# sack rate; scoring either would be measuring a zero that was never in question.
TABLE_POSITIONS = {"qb": ("QB",), "skill": ("RB", "WR", "TE")}


def to_date(metric: priors.Metric, target: int, settings: Settings) -> pl.DataFrame | None:
    """This season so far as one more season of the player's own history, or None if it cannot be.

    Both halves of that sentence are load-bearing. A season projection made in November should know
    what happened in September -- not only for the weeks already played, which `compose.actualise`
    substitutes outright, but for the weeks still to come: a receiver running a 28% target share
    through five games is telling us something about his remaining twelve that no amount of last
    year's history knows. Returning it shaped like a history row is what lets `blend_counts` weigh it
    without a special case, and weigh it *by how much of it there is*, so it grows from a rumour in
    week 1 to the dominant evidence by December on its own.

    None in four cases, and each is a refusal rather than a gap:

    - not the season in progress. The backtest projects a finished season and scores it against that
      season's own rows, so letting lag 0 in there would report a model that already knew the answer.
      This is the same guard `blend.season_weights` keeps and it is checked twice on purpose.
    - the user asked for the model on its own (`use_inseason_form` off), which is the only honest way
      to read what the preseason estimate was worth.
    - no games in the books yet.
    - the live tables cannot measure *both* sides of this metric's ratio. That is the important one.
      Nobody publishes routes run or red-zone targets within hours of a game, and half a ratio is
      worse than none of it: a quarterback's sacks with no dropbacks to divide them by would add
      sacks to the numerator, nothing to the denominator, and report his sack rate as doubled. A
      metric the weekly tables cannot close keeps its preseason estimate, which is a real limit of
      this and is written down in `src/data/inseason.py` rather than hidden here.
    """
    if target != PROJ_SEASON or not settings.use_inseason_form:
        return None
    from src.data import inseason           # local: only the projection season pays for the import

    cur = inseason.to_date(metric.table, target)
    if cur.is_empty() or not {metric.num, metric.den} <= set(cur.columns):
        return None
    cur = cur.filter(pl.col(metric.den).cast(pl.Float64).fill_null(0.0) > 0)
    return None if cur.is_empty() else cur


def own_rate(
    metric: priors.Metric, target: int, settings: Settings, hist: pl.DataFrame | None = None
) -> pl.DataFrame:
    """A player's own recency-weighted rate from before `target`, and the evidence behind it.

    Counts are blended and then divided, so `n` is a real opportunity count and `obs` is the ratio
    that count actually produced. The leakage guard is `blend_counts`, which zeroes the weight on the
    target season and every season after it -- except for the season actually in progress, where the
    games already played are evidence about the games left rather than the answer to them. `to_date`
    decides that, and returning None from it is what makes this function's old behaviour the default.
    """
    hist = priors._hist(metric.table) if hist is None else hist
    keep = ["player_id", "season", "games", metric.num, metric.den]
    past = hist.filter((pl.col("season") >= metric.since) & (pl.col("season") < target))
    past = past.select([c for c in keep if c in past.columns])

    cur = to_date(metric, target, settings)
    current = 0.0
    if cur is not None:
        # weighted like last season, because that is what it is on its way to being. The count blend
        # scales it down to the games actually in it, so this is a ceiling and not a thumb on the scale.
        current = float(settings.recency[0]) if settings.recency else 1.0
        past = pl.concat([past, cur.select([c for c in keep if c in cur.columns])],
                         how="diagonal_relaxed")
    if past.is_empty():
        return pl.DataFrame(schema={"player_id": pl.String, "obs": pl.Float64, "n": pl.Float64,
                                    "seasons_used": pl.UInt32, "last_season": pl.Int32})
    b = blend_counts(past, target, [metric.num, metric.den], by=("player_id",),
                     weights=settings.recency, current=current)
    return b.select(
        "player_id", "seasons_used", "last_season",
        ratio(metric.num, metric.den, "obs"),
        pl.col(metric.den).cast(pl.Float64).alias("n"),
    )


def _prior_frame(group: pl.DataFrame, name: str) -> pl.DataFrame:
    empty = {"position": pl.String, "slot_bucket": pl.Int32, "prior_slot": pl.Float64,
             "position_prior": pl.Float64}
    if group.is_empty():
        return pl.DataFrame(schema=empty)
    return group.filter(pl.col("metric") == name).select(
        "position", "slot_bucket", pl.col("prior").alias("prior_slot"), "position_prior"
    )


def _curve_frame(curves: pl.DataFrame, name: str, w: float) -> pl.DataFrame:
    empty = {"position": pl.String, "pick_bin": pl.Int32, "curve": pl.Float64}
    if curves.is_empty() or w <= 0.0:
        return pl.DataFrame(schema=empty)
    return curves.filter(pl.col("metric") == name).select(
        "position", "pick_bin", pl.col("value").alias("curve")
    )


def _norm_frame(norms: pl.DataFrame, name: str) -> pl.DataFrame:
    empty = {"position": pl.String, "slot_bucket": pl.Int32, "norm": pl.Float64}
    if norms.is_empty():
        return pl.DataFrame(schema=empty)
    return norms.filter(pl.col("metric") == name).select("position", "slot_bucket", "norm")


def estimate(
    names: tuple[str, ...],
    ros: pl.DataFrame,
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
    fitted: priors.Fitted | None = None,
) -> pl.DataFrame:
    """One row per rostered player per metric, long form, with every input exposed.

    `ros` is the population, from `roster.roster` -- passed in rather than fetched so this module has
    no opinion about who is projectable and no import cycle with the module that does.

    `fitted` defaults to the artifacts on disk, which is what a projection wants. The backtest passes
    a set rebuilt from seasons before its target instead; see `priors.Fitted`.
    """
    settings = settings or Settings()
    fitted = fitted or priors.fitted_saved()
    ks = fitted.k
    group, curves, norms = fitted.priors, fitted.curves, fitted.norms
    blends = fitted.blends

    frames = []
    for name in names:
        metric = priors.BY_NAME[name]
        base = ros.filter(pl.col("position").is_in(TABLE_POSITIONS[metric.table]))
        if base.is_empty():
            continue
        form, w = blends.get(name, ("additive", 0.0))
        k = float(ks.get(name, settings.default_share_k if metric.kind == "share"
                         else settings.default_rate_k))
        d = (
            base.join(_prior_frame(group, name), on=["position", "slot_bucket"], how="left")
            .join(_curve_frame(curves, name, w), on=["position", "pick_bin"], how="left")
            .join(_norm_frame(norms, name), on=["position", "slot_bucket"], how="left")
            .join(own_rate(metric, season, settings), on="player_id", how="left")
        )
        # A rookie has no history, so his prior is the whole projection; everyone else is judged on
        # the job alone, because his own play has already priced in whatever the draft said about him.
        slot = pl.coalesce("prior_slot", "position_prior")
        rookie = priors.rookie_prior(slot, form, w, bounded=metric.kind == "share")
        is_rookie_priced = pl.col("is_rookie") & pl.col("curve").is_not_null() & (w > 0.0)
        frames.append(
            d.with_columns(pl.when(is_rookie_priced).then(rookie).otherwise(slot).alias("prior"))
            .with_columns(
                pl.lit(name).alias("metric"),
                pl.lit(metric.kind).alias("kind"),
                pl.lit(metric.den).alias("units"),
                pl.lit(k).alias("k"),
                shrink("obs", "prior", "n", k).alias("used"),
                shrink_weight("n", k).alias("own_weight"),
                pl.when(is_rookie_priced).then(pl.lit("draft_blend"))
                .when(pl.col("obs").is_null() | (pl.col("n").fill_null(0.0) <= 0))
                .then(pl.lit("slot_prior"))
                .otherwise(pl.lit("blend"))
                .alias("source"),
            )
            .select(
                "season", "team", "player_id", "player", "position", "depth_slot", "slot_bucket",
                "is_rookie", "draft_pick", "metric", "kind", "units", "k", "obs", "n",
                "seasons_used", "prior_slot", "curve", "norm", "prior", "used", "own_weight",
                "source",
            )
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort(["team", "position", "depth_slot", "metric"])


_HISTORY_SCHEMA = {"season": pl.Int32, "player_id": pl.String, "player": pl.String,
                   "team": pl.String, "games": pl.Int64, "metric": pl.String, "kind": pl.String,
                   "units": pl.String, "num": pl.Float64, "n": pl.Float64, "value": pl.Float64}


def season_history(
    names: tuple[str, ...] | list[str],
    player_ids: tuple[str, ...] | list[str] | None = None,
    seasons: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    """What a player's own metric actually measured, season by season, unweighted and unshrunk.

    `estimate` returns `obs`: one recency-weighted number over a career, which is the right input to a
    blend and the wrong thing to argue with. A rookie year at 9% followed by four seasons at 26% and
    five flat seasons at 22% can produce the same `obs`, and only one of those is a player whose role
    has changed. So the same ratio is returned here per season, with its denominator beside it, and
    nothing is weighted, blended or clipped: this is the record, not an estimate of it.

    Both are needed for the same decision, which is why they are separate functions rather than one
    frame -- an override is argued from the record and applied against the estimate.

    The season in progress is a row here like any other, unfinished. It has to be: it is now part of
    what the estimate reads, and a record that stopped last December would leave a user arguing with a
    number he cannot see the reason for. `games` says how much of it there is.
    """
    frames = []
    for name in names:
        metric = priors.BY_NAME.get(name)
        if metric is None:
            continue
        hist = priors._hist(metric.table)
        if metric.num not in hist.columns or metric.den not in hist.columns:
            continue
        live = to_date(metric, PROJ_SEASON, Settings())
        if live is not None:
            hist = pl.concat([hist, live.with_columns(pl.lit(None, pl.String).alias("player"))],
                             how="diagonal_relaxed")
        d = hist.filter(pl.col("season") >= metric.since)
        if seasons:
            d = d.filter(pl.col("season").is_in(list(seasons)))
        if player_ids is not None:
            d = d.filter(pl.col("player_id").is_in(list(player_ids)))
        if d.is_empty():
            continue
        frames.append(d.select(
            pl.col("season").cast(pl.Int32),
            "player_id",
            pl.col("player").cast(pl.String) if "player" in d.columns
            else pl.lit(None, pl.String).alias("player"),
            pl.col("team").cast(pl.String),
            pl.col("games").cast(pl.Int64),
            pl.lit(name).alias("metric"),
            pl.lit(metric.kind).alias("kind"),
            pl.lit(metric.den).alias("units"),
            pl.col(metric.num).cast(pl.Float64).alias("num"),
            pl.col(metric.den).cast(pl.Float64).alias("n"),
            ratio(metric.num, metric.den, "value"),
        ))
    if not frames:
        return pl.DataFrame(schema=_HISTORY_SCHEMA)
    return pl.concat(frames, how="diagonal_relaxed").sort(["metric", "player_id", "season"])


def wide(detail: pl.DataFrame, names: tuple[str, ...] | None = None) -> pl.DataFrame:
    """`estimate` output pivoted to one row per player, for the arithmetic downstream."""
    if detail.is_empty():
        return pl.DataFrame(schema={"player_id": pl.String})
    out = detail.pivot(on="metric", index="player_id", values="used")
    if names:
        keep = ["player_id", *[n for n in names if n in out.columns]]
        return out.select(keep)
    return out
