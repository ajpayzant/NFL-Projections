"""Blend a player's own history, then shrink it toward a prior.

Two rules, and they are the reason this is a module rather than three lines inline:

**Blend counts, never rates.** A share of a season is `Σ w·numerator / Σ w·denominator` over the
seasons behind it. Averaging the seasons' *ratios* instead would weight a 20-target rookie year the
same as a 140-target follow-up. Weighting counts also means playing time takes care of itself: a
four-game season contributes four games of evidence without any special case.

**Shrink by evidence, not by season count.** `(n·obs + k·prior) / (n + k)` where `n` is the blended
denominator -- opportunities, not games and not seasons. `k` is the number of opportunities at which
a player's own rate and the group prior carry equal weight, so it is directly interpretable: a
fitted `k` of 90 targets says half of a receiver's catch rate is still his position's until he has
seen 90 of them. It is fitted per metric in `priors.py`, never guessed.

The pair composes: blend to get `(obs, n)`, shrink `obs` toward the prior with weight `n/(n+k)`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import polars as pl

DEFAULT_WEIGHTS = (5.0, 3.0, 2.0)


def season_weights(
    season_col: str = "season",
    target: int | pl.Expr = 0,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
) -> pl.Expr:
    """Recency weight for a row, by how many seasons back it is from `target`.

    Seasons at or after the target get weight 0 -- that is the leakage guard, and it lives here so
    every caller inherits it rather than remembering it.
    """
    lag = (pl.lit(target) if isinstance(target, int) else target) - pl.col(season_col)
    expr = pl.when(lag < 1).then(0.0)
    for i, w in enumerate(weights, start=1):
        expr = expr.when(lag == i).then(float(w))
    return expr.otherwise(0.0).alias("recency_weight")


def blend_counts(
    df: pl.DataFrame,
    target_season: int,
    count_cols: Iterable[str],
    by: Sequence[str] = ("player_id",),
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    season_col: str = "season",
    scale: bool = True,
) -> pl.DataFrame:
    """Recency-weighted totals of `count_cols` from seasons strictly before `target_season`.

    Returns one row per `by` group with each count blended, plus `blend_games` and `seasons_used`
    for reporting how much history is actually behind a number.

    `scale` divides by the *full* weight vector's sum, which puts the blended counts on a
    one-season scale without erasing how much history is behind them: three seasons of 600 team
    targets blends to 600, one season to 300. Ratios are unaffected -- it only fixes the units of
    `n`, and those units are what a fitted `k` is quoted in.
    """
    cols = [c for c in count_cols if c in df.columns]
    w = season_weights(season_col, target_season, weights)
    hist = df.with_columns(w).filter(pl.col("recency_weight") > 0)
    if hist.is_empty():
        schema = {c: pl.Float64 for c in cols}
        return pl.DataFrame(
            schema={
                **{k: df.schema[k] for k in by},
                **schema,
                "blend_games": pl.Float64,
                "seasons_used": pl.UInt32,
                "last_season": pl.Int32,
            }
        )
    denom = float(sum(weights)) if scale else 1.0
    aggs = [
        ((pl.col(c).cast(pl.Float64).fill_null(0.0) * pl.col("recency_weight")).sum() / denom).alias(c)
        for c in cols
    ]
    if "games" in df.columns:
        aggs.append(
            ((pl.col("games").cast(pl.Float64) * pl.col("recency_weight")).sum() / denom)
            .alias("blend_games")
        )
    return hist.group_by(list(by)).agg(
        *aggs,
        pl.col(season_col).n_unique().alias("seasons_used"),
        pl.col(season_col).max().cast(pl.Int32).alias("last_season"),
    )


def ratio(num: str, den: str, alias: str) -> pl.Expr:
    return (
        pl.when(pl.col(den) > 0).then(pl.col(num) / pl.col(den)).otherwise(None).alias(alias)
    )


def shrink(
    obs: str | pl.Expr, prior: str | pl.Expr, n: str | pl.Expr, k: float | pl.Expr
) -> pl.Expr:
    """`(n·obs + k·prior) / (n + k)`, falling back to the prior where the player has no sample.

    `k` is in the units of `n`. A missing observation is not zero evidence pulled to zero; it is no
    evidence, so the answer is the prior.

    `k` may be an expression rather than a scalar, for the one fit that needs a different constant for
    different rows: availability is fitted separately for first-string jobs and for the bench, because
    those two populations are not the same shape. See `roster.fit_availability`.
    """
    obs_e = pl.col(obs) if isinstance(obs, str) else obs
    prior_e = pl.col(prior) if isinstance(prior, str) else prior
    n_e = pl.col(n) if isinstance(n, str) else n
    n_e = n_e.cast(pl.Float64).fill_null(0.0)
    return (
        pl.when(obs_e.is_null() | (n_e <= 0))
        .then(prior_e)
        .otherwise((n_e * obs_e + k * prior_e) / (n_e + k))
    )


def shrink_weight(n: str | pl.Expr, k: float | pl.Expr) -> pl.Expr:
    """The `n/(n+k)` fraction itself -- worth surfacing so a user can see how much of a number is
    the player and how much is his position.

    A metric can fit `k = 0` -- the QB share of his team's passing touchdowns does -- and then a player
    with no sample is 0/0. That is zero evidence, so it is zero weight, and `shrink` agrees: with no
    sample it returns the prior. Left as a NaN it propagated into every mean of this column.
    """
    n_e = (pl.col(n) if isinstance(n, str) else n).cast(pl.Float64).fill_null(0.0)
    return pl.when(n_e + k <= 0).then(pl.lit(0.0)).otherwise(n_e / (n_e + k))


def regress_to_mean(value: str | pl.Expr, mean: str | pl.Expr, keep: float) -> pl.Expr:
    """`mean + keep·(value - mean)`. The team-level counterpart of `shrink`: there is no opportunity
    count to weight by at team level -- 17 games is 17 games -- so the pull is a fixed fraction."""
    v = pl.col(value) if isinstance(value, str) else value
    m = pl.col(mean) if isinstance(mean, str) else mean
    return pl.when(v.is_null()).then(m).otherwise(m + keep * (v - m))
