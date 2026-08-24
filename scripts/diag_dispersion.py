"""Where the simulated variance comes from, against how much of it the outcomes demand.

Calibration once said the intervals were short and wanted every persistent sigma doubled, which is the
kind of answer that is worth checking before it is believed: a scalar that lands on the edge of its own
grid is usually a sign that the thing being scaled is not the thing that is missing. It was not -- the
objective was absorbing a median bias into the width, and `simulate._pit_iqr` says so now -- but the
question this script was built to ask is still the right one. It switches each source of dispersion on
by itself and reports what it contributes, next to the spread the held-out season actually had.

Read the `target` column with its bias in mind, and do not use this in place of `--calibrate`. The
target is measured on players who played and finished with points, because a log ratio needs both; the
simulated spread beside it includes every draw, injuries and lost jobs and all. So `short_by` is
optimistic about how wide the sim is, and the verdict on the intervals is `simulate.coverage`, which
compares each outcome with its own distribution, drops nothing, and cuts its population on the
projection instead of the outcome for exactly the reason this column cannot. What this script is for is
the *shape* of the answer: which family of sigmas a given range is actually made of, and therefore
which one is worth measuring better.

The answer, on 2025: `share_season` and `games_ratio` are the two that matter, in that order, and the
rest are close to interchangeable. Everything else is the Poisson count, which is already in `none`.

    python scripts/diag_dispersion.py [season]
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import backtest, simulate  # noqa: E402

DRAWS = 600


def zeroed(d: simulate.Dispersion, keep: str) -> simulate.Dispersion:
    """The dispersion with every family off but `keep` (`"all"` keeps everything)."""
    off = {
        "team_week": {k: 0.0 for k in d.team_week},
        "team_season": {k: 0.0 for k in d.team_season},
        "usage_cv2": {k: 0.0 for k in d.usage_cv2},
        "event_cv2": {k: 0.0 for k in d.event_cv2},
        "share_season": {k: 0.0 for k in d.share_season},
        "rate_season": {k: 0.0 for k in d.rate_season},
        "games_ratio": {k: [1.0] for k in d.games_ratio},
    }
    if keep == "all":
        return d
    if keep == "none":
        return replace(d, **off)
    return replace(d, **{k: v for k, v in off.items() if k != keep})


def log_cv(pts: np.ndarray) -> np.ndarray:
    """Per-player robust sd of log points, from the quartiles rather than the moments.

    The moments cannot be used here. A draw where the player never took the field is a season total of
    zero, and in logs that is minus infinity however small a slice of the distribution it is -- so a
    plain `log(pts).std()` reports the injury lump and nothing else. The quartile spread is the same
    number for a lognormal and ignores a tail of zeros, which is what makes it comparable with the
    target, itself measured on players who played.
    """
    q25, q75 = np.quantile(np.maximum(pts, 1e-3), (0.25, 0.75), axis=1)
    return (np.log(q75) - np.log(q25)) / 1.349


def main(argv: list[str]) -> int:
    season = int(argv[0]) if argv else 2023
    settings = backtest.base_settings()
    disp = simulate.load()
    print(f"dispersion fitted={disp.meta.get('fitted')} calibrated={disp.meta.get('calibrated')} "
          f"scale={disp.scale}")

    wk, actual = simulate._ex_ante_weekly(season, settings)

    # the target: how far a held-out season really landed from its projection, in logs
    base = simulate.run(wk, settings, zeroed(disp, "none"), draws=8)
    tgt = base.season.join(actual, on="player_id", how="inner").filter(
        pl.col("played") & (pl.col("projected") > 20.0) & (pl.col("a_fantasy_points") > 0.0)
    ).with_columns(
        (pl.col("a_fantasy_points") / pl.col("projected")).log().alias("lr")
    )
    iqr = ((pl.col("lr").quantile(0.75) - pl.col("lr").quantile(0.25)) / 1.349)
    print(f"\nTARGET  log(actual/projected), played, projected > 20, season {season}")
    print(tgt.group_by("position", "tier").agg(
        pl.len().alias("n"), pl.col("lr").std().round(3).alias("sd"),
        iqr.round(3).alias("robust_sd"),
        pl.col("lr").mean().round(3).alias("mean"),
        pl.col("lr").kurtosis().round(2).alias("kurt"),
    ).sort("position", "tier"))

    # and each source of simulated dispersion on its own
    keys = ["none", "team_week", "team_season", "usage_cv2", "event_cv2", "share_season",
            "rate_season", "games_ratio", "all"]
    rows = []
    for keep in keys:
        sim = simulate.run(wk, settings, zeroed(disp, keep), draws=DRAWS)
        sd = log_cv(sim.points_draws)
        f = sim.season.with_columns(pl.Series("lsd", sd)).filter(pl.col("projected") > 20.0)
        for (pos, tier), sub in f.group_by(["position", "tier"], maintain_order=True):
            rows.append({"source": keep, "position": pos, "tier": tier,
                         "n": sub.height, "lsd": float(sub["lsd"].mean())})
        print(f"  ran {keep}")

    got = pl.DataFrame(rows).pivot(on="source", index=["position", "tier"], values="lsd") \
        .sort("position", "tier")
    want = tgt.group_by("position", "tier").agg(iqr.alias("target"))
    out = got.join(want, on=["position", "tier"], how="left")
    print("\nSOURCES  mean per-player sd of log season points, one family at a time")
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_rows(20)
    print(out.select(
        "position", "tier",
        *[pl.col(k).round(3) for k in keys],
        pl.col("target").round(3),
        (pl.col("target") / pl.col("all")).round(2).alias("short_by"),
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
