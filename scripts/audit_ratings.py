"""Every rating the projection uses: where it comes from, whether it is found, whether it is spent.

    python scripts/audit_ratings.py                  # the table, and the findings under it
    python scripts/audit_ratings.py --csv out.csv
    python scripts/audit_ratings.py --strict          # exit 1 if anything is wrong

There are thirty-seven ratings and three ways one of them can be wrong, which is why this is a script
rather than a paragraph in a README. A rating can be:

- **not found.** Its numerator or denominator is not a column of the table it names, so `own_rate`
  silently returns nothing and every player gets the depth-slot prior. The projection still builds and
  still looks reasonable, which is what makes this the expensive failure: a metric that quietly went to
  the prior for everybody costs accuracy without costing an error message. Upstream renames columns
  without warning -- the 2026 roster lost `draft_number` mid-season -- so this is checked rather than
  assumed.
- **found and empty.** The column is there and is null, or its denominator is zero, for most of the
  league. A share measured on 4% of players is a prior with extra steps.
- **classified wrong.** A rating that moves a projection and is not offered as a knob, or one that is
  offered and moves nothing. `overrides` derives that split from the pools rather than declaring it, so
  the audit re-derives it independently and the two have to agree.

The `spent` column is what that last check is made of, and `unused` in it is a classification rather
than a fault. Eleven ratings are estimated, fitted and then read by no stat on purpose -- the red-zone
and late-down volume, `route_participation`, `adot`, `tprr`, `rush_success_rate`, `pass_td_rate`,
`yards_per_clean_rush`. Each was measured against the projection, moved nothing, and was moved out of
the override list for it, because offering a knob that does nothing is worse than not offering it. They
are still worth computing: they are the evidence a reader argues from, and a receiver's air yards per
target is the reason one would move his `yards_per_target`. What must not happen is the two lists
drifting apart, in either direction.

The `live` column is a fifth thing and not a fault: whether the weekly tables can measure both halves of
the ratio, which is what decides whether this rating learns from the season in progress or keeps its
preseason estimate all year. Eleven of nineteen skill ratings and eight of eighteen quarterback ratings
can; the rest need the play-by-play rebuild. That is a real limit and it is meant to be visible.

`tests/test_ratings_audit.py` asserts the parts that must never regress.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROJ_SEASON, Settings  # noqa: E402
from src.model import compose, efficiency, estimate, opportunity, overrides, priors  # noqa: E402

# What "populated" has to mean before a rating is worth fitting. Not a rounded guess: below a fifth of
# the league the (position, slot) prior is carrying nearly every player anyway, so the metric is a prior
# wearing a player's name. `rz_carry_share` sits at 0.30 and is genuinely thin; anything under this is
# a column that is not really there.
MIN_COVERAGE = 0.20

# Where each rating is spent. Derived from the same constants the engine reads rather than listed, so a
# rating that stops being used stops reading as used here without anybody remembering to edit this file.
_POOL_OF = {s: p.name for p in opportunity.POOLS for s in p.shares}


def spent_how(name: str) -> str:
    """What the stat line actually does with this rating. The column that separates knob from evidence.

    Four answers. `rate` is multiplied into a count to produce a stat. `pool` is a share of something a
    stat is divided out of. `playing time` is the exception the whole engine has exactly one of -- the
    snap shares divide a pool nothing is divided by, so by shape they are evidence, and they are spent
    anyway as a multiplier on every other claim the man makes. `unused` is a rating that is estimated and
    then read by no stat: evidence, which is a real answer and not a fault, and the one thing that must
    then also be true of it is that nobody is offered it as a knob.
    """
    if name in overrides.PLAYING_TIME_METRICS:
        return "playing time"
    if name in compose.STAT_RATES:
        return "rate"
    pool = _POOL_OF.get(name)
    if pool in compose.STAT_POOLS:
        return "pool"
    return "unused"


def audit(season: int = PROJ_SEASON) -> pl.DataFrame:
    """One row per rating: its source, its coverage, its fit, and what is done with it."""
    settings = Settings()
    fitted = priors.fitted_saved()
    rows = []
    for m in priors.METRICS:
        hist = priors._hist(m.table)
        have_num, have_den = m.num in hist.columns, m.den in hist.columns
        d = hist.filter(pl.col("season") >= m.since)
        n_rows = d.height

        # coverage is "has a denominator to divide by", not "is not null": a receiver with a null target
        # count and one with zero targets are the same absence of evidence
        if have_den and n_rows:
            with_den = d.filter(pl.col(m.den).cast(pl.Float64).fill_null(0.0) > 0).height
            coverage = with_den / n_rows
        else:
            with_den, coverage = 0, 0.0

        mean = None
        if have_num and have_den and with_den:
            got = d.filter(pl.col(m.den).cast(pl.Float64).fill_null(0.0) > 0).select(
                (pl.col(m.num).cast(pl.Float64).sum() / pl.col(m.den).cast(pl.Float64).sum())
            )
            mean = float(got.to_series()[0])

        # what the fit made of it, and how much of a typical player's number is his own rather than his
        # job's -- `k` is in the metric's own denominator on a one-season scale, so this is readable
        k = float(fitted.k.get(m.name, settings.default_share_k if m.kind == "share"
                               else settings.default_rate_k))
        typical_n = None
        if have_den and with_den:
            typical_n = float(d.filter(pl.col(m.den).cast(pl.Float64) > 0)[m.den]
                              .cast(pl.Float64).median() or 0.0)
        own_weight = None if typical_n is None else typical_n / (typical_n + k)

        rows.append({
            "metric": m.name,
            "table": m.table,
            "kind": m.kind,
            "num": m.num,
            "den": m.den,
            "found": have_num and have_den,
            "since": m.since,
            "rows": n_rows,
            "coverage": round(coverage, 3),
            "league_mean": None if mean is None else round(mean, 4),
            "k": k,
            "own_weight": None if own_weight is None else round(own_weight, 3),
            "spent": spent_how(m.name),
            "editable": m.name in overrides.PLAYER_FIELDS,
            "live": estimate.to_date(m, season, settings) is not None,
        })
    return pl.DataFrame(rows, infer_schema_length=None)


def findings(a: pl.DataFrame) -> list[str]:
    """The rows that are wrong, in the order they would cost accuracy. Empty is the answer we want."""
    out = []
    missing = a.filter(~pl.col("found"))
    for r in missing.iter_rows(named=True):
        out.append(f"NOT FOUND  {r['metric']}: {r['table']} has no {r['num']!r}/{r['den']!r} -- every "
                   f"player is getting the depth-slot prior for this and nothing says so")
    thin = a.filter(pl.col("found") & (pl.col("coverage") < MIN_COVERAGE))
    for r in thin.iter_rows(named=True):
        out.append(f"THIN       {r['metric']}: only {r['coverage']:.0%} of rows have a denominator, so "
                   f"this is close to a prior with extra steps")
    # `spent == "unused"` is a classification and not a fault. Eleven ratings are measured, fitted and
    # read by no stat on purpose: red-zone and late-down volume, route participation, `adot`, `tprr`,
    # `rush_success_rate`, `pass_td_rate`, `yards_per_clean_rush`. They are the evidence a reader argues
    # from -- a receiver's air yards per target is why one would move his `yards_per_target` -- and the
    # measurement that moved them out of the override list is the reason they are shown rather than
    # offered. What must never regress is that the two lists agree, which is the pair of checks below.
    # A metric that becomes unused *and stays editable* is the fault; being unused is not.
    for r in a.filter((pl.col("spent") == "unused") & pl.col("editable")).iter_rows(named=True):
        out.append(f"MISLABEL   {r['metric']}: editable but nothing reads it -- a knob that moves "
                   f"nothing, which teaches a reader that the knobs move nothing")
    for r in a.filter((pl.col("spent") != "unused") & ~pl.col("editable")).iter_rows(named=True):
        out.append(f"HIDDEN     {r['metric']}: a stat is built from it but a user cannot say so")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, default=PROJ_SEASON)
    p.add_argument("--csv", type=Path, help="write the table here as well as printing it")
    p.add_argument("--strict", action="store_true", help="exit 1 if there is anything to report")
    args = p.parse_args(argv)

    a = audit(args.season)
    show = a.select("metric", "table", "kind", "found", "coverage", "league_mean", "k", "own_weight",
                    "spent", "editable", "live").sort(["table", "kind", "metric"])
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(show)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        a.write_csv(args.csv)
        print(f"\nwrote {args.csv}")

    print(f"\n{a.height} ratings · {a['found'].sum()} found · {a['editable'].sum()} knobs · "
          f"{a.filter(pl.col('spent') == 'unused').height} evidence · "
          f"{a['live'].sum()} can learn from the season in progress")
    # Which ratings cannot, and why, because this is the honest limit rather than a to-do: routes run,
    # red-zone volume and the scramble/designed split are play-by-play facts that arrive with the weekly
    # rebuild, and a rating denominated in one of them keeps its preseason estimate all season.
    waiting = a.filter(~pl.col("live") & pl.col("found"))["metric"].to_list()
    if waiting:
        print(f"waiting on the weekly rebuild: {', '.join(waiting)}")

    found = findings(a)
    if not found:
        print("\nno findings: every rating resolves to a real column, is populated on enough of the "
              "league to be a measurement, and is offered as a knob exactly when a stat is built from it")
        return 0
    print(f"\n{len(found)} findings:")
    for line in found:
        print(f"  {line}")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
