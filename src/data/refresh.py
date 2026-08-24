"""Refresh the light 2026 tables this project owns.

Four tables decide who is on the field and who they play: rosters, depth charts, schedules and
injuries. They are small, they change daily in season, and a projection built on a stale one is
wrong in a way that no amount of modelling fixes -- so this project fetches them itself rather than
waiting on the pbp pipeline that owns the heavy played-game tables.

Writes to `OWN_RAW/<dataset>/season=<year>/<dataset>.parquet`, which `lake.read` prefers over the
shared copy. Nothing here touches `processed/`.

    python -m src.data.refresh                    # the projection season
    python -m src.data.refresh --seasons 2025 2026
    python -m src.data.refresh --check            # what is on disk, fetch nothing
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

import polars as pl

from src.config import OWN_RAW, PROJ_SEASON, ensure_dirs
from src.data import lake

# dataset -> (loader name, is the loader season-filterable)
SOURCES: dict[str, tuple[str, bool]] = {
    "rosters": ("load_rosters", True),
    "depth_charts": ("load_depth_charts", True),
    "schedules": ("load_schedules", True),
    "injuries": ("load_injuries", True),
    "draft_picks": ("load_draft_picks", True),
}


def _loader(name: str):
    import nflreadpy as nfl

    fn = getattr(nfl, SOURCES[name][0], None)
    if fn is None:  # pragma: no cover - upstream rename
        raise AttributeError(f"nflreadpy has no {SOURCES[name][0]}")
    return fn


def fetch(dataset: str, season: int) -> pl.DataFrame:
    df = _loader(dataset)(season)
    if "season" not in df.columns:
        df = df.with_columns(pl.lit(season, pl.Int32).alias("season"))
    return df.filter(pl.col("season") == season)


def write(dataset: str, season: int, df: pl.DataFrame) -> int:
    out = OWN_RAW / dataset / f"season={season}"
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / f"{dataset}.parquet")
    return df.height


def refresh(datasets: list[str], seasons: list[int]) -> dict[tuple[str, int], int | str]:
    """Fetch and write each (dataset, season). A failure is recorded, not raised.

    One table failing upstream should not cost the others: a depth chart that has not been
    published yet is a normal Tuesday, and the caller decides whether the gap matters.
    """
    ensure_dirs()
    results: dict[tuple[str, int], int | str] = {}
    for dataset in datasets:
        for season in seasons:
            try:
                df = fetch(dataset, season)
            except Exception as exc:  # noqa: BLE001 - upstream can fail any number of ways
                results[(dataset, season)] = f"FAILED: {type(exc).__name__}: {exc}"
                continue
            if df.is_empty():
                results[(dataset, season)] = "empty upstream"
                continue
            results[(dataset, season)] = write(dataset, season, df)
    lake.clear_cache()
    return results


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=sorted(SOURCES), choices=sorted(SOURCES))
    p.add_argument("--seasons", nargs="+", type=int, default=[PROJ_SEASON])
    p.add_argument("--check", action="store_true", help="report what is on disk and exit")
    args = p.parse_args(argv)

    if args.check:
        print(lake.status().to_pandas().to_string(index=False))
        return 0

    print(f"refresh {datetime.now(UTC):%Y-%m-%d %H:%M} UTC -> {OWN_RAW}")
    results = refresh(args.datasets, args.seasons)
    failed = 0
    for (dataset, season), outcome in sorted(results.items()):
        print(f"  {dataset:<14} {season}  {outcome}")
        if isinstance(outcome, str):
            failed += 1
    # Rosters and depth charts are the two that change who gets projected at all. An empty
    # schedule in August is a broken fetch, not a quiet season -- so any miss on these is an error.
    essential = {(d, s) for (d, s) in results if d in ("rosters", "depth_charts", "schedules")}
    if any(isinstance(results[k], str) for k in essential):
        print("essential table missing -- projections would run on the previous snapshot", file=sys.stderr)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
