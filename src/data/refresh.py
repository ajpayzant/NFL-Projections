"""Refresh the light 2026 tables this project owns.

Four tables decide who is on the field and who they play: rosters, depth charts, schedules and
injuries. They are small, they change daily in season, and a projection built on a stale one is
wrong in a way that no amount of modelling fixes -- so this project fetches them itself rather than
waiting on the pbp pipeline that owns the heavy played-game tables.

Three more decide what has already happened: `player_stats`, `snap_counts` and `team_stats`, the weekly
results. They are here for the same reason and a sharper one. The heavy `processed/` tables carry five
years of history and are rebuilt on the other repo's schedule, which is right for history and hopeless
for last Sunday -- so through August 2026 the app had no way to know the season had started, and would
have projected week 1 from priors in December. These three are published within hours of a game.
`src/data/inseason.py` reads them; `compose.actualise` spends them.

Writes to `OWN_RAW/<dataset>/season=<year>/<dataset>.parquet`, which `lake.read` prefers over the
shared copy. Nothing here touches `processed/`.

    python -m src.data.refresh                    # the projection season
    python -m src.data.refresh --seasons 2025 2026
    python -m src.data.refresh --check            # what is on disk, fetch nothing
    python -m src.data.refresh --datasets player_stats snap_counts team_stats
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

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
    # results to date, which is what makes an in-season projection an in-season projection
    "player_stats": ("load_player_stats", True),
    "snap_counts": ("load_snap_counts", True),
    "team_stats": ("load_team_stats", True),
}

# Tables that are empty until the season starts, and whose emptiness in August is not a failure. Kept
# apart from the essential three so a missing week-1 stat line in July does not report as a broken
# refresh, and so that `main` can still say the season has not begun rather than nothing.
RESULTS = ("player_stats", "snap_counts", "team_stats")


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


def dropped_columns(target: Path, df: pl.DataFrame) -> list[str]:
    """Columns the file at `target` has that the replacement does not.

    Upstream reshapes without warning: the 2026 season roster arrived for months with `draft_number`
    and then stopped, and the refresh wrote the narrower file over the wider one without a word.
    Everything downstream is written to survive that now, but a column vanishing is still a change in
    what can be projected, and it belongs in the log rather than in a traceback a week later.
    """
    if not target.is_file():
        return []
    try:
        before = set(pl.read_parquet_schema(target))
    except Exception:                    # noqa: BLE001 - an unreadable old file is what we are replacing
        return []
    return sorted(before - set(df.columns))


def write(dataset: str, season: int, df: pl.DataFrame) -> tuple[int, list[str]]:
    """Replace the snapshot, atomically, and say which columns it lost on the way.

    Atomically because the app reads this directory while the scheduled task writes it: a torn parquet
    is not a stale projection, it is a crash on the next page load. `os.replace` makes the swap a single
    step, so a reader sees either the old file or the new one.
    """
    out = OWN_RAW / dataset / f"season={season}"
    out.mkdir(parents=True, exist_ok=True)
    final = out / f"{dataset}.parquet"
    lost = dropped_columns(final, df)
    # The temp file must not end in .parquet: `lake._files` globs the partition directory, and a
    # half-written sibling would be concatenated into the frame.
    tmp = out / f".{dataset}.parquet.tmp"
    df.write_parquet(tmp)
    os.replace(tmp, final)
    return df.height, lost


def missed(outcome: int | str) -> bool:
    """Did this (dataset, season) end up with nothing written?

    An outcome is a row count when the write succeeded and a sentence when it did not -- except for a
    write that succeeded with fewer columns than before, which is also a sentence and is emphatically
    not a miss. The snapshot is on disk and current; one column of it is gone.
    """
    return isinstance(outcome, str) and not outcome[:1].isdigit()


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
            rows, lost = write(dataset, season, df)
            results[(dataset, season)] = (
                f"{rows} rows  UPSTREAM DROPPED: {', '.join(lost)}" if lost else rows
            )
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
        # A results table with nothing in it is a season that has not kicked off, not a broken fetch.
        # Reported, because "no games yet" is worth reading, and not counted, because in July it is true.
        if missed(outcome) and not (dataset in RESULTS and outcome == "empty upstream"):
            failed += 1
    # Rosters and depth charts are the two that change who gets projected at all. An empty
    # schedule in August is a broken fetch, not a quiet season -- so any miss on these is an error.
    essential = {(d, s) for (d, s) in results if d in ("rosters", "depth_charts", "schedules")}
    if any(missed(results[k]) for k in essential):
        print("essential table missing -- projections would run on the previous snapshot", file=sys.stderr)
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
