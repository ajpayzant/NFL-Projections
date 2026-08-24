"""Read the parquet lake, and say how old it is.

The lake has two shapes. Most datasets are partitioned by season
(`processed/team_games/season=2025/*.parquet`); a few that arrive whole are a single file
(`raw/schedules.parquet`). `read` handles both so no caller has to know which it is asking for.

Precedence matters for the in-progress season. The four light tables this project refreshes itself
-- rosters, depth charts, schedules, injuries -- are written under `OWN_RAW`, and a season present
there wins over the same season in the shared lake. That is what lets a Tuesday roster refresh here
take effect without waiting on the other repo's pipeline, while the heavy played-game tables keep
coming from the one place that builds them.

Every read is memoised on (dataset, layer, seasons). The frames are large and pure; a page that asks
for the same one twice should pay once.
"""

from __future__ import annotations

import glob
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path

import polars as pl

from src.config import (
    HISTORY_FROM,
    LAKE,
    LAST_COMPLETE_SEASON,
    OWN_RAW,
    PROCESSED,
    PROJ_SEASON,
    RAW,
)

# Datasets we refresh ourselves; OWN_RAW wins over the shared lake for these.
OWNED = ("rosters", "depth_charts", "schedules", "injuries", "draft_picks")

# What the engine actually reads, and where from. Anything not listed here is not a dependency.
PROCESSED_TABLES = ("team_games", "player_usage", "passer_games", "player_games")
RAW_TABLES = ("rosters", "rosters_weekly", "depth_charts", "schedules", "injuries", "draft_picks",
              "combine")


def _roots(layer: str, dataset: str) -> list[Path]:
    """Directories to look in, highest precedence first."""
    if layer == "processed":
        return [PROCESSED]
    roots = [RAW]
    if dataset in OWNED:
        roots.insert(0, OWN_RAW)
    return roots


def _files(dataset: str, layer: str, seasons: tuple[int, ...] | None) -> list[str]:
    """Resolve a dataset to a concrete file list, preferring our own copy of an owned table."""
    out: list[str] = []
    for root in _roots(layer, dataset):
        found: dict[int, list[str]] = {}
        for path in sorted(glob.glob(str(root / dataset / "season=*" / "*.parquet"))):
            season = int(Path(path).parent.name.split("=")[1])
            if seasons is None or season in seasons:
                found.setdefault(season, []).append(path)
        # a season we already have from a higher-precedence root is not re-read from a lower one
        have = {int(Path(p).parent.name.split("=")[1]) for p in out if "season=" in p}
        for season, paths in found.items():
            if season not in have:
                out.extend(paths)
        # A dataset that arrives whole is a single file beside the directory. The directory can
        # exist and hold nothing but a manifest -- `schedules` is exactly that -- so the flat file
        # is checked whenever the partitions produced nothing, not only when the directory is absent.
        flat = root / f"{dataset}.parquet"
        if not out and flat.is_file():
            out.append(str(flat))
    return out


@lru_cache(maxsize=64)
def read(dataset: str, layer: str = "processed", seasons: tuple[int, ...] | None = None) -> pl.DataFrame:
    """One dataset as a single frame. Season filtering is applied to flat files too.

    Raises rather than returning empty: a missing table is a broken install or an un-run refresh,
    and a caller that silently projects off nothing produces numbers that look real.
    """
    files = _files(dataset, layer, seasons)
    if not files:
        looked = ", ".join(str(r / dataset) for r in _roots(layer, dataset))
        raise FileNotFoundError(f"no parquet for {layer}/{dataset} (seasons={seasons}); looked in {looked}")
    frames = []
    for path in files:
        df = pl.read_parquet(path)
        if "season" not in df.columns:
            season = int(Path(path).parent.name.split("=")[1])
            df = df.with_columns(pl.lit(season, pl.Int32).alias("season"))
        frames.append(df)
    out = pl.concat(frames, how="diagonal_relaxed")
    out = out.with_columns(pl.col("season").cast(pl.Int32))
    if seasons is not None:
        out = out.filter(pl.col("season").is_in(list(seasons)))
    return out


def history_seasons(first: int = HISTORY_FROM, last: int = LAST_COMPLETE_SEASON) -> tuple[int, ...]:
    return tuple(range(first, last + 1))


def regular_season(df: pl.DataFrame) -> pl.DataFrame:
    """REG rows only. Playoff usage is real football but it is not what a season projection covers."""
    return df.filter(pl.col("season_type") == "REG") if "season_type" in df.columns else df


def clear_cache() -> None:
    read.cache_clear()


# --------------------------------------------------------------------------- #
# freshness
# --------------------------------------------------------------------------- #
def _newest(dataset: str, layer: str) -> tuple[datetime | None, int, list[int]]:
    files = _files(dataset, layer, None)
    if not files:
        return None, 0, []
    mtime = max(Path(f).stat().st_mtime for f in files)
    seasons = sorted({int(Path(f).parent.name.split("=")[1]) for f in files if "season=" in f})
    return datetime.fromtimestamp(mtime, UTC), len(files), seasons


def status() -> pl.DataFrame:
    """One row per table the engine reads: where it came from, how old, which seasons.

    The `stale` flag is deliberately conservative. A processed table without the projection season
    is normal in August and alarming in November, so the check is 'does it cover every season we
    claim to read' rather than a fixed age in days.
    """
    rows = []
    now = datetime.now(UTC)
    for layer, tables in (("processed", PROCESSED_TABLES), ("raw", RAW_TABLES)):
        for dataset in tables:
            newest, n_files, seasons = _newest(dataset, layer)
            owned = layer == "raw" and dataset in OWNED and any(
                Path(f).is_relative_to(OWN_RAW) for f in _files(dataset, layer, None)
            )
            wants_proj_season = layer == "raw" or dataset in ("player_usage", "team_games")
            has_proj = (not seasons) or (PROJ_SEASON in seasons)
            rows.append(
                {
                    "table": f"{layer}/{dataset}",
                    "source": "own refresh" if owned else "shared lake",
                    "files": n_files,
                    "seasons": f"{min(seasons)}-{max(seasons)}" if seasons else "flat file",
                    "updated": newest.replace(tzinfo=None) if newest else None,
                    "age_days": round((now - newest).total_seconds() / 86400, 1) if newest else None,
                    "missing_proj_season": bool(wants_proj_season and seasons and not has_proj),
                }
            )
    return pl.DataFrame(rows)


def current_week(season: int = PROJ_SEASON, today: date | None = None) -> int:
    """Which week of `season` we are in now: 0 before it starts, 18 once it is over.

    Taken from the schedule rather than from a fixed start date, because the calendar moves every year
    and a hard-coded September date silently drifts. The week is the highest one whose first game has
    already kicked off, so during week 5 this reads 5 and the tables that should carry week-5 usage can
    be checked against it. Returns 0 when the schedule is missing, which reads as "no season in
    progress" and is the honest answer to a question we cannot resolve.

    Regular season only: the playoffs are weeks 19-22 in the same table, and a projection that covers
    17 games is not four weeks behind in February.
    """
    try:
        sched = read("schedules", "raw", (season,))
    except Exception:                                   # noqa: BLE001 -- absence is an answer here
        return 0
    if sched.is_empty() or not {"week", "gameday"} <= set(sched.columns):
        return 0
    for col in ("game_type", "season_type"):
        if col in sched.columns:
            sched = sched.filter(pl.col(col) == "REG")
            break
    now = today or datetime.now(UTC).date()
    started = sched.with_columns(
        pl.col("gameday").cast(pl.String).str.to_date("%Y-%m-%d", strict=False).alias("_d")
    ).filter(pl.col("_d").is_not_null() & (pl.col("_d") <= now))
    return 0 if started.is_empty() else int(started["week"].max())


def staleness(season: int = PROJ_SEASON, today: date | None = None) -> dict:
    """Is `processed/` keeping up with the season? The one freshness question with a right answer.

    `status` reports every table's age, which is necessary and not sufficient: a table refreshed
    yesterday can still be missing last Sunday's games. This compares the weeks actually present in
    the processed usage tables against the week the calendar says we are in, and that difference is
    the number a user needs -- before week 1 it is vacuously fine, and in November a two-week gap means
    the projection is being made from stale usage.
    """
    week = current_week(season, today)
    out = {"season": season, "current_week": week, "in_season": week > 0,
           "weeks_behind": 0, "latest_week": None, "tables": {}, "stale": False}
    if week == 0:
        return out
    latest = []
    for dataset in ("team_games", "player_usage"):
        try:
            df = read(dataset, "processed", (season,))
        except Exception:                               # noqa: BLE001 -- a missing table is a finding
            out["tables"][dataset] = None
            continue
        got = None if df.is_empty() or "week" not in df.columns else int(df["week"].max())
        out["tables"][dataset] = got
        if got is not None:
            latest.append(got)
    if latest:
        out["latest_week"] = min(latest)
        out["weeks_behind"] = max(week - min(latest), 0)
    out["stale"] = bool(out["latest_week"] is None or out["weeks_behind"] >= 1)
    return out


def lake_present() -> bool:
    return LAKE.is_dir() and (PROCESSED / "team_games").is_dir()
