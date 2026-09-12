"""Bring the data under the projection up to date. Safe to run any number of times.

    python scripts/update_data.py                 # light: the eight tables this repo owns
    python scripts/update_data.py --full          # light, after rebuilding the played-game lake
    python scripts/update_data.py --dry-run       # print the commands and stop

Two speeds, because the data has two speeds:

- **Light** (seconds, daily). Rosters, depth charts, schedules, injuries and draft picks change every
  day in season and decide who is projected at all. So do the three weekly results tables --
  player_stats, snap_counts and team_stats -- which are published within hours of a game and are what
  make an in-season projection an in-season projection: a played week becomes what happened in it, and
  the weeks still to come learn from it. This is `src.data.refresh`, which this repo owns.
- **Heavy** (minutes, weekly). player_games, team_games, player_usage and passer_games only change
  after games are played. They carry the play-by-play detail the light tables cannot -- routes run,
  red-zone volume, the scramble/designed split -- so the metrics denominated in those keep waiting on
  this pass, and the light one is not a substitute for it. They are built by the engine repo --
  `~/nfl-projection-system`, or
  `$NFLSP_ENGINE_DIR` -- whose `bootstrap_data -> build_tables -> build_aggregates` chain is
  season-scoped and idempotent, so re-running it for the projection season is the whole update. This
  script does not reimplement any of it; if that repo is not on this machine the heavy pass is skipped
  with a line in the log and the light pass still runs.

  Mid-season the heavy pass is deliberately partial: three of those tables are derived from a
  participation file nflverse only publishes once a season is over, so asking for them fails the chain
  and leaves the three that *are* buildable unbuilt. `IN_SEASON_TABLES` below is the honest list, and
  the run reports `ok (partial)` rather than the `failed` it used to report every week of the year.

Everything is appended to `build/review/update.log` and the finish is stamped into
`build/review/last_update.json`, which the app's sidebar reads: `st.cache_data` lives in the server
process, so a server that was already running when this ran keeps serving the old frames until it is
restarted, and the app should be the one to say so rather than the user finding out from a number.

Exit codes: 0 up to date, 1 something non-essential was missed (the preseason always lands here --
nflreadpy refuses 2026 injuries until the season starts), 2 nothing usable ran.

`scripts/weekly_update.bat` is the wrapper the Windows scheduled tasks call.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import PROJ_SEASON  # noqa: E402

LOG = ROOT / "build" / "review" / "update.log"
STAMP = ROOT / "build" / "review" / "last_update.json"

# Refetch on the heavy pass: without --force, bootstrap_data leaves a dataset it already has on disk
# alone, which is right for a first install and wrong for a Tuesday -- last week's pbp is on disk and
# is exactly the file that needs replacing.
HEAVY = (
    ("scripts/bootstrap_data.py", "--essential-only", "--force"),
    ("scripts/build_tables.py",),
    ("scripts/build_aggregates.py",),
)

# The derived tables that need no participation file, which is all of them except `routes`, `backfield`
# and the `player_usage` built from those two.
#
# Participation -- which eleven men were on the field for a play -- comes to nflverse from FTN and is
# published only after a season has finished: `nflreadpy.load_participation` caps its own season argument
# at `current_season - 1` and says so in a comment. So for the season in progress those three tables
# cannot be built at all, and asking for them does not merely skip them: `build_all` raises on the missing
# file and the whole chain stops, which is how a weekly refresh comes to report "failed at
# build_aggregates" every week of the year and how the three tables that *can* be built never get built.
#
# Asking only for what is possible is the difference between a pass that reports a real failure and one
# that cries wolf. What it costs is the eight skill ratings denominated in route, red-zone and late-down
# volume -- they keep their preseason estimate until the offseason rebuild, and `src/data/inseason.py`
# says so beside the ten quarterback ratings these three tables do close.
IN_SEASON_TABLES = ("team_games", "player_games", "passer_games")


def engine_dir() -> Path:
    return Path(os.environ.get("NFLSP_ENGINE_DIR", Path.home() / "nfl-projection-system"))


def python_for(repo: Path) -> str:
    """That repo's own interpreter if it has one, else the one running this."""
    own = repo / ".venv" / "Scripts" / "python.exe"
    if own.exists():
        return str(own)
    own = repo / ".venv" / "bin" / "python"
    return str(own) if own.exists() else sys.executable


def say(handle, text: str) -> None:
    """Every line goes to both the log and the console: the log is for the scheduled runs, the
    console for the person who ran it by hand and is waiting."""
    print(text)
    handle.write(text + "\n")
    handle.flush()


def run(handle, cmd: list[str], cwd: Path, dry: bool) -> int:
    say(handle, f"$ {' '.join(cmd)}   [{cwd}]")
    if dry:
        return 0
    env = {**os.environ, "PYTHONUTF8": "1"}
    # Child output is interleaved into the same log, so a failure is diagnosable from one file.
    done = subprocess.run(cmd, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    handle.flush()
    say(handle, f"  -> exit {done.returncode}")
    return done.returncode


def has_participation(repo: Path, season: int) -> bool:
    """Is there a participation file for this season yet?

    Asked of the disk rather than assumed from the calendar, so that the week FTN publishes last season
    the full chain resumes on its own -- and so a machine that has the file for a season already gets
    the routes tables rebuilt even if that season is nominally the projection season.
    """
    return any((repo / "data" / "raw" / "participation").glob(f"season={season}/*.parquet"))


def heavy_pass(handle, season: int, dry: bool) -> str:
    repo = engine_dir()
    if not (repo / HEAVY[0][0]).exists():
        say(handle, f"heavy: no engine repo at {repo} -- skipped, the played-game tables stay as they are")
        return "no engine repo"
    py = python_for(repo)
    limited = not has_participation(repo, season)      # asked on a dry run too, so it prints the truth
    if limited:
        say(handle, f"heavy: no participation file for {season} yet, so routes, backfield and "
                    f"player_usage cannot be built -- asking build_aggregates for "
                    f"{', '.join(IN_SEASON_TABLES)} only. The eight route/red-zone/late-down skill "
                    f"ratings keep their preseason estimate; everything else still learns from the "
                    f"weeks played.")
    for script, *flags in HEAVY:
        if limited and script.endswith("build_aggregates.py"):
            flags = [*flags, "--tables", *IN_SEASON_TABLES]
        code = run(handle, [py, script, *flags, "--seasons", str(season)], repo, dry)
        if code != 0:
            say(handle, f"heavy: {script} failed -- stopping the chain, the later steps read its output")
            return f"failed at {Path(script).name} ({code})"
    return "ok (partial)" if limited else "ok"


def light_pass(handle, season: int, dry: bool) -> int:
    py = sys.executable
    code = run(handle, [py, "-m", "src.data.refresh", "--seasons", str(season)], ROOT, dry)
    if code == 1:
        say(handle, "light: a non-essential table was missed. Until the season starts nflreadpy "
                    "refuses 2026 injuries outright, so this is the normal preseason result.")
    elif code >= 2:
        say(handle, "light: an essential table (rosters, depth charts, schedules) was missed -- the "
                    "projection would run on the previous snapshot.")
    run(handle, [py, "-m", "src.data.refresh", "--check"], ROOT, dry)
    return code


def inseason_state(handle, season: int, dry: bool) -> dict:
    """How far into the season the results now reach, which is the point of having refreshed them.

    Logged rather than merely written, because it is the one line of the run that says whether the
    projection changed: a refresh that reports eight successful fetches and still knows about no games
    is a refresh that did nothing for the numbers on the board.
    """
    if dry:
        return {}
    from src.data import inseason, lake

    lake.clear_cache()
    inseason.clear_cache()
    state = {"weeks_complete": inseason.weeks_complete(season),
             "weeks_played": inseason.weeks_played(season),
             "result_rows": int(inseason.player_weeks(season).height)}
    if state["weeks_played"] == 0:
        say(handle, "in season: no completed games -- every week on the board is a projection")
    else:
        say(handle, f"in season: complete through week {state['weeks_complete']}, "
                    f"some teams through {state['weeks_played']}, "
                    f"{state['result_rows']} player-game results readable")
    return state


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full", action="store_true",
                   help="rebuild the played-game lake first (minutes, once a week is enough)")
    p.add_argument("--season", type=int, default=PROJ_SEASON)
    p.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    args = p.parse_args(argv)

    LOG.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now().astimezone()
    with LOG.open("a", encoding="utf-8") as handle:
        say(handle, "")
        say(handle, f"===== {started:%Y-%m-%d %H:%M %Z}  {'full' if args.full else 'light'}  "
                    f"season {args.season} =====")
        heavy = heavy_pass(handle, args.season, args.dry_run) if args.full else "skipped"
        light = light_pass(handle, args.season, args.dry_run)
        state = inseason_state(handle, args.season, args.dry_run)
        finished = datetime.now().astimezone()
        say(handle, f"finished {finished:%H:%M:%S} after {(finished - started).seconds}s  "
                    f"heavy={heavy}  light=exit {light}")

    if args.dry_run:
        return 0
    STAMP.write_text(json.dumps({
        "finished": finished.isoformat(timespec="seconds"),
        "mode": "full" if args.full else "light",
        "season": args.season,
        "heavy": heavy,
        "light": light,
        "seconds": (finished - started).seconds,
        **state,
    }, indent=2) + "\n", encoding="utf-8")
    return 2 if (light >= 2 or heavy.startswith("failed")) else (1 if light == 1 else 0)


if __name__ == "__main__":
    raise SystemExit(main())
