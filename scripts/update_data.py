"""Bring the data under the projection up to date. Safe to run any number of times.

    python scripts/update_data.py                 # light: the four tables this repo owns
    python scripts/update_data.py --full          # light, after rebuilding the played-game lake
    python scripts/update_data.py --dry-run       # print the commands and stop

Two speeds, because the data has two speeds:

- **Light** (seconds, daily). Rosters, depth charts, schedules, injuries and draft picks change every
  day in season and decide who is projected at all. This is `src.data.refresh`, which this repo owns.
- **Heavy** (minutes, weekly). player_games, team_games, player_usage and passer_games only change
  after games are played. They are built by the engine repo -- `~/nfl-projection-system`, or
  `$NFLSP_ENGINE_DIR` -- whose `bootstrap_data -> build_tables -> build_aggregates` chain is
  season-scoped and idempotent, so re-running it for the projection season is the whole update. This
  script does not reimplement any of it; if that repo is not on this machine the heavy pass is skipped
  with a line in the log and the light pass still runs.

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


def heavy_pass(handle, season: int, dry: bool) -> str:
    repo = engine_dir()
    if not (repo / HEAVY[0][0]).exists():
        say(handle, f"heavy: no engine repo at {repo} -- skipped, the played-game tables stay as they are")
        return "no engine repo"
    py = python_for(repo)
    for script, *flags in HEAVY:
        code = run(handle, [py, script, *flags, "--seasons", str(season)], repo, dry)
        if code != 0:
            say(handle, f"heavy: {script} failed -- stopping the chain, the later steps read its output")
            return f"failed at {Path(script).name} ({code})"
    return "ok"


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
    }, indent=2) + "\n", encoding="utf-8")
    return 2 if (light >= 2 or heavy.startswith("failed")) else (1 if light == 1 else 0)


if __name__ == "__main__":
    raise SystemExit(main())
