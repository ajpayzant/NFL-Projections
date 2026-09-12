"""The scheduled data update: the parts that have to be right when nobody is watching.

This runs twice a day from a Windows scheduled task, on a machine where the only evidence anybody
will ever look at is `build/review/update.log`. So what is worth pinning is not the fetching -- that
is nflreadpy's and the engine repo's job, and both are tested where they live -- but the wiring around
it: the three engine steps in the order their outputs depend on each other, a missing engine repo
costing the heavy pass and nothing else, and the exit code meaning what the task's history will say it
means.

Every test here is a dry run. A real one downloads a season of play-by-play.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    """`scripts/update_data.py` is a script, not a package member, so it is loaded by path."""
    spec = importlib.util.spec_from_file_location("update_data", ROOT / "scripts" / "update_data.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["update_data"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def update(tmp_path, monkeypatch):
    """The module with its log and stamp pointed somewhere disposable."""
    mod = _module()
    monkeypatch.setattr(mod, "LOG", tmp_path / "update.log")
    monkeypatch.setattr(mod, "STAMP", tmp_path / "last_update.json")
    return mod


def _log(update) -> str:
    return update.LOG.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# the heavy pass
# --------------------------------------------------------------------------- #
def test_the_three_engine_steps_run_in_the_order_their_outputs_depend_on(update):
    """bootstrap ingests, build_tables reads what was ingested, build_aggregates reads the tables.

    Any other order produces aggregates over last week's tables and reports success doing it.
    """
    assert [Path(script).name for script, *_ in update.HEAVY] == [
        "bootstrap_data.py", "build_tables.py", "build_aggregates.py"]


def test_the_heavy_pass_refetches_rather_than_trusting_what_is_on_disk(update):
    """Without --force, bootstrap_data leaves a dataset it already has alone -- which is right for a
    first install and wrong for a Tuesday, when last week's pbp is on disk and is the file to replace."""
    bootstrap = update.HEAVY[0]
    assert "--force" in bootstrap


def test_a_missing_engine_repo_skips_the_heavy_pass_and_says_so(update, tmp_path, monkeypatch):
    monkeypatch.setenv("NFLSP_ENGINE_DIR", str(tmp_path / "not-here"))
    assert update.main(["--full", "--dry-run"]) == 0
    log = _log(update)
    assert "no engine repo" in log
    assert "src.data.refresh" in log, "the light pass must still run"


def _fake_engine(tmp_path, monkeypatch, participation_seasons=()):
    """An engine repo with the three scripts and whichever participation files it has been given."""
    repo = tmp_path / "engine"
    (repo / "scripts").mkdir(parents=True)
    for script, *_ in _module().HEAVY:
        (repo / script).write_text("")
    for season in participation_seasons:
        part = repo / "data" / "raw" / "participation" / f"season={season}"
        part.mkdir(parents=True)
        (part / "part.parquet").write_text("")
    monkeypatch.setenv("NFLSP_ENGINE_DIR", str(repo))
    return repo


def test_a_season_with_no_participation_file_asks_only_for_the_tables_that_do_not_need_one(
        update, tmp_path, monkeypatch):
    """The bug this exists to prevent ran every week of last season and reported itself honestly.

    `routes`, `backfield` and the `player_usage` built from them come from a participation dataset
    published only after a season ends -- so mid-season `build_all` raised on the missing file, the chain
    stopped, and the three tables that *were* buildable never got built. Asking for what is possible turns
    a weekly `failed` into a weekly `ok (partial)`, and is the difference between the quarterback ratings
    learning from the season and holding their August estimate until February.
    """
    _fake_engine(tmp_path, monkeypatch, participation_seasons=(2024,))
    assert update.main(["--full", "--dry-run", "--season", "2026"]) == 0
    log = _log(update)
    assert "--tables team_games player_games passer_games" in log
    assert "no participation file for 2026" in log
    assert "routes, backfield and player_usage cannot be built" in log


def test_the_full_chain_returns_the_week_the_participation_file_lands(update, tmp_path, monkeypatch):
    """Asked of the disk rather than the calendar, so nobody has to remember to turn it back on."""
    _fake_engine(tmp_path, monkeypatch, participation_seasons=(2024, 2025, 2026))
    assert update.main(["--full", "--dry-run", "--season", "2026"]) == 0
    log = _log(update)
    assert "--tables" not in log, "with participation on disk, every aggregate can be built"
    assert "build_aggregates.py --seasons 2026" in log


def test_the_engine_repos_own_interpreter_is_preferred(update, tmp_path):
    bare = tmp_path / "bare"
    bare.mkdir()
    assert update.python_for(bare) == sys.executable

    venv = tmp_path / "with_venv"
    scripts = venv / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    (scripts / "python.exe").write_text("")
    assert update.python_for(venv) == str(scripts / "python.exe")


# --------------------------------------------------------------------------- #
# what a run leaves behind
# --------------------------------------------------------------------------- #
def test_a_light_run_names_only_the_light_commands(update, monkeypatch):
    monkeypatch.setenv("NFLSP_ENGINE_DIR", str(ROOT))
    assert update.main(["--dry-run"]) == 0
    log = _log(update)
    assert "src.data.refresh" in log
    assert "bootstrap_data" not in log, "the weekly rebuild is not what a daily refresh does"


def test_a_dry_run_leaves_no_stamp(update):
    assert update.main(["--dry-run"]) == 0
    assert not update.STAMP.exists(), "nothing was updated, so nothing may claim it was"


def test_every_run_appends_rather_than_replacing_the_log(update):
    update.main(["--dry-run"])
    first = _log(update)
    update.main(["--dry-run"])
    assert _log(update).startswith(first)
    assert _log(update).count("=====") == 4        # two runs, two banner lines each side


def test_the_stamp_says_when_what_and_how_it_went(update, monkeypatch):
    """The app's sidebar reads this file to decide whether the server owes a restart, so the finish
    time and the mode are the two fields that cannot go missing."""
    monkeypatch.setattr(update, "run", lambda handle, cmd, cwd, dry: 0)
    assert update.main([]) == 0
    stamp = json.loads(update.STAMP.read_text(encoding="utf-8"))
    assert set(stamp) >= {"finished", "mode", "season", "heavy", "light"}
    assert stamp["mode"] == "light"
    assert stamp["heavy"] == "skipped"
    assert stamp["finished"][:2] == "20"


@pytest.mark.parametrize("light,heavy,want", [
    (0, "skipped", 0),
    (1, "skipped", 1),          # the preseason answer: nflreadpy refuses this season's injuries
    (2, "skipped", 2),          # rosters, depth charts or schedules missing
    (0, "failed at build_tables.py (1)", 2),
    # mid-season the heavy pass cannot build the participation tables and does not pretend to try. That
    # is the normal in-season result and must not be reported to the task scheduler as a failure, or the
    # one week something is genuinely broken looks exactly like the twenty before it.
    (0, "ok (partial)", 0),
])
def test_the_exit_code_is_how_bad_it_was(update, monkeypatch, light, heavy, want):
    monkeypatch.setattr(update, "light_pass", lambda handle, season, dry: light)
    monkeypatch.setattr(update, "heavy_pass", lambda handle, season, dry: heavy)
    assert update.main(["--full"]) == want
