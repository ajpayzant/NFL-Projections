"""The app's own logic: the parts of `app/ui.py` that are not Streamlit calls.

Almost everything in `ui` is chrome, and chrome is checked by running the pages -- `scripts/smoke_app.py`
does that in a real session, including clicking a reset and typing a multiplier. What is left, and what
is here, is the translation between an edited grid and a scenario: the one piece of the editing path
that a test can pin down exactly, and the one that would silently record the wrong `base` or the wrong
week if it were wrong.

`import ui` outside `streamlit run` warns about session state and then works, which is why the module
keeps a fallback for the no-session case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import ui  # noqa: E402

from src.model.overrides import Override, Scenario  # noqa: E402


@pytest.fixture(autouse=True)
def clean() -> None:
    """Every test starts from the engine's own answer, since the live scenario is module state."""
    ui.set_live(Scenario())


@pytest.fixture
def before() -> pl.DataFrame:
    return pl.DataFrame({
        "player_id": ["p1", "p2"],
        "player": ["A", "B"],
        "target_share": [0.24, 0.10],
        "catch_rate": [0.68, 0.62],
    })


FIELDS = ("target_share", "catch_rate")


def test_a_grid_nobody_touched_produces_nothing(before) -> None:
    assert ui.edits_from_grid(before, before, "player", "player_id", FIELDS) == []


def test_a_changed_cell_becomes_a_set_edit_on_that_player_alone(before) -> None:
    after = before.with_columns(
        pl.when(pl.col("player_id") == "p1").then(0.32).otherwise(pl.col("target_share"))
        .alias("target_share")
    )
    got = ui.edits_from_grid(before, after, "player", "player_id", FIELDS)
    assert len(got) == 1
    o = got[0]
    assert (o.level, o.key, o.field, o.mode, o.value) == ("player", "p1", "target_share", "set", 0.32)
    assert o.base == 0.24            # what it was worth before the edit, for the diff
    assert o.week is None


def test_the_recorded_base_stays_the_engines_number_across_a_second_edit(before) -> None:
    ui.edit(Override("player", "p1", "target_share", "set", 0.30, base=0.24))
    after = before.with_columns(pl.Series("target_share", [0.35, 0.10]))
    got = ui.edits_from_grid(before, after, "player", "player_id", FIELDS)
    assert got[0].value == 0.35
    assert got[0].base == 0.24


def test_a_weekly_grid_keys_each_edit_to_its_week() -> None:
    was = pl.DataFrame({"team": ["PHI"] * 3, "week": [1, 2, 3], "targets": [34.0, 30.0, 31.0]})
    now = was.with_columns(pl.Series("targets", [34.0, 40.0, 31.0]))
    got = ui.edits_from_grid(was, now, "team", "team", ("targets",), week_col="week")
    assert len(got) == 1
    assert (got[0].week, got[0].value, got[0].base) == (2, 40.0, 30.0)


def test_a_grid_of_a_different_shape_is_ignored_rather_than_misaligned(before) -> None:
    """Row order is the only thing tying the two frames together, so a different height is not an edit."""
    assert ui.edits_from_grid(before, before.head(1), "player", "player_id", FIELDS) == []


def test_the_view_is_the_scenarios_content_and_settings_follow_it() -> None:
    ui.set_live(Scenario(name="mine").patch_league(normalize_pools=False, scoring="half_ppr"))
    v = ui.view()
    assert v.payload == ui.live().content_json()
    assert v.scenario.digest == ui.live().digest
    assert v.settings.normalize_pools is False
    assert v.settings.scoring.reception == 0.5
    assert v.scoring == "half_ppr"


def test_renaming_the_live_scenario_does_not_change_what_is_cached() -> None:
    ui.edit(Override("player", "p1", "target_share", "set", 0.30))
    first = ui.view().payload
    ui.set_live(ui.live().rename("consensus"))
    assert ui.view().payload == first
