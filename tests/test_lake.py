"""How old the data is, and — the question age cannot answer — whether it has kept up with the season.

`status` says when a table was last written, which is necessary and not sufficient: a table refreshed
this morning can still be missing last Sunday's games. `current_week` and `staleness` answer the
question that has a right answer, so what is pinned here is the arithmetic of "behind", the boundaries
either side of a season, and that a missing table is reported as a finding rather than as zero weeks
behind.

The week is read from the schedule on purpose, so these tests drive it with an explicit `today` and a
fake lake. A test that used the real clock would pass in September and fail in February.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.config import PROJ_SEASON
from src.data import lake

# Five weeks of a fictional season, one game each, a week apart.
SCHEDULE = pl.DataFrame({
    "season": [2026] * 5,
    "week": [1, 2, 3, 4, 5],
    "gameday": ["2026-09-10", "2026-09-17", "2026-09-24", "2026-10-01", "2026-10-08"],
})


def fake_lake(monkeypatch, **tables: pl.DataFrame | None) -> None:
    """Point `read` at frames we control. `None` means the table is absent and raises, as it does."""

    def read(dataset: str, layer: str = "processed", seasons=None) -> pl.DataFrame:
        if dataset not in tables or tables[dataset] is None:
            raise FileNotFoundError(dataset)
        return tables[dataset]

    monkeypatch.setattr(lake, "read", read)


def usage(max_week: int) -> pl.DataFrame:
    return pl.DataFrame({"season": [2026] * max_week, "week": list(range(1, max_week + 1))})


# --------------------------------------------------------------------------- #
# which week it is
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("today", "want"), [
    (date(2026, 8, 24), 0),         # before the opener there is no week to be behind on
    (date(2026, 9, 10), 1),         # the day of the first game is week 1, not week 0
    (date(2026, 9, 23), 2),         # mid-week is still the week whose games have kicked off
    (date(2026, 10, 8), 5),
    (date(2027, 3, 1), 5),          # after the last scheduled game it stops, it does not keep counting
])
def test_the_week_comes_from_the_schedule_and_not_from_a_hard_coded_date(monkeypatch, today, want):
    fake_lake(monkeypatch, schedules=SCHEDULE)
    assert lake.current_week(2026, today) == want


def test_the_playoffs_are_not_more_weeks_to_be_behind_on(monkeypatch):
    """Weeks 19-22 live in the same table, and a 17-game projection is not stale in February."""
    post = pl.DataFrame({
        "season": [2026] * 7, "week": [1, 2, 3, 4, 5, 19, 20],
        "gameday": [*SCHEDULE["gameday"].to_list(), "2027-01-16", "2027-01-23"],
        "game_type": ["REG"] * 5 + ["WC", "DIV"],
    })
    fake_lake(monkeypatch, schedules=post)
    assert lake.current_week(2026, date(2027, 2, 1)) == 5


def test_no_schedule_reads_as_no_season_in_progress(monkeypatch):
    """The honest answer to a question we cannot resolve, and it must not raise on a fresh install."""
    fake_lake(monkeypatch, schedules=None)
    assert lake.current_week(2026, date(2026, 10, 1)) == 0
    fake_lake(monkeypatch, schedules=SCHEDULE.head(0))
    assert lake.current_week(2026, date(2026, 10, 1)) == 0
    fake_lake(monkeypatch, schedules=SCHEDULE.drop("gameday"))
    assert lake.current_week(2026, date(2026, 10, 1)) == 0


# --------------------------------------------------------------------------- #
# whether processed/ has kept up
# --------------------------------------------------------------------------- #
def test_before_the_season_nothing_is_stale(monkeypatch):
    fake_lake(monkeypatch, schedules=SCHEDULE)
    got = lake.staleness(2026, date(2026, 8, 24))
    assert got["current_week"] == 0
    assert got["in_season"] is False
    assert got["stale"] is False
    assert got["weeks_behind"] == 0


def test_a_lake_carrying_this_week_is_current(monkeypatch):
    fake_lake(monkeypatch, schedules=SCHEDULE, team_games=usage(4), player_usage=usage(4))
    got = lake.staleness(2026, date(2026, 10, 1))
    assert got["current_week"] == 4
    assert got["latest_week"] == 4
    assert got["weeks_behind"] == 0
    assert got["stale"] is False


def test_the_lagging_table_sets_the_verdict(monkeypatch):
    """`min`, not `max`: a projection reads both tables, so it is as stale as the worse of them."""
    fake_lake(monkeypatch, schedules=SCHEDULE, team_games=usage(5), player_usage=usage(3))
    got = lake.staleness(2026, date(2026, 10, 8))
    assert got["tables"] == {"team_games": 5, "player_usage": 3}
    assert got["latest_week"] == 3
    assert got["weeks_behind"] == 2
    assert got["stale"] is True


def test_a_missing_table_is_a_finding_and_not_zero_weeks_behind(monkeypatch):
    """The failure this guards: an absent table reported as 'current' is the worst possible answer."""
    fake_lake(monkeypatch, schedules=SCHEDULE, team_games=None, player_usage=None)
    got = lake.staleness(2026, date(2026, 10, 8))
    assert got["tables"] == {"team_games": None, "player_usage": None}
    assert got["latest_week"] is None
    assert got["stale"] is True


def test_an_empty_partition_counts_as_missing(monkeypatch):
    fake_lake(monkeypatch, schedules=SCHEDULE, team_games=usage(0), player_usage=usage(2))
    got = lake.staleness(2026, date(2026, 10, 8))
    assert got["tables"]["team_games"] is None
    assert got["latest_week"] == 2
    assert got["stale"] is True


# --------------------------------------------------------------------------- #
# against the real lake
# --------------------------------------------------------------------------- #
def test_the_real_schedule_answers_for_the_projection_season():
    """Not an assertion about today: only that the real table parses and stays inside the calendar."""
    if not lake.lake_present():
        pytest.skip("no lake on this machine")
    week = lake.current_week(PROJ_SEASON, date(2026, 1, 1))
    assert week == 0, "no 2026 game has been played on new year's day"
    assert lake.current_week(PROJ_SEASON, date(2027, 6, 1)) in range(17, 19)
    live = lake.staleness(PROJ_SEASON)
    assert 0 <= live["current_week"] <= 18
    assert set(live) == {"season", "current_week", "in_season", "weeks_behind", "latest_week",
                        "tables", "stale"}
