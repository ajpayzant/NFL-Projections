"""Projected records: the arithmetic that has to hold, and the one fitted number.

A record is the only thing in the app a football person will check against instinct before checking
anything else, so the invariants here are the ones that make it readable at all -- somebody has to win
every game, a season is seventeen of them, and a better team is favoured. The fitted width of the
margin distribution is asserted to be *in a plausible range* rather than equal to a value, because it
is measured from the lake and a data refresh is allowed to move it.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.config import PROJ_SEASON
from src.model import overrides, standings


@pytest.fixture(scope="module")
def env() -> pl.DataFrame:
    return overrides.run(None, PROJ_SEASON).env


@pytest.fixture(scope="module")
def table(env) -> pl.DataFrame:
    return standings.standings(env)


# --------------------------------------------------------------------------- #
# the probability itself
# --------------------------------------------------------------------------- #
def test_a_pick_em_game_is_a_coin_toss():
    assert standings.win_probability(0.0, 13.0) == pytest.approx(0.5)


def test_the_favourite_is_more_likely_to_win_than_the_underdog():
    assert standings.win_probability(7.0, 13.0) > 0.5 > standings.win_probability(-7.0, 13.0)


def test_the_two_sides_of_one_game_are_one_probability_between_them():
    """Anything else and expected wins would not add up to the games actually being played."""
    for margin in (0.0, 1.5, 3.0, 7.0, 14.0, 24.0):
        assert (standings.win_probability(margin, 12.7)
                + standings.win_probability(-margin, 12.7)) == pytest.approx(1.0)


def test_a_bigger_favourite_is_a_bigger_favourite():
    ladder = [standings.win_probability(m, 13.0) for m in (0, 3, 7, 10, 14)]
    assert ladder == sorted(ladder)


def test_a_narrower_error_makes_the_same_edge_worth_more():
    """The width is what turns three points into a probability, so it has to be doing that work."""
    assert standings.win_probability(3.0, 8.0) > standings.win_probability(3.0, 18.0)


def test_no_error_at_all_makes_the_favourite_certain():
    assert standings.win_probability(3.0, 0.0) == 1.0
    assert standings.win_probability(-3.0, 0.0) == 0.0


# --------------------------------------------------------------------------- #
# the fitted width
# --------------------------------------------------------------------------- #
def test_the_margin_error_is_measured_and_lands_where_football_says_it_should():
    """Twelve to fifteen points is the range every published estimate of this sits in."""
    assert 10.0 < standings.margin_sigma() < 16.0


def test_the_fit_is_scored_and_beats_a_coin_toss():
    got = standings.accuracy()
    assert got["games"] > 1000
    assert 0.15 < got["brier"] < 0.25          # 0.25 is what always saying 50% would score
    assert 0.60 < got["straight_up"] < 0.75


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
def test_every_team_gets_a_row_in_a_division():
    assert table_teams(standings.DIVISIONS) == 32
    assert len(standings.DIVISION_OF) == 32
    assert set(standings.CONFERENCE_OF.values()) == {"AFC", "NFC"}


def table_teams(divisions) -> int:
    return len({t for teams in divisions.values() for t in teams})


def test_the_league_projects_to_win_exactly_half_its_games(table):
    """The invariant a record table lives or dies on: one winner a game, no more and no fewer."""
    assert table["expected_wins"].sum() == pytest.approx(table["games"].sum() / 2, abs=1e-6)


def test_a_projected_season_is_seventeen_games_for_everybody(table):
    assert table["games"].unique().to_list() == [17]
    assert (table["expected_wins"] + table["expected_losses"]).to_list() == pytest.approx(
        [17.0] * table.height)


def test_nobody_is_projected_to_win_a_game_they_cannot(table):
    assert table["expected_wins"].min() >= 0.0
    assert table["expected_wins"].max() <= float(table["games"].max())


def test_the_team_that_outscores_its_schedule_is_the_team_projected_to_win(table):
    """Not a tautology: it goes through seventeen separate probabilities, and could fail to survive."""
    ranked = table.sort("expected_wins", descending=True)
    assert ranked["point_margin"][0] > ranked["point_margin"][-1]
    assert ranked.select(pl.corr("expected_wins", "point_margin")).item() > 0.9


def test_division_wins_cannot_exceed_the_division_games_played(table):
    """Six of a team's seventeen are inside the division, so this is the ceiling."""
    assert table["division_wins"].max() <= 6.0
    assert table["division_wins"].min() >= 0.0


def test_editing_a_team_up_moves_that_team_up_the_standings():
    """The whole point of deriving this from `implied_points`: an override has to reach it."""
    bare = standings.standings(overrides.run(None, PROJ_SEASON).env)
    was = bare.filter(pl.col("team") == "DET").row(0, named=True)
    edits = tuple(
        overrides.Override("team", "DET", "implied_points", "multiply", 1.25, week=w)
        for w in range(1, 19)
    )
    after = standings.standings(
        overrides.run(overrides.Scenario().set(*edits), PROJ_SEASON).env)
    now = after.filter(pl.col("team") == "DET").row(0, named=True)
    assert now["expected_wins"] > was["expected_wins"] + 0.5
    assert now["points_for"] > was["points_for"]
    # and somebody had to pay for it, because the league still only plays 272 games
    assert after["expected_wins"].sum() == pytest.approx(bare["expected_wins"].sum(), abs=1e-6)


def test_an_empty_environment_gives_an_empty_table_with_the_same_columns(table):
    empty = standings.standings(table.head(0).select("team").with_columns(
        pl.lit(0).alias("week"), pl.lit("").alias("opponent"),
        pl.lit(0.0).alias("implied_points")))
    assert empty.is_empty()
    assert "expected_wins" in empty.columns


# --------------------------------------------------------------------------- #
# per game
# --------------------------------------------------------------------------- #
def test_a_game_carries_both_sides_scoring_levels(env):
    games = standings.game_margins(env)
    assert games.height == env.height
    assert games["win_prob"].is_between(0.0, 1.0).all()
    assert games["margin"].abs().max() < 40.0        # a projected 40-point spread is a broken chain


def test_the_home_and_away_rows_of_one_game_disagree_exactly(env):
    games = standings.game_margins(env)
    one = games.filter(pl.col("game_id") == games["game_id"][0])
    assert one.height == 2
    assert one["margin"].sum() == pytest.approx(0.0, abs=1e-9)
    assert one["win_prob"].sum() == pytest.approx(1.0, abs=1e-9)
