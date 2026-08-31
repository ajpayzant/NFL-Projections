"""Projected records, from the same per-game scoring level every other page reads.

The app could already say what a team was projected to *do* -- its plays, its dropbacks, the points its
offence was worth -- and could not say what any of that was projected to be *worth*, which is the one
number a football person checks a projection against. A team that comes out at 4-13 when the roster
looks like a contender is a projection with something wrong in it, and no per-player table shows that.

The chain is deliberately short, because a long one would be a second model of the season sitting
beside the one that produced the board:

1. **A game's margin is the difference in `implied_points`.** That column is the blended market/model
   scoring level `team.game_environment` already produces, one row per team per game, and it is an
   editable team field -- so a projected record moves when somebody says this offence is better than
   the market thinks. Nothing here re-derives it.
2. **A margin becomes a win probability through a normal CDF.** The only fitted quantity is that
   distribution's width, and it is measured rather than assumed: `margin_sigma` regresses ten seasons
   of actual margins on the same `implied_points` difference and takes the residual spread.
3. **A record is the sum of seventeen win probabilities.** Expected wins, not simulated ones. A team
   favoured in eleven games by three points apiece has not won eleven games, and adding probabilities
   says so where counting favourites would not.

What this deliberately does *not* do is decide tiebreakers or seed a playoff field. Expected wins are
continuous, so two teams are never actually tied, and a projected 9.4-7.6 does not contain the
information a tiebreaker needs. The division tables sort on expected wins and say so.
"""
from __future__ import annotations

import math
from functools import lru_cache

import polars as pl

from src.data import lake

# Divisions are a fact about the league, not something in the lake -- the tables carry team codes and
# nothing else. `LA` rather than `LAR` because that is what the play-by-play uses.
DIVISIONS: dict[str, tuple[str, ...]] = {
    "AFC East": ("BUF", "MIA", "NE", "NYJ"),
    "AFC North": ("BAL", "CIN", "CLE", "PIT"),
    "AFC South": ("HOU", "IND", "JAX", "TEN"),
    "AFC West": ("DEN", "KC", "LAC", "LV"),
    "NFC East": ("DAL", "NYG", "PHI", "WAS"),
    "NFC North": ("CHI", "DET", "GB", "MIN"),
    "NFC South": ("ATL", "CAR", "NO", "TB"),
    "NFC West": ("ARI", "LA", "SEA", "SF"),
}

DIVISION_OF: dict[str, str] = {t: d for d, teams in DIVISIONS.items() for t in teams}
CONFERENCE_OF: dict[str, str] = {t: d[:3] for t, d in DIVISION_OF.items()}

# The fallback if the lake cannot be read. Measured at 12.72 over 2016-2025; a round 13 is the number
# the literature uses, and being explicit about which one is in force matters more than the third digit.
SIGMA_FALLBACK = 13.0


def win_probability(margin: float, sigma: float = SIGMA_FALLBACK) -> float:
    """The chance a team favoured by `margin` points wins, under a normal error of width `sigma`.

    A tie is half a win to each side, which is what the CDF gives at a margin of zero without being
    asked. Ties are 0.4% of games, so nothing here is worth a special case.
    """
    if sigma <= 0:
        return 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
    return 0.5 * (1.0 + math.erf(float(margin) / (sigma * math.sqrt(2.0))))


def _margin_expr() -> pl.Expr:
    return pl.col("implied_points") - pl.col("implied_points_opp")


def with_opponent(env: pl.DataFrame) -> pl.DataFrame:
    """One row per team-game, with the opponent's scoring level joined onto it.

    `game_environment` carries both sides of a game as two rows; a margin needs them on one. Joined on
    (season, week, opponent) rather than on `game_id` so this works on a frame that has been filtered
    to one team, which is how the pages use it.
    """
    keys = [c for c in ("season", "week") if c in env.columns]
    other = env.select(
        *keys,
        pl.col("team").alias("opponent"),
        pl.col("implied_points").alias("implied_points_opp"),
    )
    return env.join(other, on=[*keys, "opponent"], how="left")


@lru_cache(maxsize=4)
def margin_sigma(seasons: tuple[int, ...] | None = None) -> float:
    """How wide the error on a projected margin actually is, in points, measured over `seasons`.

    The residual standard deviation of (actual margin) minus (the difference in `implied_points`) --
    i.e. exactly the quantity `standings` turns into a probability, scored against what happened. Not
    a constant from a paper: the app's own `implied_points` is a blend of market and model, so its
    error is its own and has to be measured on it.
    """
    try:
        seasons = seasons or lake.history_seasons()
        tg = lake.regular_season(lake.read("team_games", seasons=seasons))
        need = ("season", "week", "team", "opponent", "points", "points_allowed", "implied_points")
        if any(c not in tg.columns for c in need):
            return SIGMA_FALLBACK
        paired = with_opponent(tg.select(need)).drop_nulls(
            ["points", "points_allowed", "implied_points", "implied_points_opp"])
        if paired.height < 200:
            return SIGMA_FALLBACK
        resid = paired.select(
            ((pl.col("points") - pl.col("points_allowed")) - _margin_expr()).alias("e")
        )["e"]
        sd = resid.std()
        return float(sd) if sd and sd > 0 else SIGMA_FALLBACK
    except Exception:                     # noqa: BLE001 -- a missing table is a fallback, not a crash
        return SIGMA_FALLBACK


def game_margins(env: pl.DataFrame, sigma: float | None = None) -> pl.DataFrame:
    """Every projected game as a margin and a win probability, one row per team per game."""
    s = float(sigma if sigma is not None else margin_sigma())
    paired = with_opponent(env)
    keep = [c for c in ("season", "week", "game_id", "team", "opponent", "is_home", "div_game",
                        "implied_points", "implied_points_opp", "has_market") if c in paired.columns]
    out = paired.select(keep).with_columns(_margin_expr().alias("margin"))
    return out.with_columns(
        pl.col("margin")
        .map_elements(lambda m: win_probability(m, s), return_dtype=pl.Float64)
        .alias("win_prob")
    )


def standings(env: pl.DataFrame, sigma: float | None = None) -> pl.DataFrame:
    """Projected record per team: expected wins, points for and against, and the division it is in.

    Expected wins are a sum of probabilities, so they are fractional on purpose. `wins` and `losses`
    are the same number rounded for reading; `expected_wins` is the one to compare two teams on.
    """
    games = game_margins(env, sigma)
    if games.is_empty():
        return pl.DataFrame(schema={"team": pl.String, "division": pl.String, "conference": pl.String,
                                   "games": pl.Int64, "expected_wins": pl.Float64,
                                   "expected_losses": pl.Float64, "wins": pl.Int64,
                                   "losses": pl.Int64, "points_for": pl.Float64,
                                   "points_against": pl.Float64, "point_margin": pl.Float64,
                                   "points_per_game": pl.Float64, "allowed_per_game": pl.Float64,
                                   "division_wins": pl.Float64, "toughest_week": pl.Int64,
                                   "easiest_week": pl.Int64})
    div = pl.col("div_game") if "div_game" in games.columns else pl.lit(False)
    agg = games.group_by("team").agg(
        pl.len().alias("games"),
        pl.col("win_prob").sum().alias("expected_wins"),
        pl.col("implied_points").sum().alias("points_for"),
        pl.col("implied_points_opp").sum().alias("points_against"),
        pl.col("win_prob").filter(div).sum().alias("division_wins"),
        pl.col("week").sort_by("win_prob").first().alias("toughest_week"),
        pl.col("week").sort_by("win_prob").last().alias("easiest_week"),
    )
    return agg.with_columns(
        pl.col("team").replace_strict(DIVISION_OF, default="—").alias("division"),
        pl.col("team").replace_strict(CONFERENCE_OF, default="—").alias("conference"),
        (pl.col("games") - pl.col("expected_wins")).alias("expected_losses"),
        (pl.col("points_for") - pl.col("points_against")).alias("point_margin"),
        (pl.col("points_for") / pl.col("games")).alias("points_per_game"),
        (pl.col("points_against") / pl.col("games")).alias("allowed_per_game"),
    ).with_columns(
        pl.col("expected_wins").round(0).cast(pl.Int64).alias("wins"),
        pl.col("expected_losses").round(0).cast(pl.Int64).alias("losses"),
    ).select(
        "team", "division", "conference", "games", "wins", "losses", "expected_wins",
        "expected_losses", "points_for", "points_against", "point_margin", "points_per_game",
        "allowed_per_game", "division_wins", "toughest_week", "easiest_week",
    ).sort(["division", "expected_wins"], descending=[False, True])


def accuracy(seasons: tuple[int, ...] | None = None) -> dict[str, float]:
    """How well this win probability did on the seasons it was fitted from.

    On the page beside the standings, because a projected record with no error attached invites being
    read as a forecast of nine wins rather than of somewhere between six and twelve. `brier` is the
    mean squared error of the probability; `straight_up` is how often the favourite actually won.
    """
    out = {"sigma": margin_sigma(seasons), "games": 0.0, "brier": float("nan"),
           "straight_up": float("nan")}
    try:
        seasons = seasons or lake.history_seasons()
        tg = lake.regular_season(lake.read("team_games", seasons=seasons))
        paired = with_opponent(tg.select("season", "week", "team", "opponent", "points",
                                         "points_allowed", "implied_points")).drop_nulls(
            ["points", "points_allowed", "implied_points", "implied_points_opp"])
        if paired.is_empty():
            return out
        s = out["sigma"]
        scored = paired.with_columns(
            (pl.col("points") - pl.col("points_allowed")).alias("actual"),
            _margin_expr().alias("margin"),
        ).with_columns(
            pl.col("margin").map_elements(lambda m: win_probability(m, s), return_dtype=pl.Float64)
            .alias("p"),
            pl.when(pl.col("actual") > 0).then(1.0)
            .when(pl.col("actual") < 0).then(0.0).otherwise(0.5).alias("won"),
        )
        decided = scored.filter(pl.col("actual") != 0)
        out["games"] = float(scored.height)
        out["brier"] = float(((pl.Series(scored["p"]) - pl.Series(scored["won"])) ** 2).mean())
        out["straight_up"] = float(
            decided.select(((pl.col("margin") > 0) == (pl.col("actual") > 0)).mean()).item())
    except Exception:                     # noqa: BLE001
        return out
    return out
