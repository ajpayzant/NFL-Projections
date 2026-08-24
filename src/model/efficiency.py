"""What a player does with an opportunity once he has it, and how the game changes that.

Two parts, deliberately separated:

1. **The player's own rate**, from the one estimator in `estimate` -- his recency-weighted history
   shrunk toward his (position, depth-slot) prior by `n / (n + k)`, with `k` fitted per rate. This is
   where the workbook was worst: it took a 40-carry sample of yards per carry at face value, and a
   backup with one 30-yard run read as a 6.0 YPC back for a season.
2. **The game's multiplier**, from the team layer's per-game chain divided by its own season-flat
   estimate. If Philadelphia's week-4 environment projects 4.6 yards a carry against a season-long 4.3,
   every Eagles ball-carrier's YPC is multiplied by 1.07 that week. The factor is the team's, not the
   player's, because a defence is faced by the offence rather than by one runner.

**Only the rates the team layer actually has an opinion about are adjusted.** Yards per attempt and
yards per carry are projected per game by `team.py`, so those two families move. Catch rate, aDOT,
interception rate and sack rate are left flat: there is a per-game story for each, but the team layer
does not project them and inventing a factor here would be a number with no measurement behind it.

**Efficiency is mostly not projectable and the fit says so.** Yards per carry gains 4.0% over a
player's own history and yards per target 4.7%, against 19-23% for the shares. That is the finding,
not a failure -- it is why the engine spends its effort on opportunity and lets efficiency regress.

    python -m src.model.efficiency
"""

from __future__ import annotations

import argparse

import polars as pl

from src.config import PROJ_SEASON, Settings
from src.model import estimate, priors, roster

SKILL_RATES = ("catch_rate", "yards_per_target", "yards_per_carry", "adot", "tprr",
               "rush_success_rate", "fumble_rate")

QB_RATES = ("attempt_rate", "completion_pct", "yards_per_attempt", "pass_td_rate", "int_rate",
            "sack_rate", "scramble_rate", "air_yards_per_attempt", "yards_per_clean_rush",
            "scramble_ypc", "designed_rush_ypc", "qb_fumble_rate")

RATE_METRICS = SKILL_RATES + QB_RATES

# rate metric -> (per-game team estimate, season-flat team estimate). The ratio is the multiplier.
# Absent from this map means "no per-game adjustment", which is the honest default.
CONTEXT_OF = {
    "yards_per_target": ("team_yards_per_attempt", "team_est_yards_per_attempt"),
    "yards_per_attempt": ("team_yards_per_attempt", "team_est_yards_per_attempt"),
    "yards_per_carry": ("team_yards_per_carry", "team_est_yards_per_carry"),
    "yards_per_clean_rush": ("team_yards_per_carry", "team_est_yards_per_carry"),
    "designed_rush_ypc": ("team_yards_per_carry", "team_est_yards_per_carry"),
    "scramble_ypc": ("team_yards_per_carry", "team_est_yards_per_carry"),
    "rush_success_rate": ("team_success_rate", "team_est_success_rate"),
}

# A single game's environment may not double or halve a rate. The chain is a product of small factors
# and the tails are where a missing market line or a thin split bucket shows up.
FACTOR_CLIP = (0.80, 1.25)


def rates(
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
    ros: pl.DataFrame | None = None,
    names: tuple[str, ...] = RATE_METRICS,
    fitted: priors.Fitted | None = None,
) -> pl.DataFrame:
    """One row per player, one column per rate. Same estimator as every share."""
    settings = settings or Settings()
    ros = roster.roster(season) if ros is None else ros
    return estimate.wide(estimate.estimate(names, ros, season, settings, fitted), names)


def rate_detail(
    season: int = PROJ_SEASON,
    settings: Settings | None = None,
    ros: pl.DataFrame | None = None,
    names: tuple[str, ...] = RATE_METRICS,
    fitted: priors.Fitted | None = None,
) -> pl.DataFrame:
    """The long form: obs, n, prior, used and own_weight per rate, for the player page."""
    settings = settings or Settings()
    ros = roster.roster(season) if ros is None else ros
    return estimate.estimate(names, ros, season, settings, fitted)


def context_factors(frame: pl.DataFrame, settings: Settings | None = None) -> pl.DataFrame:
    """Add `f_<rate>` for every rate the team layer projects per game; 1.0 for the rest.

    `frame` is a player-game frame carrying the `team_*` columns from `opportunity`. The factor is
    clipped, and the clip is reported rather than hidden: a factor at the bound means the game
    environment is being trusted further than the split behind it can support.
    """
    settings = settings or Settings()
    exprs = []
    for name in RATE_METRICS:
        spec = CONTEXT_OF.get(name)
        if spec is None or not settings.use_context_factors:
            exprs.append(pl.lit(1.0).alias(f"f_{name}"))
            continue
        game, flat = spec
        if game not in frame.columns or flat not in frame.columns:
            exprs.append(pl.lit(1.0).alias(f"f_{name}"))
            continue
        exprs.append(
            pl.when(pl.col(flat) > 0)
            .then((pl.col(game) / pl.col(flat)).clip(*FACTOR_CLIP))
            .otherwise(pl.lit(1.0))
            .alias(f"f_{name}")
        )
    return frame.with_columns(exprs)


def adjusted(frame: pl.DataFrame, player_rates: pl.DataFrame,
             settings: Settings | None = None) -> pl.DataFrame:
    """Join a player's rates onto his games and apply the game factor. `used_<rate>` is the answer."""
    have = [c for c in RATE_METRICS if c in player_rates.columns]
    out = context_factors(
        frame.join(player_rates.select("player_id", *have), on="player_id", how="left"), settings
    )
    return out.with_columns(
        [(pl.col(c) * pl.col(f"f_{c}")).alias(f"used_{c}") for c in have]
    )


def _report(season: int, settings: Settings) -> None:
    from src.model import opportunity

    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_rows(40)
    pl.Config.set_fmt_float("mixed")

    detail = rate_detail(season, settings)
    print(f"\nRATES  {season}: how much of each answer is the player rather than his position")
    print(detail.group_by("metric").agg(
        pl.col("k").first(),
        pl.col("used").mean().round(3).alias("mean_used"),
        pl.col("own_weight").mean().round(2).alias("mean_own_weight"),
        (pl.col("source") == "blend").sum().alias("from_blend"),
        (pl.col("source") == "slot_prior").sum().alias("from_slot"),
        (pl.col("source") == "draft_blend").sum().alias("from_draft"),
    ).sort("mean_own_weight", descending=True))

    opp = opportunity.opportunity(season, settings)
    adj = adjusted(opp, rates(season, settings), settings)
    fcols = sorted({f"f_{n}" for n in CONTEXT_OF})
    # per team-game, not per team: the factor is a property of one matchup, and averaging a team's
    # season first would report the spread of team means and understate the real range by half.
    per_game = adj.group_by(["game_id", "team"]).agg([pl.col(c).mean() for c in fcols])
    print(f"\nGAME FACTORS  the per-game efficiency multiplier over {per_game.height} team-games")
    print(pl.DataFrame({
        "factor": fcols,
        "min": [round(float(per_game[c].min()), 3) for c in fcols],
        "p10": [round(float(per_game[c].quantile(0.10)), 3) for c in fcols],
        "median": [round(float(per_game[c].median()), 3) for c in fcols],
        "p90": [round(float(per_game[c].quantile(0.90)), 3) for c in fcols],
        "max": [round(float(per_game[c].max()), 3) for c in fcols],
    }))
    at_bound = adj.select(
        [(pl.col(c).is_between(*FACTOR_CLIP, closed="none").not_()).mean().alias(c) for c in fcols]
    )
    print("\nfraction of player-games where the factor hit the clip")
    print(at_bound.select([pl.col(c).round(4) for c in fcols]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, default=PROJ_SEASON)
    a = p.parse_args(argv)
    _report(a.season, Settings())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
