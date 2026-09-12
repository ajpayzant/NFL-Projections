"""Every path, season and tunable in one place.

Two rules this module exists to enforce:

- **No magic numbers downstream.** A scoring rule, a recency weight or a shrinkage constant is a
  value a user is entitled to change, so it lives in a dataclass with a default rather than inline
  in the formula that reads it. `Settings` is what the app's adjustment layer edits.
- **The lake is somebody else's.** The heavy `processed/` tables are built by the pbp pipeline in
  `~/nfl-projection-system`; this project reads them and never writes them. `LAKE` points at that
  repo's `data/` and `OWN` points at ours. Anything we generate goes under `OWN`, so a reader can
  tell at a glance which numbers we are responsible for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The shared parquet lake. Override with NFLSP_DATA_DIR to point at a copy.
LAKE = Path(os.environ.get("NFLSP_DATA_DIR", Path.home() / "nfl-projection-system" / "data"))
OWN = ROOT / "data"

RAW = LAKE / "raw"
PROCESSED = LAKE / "processed"
OWN_RAW = OWN / "raw"          # the light 2026 tables we refresh ourselves
SCENARIOS = OWN / "scenarios"
CACHE = OWN / "cache"
FITTED = OWN / "fitted"        # priors, shrinkage constants, dispersion — our outputs

PROJ_SEASON = 2026
LAST_COMPLETE_SEASON = 2025
HISTORY_FROM = 2016            # earliest season with a complete processed table
PARTICIPATION_FROM = 2021      # snap/route participation is only reliable from here
REG_WEEKS = 18                 # 18-week calendar, 17 games per team


@dataclass(frozen=True)
class Scoring:
    """Fantasy scoring. Defaults are full PPR, 4-point passing touchdowns."""

    pass_yard: float = 0.04
    pass_td: float = 4.0
    interception: float = -2.0
    completion: float = 0.0
    rush_yard: float = 0.1
    rush_td: float = 6.0
    rec_yard: float = 0.1
    rec_td: float = 6.0
    reception: float = 1.0
    fumble_lost: float = -2.0

    @classmethod
    def preset(cls, name: str) -> Scoring:
        presets = {
            "ppr": cls(),
            "half_ppr": cls(reception=0.5),
            "standard": cls(reception=0.0),
            "ppr_6td": cls(pass_td=6.0),
            "superflex_ppr": cls(),  # scoring is identical; the difference is roster construction
        }
        if name not in presets:
            raise KeyError(f"unknown scoring preset {name!r}; have {sorted(presets)}")
        return presets[name]


@dataclass(frozen=True)
class Settings:
    """Everything the projection depends on that is not data.

    `recency` weights the seasons behind a player's own history, most recent first. They are
    relative, not normalised: what matters is their ratio. Blending happens on *counts* rather than
    on rates, so a four-game season already contributes four games of weight and these numbers do
    not have to compensate for playing time.
    """

    scoring: Scoring = field(default_factory=Scoring)
    recency: tuple[float, ...] = (5.0, 3.0, 2.0)

    # Team season shape: estimate = league_mean + keep * (recent_form - league_mean), where
    # recent_form = w * last_season + (1 - w) * the season before. Grid-searched on 2019-2025 in
    # the workbook this replaces; re-fitted by scripts/fit_priors.py over the wider metric set.
    team_weight_recent: float = 0.80
    team_keep_vs_mean: float = 0.40

    # Shrinkage of a player's own rate toward his position/depth prior: n / (n + k) on his blended
    # opportunity count. Loaded from FITTED/shrinkage.json when fitted; these are the fallbacks.
    default_share_k: float = 60.0
    default_rate_k: float = 120.0

    # Per-game context factors. `context_k` shrinks a split's bucket factor toward 1 by
    # n / (n + k) team-games, so a 40-game bucket cannot swing a projection the way a 2,000-game one
    # can. `market_weight` is how much of a game's scoring level comes from the posted line rather
    # than from our own team estimates; None means use the fitted value in FITTED/team_market.json.
    context_k: float = 200.0
    market_weight: float | None = None
    use_context_factors: bool = True

    # Off by default: the schedule is allowed to move a team's season total, because a genuinely
    # easy schedule is worth something. Turning this on renormalises each team's 17 factors to
    # average 1, so the schedule only redistributes within the season.
    schedule_renormalise: bool = False

    normalize_pools: bool = True     # rescale exclusive team pools to sum to 1
    games_projected: int = 17        # per team; per player it is availability-adjusted

    # In season, a week that has been played is replaced by what happened in it, so a season total is
    # results-to-date plus the projected rest rather than a forecast of a game whose score is known.
    # On by default because that is what a projection made in November means; off is the honest way to
    # ask what the model would say on its own, which is the only way to read its own accuracy.
    # `compose.actualise` ignores it for any season but the one in progress -- the backtest scores a
    # finished season against these same tables, and substituting them there would report a perfect model.
    use_actuals: bool = True

    # And the games not yet played learn from the ones that were. `use_actuals` replaces a finished week
    # with its result; this is the other half, and the half that changes the *rest* of the season: a
    # receiver running a 28% target share through five games is evidence about his remaining twelve.
    # `estimate.to_date` shapes the season so far as one more season of the player's own history, so it
    # is blended and shrunk by exactly the machinery three years of history go through, and weighted by
    # how much of it there is -- a rumour in week 1, the dominant evidence by December, with no schedule
    # anybody chose. Off is the model on its own, which is the only way to read what August was worth.
    # Ignored for any season but the one in progress, for the same reason as `use_actuals`.
    use_inseason_form: bool = True

    # Who pays when a room claims more of a pool than the pool holds. The residual is taken in
    # proportion to `claim ** pool_tilt`, so the exponent is the whole behaviour:
    #
    #   1.0  every claim loses the same *fraction* of itself -- the flat proportional rescale
    #   <1   a small claim loses a larger fraction than a big one
    #   0.0  every claim loses the same *absolute* amount, so the bench empties first
    #
    # 1.0 reproduces the old rescale exactly, which is why it is the fallback rather than a value.
    # The number lives in FITTED/pool_tilt.json and is measured the same way every other constant here
    # is -- on held-out seasons, `python -m src.model.backtest --fit-tilt` -- and unlike the others, the
    # measurement came back indifferent. Held-out per-game target and carry MAE moves by 0.04% across
    # the whole grid from 1.0 down to 0.0, which is noise; the calibration slope of WR targets per game
    # is 0.95, meaning the projections are already a shade too spread, which argues mildly *against*
    # taking more from the bench. So this exponent is a preference the accuracy measurement does not
    # object to rather than a result it produced, and the default is the mildest tilt on the grid.
    # Read `choose_tilt` before quoting a fitted value for it.
    pool_tilt: float | None = None    # None means use the fitted value
    # An explicit share override is held at what was typed and the residual is taken from the rest of
    # the room instead. Off, a number a user typed is silently rescaled with everything else -- which
    # made "set his target share to 0.30" deliver 0.2545 and report itself applied.
    lock_edited_shares: bool = True

    # Preseason roster status -> the fraction of his fitted expected games a player is credited with.
    # The one number in the engine that is stated rather than fitted: 20 of 2026's 915 offensive
    # players are anything but ACT, and the historical week-1 status column is missing for 2017-2018
    # and inconsistent elsewhere, so there is no honest sample to measure against. Out in the open
    # here so it can be moved, and reported as an assumption rather than a result.
    status_availability: dict[str, float] = field(default_factory=lambda: {
        "ACT": 1.0, "DEV": 1.0,            # active, or practice squad and promotable
        "E14": 0.75,                       # exempt, expected back
        "PUP": 0.55, "NON": 0.50, "SUS": 0.60,
        "RES": 0.30,                       # injured reserve in August: often but not always the year
        "CUT": 0.0, "RET": 0.0, "EXE": 0.0, "TRC": 0.0,
    })
    simulation_draws: int = 10_000
    starters: dict[str, int] = field(default_factory=lambda: {"QB": 12, "RB": 24, "WR": 36, "TE": 12})
    tier_size: int = 6

    def with_scoring(self, name: str) -> Settings:
        return replace(self, scoring=Scoring.preset(name))


POSITIONS = ("QB", "RB", "WR", "TE")

# Offensive positions worth carrying through the roster. FB is folded into RB everywhere: the
# depth charts list them separately but they compete for the same carries.
OFFENSE_POSITIONS = ("QB", "RB", "FB", "WR", "TE")

# Abbreviations used by rosters, depth charts and the draft table, mapped to the processed
# tables' vocabulary. Without this every Cardinal reads as having changed teams.
TEAM_FIXUP = {
    "AZ": "ARI", "ARZ": "ARI", "LAR": "LA", "RAM": "LA", "WSH": "WAS", "JAC": "JAX",
    "CLV": "CLE", "BLT": "BAL", "HST": "HOU", "SL": "LA", "OAK": "LV", "SD": "LAC",
    # pro-football-reference style, which the draft table uses
    "LVR": "LV", "NOR": "NO", "GNB": "GB", "KAN": "KC", "NWE": "NE", "SFO": "SF",
    "TAM": "TB", "RAI": "LV", "STL": "LA", "PHO": "ARI",
}


def ensure_dirs() -> None:
    for d in (OWN, OWN_RAW, SCENARIOS, CACHE, FITTED):
        d.mkdir(parents=True, exist_ok=True)
