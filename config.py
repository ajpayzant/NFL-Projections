"""Paths and constants for the NFL game projections tool."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = DATA / "models"
LIVE = DATA / "live"
OVERRIDES = DATA / "overrides"

# The projection engine (~/nfl-projection-system): its data lake, feature tables and fitted chain.
ENGINE = Path(os.environ.get("NFL_ENGINE_DIR", Path.home() / "nfl-projection-system"))

CURRENT_SEASON = 2026
FIRST_SEASON = 2016
POSITIONS = ("QB", "RB", "WR", "TE")
CANDIDATE_STATUSES = ("ACT", "INA")   # on the 53-man roster; INA is game-day inactive, only known at kickoff
ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
REQUEST_TIMEOUT = 30


def engine_on_path() -> None:
    """Make the engine importable (`from src... import ...`)."""
    if str(ENGINE) not in sys.path:
        sys.path.insert(0, str(ENGINE))
