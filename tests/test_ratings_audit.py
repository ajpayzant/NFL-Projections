"""The ratings themselves: found, populated, spent, and offered exactly when they are spent.

`scripts/audit_ratings.py` is the readable form of this -- one row per rating with its source, its
coverage and what the stat line does with it. These are the parts of that table that must never regress,
and they run against the real lake for the same reason the audit does: the failure being guarded against
is an upstream column quietly changing its name, which no fixture can reproduce.

Three claims, in the order they would cost accuracy:

1. **found.** A rating whose numerator or denominator is not a column of its table returns nothing from
   `own_rate` and hands every player the depth-slot prior. The projection still builds. Nothing says so.
2. **populated.** A share measured on a twentieth of the league is that prior with extra steps.
3. **classified.** `overrides.PLAYER_FIELDS` derives the knob list from the pools; the audit re-derives
   it from `compose`'s own constants. A rating a stat is built from and a user cannot type, or one he can
   type that reaches nothing, is the same defect seen from either side.

The fourth section is about the season in progress, which is not a fault but is a limit worth pinning
down: which ratings can close both halves of their ratio from the weekly results tables, and therefore
which of them learn from September rather than keeping their August estimate until the play-by-play
rebuild lands.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import ui  # noqa: E402

from src.config import PROJ_SEASON, Settings  # noqa: E402
from src.model import compose, opportunity, overrides, priors  # noqa: E402
from scripts.audit_ratings import MIN_COVERAGE, audit, findings, spent_how  # noqa: E402


@pytest.fixture(scope="module")
def table() -> pl.DataFrame:
    return audit(PROJ_SEASON)


# --------------------------------------------------------------------------- #
# found, populated, classified
# --------------------------------------------------------------------------- #
def test_every_rating_resolves_to_a_real_column_of_its_own_table(table):
    """The expensive failure, because it is the silent one.

    `estimate.own_rate` selects `metric.num` and `metric.den` out of the history frame and returns an
    empty answer if they are not there, so a renamed column costs the whole league its own record for
    that rating and reads on screen as everybody being exactly average at his job.
    """
    assert table.height == len(priors.METRICS)
    missing = table.filter(~pl.col("found"))
    assert missing.is_empty(), missing.select("metric", "table", "num", "den").rows()


def test_every_rating_is_measured_on_enough_of_the_league_to_be_a_measurement(table):
    thin = table.filter(pl.col("coverage") < MIN_COVERAGE)
    assert thin.is_empty(), thin.select("metric", "coverage").rows()
    # and the shrinkage is real: a typical player's own number carries some of his estimate, or the
    # metric is a prior that has been given a player's name
    assert table["own_weight"].min() > 0.0


def test_a_knob_is_a_rating_the_projection_spends_and_evidence_is_one_it_does_not(table):
    """The two lists are derived independently and have to agree, in both directions.

    `overrides` builds the editable set from the pools and the rate list; `spent_how` re-derives it from
    `compose.STAT_RATES`, `compose.STAT_POOLS` and `opportunity.POOLS`. A knob that moves nothing teaches
    a reader that the knobs move nothing, and a rating a stat is built from that he cannot type is a
    number he can see being wrong and cannot argue with.
    """
    knob_but_dead = table.filter(pl.col("editable") & (pl.col("spent") == "unused"))
    assert knob_but_dead.is_empty(), knob_but_dead["metric"].to_list()
    spent_but_hidden = table.filter(~pl.col("editable") & (pl.col("spent") != "unused"))
    assert spent_but_hidden.is_empty(), spent_but_hidden["metric"].to_list()


def test_the_audit_reports_nothing(table):
    assert findings(table) == []


def test_the_snap_shares_are_the_one_kind_of_evidence_that_is_also_a_knob():
    """Playing time is the engine's single declared exception and it is asserted rather than described.

    By shape a snap share is evidence: no stat is divided out of a team's snaps. It is spent anyway, as a
    multiplier on every other claim a man makes, which is what makes it editable for every offensive
    player. If that ever silently stopped being true the audit would keep reporting it as a knob.
    """
    for metric in overrides.PLAYING_TIME_METRICS:
        assert metric in priors.BY_NAME, metric
        assert metric in overrides.PLAYER_FIELDS, metric
        assert spent_how(metric) == "playing time", metric
        assert metric not in compose.STAT_RATES, f"{metric} is a share, not a rate"
    # ...and it is genuinely not a pool anything is divided out of, which is why it needed the exception
    pool_of = {s: p.name for p in opportunity.POOLS for s in p.shares}
    assert pool_of["snap_share"] not in compose.STAT_POOLS


def test_every_editable_rating_is_offered_somewhere_a_user_can_reach_it(table):
    """A knob nobody is shown is a knob nobody turns.

    The room sheets are the surface, so the check is that the editable ratings are covered by them plus
    the two numbers the participation frame puts on every sheet -- playing time and attendance, which
    reach it without being named in a room list.
    """
    offered = {c for fields in ui.ROOM_INPUTS.values() for c in fields}
    offered |= {"snap_share", "route_participation", *overrides.AVAILABILITY_FIELDS}
    knobs = set(table.filter(pl.col("editable"))["metric"].to_list())
    assert knobs <= offered, sorted(knobs - offered)


def test_a_room_sheet_names_real_ratings_and_lets_a_user_type_exactly_the_ones_it_spends(table):
    """The page's own lists, held to the same split as the engine's.

    `ROOM_INPUTS` mixes both on purpose -- the knobs a room is argued with and the measurements that
    justify moving one -- and the grid is handed the intersection with `PLAYER_FIELDS` rather than the
    list itself. So a typo here does not raise: the column is simply absent and the number a reader
    wanted is not on the sheet. This is what notices.
    """
    spent = dict(zip(table["metric"].to_list(), table["spent"].to_list(), strict=True))
    for pos, fields in ui.ROOM_INPUTS.items():
        assert len(set(fields)) == len(fields), f"{pos} lists a rating twice"
        assert ui.ROOM_HEADLINE[pos] in fields, f"{pos} opens on a rating it does not show"
        for name in fields:
            assert name in priors.BY_NAME, f"{pos}: {name} is not a rating"
            editable = name in overrides.PLAYER_FIELDS
            assert editable == (spent[name] != "unused"), f"{pos}: {name} is {spent[name]} and " \
                                                          f"{'' if editable else 'not '}editable"


# --------------------------------------------------------------------------- #
# what the season in progress can and cannot teach
# --------------------------------------------------------------------------- #
# Denominated in something the weekly results tables do not publish: routes run, red-zone and late-down
# volume, rush successes, dropbacks, and the scramble/designed split. These keep their preseason estimate
# all season and that is a data limit, not a bug -- the play-by-play rebuild is what closes them. Listed
# rather than counted so that a rating quietly falling off the live side has to be acknowledged here.
WAITING = {
    "rz_target_share", "rz_carry_share", "inside_5_carry_share", "short_yardage_carry_share",
    "late_down_target_share", "route_participation", "tprr", "rush_success_rate",
    "clean_rush_share", "yards_per_clean_rush", "dropback_share", "attempt_rate", "sack_rate",
    "scramble_rate", "designed_rush_share", "designed_rush_ypc", "scramble_ypc", "qb_fumble_rate",
}


def test_the_ratings_that_learn_from_this_season_are_the_ones_the_weekly_tables_can_measure(table):
    live = set(table.filter(pl.col("live"))["metric"].to_list())
    waiting = set(table.filter(~pl.col("live"))["metric"].to_list())
    assert waiting == WAITING, {"unexpectedly waiting": sorted(waiting - WAITING),
                               "unexpectedly live": sorted(WAITING - waiting)}
    # the ones that matter most are on the live side: volume and the rates a stat is built from
    assert {"target_share", "carry_share", "snap_share", "qb_snap_share", "catch_rate",
            "yards_per_target", "yards_per_carry", "yards_per_attempt"} <= live


def test_nothing_learns_from_a_season_that_is_already_over(table):
    """The leakage guard from the audit's side, which is the side a reader would check it from."""
    from src.model import estimate

    for m in priors.METRICS:
        assert estimate.to_date(m, 2024, Settings()) is None, m.name


def test_asking_for_the_model_alone_turns_the_whole_live_side_off():
    from src.model import estimate

    off = Settings(use_inseason_form=False)
    for m in priors.METRICS:
        assert estimate.to_date(m, PROJ_SEASON, off) is None, m.name
