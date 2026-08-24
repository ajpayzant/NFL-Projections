"""The projectable roster: membership, depth, expected games, participation.

Real lake, football invariants rather than fixed numbers, as in the other suites. The four that carry
the most weight here are the population guard (every rostered player is projected, and nobody else),
the product guard (a weekly contribution is games x participation and cannot exceed either), the
leakage guard (an availability fit for season S must not read season S), and the identity guard (five
skill players are on the field for every snap, so the team sums have to come back to five).
"""

from __future__ import annotations

import polars as pl
import pytest

from src.config import PROJ_SEASON, REG_WEEKS, Settings
from src.data import depth, lake
from src.model import priors, roster


@pytest.fixture(scope="module")
def ros() -> pl.DataFrame:
    return roster.roster(PROJ_SEASON)


@pytest.fixture(scope="module")
def part() -> pl.DataFrame:
    return roster.participation(PROJ_SEASON)


@pytest.fixture(scope="module")
def detail() -> pl.DataFrame:
    return roster.participation_detail(PROJ_SEASON)


@pytest.fixture(scope="module")
def fit() -> dict:
    f = roster.load_availability()
    if not f:
        pytest.skip("no fitted availability on disk; run python -m src.model.roster --fit")
    return f


@pytest.fixture(scope="module")
def slot_prior() -> pl.DataFrame:
    p = roster.load_slot_games()
    if p.is_empty():
        pytest.skip("no fitted slot prior on disk")
    return p


# --------------------------------------------------------------------------- #
# membership: the roster decides who exists
# --------------------------------------------------------------------------- #
def test_the_roster_is_the_population_and_the_chart_cannot_add_to_it(ros):
    raw = lake.read("rosters", layer="raw", seasons=(PROJ_SEASON,))
    raw = raw.filter(pl.col("week") == pl.col("week").min())
    expected = set(
        raw.filter(pl.col("position").is_in(["QB", "RB", "FB", "WR", "TE"]))["gsis_id"]
        .drop_nulls().to_list()
    )
    assert set(ros["player_id"].to_list()) == expected
    # the chart lists plenty of players who have moved on; none of them may appear
    chart_only = set(depth.depth_chart(PROJ_SEASON, "latest")["player_id"]) - expected
    assert chart_only, "no chart-only players at all would mean the sources agree, which they do not"
    assert not chart_only & set(ros["player_id"])


def test_every_team_is_present_with_a_believable_offence(ros):
    assert ros["team"].n_unique() == 32
    per_team = ros.group_by("team").len()["len"]
    assert per_team.min() >= 20
    per_pos = ros.group_by(["team", "position"]).len()
    for pos, lo in (("QB", 2), ("RB", 3), ("WR", 5), ("TE", 3)):
        sub = per_pos.filter(pl.col("position") == pos)
        assert sub.height == 32, pos
        assert sub["len"].min() >= lo, pos


def test_depth_slots_are_contiguous_from_one_within_a_team_and_position(ros):
    counts = ros.group_by(["team", "position"]).agg(
        pl.col("depth_slot").min().alias("lo"),
        pl.col("depth_slot").max().alias("hi"),
        pl.col("depth_slot").n_unique().alias("distinct"),
        pl.len().alias("n"),
    )
    assert (counts["lo"] == 1).all()
    assert (counts["hi"] == counts["n"]).all()
    assert (counts["distinct"] == counts["n"]).all(), "two players cannot share a slot"


def test_a_player_the_chart_omits_ranks_behind_every_player_it_lists(ros):
    """Membership from the roster, order from the chart -- so an omission cannot outrank a starter."""
    per_group = ros.group_by(["team", "position"]).agg(
        pl.col("depth_slot").filter(pl.col("charted")).max().alias("last_charted"),
        pl.col("depth_slot").filter(~pl.col("charted")).min().alias("first_unlisted"),
    ).drop_nulls()
    assert per_group.height > 20, "expected plenty of teams with an unlisted player somewhere"
    assert (per_group["first_unlisted"] > per_group["last_charted"]).all()


def test_a_player_the_chart_puts_on_another_team_is_projected_where_he_is_rostered(ros):
    moved = ros.filter("team_disagreement")
    assert moved.height > 0, "August always has a few of these; none found means the flag is dead"
    chart = depth.depth_chart(PROJ_SEASON, "latest").select("player_id", pl.col("team").alias("old"))
    j = moved.join(chart, on="player_id", how="inner")
    assert j.height == moved.height
    assert (j["team"] != j["old"]).all()


def test_buckets_never_exceed_their_caps(ros):
    assert (ros["avail_slot"] <= roster.AVAIL_SLOT_CAP).all()
    caps = {**priors.SLOT_CAP}
    for pos, cap in caps.items():
        sub = ros.filter(pl.col("position") == pos)
        if sub.is_empty():
            continue
        assert sub["slot_bucket"].max() <= cap, pos
    assert (ros["slot_bucket"] <= ros["depth_slot"]).all()


# --------------------------------------------------------------------------- #
# availability: fitted blend x presence x status
# --------------------------------------------------------------------------- #
def test_the_blend_beats_both_of_the_things_it_blends(fit):
    assert fit["mae"] <= min(fit["mae_own_record"], fit["mae_slot_prior"]) + 1e-9
    assert fit["gain_vs_own_pct"] > 0 and fit["gain_vs_prior_pct"] > 0
    assert fit["k"] in roster.GAMES_K_GRID or fit["k"] == float("inf")
    assert fit["n"] > 3000


def test_the_slot_prior_and_the_presence_curve_both_fall_with_depth(slot_prior):
    for pos in slot_prior["position"].unique().to_list():
        sub = slot_prior.filter(pl.col("position") == pos).sort("avail_slot")
        for col in ("prior_games", "presence"):
            vals = sub[col].to_list()
            assert vals == sorted(vals, reverse=True), f"{pos}/{col}: {vals}"
        assert sub["presence"].max() <= 1.0 + 1e-9
        assert sub["presence"].min() >= 0.0
        assert sub["prior_games"].max() <= float(REG_WEEKS - 1)


def test_presence_is_one_where_every_team_fills_the_job_and_small_where_few_do(slot_prior):
    starters = slot_prior.filter(pl.col("avail_slot") == 1)
    assert starters.height >= 4
    assert (starters["presence"] > 0.99).all()
    # and the bottom of the chart must be genuinely unlikely, or the cut is not being modelled
    deep = slot_prior.filter(pl.col("avail_slot") >= 10)
    assert deep.height > 0
    assert deep["presence"].max() < 0.4


def test_expected_games_is_the_product_of_the_three_factors_and_stays_in_range(part):
    p = part.with_columns(
        pl.min_horizontal(
            pl.col("games_if_available") * pl.col("presence") * pl.col("status_factor"),
            pl.lit(float(REG_WEEKS - 1)),
        ).alias("want")
    )
    assert (p["expected_games"] - p["want"]).abs().max() == pytest.approx(0.0, abs=1e-9)
    assert part["expected_games"].min() >= 0.0
    assert part["expected_games"].max() <= float(REG_WEEKS - 1)
    assert part["expected_games"].null_count() == 0
    assert 0.0 <= part["active_weeks"].min() and part["active_weeks"].max() <= 1.0


def test_expected_games_falls_as_the_roster_deepens(part):
    by_slot = (
        part.filter(pl.col("position") == "WR")
        .group_by("avail_slot").agg(pl.col("expected_games").mean().alias("g"))
        .sort("avail_slot")
    )
    top, bottom = by_slot["g"][0], by_slot["g"][-1]
    assert top > 10.0, "a WR1 should be projected for most of a season"
    assert bottom < 2.0, "the bottom of a 90-man roster should be projected for almost nothing"
    # broadly monotone: allow one local inversion from a veteran buried on a chart
    drops = sum(1 for a, b in zip(by_slot["g"][:-1], by_slot["g"][1:], strict=True) if b > a)
    assert drops <= 1, by_slot


def test_a_retired_player_is_projected_for_no_games_at_all(part):
    zeros = {s for s, f in Settings().status_availability.items() if f == 0.0}
    out = part.filter(pl.col("status").is_in(list(zeros)))
    if out.is_empty():
        pytest.skip("no non-participating statuses on the current roster")
    assert (out["expected_games"] == 0.0).all()
    assert (out["status_factor"] == 0.0).all()
    # and he is still on the board, because a user needs to see that he is gone
    assert out.height == part.filter(pl.col("status").is_in(list(zeros))).height


def test_moving_a_status_multiplier_moves_only_that_status(part):
    s = Settings()
    harsher = Settings(status_availability={**s.status_availability, "RES": 0.0})
    alt = roster.participation(PROJ_SEASON, harsher)
    j = part.join(alt, on="player_id", suffix="_alt")
    res = j.filter(pl.col("status") == "RES")
    if res.is_empty():
        pytest.skip("nobody on injured reserve")
    assert (res["expected_games_alt"] == 0.0).all()
    rest = j.filter(pl.col("status") != "RES")
    assert (rest["expected_games"] - rest["expected_games_alt"]).abs().max() == pytest.approx(0.0)


def test_the_availability_fit_never_reads_the_season_it_projects():
    """Corrupt the target season's games and neither the prior nor the own record may move."""
    seasons = tuple(range(2016, 2026))
    panel = roster._avail_panel(seasons)
    target = 2025
    poisoned = panel.with_columns(
        pl.when(pl.col("season") == target).then(pl.lit(99.0)).otherwise(pl.col("games")).alias("games")
    )
    clean_prior = roster.slot_games_prior(panel, before=target)
    dirty_prior = roster.slot_games_prior(poisoned, before=target)
    j = clean_prior.join(dirty_prior, on=["position", "avail_slot"], suffix="_p")
    assert j.height > 20
    assert (j["prior_games"] - j["prior_games_p"]).abs().max() == pytest.approx(0.0)

    hist = panel.group_by(["season", "player_id"]).agg(pl.col("games").max())
    bad = poisoned.group_by(["season", "player_id"]).agg(pl.col("games").max())
    settings = Settings()
    a = roster._own_games(hist, target, settings)
    b = roster._own_games(bad, target, settings)
    k = a.join(b, on="player_id", suffix="_p")
    assert k.height > 500
    assert (k["obs_games"] - k["obs_games_p"]).abs().max() == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# participation: the player, his job, and the blend of the two
# --------------------------------------------------------------------------- #
def test_every_participation_rate_is_a_share_of_something(part):
    for m in roster.PARTICIPATION_METRICS:
        if m not in part.columns:
            continue
        v = part[m].drop_nulls()
        assert v.min() >= 0.0, m
        assert v.max() <= 1.0, m


def test_the_blend_lies_between_the_player_and_his_job_and_weights_by_evidence(detail):
    d = detail.drop_nulls(["obs", "prior", "used"])
    lo, hi = pl.min_horizontal("obs", "prior"), pl.max_horizontal("obs", "prior")
    off = d.with_columns((pl.col("used") < lo - 1e-9).alias("under"),
                         (pl.col("used") > hi + 1e-9).alias("over"))
    assert off["under"].sum() == 0
    assert off["over"].sum() == 0
    w = d.with_columns((pl.col("n") / (pl.col("n") + pl.col("k"))).alias("want"))
    assert (w["own_weight"] - w["want"]).abs().max() == pytest.approx(0.0, abs=1e-9)


def test_a_player_with_no_history_is_exactly_his_job(detail):
    fresh = detail.filter(pl.col("obs").is_null() | (pl.col("n").fill_null(0.0) <= 0))
    assert fresh.height > 100, "a 90-man roster is full of players with no history"
    assert (fresh["own_weight"] == 0.0).all()
    assert (fresh["used"] - fresh["prior"]).abs().max() == pytest.approx(0.0)
    assert fresh["used"].null_count() == 0, "everybody gets a number; that is the point of the prior"


def test_only_rookies_with_a_curve_are_priced_off_draft_capital(detail):
    rook = detail.filter(pl.col("source") == "draft_blend")
    assert rook.height > 0
    assert rook["is_rookie"].all()
    assert rook["curve"].null_count() == 0
    assert rook["draft_pick"].null_count() == 0, "undrafted rookies have no capital to price"
    # a veteran's own play has already priced in whatever the draft said about him
    vets = detail.filter(~pl.col("is_rookie"))
    assert (vets["source"] != "draft_blend").all()


def test_the_rookie_form_uses_the_depth_slot_as_well_as_the_pick(detail):
    """The whole point of `capital_ratio`: two rookies with one pick bin and different jobs differ."""
    blend = priors.load_rookie_blend()
    ratio_metrics = [
        m for m in roster.PARTICIPATION_METRICS
        if blend.get(m, ("additive", 0.0))[0] == "capital_ratio" and blend[m][1] > 0.0
    ]
    if not ratio_metrics:
        pytest.skip("no participation metric fitted to the capital-ratio form")
    m = ratio_metrics[0]
    r = detail.filter((pl.col("metric") == m) & (pl.col("source") == "draft_blend"))
    spread = r.group_by(["position", "curve"]).agg(
        pl.col("prior").n_unique().alias("priors"), pl.len().alias("n")
    ).filter(pl.col("n") > 1)
    assert spread.height > 0
    assert spread["priors"].max() > 1, "same pick bin, different slots, identical prior"


def test_an_earlier_pick_is_never_priced_below_a_later_one_at_the_same_slot(detail):
    r = detail.filter((pl.col("metric") == "snap_share") & (pl.col("source") == "draft_blend"))
    if r.is_empty():
        pytest.skip("no drafted rookies priced off the curve")
    for (pos, bucket), sub in r.group_by(["position", "slot_bucket"]):
        s = sub.sort("draft_pick")
        vals = s["prior"].to_list()
        assert vals == sorted(vals, reverse=True), f"{pos}/{bucket}: {list(zip(s['draft_pick'], vals))}"


def test_a_weekly_contribution_is_the_product_and_so_below_both_factors(part):
    for m in roster.PARTICIPATION_METRICS:
        col = f"weekly_{m}"
        if col not in part.columns:
            continue
        p = part.with_columns((pl.col(m).fill_null(0.0) * pl.col("active_weeks")).alias("want"))
        assert (p[col] - p["want"]).abs().max() == pytest.approx(0.0, abs=1e-12)
        assert (p[col] <= p[m].fill_null(0.0) + 1e-9).all(), m
        assert (p[col] <= p["active_weeks"] + 1e-9).all(), m


# --------------------------------------------------------------------------- #
# the identity: five skill players on the field, every snap
# --------------------------------------------------------------------------- #
def test_the_measured_benchmark_is_the_five_man_identity_net_of_denominators():
    bm = roster.measure_benchmark()
    assert 4.5 < bm["snap_share"] < 5.0
    assert 4.5 < bm["route_participation"] < 5.2
    assert 0.9 < bm["rush_participation"] < 1.4      # one designed rush has one carrier
    assert 0.9 < bm["dropback_share"] < 1.1          # one quarterback drops back


def test_projected_team_sums_return_to_the_measured_identity(part, fit):
    """Nothing here is tuned to make this true, which is the only reason it is worth checking."""
    diag = roster.team_diagnostic(part, fit["benchmark"])
    assert diag.height == len([c for c in part.columns if c.startswith("weekly_")])
    for row in diag.iter_rows(named=True):
        assert row["benchmark"] is not None, row["metric"]
        assert abs(row["gap_pct"]) < 5.0, row
        # and no single team may be wildly off, even where the mean is right
        assert row["projected_min"] > 0.5 * row["benchmark"], row
        assert row["projected_max"] < 1.5 * row["benchmark"], row


def test_the_fitted_population_is_smaller_than_a_ninety_man_roster(fit, part):
    """The fact that makes `presence` necessary; if it ever stops being true, drop the factor."""
    assert fit["population_per_team"] < part.height / 32
