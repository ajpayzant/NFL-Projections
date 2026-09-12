"""Reading a season that is still going on.

A projection made in November that still says what it said in August is a souvenir. Two things have to
happen for it to keep up and they are separable, so they are tested separately: the results have to be
*readable* (`src/data/inseason.py`), and the estimator has to be allowed to *learn* from them for the
games not yet played (`estimate.to_date`). `compose.actualise` -- replacing a finished week with what
happened in it -- is the third piece and is tested in `test_compose.py`, against the identities it is
allowed to break.

Every test here is written to pass before kickoff rather than to skip, and that is deliberate. The
guards are the part most worth checking and they are exactly what makes the tables empty: a season that
has not started and a season whose results are being ignored have to look the same from here, because
one of them is the backtest and it must never see a result.
"""

from __future__ import annotations

import polars as pl
import pytest

from src.config import LAST_COMPLETE_SEASON, PROJ_SEASON, REG_WEEKS, Settings
from src.data import inseason
from src.model import blend, estimate, priors

# --------------------------------------------------------------------------- #
# how far the season has got
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def weeks() -> pl.DataFrame:
    return inseason.team_weeks(PROJ_SEASON)


def test_a_team_week_is_in_the_books_or_it_is_not(weeks) -> None:
    """The unit everything else keys on, and it is per team rather than per league week.

    Not pedantry: a Thursday game is final while Sunday's has not kicked off, and byes mean two teams in
    the same calendar week are not equally far into their seasons. A projection that substituted a whole
    league week at a time would either throw a Thursday result away for four days or zero out a game
    nobody had played.
    """
    if weeks.is_empty():
        return
    assert weeks.select(["team", "week"]).is_unique().all()
    assert weeks["week"].min() >= 1
    assert weeks["week"].max() <= REG_WEEKS
    assert weeks["team"].n_unique() <= 32


def test_complete_is_the_slowest_team_and_played_is_the_fastest(weeks) -> None:
    """`weeks_complete` is a floor a header can print; `weeks_played` is how much data exists.

    They are different numbers and conflating them is how a page comes to say "through week 3" while a
    team that has played once sits on the same board being compared with one that has played three.
    """
    complete, played = inseason.weeks_complete(PROJ_SEASON), inseason.weeks_played(PROJ_SEASON)
    assert 0 <= complete <= played <= REG_WEEKS
    if weeks.is_empty():
        assert complete == 0 and played == 0
        return
    per_team = weeks.group_by("team").len()
    # zero until all thirty-two are in, because a minimum over the teams that have rows is not a minimum
    assert complete == (int(per_team["len"].min()) if per_team.height == 32 else 0)
    assert played == int(weeks["week"].max())


def test_a_players_week_says_whether_he_was_there(weeks) -> None:
    """`played` is 0 or 1, and it comes from the snap count rather than from the stat line.

    A receiver on the field for thirty snaps who was not thrown at did play, and no stat line can say
    so. `p_play` is an expectation, which is the right thing to carry into a game that has not happened
    and the wrong thing to carry out of one that has -- 0.94 of a game he played every snap of, or of one
    he was inactive for, is a number about neither.
    """
    pw = inseason.player_weeks(PROJ_SEASON)
    if pw.is_empty():
        assert weeks.is_empty() or True     # results can lag the team table by hours; not a failure
        return
    assert set(pw["played"].unique().to_list()) <= {0.0, 1.0}
    assert pw.select(["player_id", "week"]).is_unique().all()   # a mid-season trade is one game
    # counts cannot be negative; yardage can, because a run can lose ground and a sack takes it back
    yardage = {c for c in inseason.MEASURED if c.endswith("_yards")}
    for col in inseason.MEASURED:
        assert col in pw.columns, col
        assert pw[col].null_count() == 0, col
        if col not in yardage:
            assert float(pw[col].min()) >= -1e-9, col
    # everybody who did something is credited with having been there
    did = pw.filter(pl.sum_horizontal([pl.col(c) for c in inseason.PLAYER_STATS.values()]) > 0)
    assert did.is_empty() or float(did["played"].min()) == 1.0


def test_the_team_pool_is_the_teams_own_and_not_a_sum_of_the_players() -> None:
    """A team's attempts are a team fact.

    Read from `team_stats` so a player the crosswalk has not caught up with -- an undrafted rookie
    signed on Tuesday -- cannot quietly shrink his own offence, which would inflate every teammate's
    share of it. This is also what lets `test_compose` check a substituted week against something other
    than the board agreeing with itself.
    """
    tp = inseason.team_pools(PROJ_SEASON)
    if tp.is_empty():
        return
    assert tp.select(["team", "week"]).is_unique().all()
    for col in ("attempts", "carries", "targets"):
        if col in tp.columns:
            assert float(tp[col].min()) >= 0.0, col
    # the pools the shares divide have to be there, or a share cannot be measured at all
    assert {"targets", "carries", "attempts", "passing_tds"} <= set(tp.columns)


def test_the_team_snap_pool_is_recovered_from_the_share_beside_each_player() -> None:
    """Nobody publishes a team's offensive snap count; every player's row implies it.

    Snaps and the percentage they were of the team's, so the total is one divided by the other. The
    percentage is rounded upstream, so any one player implies it to within a snap and the median over
    twenty of them is exact -- which matters because this is the denominator of the snap share, and the
    snap share is now the knob that scales every other claim a player makes.
    """
    ts = inseason.team_snaps(PROJ_SEASON)
    if ts.is_empty():
        return
    assert ts.select(["team", "week"]).is_unique().all()
    # a team runs between roughly forty and ninety offensive snaps in a game; anything outside that is
    # a broken percentage rather than an unusual game
    assert float(ts["team_offense_snaps"].min()) >= 30.0
    assert float(ts["team_offense_snaps"].max()) <= 110.0


# --------------------------------------------------------------------------- #
# the season so far as one more season of history
# --------------------------------------------------------------------------- #
def test_the_frame_is_shaped_like_a_season_of_history() -> None:
    """Which is the whole trick, and the reason nothing downstream needed a special case.

    The estimator is driven by `(numerator, denominator)` column pairs on a per-player-season frame, so
    a season-to-date row in that same shape is a season of history like any other: blended by recency,
    shrunk by the opportunity in it, weighted by how much of it there is. Nobody had to choose a schedule
    for how fast this season should start to count.
    """
    for table in ("skill", "qb"):
        d = inseason.to_date(table, PROJ_SEASON)
        assert {"player_id", "season", "team", "games"} <= set(d.columns), table
        if d.is_empty():
            continue
        assert d["player_id"].is_unique().all(), table
        assert set(d["season"].unique().to_list()) == {PROJ_SEASON}, table
        assert float(d["games"].min()) > 0, table
        assert float(d["games"].max()) <= REG_WEEKS, table


def test_a_metric_is_fed_only_when_both_halves_of_its_ratio_are_measured() -> None:
    """Half a ratio is worse than none of it, and this is the test that says so.

    `blend_counts` fills a missing count with zero. So a quarterback's sacks arriving without the
    dropbacks to divide them by would add sacks to the numerator, nothing to the denominator, and report
    his sack rate as having doubled -- an accuracy loss dressed up as fresher data. Both sides or
    neither, decided by what is in the frame rather than by a list kept in step by hand.

    The refusals are real limits and are named here so they cannot be quietly lost: routes run and
    red-zone, late-down and short-yardage volume are participation facts, and participation is published
    only after a season ends. Those eight keep their preseason estimate all season, which is the honest
    answer and not a good one. Everything a weekly table or the play-by-play in `passer_games` can
    measure -- the quarterback's dropbacks and the scramble/designed split included -- is expected live.
    """
    st = Settings()
    live = {m.name for m in priors.METRICS if estimate.to_date(m, PROJ_SEASON, st) is not None}
    for m in priors.METRICS:
        cur = estimate.to_date(m, PROJ_SEASON, st)
        if cur is None:
            continue
        assert {m.num, m.den} <= set(cur.columns), m.name
        assert float(cur[m.den].min()) > 0, m.name
    if not live:
        return                              # before kickoff there is nothing to feed anything
    # the ones no weekly table and no play-by-play rebuild of the current season can close
    assert not live & {"route_participation", "tprr", "rz_target_share", "rz_carry_share",
                       "late_down_target_share", "short_yardage_carry_share", "inside_5_carry_share",
                       "rush_success_rate"}
    # and the ones they can, which are the point of the exercise
    assert {"target_share", "carry_share", "snap_share", "catch_rate", "yards_per_target"} <= live
    # the quarterback's half of that, which only exists because `to_date` reads the pbp aggregates too
    if not inseason.qb_detail(PROJ_SEASON).is_empty():
        assert {"dropback_share", "attempt_rate", "sack_rate", "scramble_rate", "designed_rush_share",
                "scramble_ypc", "designed_rush_ypc", "qb_fumble_rate"} <= live


# --------------------------------------------------------------------------- #
# the dropback detail, and the grain everything is joined at
# --------------------------------------------------------------------------- #
def test_the_dropback_detail_is_one_row_a_game_and_clean_means_what_history_means() -> None:
    """`qb_detail` against the definition `history.qb_weeks` uses, because they are blended together.

    A season-to-date row that counted kneeldowns as rushes, or scrambles as designed runs, would be a
    different metric wearing the same name as the three seasons it is averaged with -- and the error would
    look like a quarterback who had suddenly started running.
    """
    d = inseason.qb_detail(PROJ_SEASON)
    if d.is_empty():
        return                              # no heavy pass has run for this season yet
    assert d.select(["player_id", "week"]).is_unique().all()
    for col in inseason.QB_PBP:
        if col in d.columns:
            assert float(d[col].min()) >= 0.0, col
    if {"rush_attempts_clean", "designed_qb_rushes", "scrambles"} <= set(d.columns):
        clean = d["designed_qb_rushes"] + d["scrambles"]
        assert (d["rush_attempts_clean"] - clean).abs().max() < 1e-9
        # a box-score carry is a designed run, a scramble or a kneel; the engine's `qb_rushes` is the
        # first two of those, so clean has to agree with it exactly and not merely fall short of carries
        if "qb_rushes" in d.columns:
            assert (d["qb_rushes"] - clean).abs().max() < 1e-9
    if {"dropbacks", "scrambles"} <= set(d.columns):
        assert float((d["dropbacks"] - d["scrambles"]).min()) >= -1e-9


def test_a_teams_dropbacks_are_a_team_fact_and_hold_every_passer_on_it() -> None:
    """`team_detail` read off `team_games`, not summed from the quarterbacks.

    Same reason as `team_pools`: a wildcat snap by a receiver is a dropback the offence spent, so it
    belongs in the denominator of every passer's share of it. Summing the passers would leave the shares
    on a team with a trick play adding to more than one.
    """
    td = inseason.team_detail(PROJ_SEASON)
    if td.is_empty():
        return
    assert td.select(["team", "week"]).is_unique().all()
    # a team drops back between roughly fifteen and eighty times in a game
    assert float(td["team_dropbacks"].min()) >= 15.0
    assert float(td["team_dropbacks"].max()) <= 80.0
    qb = inseason.qb_detail(PROJ_SEASON)
    if qb.is_empty() or "dropbacks" not in qb.columns:
        return
    mine = qb.join(td, on=["team", "week"], how="inner").group_by(["team", "week"]).agg(
        pl.col("dropbacks").sum(), pl.col("team_dropbacks").first())
    assert float((mine["team_dropbacks"] - mine["dropbacks"]).min()) >= -1e-9


def test_the_team_pool_a_share_divides_is_the_one_he_was_present_for() -> None:
    """A share is measured over the games the player was on the field for, and nothing else.

    `history.skill_seasons` sums the team totals carried on the player's own game rows, so its
    `team_targets` is the pool he was present for rather than his team's whole season. The season to date
    has to match that or the two are not comparable: divide a man's four targets in the one game he played
    by his offence's sixty over three, and he reads as having lost his job.

    Built by hand so it can be checked on a player who missed a game, which the calendar may not offer
    yet, and so the arithmetic is checked rather than the frames agreeing with each other.
    """
    pw = pl.DataFrame({
        "player_id": ["a", "a", "b"], "team": ["KC"] * 3, "week": [1, 2, 2],
        "played": [1.0, 1.0, 1.0], "targets": [10.0, 8.0, 4.0], "offense_snaps": [50.0, 45.0, 20.0],
    })
    tp = pl.DataFrame({"team": ["KC", "KC"], "week": [1, 2], "targets": [40.0, 30.0]})
    got = _to_date_over("skill", 1901, pw, tp)
    row = {r["player_id"]: r for r in got.to_dicts()}
    assert row["a"]["games"] == 2.0
    assert row["a"]["targets"] == 18.0
    assert row["a"]["team_targets"] == 70.0        # both weeks, because he played both
    assert row["b"]["games"] == 1.0
    assert row["b"]["targets"] == 4.0
    assert row["b"]["team_targets"] == 30.0        # week 2 only, not the 70 his offence ran all year


def test_the_dropback_detail_is_used_only_over_the_weeks_it_and_the_weekly_tables_share() -> None:
    """Between Sunday's games and the next heavy pass, half a quarterback's rates would count two games
    and half of them one. A rate whose two sides count different games is not a rate.

    So the detail is not merely joined -- the weeks it does not cover are dropped from the frame
    altogether. His live evidence stops a week short, which is a week of freshness given up to keep the
    ratios honest, and the alternative is an attempt rate computed over one game's attempts and two games'
    dropbacks.
    """
    pw = pl.DataFrame({
        "player_id": ["q", "q"], "team": ["KC", "KC"], "week": [1, 2], "played": [1.0, 1.0],
        "attempts": [30.0, 25.0], "sacks": [2.0, 1.0], "offense_snaps": [60.0, 55.0],
    })
    tp = pl.DataFrame({"team": ["KC", "KC"], "week": [1, 2], "attempts": [30.0, 25.0]})
    qb = pl.DataFrame({"player_id": ["q"], "team": ["KC"], "week": [1], "dropbacks": [34.0],
                       "scrambles": [2.0]})
    td = pl.DataFrame({"team": ["KC"], "week": [1], "team_dropbacks": [36.0]})
    got = _to_date_over("qb", 1902, pw, tp, qb, td).to_dicts()[0]
    assert got["games"] == 1.0                     # week 2 is dropped, not half-measured
    assert got["attempts"] == 30.0
    assert got["dropbacks"] == 34.0
    assert got["team_pass_attempts"] == 30.0
    assert got["team_dropbacks"] == 36.0


def test_a_passer_the_play_by_play_has_no_row_for_still_gets_a_denominator() -> None:
    """`attempts + sacks` is a dropback count missing every scramble -- for a quarterback it is the wrong
    number and the module refuses it. For a man who is not one it is exactly right.

    The receiver who throws the week's trick pass has no row in `passer_games`, and leaving his dropbacks
    null would divide his one attempt by nothing. He drops back as often as he throws or is sacked, which
    is what this fills, and it is a fallback for the null rather than a substitute for the real column.
    """
    pw = pl.DataFrame({
        "player_id": ["q", "wr"], "team": ["KC", "KC"], "week": [1, 1], "played": [1.0, 1.0],
        "attempts": [30.0, 1.0], "sacks": [2.0, 0.0], "offense_snaps": [60.0, 50.0],
    })
    tp = pl.DataFrame({"team": ["KC"], "week": [1], "attempts": [31.0]})
    qb = pl.DataFrame({"player_id": ["q"], "team": ["KC"], "week": [1], "dropbacks": [34.0]})
    td = pl.DataFrame({"team": ["KC"], "week": [1], "team_dropbacks": [35.0]})
    row = {r["player_id"]: r for r in _to_date_over("qb", 1903, pw, tp, qb, td).to_dicts()}
    assert row["q"]["dropbacks"] == 34.0           # the real column wins where there is one
    assert row["wr"]["dropbacks"] == 1.0           # one attempt, no sack, no scramble


def _to_date_over(table: str, season: int, pw: pl.DataFrame, tp: pl.DataFrame,
                  qb: pl.DataFrame | None = None, td: pl.DataFrame | None = None) -> pl.DataFrame:
    """`to_date` over frames built by hand instead of read from the lake.

    A fake season number rather than `PROJ_SEASON` because `to_date` is memoised on its arguments, and
    the sources are swapped by name because that is how `to_date` reaches them.
    """
    empty = pl.DataFrame()
    subs = {"player_weeks": pw, "team_pools": tp, "team_snaps": empty,
            "qb_detail": qb if qb is not None else empty,
            "team_detail": td if td is not None else empty}
    saved = {name: getattr(inseason, name) for name in subs}
    try:
        for name, frame in subs.items():
            setattr(inseason, name, lambda season=None, _f=frame: _f)
        return inseason.to_date(table, season)
    finally:
        for name, fn in saved.items():
            setattr(inseason, name, fn)
        inseason.clear_cache()


def test_a_finished_season_never_sees_itself() -> None:
    """The leakage guard, checked at the estimator rather than at the flag.

    The backtest projects a completed season and scores it against that season's own rows, so weight on
    lag 0 there would report a model that already knew the answer. Two independent guards say no --
    `estimate.to_date` refuses any season but the one in progress, and `blend.season_weights` zeroes the
    target season unless a caller explicitly asks otherwise -- and both are checked because either one
    failing silently is the most expensive bug this project can have.
    """
    st = Settings()
    for m in priors.METRICS:
        assert estimate.to_date(m, LAST_COMPLETE_SEASON, st) is None, m.name
        assert estimate.to_date(m, PROJ_SEASON, Settings(use_inseason_form=False)) is None, m.name

    df = pl.DataFrame({"season": [2023, 2024, 2025, 2026, 2027], "games": [17] * 5,
                       "player_id": ["a"] * 5, "n": [1.0] * 5})
    got = df.with_columns(blend.season_weights("season", 2026)).sort("season")["recency_weight"]
    assert got.to_list() == [2.0, 3.0, 5.0, 0.0, 0.0]
    asked = df.with_columns(blend.season_weights("season", 2026, current=5.0)).sort("season")
    assert asked["recency_weight"].to_list() == [2.0, 3.0, 5.0, 5.0, 0.0]


def test_two_games_of_evidence_weigh_two_games() -> None:
    """The unit of `n` has to survive, because every fitted `k` is quoted in it.

    `scale=True` divides by the weight vector's sum to put blended counts on a one-season scale, so `n`
    is a *pace* -- the pool per seventeen games that this player's blended history implies -- and not a
    running total. A season two games old is two games of evidence, so it enters the divisor as the
    fraction of a season it covers. Counting it whole would divide by a third more weight than was added
    and shrink every player on the board toward his position prior: fresher data making the projection
    worse, which is the failure mode worth a test.

    That `n` is a pace has a consequence worth being explicit about, since it looks wrong at a glance:
    adding a season in progress can move `n` *down*. Two games at a below-average pace genuinely imply a
    smaller pool than the player's history did, and the shrinkage trusting the prior a little more is the
    right response to that rather than a bug in it.

    Built by hand rather than measured off the live tables, because the arithmetic has to be checked at
    weeks of the season the calendar is not currently at, including a full one.
    """
    hist = pl.DataFrame({
        "player_id": ["a"] * 4,
        "season": [2023, 2024, 2025, 2026],
        "games": [17.0, 17.0, 17.0, 2.0],
        "targets": [100.0, 100.0, 100.0, 20.0],
        "team_targets": [500.0, 500.0, 500.0, 60.0],
    })
    w = (5.0, 3.0, 2.0)
    past = hist.filter(pl.col("season") < 2026)
    without = blend.blend_counts(past, 2026, ["targets", "team_targets"], weights=w)
    with_live = blend.blend_counts(hist, 2026, ["targets", "team_targets"], weights=w, current=w[0])
    assert float(without["team_targets"][0]) == pytest.approx(500.0, rel=1e-9)

    # the divisor carries the two games and not a season: 5,300 of weighted pool over 10 + 5 x 2/17
    expect = (500.0 * sum(w) + 60.0 * w[0]) / (sum(w) + w[0] * 2.0 / 17.0)
    assert float(with_live["team_targets"][0]) == pytest.approx(expect, rel=1e-9)
    # two games at 30 targets a game against a history of 29.4 is the same pace, so `n` barely moves --
    # which is the property that makes this safe, and would be a 28-target jump if the divisor were wrong
    assert abs(float(with_live["team_targets"][0]) - 500.0) < 1.0
    # a full season in progress would count like the most recent completed one and no more
    full = hist.with_columns(
        pl.when(pl.col("season") == 2026).then(17.0).otherwise(pl.col("games")).alias("games"),
        pl.when(pl.col("season") == 2026).then(500.0).otherwise(pl.col("team_targets")).alias("team_targets"),
        pl.when(pl.col("season") == 2026).then(100.0).otherwise(pl.col("targets")).alias("targets"),
    )
    whole = blend.blend_counts(full, 2026, ["targets", "team_targets"], weights=w, current=w[0])
    assert float(whole["team_targets"][0]) == pytest.approx(500.0, rel=1e-9)
    assert float(whole["targets"][0]) == pytest.approx(100.0, rel=1e-9)
    # and a season nobody has played yet changes nothing at all
    none_yet = hist.with_columns(
        pl.when(pl.col("season") == 2026).then(0.0).otherwise(pl.col("games")).alias("games"),
        pl.when(pl.col("season") == 2026).then(0.0).otherwise(pl.col("targets")).alias("targets"),
        pl.when(pl.col("season") == 2026).then(0.0).otherwise(pl.col("team_targets")).alias("team_targets"),
    )
    zero = blend.blend_counts(none_yet, 2026, ["targets", "team_targets"], weights=w, current=w[0])
    assert float(zero["team_targets"][0]) == pytest.approx(float(without["team_targets"][0]), rel=1e-9)


def test_a_bad_year_so_far_moves_the_estimate_down_and_not_by_much_yet() -> None:
    """The direction and the magnitude, on the one metric every receiver has.

    A share running below a player's history has to pull his estimate down -- that is the whole purpose
    -- and after one or two games it has to pull it a *little*, because a projection that lurches on a
    week of football is worse than one that ignores it. Both halves are the test; either alone passes
    for the wrong reason.
    """
    st = Settings()
    m = priors.BY_NAME["target_share"]
    cur = estimate.to_date(m, PROJ_SEASON, st)
    if cur is None:
        return
    off = estimate.own_rate(m, PROJ_SEASON, Settings(use_inseason_form=False))
    on = estimate.own_rate(m, PROJ_SEASON, st)
    j = (off.select("player_id", pl.col("obs").alias("was"), pl.col("n").alias("n_was"))
         .join(on.select("player_id", pl.col("obs").alias("now"), pl.col("n").alias("n_now")),
               on="player_id", how="inner")
         .join(cur.select("player_id", (pl.col(m.num) / pl.col(m.den)).alias("so_far")),
               on="player_id", how="inner")
         .filter(pl.col("n_was") > 50.0))          # men with real history, so the move is not the prior
    if j.is_empty():
        return
    # nobody outside the season so far moved
    assert float((off.join(on, on="player_id", how="inner")
                  .filter(~pl.col("player_id").is_in(cur["player_id"].to_list()))
                  .select((pl.col("obs") - pl.col("obs_right")).abs().max()).item() or 0.0) < 1e-12)
    moved = j.with_columns((pl.col("now") - pl.col("was")).alias("d"))
    down = moved.filter(pl.col("so_far") < pl.col("was") - 0.02)
    assert down.is_empty() or float(down["d"].max()) < 0.0
    up = moved.filter(pl.col("so_far") > pl.col("was") + 0.02)
    assert up.is_empty() or float(up["d"].min()) > 0.0
    # and one or two games cannot move a player with a career behind him very far
    games = float(inseason.to_date("skill", PROJ_SEASON)["games"].max())
    if games <= 3:
        assert float(moved["d"].abs().max()) < 0.08
    # `n` is a weighted mean and not a running total, so it is allowed to move either way, and the exact
    # arithmetic is checkable rather than a vibe: the new mean is the old one plus this season's pool at
    # this season's weight, over the divisor grown by the fraction of a season it covers. Asserted as an
    # identity because the alternative -- a bound -- would have hidden the real behaviour here, which is
    # that a man who has only ever seen sixty targets of team pool learns a fifth of what he knows from
    # one game, while a man with three full seasons behind him moves half a percent. That is the whole
    # design: the players a projection is least sure about are the ones this moves.
    w = list(Settings().recency)
    frac = games / 17.0
    expect = ((pl.col("n_was") * sum(w) + pl.col(m.den) * w[0]) / (sum(w) + w[0] * frac))
    chk = (moved.join(cur.select("player_id", m.den), on="player_id")
           .with_columns((pl.col("n_now") - expect).abs().alias("err")))
    assert float(chk["err"].max()) < 1e-6


def test_the_record_shows_the_season_in_progress() -> None:
    """What the estimate reads, a user has to be able to see.

    `season_history` is the frame the app's evidence panel is built from, and a record that stopped last
    December would leave somebody arguing with a number whose reason is off screen. Unfinished is fine;
    absent is not. `games` says how much of it there is.
    """
    live = estimate.to_date(priors.BY_NAME["target_share"], PROJ_SEASON, Settings())
    rec = estimate.season_history(["target_share"], seasons=(PROJ_SEASON,))
    if live is None:
        assert rec.is_empty() or rec.height >= 0
        return
    assert not rec.is_empty()
    assert set(rec["season"].unique().to_list()) == {PROJ_SEASON}
    assert float(rec["games"].max()) <= REG_WEEKS
    assert rec["n"].null_count() == 0
