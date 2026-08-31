"""What else the season could do: correlated Monte Carlo over the projection the engine just made.

Every number up to here is an expectation, and an expectation is the least interesting thing about a
season. Two backs projected for 250 points are not the same asset if one of them gets there on 260
carries and the other on a 14% chance of 340 and a 30% chance of missing six games. This module is the
distribution around the point estimate, and it is built out of the same frame the board is: the weekly
composition, edits included, so a share moved on the team page moves the range with it.

**Three correlated levels, in the order the projection was built.**

1. **The game.** One shock per game, shared by the two teams. Its sign is measured, not assumed, and
   the measurement is the interesting part: volume is *anti*-correlated within a game (plays -0.46,
   carries -0.53) because there is one clock and the team that leads runs it out, while touchdowns are
   mildly positive (+0.14). The shootout that lifts both teams' receivers is real for scoring and false
   for volume, so one correlation per channel rather than one for the game.
2. **The team.** A persistent component -- our season estimate of this offence can be wrong all year --
   and a transient one for the week. Both from history: a team's weekly volume moves 13-28% around its
   own season mean depending on the channel.
3. **The player.** His share of the pool and his efficiency, each with a persistent and a transient
   part. Shares are renormalised inside the team on every draw, so a share is still taken from a
   teammate rather than invented, and a backup inherits the starter's targets on the draws where the
   starter is hurt.

**Availability is drawn, not assumed.** The season-long injury multiplier is resampled from the
measured distribution of actual games over projected games -- centred to mean 1 per position, so the
sim does not quietly re-level the projection, with the measured bias reported instead.

**Counts are Poisson, rates are lognormal.** A touchdown is not a small continuous quantity: team
touchdowns come back at a coefficient of variation of 1.0 per game, which is Poisson to two decimals,
and a receiver's ceiling is mostly the week he catches three. Yardage per opportunity is measured as a
per-event dispersion and averaged over the week's events, so a twelve-target receiver's yards per
target is narrower than a two-target back's -- which is the difference between a safe projection and a
volatile one, and the reason a single blanket sigma cannot produce a useful volatility score.

**Then it is calibrated against out-of-sample coverage.** A structural simulation is always too
narrow, because it knows only the errors it was told about. `calibrate()` composes the held-out
2021-2025 seasons ex ante, runs this sim on each, and scales the persistent sigmas per position until a
5-95 interval really does contain 90% of outcomes. The scalars, the coverage before and after, and the
level bias the coverage check exposes are all in `dispersion.json`.

    python -m src.model.simulate --fit           # measure dispersion from history + the backtest
    python -m src.model.simulate --calibrate     # then scale it to hit out-of-sample coverage
    python -m src.model.simulate                 # ranges for 2026
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl

from src.config import (
    FITTED,
    HISTORY_FROM,
    LAST_COMPLETE_SEASON,
    PROJ_SEASON,
    Scoring,
    Settings,
    ensure_dirs,
)
from src.data import history, lake

DISPERSION_PATH = FITTED / "dispersion.json"

# Fixed, so two looks at the same board give the same floor. The sim is a measurement of the
# projection, not a lottery ticket, and a range that moves on every rerun cannot be argued with.
SEED = 20260823

POSITIONS = ("QB", "RB", "WR", "TE")
EPS = 1e-9

# Uncertainty is not the same shape for a franchise quarterback and for the third man in a camp
# competition, and pooling them is the difference between a range worth reading and a range worth
# nothing. Measured pooled, a quarterback's fifth percentile lands near fifty points, because the
# sample that produced it is four parts backup who never took the field. So every player-level sigma is
# measured, and looked up, within a tier: how far up his own position's board the projection put him.
#
#   starter   inside the startable count at his position -- the players a lineup is drawn from
#   depth     everybody else the projection named
TIERS = ("starter", "depth")
STARTERS = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}

# How many quantiles the availability sample is stored as. A nuisance dimension does not need a
# thousand atoms, and a fixed count is what lets `_avail_scale` solve for every player at once.
ATOMS = 128


# --------------------------------------------------------------------------- #
# what gets shocked
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Channel:
    """A volume pool a player takes a share of, and the history columns that measure its spread."""

    name: str
    count: str                 # the count column in the weekly frame
    team_metric: str           # per-game column in `history.team_seasons`, for the persistent part
    team_week: str             # per-game column in `history.team_weeks`, for the transient part
    share: tuple[str, str]     # (numerator, denominator) in the weekly player history


CHANNELS = (
    Channel("targets", "targets", "targets_per_game", "targets", ("targets", "team_targets")),
    Channel("carries", "carries", "rush_att_per_game", "carries", ("carries", "team_carries")),
    Channel("attempts", "attempts", "pass_att_per_game", "pass_attempts",
            ("attempts", "team_pass_attempts")),
)
BY_CHANNEL = {c.name: c for c in CHANNELS}


@dataclass(frozen=True)
class Group:
    """A block of fantasy points that moves together, and what moves it.

    `channel` is the volume it rides on; `count` says the outcome is a small integer and is drawn
    Poisson rather than scaled. `rate` is the (numerator, denominator) whose per-event dispersion is
    the transient part of a continuous group -- absent means the group has no measurable rate and gets
    the position's pooled figure.
    """

    name: str
    channel: str | None
    stats: tuple[str, ...]
    count: bool = False
    rate: tuple[str, str] | None = None


GROUPS = (
    Group("pass_yards", "attempts", ("passing_yards", "completions"),
          rate=("passing_yards", "attempts")),
    Group("pass_tds", "attempts", ("passing_tds",), count=True),
    Group("interceptions", "attempts", ("interceptions",), count=True),
    Group("rush_yards", "carries", ("rushing_yards",), rate=("rushing_yards", "carries")),
    Group("rush_tds", "carries", ("rushing_tds",), count=True),
    Group("receptions", "targets", ("receptions",), rate=("receptions", "targets")),
    Group("rec_yards", "targets", ("receiving_yards",), rate=("receiving_yards", "targets")),
    Group("rec_tds", "targets", ("receiving_tds",), count=True),
    # A fumble rides on nothing in particular -- a QB's is per dropback, a back's per touch -- and it is
    # worth two points, so it gets its own draw and no volume channel rather than a fourth pool.
    Group("fumbles", None, ("fumbles_lost",), count=True),
)

# Season totals reported with a full range for the players the caller asks about by name. Everybody
# gets a range on fantasy points; carrying nine of these for 900 players would be 300MB of draws for a
# table nobody reads at that width.
TRACKED = ("targets", "carries", "receptions", "receiving_yards", "receiving_tds", "rushing_yards",
           "rushing_tds", "attempts", "passing_yards", "passing_tds")

QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


# --------------------------------------------------------------------------- #
# dispersion
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Dispersion:
    """Every sigma the simulation needs, each one measured somewhere.

    Keys are strings because this round-trips through JSON: `"WR:targets"`, `"RB:rush_yards"`. The
    `_get` helpers fall back from the specific to the pooled value so a missing position or a new
    group degrades to the population figure rather than to zero.
    """

    team_week: dict[str, float] = field(default_factory=dict)     # channel -> log sd, week to week
    team_rho: dict[str, float] = field(default_factory=dict)      # channel -> within-game correlation
    team_season: dict[str, float] = field(default_factory=dict)   # channel -> log sd of the estimate
    usage_cv2: dict[str, float] = field(default_factory=dict)     # pos:channel -> squared CV, non-Poisson part
    event_cv2: dict[str, float] = field(default_factory=dict)     # pos:group -> per-event squared CV
    share_season: dict[str, float] = field(default_factory=dict)  # pos:tier:channel -> log sd, season share
    rate_season: dict[str, float] = field(default_factory=dict)   # pos:tier:group -> log sd, season rate
    games_ratio: dict[str, list[float]] = field(default_factory=dict)   # pos:tier -> centred sample
    boom: dict[str, float] = field(default_factory=dict)          # pos -> weekly points
    bust: dict[str, float] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)         # pos -> calibration multiplier
    meta: dict = field(default_factory=dict)

    # ---- lookups ---------------------------------------------------------- #
    def k_team_week(self, channel: str) -> float:
        return float(self.team_week.get(channel, 0.20))

    def k_team_rho(self, channel: str) -> float:
        return float(self.team_rho.get(channel, 0.0))

    def k_team_season(self, channel: str) -> float:
        return float(self.team_season.get(channel, 0.15))

    def k_usage(self, pos: str, channel: str) -> float:
        return float(self.usage_cv2.get(f"{pos}:{channel}",
                                        self.usage_cv2.get(f"ALL:{channel}", 0.25)))

    def k_event(self, pos: str, group: str) -> float:
        return float(self.event_cv2.get(f"{pos}:{group}",
                                        self.event_cv2.get(f"ALL:{group}", 1.0)))

    def k_share_season(self, pos: str, tier: str, channel: str) -> float:
        got = self.share_season.get(f"{pos}:{tier}:{channel}",
                                    self.share_season.get(f"ALL:{tier}:{channel}", 0.40))
        return self.scale.get(pos, 1.0) * float(got)

    def k_rate_season(self, pos: str, tier: str, group: str) -> float:
        got = self.rate_season.get(f"{pos}:{tier}:{group}",
                                   self.rate_season.get(f"ALL:{tier}:{group}", 0.15))
        return self.scale.get(pos, 1.0) * float(got)

    def games_sample(self, pos: str, tier: str = "starter") -> np.ndarray:
        """Always `ATOMS` long, so every player's sample is one row of one array.

        A stored sample that is a different length -- a hand-written one, or a file from before the
        quantile form -- is requantised rather than rejected.
        """
        got = (self.games_ratio.get(f"{pos}:{tier}") or self.games_ratio.get(f"ALL:{tier}")
               or self.games_ratio.get(pos) or [1.0])
        a = np.asarray(got, dtype=np.float32)
        if a.size == ATOMS:
            return a
        return np.quantile(a, np.linspace(0.5 / ATOMS, 1 - 0.5 / ATOMS, ATOMS)).astype(np.float32)

    def thresholds(self, pos: str) -> tuple[float, float]:
        return float(self.boom.get(pos, 20.0)), float(self.bust.get(pos, 6.0))

    # ---- disk ------------------------------------------------------------- #
    def to_json(self) -> str:
        return json.dumps({k: v for k, v in self.__dict__.items()}, indent=2, sort_keys=True)

    def save(self, path=None) -> str:
        ensure_dirs()
        p = path or DISPERSION_PATH
        p.write_text(self.to_json(), encoding="utf-8")
        return str(p)

    @classmethod
    def from_dict(cls, d: dict) -> Dispersion:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def load(path=None) -> Dispersion:
    """The fitted dispersion, or the defaults in `Dispersion` if nobody has fitted it yet.

    The fallbacks are the pooled measurements from `--fit` rounded off, so an unfitted app produces
    ranges that are the right order of magnitude rather than either zero or nonsense. Which one is in
    use is in `meta`, and the app says so.
    """
    p = path or DISPERSION_PATH
    if not p.exists():
        return Dispersion(meta={"fitted": False})
    return Dispersion.from_dict(json.loads(p.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def _log_sd(values: np.ndarray) -> float:
    v = values[np.isfinite(values) & (values > 0)]
    return float(np.log(v).std()) if v.size > 8 else float("nan")


def _team_transient(seasons: tuple[int, ...]) -> tuple[dict, dict, dict]:
    """A team's weekly volume against its own season mean, and how the two teams in a game covary."""
    tw = history.team_weeks(seasons)
    week, rho, n = {}, {}, {}
    for ch in CHANNELS:
        if ch.team_week not in tw.columns:
            continue
        d = tw.select("season", "team", "opponent", "game_id",
                      pl.col(ch.team_week).cast(pl.Float64).alias("v"))
        d = d.with_columns(pl.col("v").mean().over(["season", "team"]).alias("mu")).filter(
            pl.col("mu") > 0.5
        )
        d = d.with_columns(((pl.col("v") + 0.5) / (pl.col("mu") + 0.5)).log().alias("r"))
        week[ch.name] = float(d["r"].std())
        pair = d.join(
            d.select("game_id", pl.col("team").alias("opponent"), pl.col("r").alias("r2")),
            on=["game_id", "opponent"], how="inner",
        )
        rho[ch.name] = float(np.corrcoef(pair["r"].to_numpy(), pair["r2"].to_numpy())[0, 1])
        n[ch.name] = int(d.height)
    return week, rho, n


def _team_persistent(seasons: tuple[int, ...], settings: Settings) -> tuple[dict, dict]:
    """How wrong the team's *season* per-game volume estimate is, in the engine's own form.

    `estimate = league_mean + keep * (w * last + (1-w) * prev - league_mean)`, which is stage one of
    the team model with the settings the projection will use, scored on the season it did not see.
    """
    ts = history.team_seasons(seasons)
    out, n = {}, {}
    w, keep = settings.team_weight_recent, settings.team_keep_vs_mean
    for ch in CHANNELS:
        m = ch.team_metric
        if m not in ts.columns:
            continue
        d = ts.select("season", "team", pl.col(m).cast(pl.Float64).alias("v"))
        d = d.with_columns(pl.col("v").mean().over("season").alias("lmean"))
        lag1 = d.select("team", (pl.col("season") + 1).alias("season"),
                        pl.col("v").alias("last"), pl.col("lmean").alias("l1"))
        lag2 = d.select("team", (pl.col("season") + 2).alias("season"),
                        pl.col("v").alias("prev"), pl.col("lmean").alias("l2"))
        j = d.join(lag1, on=["team", "season"]).join(lag2, on=["team", "season"]).drop_nulls()
        if j.is_empty():
            continue
        est = (w * j["l1"] + (1 - w) * j["l2"]) + keep * (
            (w * j["last"] + (1 - w) * j["prev"]) - (w * j["l1"] + (1 - w) * j["l2"])
        )
        out[ch.name] = _log_sd((j["v"] / est).to_numpy())
        n[ch.name] = int(j.height)
    return out, n


def _player_weeks(seasons: tuple[int, ...]) -> pl.DataFrame:
    """Every player-week the dispersion measurements need, skill players and quarterbacks together."""
    sw = history.skill_weeks(seasons)
    qw = history.qb_weeks(seasons)
    want = ["season", "player_id", "position", "targets", "carries", "receptions",
            "receiving_yards", "rushing_yards", "attempts", "passing_yards",
            "team_targets", "team_carries", "team_pass_attempts"]
    frames = []
    for f in (sw, qw):
        have = [c for c in want if c in f.columns]
        missing = [pl.lit(None, dtype=pl.Float64).alias(c) for c in want if c not in f.columns]
        frames.append(f.select(*have, *missing).select(want))
    return pl.concat(frames, how="vertical_relaxed")


def _ratio_spread(
    panel: pl.DataFrame, num: str, den: str, pos: str, min_week: float, min_season: float,
    min_num: float = 0.0,
) -> tuple[float, float, int]:
    """Squared CV of a weekly ratio against the player's own season ratio, and the Poisson part of it.

    Two numbers come out and they are used differently. The observed squared CV is what the data says.
    `1/n` averaged over the same rows is how much of it is simply the arithmetic of small counts -- a
    receiver who averages four targets cannot help varying. The difference is the part that belongs to
    usage, and it is the part that is carried forward to a player whose count is different.

    `min_num` is on the player's own season total and it is what makes a *share* measurable. For a rate
    the denominator is the player's own count, so bounding that is enough; for a share the denominator
    is the team's, which every player on the roster clears, and without a floor on the numerator the
    number that comes back is the spread of a receiver's one carry a year rather than of a usage share.
    """
    d = panel.filter(pl.col("position") == pos).select(
        "season", "player_id",
        pl.col(num).cast(pl.Float64).alias("n"), pl.col(den).cast(pl.Float64).alias("d"),
    ).drop_nulls()
    d = d.with_columns(
        pl.col("n").sum().over(["season", "player_id"]).alias("sn"),
        pl.col("d").sum().over(["season", "player_id"]).alias("sd"),
        pl.len().over(["season", "player_id"]).alias("g"),
    ).filter(
        (pl.col("g") >= 8) & (pl.col("sd") >= min_season) & (pl.col("d") >= min_week)
        & (pl.col("sn") > min_num)
    )
    if d.height < 50:
        return float("nan"), float("nan"), int(d.height)
    r = ((d["n"] / d["d"]) / (d["sn"] / d["sd"])).to_numpy()
    r = r[np.isfinite(r)]
    obs = float(np.var(r))
    poisson = float(np.mean(1.0 / np.maximum(d["d"].to_numpy(), 1.0)))
    return obs, poisson, int(len(r))


def _weekly_dispersion(seasons: tuple[int, ...]) -> tuple[dict, dict, dict]:
    """Usage dispersion per position and channel, and per-event dispersion per position and group."""
    panel = _player_weeks(seasons)
    usage, event, n = {}, {}, {}
    for pos in POSITIONS:
        for ch in CHANNELS:
            num, den = ch.share
            obs, poisson, rows = _ratio_spread(panel, num, den, pos, 1.0, 40.0, min_num=30.0)
            if not np.isfinite(obs):
                continue
            # a share's weekly wobble is the small-count arithmetic plus the coach; only the coach
            # transfers to a player with a different count, so the arithmetic comes out here and goes
            # back in at simulation time against his own projected count.
            # bounded above because a share shock is zero-sum inside the team: an unbounded draw on a
            # player with almost no volume would take real carries off the back beside him
            usage[f"{pos}:{ch.name}"] = float(np.clip(obs - poisson, 0.01, 1.5))
            n[f"usage:{pos}:{ch.name}"] = rows
        for g in GROUPS:
            if g.rate is None:
                continue
            num, den = g.rate
            obs, poisson, rows = _ratio_spread(panel, num, den, pos, 2.0, 30.0)
            if not np.isfinite(obs):
                continue
            # a rate's wobble is per-event noise averaged over the week's events, so the transferable
            # number is the variance of one event: obs * n, averaged the same way
            event[f"{pos}:{g.name}"] = max(obs / max(poisson, 1e-3), 0.01)
            n[f"event:{pos}:{g.name}"] = rows
    return usage, event, n


def _tiered(players: pl.DataFrame) -> pl.DataFrame:
    """Tag each backtest row with the tier the projection put it in, before the season was played."""
    return players.with_columns(
        pl.col("fantasy_points").rank("min", descending=True).over(["season", "variant", "position"])
        .alias("proj_rank")
    ).with_columns(
        pl.when(pl.col("proj_rank")
                <= pl.col("position").replace_strict(STARTERS, default=0, return_dtype=pl.Int32))
        .then(pl.lit("starter")).otherwise(pl.lit("depth")).alias("tier")
    )


def _season_errors(players: pl.DataFrame, team_season: dict) -> tuple[dict, dict, dict]:
    """Persistent error, from the backtest: how wrong a season's shares and rates really were.

    The share figure is a per-game volume error, which contains the team's error as well as the
    player's, so the team's measured season sigma comes out in quadrature -- otherwise the same
    uncertainty is counted twice, once at each level, and the intervals come out wide for a reason that
    is not real.

    Measured per tier, and the two tiers are not close: a projected starter's share of the pool comes
    back within a quarter of what was projected, a depth player's is a coin toss on whether he was in
    the rotation at all. A pooled figure is neither.
    """
    full = _tiered(players.filter(pl.col("variant") == "full"))
    share, rate, n = {}, {}, {}
    for pos in POSITIONS:
        for tier in TIERS:
            sub = full.filter(
                (pl.col("position") == pos) & (pl.col("tier") == tier)
                & (pl.col("a_games") >= 4) & (pl.col("games") >= 4)
            )
            for ch in CHANNELS:
                c, a = ch.count, f"a_{ch.count}"
                if c not in sub.columns or a not in sub.columns:
                    continue
                d = sub.filter((pl.col(c) > 1.0) & (pl.col(a) > 1.0))
                sd = _log_sd(((d[a] / d["a_games"]) / (d[c] / d["games"])).to_numpy())
                if not np.isfinite(sd):
                    continue
                team = team_season.get(ch.name, 0.0)
                share[f"{pos}:{tier}:{ch.name}"] = float(np.sqrt(max(sd**2 - team**2, 0.05**2)))
                n[f"share:{pos}:{tier}:{ch.name}"] = int(d.height)
            for g in GROUPS:
                if g.rate is None:
                    continue
                num, den = g.rate
                if num not in sub.columns or f"a_{num}" not in sub.columns:
                    continue
                d = sub.filter((pl.col(den) > 5.0) & (pl.col(f"a_{den}") > 5.0)
                               & (pl.col(num) > 1.0) & (pl.col(f"a_{num}") > 1.0))
                sd = _log_sd(((d[f"a_{num}"] / d[f"a_{den}"]) / (d[num] / d[den])).to_numpy())
                if np.isfinite(sd):
                    rate[f"{pos}:{tier}:{g.name}"] = sd
                    n[f"rate:{pos}:{tier}:{g.name}"] = int(d.height)
    # a touchdown rate is not measurable per season at this sample size without the noise swamping the
    # signal, so a count group inherits the yardage-rate error of the channel it rides on
    for pos in POSITIONS:
        for tier in TIERS:
            for g in GROUPS:
                if not g.count or g.channel is None:
                    continue
                src = {"targets": "rec_yards", "carries": "rush_yards",
                       "attempts": "pass_yards"}[g.channel]
                rate.setdefault(f"{pos}:{tier}:{g.name}",
                                rate.get(f"{pos}:{tier}:{src}", 0.30))
    return share, rate, n


def _availability(players: pl.DataFrame) -> tuple[dict, dict]:
    """The measured distribution of actual games over projected games, centred to mean 1.

    Resampled rather than fitted: the shape is what matters and it is not a Beta. Most of the sample is
    a player who played nearly every week, and there is a hard lump at zero for the ones who never took
    the field. Centring is deliberate -- the mean of the raw ratio is a *bias* in expected games, not a
    spread, and correcting a bias silently inside a range model would hide it. The number that was
    divided out is in `meta`.

    Per tier, because the lump at zero is almost entirely depth: a projected starter misses games, a
    projected fifth receiver never appears, and one sample cannot describe both. Stored as `ATOMS`
    quantiles rather than the raw sample so the shape survives -- including the lump at zero -- while
    the arithmetic in `_avail_scale` stays the same size for every position.
    """
    full = _tiered(players.filter((pl.col("variant") == "full") & (pl.col("games") >= 4)))
    out, bias = {}, {}
    for pos in POSITIONS:
        for tier in TIERS:
            r = (full.filter((pl.col("position") == pos) & (pl.col("tier") == tier))
                 .select((pl.col("a_games") / pl.col("games")).alias("r"))["r"].to_numpy())
            r = r[np.isfinite(r)]
            if r.size < 40:
                continue
            mean = float(r.mean())
            bias[f"{pos}:{tier}"] = mean
            q = np.quantile(r / max(mean, EPS), np.linspace(0.5 / ATOMS, 1 - 0.5 / ATOMS, ATOMS))
            out[f"{pos}:{tier}"] = [round(float(x), 4) for x in q]
    return out, bias


def _thresholds(seasons: tuple[int, ...], scoring: Scoring) -> tuple[dict, dict]:
    """Boom and bust in points, measured: the 85th and 25th percentile of a startable week.

    Absolute thresholds only mean something against the population that produces them, so they come
    from the weeks of players who were startable that season -- the top 12/24/36/12 by season points at
    the position. A user can move them; this is where they start.
    """
    pg = lake.regular_season(lake.read("player_games", seasons=seasons))
    pg = pg.with_columns(history.fantasy_points(pg.columns, scoring),
                         pl.col("position").replace({"FB": "RB"}))
    season_tot = pg.group_by(["season", "player_id", "position"]).agg(
        pl.col("fantasy_points").sum().alias("total")
    )
    keep = {"QB": 12, "RB": 24, "WR": 36, "TE": 12}
    boom, bust = {}, {}
    for pos, n in keep.items():
        top = (season_tot.filter(pl.col("position") == pos)
               .with_columns(pl.col("total").rank("min", descending=True).over("season").alias("rk"))
               .filter(pl.col("rk") <= n).select("season", "player_id"))
        weeks = pg.join(top, on=["season", "player_id"], how="semi")["fantasy_points"].to_numpy()
        if weeks.size < 100:
            continue
        boom[pos] = float(np.round(np.quantile(weeks, 0.85), 1))
        bust[pos] = float(np.round(np.quantile(weeks, 0.25), 1))
    return boom, bust


def measure(
    seasons: tuple[int, ...] | None = None,
    settings: Settings | None = None,
    players: pl.DataFrame | None = None,
) -> Dispersion:
    """Fit every sigma from history and from the saved backtest. Nothing here is assumed."""
    settings = settings or Settings()
    seasons = seasons or lake.history_seasons()
    if players is None:
        from src.model import backtest
        if not backtest.PLAYER_PATH.exists():
            raise FileNotFoundError(
                f"{backtest.PLAYER_PATH} is missing; run `python -m src.model.backtest` first -- the "
                "persistent half of the dispersion is measured from its held-out residuals"
            )
        players = pl.read_parquet(backtest.PLAYER_PATH)

    team_week, team_rho, n_team = _team_transient(seasons)
    team_season, n_ts = _team_persistent(seasons, settings)
    usage, event, n_week = _weekly_dispersion(seasons)
    share, rate, n_season = _season_errors(players, team_season)
    games, games_bias = _availability(players)
    boom, bust = _thresholds(seasons, settings.scoring)

    return Dispersion(
        team_week=team_week, team_rho=team_rho, team_season=team_season,
        usage_cv2=usage, event_cv2=event, share_season=share, rate_season=rate,
        games_ratio=games, boom=boom, bust=bust, scale={p: 1.0 for p in POSITIONS},
        meta={
            "fitted": True,
            "seasons": [int(min(seasons)), int(max(seasons))],
            "backtest_seasons": sorted({int(s) for s in players["season"].unique()}),
            "games_bias": {k: round(v, 4) for k, v in games_bias.items()},
            "n": {**{f"team_week:{k}": v for k, v in n_team.items()},
                  **{f"team_season:{k}": v for k, v in n_ts.items()},
                  **n_week, **n_season},
            "calibrated": False,
        },
    )


# --------------------------------------------------------------------------- #
# the simulation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class Sim:
    """The result of one run: a season frame everybody is in, and full draws for the ones asked for.

    `points_draws` is (players, draws) of season fantasy points, in the row order of `season`. It is
    kept rather than reduced to quantiles because "what are the odds he clears 250" is a question a
    user asks *after* the sim has run, and answering it from five stored quantiles would be a guess.
    """

    season: pl.DataFrame
    weekly: pl.DataFrame
    stats: pl.DataFrame
    points_draws: np.ndarray
    index: dict[str, int]
    draws: int
    dispersion: Dispersion

    def p_over(self, player_id: str, x: float) -> float:
        i = self.index.get(player_id)
        if i is None:
            return float("nan")
        return float((self.points_draws[i] > x).mean())

    def draws_of(self, player_id: str) -> np.ndarray:
        i = self.index.get(player_id)
        return self.points_draws[i] if i is not None else np.zeros(0, dtype=np.float32)

    def histogram(self, player_id: str, bins: int = 40) -> pl.DataFrame:
        """The distribution as `points` / `share` / `cumulative`, for plotting rather than tabulating.

        Binned here rather than in the app because a plot of a distribution is the distribution: a page
        that chose its own bins could show a bimodal season as a smooth one. The lump at zero -- the
        draws where he never took the field -- gets its own bin for the same reason, since it is the
        single most decision-relevant feature of a fragile player's range and averaging it into the first
        bin of a wide histogram hides it.
        """
        d = self.draws_of(player_id)
        if d.size == 0:
            return pl.DataFrame({"points": [], "share": [], "cumulative": []},
                                schema={"points": pl.Float64, "share": pl.Float64,
                                        "cumulative": pl.Float64})
        hi = float(np.quantile(d, 0.999))
        edges = np.concatenate(([0.0, 1e-6], np.linspace(1e-6, max(hi, 1.0), bins)[1:]))
        counts, _ = np.histogram(np.clip(d, 0.0, edges[-1]), bins=edges)
        mid = (edges[:-1] + edges[1:]) / 2.0
        share = counts / float(d.size)
        return pl.DataFrame({"points": mid, "share": share, "cumulative": np.cumsum(share)})


def _scoring_weights(scoring: Scoring) -> dict[str, float]:
    """The scoring rule as a dict over the projection's own stat names."""
    return {
        "passing_yards": scoring.pass_yard, "passing_tds": scoring.pass_td,
        "interceptions": scoring.interception, "completions": scoring.completion,
        "rushing_yards": scoring.rush_yard, "rushing_tds": scoring.rush_td,
        "receiving_yards": scoring.rec_yard, "receiving_tds": scoring.rec_td,
        "receptions": scoring.reception, "fumbles_lost": scoring.fumble_lost,
    }


@dataclass
class _Sub:
    """One volume pool inside one week: the rows that have any of it, and how to sum them per team.

    Row subsets are the difference between a simulation that runs in three seconds and one that runs in
    ninety. Nine hundred players are in a week and forty of them throw, so drawing a passing shock for
    every row spends nearly all of its time multiplying zero by noise. Because the week is sorted by
    team, a subset stays sorted, so the pool sums are still one `reduceat`.
    """

    rows: np.ndarray            # index into the week's rows
    base: np.ndarray            # per-play count at those rows, strictly positive
    tstart: np.ndarray          # block boundaries within the subset, for reduceat
    tof: np.ndarray             # subset row -> subset block
    tblock: np.ndarray          # subset block -> the week's team block


@dataclass
class _Grp:
    """One block of points inside one week, on the rows that have any of it."""

    rows: np.ndarray
    points: np.ndarray          # per-play fantasy points
    events: np.ndarray          # per-play count of the rate's denominator
    tds: np.ndarray             # per-play expected count, for a Poisson group
    per_td: float               # points per event, for a Poisson group


@dataclass
class _Week:
    """One week of the projection as numpy, sorted by team so a pool can be summed with `reduceat`."""

    week: int
    rows: int
    pidx: np.ndarray            # row -> player index
    unique: bool                # is every player in this week exactly once
    gof: np.ndarray             # team block -> game index
    sign: np.ndarray            # team block -> +1/-1, which side of the shared game shock
    n_teams: int
    p_play: np.ndarray
    chan: dict[str, _Sub]
    grp: dict[str, _Grp]
    stats: dict[str, np.ndarray]       # tracked stat -> per-play value, full length
    stat_group: dict[str, str]         # tracked stat -> the group whose shock moves it


def _blocks(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Block starts and a row->block map for an already-sorted key column."""
    if keys.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    edge = np.r_[True, keys[1:] != keys[:-1]]
    return np.flatnonzero(edge), np.cumsum(edge) - 1


def _prepare(
    wk: pl.DataFrame, settings: Settings
) -> tuple[list[_Week], list[str], np.ndarray, np.ndarray]:
    """Slice the weekly projection into per-week numpy blocks, once.

    Everything is stored *per play*: the projection's counts already have availability inside them, and
    the simulation draws availability itself, so `p_play` has to come out first or it would be applied
    twice. A player whose availability is zero has no per-play line to draw and drops out.
    """
    weights = _scoring_weights(settings.scoring)
    players = wk["player_id"].unique(maintain_order=True).sort().to_list()
    pmap = {p: i for i, p in enumerate(players)}
    who = (wk.group_by("player_id").agg(pl.col("position").first(),
                                        pl.col("fantasy_points").sum().alias("total"))
           .with_columns(pl.col("total").rank("min", descending=True).over("position").alias("rk"))
           .with_columns(
               pl.when(pl.col("rk") <= pl.col("position")
                       .replace_strict(STARTERS, default=0, return_dtype=pl.Int32))
               .then(pl.lit("starter")).otherwise(pl.lit("depth")).alias("tier")))
    pos_of = dict(who.select("player_id", "position").iter_rows())
    tier_of = dict(who.select("player_id", "tier").iter_rows())
    positions = np.array([pos_of.get(p, "WR") for p in players])
    tiers = np.array([tier_of.get(p, "depth") for p in players])

    stat_group = {}
    for g in GROUPS:
        for s in g.stats:
            if s in TRACKED:
                stat_group[s] = g.name
    for s in TRACKED:                      # a count with no scoring weight still rides its channel
        if s not in stat_group:
            for g in GROUPS:
                if g.channel == s:
                    stat_group[s] = g.name
                    break

    out: list[_Week] = []
    for week in sorted(wk["week"].unique().to_list()):
        w = wk.filter(pl.col("week") == week).sort(["team", "player_id"])
        p = w["p_play"].to_numpy().astype(np.float32)
        live = p > 1e-6
        w, p = w.filter(live), p[live]
        if w.is_empty():
            continue
        teams = w["team"].to_numpy()
        tstart, tof = _blocks(teams)
        team_names = teams[tstart]
        # the two teams in a game share one shock; which of them takes it positively is arbitrary but
        # has to be stable, so it is alphabetical
        games = w["game_id"].to_numpy()[tstart]
        gmap = {g: i for i, g in enumerate(dict.fromkeys(games))}
        gof = np.array([gmap[g] for g in games])
        sign = np.ones(len(team_names), dtype=np.float32)
        for gi in range(len(gmap)):
            in_game = np.flatnonzero(gof == gi)
            if in_game.size == 2:
                sign[in_game[np.argmax(team_names[in_game])]] = -1.0

        def per_play(col: str) -> np.ndarray:
            v = (w[col].to_numpy().astype(np.float32) if col in w.columns
                 else np.zeros(len(p), np.float32))
            return np.nan_to_num(v) / p

        chan = {}
        counts = {}
        for c in CHANNELS:
            base = per_play(c.count)
            counts[c.name] = base
            rows = np.flatnonzero(base > 1e-6)
            starts, sof = _blocks(tof[rows])
            chan[c.name] = _Sub(
                rows=rows, base=base[rows], tstart=starts, tof=sof,
                tblock=tof[rows][starts] if rows.size else np.zeros(0, dtype=int),
            )

        grp, stats = {}, {}
        for g in GROUPS:
            pts = np.zeros(len(p), np.float32)
            for s in g.stats:
                if s in w.columns and weights.get(s):
                    pts = pts + per_play(s) * np.float32(weights[s])
            tds = per_play(g.stats[0]) if g.count else np.zeros(len(p), np.float32)
            events = counts[g.channel] if g.channel else np.zeros(len(p), np.float32)
            live_g = (np.abs(pts) > 1e-9) | (tds > 1e-9)
            # a group rides its channel, so its rows are a subset of the channel's: the ratio the
            # channel produces is only ever read where the channel had volume to begin with
            if g.channel:
                live_g &= counts[g.channel] > 1e-6
            rows = np.flatnonzero(live_g)
            grp[g.name] = _Grp(rows=rows, points=pts[rows], events=events[rows], tds=tds[rows],
                               per_td=float(weights.get(g.stats[0], 0.0)) if g.count else 0.0)
        for s in TRACKED:
            if s in w.columns:
                stats[s] = per_play(s)

        pidx = np.array([pmap[x] for x in w["player_id"]])
        out.append(_Week(
            week=int(week), rows=len(p), pidx=pidx, unique=len(np.unique(pidx)) == len(pidx),
            gof=gof, sign=sign, n_teams=len(team_names), p_play=p, chan=chan, grp=grp,
            stats=stats, stat_group=stat_group,
        ))
    return out, players, positions, tiers


def _sigma(cv2: np.ndarray) -> np.ndarray:
    """Lognormal sigma with the given squared coefficient of variation."""
    return np.sqrt(np.log1p(np.maximum(cv2, 1e-6))).astype(np.float32)


def _avail_scale(pp: np.ndarray, atoms: np.ndarray, iters: int = 30) -> np.ndarray:
    """Per player, the multiplier on the availability sample that keeps his expected games honest.

    A week is played with probability `p_play * ratio`, and a probability cannot exceed one, so the
    clip at one silently eats the upper tail: a backup drawn to play twice as often as projected gets
    capped, nothing compensates, and his expected season lands eight per cent under the projection it
    was supposed to be a range around. The clip is not negotiable, so the sample is scaled instead --
    one number per player, solved so that the expected games come back to `sum(p_play)`.

    `f(k) = sum_w mean_j min(p_w * k * a_j, 1)` is increasing in `k` and bounded by the number of
    weeks, so a bisection converges on it in a couple of dozen halvings. Where the projection already
    has a player at essentially every week there is nothing to solve and `k` stays at one.
    """
    want = pp.sum(1)
    a = atoms[:, None, :]                      # one row of atoms per player: (players, 1, ATOMS)
    # clipping can only ever lose games, so the answer is at or above one; the upper bound is generous
    # for a deep backup whose sample is mostly zero
    lo = np.full(pp.shape[0], 0.9, np.float32)
    hi = np.full(pp.shape[0], 20.0, np.float32)

    def f(k: np.ndarray) -> np.ndarray:
        return np.minimum(pp[:, :, None] * k[:, None, None] * a, 1.0).mean(2).sum(1)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_low = f(mid) < want
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    k = 0.5 * (lo + hi)
    return np.where(want > 1e-6, k, 1.0).astype(np.float32)


def run(
    wk: pl.DataFrame,
    settings: Settings | None = None,
    disp: Dispersion | None = None,
    draws: int | None = None,
    seed: int = SEED,
    detail: tuple[str, ...] = (),
    chunk: int = 1000,
) -> Sim:
    """Simulate the season the weekly frame describes.

    `wk` is `compose.weekly` -- the current one, edits included. Nothing is re-projected here: the sim
    shocks what the projection said, which is what makes a range respond to an override instead of
    quietly describing the unedited engine.
    """
    settings = settings or Settings()
    disp = disp or load()
    n_draws = int(draws or settings.simulation_draws)
    weeks, players, positions, tiers = _prepare(wk, settings)
    n_players = len(players)
    rng = np.random.default_rng(seed)

    # per-player sigmas, looked up once
    pt = list(zip(positions, tiers, strict=True))
    share_sd = {c.name: np.array([disp.k_share_season(p, t, c.name) for p, t in pt], np.float32)
                for c in CHANNELS}
    rate_sd = {g.name: np.array([disp.k_rate_season(p, t, g.name) for p, t in pt], np.float32)
               for g in GROUPS}
    usage_cv2 = {c.name: np.array([disp.k_usage(p, c.name) for p in positions], np.float32)
                 for c in CHANNELS}
    event_cv2 = {g.name: np.array([disp.k_event(p, g.name) for p in positions], np.float32)
                 for g in GROUPS if g.rate is not None}
    # the availability sample each player draws from, and the scale that keeps his expected games on
    # the projection's own number despite the clip at one
    atoms = np.stack([disp.games_sample(*k) for k in pt]).astype(np.float32)
    pp = np.zeros((n_players, max(len(weeks), 1)), np.float32)
    for wi, w in enumerate(weeks):
        pp[w.pidx, wi] = w.p_play
    atoms *= _avail_scale(pp, atoms)[:, None]
    boom_line = np.array([disp.thresholds(p)[0] for p in positions], np.float32)
    bust_line = np.array([disp.thresholds(p)[1] for p in positions], np.float32)

    n_teams = max((w.n_teams for w in weeks), default=0)
    team_sd = {c.name: np.float32(disp.k_team_season(c.name)) for c in CHANNELS}
    week_sd = {c.name: np.float32(disp.k_team_week(c.name)) for c in CHANNELS}
    rho = {c.name: float(disp.k_team_rho(c.name)) for c in CHANNELS}

    points_draws = np.zeros((n_players, n_draws), np.float32)
    games_draws = np.zeros((n_players, n_draws), np.float32)
    booms = np.zeros(n_players, np.float64)
    busts = np.zeros(n_players, np.float64)

    dset = [p for p in detail if p in set(players)]
    didx = np.array([players.index(p) for p in dset], dtype=int) if dset else np.zeros(0, int)
    stat_draws = {s: np.zeros((len(dset), n_draws), np.float32) for s in TRACKED} if dset else {}
    weekly_pts = (np.zeros((len(dset), len(weeks), n_draws), np.float32) if dset else None)

    # A team's identity is only needed within a week -- the teams playing in week 3 are the same 32 --
    # so the persistent team draw is indexed by the block position, which is stable because every week
    # is sorted by team name and every team plays every week bar its bye. The bye week shifts the
    # blocks by one; the cost is that a team's persistent shock is not literally the same number on the
    # far side of its bye, and that is worth the memory it saves.
    for start in range(0, n_draws, chunk):
        c = min(chunk, n_draws - start)
        sl = slice(start, start + c)

        z_team = {ch.name: rng.standard_normal((n_teams + 1, c), np.float32) for ch in CHANNELS}
        z_share = {ch.name: rng.standard_normal((n_players, c), np.float32) for ch in CHANNELS}
        z_rate = {g.name: rng.standard_normal((n_players, c), np.float32) for g in GROUPS}
        avail = atoms[np.arange(n_players)[:, None], rng.integers(0, ATOMS, (n_players, c))]

        season_pts = np.zeros((n_players, c), np.float32)
        season_games = np.zeros((n_players, c), np.float32)
        season_stats = {s: np.zeros((len(dset), c), np.float32) for s in stat_draws}

        for wi, w in enumerate(weeks):
            # --- level 1 and 2: the game and the team ------------------------ #
            team_shock = {}
            for ch in CHANNELS:
                s_w, s_s = week_sd[ch.name], team_sd[ch.name]
                r = rho[ch.name]
                shared = rng.standard_normal((int(w.gof.max()) + 1, c), np.float32)
                own = rng.standard_normal((w.n_teams, c), np.float32)
                # one draw per game, shared by both teams; the second team takes it with the sign
                # flipped when the measured correlation is negative, which for volume it is -- there is
                # one clock, and the team in front runs it down
                flip = w.sign[:, None] if r < 0 else np.float32(1.0)
                mix = (np.sqrt(abs(r)) * flip * shared[w.gof]
                       + np.sqrt(max(1.0 - abs(r), 0.0)) * own)
                team_shock[ch.name] = np.exp(
                    s_w * mix - 0.5 * s_w**2 + s_s * z_team[ch.name][:w.n_teams] - 0.5 * s_s**2
                )

            # --- availability ------------------------------------------------ #
            p_avail = np.clip(w.p_play[:, None] * avail[w.pidx], 0.0, 1.0)
            playing = (rng.random((w.rows, c), np.float32) < p_avail).astype(np.float32)
            season_games[w.pidx] += playing

            # --- level 3: the player's share of the pool --------------------- #
            ratio = {}
            for ch in CHANNELS:
                s = w.chan[ch.name]
                if s.rows.size == 0:
                    continue
                pid_s = w.pidx[s.rows]
                on = playing[s.rows]
                cv2 = usage_cv2[ch.name][pid_s] + 1.0 / np.maximum(s.base, 1.0)
                sig = _sigma(cv2)[:, None]
                sd_s = share_sd[ch.name][pid_s][:, None]
                shock = np.exp(sig * rng.standard_normal((s.rows.size, c), np.float32) - 0.5 * sig**2
                               + sd_s * z_share[ch.name][pid_s] - 0.5 * sd_s**2)
                got = s.base[:, None] * on * shock
                if settings.normalize_pools:
                    # The shares are renormalised inside the team on every draw, exactly as
                    # `opportunity` renormalises them once: a share is taken from a teammate rather
                    # than invented, so an override stays zero-sum and a shock cannot conjure volume.
                    #
                    # The target is the pool over the players drawn available, not the projected team
                    # total. Aiming at the projected total would hand an absent starter's carries to
                    # his backup on every draw -- which sounds like the right behaviour and is in fact
                    # counting the same thing twice, because `p_play` already spread that absence
                    # across the room: a second-string quarterback projects 0.9 attempts a game *only*
                    # because the starter is sometimes hurt. Measured, the double count cost a whole
                    # position group ten per cent of its projection, and on the draws where a shallow
                    # room came back empty the pool had nowhere to go at all.
                    want = np.add.reduceat(s.base[:, None] * on, s.tstart, axis=0)
                    have = np.add.reduceat(got, s.tstart, axis=0)
                    factor = np.where(have > EPS, want / np.maximum(have, EPS), 1.0)
                    got *= factor[s.tof]
                got *= team_shock[ch.name][s.tblock][s.tof]
                # as a multiple of what the projection had, which is the form every group wants it in
                full = np.zeros((w.rows, c), np.float32)
                full[s.rows] = got / s.base[:, None]
                ratio[ch.name] = full

            # --- the week's points ------------------------------------------- #
            pts = np.zeros((w.rows, c), np.float32)
            group_mult: dict[str, np.ndarray] = {}
            for g in GROUPS:
                s = w.grp[g.name]
                if s.rows.size == 0:
                    continue
                pid_s = w.pidx[s.rows]
                rat = ratio[g.channel][s.rows] if g.channel else playing[s.rows]
                sd_r = rate_sd[g.name][pid_s][:, None]
                eff = np.exp(sd_r * z_rate[g.name][pid_s] - 0.5 * sd_r**2)
                if g.rate is not None:
                    # per-event noise averaged over the events the week actually produced, so twelve
                    # targets is a narrower yards-per-target than two. A count group skips this: the
                    # week-to-week noise in a touchdown is the Poisson draw itself, and drawing a
                    # lognormal on top of it would be the same randomness twice
                    ev = np.maximum(s.events[:, None] * rat, 0.25)
                    sig = _sigma(event_cv2[g.name][pid_s][:, None] / ev)
                    eff *= np.exp(sig * rng.standard_normal((s.rows.size, c), np.float32)
                                  - 0.5 * sig**2)
                if g.count:
                    lam = np.maximum(s.tds[:, None] * rat * eff, 0.0)
                    drawn = rng.poisson(lam).astype(np.float32)
                    pts[s.rows] += drawn * np.float32(s.per_td)
                    if dset:
                        group_mult[g.name] = np.where(lam > EPS, drawn / np.maximum(lam, EPS), 0.0)
                else:
                    pts[s.rows] += s.points[:, None] * rat * eff
                    if dset:
                        group_mult[g.name] = rat * eff

            if w.unique:
                season_pts[w.pidx] += pts
            else:
                np.add.at(season_pts, w.pidx, pts)
            booms += np.bincount(w.pidx, weights=(pts >= boom_line[w.pidx][:, None]).sum(1),
                                 minlength=n_players)
            busts += np.bincount(w.pidx, weights=(pts <= bust_line[w.pidx][:, None]).sum(1),
                                 minlength=n_players)

            if dset:
                take = {p: i for i, p in enumerate(didx)}
                rows = np.array([i for i, p in enumerate(w.pidx) if p in take], dtype=int)
                if rows.size:
                    to = [take[p] for p in w.pidx[rows]]
                    # `weekly_pts` spans every draw, so it takes the chunk's slice -- unlike
                    # `season_stats`, which is a per-chunk accumulator copied out below
                    weekly_pts[to, wi, sl] = pts[rows]
                    for s, arr in season_stats.items():
                        if s not in w.stats:
                            continue
                        gname = w.stat_group.get(s, "")
                        m = group_mult.get(gname)
                        if m is None:
                            arr[to] += w.stats[s][rows][:, None] * playing[rows]
                            continue
                        where = np.searchsorted(w.grp[gname].rows, rows)
                        ok = (where < w.grp[gname].rows.size) & (
                            w.grp[gname].rows[np.minimum(where, w.grp[gname].rows.size - 1)] == rows
                        )
                        if ok.any():
                            arr[np.array(to)[ok]] += w.stats[s][rows[ok]][:, None] * m[where[ok]]

        points_draws[:, sl] = season_pts
        games_draws[:, sl] = season_games
        for s, arr in season_stats.items():
            stat_draws[s][:, sl] = arr

    n_weeks = max(len(weeks), 1)
    season = _season_frame(wk, players, positions, tiers, points_draws, games_draws,
                           booms / (n_weeks * n_draws), busts / (n_weeks * n_draws))
    # `_season_frame` sorts by median, so the draws are permuted to match it rather than left in the
    # engine's own order. A caller who attaches a per-player number computed from `points_draws` back
    # onto `season` -- exactly what the variance diagnostic does -- gets silence and wrong rows
    # otherwise, and the two orders differ by enough that the mistake still looks plausible.
    where = {p: i for i, p in enumerate(players)}
    order = np.fromiter((where[p] for p in season["player_id"]), np.int64, season.height)
    return Sim(
        season=season,
        weekly=(_weekly_frame(dset, weeks, weekly_pts, boom_line[didx], bust_line[didx])
                if dset else _empty_weekly()),
        stats=_stat_frame(dset, stat_draws) if dset else _empty_stats(),
        points_draws=points_draws[order],
        index={p: i for i, p in enumerate(season["player_id"])},
        draws=n_draws,
        dispersion=disp,
    )


def _q(a: np.ndarray, qs=QUANTILES) -> np.ndarray:
    return np.quantile(a, qs, axis=1)


def _season_frame(wk, players, positions, tiers, pts, games, boom, bust) -> pl.DataFrame:
    """One row per player: the point projection beside the distribution around it."""
    q = _q(pts)
    gq = _q(games)
    mean = pts.mean(1)
    sd = pts.std(1)
    point = (wk.group_by("player_id").agg(pl.col("fantasy_points").sum().alias("projected"),
                                          pl.col("p_play").sum().alias("projected_games"),
                                          pl.col("player").first(), pl.col("team").first()))
    out = pl.DataFrame({
        "player_id": players,
        "position": positions,
        "tier": tiers,
        "sim_mean": mean,
        "sim_sd": sd,
        "p5": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4],
        "games_mean": games.mean(1),
        "games_p5": gq[0], "games_p50": gq[2], "games_p95": gq[4],
        "boom_rate": boom, "bust_rate": bust,
    }).join(point, on="player_id", how="left")
    return out.with_columns(
        pl.col("p5").alias("floor"), pl.col("p95").alias("ceiling"),
        (pl.col("p95") - pl.col("p5")).alias("range"),
        # the volatility score: spread per point of projection, so a 40-point sd on a 300-point back is
        # not read as the same risk as a 40-point sd on an 80-point one
        pl.when(pl.col("sim_mean") > 1.0).then(pl.col("sim_sd") / pl.col("sim_mean"))
        .otherwise(None).alias("volatility"),
        (pl.col("sim_mean") - pl.col("projected")).alias("sim_vs_projected"),
    ).sort("p50", descending=True).with_columns(
        pl.col("p50").rank("min", descending=True).over("position").cast(pl.Int32)
        .alias("median_rank"),
        pl.col("p95").rank("min", descending=True).over("position").cast(pl.Int32)
        .alias("ceiling_rank"),
        pl.col("p5").rank("min", descending=True).over("position").cast(pl.Int32).alias("floor_rank"),
    )


def _weekly_frame(dset, weeks, weekly_pts, boom, bust) -> pl.DataFrame:
    """Each of the player's weeks as a distribution. A bye is absent rather than a zero."""
    rows = []
    for i, pid in enumerate(dset):
        for wi, w in enumerate(weeks):
            a = weekly_pts[i, wi]
            q = np.quantile(a, QUANTILES)
            rows.append({"player_id": pid, "week": w.week, "mean": float(a.mean()),
                         "p5": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
                         "p75": float(q[3]), "p95": float(q[4]),
                         "boom_rate": float((a >= boom[i]).mean()),
                         "bust_rate": float((a <= bust[i]).mean())})
    return pl.DataFrame(rows, schema=_WEEKLY_SCHEMA)


_WEEKLY_SCHEMA = {"player_id": pl.String, "week": pl.Int64, "mean": pl.Float64, "p5": pl.Float64,
                  "p25": pl.Float64, "p50": pl.Float64, "p75": pl.Float64, "p95": pl.Float64,
                  "boom_rate": pl.Float64, "bust_rate": pl.Float64}
_STAT_SCHEMA = {"player_id": pl.String, "stat": pl.String, "mean": pl.Float64, "p5": pl.Float64,
                "p25": pl.Float64, "p50": pl.Float64, "p75": pl.Float64, "p95": pl.Float64}


def _empty_weekly() -> pl.DataFrame:
    return pl.DataFrame(schema=_WEEKLY_SCHEMA)


def _empty_stats() -> pl.DataFrame:
    return pl.DataFrame(schema=_STAT_SCHEMA)


def _stat_frame(dset, stat_draws) -> pl.DataFrame:
    rows = []
    for i, pid in enumerate(dset):
        for s, arr in stat_draws.items():
            a = arr[i]
            if not np.any(a):
                continue
            q = np.quantile(a, QUANTILES)
            rows.append({"player_id": pid, "stat": s, "mean": float(a.mean()),
                         "p5": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
                         "p75": float(q[3]), "p95": float(q[4])})
    return pl.DataFrame(rows, schema=_STAT_SCHEMA)


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
COVER = ((0.05, 0.95, 0.90), (0.25, 0.75, 0.50))

# the projected total from which a season is worth putting an interval around. Fifty points is a bench
# player over seventeen games, and below it the distribution is mostly the lump at zero: an interval is
# still drawn but it is not a quantile any more, so it is not what the scale should be fitted on.
STARTABLE = 50.0


def _pit(pts: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Where each outcome fell in its own simulated distribution. Uniform is calibrated.

    The mid-PIT rather than the plain one: half the mass of a tie is counted, which matters here because
    a simulated season is discrete in two places that are not edge cases -- the touchdown counts, and the
    lump of draws at exactly zero for a player who might miss the year. Under the plain
    `(draws < actual)` every one of those ties lands at the bottom of its own interval, and the
    uniformity test then reads a correct model as biased low.
    """
    a = actual[:, None]
    return 0.5 * ((pts < a).mean(1) + (pts <= a).mean(1))


def _cvm(pit: np.ndarray) -> float:
    """Cramer-von Mises distance of the PIT from uniform: one number for "is this calibrated".

    This is what the scale is fitted on, in place of the coverage of two nested intervals. Two
    intervals turn out not to identify a scale: a structural simulation cannot reach 0.90 at the
    extremes whatever the scale, so the P5-P95 term is a near-constant slope pushing the scale up, while
    the P25-P75 term pulls it back, and the winner is wherever those two cross -- which moved from 1.5 to
    2.5 between two runs of the same fit. The whole PIT is stable because it is not being asked to
    balance two points against each other.
    """
    u = np.sort(pit[np.isfinite(pit)])
    n = u.size
    if n < 2:
        return float("nan")
    grid = (2.0 * np.arange(1, n + 1) - 1.0) / (2.0 * n)
    # the statistic itself, undivided: its expectation under uniformity is 1/6 whatever `n` is, so the
    # figure for a 400-quarterback season is comparable with the one for 1200 receivers
    return float(((u - grid) ** 2).sum() + 1.0 / (12.0 * n))


def _pit_iqr(pit: np.ndarray) -> float:
    """The middle half of the PIT, which is what a width should be fitted on. Uniform gives 0.5.

    `_cvm` is the right verdict on a whole distribution and the wrong objective for this one scalar,
    because it cannot tell a bad width from a bad centre. Inflating a persistent sigma is nearly
    mean-preserving and therefore *median-lowering*, so a position the engine over-projects can have its
    PIT pushed towards uniform by widening it -- and the fit will do exactly that: measured on 2021-2025
    the quarterback scale ran to the top of its own grid at 2.5 while the median bias it was flattening
    went from -18 points to +26. That trades a stated bias for an unstated one, and a range that is wide
    because the middle is wrong is worse than either error alone, since it also reads as low confidence.

    The interquartile range of the PIT is invariant to any shift of the whole distribution, so it scores
    only the spread. Whatever location error is left over stays visible as `median_bias`, which is a
    number for the projection to answer for rather than the interval.
    """
    u = pit[np.isfinite(pit)]
    if u.size < 4:
        return float("nan")
    q25, q75 = np.quantile(u, (0.25, 0.75))
    return float(q75 - q25)


def coverage(sim: Sim, actual: pl.DataFrame, view: str = "all", min_projected: float = 0.0,
             min_n: int = 10) -> pl.DataFrame:
    """Did the intervals contain the outcomes? One row per position plus the pooled answer.

    **The population has to be chosen on the projection, never on the outcome.** `played` looks like
    the fair cut -- why charge a range for a man who was cut in August -- but it conditions on an event
    the projection itself assigns probability to, so it keeps only the seasons that beat the `p_play`
    the model applied and the PIT comes back shifted right by construction. Measured on 2024 and 2025
    that alone is worth about +0.06 of mean PIT and reads as an upside tail that is short when it is
    not. `regulars` is worse for the same reason, and harder.

    So `view` defaults to `all` -- everyone the board listed, a zero for whoever never took the field,
    which is exactly what the simulated injury lump is there to cover -- and the population is trimmed
    by `min_projected` instead. That cut is unconditional, and it earns its place: below about 20
    projected points a player's distribution is mostly the lump at zero, so his PIT is a discrete atom
    rather than a quantile and hundreds of 0-against-0 rows drown the players a user is deciding
    between. Report both: the whole board, and the startable part of it.
    """
    j = sim.season.join(actual, on="player_id", how="inner")
    if view in j.columns:
        j = j.filter(pl.col(view))
    if min_projected > 0.0:
        j = j.filter(pl.col("projected") > min_projected)
    idx = np.array([sim.index[p] for p in j["player_id"]])
    pts = sim.points_draws[idx]
    a = j["a_fantasy_points"].to_numpy()
    pit = _pit(pts, a)
    rows = []
    for pos in (*POSITIONS, "ALL"):
        m = np.ones(len(a), bool) if pos == "ALL" else (j["position"].to_numpy() == pos)
        if m.sum() < min_n:
            continue
        row = {"position": pos, "n": int(m.sum()), "pit_mean": float(pit[m].mean()),
               "cvm": _cvm(pit[m]), "pit_iqr": _pit_iqr(pit[m])}
        for lo, hi, want in COVER:
            inside = ((pit[m] >= lo) & (pit[m] <= hi)).mean()
            row[f"cover_{round((hi - lo) * 100)}"] = float(inside)
            row[f"want_{round((hi - lo) * 100)}"] = want
        # the two tails are the complement of the widest band, strictly. `cover_90` is closed on both
        # edges, so an outcome landing exactly on P5 was counted as covered *and* below, and the three
        # shares summed to more than one -- by 2 points of a hundred at 120 draws. A simulated season is
        # discrete in two places, so landing exactly on an edge is common rather than a curiosity, and a
        # floor that contains the outcome is not a floor that was missed.
        lo, hi, _ = COVER[0]
        row["below"] = float((pit[m] < lo).mean())
        row["above"] = float((pit[m] > hi).mean())
        row["median_bias"] = float(np.median(a[m] - j["p50"].to_numpy()[m]))
        rows.append(row)
    return pl.DataFrame(rows)


def _ex_ante_weekly(season: int, settings: Settings):
    """The held-out weekly projection for a played season, and what actually happened."""
    from src.model import backtest
    ex = backtest.ex_ante(season, settings)
    wk = backtest.project_weekly(season, settings, ex)
    seasonal = backtest.compose.seasonal(wk, season, settings)
    actual = backtest.score_frame(seasonal, ex).select(
        "player_id", "a_games", "a_fantasy_points", "played", "regulars", "starters"
    )
    return wk, actual


def calibrate(
    targets: tuple[int, ...] | None = None,
    disp: Dispersion | None = None,
    settings: Settings | None = None,
    draws: int = 2000,
    grid: tuple[float, ...] = (0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5),
    verbose: bool = True,
) -> tuple[Dispersion, pl.DataFrame]:
    """Scale the persistent sigmas per position until the intervals cover what they claim to.

    The structural sim is narrow by construction: it knows about the errors it was told about and
    nothing about the ones nobody measured -- a coach nobody expected, a scheme change, a trade. One
    scalar per position over the *season* sigmas is the smallest correction that fixes that, and it is
    fitted on held-out seasons rather than chosen. The transient sigmas are left alone: week-to-week
    spread is directly observed and does not need a fudge.

    Fitted from `STARTABLE` projected points up. The cut is on the projection, not the outcome -- see
    `coverage` for why the other way round is not allowed -- and it changes the answer rather than the
    presentation: a fifth receiver's simulated season is mostly the lump at zero, so his PIT is an atom
    no choice of sigma can spread out, and fitting over the whole board lets several hundred such atoms
    outvote the players the scalar is actually for.
    """
    from src.model import backtest
    settings = settings or backtest.base_settings()
    disp = disp or load()
    targets = targets or tuple(range(2021, LAST_COMPLETE_SEASON + 1))

    frames = [_ex_ante_weekly(s, settings) for s in targets]
    rows = []
    for scale in grid:
        trial = replace(disp, scale={p: scale for p in POSITIONS})
        per = []
        for (wk, actual), season in zip(frames, targets, strict=True):
            sim = run(wk, settings, trial, draws=draws)
            per.append(coverage(sim, actual, min_projected=STARTABLE)
                       .with_columns(pl.lit(season).alias("season"), pl.lit(scale).alias("scale")))
        got = pl.concat(per)
        pooled = got.filter(pl.col("position") != "ALL")
        rows.append(got)
        if verbose:
            allp = got.filter(pl.col("position") == "ALL")
            print(f"  scale {scale:<5} pit_iqr {allp['pit_iqr'].mean():.3f} (want 0.500)  "
                  f"cvm {allp['cvm'].mean():.5f}  cover90 {allp['cover_90'].mean():.3f}  "
                  f"cover50 {allp['cover_50'].mean():.3f}  pit {allp['pit_mean'].mean():.3f}  "
                  f"bias {allp['median_bias'].mean():+.1f}  n {int(pooled['n'].sum())}")
    table = pl.concat(rows)

    # one scalar per position: the grid point whose PIT is the right *width*, and where the surface is
    # flat the smallest such scale. Parsimony is the right tie-break for an inflation factor -- it is
    # there to cover errors nobody measured, and a flat ridge is the data saying it cannot tell 1.5 from
    # 2.5, in which case claiming 2.5 is claiming an uncertainty that was not demonstrated.
    # Fitted on `_pit_iqr` and not on `_cvm`: see `_pit_iqr` for why the whole-PIT distance is the wrong
    # objective for a width. `cvm` is still carried through the table, as the verdict rather than the aim.
    TOL = 1.10
    chosen, ridge = {}, {}
    for pos in POSITIONS:
        sub = table.filter(pl.col("position") == pos)
        if sub.is_empty():
            chosen[pos] = 1.0
            continue
        by = sub.group_by("scale").agg(pl.col("pit_iqr").mean()).sort("scale")
        loss = np.abs(by["pit_iqr"].to_numpy() - 0.5)
        best = float(np.nanmin(loss))
        # `best` can be ~0, so the tolerance has to be an absolute floor as well as a ratio
        ok = np.flatnonzero(loss <= max(best * TOL, best + 0.01))
        chosen[pos] = float(by["scale"][int(ok[0])])
        ridge[pos] = [float(by["scale"][int(i)]) for i in ok]

    # what the chosen scale actually delivered, recorded beside it: an interval is a claim, and the app
    # says how well the claim held on seasons the model had not seen
    got = table.filter(pl.col("position") != "ALL").join(
        pl.DataFrame({"position": list(chosen), "scale": list(chosen.values())}),
        on=["position", "scale"], how="semi",
    )
    cover = {c: round(float((got[c] * got["n"]).sum() / max(int(got["n"].sum()), 1)), 3)
             for c in ("cover_90", "cover_50", "pit_mean", "pit_iqr", "below", "above",
                       "median_bias")} if not got.is_empty() else {}
    out = replace(disp, scale=chosen,
                  meta={**disp.meta, "calibrated": True,
                        "calibration_targets": list(targets),
                        "calibration_draws": draws,
                        # the population the coverage below is measured over, without which the number
                        # cannot be read: a coverage is a claim about a set of players
                        "calibration_min_projected": STARTABLE,
                        "coverage": cover,
                        # every scale the fit could not distinguish from the chosen one
                        "scale_ridge": ridge})
    return out, table


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def _dispersion_report(d: Dispersion) -> None:
    pl.Config.set_tbl_rows(50)
    print("\nTEAM  volume dispersion per channel: within a season, and how wrong the season is")
    print(pl.DataFrame([
        {"channel": c.name, "week_log_sd": d.k_team_week(c.name),
         "within_game_rho": d.k_team_rho(c.name), "season_log_sd": d.k_team_season(c.name)}
        for c in CHANNELS
    ]).with_columns(pl.col(pl.Float64).round(3)))

    print("\nPLAYER  the transient squared CV of a share, and the season error of it per tier")
    print(pl.DataFrame([
        {"position": p, "channel": c.name, "usage_cv2": d.k_usage(p, c.name),
         **{f"share_sd_{t}": d.k_share_season(p, t, c.name) for t in TIERS}}
        for p in POSITIONS for c in CHANNELS
    ]).with_columns(pl.col(pl.Float64).round(3)))

    print("\nRATES  per-event squared CV and the season error, per position, tier and group")
    print(pl.DataFrame([
        {"position": p, "group": g.name, "kind": "poisson" if g.count else "rate",
         "event_cv2": d.k_event(p, g.name),
         **{f"rate_sd_{t}": d.k_rate_season(p, t, g.name) for t in TIERS}}
        for p in POSITIONS for g in GROUPS
    ]).with_columns(pl.col(pl.Float64).round(3)))

    print("\nAVAILABILITY  the resampled actual/projected games ratio, and the bias divided out")
    print(pl.DataFrame([
        {"position": p, "tier": t, "n": len(d.games_sample(p, t)),
         "p5": float(np.quantile(d.games_sample(p, t), 0.05)),
         "p25": float(np.quantile(d.games_sample(p, t), 0.25)),
         "p50": float(np.quantile(d.games_sample(p, t), 0.50)),
         "p95": float(np.quantile(d.games_sample(p, t), 0.95)),
         "measured_bias": d.meta.get("games_bias", {}).get(f"{p}:{t}"),
         "boom": d.thresholds(p)[0], "bust": d.thresholds(p)[1],
         "scale": d.scale.get(p, 1.0)}
        for p in POSITIONS for t in TIERS
    ]).with_columns(pl.col(pl.Float64).round(3)))


def _report(season: int, settings: Settings, draws: int) -> None:
    from src.model import compose
    pl.Config.set_tbl_width_chars(240)
    pl.Config.set_fmt_float("mixed")
    d = load()
    _dispersion_report(d)
    if not d.meta.get("fitted"):
        print("\nNOT FITTED  the numbers above are the module's fallbacks; run --fit")

    wk = compose.weekly(season, settings)
    sim = run(wk, settings, d, draws=draws)
    print(f"\nSIMULATION  {season}, {sim.draws:,} draws, {sim.season.height} players")
    for pos in POSITIONS:
        print(f"\nTOP {pos}  by median")
        print(sim.season.filter(pl.col("position") == pos).head(12).select(
            "median_rank", "player", "team", pl.col("projected").round(1),
            pl.col("p50").round(1), pl.col("p5").round(1), pl.col("p95").round(1),
            pl.col("volatility").round(3), pl.col("boom_rate").round(3),
            pl.col("bust_rate").round(3), pl.col("games_p50").round(1),
            pl.col("ceiling_rank"), pl.col("floor_rank"),
        ))
    print("\nMEDIAN AGAINST THE POINT PROJECTION  per position; the gap is the redistribution of an "
          "absent player's pool, which a point projection cannot express")
    print(sim.season.group_by("position").agg(
        pl.len().alias("n"),
        pl.col("projected").sum().round(0).alias("projected_total"),
        pl.col("sim_mean").sum().round(0).alias("sim_total"),
        (100.0 * (pl.col("sim_mean").sum() / pl.col("projected").sum() - 1)).round(2)
        .alias("gap_pct"),
        pl.col("volatility").median().round(3).alias("median_volatility"),
    ).sort("position"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fit", action="store_true", help="measure dispersion and save it")
    p.add_argument("--calibrate", action="store_true",
                   help="scale the persistent sigmas to hit out-of-sample coverage, and save")
    p.add_argument("--season", type=int, default=PROJ_SEASON)
    p.add_argument("--draws", type=int, default=None)
    p.add_argument("--targets", type=int, nargs="+", default=None)
    a = p.parse_args(argv)

    ensure_dirs()
    settings = Settings()
    if a.fit:
        d = measure(tuple(range(HISTORY_FROM, LAST_COMPLETE_SEASON + 1)), settings)
        print(f"wrote {d.save()}")
        _dispersion_report(d)
    if a.calibrate:
        before = load()
        print(f"\nCALIBRATION  held-out seasons projected over {STARTABLE:.0f} points, at each scale")
        d, table = calibrate(tuple(a.targets) if a.targets else None, before,
                             draws=a.draws or 2000)
        print(f"\nchosen scale per position: {d.scale}")
        print(f"scales the fit could not tell apart: {d.meta.get('scale_ridge')}")
        print(f"wrote {d.save()}")
        print("\nCOVERAGE AT THE CHOSEN SCALE")
        print(table.filter(
            pl.col("scale").is_in([d.scale[p] for p in POSITIONS])
        ).group_by("position", "scale").agg(
            pl.col("n").sum(), pl.col("pit_iqr").mean().round(3), pl.col("cvm").mean().round(5),
            pl.col("cover_90").mean().round(3),
            pl.col("cover_50").mean().round(3), pl.col("below").mean().round(3),
            pl.col("above").mean().round(3), pl.col("median_bias").mean().round(1),
        ).sort("position"))
        print("  `pit_iqr` is what the scale was fitted on and wants 0.500; `median_bias` is what is "
              "left over, and it is the projection's to answer for rather than the interval's")
    if not (a.fit or a.calibrate):
        _report(a.season, settings, a.draws or settings.simulation_draws)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
