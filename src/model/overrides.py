"""The adjustment layer: a user's edits, what they were before, and the projection that follows.

The workbook got one thing exactly right -- `BASE`, `x Adj`, `USED` in three adjacent cells, so an
edit was always visible as an edit. This module is that triplet for an engine with no cells in it.

Three levels, because that is where the arithmetic actually has seams:

- **league** -- scoring, normalisation, the per-game chain, the market weight, and `k_scale`, which
  multiplies every fitted shrinkage constant. `k_scale = 0` is a player's own history at face value
  and a large one is his depth slot alone; those are the two ablations the backtest scored, so a user
  can reproduce the experiment rather than take its word for it.
- **team** -- any per-game number in the team environment, for the whole season or for one week. Pace
  and pools are the useful knobs; edit `plays` or `dropbacks` and the designed-run count derives from
  the edit rather than contradicting it.
- **player** -- his availability, each of his shares, each of his rates. Set a number or multiply it.

**An override is applied, never assumed.** Every one comes back in `provenance` with the value it was
recorded against, the value it actually found when it ran, and the value it produced. A field that no
longer exists is reported as unapplied rather than silently dropped, because a scenario written in
August should say so in November instead of quietly meaning nothing.

**Overriding a share does not exempt it from the pool.** Push a receiver to a 30% target share with
normalisation on and his teammates are scaled down to keep the team's targets whole -- which is the
point of the pool, and visible in the audit on the team page. With it off he simply gets 30% of a
team that now throws more than it was projected to. Both are defensible; neither is silent.

    python -m src.model.overrides --demo        # what one team edit and one player edit do
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.config import PROJ_SEASON, REG_WEEKS, SCENARIOS, Settings, ensure_dirs
from src.model import compose, efficiency, opportunity, priors, roster, team

BASELINE = "baseline"          # the reserved name for "no edits at all"
MODES = ("set", "multiply")

# Settings fields a scenario is allowed to patch. Not every field: `status_availability` is a stated
# assumption rather than a knob, and `simulation_draws` belongs to the simulation.
LEAGUE_FIELDS = ("normalize_pools", "use_context_factors", "market_weight", "schedule_renormalise",
                 "context_k", "team_weight_recent", "team_keep_vs_mean", "games_projected",
                 "tier_size", "recency")

# Per-game team numbers worth editing: the pools the players divide, and the rates their efficiency
# is scaled by. Deliberately not every column in the environment -- editing `points` alone would move
# nothing downstream, and offering a knob that does nothing is worse than not offering it.
TEAM_FIELDS = ("plays", "dropbacks", "pass_attempts", "carries", "targets", "air_yards",
               "late_down_targets", "pass_tds", "rush_tds", "red_zone_trips", "red_zone_targets",
               "red_zone_carries", "inside_5_carries", "short_yardage_carries", "seconds_per_play",
               "yards_per_attempt", "yards_per_carry", "success_rate", "implied_points")

# `expected_games` is the availability knob; `active_weeks` is derived from it rather than edited, so
# the two cannot be set to disagree.
AVAILABILITY_FIELDS = ("expected_games",)
# Deduplicated because the two frames overlap: `snap_share` and the participation metrics are both a
# player's own number and a share of a non-exclusive pool, so they appear in each list.
PLAYER_FIELDS = tuple(dict.fromkeys(
    AVAILABILITY_FIELDS + roster.PARTICIPATION_METRICS
    + opportunity.SHARE_METRICS + efficiency.RATE_METRICS
))
FIELDS = {"league": LEAGUE_FIELDS, "team": TEAM_FIELDS, "player": PLAYER_FIELDS}

DEFAULT_SCORING = "ppr"        # `Scoring()` itself, so selecting it is not an edit


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def league_defaults() -> dict[str, Any]:
    """What the league knobs are worth with nothing overridden."""
    base = Settings()
    return {k: getattr(base, k) for k in LEAGUE_FIELDS}


def _typed(value: Any, default: Any) -> Any:
    """A JSON round-trip turns `recency`'s tuple into a list. Put it back before comparing or using it."""
    if isinstance(default, tuple) and isinstance(value, list):
        return tuple(value)
    return value


def clean_league(values: dict[str, Any]) -> dict[str, Any]:
    """Only the knobs that actually differ from the engine's own default.

    A scenario that records `normalize_pools = True` is not the baseline any more -- different digest,
    different cache entry, an "edited" badge in the sidebar -- even though it means nothing. So a knob
    set back to its default is dropped rather than stored, and the baseline stays the baseline.
    """
    d = league_defaults()
    return {k: _typed(v, d[k]) for k, v in values.items()
            if k in d and _typed(v, d[k]) != d[k]}


# --------------------------------------------------------------------------- #
# the data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Override:
    """One edit. `base` is what it was worth when the edit was made, kept for the diff."""

    level: str                      # "team" | "player"
    key: str                        # team code or player_id
    field: str                      # the column being edited
    mode: str = "set"               # "set" an absolute value or "multiply" the projection
    value: float = 1.0
    week: int | None = None         # team edits only; None means all seventeen games
    base: float | None = None       # recorded at edit time, not at apply time
    note: str = ""
    at: str = ""

    def __post_init__(self) -> None:
        if self.level not in ("team", "player"):
            raise ValueError(f"level must be team or player, got {self.level!r}")
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    @property
    def id(self) -> tuple[str, str, str, int | None]:
        """What makes two edits the same edit, so setting one twice replaces rather than stacks."""
        return (self.level, self.key, self.field, self.week)

    @property
    def label(self) -> str:
        where = f"{self.key} wk{self.week}" if self.week is not None else self.key
        op = "=" if self.mode == "set" else "x"
        return f"{where} {self.field} {op} {self.value:g}"

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "key": self.key, "field": self.field, "mode": self.mode,
                "value": self.value, "week": self.week, "base": self.base, "note": self.note,
                "at": self.at}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Override:
        return cls(**{k: d.get(k) for k in
                      ("level", "key", "field", "mode", "value", "week", "base", "note", "at")
                      if d.get(k) is not None or k in ("week", "base")})


@dataclass(frozen=True)
class Scenario:
    """A named set of edits: the league patch, the shrinkage scale, and every team and player edit.

    Two scenarios with the same content have the same `digest`, which is what the app caches on. That
    is the whole reason this is a value object rather than mutable state: an edit produces a new
    scenario, and a projection is a pure function of one.
    """

    name: str = BASELINE
    scoring: str | None = None                       # a preset name, None means leave it alone
    league: dict[str, Any] = field(default_factory=dict)
    k_scale: float = 1.0
    items: tuple[Override, ...] = ()
    created: str = ""
    updated: str = ""

    # ---- content -------------------------------------------------------- #
    @property
    def is_baseline(self) -> bool:
        return not self.items and not self.league and self.scoring is None and self.k_scale == 1.0

    def at_level(self, level: str) -> tuple[Override, ...]:
        return tuple(o for o in self.items if o.level == level)

    def touching(self, level: str, key: str) -> tuple[Override, ...]:
        return tuple(o for o in self.items if o.level == level and o.key == key)

    def set(self, *edits: Override) -> Scenario:
        """Add or replace edits, keyed on `Override.id`, newest wins."""
        by_id = {o.id: o for o in self.items}
        for edit in edits:
            if edit.field not in FIELDS[edit.level]:
                raise ValueError(f"{edit.field!r} is not editable at the {edit.level} level")
            by_id[edit.id] = replace(edit, at=edit.at or _now())
        return replace(self, items=tuple(by_id.values()), updated=_now())

    def clear(self, level: str | None = None, key: str | None = None,
              field_name: str | None = None, week: int | None = None) -> Scenario:
        """Reset: everything, one level, one team or player, or one knob. Per-knob is the point."""
        def keep(o: Override) -> bool:
            return not (
                (level is None or o.level == level)
                and (key is None or o.key == key)
                and (field_name is None or o.field == field_name)
                and (week is None or o.week == week)
            )
        return replace(self, items=tuple(o for o in self.items if keep(o)), updated=_now())

    def rename(self, name: str) -> Scenario:
        """A new name and nothing else. The digest is unchanged, so a rename costs no recomputation."""
        return replace(self, name=name or BASELINE)

    def patch_league(self, **values: Any) -> Scenario:
        """Merge league knobs in. Anything left at its default is dropped, so `is_baseline` holds."""
        bad = [k for k in values if k not in LEAGUE_FIELDS and k not in ("scoring", "k_scale")]
        if bad:
            raise ValueError(f"not league-editable: {bad}")
        scoring = values.pop("scoring", self.scoring)
        if scoring == DEFAULT_SCORING:
            scoring = None
        k_scale = float(values.pop("k_scale", self.k_scale))
        return replace(self, scoring=scoring, k_scale=k_scale,
                       league=clean_league({**self.league, **values}), updated=_now())

    # ---- serialisation -------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "scoring": self.scoring, "league": self.league,
                "k_scale": self.k_scale, "created": self.created, "updated": self.updated,
                "overrides": [o.to_dict() for o in self.items]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Scenario:
        return cls(
            name=d.get("name") or BASELINE,
            scoring=d.get("scoring"),
            league=dict(d.get("league") or {}),
            k_scale=float(d.get("k_scale", 1.0)),
            items=tuple(Override.from_dict(o) for o in d.get("overrides") or ()),
            created=d.get("created", ""),
            updated=d.get("updated", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def content_json(self) -> str:
        """The scenario without its name or timestamps: canonical, and safe as a cache key.

        Renaming a scenario must not throw away a projection, and saving one twice without editing it
        must not either, so what the app caches on is the content and nothing else.
        """
        body = {k: v for k, v in self.to_dict().items()
                if k not in ("name", "created", "updated")}
        body["overrides"] = sorted(
            ({k: v for k, v in o.items() if k not in ("note", "at", "base")}
             for o in body["overrides"]),
            key=lambda o: (o["level"], o["key"] or "", o["field"], o["week"] if o["week"] else -1),
        )
        return json.dumps(body, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Scenario:
        return cls.from_dict(json.loads(text))

    @property
    def digest(self) -> str:
        """A short stable hash of `content_json`, for labelling a run in the UI and in a filename."""
        return hashlib.sha1(self.content_json().encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# scenarios on disk
# --------------------------------------------------------------------------- #
def path(name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name).strip() or "unnamed"
    return SCENARIOS / f"{safe}.json"


def names() -> list[str]:
    if not SCENARIOS.is_dir():
        return []
    out = []
    for p in sorted(SCENARIOS.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")).get("name") or p.stem)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def save(scenario: Scenario) -> Path:
    ensure_dirs()
    now = _now()
    sc = replace(scenario, created=scenario.created or now, updated=now)
    p = path(sc.name)
    p.write_text(sc.to_json(), encoding="utf-8")
    return p


def load(name: str) -> Scenario:
    p = path(name)
    if not p.is_file():
        return Scenario(name=name)
    return Scenario.from_json(p.read_text(encoding="utf-8"))


def delete(name: str) -> bool:
    p = path(name)
    if p.is_file():
        p.unlink()
        return True
    return False


# --------------------------------------------------------------------------- #
# applying an edit
# --------------------------------------------------------------------------- #
def settings_for(scenario: Scenario, settings: Settings | None = None) -> Settings:
    """The league patch as a `Settings`. Everything else in the engine reads this one object."""
    st = settings or Settings()
    if scenario.league:
        d = league_defaults()
        st = replace(st, **{k: _typed(v, d[k]) for k, v in scenario.league.items()
                            if k in LEAGUE_FIELDS})
    if scenario.scoring:
        st = st.with_scoring(scenario.scoring)
    return st


def fitted_for(scenario: Scenario, fitted: priors.Fitted | None = None) -> priors.Fitted:
    """The fitted artifacts with `k_scale` applied to every shrinkage constant.

    Every metric is written out rather than scaled in place: a name missing from the dict would fall
    back to the `Settings` default and quietly escape the scale.
    """
    f = fitted or priors.fitted_saved()
    if scenario.k_scale == 1.0:
        return f
    st = Settings()
    scaled = {
        m.name: float(f.k.get(m.name, st.default_share_k if m.kind == "share" else st.default_rate_k))
        * scenario.k_scale
        for m in priors.METRICS
    }
    return replace(f, k=scaled)


def _one(frame: pl.DataFrame, o: Override, key_col: str, week_col: str | None) -> tuple[pl.DataFrame, dict]:
    """Apply a single edit and report what it did, including when it did nothing."""
    record = {"level": o.level, "key": o.key, "field": o.field, "week": o.week, "mode": o.mode,
              "value": float(o.value), "base_recorded": o.base, "note": o.note, "at": o.at,
              "base_now": None, "used_now": None, "rows": 0, "applied": False, "reason": ""}
    if o.field not in frame.columns:
        record["reason"] = "no such column in this frame"
        return frame, record
    mask = pl.col(key_col) == o.key
    if o.week is not None:
        if week_col is None or week_col not in frame.columns:
            record["reason"] = "this frame has no weeks to target"
            return frame, record
        mask = mask & (pl.col(week_col) == o.week)
    hit = frame.filter(mask)
    if hit.is_empty():
        record["reason"] = f"no rows for {o.key}"
        return frame, record

    col = pl.col(o.field).cast(pl.Float64)
    new = pl.lit(float(o.value)) if o.mode == "set" else col * float(o.value)
    out = frame.with_columns(pl.when(mask).then(new).otherwise(col).alias(o.field))
    record.update(
        base_now=float(hit[o.field].cast(pl.Float64).mean()),
        used_now=float(out.filter(mask)[o.field].mean()),
        rows=hit.height,
        applied=True,
    )
    return out, record


def apply(
    frame: pl.DataFrame,
    scenario: Scenario,
    level: str,
    key_col: str,
    week_col: str | None = None,
    fields: tuple[str, ...] | None = None,
) -> tuple[pl.DataFrame, list[dict]]:
    """Every edit at `level` whose field is in this frame, applied in the order they were made.

    `fields` narrows it further, which is how the same player edit lands on the frame that owns it:
    availability on the participation frame, shares on the share frame, rates on the rate frame.
    """
    log: list[dict] = []
    out = frame
    for o in scenario.at_level(level):
        if fields is not None and o.field not in fields:
            continue
        out, record = _one(out, o, key_col, week_col)
        log.append(record)
    return out, log


def _rederive_availability(part: pl.DataFrame) -> pl.DataFrame:
    """`active_weeks` follows `expected_games`, so an edit to games cannot leave them disagreeing."""
    if "expected_games" not in part.columns:
        return part
    return part.with_columns(
        pl.col("expected_games").clip(0.0, float(REG_WEEKS - 1)).alias("expected_games")
    ).with_columns(
        (pl.col("expected_games") / float(REG_WEEKS - 1)).clip(0.0, 1.0).alias("active_weeks")
    )


# --------------------------------------------------------------------------- #
# the projection under a scenario
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Run:
    """Every frame a page needs, all of them produced under one scenario.

    One object because the alternative is a page reading an edited share frame beside an unedited
    environment, which would be a projection nobody ran.
    """

    scenario: Scenario
    season: int
    settings: Settings
    roster: pl.DataFrame
    part: pl.DataFrame
    shares: pl.DataFrame
    rates: pl.DataFrame
    env: pl.DataFrame
    opp: pl.DataFrame
    weekly: pl.DataFrame
    season_frame: pl.DataFrame
    board: pl.DataFrame
    provenance: pl.DataFrame

    @property
    def digest(self) -> str:
        return self.scenario.digest


PROVENANCE_SCHEMA = {
    "level": pl.String, "key": pl.String, "field": pl.String, "week": pl.Int64, "mode": pl.String,
    "value": pl.Float64, "base_recorded": pl.Float64, "base_now": pl.Float64,
    "used_now": pl.Float64, "rows": pl.Int64, "applied": pl.Boolean, "reason": pl.String,
    "note": pl.String, "at": pl.String, "stage": pl.String,
}


def run(
    scenario: Scenario | None = None,
    season: int = PROJ_SEASON,
    prev: pl.DataFrame | None = None,
) -> Run:
    """Project `season` with every edit in `scenario` applied at the point it belongs.

    The order is the engine's own: the league patch first because everything downstream reads it,
    then availability, then shares and rates, then the team environment, and only then the pools --
    so a share edit is still subject to normalisation and a team edit still divides among the players
    who were there to receive it.
    """
    sc = scenario or Scenario()
    st = settings_for(sc)
    fitted = fitted_for(sc)
    log: list[dict] = []

    def note(stage: str, records: list[dict]) -> None:
        log.extend({**r, "stage": stage} for r in records)

    ros = roster.roster(season)
    part = roster.participation(season, st, ros=ros, fitted=fitted)
    part, rec = apply(part, sc, "player", "player_id", fields=AVAILABILITY_FIELDS)
    note("availability", rec)
    part = _rederive_availability(part)
    # participation metrics live in both frames: the pools read the share frame, the team page reads
    # this one, and an edit that landed on only one of them would show a number the pool never used.
    part, rec = apply(part, sc, "player", "player_id", fields=roster.PARTICIPATION_METRICS)
    note("participation", rec)

    shares = opportunity.player_shares(season, st, ros=ros, fitted=fitted)
    shares, rec = apply(shares, sc, "player", "player_id", fields=opportunity.SHARE_METRICS)
    note("shares", rec)

    rates = efficiency.rates(season, st, ros=ros, fitted=fitted)
    rates, rec = apply(rates, sc, "player", "player_id", fields=efficiency.RATE_METRICS)
    note("rates", rec)

    env = team.game_environment(season, st)
    env, rec = apply(env, sc, "team", "team", week_col="week", fields=TEAM_FIELDS)
    note("environment", rec)

    opp = opportunity.opportunity(season, st, shares=shares, part=part, fitted=fitted, env=env)
    wk = compose.weekly(season, st, opp=opp, player_rates=rates)
    yr = compose.seasonal(wk, season, st)
    board = compose.board(yr, st, prev)

    prov = pl.DataFrame(log, schema=PROVENANCE_SCHEMA) if log else pl.DataFrame(schema=PROVENANCE_SCHEMA)
    return Run(scenario=sc, season=season, settings=st, roster=ros, part=part, shares=shares,
               rates=rates, env=env, opp=opp, weekly=wk, season_frame=yr, board=board,
               provenance=prov)


# --------------------------------------------------------------------------- #
# what changed
# --------------------------------------------------------------------------- #
DIFF_COLUMNS = ("fantasy_points", "points_per_game", "games", "targets", "carries", "receptions",
                "receiving_yards", "rushing_yards", "attempts", "passing_yards", "passing_tds",
                "receiving_tds", "rushing_tds", "position_rank", "overall_rank")


def diff(base: pl.DataFrame, scenario_board: pl.DataFrame, threshold: float = 0.05) -> pl.DataFrame:
    """Player-level difference between two boards, biggest move first.

    The table that answers "what did my edit actually do", including to everybody the edit was not
    about: pushing one receiver's share up moves his whole receiving room, and that consequence is
    the most important thing on the page. `threshold` is in fantasy points and only hides rounding.
    """
    cols = [c for c in DIFF_COLUMNS if c in base.columns and c in scenario_board.columns]
    a = base.select("player_id", "player", "position", "team", *cols)
    b = scenario_board.select(
        "player_id", *[pl.col(c).alias(f"new_{c}") for c in cols],
        pl.col("team").alias("new_team"),
    )
    out = a.join(b, on="player_id", how="full", coalesce=True).with_columns(
        *[(pl.col(f"new_{c}") - pl.col(c)).alias(f"d_{c}") for c in cols]
    )
    moved = (pl.col("d_fantasy_points").abs() >= threshold)
    appeared = pl.col("fantasy_points").is_null() | pl.col("new_fantasy_points").is_null()
    return out.filter(moved | appeared).sort(
        pl.col("d_fantasy_points").abs(), descending=True, nulls_last=True
    )


def summary(base: pl.DataFrame, scenario_board: pl.DataFrame,
            threshold: float = 0.05) -> pl.DataFrame:
    """How far the board moved in total, per position: the headline above the diff table.

    `net` against `abs` is the reading worth making. A team edit that adds volume shows both moving
    together; a share edit inside one receiving room shows a large `abs` and a `net` near zero,
    because normalisation took from the teammates what it gave to the player.
    """
    d = diff(base, scenario_board, threshold=threshold)
    return (
        d.group_by("position").agg(
            pl.len().alias("players_moved"),
            pl.col("d_fantasy_points").abs().sum().alias("abs_points_moved"),
            pl.col("d_fantasy_points").sum().alias("net_points_moved"),
            pl.col("d_fantasy_points").max().alias("biggest_gain"),
            pl.col("d_fantasy_points").min().alias("biggest_loss"),
        ).sort("abs_points_moved", descending=True)
    )


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def demo_scenario(player_id: str, team: str = "PHI") -> Scenario:
    """One edit of each kind, so the report and the tests have something real to move.

    The player is passed in rather than hardcoded: an id that has left the league would make this a
    scenario of four edits that do nothing, which is exactly the failure the provenance table exists
    to catch and a poor thing to demonstrate it with.
    """
    return Scenario(name="demo").set(
        Override("team", team, "targets", "multiply", 1.10, note="more dropbacks than projected"),
        Override("team", team, "carries", "set", 18.0, week=1, note="one week only"),
        Override("player", player_id, "target_share", "set", 0.30, note="alpha receiver"),
        Override("player", player_id, "expected_games", "set", 17.0),
    )


def _report(season: int, scenario: Scenario | None) -> None:
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_rows(30)
    pl.Config.set_fmt_float("mixed")

    base = run(Scenario(), season)
    if scenario is None:
        top = base.board.filter(pl.col("position") == "WR").row(0, named=True)
        scenario = demo_scenario(top["player_id"], top["team"])
        print(f"demo: {top['player']} of {top['team']}, and {top['team']}'s own volume")
    got = run(scenario, season)

    print(f"\nSCENARIO {scenario.name!r}  digest {scenario.digest}  "
          f"{len(scenario.items)} edits  k_scale {scenario.k_scale:g}")
    print(got.provenance.select("stage", "level", "key", "field", "week", "mode", "value",
                                pl.col("base_now").round(4), pl.col("used_now").round(4),
                                "rows", "applied", "reason"))

    print("\nBOARD MOVEMENT")
    print(summary(base.board, got.board).with_columns(pl.col("^.*points.*$").round(1)))

    d = diff(base.board, got.board)
    print(f"\nBIGGEST MOVES  {d.height} players moved by more than 0.05 points")
    print(d.select("player", "position", "team", pl.col("fantasy_points").round(1),
                   pl.col("new_fantasy_points").round(1), pl.col("d_fantasy_points").round(1),
                   pl.col("d_targets").round(1), pl.col("d_carries").round(1),
                   "position_rank", "new_position_rank").head(20))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, default=PROJ_SEASON)
    p.add_argument("--scenario", help="a saved scenario name")
    p.add_argument("--demo", action="store_true", help="one edit of each kind")
    p.add_argument("--list", action="store_true", help="the scenarios on disk")
    a = p.parse_args(argv)
    if a.list:
        for name in names() or ["(none saved)"]:
            print(name)
        return 0
    _report(a.season, None if a.demo or not a.scenario else load(a.scenario))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
