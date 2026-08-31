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
  For the whole season, or for one game: an edit carrying a `week` is applied to that week alone, on the
  player-game frame, before the team's pools are divided. So "he is out in week 5" hands his work to the
  room in week 5 and promotes the backup quarterback in week 5, and the other sixteen games are
  untouched. A per-game edit and a season edit on the same field are different edits and both are kept.
- **depth** -- where he sits on the chart. Not a number like the others and it is worth saying why it is
  still an override: `depth_slot` is an *ordering*, so setting one man's slot moves everybody behind him
  and re-prices all of them. It is applied to the roster frame before anything is estimated, so a
  promoted backup is priced as a starter -- his slot's prior, his slot's survival rate, his place in the
  quarterback queue -- rather than being handed a starter's share while still being priced as a backup.
- **roster** -- whether he is on this team at all. `on_roster = 0` is a release, and it is the only edit
  that *removes a row* rather than changing a number in one: it is applied before the chart is settled,
  so his room renumbers around the hole, the men behind him are re-priced as what they moved up to, his
  share of every pool is divided among the players who are left, and nothing downstream -- board,
  weekly, exports, simulation -- has a row for him to be counted in. That is the difference between a
  release and zeroing a man's availability: zero games still leaves him holding his slot in the room.

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

# The name a scenario gets the moment it stops being the baseline, so that autosave has somewhere to
# write without asking anybody to name their work first. A session spent going team by team is hours
# of typing, and the failure it must not have is "the browser reloaded and it is all gone".
WORKING = "working"

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

# Where he sits on the chart. Kept out of `PLAYER_FIELDS` on purpose: every name in that tuple is a
# number applied to a column in place, and this one is a position in an ordering that renumbers the
# whole room. It is a player edit -- same level, same scenario, same provenance -- applied at its own
# stage, and `DEPTH_FIELDS` is how the two are told apart.
DEPTH_FIELD = "depth_slot"
DEPTH_FIELDS = (DEPTH_FIELD,)

# Whether he is on this roster at all. Kept apart from every other field for the same reason as
# `depth_slot` and then one more: this is the only edit that takes a row out of the projection instead
# of changing a number in it, so it is applied first, before the chart is settled and before anything is
# estimated off it. `0` is the only value that means anything -- there is no half a roster spot -- and
# the release is a season-long fact, so a single week is `p_play = 0` and not this.
ROSTER_FIELD = "on_roster"
ROSTER_FIELDS = (ROSTER_FIELD,)
OFF_ROSTER = 0.0

# Deduplicated because the two frames overlap: `snap_share` and the participation metrics are both a
# player's own number and a share of a non-exclusive pool, so they appear in each list.
PLAYER_FIELDS = tuple(dict.fromkeys(
    AVAILABILITY_FIELDS + roster.PARTICIPATION_METRICS
    + opportunity.SHARE_METRICS + efficiency.RATE_METRICS
))

# His chance of playing in one particular game. Only ever a per-game edit, which is why it is not in
# `PLAYER_FIELDS`: over a season the knob is `expected_games`, a count of games, and a count cannot say
# *which* games. `p_play` is how "he is out in week 5" is expressed, and a season-wide `p_play` edit
# would be a second, contradicting way to say what `expected_games` already says.
GAME_ONLY_FIELDS = ("p_play",)

# What a player edit can say about one game: whether he is out there, and what his job is when he is.
# Both live on the frame `opportunity` divides the pools from, so a per-game edit is subject to
# normalisation and to the quarterback queue exactly as the season-level one is.
GAME_FIELDS = GAME_ONLY_FIELDS + opportunity.SHARE_METRICS

# The rates are per-game too, but on the next frame along: they are joined to his games and multiplied
# by that game's environment factor, so an edit belongs after the join and before the stat line.
GAME_RATE_FIELDS = efficiency.RATE_METRICS

# Everything a week can be attached to. A player field outside this set is one number for the season.
GAME_ALL_FIELDS = tuple(dict.fromkeys(GAME_FIELDS + GAME_RATE_FIELDS))
FIELDS = {"league": LEAGUE_FIELDS, "team": TEAM_FIELDS,
          "player": PLAYER_FIELDS + DEPTH_FIELDS + ROSTER_FIELDS + GAME_ONLY_FIELDS}

DEFAULT_SCORING = "ppr"        # `Scoring()` itself, so selecting it is not an edit


def _now() -> str:
    """An ISO stamp with microseconds, because it is sorted on as well as read.

    Second resolution was enough while this was only ever displayed -- and wrong the moment `names()`
    started ordering saved scenarios by it. Two saves inside one second is normal while typing, and on
    Windows the file's own mtime ticks every ~16ms, so both the stamp and the mtime tie and the order
    falls back to the alphabet. Nothing reading this needs the truncation: the sidebar slices out `HH:MM`.
    """
    return datetime.now(UTC).isoformat()


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
    week: int | None = None         # one game; None means every game of the season
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
            if edit.field in GAME_ONLY_FIELDS and edit.week is None:
                # refused rather than reported: over a whole season this is `expected_games`, and two
                # ways to say the same thing is two ways for them to disagree
                raise ValueError(f"{edit.field!r} is a per-game edit and needs a week")
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


# Bookkeeping the app writes for itself, not scenarios anybody named. Kept in the same directory so
# there is one place to look, and hidden from `names()` so it never appears as something to load.
# A function rather than a constant because `SCENARIOS` is what a test redirects, and a path resolved
# at import time would keep writing to the real directory while the test believed otherwise.
POINTER_NAME = "_last.json"


def pointer_path() -> Path:
    return SCENARIOS / POINTER_NAME


def names() -> list[str]:
    """Every named scenario on disk, newest-touched first, excluding the app's own bookkeeping."""
    if not SCENARIOS.is_dir():
        return []
    found: list[tuple[str, float, str]] = []
    for p in sorted(SCENARIOS.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            when = p.stat().st_mtime
        except (OSError, json.JSONDecodeError):
            continue
        # `updated` carries microseconds, so it orders two saves made seconds apart while typing. The
        # file's own timestamp is still the tie-break, for a scenario written by an older build whose
        # stamp stops at the second
        found.append((str(d.get("updated") or ""), when, d.get("name") or p.stem))
    # newest first, because the one being worked on is the one wanted, and a preseason pass across the
    # league produces enough saved scenarios that alphabetical stops being a useful order
    return [name for _, _, name in sorted(found, key=lambda r: (r[0], r[1]), reverse=True)]


def summaries() -> list[dict[str, Any]]:
    """`names()` with the size and age of each, for a picker that has to say what it is offering."""
    out: list[dict[str, Any]] = []
    for name in names():
        try:
            d = json.loads(path(name).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # "overrides" is what `Scenario.to_dict` calls the list; `items` is the attribute name only
        out.append({"name": d.get("name") or name, "edits": len(d.get("overrides") or ()),
                    "league_knobs": len(d.get("league") or {}), "created": d.get("created") or "",
                    "updated": d.get("updated") or ""})
    return out


def save(scenario: Scenario) -> Path:
    ensure_dirs()
    now = _now()
    sc = replace(scenario, created=scenario.created or now, updated=now)
    p = path(sc.name)
    p.write_text(sc.to_json(), encoding="utf-8")
    remember(sc.name)
    return p


def remember(name: str) -> None:
    """Record which scenario was last written, so a new session can pick the work back up.

    Deliberately a pointer rather than a copy: two files that both claim to be the scenario is a way
    for them to disagree, and the one on disk under its own name is the one that matters.
    """
    ensure_dirs()
    try:
        pointer_path().write_text(json.dumps({"name": name, "at": _now()}), encoding="utf-8")
    except OSError:
        pass                       # a pointer that cannot be written is not worth failing an edit over


def last_used() -> str | None:
    """The name recorded by `remember`, if that scenario is still on disk. `None` otherwise."""
    try:
        name = json.loads(pointer_path().read_text(encoding="utf-8")).get("name")
    except (OSError, json.JSONDecodeError):
        return None
    if not name or not path(str(name)).is_file():
        return None
    return str(name)


def load(name: str) -> Scenario:
    p = path(name)
    if not p.is_file():
        return Scenario(name=name)
    return Scenario.from_json(p.read_text(encoding="utf-8"))


def delete(name: str) -> bool:
    p = path(name)
    if p.is_file():
        p.unlink()
        if last_used() is None or last_used() == name:
            # the pointer would otherwise send the next session to a file that is no longer there
            pointer_path().unlink(missing_ok=True)
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


def _record(o: Override) -> dict:
    """The provenance row for one edit before anything has been tried: applied nothing, changed nothing."""
    return {"level": o.level, "key": o.key, "field": o.field, "week": o.week, "mode": o.mode,
            "value": float(o.value), "base_recorded": o.base, "note": o.note, "at": o.at,
            "base_now": None, "used_now": None, "rows": 0, "applied": False, "reason": ""}


def _one(frame: pl.DataFrame, o: Override, key_col: str, week_col: str | None) -> tuple[pl.DataFrame, dict]:
    """Apply a single edit and report what it did, including when it did nothing."""
    record = _record(o)
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

    was = hit[o.field].cast(pl.Float64).mean()
    if was is None:
        # The rows exist and the number does not: a receiver has a row in the rate frame and no
        # completion percentage in it. Reported unapplied rather than written, because there is
        # nothing to multiply, nothing to record as a base, and no reading of the season in which a
        # receiver throws -- and because a set on a null cell used to reach `float(None)` below.
        record["reason"] = "no number for this player to change"
        return frame, record

    col = pl.col(o.field).cast(pl.Float64)
    new = pl.lit(float(o.value)) if o.mode == "set" else col * float(o.value)
    out = frame.with_columns(pl.when(mask).then(new).otherwise(col).alias(o.field))
    record.update(
        base_now=float(was),
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
    weeks: str = "any",
) -> tuple[pl.DataFrame, list[dict]]:
    """Every edit at `level` whose field is in this frame, applied in the order they were made.

    `fields` narrows it further, which is how the same player edit lands on the frame that owns it:
    availability on the participation frame, shares on the share frame, rates on the rate frame.

    `weeks` says which edits a stage owns -- `"any"`, `"only"` the ones carrying a week, or `"never"`
    those. A player's share is one number for the season and his share in week 5 is a different edit on
    a different frame, so the season stages take `"never"` and the per-game stages `"only"`. Without the
    split the season stage would report a week-scoped edit unapplied, the game stage would then apply it,
    and the provenance log would contradict itself about the same edit.
    """
    if weeks not in ("any", "only", "never"):
        raise ValueError(f"weeks must be any, only or never, got {weeks!r}")
    log: list[dict] = []
    out = frame
    for o in scenario.at_level(level):
        if fields is not None and o.field not in fields:
            continue
        if (weeks == "only" and o.week is None) or (weeks == "never" and o.week is not None):
            continue
        out, record = _one(out, o, key_col, week_col)
        log.append(record)
    return out, log


def _season_grain(scenario: Scenario) -> list[dict]:
    """Week-scoped player edits on a field no per-game frame carries, reported rather than dropped.

    `depth_slot` is the case this exists for. A chart is not a per-game thing -- it is the row every
    prior and the whole quarterback queue are read off, settled before anything is estimated -- so
    "week 5, depth 1" is not a claim the engine can act on. Saying so is the point: an edit that
    quietly means nothing is the failure this module exists to prevent, and the stages that skip it
    are silent by design.
    """
    out = []
    for o in scenario.at_level("player"):
        if o.week is None or o.field in GAME_ALL_FIELDS:
            continue
        rec = _record(o)
        rec["reason"] = (
            "a roster spot is not a per-game thing -- for one week, set `p_play` to 0 in that week"
            if o.field == ROSTER_FIELD else
            "this field is one number for the whole season, so a single week cannot be targeted"
        )
        out.append(rec)
    return out


def _renumber(ros: pl.DataFrame) -> pl.DataFrame:
    """Close the gaps a release left, and re-derive everything keyed on a slot.

    A room reads 1, 2, 3 with nobody sharing a number, so taking the second back out of it has to make
    the third back the second -- and `slot_bucket` and `avail_slot` have to follow, because they are the
    rows the priors and the survival curve are read off. A no-op on a room that lost nobody: the slots
    are already 1..n in order, so counting them back out returns what was there.
    """
    out = (
        ros.sort(["team", "position", DEPTH_FIELD, "player_id"])
        .with_columns(pl.col("player_id").cum_count().over(["team", "position"])
                      .cast(pl.Int32).alias(DEPTH_FIELD))
    )
    cap = pl.col("position").replace_strict(priors.SLOT_CAP, default=4, return_dtype=pl.Int32)
    return out.with_columns(
        pl.min_horizontal(DEPTH_FIELD, cap).alias("slot_bucket"),
        pl.min_horizontal(DEPTH_FIELD, pl.lit(roster.AVAIL_SLOT_CAP, pl.Int32)).alias("avail_slot"),
    )


def _release(ros: pl.DataFrame, scenario: Scenario) -> tuple[pl.DataFrame, list[dict]]:
    """Take released players off the roster frame, before a single number is estimated from it.

    The first stage of the run, and the only one that changes the *shape* of what follows. Everything
    downstream is built from this frame -- participation, the shares that divide the pools, the rates,
    the per-game grid, the board, the exports, the simulation -- so a man dropped here is not projected
    small, he is not projected at all, and the pool he was holding is divided among the men who remain.

    Two refusals rather than a guess. A roster spot cannot be multiplied, and a room cannot be emptied:
    a team whose quarterback room is empty has dropbacks with nobody to divide them, which is not a
    projection of anything, so the last man in a room stays and the log says why.
    """
    asks: list[tuple[dict, str]] = []
    wanted: list[str] = []
    for o in scenario.at_level("player"):
        if o.field != ROSTER_FIELD or o.week is not None:
            # a week-scoped release is reported by `_season_grain`, not here, so that one edit is never
            # both applied and unapplied in the same log
            continue
        rec = _record(o)
        if o.mode != "set":
            rec["reason"] = "a roster spot is set, not multiplied"
        elif float(o.value) != OFF_ROSTER:
            rec["reason"] = f"{ROSTER_FIELD} is {OFF_ROSTER:g} to release him; anything else is what " \
                            "the roster already says, so there is nothing to apply"
        else:
            wanted.append(o.key)
        asks.append((rec, o.key))
    if not wanted:
        return ros, [rec for rec, _ in asks]

    here = set(ros["player_id"].to_list())
    rooms: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for pid, tm, pos, slot in zip(ros["player_id"], ros["team"], ros["position"],
                                  ros[DEPTH_FIELD], strict=True):
        rooms.setdefault((tm, pos), []).append((int(slot), pid))

    asked = set(wanted)
    kept_back: dict[str, str] = {}
    for (tm, pos), members in rooms.items():
        if not all(pid in asked for _slot, pid in members):
            continue
        _slot, last = max(members)
        asked.discard(last)
        kept_back[last] = (f"releasing him would empty {tm}'s {pos} room, and a pool with nobody to "
                           "divide it is not a projection -- the last man in a room stays")

    out = _renumber(ros.filter(~pl.col("player_id").is_in(list(asked)))) if asked else ros
    for rec, pid in asks:
        if rec["reason"]:
            continue
        if pid in kept_back:
            rec["reason"] = kept_back[pid]
        elif pid not in here:
            rec["reason"] = f"no rows for {pid}"
        else:
            rec.update(base_now=1.0, used_now=OFF_ROSTER, rows=1, applied=True)
    return out, [rec for rec, _ in asks]


def _place(members: list[str], asked: dict[str, tuple[int, str]]) -> dict[str, int]:
    """One room, rebuilt: the men who were asked for at the slots asked for, everybody else around them.

    `members` is the room in the order the chart has it and `asked` is player -> (slot, when). Two men
    sent to the same slot are settled by *when*, most recent first, because `Scenario.set` keys on the
    override id and so the order the edits are stored in is not the order they were made. The loser takes
    the nearest slot behind the one he wanted, or the nearest in front if the room ends before that.
    """
    n = len(members)
    movers = sorted(asked.items(), key=lambda kv: kv[1][1], reverse=True)   # most recent edit first
    movers.sort(key=lambda kv: kv[1][0])                    # stable, so it still leads its own slot
    out: dict[str, int] = {}
    taken: set[int] = set()
    for pid, (slot, _at) in movers:
        want = min(max(1, slot), n)
        free = next((s for s in range(want, n + 1) if s not in taken),
                    next((s for s in range(want - 1, 0, -1) if s not in taken), want))
        out[pid] = free
        taken.add(free)
    spare = (s for s in range(1, n + 1) if s not in taken)
    for pid in members:
        if pid not in out:
            out[pid] = next(spare)
    return out


def _reslot(ros: pl.DataFrame, scenario: Scenario) -> tuple[pl.DataFrame, list[dict]]:
    """Move players on the depth chart, and renumber every room a move touched.

    A depth slot is a rank, not a quantity, so this is the one override that cannot be applied cell by
    cell. Setting a third receiver to slot 1 has to push the men in front of him back, or the room ends
    up with two number ones and the priors are read off the wrong rows.

    So a room is rebuilt rather than re-sorted: the men who were asked for are *placed* at the slots
    they were asked for, and everybody else falls into the slots left over, keeping the order the
    published chart had them in. Re-sorting was the first attempt and it only worked upwards -- sending
    the starter to 2 left him first anyway, because no one else claimed slot 1 and a rank nobody
    contests is a rank that does not move. Placement handles both directions: send him to 2 and the man
    behind him takes the room's first slot.

    `slot_bucket` and `avail_slot` are re-derived rather than carried, because they are what the priors
    and the survival curve are keyed on: a promotion that left them behind would give a man a starter's
    share of the pool at a backup's expected games, which is a projection of nobody.
    """
    asks: list[tuple[dict, str]] = []
    wanted: dict[str, tuple[int, str]] = {}
    for o in scenario.at_level("player"):
        if o.field != DEPTH_FIELD or o.week is not None:
            # a chart is settled before the season rather than per game, so a week-scoped move is not a
            # move this stage can make. `_season_grain` is the one that reports it, and reporting it
            # twice -- applied here, unapplied there -- would be a log arguing with itself.
            continue
        rec = _record(o)
        if o.mode != "set":
            # There is no sensible reading of "twice as deep on the chart", and guessing one would be
            # worse than saying so: an unapplied edit is reported, never silently reinterpreted.
            rec["reason"] = "a depth slot is set, not multiplied"
        else:
            wanted[o.key] = (max(1, int(round(float(o.value)))), o.at or "")
        asks.append((rec, o.key))
    if not wanted:
        return ros, [rec for rec, _ in asks]

    before = dict(zip(ros["player_id"].to_list(), ros[DEPTH_FIELD].to_list(), strict=True))
    # sorted rather than trusted: "everybody else keeps the order the chart had" is only true if the
    # rooms are read in slot order in the first place
    ordered = ros.sort(["team", "position", DEPTH_FIELD])
    rooms: dict[tuple[str, str], list[str]] = {}
    for pid, tm, pos in zip(ordered["player_id"], ordered["team"], ordered["position"], strict=True):
        rooms.setdefault((tm, pos), []).append(pid)

    placed: dict[str, int] = {}
    for members in rooms.values():
        if not any(pid in wanted for pid in members):
            continue
        placed.update(_place(members, {pid: wanted[pid] for pid in members if pid in wanted}))

    out = ros.with_columns(
        pl.col("player_id").replace_strict(placed, default=None, return_dtype=pl.Int32)
        .fill_null(pl.col(DEPTH_FIELD)).alias(DEPTH_FIELD)
    ) if placed else ros
    cap = pl.col("position").replace_strict(priors.SLOT_CAP, default=4, return_dtype=pl.Int32)
    out = out.with_columns(
        pl.min_horizontal(DEPTH_FIELD, cap).alias("slot_bucket"),
        pl.min_horizontal(DEPTH_FIELD, pl.lit(roster.AVAIL_SLOT_CAP, pl.Int32)).alias("avail_slot"),
    )

    after = dict(zip(out["player_id"].to_list(), out[DEPTH_FIELD].to_list(), strict=True))
    for rec, pid in asks:
        if rec["reason"]:
            continue
        if pid not in after:
            rec["reason"] = f"no rows for {pid}"
            continue
        rec.update(base_now=float(before[pid]), used_now=float(after[pid]), rows=1, applied=True)
    return out.sort(["team", "position", DEPTH_FIELD]), [rec for rec, _ in asks]


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

    `opp` and `adj` are the two per-game frames: what each man is projected to get in each game, and
    the rates he is projected to get it at with that game's factor beside them. They are what a per-game
    edit is made against, so they are handed back rather than left inside the run.
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
    adj: pl.DataFrame
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

    The order is the engine's own: the league patch first because everything downstream reads it, then
    the depth chart because every prior is keyed on it, then availability, then shares and rates, then
    the team environment, and only then the pools -- so a promoted player is priced as what he was
    promoted to, a share edit is still subject to normalisation, and a team edit still divides among the
    players who were there to receive it.

    A player edit carrying a week is held back from those season stages and applied inside the pool
    division instead, on the one frame that has a row per player per game. Same scenario, same
    provenance, one stage later.
    """
    sc = scenario or Scenario()
    st = settings_for(sc)
    fitted = fitted_for(sc)
    log: list[dict] = []

    def note(stage: str, records: list[dict]) -> None:
        log.extend({**r, "stage": stage} for r in records)

    ros = roster.roster(season)
    # before the chart, because a released player is not a slot to be placed: he is a row the rest of
    # the run must never see, and the room has to renumber around the hole before anybody is placed in it
    ros, rec = _release(ros, sc)
    note("roster", rec)
    # then the chart, because a slot is not a number applied to a projection -- it is the row the priors,
    # the survival curve and the quarterback queue are all read off, so it has to be settled before any
    # of them are asked
    ros, rec = _reslot(ros, sc)
    note("depth", rec)
    part = roster.participation(season, st, ros=ros, fitted=fitted)
    part, rec = apply(part, sc, "player", "player_id", fields=AVAILABILITY_FIELDS, weeks="never")
    note("availability", rec)
    part = _rederive_availability(part)
    # participation metrics live in both frames: the pools read the share frame, the team page reads
    # this one, and an edit that landed on only one of them would show a number the pool never used.
    part, rec = apply(part, sc, "player", "player_id", fields=roster.PARTICIPATION_METRICS,
                      weeks="never")
    note("participation", rec)

    shares = opportunity.player_shares(season, st, ros=ros, fitted=fitted)
    shares, rec = apply(shares, sc, "player", "player_id", fields=opportunity.SHARE_METRICS,
                        weeks="never")
    note("shares", rec)

    rates = efficiency.rates(season, st, ros=ros, fitted=fitted)
    rates, rec = apply(rates, sc, "player", "player_id", fields=efficiency.RATE_METRICS,
                       weeks="never")
    note("rates", rec)

    env = team.game_environment(season, st)
    env, rec = apply(env, sc, "team", "team", week_col="week", fields=TEAM_FIELDS)
    note("environment", rec)

    def one_game(grid: pl.DataFrame) -> pl.DataFrame:
        """The per-game player edits, inside the pool division rather than after it."""
        edited, records = apply(grid, sc, "player", "player_id", week_col="week",
                                fields=GAME_FIELDS, weeks="only")
        note("game", records)
        return edited

    opp = opportunity.opportunity(season, st, shares=shares, part=part, fitted=fitted, env=env,
                                  on_grid=one_game)
    # the rates cannot ride along in `one_game`: they are not joined to his games until here, and the
    # `used_` answer the stat line reads is the product of the rate and this game's factor, so the edit
    # goes between the join and the product and the product is then re-derived.
    adj = efficiency.adjusted(opp, rates, st)
    adj, rec = apply(adj, sc, "player", "player_id", week_col="week",
                     fields=GAME_RATE_FIELDS, weeks="only")
    note("game rates", rec)
    adj = efficiency.use_rates(adj)

    wk = compose.weekly(season, st, opp=opp, player_rates=rates, adj=adj)
    yr = compose.seasonal(wk, season, st)
    board = compose.board(yr, st, prev)
    # the season stages skipped these and no per-game stage claimed them, so nothing else would say so
    note("season grain", _season_grain(sc))

    prov = pl.DataFrame(log, schema=PROVENANCE_SCHEMA) if log else pl.DataFrame(schema=PROVENANCE_SCHEMA)
    return Run(scenario=sc, season=season, settings=st, roster=ros, part=part, shares=shares,
               rates=rates, env=env, opp=opp, adj=adj, weekly=wk, season_frame=yr, board=board,
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
