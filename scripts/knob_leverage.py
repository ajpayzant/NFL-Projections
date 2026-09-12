"""Which knobs move a projection, and by how much. The measurement the override surface is built on.

The app offers a number of editable fields per player and per team. Some are the projection's own
load-bearing numbers and some reach nothing the stat line reads, and for a while nothing distinguished
the two -- so a user deciding what was worth overriding had no way to tell a target share from a knob
that did nothing, and the per-position shortlists in `ui.KNOBS` were ordered by argument rather than by
effect. They led with `pass_td_rate`, which moves nothing at all.

This is what settled it. Each field is nudged by +10% on the best eligible man on all 32 teams -- one
edit per team, so no two contend for the same pool -- the whole engine is re-run through
`overrides.run`, and what moved is recorded:

    own_pts        the median move in the edited player's own projected points
    own_%          the same as a share of his season, which is the number a reader feels
    own_max        the largest single move, because a median hides the case that matters
    team_pts       the median move in his team's total -- zero means the edit only redistributed
    others_moved   how many other players moved with him, which is the same thing from the other side
    moves          the projected columns that changed, so a zero in points is not read as "no effect"

`depth_slot` and `on_roster` are not multiplied, because there is no 10% of a roster spot: the backup is
promoted to slot 1 and the starter is released. `depth_slot` is probed on a man who is not already slot 1,
or the edit is a no-op that reads as a dead knob.

One limit to read the output with: this measures the player board and the team totals summed off it, so a
field can score zero here and still be worth offering if it moves a surface the board does not cover.
`implied_points` is exactly that -- no player stat, but `standings` derives the whole league table from
it -- and it stays a knob for that reason. A zero is a question, not a verdict.

    python -m scripts.knob_leverage                  # every editable field, league-wide
    python -m scripts.knob_leverage --by-position     # the ranking `ui.KNOBS` is ordered by
    python -m scripts.knob_leverage --team            # the team environment's knobs
    python -m scripts.knob_leverage --evidence        # the fields the app deliberately does not offer

The last one is the guard rail: every field in `overrides.EVIDENCE_FIELDS` should move no projected
stat. If one of them starts moving points, it stopped being evidence and should be offered as a knob.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("POLARS_MAX_THREADS", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl  # noqa: E402

from src.model import overrides  # noqa: E402

BUMP = 1.10
MOVED = 0.25          # fantasy points, below which a teammate did not really move
POSITIONS = ("QB", "RB", "WR", "TE")

# not multiplied: promote to the top of the room, and release
SPECIAL = {overrides.DEPTH_FIELD: ("set", 1.0), overrides.ROSTER_FIELD: ("set", 0.0)}


def _frames(run: overrides.Run) -> dict[str, tuple[str, pl.DataFrame]]:
    """Which frame owns each field, so eligibility is read off the number the engine actually used."""
    out: dict[str, tuple[str, pl.DataFrame]] = {}
    for name, fr in (("availability", run.part), ("shares", run.shares), ("rates", run.rates)):
        for c in fr.columns:
            out.setdefault(c, (name, fr))
    return out


def _eligible(field: str, frame: pl.DataFrame, board: pl.DataFrame,
              position: str | None = None) -> pl.DataFrame | None:
    """Everyone with a non-zero value for this field, best projection first.

    `position` matters more than it looks. The best man in the league with any carry share at all is a
    quarterback, so a league-wide sweep ranks the running back's most important knob near the bottom;
    a position's shortlist has to be measured on that position.
    """
    if field not in frame.columns:
        return None
    have = frame.select("player_id", pl.col(field).cast(pl.Float64).alias("v")).filter(
        pl.col("v").is_not_null() & (pl.col("v").abs() > 1e-9))
    who = board.select("player_id", "player", "team", "position", "fantasy_points")
    if position is not None:
        who = who.filter(pl.col("position") == position)
    return have.join(who, on="player_id", how="inner").sort("fantasy_points", descending=True)


def _stat_moves(base: pl.DataFrame, after: pl.DataFrame, ids: set[str]) -> str:
    """Which projected numbers moved, so a zero in points is not read as a knob that does nothing.

    Several fields feed a count nobody scores -- completions, air yards, snaps -- and a field that moves
    those is informative even where it cannot move a fantasy total.
    """
    skip = ("season", "draft_pick", "depth_slot", "slot_bucket", "overall_rank", "position_rank",
            "drop_next", "vs_starter")
    cols = [c for c in base.columns if base.schema[c].is_numeric() and c not in skip]
    b = base.filter(pl.col("player_id").is_in(ids)).select("player_id", *cols)
    a = after.filter(pl.col("player_id").is_in(ids)).select(
        "player_id", *[pl.col(c).alias(f"n_{c}") for c in cols])
    j = b.join(a, on="player_id", how="inner")
    got = []
    for c in cols:
        d = (j[f"n_{c}"].cast(pl.Float64) - j[c].cast(pl.Float64)).abs()
        rel = float((d / j[c].cast(pl.Float64).abs().clip(1e-6)).median() or 0.0)
        if rel > 0.001:
            got.append((rel, c))
    got.sort(reverse=True)
    return ", ".join(f"{c} {100 * r:.0f}%" for r, c in got[:3]) or "nothing"


def probe(field: str, keys: list[str], base: pl.DataFrame, *, level: str = "player",
          ids: set[str] | None = None) -> dict:
    """One field, one run of the whole engine, and what moved."""
    mode, value = SPECIAL.get(field, ("multiply", BUMP))
    items = tuple(overrides.Override(level=level, key=k, field=field, mode=mode, value=value)
                  for k in keys)
    run = overrides.run(overrides.Scenario(name=f"probe_{field}", items=items))
    if ids is None:
        ids = set(keys)
    b = base.select("player_id", "team", pl.col("fantasy_points").alias("b_pts"))
    n = run.board.select("player_id", pl.col("fantasy_points").alias("n_pts"))
    j = b.join(n, on="player_id", how="left").with_columns(
        (pl.col("n_pts").fill_null(0.0) - pl.col("b_pts")).alias("d"))
    edited = j.filter(pl.col("player_id").is_in(ids))
    others = j.filter(~pl.col("player_id").is_in(ids))
    team = j.group_by("team").agg(pl.col("d").sum().alias("team_d"))
    return {
        "field": field,
        "edit": f"{mode} {value:g}",
        "own_pts": round(float(edited["d"].abs().median() or 0.0), 2),
        "own_%": round(float((edited["d"].abs() / edited["b_pts"].clip(1.0)).median() or 0.0) * 100, 2),
        "own_max": round(float(edited["d"].abs().max() or 0.0), 1),
        "team_pts": round(float(team["team_d"].abs().median() or 0.0), 2),
        "others_moved": int(others.filter(pl.col("d").abs() > MOVED).height),
        "moves": _stat_moves(base, run.board, ids),
        "n": len(keys),
    }


def player_fields(evidence: bool = False) -> list[str]:
    if evidence:
        return list(overrides.EVIDENCE_FIELDS)
    return [f for f in overrides.FIELDS["player"] if f not in overrides.GAME_ONLY_FIELDS]


def sweep(run: overrides.Run, fields: list[str], position: str | None = None,
          per_team: int = 1) -> pl.DataFrame:
    owners = _frames(run)
    board = run.board
    rows = []
    for field in fields:
        if field == overrides.ROSTER_FIELD:
            frame = run.roster.with_columns(pl.lit(1.0).alias(field))
        elif field in owners:
            frame = owners[field][1]
        else:
            continue
        cand = _eligible(field, frame, board, position)
        if cand is None or cand.is_empty():
            continue
        if field == overrides.DEPTH_FIELD:
            cand = cand.filter(pl.col("v") >= 2.0)
        picked = cand.group_by("team").head(per_team)
        if picked.height < 8:
            continue
        rows.append(probe(field, picked["player_id"].to_list(), board))
    return pl.DataFrame(rows).sort("own_pts", descending=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--by-position", action="store_true", help="rank per position, as ui.KNOBS is")
    p.add_argument("--team", action="store_true", help="the team environment's knobs instead")
    p.add_argument("--evidence", action="store_true",
                   help="the fields the app does not offer, which should move nothing")
    args = p.parse_args(argv)

    pl.Config.set_tbl_width_chars(210)
    pl.Config.set_tbl_cols(12)
    pl.Config.set_tbl_rows(60)
    t0 = time.time()
    base = overrides.run()

    if args.team:
        teams = base.board["team"].unique().sort().to_list()
        leaders = set(base.board.sort("fantasy_points", descending=True)
                      .group_by("team").head(1)["player_id"])
        offered = list(overrides.TEAM_FIELDS) + list(overrides.TEAM_EVIDENCE_FIELDS)
        rows = [probe(f, teams, base.board, level="team", ids=leaders)
                for f in offered if f in base.env.columns]
        out = pl.DataFrame(rows).sort("team_pts", descending=True).with_columns(
            pl.col("field").is_in(list(overrides.TEAM_FIELDS)).alias("offered"))
        print("\nTEAM KNOBS   +10% on all 32 teams, own_* is the man each team leans on most")
        print(out)
        print(f"\n{time.time() - t0:.0f}s")
        return 0

    fields = player_fields(args.evidence)
    if args.by_position:
        for pos in POSITIONS:
            got = sweep(base, fields, position=pos)
            print(f"\n{pos}   +10% on the best {pos} of each team   ({time.time() - t0:.0f}s)")
            print(got)
        print("\nshortlist in measured order, knobs worth a first screen:")
        for pos in POSITIONS:
            got = sweep(base, fields, position=pos)
            live = got.filter((pl.col("own_pts") > 0.05)
                              & ~pl.col("field").is_in([overrides.DEPTH_FIELD,
                                                        overrides.ROSTER_FIELD]))["field"].to_list()
            print(f'    "{pos}": (' + ", ".join(f'"{f}"' for f in live) + "),")
    else:
        got = sweep(base, fields)
        title = "EVIDENCE FIELDS, which should move no projected stat" if args.evidence \
            else "PLAYER KNOBS   +10% on the top eligible man on each of the 32 teams"
        print(f"\n{title}")
        print(got)
        if args.evidence:
            bad = got.filter(pl.col("own_pts") > 0.05)
            print("\n" + ("every evidence field moved no points, as declared" if bad.is_empty()
                          else f"{bad.height} moved points and should be offered as knobs:\n{bad}"))
    print(f"\n{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
