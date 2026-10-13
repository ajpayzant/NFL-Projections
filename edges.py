"""Compare sportsbook prices (odds.py) with the model, for the Edges tab and the line checker.

Nothing here changes a projection. Each book's line is priced with the tool's own distributions:
    player props    dist.py around the blended projection, for a game he plays (books void otherwise)
    anytime TD      P(1+ rushing or receiving TD); books post it as Yes only, so the book's probability
                    keeps its margin
    game lines      dist.GameDist on the market-free projected margin and total (key numbers included)
For each bet: the book's price, its no-vig probability (both sides), ours, the edge and the EV.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

import numpy as np
import pandas as pd

import dist

# market -> (bet type shown, projection column, distribution key)
PROPS = {"player_pass_yds": ("Pass yds", "passing_yards", "passing_yards"),
         "player_pass_tds": ("Pass TDs", "passing_tds", "passing_tds"),
         "player_pass_completions": ("Completions", "completions", "completions"),
         "player_pass_attempts": ("Pass attempts", "attempts", "attempts"),
         "player_pass_interceptions": ("Interceptions", "interceptions", "interceptions"),
         "player_rush_yds": ("Rush yds", "rushing_yards", "rushing_yards"),
         "player_rush_attempts": ("Carries", "carries", "carries"),
         "player_receptions": ("Receptions", "receptions", "receptions"),
         "player_reception_yds": ("Rec yds", "receiving_yards", "receiving_yards"),
         "player_rush_reception_yds": ("Rush + rec yds", "rush_rec_yards", "rush_rec_yards"),
         "player_pass_rush_yds": ("Pass + rush yds", "pass_rush_yards", "pass_rush_yards")}
YES_NO = {"player_anytime_td": ("Anytime TD", "p_anytime_td")}
GAME = ("h2h", "spreads", "totals")
KIND = {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total", **{k: v[0] for k, v in PROPS.items()},
        **{k: v[0] for k, v in YES_NO.items()}}
WORDS = {"Pass yds": "passing yards", "Pass TDs": "passing TDs", "Completions": "completions", "Pass attempts": "attempts",
         "Interceptions": "interceptions", "Rush yds": "rushing yards", "Carries": "carries", "Receptions": "receptions",
         "Rec yds": "receiving yards", "Rush + rec yds": "rush + rec yards", "Pass + rush yds": "pass + rush yards"}
POSITION = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE"}


def key(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}


def plain(name: str) -> str:
    return key(" ".join(p for p in str(name).replace(".", " ").split() if key(p) not in SUFFIX))


def surname(name: str) -> str:
    parts = [p for p in str(name).replace(".", " ").split() if key(p) not in SUFFIX]
    return key(parts[-1]) if parts else ""


def same_first(a: str, b: str) -> bool:
    fa, fb = key(str(a).split()[0]), key(str(b).split()[0])
    return fa[:1] == fb[:1] or difflib.SequenceMatcher(None, fa, fb).ratio() >= 0.75


def nick(team_name: str) -> str:
    """'Seattle Seahawks' -> 'seahawks'; the part every source agrees on."""
    return key(str(team_name).split()[-1])


def decimal(price) -> np.ndarray:
    price = np.asarray(price, float)
    return np.where(price > 0, 1 + price / 100, 1 + 100 / np.abs(price))


# ---------------------------------------------------------------- pricing one line

def prop_win_push(tables: dict, row, market: str, line: float, over: bool) -> tuple[float, float]:
    _, col, dkey = PROPS[market]
    group = {"QB": "QB", "RB": "RB"}.get(row["projected_position"], "WR+TE")
    if f"{group}|{dkey}" not in tables:
        return np.nan, 0.0
    return dist.win_push(tables, group, dkey, float(mean_of(row, col)), float(line), over)


def mean_of(row, col: str) -> float:
    if col == "rush_rec_yards":
        return float(np.nan_to_num(row.get("rushing_yards", 0.0)) + np.nan_to_num(row.get("receiving_yards", 0.0)))
    if col == "pass_rush_yards":
        return float(np.nan_to_num(row.get("passing_yards", 0.0)) + np.nan_to_num(row.get("rushing_yards", 0.0)))
    v = row.get(col, np.nan)
    return float(v) if v is not None else np.nan


def game_win_push(margin_dist, total_dist, g, market: str, home_side: bool, line: float = 0.0,
                  over: bool = True) -> tuple[float, float]:
    """(P(win), P(push)). line = this team's handicap for a spread (e.g. -3.5)."""
    if market == "Total":
        gt, eq = total_dist.p_over(g["total_hat"], line)
        return (gt if over else 1 - gt - eq), eq
    m = g["margin_hat"] if home_side else -g["margin_hat"]
    if market == "Moneyline":
        gt, eq = margin_dist.p_over(m, 0.0)
        return gt / max(1 - eq, 1e-9), 0.0                     # a tie goes to overtime
    gt, eq = margin_dist.p_over(m, -line)                      # covers when his margin beats -line
    return gt, eq


def evaluate(p_win: float, p_push: float, odds: float, other: float | None = None) -> dict:
    dec = float(decimal([odds])[0])
    p = p_win / max(1 - p_push, 1e-9)
    implied = 1 / dec
    if other is not None:
        o = 1 / float(decimal([other])[0])
        book_p, novig = implied / (implied + o), True
    else:
        book_p, novig = implied, False
    ev = 100 * (p_win * (dec - 1) - (1 - p_win - p_push))
    return {"p": p, "push": p_push, "book_p": book_p, "novig": novig, "edge": p - book_p, "ev": ev, "dec": dec}


# ---------------------------------------------------------------- the whole board

def compare(odds: pd.DataFrame, games: pd.DataFrame, players: pd.DataFrame, tables: dict, margin_dist, total_dist,
            now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    now = now or pd.Timestamp.now(tz="UTC")
    o = odds.copy()
    gid = {nick(h) + "|" + nick(a): g for h, a, g in zip(games.home_name, games.away_name, games.game_id)}
    o["game_id"] = [gid.get(nick(h) + "|" + nick(a)) for h, a in zip(o.home, o.away)]
    o = o[o.game_id.notna() & (pd.to_datetime(o.commence_time, utc=True) > now)].copy()
    stats = {"prices": len(o)}
    if o.empty:
        return pd.DataFrame(), stats
    o["dec"] = decimal(o.price)
    o["pair"] = np.where(o.market == "spreads", o.point.abs(), o.point.fillna(0.0))
    imp = 1 / o.dec
    grp = [o.event_id, o.book, o.market, o.player.fillna(""), o.pair]
    o["novig"] = (imp / imp.groupby(grp).transform("sum")).where(imp.groupby(grp).transform("size") == 2)
    yes_only = o.market.isin(list(YES_NO)) & o.novig.isna()
    o["book_p"] = o.novig.where(~yes_only, 1 / o.dec)
    o["one_sided"] = yes_only
    for c in ("p_model", "p_push", "proj", "proj_own", "proj_opp"):
        o[c] = np.nan
    o["team"], o["pos"], o["pid"] = None, None, None
    gm = games.set_index("game_id")
    # game lines
    for i, r in o[o.market.isin(GAME)].iterrows():
        g = gm.loc[r.game_id]
        home = nick(r["name"]) == nick(g.home_name)
        mk = {"h2h": "Moneyline", "spreads": "Spread", "totals": "Total"}[r.market]
        w, push = game_win_push(margin_dist, total_dist, g, mk, home, 0.0 if r.market == "h2h" else float(r.point),
                                str(r["name"]).lower() == "over")
        o.at[i, "p_model"], o.at[i, "p_push"] = w / max(1 - push, 1e-9), push
        if r.market == "totals":
            o.at[i, "proj"] = g.total_hat
        else:
            o.at[i, "proj_own"], o.at[i, "proj_opp"] = (g.home_pts, g.away_pts) if home else (g.away_pts, g.home_pts)
    # player props
    pr = players.assign(k=players.player_name.map(plain), sur=players.player_name.map(surname))
    look = {(g, k): r for g, k, r in zip(pr.game_id, pr.k, pr.to_dict("records"))}
    once = pr.groupby(["game_id", "sur"]).sur.transform("size") == 1
    by_sur = {(g, s): r for g, s, r, u in zip(pr.game_id, pr.sur, pr.to_dict("records"), once) if u}
    sk = o[o.market.isin(list(PROPS) + list(YES_NO))]
    matched = 0
    for i, r in sk.iterrows():
        row = look.get((r.game_id, plain(r.player)))
        if row is None:
            row = by_sur.get((r.game_id, surname(r.player)))
            if row is not None and not same_first(row["player_name"], r.player):
                row = None
        if row is None:
            continue
        side = str(r["name"]).lower()
        if r.market in YES_NO:
            p = float(row[YES_NO[r.market][1]])
            o.at[i, "p_model"] = p if side == "yes" else 1 - p
            o.at[i, "proj"] = row.get("anytime_td", np.nan)
        else:
            if r.point is None or not np.isfinite(r.point):
                continue
            w, push = prop_win_push(tables, row, r.market, float(r.point), side == "over")
            if not np.isfinite(w):
                continue
            o.at[i, "p_model"], o.at[i, "p_push"] = w / max(1 - push, 1e-9), push
            o.at[i, "proj"] = mean_of(row, PROPS[r.market][1])
        o.at[i, "team"], o.at[i, "pos"], o.at[i, "pid"] = row["team"], row["projected_position"], row["player_id"]
        matched += 1
    stats["player props"], stats["player props matched"] = len(sk), matched
    o = o[o.p_model.notna()].copy()
    o["p_push"] = o.p_push.fillna(0.0)
    o["kind"] = o.market.map(KIND)
    o["position"] = np.where(o.market.isin(GAME), "Game", o.pos.fillna(""))
    o["game"] = [f"{gm.loc[g].away} @ {gm.loc[g].home}" for g in o.game_id]
    o["bet"] = [_label(r) for r in o.itertuples(index=False)]
    o["edge"] = o.p_model - o.book_p
    o["ev"] = o.p_model * (1 - o.p_push) * (o.dec - 1) - (1 - o.p_model) * (1 - o.p_push)
    abbr = {nick(n): a for n, a in zip(games.home_name, games.home)} | {nick(n): a for n, a in zip(games.away_name, games.away)}
    o["subject"] = [r.player if r.market not in GAME else ("Game total" if r.market == "totals" else abbr.get(nick(r.name), r.name))
                    for r in o.itertuples(index=False)]
    o["line"] = [_line(r) for r in o.itertuples(index=False)]
    o["proj_text"] = [_proj_text(r) for r in o.itertuples(index=False)]
    return o, stats


def _line(r) -> str:
    pt = f"{r.point:g}" if r.point is not None and np.isfinite(r.point) else ""
    if r.market == "h2h":
        return "Moneyline"
    if r.market == "spreads":
        return f"Spread {'+' if r.point > 0 else ''}{pt}"
    if r.market in YES_NO:
        return f"Anytime TD: {r.name}"
    side = "Over" if str(r.name).lower() == "over" else "Under"
    return f"{side} {pt} {'points' if r.market == 'totals' else WORDS[KIND[r.market]]}"


def _proj_text(r) -> str:
    if r.market in ("h2h", "spreads"):
        return f"{r.proj_own:.1f}–{r.proj_opp:.1f}" if np.isfinite(r.proj_own) else ""
    if not np.isfinite(r.proj):
        return ""
    if r.market == "totals":
        return f"{r.proj:.1f} total"
    if r.market in YES_NO:
        return f"{r.proj:.2f} TDs"
    return f"{r.proj:.{2 if r.market in ('player_pass_tds', 'player_pass_interceptions') else 1}f} {WORDS[KIND[r.market]]}"


def _label(r) -> str:
    pt = f"{r.point:g}" if r.point is not None and np.isfinite(r.point) else ""
    if r.market == "h2h":
        return f"{r.name} ML"
    if r.market == "spreads":
        return f"{r.name} {'+' if r.point > 0 else ''}{pt}"
    if r.market == "totals":
        return f"{r.name} {pt}"
    if r.market in YES_NO:
        return f"{r.player} anytime TD {r.name}"
    return f"{r.player} {'o' if str(r.name).lower() == 'over' else 'u'}{pt} {KIND[r.market]}"
