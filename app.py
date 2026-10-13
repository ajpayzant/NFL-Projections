"""NFL player game projections: Slate · Game · Depth charts · Player · Edges · Fantasy · Results · How it works.

    run_app.bat            # http://localhost:8690
"""
from __future__ import annotations

import json
import os
import pickle

import altair as alt
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st

import adjust
import cloud
import config as C
import dist
import edges
import odds
import tracking

st.set_page_config(page_title="NFL Player Projections", layout="wide")


@st.cache_data(ttl=600, show_spinner="Loading the latest projections...")
def _sync() -> str | None:
    """On Streamlit Cloud: pull the refresh's newest files (no-op locally)."""
    return cloud.sync()


_sync()
PROJ, ACT, MUTED = "#2a78d6", "#eb6834", "#8a8984"   # same palette as the NHL and NBA tools
GROUP = {"QB": "QB", "RB": "RB", "WR": "WR+TE", "TE": "WR+TE"}

# per position: (header, statistic, format) for the projection columns, and (header, statistic, line) for props
STATS = {"QB": [("Pass yds", "passing_yards", "%.0f"), ("Pass TD", "passing_tds", "%.2f"), ("INT", "interceptions", "%.2f"),
                ("Att", "attempts", "%.1f"), ("Rush yds", "rushing_yards", "%.0f"), ("Rush TD", "rushing_tds", "%.2f")],
         "RB": [("Carries", "carries", "%.1f"), ("Rush yds", "rushing_yards", "%.0f"), ("Rec", "receptions", "%.1f"),
                ("Rec yds", "receiving_yards", "%.0f"), ("TD", "anytime_td", "%.2f")],
         "WR": [("Tgt", "targets", "%.1f"), ("Rec", "receptions", "%.1f"), ("Rec yds", "receiving_yards", "%.0f"),
                ("Rec TD", "receiving_tds", "%.2f")],
         "TE": [("Tgt", "targets", "%.1f"), ("Rec", "receptions", "%.1f"), ("Rec yds", "receiving_yards", "%.0f"),
                ("Rec TD", "receiving_tds", "%.2f")]}
PROPS = {"QB": [("o224.5 PY", "passing_yards", 224.5), ("o249.5 PY", "passing_yards", 249.5), ("o1.5 PTD", "passing_tds", 1.5),
                ("o0.5 INT", "interceptions", 0.5), ("o19.5 RuY", "rushing_yards", 19.5)],
         "RB": [("o49.5 RuY", "rushing_yards", 49.5), ("o69.5 RuY", "rushing_yards", 69.5), ("o14.5 Car", "carries", 14.5),
                ("o2.5 Rec", "receptions", 2.5), ("Any TD", "anytime_td", 0.5)],
         "WR": [("o3.5 Rec", "receptions", 3.5), ("o4.5 Rec", "receptions", 4.5), ("o49.5 RY", "receiving_yards", 49.5),
                ("o69.5 RY", "receiving_yards", 69.5), ("Any TD", "anytime_td", 0.5)],
         "TE": [("o2.5 Rec", "receptions", 2.5), ("o3.5 Rec", "receptions", 3.5), ("o29.5 RY", "receiving_yards", 29.5),
                ("o44.5 RY", "receiving_yards", 44.5), ("Any TD", "anytime_td", 0.5)]}
SCORING = {"PPR": "fantasy_ppr", "Half PPR": "fantasy_half", "Standard": "fantasy_std"}


# ---------------------------------------------------------------- data

def _mtime(p) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource
def _load(mtime: float) -> dict | None:
    f = C.LIVE / "players.parquet"
    if not f.exists():
        return None
    meta = json.loads((C.LIVE / "meta.json").read_text())
    tables = pickle.load(open(C.MODELS / "dist.pkl", "rb"))
    oos = pl.read_parquet(C.MODELS / "gamelines_oos.parquet")
    return {"meta": meta, "players": pl.read_parquet(f), "backup": pl.read_parquet(C.LIVE / "players_backup_qb.parquet"),
            "games": pl.read_parquet(C.LIVE / "games.parquet"), "cand": pl.read_parquet(C.LIVE / "candidates.parquet"),
            "history": pl.read_parquet(C.LIVE / "history.parquet"), "tables": tables,
            "margin": dist.GameDist(oos["pred_margin"].to_numpy(), oos["margin"].to_numpy()),
            "total": dist.GameDist(oos["pred_total"].to_numpy(), oos["total"].to_numpy(), bandwidth=2.5)}


def data() -> dict:
    d = _load(_mtime(C.LIVE / "meta.json"))
    if d is None:
        st.warning("No projections yet. Run `python live.py` (or refresh.bat) first.")
        st.stop()
    return d


@st.cache_data(show_spinner=False, max_entries=16)
def _players(built: str, ov_key: str) -> pd.DataFrame:
    d = data()
    return adjust.apply(d["players"], d["backup"], adjust.load(), d["meta"]).to_pandas()


def players() -> pd.DataFrame:
    """The week's projections with the Depth charts tab's edits applied."""
    return _players(data()["meta"]["built_at"], json.dumps(adjust.load(), sort_keys=True))


def games() -> pd.DataFrame:
    return data()["games"].to_pandas()


def american(p) -> str:
    """Fair (no-vig) American odds for probability p."""
    if p is None or not np.isfinite(p) or p <= 0 or p >= 1:
        return "—"
    p = min(max(p, 0.005), 0.995)
    o = -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
    o = 5 * round(o / 5)
    return "EVEN" if abs(o) <= 100 else f"{o:+d}"


def odds_mode() -> bool:
    return st.session_state.get("prob_view", "%") == "Fair odds"


def odds_toggle():
    st.sidebar.segmented_control("Probabilities", ["%", "Fair odds"], default="%", key="prob_view",
                                 help="Fair odds = the no-vig American price implied by the model's probability.")


def p_over(row, stat: str, line: float) -> float:
    g = GROUP[row["projected_position"]]
    if stat == "anytime_td":
        return float(row["p_anytime_td"])
    return float(dist.p_over(data()["tables"], g, stat, [row[stat]], line)[0])


def when(ts) -> str:
    try:
        return pd.Timestamp(ts).strftime("%a %b %d · %I:%M %p ET")
    except Exception:
        return ""


def week_title() -> str:
    m = data()["meta"]
    return f"Week {m['week']}, {m['season']}"


def pick_game(key: str = "game"):
    g = games()
    lab = {r.game_id: f"{r.away} @ {r.home}" for r in g.itertuples()}
    gid = st.sidebar.selectbox("Game", list(lab), format_func=lab.get, key=key)
    return g[g.game_id == gid].iloc[0]


# ---------------------------------------------------------------- tables

def position_table(p: pd.DataFrame, pos: str, with_team: bool = True) -> pd.DataFrame:
    t = p[p.projected_position == pos].copy()
    t = t.sort_values("fantasy_ppr", ascending=False)
    out = pd.DataFrame({"Team": t.team, "Player": t.player_name, "Opp": t.opp, "Plays": t.p_play * 100})
    for h, s, _ in STATS[pos]:
        out[h] = t[s]
    for h, s, line in PROPS[pos]:
        out[h] = [p_over(r, s, line) for _, r in t.iterrows()]
    out["PPR"] = t.fantasy_ppr
    return out if with_team else out.drop(columns=["Team"])


def show(t: pd.DataFrame, pos: str, height="auto"):
    t = t.copy()
    pcols = [h for h, _, _ in PROPS[pos] if h in t.columns]
    if odds_mode():
        for c in pcols:
            t[c] = t[c].map(american)
        bars = []
    else:
        t[pcols] = (t[pcols] * 100).round()
        bars = pcols
    cfg = {h: st.column_config.NumberColumn(format=f, width="small") for h, _, f in STATS[pos]}
    cfg |= {c: st.column_config.TextColumn(width="small", pinned=c == "Team") for c in ("Team", "Opp")}
    cfg["Player"] = st.column_config.TextColumn("Player", pinned=True)
    cfg["Plays"] = st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100, width="small",
                                                   help="Chance he plays (the projections are for a game he plays)")
    cfg["PPR"] = st.column_config.NumberColumn(format="%.1f", width="small", help="PPR fantasy points (Fantasy tab for the rest)")
    cfg |= {c: st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100, width="small") for c in bars}
    st.dataframe(t, hide_index=True, column_config=cfg, width="stretch", height=height)


# ---------------------------------------------------------------- the game-lines board

BOARD_CSS = """<style>
.gb{font-family:"Source Sans Pro","Segoe UI",sans-serif;max-width:860px;border:1px solid #e5e4df;border-radius:10px;padding:10px 12px 8px;
    background:#fff;color:#0b0b0b}
.gb .bar{display:flex;height:6px;border-radius:3px;overflow:hidden;margin:2px 0 4px}
.gb .bar span{display:block}
.gb .barlab{display:flex;justify-content:space-between;font-size:11px;color:#52514e;margin-bottom:6px}
.gb table{width:100%;border-collapse:separate;border-spacing:4px 4px}
.gb th{font-size:11px;font-weight:600;color:#8a8984;text-align:center;padding:0 4px;text-transform:uppercase;letter-spacing:.03em}
.gb th.t{text-align:left}
.gb td{text-align:center;border-radius:7px;padding:4px 6px;background:#f6f6f4;line-height:1.15}
.gb td.t{text-align:left;background:none;padding-left:0;white-space:nowrap}
.gb td.fav{background:rgba(42,120,214,.10)}
.gb .o{font-weight:700;font-size:15px}.gb .p{font-size:11px;color:#52514e}
.gb .tm{font-weight:700;font-size:15px}.gb .xg{font-size:11px;color:#52514e}
.gb.sm{padding:8px 9px 6px}.gb.sm .o,.gb.sm .tm{font-size:13px}.gb.sm td{padding:3px 4px}
.gb .ttl{font-size:12px;font-weight:600;color:#52514e;margin-bottom:2px}
.gb .ar{display:inline-flex;flex-direction:column;margin-left:5px;vertical-align:middle;gap:2px}
.gb .ar b{display:block;font-size:8px;line-height:8px;height:8px;font-weight:400;color:#8a8984;cursor:pointer;
    user-select:none;padding:0 2px}
.gb .ar b:hover{color:#2a78d6}
</style>"""


def _cell(p: float, fav: bool) -> str:
    return f"<td class='{'fav' if fav else ''}'><div class='o'>{american(p)}</div><div class='p'>{p:.0%}</div></td>"


def _step_cell(bid: str, key: str, label: str, p: float, fav: bool, title: str) -> str:
    return (f"<td id='{bid}-{key}' class='{'fav' if fav else ''}'><div class='o'><span class='tl'>{label}</span> "
            f"<span class='to'>{american(p)}</span><span class='ar' data-g='{key[0]}' title='{title}'>"
            f"<b data-d='1'>&#9650;</b><b data-d='-1'>&#9660;</b></span></div><div class='p'>{p:.0%}</div></td>")


def _step_rows(pairs: list) -> list:
    return [[la, american(pa), f"{pa:.0%}", bool(fa), lb, american(pb), f"{pb:.0%}", bool(fb)]
            for la, pa, fa, lb, pb, fb in pairs]


def _step_script(bid: str, grp: str, keys: tuple[str, str], rows: list, start: int) -> str:
    """Steps one market of a board through its alternate lines in the browser (no rerun)."""
    return f"""<script>(function(){{
const L={json.dumps(rows)}, id="{bid}"; let i={start};
const b=document.getElementById(id); if(!b) return;
const a=document.getElementById(id+"-{keys[0]}"), c=document.getElementById(id+"-{keys[1]}");
const put=(td,l,od,pc,fav)=>{{td.querySelector(".tl").textContent=l;td.querySelector(".to").textContent=od;
  td.querySelector(".p").textContent=pc;td.classList.toggle("fav",fav);}};
b.querySelectorAll(".ar[data-g='{grp}'] b").forEach(x=>x.addEventListener("click",()=>{{
  i=Math.max(0,Math.min(L.length-1,i+Number(x.dataset.d)));const r=L[i];
  put(a,r[0],r[1],r[2],r[3]);put(c,r[4],r[5],r[6],r[7]);}}));
}})();</script>"""


def cover(margin_hat: float, k: float) -> float:
    """P(favourite wins by more than k), a push refunded (key numbers from history, dist.GameDist)."""
    over, push = data()["margin"].p_over(abs(margin_hat), k)
    return over / max(1 - push, 1e-9)


def total_over(total_hat: float, x: float) -> float:
    over, push = data()["total"].p_over(total_hat, x)
    return over / max(1 - push, 1e-9)


def board_html(g, small: bool = False, title: str = "") -> str:
    """Sportsbook-style card: win-probability bar, then away / home rows with the fair moneyline, spread and
    total. The arrows step the spread and the total through alternate lines."""
    bid = f"gb{g.game_id}{'s' if small else ''}".replace("_", "")
    m, t = g.margin_hat, g.total_hat
    pw = data()["margin"].p_over(m, 0.0)
    ph = pw[0] / max(1 - pw[1], 1e-9)
    pa = 1 - ph
    home_fav = m >= 0
    fav_m = abs(m)
    k0 = max(0.5, np.floor(fav_m) + 0.5)
    ks = [k for k in np.arange(k0 - 6, k0 + 7, 1.0) if k > 0]
    cov = {k: cover(m, k) for k in ks}
    start_k = min(ks, key=lambda k: abs(cov[k] - 0.5))
    t0 = np.floor(t) + 0.5
    ts = list(np.arange(t0 - 8, t0 + 9, 1.0))
    over = {x: total_over(t, x) for x in ts}
    start_t = min(ts, key=lambda x: abs(over[x] - 0.5))
    rows = ""
    pts = {g.home: g.home_pts, g.away: g.away_pts}
    for side, team, p_win in (("away", g.away, pa), ("home", g.home, ph)):
        fav = (side == "home") == home_fav
        c = cov[start_k]
        sp = _step_cell(bid, "sf", f"-{start_k:g}", c, c >= 0.5, "Alternate spreads") if fav else \
            _step_cell(bid, "sd", f"+{start_k:g}", 1 - c, 1 - c > 0.5, "Alternate spreads")
        o = over[start_t]
        tot = _step_cell(bid, "to", f"o{start_t:g}", o, o >= 0.5, "Alternate totals") if side == "away" else \
            _step_cell(bid, "tu", f"u{start_t:g}", 1 - o, o < 0.5, "Alternate totals")
        rows += (f"<tr><td class='t'><span class='tm'>{team}</span> <span class='xg'>{pts[team]:.1f} pts</span></td>"
                 f"{_cell(p_win, p_win >= 0.5)}{sp}{tot}</tr>")
    bar = (f"<div class='bar'><span style='width:{pa:.1%};background:#eb6834'></span>"
           f"<span style='width:{ph:.1%};background:#2a78d6'></span></div>"
           f"<div class='barlab'><span>{g.away} {pa:.0%}</span><span>win probability</span><span>{ph:.0%} {g.home}</span></div>")
    sp_rows = _step_rows([(f"-{k:g}", cov[k], cov[k] >= 0.5, f"+{k:g}", 1 - cov[k], 1 - cov[k] > 0.5) for k in ks])
    if not home_fav:
        sp_rows = [[r[4], r[5], r[6], r[7], r[0], r[1], r[2], r[3]] for r in sp_rows]
    keys_sp = ("sd", "sf") if home_fav else ("sf", "sd")
    to_rows = _step_rows([(f"o{x:g}", over[x], over[x] >= 0.5, f"u{x:g}", 1 - over[x], over[x] < 0.5) for x in ts])
    ttl = f"<div class='ttl'>{title}</div>" if title else ""
    return (BOARD_CSS + f"<div id='{bid}' class='gb{' sm' if small else ''}'>{ttl}{bar}<table><tr><th class='t'></th>"
            f"<th>Moneyline</th><th>Spread</th><th>Total</th></tr>{rows}</table></div>"
            + _step_script(bid, "s", keys_sp, sp_rows, ks.index(start_k))
            + _step_script(bid, "t", ("to", "tu"), to_rows, ts.index(start_t)))


def board(g, small=False, title=""):
    st.html(board_html(g, small, title), unsafe_allow_javascript=True)


# ---------------------------------------------------------------- Slate

def page_slate():
    d = data()
    odds_toggle()
    st.title(f"Slate · {week_title()}")
    st.caption(f"Projections built {d['meta']['built_at']}. Every projection is for a game the player plays "
               "(books void a prop otherwise); **Plays** is the chance he does. Lineups from the availability model "
               "and the injury report, with the Depth charts tab's edits on top.")
    g = games()
    with st.expander("Game lines (fair moneyline, spread, total — no market input)", expanded=True):
        cols = st.columns(2)
        for i, r in enumerate(g.itertuples()):
            with cols[i % 2]:
                board(r, small=True, title=f"{r.away} @ {r.home} · {when(r.kickoff_et)}")
        st.caption("Fair (no-vig) prices from the projected margin and total, priced on the distribution of real "
                   "NFL results (key numbers 3 and 7 included). The arrows step through alternate lines.")
    p = players()
    c1, c2 = st.columns([3, 1])
    teams = c1.multiselect("Team", sorted(p.team.unique()), key="slate_teams")
    q = p[p.team.isin(teams)] if teams else p
    tabs = st.tabs(["QB", "RB", "WR", "TE"])
    for tab, pos in zip(tabs, ("QB", "RB", "WR", "TE")):
        with tab:
            show(position_table(q, pos), pos, height=560)


# ---------------------------------------------------------------- Game

def team_panel(p: pd.DataFrame, team: str):
    for pos in ("QB", "RB", "WR", "TE"):
        t = position_table(p[p.team == team], pos, with_team=False)
        if len(t):
            st.markdown(f"**{pos}**")
            show(t, pos)


def page_game():
    odds_toggle()
    g = pick_game()
    st.title(f"{g.away} @ {g.home}")
    st.caption(when(g.kickoff_et) + (f" · {g.roof}" if isinstance(g.roof, str) else ""))
    a, b, c, d_ = st.columns(4)
    a.metric(f"{g.away} points", f"{g.away_pts:.1f}", border=True)
    b.metric(f"{g.home} points", f"{g.home_pts:.1f}", border=True)
    c.metric("Projected total", f"{g.total_hat:.1f}", border=True)
    d_.metric(f"{g.home if g.margin_hat >= 0 else g.away} by", f"{abs(g.margin_hat):.1f}", border=True)
    board(g)
    p = players()
    p = p[p.game_id == g.game_id]
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader(g.away)
        team_panel(p, g.away)
    with right:
        st.subheader(g.home)
        team_panel(p, g.home)


# ---------------------------------------------------------------- Depth charts

def page_depth():
    d = data()
    g = games()
    teams = sorted(set(g.home) | set(g.away))
    team = st.sidebar.selectbox("Team", teams, key="depth_team")
    st.title(f"Depth chart · {team}")
    cand = d["cand"].filter(pl.col("team") == team).to_pandas()
    ov = adjust.load().get(adjust.key(d["meta"], team), {})
    st.caption("Who plays and who starts at quarterback. The model's call comes from recent snaps, the depth chart and "
               "the injury report; change it here and every projection for this team updates (other teams are not "
               "affected). Saved edits are shared with everyone using the app.")
    if cloud.enabled() and adjust.backend() == "file":
        st.warning("Edits are not connected to their gist (add gist_id and github_token to the app's secrets), "
                   "so saves only last until the app restarts.")
    qbs = cand[cand.pos == "QB"].sort_values("p_start", ascending=False)
    names = dict(zip(qbs.player_id, qbs.player_name))
    model_qb = qbs[qbs.starter].player_id.iloc[0] if qbs.starter.any() else qbs.player_id.iloc[0]
    backup = d["backup"].filter((pl.col("team") == team) & (pl.col("group") == "QB"))
    swap_ok = {model_qb} | (set(backup["player_id"].to_list()) if backup.height else set())
    qb_now = ov.get("qb", model_qb)
    qb = st.radio("Starting quarterback", [q for q in qbs.player_id if q in swap_ok], horizontal=True,
                  index=[q for q in qbs.player_id if q in swap_ok].index(qb_now) if qb_now in swap_ok else 0,
                  format_func=lambda q: f"{names[q]} ({qbs.set_index('player_id').p_start[q]:.0%} to start)", key=f"qb_{team}")
    rows = cand[cand.pos != "QB"].sort_values(["pos", "p_play"], ascending=[True, False])
    out_now = set(ov.get("out", []))
    ed = pd.DataFrame({"Plays": [bool(r.in_lineup) and r.player_id not in out_now for r in rows.itertuples()],
                       "Player": rows.player_name, "Pos": rows.pos, "Chance he plays": (rows.p_play * 100).round(),
                       "Depth": rows.depth, "Injury": np.select([rows.inj_out == 1, rows.inj_doubtful == 1, rows.inj_questionable == 1],
                                                               ["Out", "Doubtful", "Questionable"], ""),
                       "id": rows.player_id, "model": rows.in_lineup})
    new = st.data_editor(ed.drop(columns=["id", "model"]), hide_index=True, width="stretch", height=560, key=f"ed_{team}",
                         disabled=["Player", "Pos", "Chance he plays", "Depth", "Injury"],
                         column_config={"Plays": st.column_config.CheckboxColumn(width="small"),
                                        "Chance he plays": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100)})
    a, b, _ = st.columns([1, 1, 4])
    if a.button("Save", type="primary", width="stretch"):
        outs = [i for i, keep, m in zip(ed.id, new.Plays, ed.model) if m and not keep]
        ins = [i for i, keep, m in zip(ed.id, new.Plays, ed.model) if keep and not m]
        adjust.save(d["meta"], team, {"qb": qb if qb != model_qb else None, "out": outs, "in": ins})
        _players.clear()
        st.rerun()
    if b.button("Reset to the model", width="stretch"):
        adjust.save(d["meta"], team, None)
        _players.clear()
        st.rerun()
    if ov.get("in"):
        st.info("Players added here are projected the next time the slate is rebuilt (python live.py); until then "
                "they carry no projection, and their teammates keep their shares.")
    st.caption("Ruling a receiver or back out hands his targets and carries to the teammates who are playing, in "
               "proportion to their own shares, at their own per-target and per-carry efficiency. A quarterback swap uses "
               "a full re-projection of the team under that quarterback.")


# ---------------------------------------------------------------- Player

LADDER = {"passing_yards": [174.5, 199.5, 224.5, 249.5, 274.5, 299.5], "passing_tds": [0.5, 1.5, 2.5, 3.5],
          "interceptions": [0.5, 1.5], "completions": [17.5, 19.5, 21.5, 23.5, 25.5], "attempts": [27.5, 30.5, 33.5, 36.5],
          "rushing_yards": [9.5, 19.5, 29.5, 39.5, 49.5, 59.5, 69.5, 79.5, 99.5], "carries": [9.5, 12.5, 14.5, 17.5, 19.5],
          "receptions": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], "targets": [3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
          "receiving_yards": [19.5, 29.5, 39.5, 49.5, 59.5, 69.5, 79.5, 99.5], "anytime_td": [0.5]}
LABEL = {"passing_yards": "Passing yards", "passing_tds": "Passing TDs", "interceptions": "Interceptions",
         "completions": "Completions", "attempts": "Pass attempts", "rushing_yards": "Rushing yards", "carries": "Carries",
         "receptions": "Receptions", "targets": "Targets", "receiving_yards": "Receiving yards", "anytime_td": "Anytime TD",
         "rushing_tds": "Rushing TDs", "receiving_tds": "Receiving TDs"}
POS_LADDER = {"QB": ["passing_yards", "passing_tds", "completions", "attempts", "interceptions", "rushing_yards"],
              "RB": ["rushing_yards", "carries", "receptions", "receiving_yards", "anytime_td"],
              "WR": ["receiving_yards", "receptions", "targets", "anytime_td"],
              "TE": ["receiving_yards", "receptions", "targets", "anytime_td"]}


def page_player():
    d = data()
    odds_toggle()
    p = players().sort_values("fantasy_ppr", ascending=False)
    lab = {r.player_id: f"{r.player_name} · {r.projected_position} · {r.team}" for r in p.itertuples()}
    pid = st.sidebar.selectbox("Player", list(lab), format_func=lab.get, key="player")
    r = p[p.player_id == pid].iloc[0]
    pos = r.projected_position
    st.title(r.player_name)
    st.caption(f"{pos} · {r.team} {'vs' if r.is_home else '@'} {r.opp} · {week_title()}")
    tiles = st.columns(6)
    tiles[0].metric("Chance he plays", f"{r.p_play:.0%}", border=True)
    for i, (h, s, f) in enumerate(STATS[pos][:4]):
        tiles[i + 1].metric(h, f % r[s], border=True)
    tiles[5].metric("PPR points", f"{r.fantasy_ppr:.1f}", border=True,
                    help=f"Range {r.fantasy_ppr_p10:.1f} to {r.fantasy_ppr_p90:.1f} (10th to 90th percentile)")
    t1, t2, t3 = st.tabs(["Props", "Fantasy", "Game log"])
    with t1:
        rows = []
        for s in POS_LADDER[pos]:
            for ln in LADDER[s]:
                po = p_over(r, s, ln)
                rows.append({"Stat": LABEL[s], "Line": f"{'o' if s != 'anytime_td' else ''}{ln:g}" if s != "anytime_td" else "Yes",
                             "Projection": r[s], "Over %": po * 100, "Over": american(po), "Under": american(1 - po)})
        lad = pd.DataFrame(rows)
        st.dataframe(lad, hide_index=True, width="stretch", height=min(35 * (len(lad) + 1) + 3, 700), column_config={
            "Projection": st.column_config.NumberColumn(format="%.1f"),
            "Over %": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100)})
        st.caption("Fair odds, no margin, for a game he plays. The Edges tab compares them with the books' prices.")
    with t2:
        f = pd.DataFrame([{"Scoring": k, "Projection": r[v], "Floor (10%)": r[f"{v}_p10"], "Median": r[f"{v}_p50"],
                           "Ceiling (90%)": r[f"{v}_p90"]} for k, v in SCORING.items()])
        st.dataframe(f, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.1f")
                                                                         for c in f.columns if c != "Scoring"})
        rank = players()
        rank = rank[rank.projected_position == pos].sort_values("fantasy_ppr", ascending=False).reset_index(drop=True)
        st.caption(f"{pos}{int(rank.index[rank.player_id == pid][0]) + 1} of {len(rank)} this week in PPR. Floor and "
                   "ceiling come from how players with similar projections actually scored.")
    with t3:
        h = d["history"].filter(pl.col("player_id") == pid).sort("kickoff_et", descending=True).to_pandas()
        if h.empty:
            st.info("No games in the last three seasons.")
        else:
            cols = {"season": "Season", "week": "Wk", "opponent": "Opp", "snap_share": "Snaps"}
            for _, s, _ in STATS[pos]:
                if s in h.columns:
                    cols[s] = LABEL.get(s, s)
            cols["fantasy_points_ppr"] = "PPR"
            show_h = h[list(cols)].rename(columns=cols)
            st.dataframe(show_h.head(25), hide_index=True, width="stretch", column_config={
                "Snaps": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "PPR": st.column_config.NumberColumn(format="%.1f")})
            this = h[h.season == data()["meta"]["season"]]
            if len(this):
                st.caption(f"This season: {len(this)} games, {this.fantasy_points_ppr.mean():.1f} PPR a game.")


# ---------------------------------------------------------------- Fantasy

def page_fantasy():
    st.title(f"Fantasy · {week_title()}")
    c1, c2, c3 = st.columns([2, 2, 3])
    scoring = c1.segmented_control("Scoring", list(SCORING), default="PPR", key="fan_scoring") or "PPR"
    pos = c2.segmented_control("Position", ["All", "QB", "RB", "WR", "TE", "FLEX"], default="All", key="fan_pos") or "All"
    col = SCORING[scoring]
    p = players()
    if pos == "FLEX":
        p = p[p.projected_position.isin(["RB", "WR", "TE"])]
    elif pos != "All":
        p = p[p.projected_position == pos]
    p = p.sort_values(col, ascending=False)
    rank = p.groupby("projected_position")[col].rank(ascending=False, method="first").astype(int)
    t = pd.DataFrame({"Rank": np.arange(1, len(p) + 1), "Pos": p.projected_position + rank.astype(str),
                      "Player": p.player_name, "Team": p.team, "Opp": [("vs " if h else "@ ") + o for h, o in zip(p.is_home, p.opp)],
                      "Projection": p[col], "Floor": p[f"{col}_p10"], "Median": p[f"{col}_p50"], "Ceiling": p[f"{col}_p90"],
                      "Plays": p.p_play * 100})
    st.dataframe(t, hide_index=True, width="stretch", height=760, column_config={
        "Player": st.column_config.TextColumn(pinned=True),
        **{c: st.column_config.NumberColumn(format="%.1f", width="small") for c in ("Projection", "Floor", "Median", "Ceiling")},
        "Floor": st.column_config.NumberColumn(format="%.1f", width="small", help="10th percentile: a bad day"),
        "Ceiling": st.column_config.NumberColumn(format="%.1f", width="small", help="90th percentile: a big day"),
        "Plays": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100, width="small",
                                                 help="Chance he plays; the projection is for a game he plays")})
    st.caption("Fantasy points from the projected passing, rushing and receiving lines (4-point passing TDs, -2 an "
               "interception, 0.1 a rushing or receiving yard, 6 a TD; receptions 1 / 0.5 / 0). Floor and ceiling are the "
               "10th and 90th percentiles of how players with similar projections actually scored. Fumbles and "
               "two-point conversions are not projected.")
    st.download_button("Download CSV", t.to_csv(index=False), f"fantasy_{scoring.replace(' ', '_').lower()}.csv", "text/csv")


# ---------------------------------------------------------------- How it works

def page_model():
    doc = C.ROOT / "HOW_IT_WORKS.md"
    st.markdown(doc.read_text(encoding="utf-8") if doc.exists() else "")


# ---------------------------------------------------------------- Edges

EDGE_POSITIONS = ["QB", "RB", "WR", "TE", "Game"]
ODDS_COOLDOWN_MIN = 2   # the Refresh odds button: stops double clicks and public-app spam


@st.cache_data(ttl=600, show_spinner=False)
def _odds(mtime: float) -> pd.DataFrame | None:
    return odds.load()


@st.cache_data(show_spinner="Comparing prices with the projections...", max_entries=8)
def _edges(odds_mtime: float, built: str, ov_key: str) -> tuple[pd.DataFrame, dict]:
    d = data()
    return edges.compare(_odds(odds_mtime), games(), players(), d["tables"], d["margin"], d["total"])


CHECK_MARKETS = {"Passing yards": "player_pass_yds", "Passing TDs": "player_pass_tds", "Completions": "player_pass_completions",
                 "Pass attempts": "player_pass_attempts", "Interceptions": "player_pass_interceptions",
                 "Rushing yards": "player_rush_yds", "Carries": "player_rush_attempts", "Receptions": "player_receptions",
                 "Receiving yards": "player_reception_yds", "Rush + rec yards": "player_rush_reception_yds",
                 "Pass + rush yards": "player_pass_rush_yds", "Anytime TD": "player_anytime_td",
                 "Moneyline": "Moneyline", "Spread": "Spread", "Total": "Total"}
MARKET_POS = {"player_pass_yds": ("QB",), "player_pass_tds": ("QB",), "player_pass_completions": ("QB",),
              "player_pass_attempts": ("QB",), "player_pass_interceptions": ("QB",), "player_pass_rush_yds": ("QB",),
              "player_rush_yds": ("QB", "RB"), "player_rush_attempts": ("RB", "QB"), "player_receptions": ("WR", "TE", "RB"),
              "player_reception_yds": ("WR", "TE", "RB"), "player_rush_reception_yds": ("RB", "WR", "TE"),
              "player_anytime_td": ("RB", "WR", "TE", "QB")}
PLAY_EDGE = 0.05   # below about 5 points the books' margin usually eats the edge


def _odds_in(label: str, default: str, key: str, help: str | None = None) -> float | None:
    raw = st.text_input(label, value=default, key=key, help=help).strip().replace("+", "")
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if abs(v) >= 100 else None


def check_line(expanded: bool = False):
    """Price one line typed in from a book against this week's projection."""
    d = data()
    with st.expander("Check a line from a book", expanded=expanded):
        g = games()
        c1, c2, c3 = st.columns([2, 2, 3])
        label = c1.selectbox("Market", list(CHECK_MARKETS), key="chk_market")
        market = CHECK_MARKETS[label]
        glab = {r.game_id: f"{r.away} @ {r.home}" for r in g.itertuples()}
        gid = c2.selectbox("Game", list(glab), format_func=glab.get, key="chk_game")
        gr = g[g.game_id == gid].iloc[0]
        d1, d2, d3, d4 = st.columns(4)
        side, line = None, None
        if market in ("Moneyline", "Spread", "Total"):
            if market == "Total":
                side = d1.segmented_control("Side", ["Over", "Under"], default="Over", key="chk_ou") or "Over"
                line = d2.number_input("Line", value=float(np.floor(gr.total_hat)) + 0.5, step=0.5, key="chk_tot")
                win_push = edges.game_win_push(d["margin"], d["total"], gr, "Total", True, line, side == "Over")
                proj_text, bet = f"{gr.total_hat:.1f} total points projected", f"{side} {line:g}, {glab[gid]}"
            else:
                team = c3.segmented_control("Team", [gr.away, gr.home], default=gr.away, key="chk_team") or gr.away
                home = team == gr.home
                fair = -(gr.margin_hat if home else -gr.margin_hat)
                line = 0.0 if market == "Moneyline" else d2.number_input(
                    "Spread", value=float(np.floor(fair)) + 0.5, step=0.5, key="chk_sp", help="This team's spread, e.g. -3.5")
                win_push = edges.game_win_push(d["margin"], d["total"], gr, market, home, line)
                own, opp = (gr.home_pts, gr.away_pts) if home else (gr.away_pts, gr.home_pts)
                proj_text = f"projected score {team} {own:.1f} – {opp:.1f}"
                bet = f"{team} {'moneyline' if market == 'Moneyline' else f'{line:+g}'}"
        else:
            p = players()
            pg = p[(p.game_id == gid) & p.projected_position.isin(MARKET_POS[market])].sort_values("fantasy_ppr", ascending=False)
            if pg.empty:
                st.info("No projected players for that market in this game.")
                return
            who = c3.selectbox("Player", pg.index, format_func=lambda i: f"{pg.at[i, 'player_name']} · {pg.at[i, 'projected_position']} · {pg.at[i, 'team']}",
                               key="chk_player")
            r = pg.loc[who]
            if market == "player_anytime_td":
                side = d1.segmented_control("Side", ["Yes", "No"], default="Yes", key="chk_yn") or "Yes"
                p_yes = float(r.p_anytime_td)
                win_push = (p_yes if side == "Yes" else 1 - p_yes, 0.0)
                proj_text, bet = f"{r.anytime_td:.2f} TDs projected", f"{r.player_name} anytime TD {side.lower()}"
            else:
                _, col, _ = edges.PROPS[market]
                mu = edges.mean_of(r, col)
                side = d1.segmented_control("Side", ["Over", "Under"], default="Over", key="chk_p_ou") or "Over"
                line = d2.number_input("Line", value=float(np.floor(mu)) + 0.5, step=0.5, key=f"chk_line_{market}")
                win_push = edges.prop_win_push(d["tables"], r, market, line, side == "Over")
                proj_text, bet = f"{mu:.1f} {label.lower()} projected", f"{r.player_name} {side.lower()} {line:g} {label.lower()}"
        with d3:
            price = _odds_in("Book odds", "-110", "chk_odds", "American odds, e.g. -115 or +130")
        with d4:
            other = _odds_in("Other side's odds (optional)", "", "chk_other",
                             "Adding the other side's price removes the book's margin from its probability")
        if price is None:
            st.warning("Enter the book's odds as an American price, e.g. -115 or +130.")
            return
        ev = edges.evaluate(*win_push, price, other)
        a, b, c, dd, e = st.columns(5)
        a.metric("Our probability", f"{ev['p']:.1%}", border=True, help="Given no push" if ev["push"] > 0 else None)
        b.metric("Our fair odds", american(ev["p"]), border=True)
        c.metric("Book probability", f"{ev['book_p']:.1%}", border=True,
                 help="Margin removed" if ev["novig"] else "Includes the book's margin; add the other side to remove it")
        dd.metric("Edge", f"{100 * ev['edge']:+.1f} pts", border=True)
        e.metric("EV per $100", f"{ev['ev']:+.2f}", border=True)
        msg = f"**{bet}** at {price:+.0f}: {proj_text}."
        if ev["ev"] <= 0 or ev["edge"] < 0.02:
            st.error(f"No edge. {msg} The book's price is at or above the model's fair value.")
        elif ev["edge"] < PLAY_EDGE:
            st.warning(f"Small edge. {msg} Edges this small rarely beat the books' margin.")
        else:
            st.success(f"Edge worth a look. {msg} Check the injury report and the depth chart before playing.")
        if ev["push"] > 0:
            st.caption(f"Push chance {ev['push']:.0%}, refunded.")


def page_edges():
    odds_mtime = _mtime(odds.ODDS_FILE)
    st.title(f"Edges · {week_title()}")
    o = _odds(odds_mtime) if odds.ODDS_FILE.exists() else None
    head, btn = st.columns([5, 1], vertical_alignment="bottom")
    has_key = odds.api_key() is not None
    last = pd.Timestamp(o.fetched_at.iloc[0]) if o is not None and len(o) else None
    wait = last is not None and pd.Timestamp.now(tz="UTC") - last < pd.Timedelta(minutes=ODDS_COOLDOWN_MIN)
    if btn.button("Refresh odds", type="primary", disabled=wait or not has_key, width="stretch",
                  help=("Fetch the latest sportsbook prices now (about 3 API credits plus up to 12 per game). At most "
                        f"once every {ODDS_COOLDOWN_MIN} minutes.") if has_key else "Needs odds_api_key in the app's secrets"):
        with st.spinner("Fetching sportsbook odds..."):
            try:
                odds.fetch(verbose=False)
            except Exception as e:
                st.error(f"Could not fetch odds: {e}")
                st.stop()
        _odds.clear()
        st.rerun()
    if not has_key:
        st.warning('**Refresh odds is off: no Odds API key.** On Streamlit Cloud add `odds_api_key = "..."` under the '
                   "app's Settings -> Secrets (locally: `.streamlit/secrets.toml`). The line checker below works without it.")
    check_line(expanded=o is None or o.empty)
    if o is None or o.empty:
        if has_key:
            head.info("No sportsbook odds yet. Click **Refresh odds** to fetch them.")
        return
    df, stats_ = _edges(odds_mtime, data()["meta"]["built_at"], json.dumps(adjust.load(), sort_keys=True))
    head.caption("Sportsbook prices compared with the model's own probability for the same bet, by position. "
                 "Odds update only when you click Refresh odds.")
    if df.empty:
        st.info("No pregame odds match this week's games.")
        return
    shown = df[df.book_p.notna()].sort_values("dec", ascending=False).drop_duplicates(["game_id", "market", "bet"])
    t_up = pd.Timestamp(o.fetched_at.iloc[0]).tz_convert("America/New_York")
    a, b, c, d_ = st.columns(4)
    a.metric("Odds updated", t_up.strftime("%a %I:%M %p ET"), border=True)
    b.metric("Sportsbooks", f"{df.book.nunique()}", border=True)
    c.metric("Bets priced", f"{len(shown):,}", border=True)
    d_.metric("Positive EV", f"{(shown.ev > 0).sum():,}", border=True)
    pos_tab = st.segmented_control("Position", ["All"] + EDGE_POSITIONS, default="All", key="edge_pos") or "All"
    c1, c2, c3, c4 = st.columns(4)
    game_order = df.sort_values("commence_time").game.drop_duplicates().tolist()
    games_sel = c1.multiselect("Game", game_order, key="edge_games")
    pool = df if pos_tab == "All" else df[df.position == pos_tab]
    pool = pool[pool.game.isin(games_sel)] if games_sel else pool
    plab = pool[pool.team.notna()].drop_duplicates("subject")
    plab = dict(zip(plab.subject, plab.subject + " (" + plab.pos.astype(str) + ", " + plab.team.astype(str) + ")"))
    sel_players = c2.multiselect("Player", sorted(plab), format_func=plab.get, key="edge_players", placeholder="Search a player")
    kinds = [k for k in edges.KIND.values() if k in set(pool.kind)]
    kind = c3.multiselect("Bet type", kinds, key="edge_type")
    books = c4.multiselect("Sportsbooks", sorted(df.book.unique()), key="edge_books")
    c5, c6, c7, c8, _ = st.columns([1, 1, 1, 1, 2], vertical_alignment="bottom")
    min_ev = c5.number_input("Min EV %", value=0.0, step=1.0, format="%.0f", key="edge_min", disabled=bool(sel_players))
    sort = c6.selectbox("Sort by", ["EV", "Edge"], key="edge_sort")
    best = c7.toggle("Best price", value=True, key="edge_best")
    both = c8.toggle("Both sides", value=True, key="edge_both",
                     help="Only bets the book posts both sides of; anytime TDs (Yes only) are always shown, against the "
                          "book's probability with its margin left in.")
    q = pool[pool.subject.isin(sel_players)] if sel_players else pool
    q = q[q.kind.isin(kind)] if kind else q
    q = q[q.book.isin(books)] if books else q
    q = q[q.novig.notna() | q.one_sided] if both else q
    if best:
        q = q.sort_values("dec", ascending=False).drop_duplicates(["game_id", "market", "bet"])
    q = q if sel_players else q[q.ev * 100 >= min_ev]
    q = q.sort_values("ev" if sort == "EV" else "edge", ascending=False)
    t = pd.DataFrame({"Game": q.game, "Pos": q.position, "Bet": q.subject, "Line": q.line, "Projection": q.proj_text,
                      "Book": q.book, "Book odds": [f"{x:+.0f}" for x in q.price], "Our odds": q.p_model.map(american),
                      "Book %": (q.book_p * 100).round(), "Our %": (q.p_model * 100).round(), "Edge": q.edge * 100, "EV": q.ev * 100})
    bar = lambda h: st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100, width="small", help=h)
    st.dataframe(t, hide_index=True, width="stretch", height=720, column_config={
        "Game": st.column_config.TextColumn(width="small", pinned=True),
        "Pos": st.column_config.TextColumn(width="small"),
        "Bet": st.column_config.TextColumn(width="medium", pinned=True),
        "Book": st.column_config.TextColumn(width="small"),
        "Book odds": st.column_config.TextColumn(width="small"), "Our odds": st.column_config.TextColumn(width="small"),
        "Book %": bar("The book's probability with its margin removed (anytime TD: with it)"),
        "Our %": bar("The model's probability, for a game he plays"),
        "Edge": st.column_config.NumberColumn(format="%+.1f", width="small", help="Our % minus Book %"),
        "EV": st.column_config.NumberColumn(format="%+.1f%%", width="small", help="Expected profit per $100 at the book's odds")})
    st.caption("**Book %** the book's probability without its margin · **Our %** the model's · **Edge** Our % − Book % · "
               "**EV** expected profit per $100. Player props are priced for a game he plays (books void them otherwise); "
               "game lines come from the market-free score model. A big edge is more often news the model has not seen "
               "(an injury, a depth-chart change) than a mispriced line: check the injury report first. "
               f"Matched {stats_.get('player props matched', 0):,} of {stats_.get('player props', 0):,} player prices.")
    st.download_button("Download CSV", t.to_csv(index=False), "edges.csv", "text/csv")


# ---------------------------------------------------------------- Results

EDGE_MIN = {"All bets": None, "Edge 0%+": 0.0, "Edge 2%+": 0.02, "Edge 5%+": 0.05, "Edge 8%+": 0.08, "Edge 12%+": 0.12}
BUCKET_LO = {lab: lo for lo, _, lab in tracking.BUCKETS}


@st.cache_data(ttl=600, show_spinner=False)
def _results(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def page_results():
    st.title("Results")
    st.caption("The model's final projections graded against the sportsbooks' opening (10 AM ET game day) and closing "
               "lines: 1-unit bets at the best price of DraftKings, FanDuel, BetMGM, Caesars and Fanatics.")
    p = tracking.RESULTS_LIVE
    r = _results(str(p), p.stat().st_mtime) if p.exists() else pd.DataFrame()
    if r.empty:
        st.info("No graded games yet. `python tracking.py snapshot` stores each week's final projections and "
                "`python tracking.py grade` grades finished games against their opening and closing lines.")
        return
    c1, c2, c3, c4 = st.columns([1, 2, 2, 1], vertical_alignment="bottom")
    snap = c1.segmented_control("Price", ["Close", "Open"], default="Close", key="res_snap") or "Close"
    r = r[r.snap == snap.lower()]
    pos = c2.multiselect("Position", sorted(r.position.unique()), key="res_pos")
    kind = c3.multiselect("Bet type", sorted(r.kind.unique()), key="res_kind")
    th = EDGE_MIN[c4.selectbox("Bets", list(EDGE_MIN), index=3, key="res_edge")]
    q = r[r.position.isin(pos)] if pos else r
    q = q[q.kind.isin(kind)] if kind else q
    sel = q if th is None else q[q.bucket.map(BUCKET_LO) >= th]
    bets, wins, pnl = sel.bets.sum(), sel.wins.sum(), sel.pnl.sum()
    a, b, c, d = st.columns(4)
    a.metric("Bets", f"{int(bets):,}", border=True)
    b.metric("Win rate", f"{wins / bets:.1%}" if bets else "—", border=True)
    c.metric("ROI", f"{pnl / bets:+.1%}" if bets else "—", border=True)
    d.metric("Profit", f"{pnl:+,.1f} u" if bets else "—", border=True)
    g = sel.groupby(["position", "kind", "side"], observed=True)[["bets", "wins", "pnl"]].sum().reset_index()
    g = g[g.bets > 0].assign(**{"Win %": lambda x: 100 * x.wins / x.bets, "ROI": lambda x: 100 * x.pnl / x.bets})
    st.dataframe(g.rename(columns={"position": "Pos", "kind": "Market", "side": "Side", "bets": "Bets", "pnl": "Profit (u)"})
                 [["Pos", "Market", "Side", "Bets", "Win %", "ROI", "Profit (u)"]], hide_index=True, width="stretch",
                 column_config={"Win %": st.column_config.NumberColumn(format="%.1f%%"), "ROI": st.column_config.NumberColumn(format="%+.1f%%"),
                                "Profit (u)": st.column_config.NumberColumn(format="%+.1f")})
    acc = q[q.primary]
    if acc.bets.sum():
        st.caption(f"Accuracy (log loss, lower is better): model **{acc.ll_model.sum() / acc.bets.sum():.4f}** · books "
                   f"**{acc.ll_mkt.sum() / acc.bets.sum():.4f}**, on {int(acc.bets.sum()):,} bets.")


_start = os.environ.get("APP_PAGE", "slate")  # headless tests pick the page to open
PAGES = [("slate", page_slate, "Slate"), ("game", page_game, "Game"), ("depth", page_depth, "Depth charts"),
         ("player", page_player, "Player"), ("edges", page_edges, "Edges"), ("fantasy", page_fantasy, "Fantasy"),
         ("results", page_results, "Results"), ("model", page_model, "How it works")]
st.navigation([st.Page(f, title=t, url_path=k, default=_start == k) for k, f, t in PAGES]).run()
