"""Market-free game model variants, walk-forward 2019-2026: margin MAE, ML log loss, total MAE."""
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gamelines as G

allg = G.game_rows(list(range(2016, 2027)))
cols = G.feature_cols(allg)
CORE = re.compile(r"^(opp_)?(off|def)_(points|epa_per_play|epa_per_dropback|epa_per_rush|success_rate|plays|yards_per_attempt|explosive_pass_rate|sack_rate)_(est|ewma|season)$")
core = [c for c in cols if CORE.match(c)] + [c for c in cols if c in ("rest_days", "opp_rest_days", "qb_change", "qb_games_with_team_prior",
                                                                      "coach_change", "is_indoor", "div_game", "neutral_site", "off_bye", "short_week", "league_year", "temp", "wind")]
print("all features", len(cols), "| core", len(core))


def lgbm(params, rounds):
    def f(trX, y, teX):
        return lgb.train(params, lgb.Dataset(trX, y), rounds).predict(teX)
    return f


def ridge(trX, y, teX):
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 13)))
    return m.fit(trX, y).predict(teX)


VARIANTS = {"lgbm all (current)": (cols, lgbm(G.PARAMS, G.ROUNDS)),
            "ridge all": (cols, ridge),
            "ridge core": (core, ridge),
            "lgbm core": (core, lgbm({**G.PARAMS, "feature_fraction": 0.7, "num_leaves": 7}, 500))}
res = {k: {"m": [], "t": [], "ym": [], "yt": [], "ms": [], "mt": []} for k in VARIANTS}
for s in range(2019, 2027):
    tr = allg.filter((pl.col("season") < s) & pl.col("margin").is_not_null())
    te = allg.filter((pl.col("season") == s) & pl.col("margin").is_not_null() & pl.col("mkt_spread").is_not_null())
    for k, (cc, fn) in VARIANTS.items():
        trX, teX = G._X(tr, cc), G._X(te, cc)
        for tgt, key in (("margin", "m"), ("total", "t")):
            res[k][key].append(fn(trX, tr[tgt].to_numpy().astype(float), teX))
        res[k]["ym"].append(te["margin"].to_numpy().astype(float)); res[k]["yt"].append(te["total"].to_numpy().astype(float))
        res[k]["ms"].append(te["mkt_spread"].to_numpy().astype(float)); res[k]["mt"].append(te["mkt_total"].to_numpy().astype(float))
    print("season", s, "done", flush=True)
for k, r in res.items():
    m, t, ym, yt, ms, mt = (np.concatenate(r[x]) for x in ("m", "t", "ym", "yt", "ms", "mt"))
    keep = ym != 0
    sd = np.std(ym - m)
    ll = lambda p: float(-np.mean((ym[keep] > 0) * np.log(np.clip(p[keep], 1e-6, 1)) + (ym[keep] < 0) * np.log(np.clip(1 - p[keep], 1e-6, 1))))
    b = 0.5 * m + 0.5 * ms
    print(f"{k:20s} margin MAE {np.abs(m - ym).mean():.3f} | ML ll {ll(stats.norm.cdf(m / sd)):.4f} | total MAE {np.abs(t - yt).mean():.3f} "
          f"| corr w/ market {np.corrcoef(m, ms)[0, 1]:.3f}")
print(f"{'market':20s} margin MAE {np.abs(ms - ym).mean():.3f} | ML ll {ll(stats.norm.cdf(ms / np.std(ym - ms))):.4f} | total MAE {np.abs(mt - yt).mean():.3f}")
