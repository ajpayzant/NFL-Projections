import pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import polars as pl
import config as C
import availability as A
C.engine_on_path()
from src.projections import week
models = pickle.load(open(A.MODEL_FILE, "rb"))
f = A.frame([2026])
for w in (5, 6):
    pred = A.predict(models, f.filter(pl.col("week") == w))
    lu = A.lineup(pred).select("team", "player_id")
    fc = week.project(2026, w, lineup=lu)
    for pop in ("receiver", "running_back"):
        r = fc.players[pop]
        shares = [c for c in r.columns if c.startswith("p_") and c.endswith("share") and not c.endswith("_raw")]
        sums = r.group_by("team").agg([pl.col(c).sum() for c in shares] + [pl.len().alias("n")]).select(pl.exclude("team").mean())
        print(w, pop, sums.to_pandas().round(3).to_dict("records"))
    fa = week.project(2026, w) if w == 5 else None
    if fa is not None:
        for pop in ("receiver", "running_back"):
            r = fa.players[pop]
            shares = [c for c in r.columns if c.startswith("p_") and c.endswith("share") and not c.endswith("_raw")]
            print(w, pop, "ACTUAL participants", r.group_by("team").agg([pl.col(c).sum() for c in shares] + [pl.len().alias("n")]).select(pl.exclude("team").mean()).to_pandas().round(3).to_dict("records"))
