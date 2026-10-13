import pickle, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import polars as pl
import config as C
import availability as A
C.engine_on_path()
from src.features import expected
from src.data import storage
from src.projections import week
models = pickle.load(open(A.MODEL_FILE, "rb"))
pred = A.predict(models, A.frame([2026]).filter(pl.col("week") == 6))
lu = A.lineup(pred).join(pred.select("game_id", "team", "player_id", "season", "week", "pos"), on=["game_id", "team", "player_id"])
expected.write(lu.select("season", "week", "team", "player_id", pl.col("pos").alias("position")))
r = subprocess.run([sys.executable, "scripts/build_features.py", "--seasons", "2026", "--include-scheduled"], cwd=C.ENGINE, capture_output=True, text=True)
print("features:", r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-500:])
fc = week.project(2026, 6, lineup=lu.select("team", "player_id"))
for pop in ("receiver", "running_back"):
    rr = fc.players[pop]
    shares = [c for c in rr.columns if c.startswith("p_") and c.endswith("share") and not c.endswith("_raw")]
    print(pop, rr.group_by("team").agg([pl.col(c).sum() for c in shares] + [pl.len().alias("n")]).select(pl.exclude("team").mean()).to_pandas().round(3).to_dict("records"))
sk = pl.concat([fc.players["running_back"].select("team", "p_targets_chain", "p_carries_chain"),
                fc.players["receiver"].select("team", "p_targets_chain").with_columns(pl.lit(0.0).alias("p_carries_chain"))], how="diagonal")
ts = sk.group_by("team").agg(pl.col("p_targets_chain").sum(), pl.col("p_carries_chain").sum())
print("team skill targets", round(ts["p_targets_chain"].mean(), 1), "(2026 actual 31.5) | RB carries", round(ts["p_carries_chain"].mean(), 1), "(actual 22.1)")
q = fc.players["quarterback"]; print("QB dropback share", round(q["p_dropback_share"].mean(), 3), "passing yds", round(q["p_passing_yards_chain"].mean(), 1))
wr = fc.players["receiver"].sort("p_receiving_yards_chain", descending=True)
print(wr.select("player_name", "team", "p_targets_chain", "p_receptions_chain", "p_receiving_yards_chain").head(8).to_pandas().round(1).to_string(index=False))
rb = fc.players["running_back"].sort("p_rushing_yards_chain", descending=True)
print(rb.select("player_name", "team", "p_carries_chain", "p_rushing_yards_chain", "p_targets_chain").head(8).to_pandas().round(1).to_string(index=False))
