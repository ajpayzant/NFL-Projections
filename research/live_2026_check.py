"""Out-of-sample check of the live engine path on 2026 weeks 2-5 (models trained through 2025).
Each week is projected on its actual participants -- the backtest's population -- and scored
against the box score, next to the 2021-2025 replay's bias for the same lines."""
import polars as pl
from src.data import storage
from src.projections import week

pu = storage.load("player_usage", layer="processed", seasons=[2026])
pg = storage.load("player_games", layer="processed", seasons=[2026])
LINES = {"quarterback": ["attempts", "completions", "passing_yards", "passing_tds", "rushing_yards"],
         "running_back": ["carries", "rushing_yards", "targets", "receptions", "receiving_yards"],
         "receiver": ["targets", "receptions", "receiving_yards", "receiving_tds"]}
rows = []
for w in (2, 3, 4, 5):
    f = week.project(2026, w)
    for pop, stats in LINES.items():
        p = f.players.get(pop)
        if p is None or p.is_empty():
            continue
        j = p.join(pg.select("game_id", "player_id", *[pl.col(s).alias(f"a_{s}") for s in stats]), on=["game_id", "player_id"], how="inner")
        for s in stats:
            col = f"p_{s}_chain" if f"p_{s}_chain" in j.columns else f"p_{s}"
            if col not in j.columns:
                continue
            rows.append({"pop": pop, "stat": s, "week": w, "n": j.height, "proj": j[col].mean(), "actual": j[f"a_{s}"].mean(),
                         "mae": (j[col] - j[f"a_{s}"]).abs().mean()})
R = pl.DataFrame(rows).group_by("pop", "stat").agg(pl.col("n").sum(), pl.col("proj").mean(), pl.col("actual").mean(), pl.col("mae").mean())
R = R.with_columns(((pl.col("proj") / pl.col("actual") - 1) * 100).alias("bias_%")).sort("pop", "stat")
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(160)
print(R.with_columns(pl.col(pl.Float64).round(2)))
