"""Blending and shrinkage arithmetic, including the leakage guard."""

from __future__ import annotations

import polars as pl
import pytest

from src.model.blend import blend_counts, regress_to_mean, season_weights, shrink, shrink_weight


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["a", "a", "a", "a", "b"],
            "season": [2022, 2023, 2024, 2025, 2024],
            "targets": [10, 20, 40, 999, 5],
            "team_targets": [100, 100, 100, 999, 100],
            "games": [17, 17, 17, 17, 8],
        }
    )


def test_season_weights_ignores_the_target_season_and_later():
    w = _frame().select("season", season_weights(target=2025))["recency_weight"].to_list()
    #     2022 (3 back)  2023  2024  2025 = the target itself  2024
    assert w == [2.0, 3.0, 5.0, 0.0, 5.0]


def test_blend_counts_excludes_the_target_season():
    out = blend_counts(_frame(), 2025, ["targets", "team_targets"], scale=False)
    a = out.filter(pl.col("player_id") == "a")
    # 2024*5 + 2023*3 + 2022*2 = 40*5 + 20*3 + 10*2 = 280; the 999 in 2025 must not appear
    assert a["targets"][0] == pytest.approx(280.0)
    assert a["team_targets"][0] == pytest.approx(1000.0)
    assert a["seasons_used"][0] == 3


def test_scaling_preserves_ratios_and_puts_counts_on_a_one_season_footing():
    raw = blend_counts(_frame(), 2025, ["targets", "team_targets"], scale=False)
    scaled = blend_counts(_frame(), 2025, ["targets", "team_targets"], scale=True)
    r, s = raw.filter(pl.col("player_id") == "a"), scaled.filter(pl.col("player_id") == "a")
    assert s["targets"][0] == pytest.approx(r["targets"][0] / 10.0)
    ratio_raw = r["targets"][0] / r["team_targets"][0]
    ratio_scaled = s["targets"][0] / s["team_targets"][0]
    assert ratio_raw == pytest.approx(ratio_scaled)
    # a full three seasons of a 100-target pool blends back to roughly one season of it
    assert s["team_targets"][0] == pytest.approx(100.0)


def test_a_single_season_of_history_carries_less_evidence_than_three():
    out = blend_counts(_frame(), 2025, ["team_targets"])
    one = out.filter(pl.col("player_id") == "b")["team_targets"][0]
    three = out.filter(pl.col("player_id") == "a")["team_targets"][0]
    assert one < three


def _used(df: pl.DataFrame, k: float) -> list[float]:
    return df.select(shrink("obs", "prior", "n", k).alias("used"))["used"].to_list()


def test_shrink_endpoints_and_monotonicity():
    df = pl.DataFrame({"obs": [0.30, None], "prior": [0.10, 0.10], "n": [100.0, 0.0]})
    assert _used(df, 0.0) == [0.30, 0.10]
    # no sample means no evidence: the answer is the prior, not zero
    assert _used(df, 50.0)[1] == pytest.approx(0.10)
    values = [_used(df, k)[0] for k in (0, 25, 100, 10_000, 1_000_000)]
    assert values == sorted(values, reverse=True)
    assert values[-1] == pytest.approx(0.10, abs=1e-3)


def test_shrink_weight_is_the_fraction_of_the_player_in_the_answer():
    df = pl.DataFrame({"n": [60.0]})
    assert df.select(shrink_weight("n", 60.0).alias("w"))["w"][0] == pytest.approx(0.5)


def test_regress_to_mean_keeps_the_stated_fraction_of_the_signal():
    df = pl.DataFrame({"v": [30.0, None], "m": [20.0, 20.0]})
    out = df.select(regress_to_mean("v", "m", 0.4).alias("est"))["est"].to_list()
    assert out[0] == pytest.approx(24.0)
    assert out[1] == pytest.approx(20.0)
