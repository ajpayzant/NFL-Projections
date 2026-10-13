"""Outcome distributions for player props: P(over a line) and quantiles, given a projection.

No engine imports, so the app (and the cloud app) can price any line from the saved tables.
    counts (receptions, targets, carries, attempts, completions, interceptions)
        negative binomial, var = c1 mu + c2 mu^2 (calibrated within 0.3 pts at book lines)
    yards, touchdowns, fantasy points
        the empirical distribution of outcome / projection among similar projections, which the
        negative binomial under-called by 2-5 pts at the lower yardage lines (research/distributions.py)
"""
from __future__ import annotations

import numpy as np
from scipy import stats

BINS = 20
EMPIRICAL = {"passing_yards", "rushing_yards", "receiving_yards", "passing_tds", "rushing_tds", "receiving_tds",
             "fantasy_ppr", "fantasy_half", "fantasy_std", "rush_rec_yards", "pass_rush_yards", "anytime_td"}


class Empirical:
    """P(Y > line | mu) from history: outcome / projection among rows whose projection is near mu."""

    def __init__(self, mu: np.ndarray, y: np.ndarray):
        lm = np.log(np.clip(mu, 1e-3, None))
        self.edges = np.quantile(lm, np.linspace(0, 1, BINS + 1))
        self.edges[0], self.edges[-1] = -np.inf, np.inf
        b = np.clip(np.searchsorted(self.edges, lm, side="right") - 1, 0, BINS - 1)
        self.groups = [np.sort(y[(b >= k - 1) & (b <= k + 1)] / np.clip(mu[(b >= k - 1) & (b <= k + 1)], 1e-3, None))
                       for k in range(BINS)]

    def _bin(self, mu: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.edges, np.log(np.clip(mu, 1e-3, None)), side="right") - 1, 0, BINS - 1)

    def p_over(self, mu, line: float) -> np.ndarray:
        mu = np.atleast_1d(np.asarray(mu, float))
        b = self._bin(mu)
        out = np.empty(len(mu))
        for k in np.unique(b):
            i = b == k
            r = self.groups[k]
            out[i] = 1 - np.searchsorted(r, line / np.clip(mu[i], 1e-3, None), side="right") / len(r)
        return np.clip(out, 1e-4, 1 - 1e-4)

    def p_at_least_one(self, mu) -> np.ndarray:
        return self.p_over(mu, 0.5)

    def quantile(self, mu, q: float) -> np.ndarray:
        mu = np.atleast_1d(np.asarray(mu, float))
        return np.array([np.quantile(self.groups[k], q) for k in self._bin(mu)]) * mu


def nb_over(mu, c, line: float) -> np.ndarray:
    mu = np.clip(np.atleast_1d(np.asarray(mu, float)), 1e-6, None)
    var = np.maximum(c[0] * mu + c[1] * mu ** 2, mu * 1.0001)
    n = mu ** 2 / (var - mu)
    return stats.nbinom.sf(np.floor(line), n, n / (n + mu))


def p_over(tables: dict, group: str, stat: str, mu, line: float) -> np.ndarray:
    """P(over line) for a game he plays (books void a prop otherwise)."""
    d = tables[f"{group}|{stat}"]
    return d["emp"].p_over(mu, line) if stat in EMPIRICAL else nb_over(mu, d["var"], line)


def win_push(tables: dict, group: str, stat: str, mu: float, line: float, over: bool) -> tuple[float, float]:
    """(P(win), P(push)) for one over / under; whole-number lines can push."""
    gt = float(p_over(tables, group, stat, [mu], line)[0])
    if float(line).is_integer():
        ge = float(p_over(tables, group, stat, [mu], line - 0.5)[0])
        return (gt if over else 1 - ge), max(ge - gt, 0.0)
    return (gt if over else 1 - gt), 0.0


class GameDist:
    """Final margin and total given the model's projection, read from history.

    NFL margins pile up on key numbers (3, 7, 10) and a normal around the projection does not know it,
    so a spread of -3 or -3.5 was priced as if 3 were any other number. Here the outcome for a projected
    margin m is the distribution of actual margins over walk-forward games whose projection was near m
    (Gaussian kernel on the projection): it carries the key numbers and the model's own out-of-sample
    error at once (P(margin = 3) is 8.7% near a 3-point projection; a normal gives 3.1%). Same for totals.
    """

    def __init__(self, pred: np.ndarray, actual: np.ndarray, bandwidth: float = 2.0):
        self.pred, self.actual, self.h = np.asarray(pred, float), np.asarray(actual, float), bandwidth

    def pmf(self, m: float) -> tuple[np.ndarray, np.ndarray]:
        w = np.exp(-0.5 * ((self.pred - m) / self.h) ** 2)
        vals, idx = np.unique(np.round(self.actual), return_inverse=True)
        mass = np.bincount(idx, weights=w)
        return vals, mass / mass.sum()

    def p_over(self, m: float, line: float) -> tuple[float, float]:
        """(P(outcome > line), P(outcome == line))."""
        v, p = self.pmf(m)
        return float(p[v > line].sum()), float(p[v == line].sum())
