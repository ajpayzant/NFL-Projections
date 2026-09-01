"""Let the suite run on a machine that has no parquet lake -- a CI runner, or a fresh clone.

The lake is somebody else's (`~/nfl-projection-system/data/processed`, 15 MB of played-game tables)
and it is deliberately not in this repo. Without it, most of this suite has nothing to assert against:
392 tests become 34 failures and 164 errors, which is not a signal anybody can read. The point of CI
is to catch a broken import, a renamed column in a hand-built frame, a page that stopped rendering --
and every one of those is testable with no lake at all.

So when the lake is absent, a test that reaches for it **skips**, and everything else runs normally.

The mechanism is deliberately the narrowest one that works: `lake.read` is the single door onto the
lake (everything else in `src/` reads only our own `data/fitted`), and it already raises
`FileNotFoundError` naming what it looked for. Here that becomes `pytest.skip`, which pytest honours
from inside a fixture -- so a module-scoped fixture that reads the lake skips the tests that asked for
it, and the arithmetic tests in the same module that took no fixtures still run and still count.

What this does not do: soften a failure on a machine that *has* the lake. The wrapper is only
installed when there is no lake to read, so the local gate is unchanged -- all 392 tests, no skips
from here.
"""

from __future__ import annotations

import pytest

from src.config import PROCESSED
from src.data import lake


def lake_present() -> bool:
    """A lake that actually holds a table, not merely a directory somebody created."""
    return PROCESSED.is_dir() and any(PROCESSED.glob("*/season=*/*.parquet"))


def pytest_report_header() -> str:
    return (f"lake: {PROCESSED} " +
            ("present" if lake_present() else "ABSENT -- tests that read it will skip"))


def pytest_configure(config: pytest.Config) -> None:
    if lake_present():
        return

    real = lake.read

    def read_or_skip(*args, **kwargs):
        try:
            return real(*args, **kwargs)
        except FileNotFoundError as exc:
            pytest.skip(f"needs the parquet lake: {exc}")

    # `lake.clear_cache()` calls `read.cache_clear()` through the module global, and the refresh path
    # calls it -- so the replacement has to carry that attribute or it breaks a test it never touched.
    read_or_skip.cache_clear = real.cache_clear
    lake.read = read_or_skip
