"""What leaves the app has to be what was on screen, at the precision the engine produced.

Two claims worth pinning, because both fail silently. The first is precision: the app rounds for
display and the export must not, so a share that reads 0.18 on screen has to be 0.18437 in the file.
The second is that Google Sheets *degrades* -- a missing credential is a configuration state and an
export page that raises on it is a page that cannot be opened at all.
"""

from __future__ import annotations

import io
import json

import polars as pl
import pytest

from src.export import sheets


def frame() -> pl.DataFrame:
    return pl.DataFrame({
        "player": ["A", "B", "C"],
        "position": ["WR", "QB", "RB"],
        "fantasy_points": [201.456789, 88.1, 0.0],
        "target_share": [0.184376, 0.0, 0.09115],
        "overall_rank": [1, 2, 3],
        "startable": [True, False, False],
        "history": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    })


def test_the_export_keeps_the_precision_the_screen_throws_away():
    got = sheets.csv_bytes(frame()).decode()
    assert "201.456789" in got
    assert "0.184376" in got
    # the display rounding `ui.rounded` applies is genuinely lossy, so this is not a vacuous check
    rounded = sheets._flat(frame()).with_columns(pl.col(pl.Float64).round(2))
    assert "201.46" in rounded.write_csv()


def test_a_nested_column_is_named_rather_than_mangled():
    f = frame()
    assert sheets.dropped_columns(f) == ["history"]
    got = sheets.csv_bytes(f).decode()
    assert "history" not in got.splitlines()[0]
    # every other column survives
    assert got.splitlines()[0].split(",") == ["player", "position", "fantasy_points", "target_share",
                                              "overall_rank", "startable"]


def test_an_empty_table_is_not_offered_and_a_missing_one_does_not_raise():
    ex = sheets.ExportSet().add("board", frame()).add("nothing", frame().head(0)).add("absent", None)
    assert ex.names == ["board"]
    assert len(ex) == 1
    assert ex.named("board") is not None
    assert ex.named("nothing") is None


def test_the_workbook_holds_every_table_plus_a_readme_that_names_the_scenario():
    openpyxl = pytest.importorskip("openpyxl")
    ex = sheets.ExportSet().add("board", frame(), "the ranking board").add("weekly", frame())
    raw = sheets.workbook_bytes(ex, scenario="my scenario (abc123)", season=2026,
                                provenance="dispersion calibrated")
    book = openpyxl.load_workbook(io.BytesIO(raw))
    assert book.sheetnames == ["README", "board", "weekly"]
    readme = "\n".join(
        str(c.value) for row in book["README"].iter_rows() for c in row if c.value is not None
    )
    assert "my scenario (abc123)" in readme
    assert "2026" in readme
    # the README has to name every sheet, or a file with eight tabs is unreadable
    assert "sheet: board" in readme and "sheet: weekly" in readme
    # the values are the engine's, not the display's
    values = [c.value for row in book["board"].iter_rows() for c in row]
    assert pytest.approx(201.456789) in [v for v in values if isinstance(v, float)]


def test_a_sheet_name_is_legal_and_two_long_names_do_not_collide():
    taken: set[str] = set()
    first = sheets._sheet_name("x" * 40, taken)
    second = sheets._sheet_name("x" * 40, taken)
    assert len(first) <= sheets.MAX_SHEET_NAME and len(second) <= sheets.MAX_SHEET_NAME
    assert first != second, "truncation that collides would silently drop a table"
    assert sheets._sheet_name("a/b:c*d", set()) == "a-b-c-d"


def test_a_share_and_a_point_total_do_not_print_alike():
    fmt = sheets._formats(frame())
    assert fmt["target_share"] == sheets.FORMATS["share"]
    assert fmt["fantasy_points"] == sheets.FORMATS["points"]
    assert fmt["overall_rank"] == sheets.FORMATS["integer"]
    # a boolean and a string carry no number format at all
    assert "startable" not in fmt and "player" not in fmt


# --------------------------------------------------------------------------- #
# Google Sheets degrades rather than failing
# --------------------------------------------------------------------------- #
def test_sheets_says_what_is_missing_rather_than_only_that_something_is(monkeypatch, tmp_path):
    """Three separable states, because each needs a different action from the user."""
    pytest.importorskip("gspread")      # without it the first state is the only one reachable
    monkeypatch.setenv(sheets.ENV_CREDENTIALS, str(tmp_path / "nope.json"))
    absent = sheets.sheets_status()
    assert not absent.ready
    assert "credentials" in absent.reason
    assert sheets.ENV_CREDENTIALS in absent.remedy or str(tmp_path) in absent.remedy

    wrong = tmp_path / "oauth.json"
    wrong.write_text(json.dumps({"type": "authorized_user", "client_id": "x"}))
    monkeypatch.setenv(sheets.ENV_CREDENTIALS, str(wrong))
    got = sheets.sheets_status()
    assert not got.ready
    assert "service-account" in got.reason
    assert "service account" in got.remedy

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    monkeypatch.setenv(sheets.ENV_CREDENTIALS, str(broken))
    assert not sheets.sheets_status().ready

    ok = tmp_path / "sa.json"
    ok.write_text(json.dumps({"type": "service_account", "client_email": "bot@x.iam.gserviceaccount.com"}))
    monkeypatch.setenv(sheets.ENV_CREDENTIALS, str(ok))
    ready = sheets.sheets_status()
    assert ready.ready
    assert ready.account == "bot@x.iam.gserviceaccount.com"
    # the remedy is still populated when ready: the sheet must be shared with the account
    assert "shared" in ready.remedy


def test_writing_to_sheets_unconfigured_raises_something_a_user_can_act_on(monkeypatch, tmp_path):
    monkeypatch.setenv(sheets.ENV_CREDENTIALS, str(tmp_path / "nope.json"))
    with pytest.raises(RuntimeError, match="credentials"):
        sheets.to_google_sheets(sheets.ExportSet().add("board", frame()), "title")


def test_a_credential_is_never_looked_for_inside_the_repository(monkeypatch):
    """The export is worth less than the accident of committing a private key."""
    from src.config import ROOT

    monkeypatch.delenv(sheets.ENV_CREDENTIALS, raising=False)
    assert not sheets.DEFAULT_CREDENTIALS.is_relative_to(ROOT)


def test_the_cell_coercion_survives_what_the_sheets_api_will_not_take():
    assert sheets._cell(None) == ""
    assert sheets._cell(float("nan")) == ""
    assert sheets._cell(float("inf")) == ""
    assert sheets._cell(1.5) == 1.5
    assert sheets._cell(True) is True
    assert sheets._cell("x") == "x"


def test_write_files_puts_every_table_and_one_workbook_on_disk(tmp_path):
    ex = sheets.ExportSet().add("board", frame()).add("weekly", frame())
    written = sheets.write_files(ex, tmp_path, stem="proj")
    names = sorted(p.name for p in written)
    assert names == ["proj.xlsx", "proj_board.csv", "proj_weekly.csv"]
    assert all(p.stat().st_size > 0 for p in written)
