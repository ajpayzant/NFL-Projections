"""Getting a projection out of the app: CSV, a formatted workbook, and Google Sheets.

The workbook this engine replaces was also how its numbers travelled -- somebody emailed the xlsx.
So export is not a convenience here, it is the last mile, and it has one rule: **what leaves is what
was on screen.** The page hands over the frame it just rendered, filters and scenario edits already
applied, and nothing here re-derives a number. That is why this module takes frames rather than a
season and a `Settings`: an exporter that recomputed would eventually disagree with the table the
user was looking at, and the disagreement would be invisible.

**Full precision on the way out, rounding only for display.** `ui.table` rounds to two decimals
because a screen is for reading; a file is for computing with, so the frames are written as they come
out of the engine. A share that reads 0.18 on screen is 0.18437 in the CSV, and a user who divides by
it gets the engine's answer rather than the screen's.

Three destinations, and they are not the same promise:

- **CSV** -- one table, no formatting, nothing to interpret. The format that will still open in ten
  years, and the only one with no dependency.
- **Excel** -- multi-sheet and formatted: frozen header, autofit columns, a number format per column
  family, and a `README` sheet naming the scenario and the moment. This is the workbook's structure
  without its formula engine, which is the point of the rebuild -- the values are computed by the
  engine the backtest scored, so a cell cannot be edited into disagreeing with its own inputs.
- **Google Sheets** -- needs a service-account JSON, which is a credential this repo must never carry.
  So it is the one destination that can be *unavailable*, and it says so: `sheets_status` reports
  exactly what is missing -- the library, the file, or the share -- and the page shows that instead of
  the button. It degrades rather than failing, because a missing credential is a configuration state
  and not an error.

Credentials are found in this order, and none of them is a path inside the repo: `$NFLSP_GSPREAD_JSON`
if set, else `~/.config/nflsp/service-account.json`. Both are outside the tree on purpose, so no
plausible `git add` can commit one.
"""

from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

# Where a service account is looked for. Deliberately outside the repository: the export is worth less
# than the accident of committing a private key.
ENV_CREDENTIALS = "NFLSP_GSPREAD_JSON"
DEFAULT_CREDENTIALS = Path.home() / ".config" / "nflsp" / "service-account.json"

# Excel caps a sheet name at 31 characters and forbids these. Silently truncating a name would put two
# tables on one sheet, so `_sheet_name` de-duplicates as well as trims.
BAD_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")
MAX_SHEET_NAME = 31

# Number formats by column family. A share and a point total should not print the same way, and the
# alternative -- one format for every float -- is what made the old workbook unreadable.
SHARE_HINTS = ("share", "rate", "pct", "participation", "volatility", "weight", "factor", "ratio")
INTEGER_HINTS = ("rank", "week", "season", "n", "players", "files", "draft_pick", "tier", "slot")

FORMATS = {
    "share": "0.000",
    "points": "0.0",
    "count": "0.00",
    "integer": "0",
    "text": None,
}


@dataclass(frozen=True)
class Sheet:
    """One table on its way out, with the name it should carry and a line explaining it.

    `note` is written into the workbook's README rather than above the table: a sheet whose first row
    is prose cannot be read by anything but a human, and these files are meant to be loaded as well as
    looked at.
    """

    name: str
    frame: pl.DataFrame
    note: str = ""


@dataclass(frozen=True)
class Availability:
    """Whether Google Sheets can be reached, and if not, precisely what is missing.

    Three separable states, because they need three different answers from the user: install a
    package, put a file somewhere, or share a spreadsheet with the service account. Collapsing them
    into one "not configured" is what makes an integration like this impossible to set up.
    """

    ready: bool
    reason: str = ""
    remedy: str = ""
    account: str = ""
    source: str = ""

    @property
    def message(self) -> str:
        return self.reason if not self.remedy else f"{self.reason} — {self.remedy}"


@dataclass
class ExportSet:
    """The tables a page is offering, in the order they should appear in a workbook."""

    sheets: list[Sheet] = field(default_factory=list)

    def add(self, name: str, frame: pl.DataFrame | None, note: str = "") -> ExportSet:
        """Append a table, skipping one that is empty or absent. Returns self so calls can chain."""
        if frame is not None and frame.height:
            self.sheets.append(Sheet(name=name, frame=frame, note=note))
        return self

    def named(self, name: str) -> Sheet | None:
        return next((s for s in self.sheets if s.name == name), None)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.sheets]

    def __len__(self) -> int:
        return len(self.sheets)


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def csv_bytes(frame: pl.DataFrame) -> bytes:
    """One table as CSV at full precision.

    Nested columns are the one thing CSV cannot carry: a player's history sparkline is a list in the
    frame and would be written as a Python repr. They are dropped rather than mangled, and
    `dropped_columns` says which so a caller can tell the user.
    """
    return _flat(frame).write_csv().encode("utf-8")


def dropped_columns(frame: pl.DataFrame) -> list[str]:
    """Columns CSV cannot represent, so a page can name them instead of losing them silently."""
    return [n for n, t in zip(frame.columns, frame.dtypes, strict=True) if _nested(t)]


def _nested(dtype: pl.DataType) -> bool:
    return isinstance(dtype, pl.List | pl.Array | pl.Struct)


def _flat(frame: pl.DataFrame) -> pl.DataFrame:
    keep = [n for n, t in zip(frame.columns, frame.dtypes, strict=True) if not _nested(t)]
    return frame.select(keep)


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
def _sheet_name(raw: str, taken: set[str]) -> str:
    """A legal, unique Excel sheet name. Truncation that collides would drop a table."""
    clean = BAD_SHEET_CHARS.sub("-", raw).strip() or "sheet"
    out = clean[:MAX_SHEET_NAME]
    n = 2
    while out.lower() in taken:
        suffix = f"~{n}"
        out = clean[: MAX_SHEET_NAME - len(suffix)] + suffix
        n += 1
    taken.add(out.lower())
    return out


def _family(name: str, dtype: pl.DataType) -> str:
    """Which number format a column takes, from its name and its type."""
    if dtype == pl.Boolean or dtype == pl.String or not dtype.is_numeric():
        return "text"
    low = name.lower()
    if dtype.is_integer() or low in INTEGER_HINTS or low.endswith("_rank"):
        return "integer"
    if any(h in low for h in SHARE_HINTS):
        return "share"
    if "points" in low or low.startswith("p") and low[1:].isdigit():
        return "points"
    return "count"


def _formats(frame: pl.DataFrame) -> dict[str, str]:
    out = {}
    for name, dtype in zip(frame.columns, frame.dtypes, strict=True):
        fmt = FORMATS[_family(name, dtype)]
        if fmt:
            out[name] = fmt
    return out


def _readme(sheets: list[Sheet], scenario: str, season: int, provenance: str) -> pl.DataFrame:
    """The sheet that says what this file is. Every export carries one."""
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    rows = [
        {"item": "season", "value": str(season), "detail": ""},
        {"item": "scenario", "value": scenario, "detail": "the adjustments applied to every table"},
        {"item": "exported", "value": when, "detail": ""},
        {"item": "engine", "value": provenance, "detail": ""},
        {"item": "precision", "value": "full",
         "detail": "values are the engine's, not the two decimals the app displays"},
    ]
    rows += [
        {"item": f"sheet: {s.name}", "value": f"{s.frame.height:,} rows x {s.frame.width} columns",
         "detail": s.note}
        for s in sheets
    ]
    return pl.DataFrame(rows)


def workbook_bytes(
    exports: ExportSet,
    scenario: str = "baseline",
    season: int = 0,
    provenance: str = "",
) -> bytes:
    """A formatted multi-sheet workbook: README first, then one sheet per table.

    Written through a single `xlsxwriter` workbook so every sheet lands in one file, with the header
    frozen and columns autofit. `polars.write_excel` does the formatting; what is added here is the
    README, the sheet-name hygiene, and a per-column-family number format so a share and a point total
    do not print alike.
    """
    import xlsxwriter

    buffer = io.BytesIO()
    taken: set[str] = set()
    with xlsxwriter.Workbook(buffer, {"in_memory": True, "nan_inf_to_errors": True}) as book:
        readme = _readme(exports.sheets, scenario, season, provenance)
        readme.write_excel(
            workbook=book, worksheet=_sheet_name("README", taken),
            autofit=True, header_format={"bold": True},
        )
        for sheet in exports.sheets:
            frame = _flat(sheet.frame)
            frame.write_excel(
                workbook=book, worksheet=_sheet_name(sheet.name, taken),
                autofit=True, freeze_panes=(1, 0),
                header_format={"bold": True, "bg_color": "#EEEEEE", "border": 1},
                column_formats=_formats(frame),
            )
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# Google Sheets
# --------------------------------------------------------------------------- #
def credentials_path() -> Path | None:
    """Where a service account would be, if there is one. Never a path inside the repo."""
    env = os.environ.get(ENV_CREDENTIALS)
    if env:
        p = Path(env).expanduser()
        return p if p.is_file() else None
    return DEFAULT_CREDENTIALS if DEFAULT_CREDENTIALS.is_file() else None


def sheets_status() -> Availability:
    """Can we write to Google Sheets, and if not, what exactly is missing?

    Checked in the order the user has to fix it: the library, then the file, then whether the file is
    a service account at all. Nothing here contacts Google -- a status check that needed the network
    would make the page slow and would fail for a reason the user cannot act on.
    """
    try:
        import gspread  # noqa: F401
    except ImportError:
        return Availability(
            ready=False, reason="the `gspread` library is not installed",
            remedy="`pip install gspread google-auth`",
        )
    path = credentials_path()
    if path is None:
        where = os.environ.get(ENV_CREDENTIALS) or str(DEFAULT_CREDENTIALS)
        return Availability(
            ready=False, reason="no service-account credentials found",
            remedy=f"put a service-account JSON at `{where}`, or set `{ENV_CREDENTIALS}`",
        )
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Availability(ready=False, reason=f"credentials at `{path}` could not be read: {exc}",
                            remedy="check the file is the JSON Google gave you, unmodified")
    email = blob.get("client_email", "")
    if blob.get("type") != "service_account" or not email:
        return Availability(
            ready=False, reason=f"`{path}` is not a service-account key",
            remedy="download a *service account* key, not an OAuth client secret",
        )
    return Availability(ready=True, account=email, source=str(path),
                        reason=f"ready as `{email}`",
                        remedy="the target spreadsheet must be shared with that address")


SCOPES = ("https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive")


def to_google_sheets(exports: ExportSet, title: str, share_with: str = "") -> str:
    """Write every table to one spreadsheet, a worksheet each, and return its URL.

    Raises `RuntimeError` when it is not configured rather than returning a sentinel, because by the
    time this is called the page has already checked `sheets_status` and shown the reason. A service
    account owns nothing a human can see, so `share_with` is how the file reaches an actual person --
    without it the sheet exists and is invisible.
    """
    status = sheets_status()
    if not status.ready:
        raise RuntimeError(status.message)

    import gspread
    from google.oauth2.service_account import Credentials

    creds = Credentials.from_service_account_file(status.source, scopes=list(SCOPES))
    client = gspread.authorize(creds)
    book = client.create(title)
    if share_with:
        book.share(share_with, perm_type="user", role="writer")

    first = True
    for sheet in exports.sheets:
        frame = _flat(sheet.frame)
        rows = [frame.columns, *[[_cell(v) for v in row] for row in frame.iter_rows()]]
        if first:
            tab = book.sheet1
            tab.update_title(_sheet_name(sheet.name, set()))
            first = False
        else:
            tab = book.add_worksheet(title=_sheet_name(sheet.name, set()),
                                     rows=len(rows) + 1, cols=max(frame.width, 1))
        tab.update(rows, "A1")
        tab.freeze(rows=1)
    return book.url


def _cell(value) -> object:
    """JSON-safe scalars for the Sheets API, which will not take a date or a NaN."""
    if value is None:
        return ""
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else ""
    return str(value)


# --------------------------------------------------------------------------- #
# writing to disk, for the command line
# --------------------------------------------------------------------------- #
def write_files(exports: ExportSet, into: Path, stem: str = "projections") -> list[Path]:
    """Every table as CSV plus one workbook, into `into`. Returns what was written."""
    into.mkdir(parents=True, exist_ok=True)
    written = []
    for sheet in exports.sheets:
        path = into / f"{stem}_{sheet.name}.csv"
        path.write_bytes(csv_bytes(sheet.frame))
        written.append(path)
    book = into / f"{stem}.xlsx"
    book.write_bytes(workbook_bytes(exports, season=0))
    written.append(book)
    return written
