"""Screenshots of the running app (Playwright, Edge): python research/screens.py"""
from pathlib import Path
from playwright.sync_api import sync_playwright
OUT = Path(__file__).resolve().parent / "shots"; OUT.mkdir(exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge")
    pg = b.new_page(viewport={"width": 1500, "height": 1300})
    for name in ("slate", "fantasy", "edges", "player"):
        pg.goto(f"http://localhost:8690/{name}", wait_until="networkidle")
        pg.wait_for_timeout(9000)
        pg.screenshot(path=str(OUT / f"{name}.png"))
        print("saved", name)
    b.close()
