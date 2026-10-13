"""Upload the week's outputs to the repo's `data` release, which the Streamlit Cloud app reads (cloud.py).

    python publish.py

Auth: GITHUB_TOKEN in the environment, else git's credential manager for github.com. Data files go up
first and manifest.json last, so the app never sees a manifest that points ahead of its files. Only the
app's inputs are published -- projections, distribution tables and aggregated results; never raw odds.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import requests

import cloud
import config as C

API = f"https://api.github.com/repos/{cloud.DEFAULT_REPO}"


def token() -> str:
    t = os.environ.get("GITHUB_TOKEN")
    if t:
        return t
    out = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                         capture_output=True, text=True, cwd=C.ROOT).stdout
    return next(line.split("=", 1)[1] for line in out.splitlines() if line.startswith("password="))


def main() -> None:
    h = {"Authorization": f"Bearer {token()}", "Accept": "application/vnd.github+json"}
    r = requests.get(f"{API}/releases/tags/{cloud.TAG}", headers=h, timeout=30)
    if r.status_code == 404:
        r = requests.post(f"{API}/releases", headers=h, timeout=30, json={
            "tag_name": cloud.TAG, "target_commitish": "main", "name": "Projection data",
            "body": "Written by publish.py after each refresh; read by the app. Do not edit."})
    r.raise_for_status()
    rel = r.json()
    have = {a["name"]: a["id"] for a in rel.get("assets", [])}
    meta = json.loads((C.LIVE / "meta.json").read_text())
    manifest = C.LIVE / cloud.MANIFEST
    manifest.write_text(json.dumps({"built_at": meta["built_at"], "season": meta["season"], "week": meta["week"]}))
    for name, path in [*cloud.ASSETS.items(), (cloud.MANIFEST, manifest)]:
        if not path.exists():
            continue
        if name in have:
            requests.delete(f"{API}/releases/assets/{have[name]}", headers=h, timeout=30).raise_for_status()
        with open(path, "rb") as fh:
            up = requests.post(f"https://uploads.github.com/repos/{cloud.DEFAULT_REPO}/releases/{rel['id']}/assets",
                               params={"name": name}, headers={**h, "Content-Type": "application/octet-stream"},
                               data=fh, timeout=600)
        up.raise_for_status()
        print(f"published {name} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"publish failed: {e!r}", file=sys.stderr)
        sys.exit(1)
