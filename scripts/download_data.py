"""Download the Metrica Sports open tracking data into data/metrica (~230 MB)."""

import subprocess
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "data" / "metrica"

if DEST.exists():
    print(f"Already there: {DEST}")
else:
    DEST.parent.mkdir(exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/metrica-sports/sample-data", str(DEST)], check=True)
    print(f"Done: {DEST}")
