"""
download_atl24_saadiyat.py


Download ICESat-2 ATL24 (coastal & nearshore bathymetry) granules over the
Abu Dhabi coast, for the full mission, and build a
season-tagged manifest.

WHAT THIS SCRIPT DOES
    1. Logs in to NASA Earthdata (earthaccess).
    2. Searches every ATL24 granule that intersects the Abu Dhabi bounding box
       from mission start (Oct 2018) to today.
    3. Downloads them to ./atl24_data/.
    4. Writes atl24_manifest.csv, tagging every granule with date, year,
       month, meteorological season, and an Arabian-Gulf cool/warm season
       split - the columns you bin on for the time-series-with-seasonality.

PREREQUISITES
    pip install earthaccess pandas
    A free Earthdata Login account:  https://urs.earthdata.nasa.gov/
    (the script will prompt, or read EARTHDATA_USERNAME / EARTHDATA_PASSWORD,
     or a ~/.netrc entry for urs.earthdata.nasa.gov)

USAGE
    python download_atl24_saadiyat.py                 # download everything
    python download_atl24_saadiyat.py --search-only   # list granules, no download
    python download_atl24_saadiyat.py --start 2022-01-01 --end 2026-06-28

Edit the CONFIG block below to change the box, dates, or output folder.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timezone


# CONFIG  


# Area of interest: 
BBOX = (54.20, 24.40, 54.60, 24.70)   # (lon_min, lat_min, lon_max, lat_max)

# Full ICESat-2 mission data 2018-10-13; "end"

MISSION_START = "2018-10-01"
DEFAULT_END = datetime.now(timezone.utc).strftime("%Y-%m-%d")

PRODUCT = "ATL24"     # short_name
VERSION = "001"       # ATL24 is currently version 001
OUTPUT_DIR = "atl24_data"
MANIFEST_CSV = "atl24_manifest.csv"


# Season tagging
# Northern-hemisphere meteorological seasons.
_MET_SEASON = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

# Arabian Gulf practical split. The Gulf has a cool season (~Nov-Apr) and a
# hot season (~May-Oct). Water clarity, SST and seagrass growth track this,
# so it is usually the more meaningful axis for a Gulf seagrass time series.
def gulf_season(month: int) -> str:
    return "Cool (Nov-Apr)" if month in (11, 12, 1, 2, 3, 4) else "Warm (May-Oct)"


# ATL24 granule filename pattern, e.g. ATL24_20190105234815_01300205_006_02_001_01.h5

_DATE_RE = re.compile(r"ATL24_(\d{14})_")


def granule_datetime(name: str) -> datetime | None:
    """Parse the YYYYMMDDHHMMSS acquisition stamp out of an ATL24 filename."""
    m = _DATE_RE.search(name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def season_columns(dt: datetime) -> dict:
    return {
        "datetime_utc": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "date": dt.strftime("%Y-%m-%d"),
        "year": dt.year,
        "month": dt.month,
        "doy": dt.timetuple().tm_yday,
        "met_season": _MET_SEASON[dt.month],
        "gulf_season": gulf_season(dt.month),
    }



# Main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download ICESat-2 ATL24 over Abu Dhabi.")
    p.add_argument("--start", default=MISSION_START, help="ISO start date (YYYY-MM-DD)")
    p.add_argument("--end", default=DEFAULT_END, help="ISO end date (YYYY-MM-DD)")
    p.add_argument("--out", default=OUTPUT_DIR, help="download directory")
    p.add_argument("--search-only", action="store_true",
                   help="search and write the manifest, but do not download files")
    p.add_argument("--threads", type=int, default=4, help="parallel download threads")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import earthaccess
    except ImportError:
        sys.exit("earthaccess not installed.  Run:  pip install earthaccess pandas")

    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas not installed.  Run:  pip install pandas")

    print(f"[auth] logging in to NASA Earthdata ...")
    # strategy='interactive' prompts for user/pass if no .netrc / env vars.
    auth = earthaccess.login(strategy="interactive", persist=True)
    if not getattr(auth, "authenticated", True):
        sys.exit("[auth] Earthdata login failed. Check your credentials.")

    print(f"[search] {PRODUCT} v{VERSION}")
    print(f"[search] bbox (W,S,E,N) = {BBOX}")
    print(f"[search] temporal       = {args.start} .. {args.end}")

    results = earthaccess.search_data(
        short_name=PRODUCT,
        version=VERSION,
        bounding_box=BBOX,
        temporal=(args.start, args.end),
    )
    print(f"[search] found {len(results)} granule(s)")

    if not results:
        print("[search] nothing to do. Try a wider box / date range or check ATL24 version.")
        return 0

    # Build the season-tagged manifest from granule metadata.
    rows = []
    for g in results:
        # earthaccess granule -> data file name
        try:
            links = g.data_links()
            fname = os.path.basename(links[0]) if links else ""
        except Exception:
            fname = ""
        dt = granule_datetime(fname)
        row = {"granule": fname, "size_mb": round(float(g.size() or 0), 1)}
        if dt is not None:
            row.update(season_columns(dt))
        rows.append(row)

    manifest = pd.DataFrame(rows).sort_values("datetime_utc", na_position="last")
    manifest.to_csv(MANIFEST_CSV, index=False)
    print(f"[manifest] wrote {MANIFEST_CSV}  ({len(manifest)} rows)")

    # Quick console summary of seasonal coverage.
    if "gulf_season" in manifest:
        print("\n[coverage] granules per Gulf season:")
        print(manifest["gulf_season"].value_counts().to_string())
        print("\n[coverage] granules per year:")
        print(manifest["year"].value_counts().sort_index().to_string())

    if args.search_only:
        print("\n[done] --search-only set; skipping download.")
        return 0

    os.makedirs(args.out, exist_ok=True)
    print(f"\n[download] -> {args.out}/  ({len(results)} granules)")
    files = earthaccess.download(results, args.out, threads=args.threads)
    print(f"[download] saved {len(files)} file(s).")
    print(f"[next] RQ4 reads these granules directly:  python ../05_analysis/rq4_depth.py photons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
