"""
s2_pipeline.py
==============

Full Sentinel-2 download 

fetch L1C .SAFE scenes from CDSE
Selection: up to 4 scenes per month (1 per week-window),
lowest cloud cover wins each window.

PREREQUISITES
  pip install cdsetool pandas rasterio shapely numpy
  CDSE account: https://dataspace.copernicus.eu/
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path


# CONFIG

BBOX          = (54.20, 24.40, 54.60, 24.70)   # (W, S, E, N)
TILE          = "40RBN"                          # MGRS tile for Saadiyat
PRODUCT_TYPE  = "S2MSI1C"                        # L1C top-of-atmosphere
CLOUD_MAX     = 10                               # % cloud cover upper limit
ARCHIVE_START = "2018-01-01"                     # sensible start (ICESat-2: 2018-10)

# Sentinel-2 nominal band centre wavelengths (nm)
# Full visible + NIR + SWIR for seagrass / shallow water.
RRS_BANDS = [443, 492, 560, 665, 704, 740, 783, 833, 1614, 2202]

# ACOLITE target resolution (m). 10 = resample all bands to 10 m native grid.
S2_TARGET_RES = 10

# ---------------------------------------------------------------------------
_DATE_RE = re.compile(r"_MSIL1C_(\d{8})T(\d{6})_")
_TILE_RE = re.compile(r"_T(\d{2}[A-Z]{3})_")
_MET = {12: "Winter", 1: "Winter",  2: "Winter",
         3: "Spring",  4: "Spring",  5: "Spring",
         6: "Summer",  7: "Summer",  8: "Summer",
         9: "Autumn", 10: "Autumn", 11: "Autumn"}


def gulf_season(m: int) -> str:
    return "Cool (Nov-Apr)" if m in (11, 12, 1, 2, 3, 4) else "Warm (May-Oct)"


def parse_name(name: str):
    dt = None
    md = _DATE_RE.search(name)
    if md:
        dt = datetime.strptime(md.group(1) + md.group(2), "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc)
    mt = _TILE_RE.search(name)
    tile = mt.group(1) if mt else None
    return dt, tile


def get_feature_name(f: dict) -> str:
    return f.get("Name") or f.get("name") or f.get("title") or str(f.get("Id", ""))


def get_cloud_cover(f: dict):
    cc = f.get("cloudCover")
    if cc is None:
        cc = f.get("CloudCover")
    if cc is not None:
        return cc
    attrs = f.get("Attributes") or []
    if isinstance(attrs, dict):
        attrs = attrs.get("results", [])
    for attr in attrs:
        if isinstance(attr, dict) and attr.get("Name") == "cloudCover":
            return attr.get("Value")
    props = f.get("properties") or {}
    return props.get("cloudCover") if isinstance(props, dict) else None


def wkt_from_bbox(bbox) -> str:
    w, s, e, n = bbox
    return f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"


def aoi_centroid_tile(bbox):
    try:
        import mgrs
        w, s, e, n = bbox
        return mgrs.MGRS().toMGRS((s + n) / 2, (w + e) / 2, MGRSPrecision=0)
    except ImportError:
        return None


def _spin(label: str, stop: threading.Event) -> None:
    frames = [".  ", ".. ", "..."]
    i = 0
    while not stop.is_set():
        print(f"\r{label}{frames[i % len(frames)]}", end="", flush=True)
        i += 1
        stop.wait(0.4)



# Step 1 -- Scene selection: up to N scenes per month (1 per week-window)


def week_of_month(day: int) -> int:
    #Map calendar day -> week-window 1..4 (days 1-7, 8-14, 15-21, 22+).
    return min((day - 1) // 7, 3) + 1


def select_weekly(rows: list[dict], weeks_per_month: int = 4) -> list[dict]:
    """
    Keep up to 4 scenes the lowest-cloud scene in each week-window (weeks 1-4 within the month).
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    df["_dt"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df["_cc"] = pd.to_numeric(df["cloud_cover"], errors="coerce").fillna(999)
    df["_yr"] = df["_dt"].dt.year
    df["_mo"] = df["_dt"].dt.month
    df["_wk"] = df["_dt"].dt.day.apply(week_of_month)

    df = df[df["_wk"] <= weeks_per_month]
    selected = (
        df.sort_values(["_yr", "_mo", "_wk", "_cc"])
        .groupby(["_yr", "_mo", "_wk"], group_keys=False)
        .head(1)
        .sort_values("datetime_utc")
    )
    return selected.drop(columns=["_dt", "_cc", "_yr", "_mo", "_wk"]).to_dict("records")



# Step 1 -- Download


def download_scenes(args, dl_dir: Path) -> list[dict]:
    """
    Query CDSE for scenes, apply week-window selection, write manifest.csv,
    download .SAFE zips to dl_dir/.
    """
    import pandas as pd

    try:
        from cdse_client import iter_features, _make_session
    except ImportError:
        sys.exit("cdse_client.py not found. Place it in the same folder as this script.")

    session = _make_session()
    tile = args.tile or aoi_centroid_tile(BBOX) or TILE
    print(f"[tile]    target MGRS tile = T{tile}")
    wkt = wkt_from_bbox(BBOX)

    start_year = int(args.start[:4])
    end_year   = int(args.end[:4])

    all_features: list[dict] = []
    for year in range(start_year, end_year + 1):
        yr_start = max(args.start, f"{year}-01-01")
        yr_end   = min(args.end,   f"{year}-12-31")
        if yr_start > yr_end:
            continue

        label     = f"[search]  year {year} "
        stop_spin = threading.Event()
        spinner   = threading.Thread(target=_spin, args=(label, stop_spin), daemon=True)
        spinner.start()

        fq = iter_features("Sentinel2", {
            "productType":        PRODUCT_TYPE,
            "contentDateStartGt": yr_start,
            "contentDateStartLt": yr_end,
            "cloudCoverLt":       args.cloud,
            "geometry":           wkt,
        }, session=session)

        yr_count = 0
        for feat in fq:
            if yr_count == 0:
                stop_spin.set(); spinner.join()
                print(f"\r{label}")
            all_features.append(feat)
            yr_count += 1
            name     = get_feature_name(feat)
            dt, _    = parse_name(name)
            cc       = get_cloud_cover(feat)
            date_str = dt.strftime("%Y-%m-%d") if dt else "?"
            cc_str   = f"{cc:.1f}%" if cc is not None else "?%"
            print(f"    {date_str}  {name[:60]}  cloud:{cc_str}")

        if not stop_spin.is_set():
            stop_spin.set(); spinner.join()
        print(f"  -> {yr_count} scene(s)")

    print(f"[search]  {len(all_features)} total scenes over AOI")

    # Filter to target tile, build rows
    rows: list[dict] = []
    feat_map: dict[str, dict] = {}
    for f in all_features:
        name = get_feature_name(f)
        dt, ftile = parse_name(name)
        if ftile != tile:
            continue
        cc = get_cloud_cover(f)
        row = {
            "name":         name,
            "tile":         ftile,
            "cloud_cover":  cc,
            "datetime_utc": dt.strftime("%Y-%m-%d %H:%M:%S") if dt else None,
            "date":         dt.strftime("%Y-%m-%d") if dt else None,
            "year":         dt.year if dt else None,
            "month":        dt.month if dt else None,
            "met_season":   _MET.get(dt.month, "") if dt else "",
            "gulf_season":  gulf_season(dt.month) if dt else "",
        }
        rows.append(row)
        feat_map[name] = f

    if not rows:
        print(f"[filter]  no scenes on tile T{tile}")
        return []

    print(f"[filter]  {len(rows)} scene(s) on T{tile} before weekly thinning")

    if not args.no_weekly:
        rows = select_weekly(rows, weeks_per_month=args.weeks_per_month)
        print(f"[weekly]  {len(rows)} scene(s) selected "
              f"(max {args.weeks_per_month}/month, lowest cloud per week-window)")

    if not rows:
        print("[weekly]  no scenes remain after selection")
        return []

    manifest_path = Path(args.out) / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"[manifest] wrote {manifest_path}")

    df = pd.DataFrame(rows)
    print("\n[scenes per Gulf season]")
    print(df["gulf_season"].value_counts().to_string())
    print("\n[scenes per year]")
    print(df.groupby("year").size().to_string())
    print()

    if args.search_only:
        print("[done]    --search-only; nothing downloaded.")
        return []

    # Download
    try:
        from cdsetool.download import download_feature
        from cdsetool.credentials import Credentials
        from cdsetool.monitor import StatusMonitor
    except ImportError as ex:
        sys.exit(f"cdsetool not installed ({ex}).  Run: pip install cdsetool")

    user = os.environ.get("CDSE_USERNAME")
    pw   = os.environ.get("CDSE_PASSWORD")
    if not user or not pw:
        import getpass
        user = user or input("CDSE username: ").strip()
        pw   = pw   or getpass.getpass("CDSE password: ")
    creds = Credentials(user, pw)

    dl_dir.mkdir(parents=True, exist_ok=True)
    opts = {"credentials": creds, "monitor": StatusMonitor(),
            "concurrency": args.concurrency}

    kept = [feat_map[r["name"]] for r in rows if r["name"] in feat_map]
    ok = 0
    print(f"[download] {len(kept)} scene(s) -> {dl_dir}/")
    for i, f in enumerate(kept, 1):
        name = get_feature_name(f)
        try:
            res = download_feature(f, str(dl_dir), opts)
            if res:
                ok += 1
                print(f"  [{i}/{len(kept)}] {res}")
            else:
                print(f"  [{i}/{len(kept)}] FAILED {name}")
        except Exception as ex:
            print(f"  [{i}/{len(kept)}] ERROR {name}: {ex}")

    print(f"[download] {ok}/{len(kept)} downloaded")
    return kept

