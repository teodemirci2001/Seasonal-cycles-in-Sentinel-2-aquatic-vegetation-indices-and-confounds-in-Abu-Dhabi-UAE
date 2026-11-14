"""
build_image_log.py - one row per Sentinel-2 scene with acquisition date/time,
viewing/solar geometry and ACOLITE status. Reads .SAFE metadata only.

Usage:  python build_image_log.py
Writes: image_log.csv next to this script, prints a revisit summary.

Times:
  sensing_time_utc  = tile SENSING_TIME from GRANULE/*/MTD_TL.xml. This is used for
                      tide prediction (the time in the .SAFE name is the datatake
                      start, ~10 min earlier, not when this tile was imaged).
  sensing_time_gst  = same in Gulf Standard Time (UTC+4).
"""
import csv, glob, os, statistics, sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   
SAFE_DIR, ACO_DIR = ROOT / "downloaded", ROOT / "acolite"
OUT = ROOT / "image_log.csv"
GST = timedelta(hours=4)

def local(tag): return tag.rsplit("}", 1)[-1]
def find_text(root, name):
    for el in root.iter():
        if local(el.tag) == name: return el.text
    return None
def parse_t(s): return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")

def tile_angles(root):
    sun_z = sun_a = None; vz, va = [], []
    for el in root.iter():
        t = local(el.tag)
        if t == "Mean_Sun_Angle":
            for c in el:
                if local(c.tag) == "ZENITH_ANGLE": sun_z = float(c.text)
                if local(c.tag) == "AZIMUTH_ANGLE": sun_a = float(c.text)
        elif t == "Mean_Viewing_Incidence_Angle":
            for c in el:
                if local(c.tag) == "ZENITH_ANGLE": vz.append(float(c.text))
                if local(c.tag) == "AZIMUTH_ANGLE": va.append(float(c.text))
    m = lambda x: round(statistics.mean(x), 3) if x else None
    return sun_z, sun_a, m(vz), m(va)

def aco_status(name):
    d = ACO_DIR / name.replace(".SAFE", "")
    if not d.is_dir(): return "not_run"
    if glob.glob(str(d / "*_L2W.nc")): return "ok"
    if glob.glob(str(d / "*_L1R.nc")): return "failed_after_L1R"
    return "failed"

def gulf_season(m): return "Cool (Nov-Apr)" if m in (11,12,1,2,3,4) else "Warm (May-Oct)"

rows = []
safes = sorted(SAFE_DIR.glob("*.SAFE"))
print(f"[scan] {len(safes)} .SAFE scenes in {SAFE_DIR}")
for i, safe in enumerate(safes, 1):
    try:
        prod = ET.parse(safe / "MTD_MSIL1C.xml").getroot()
        tl = ET.parse(next(safe.glob("GRANULE/*/MTD_TL.xml"))).getroot()
    except Exception as ex:
        print(f"  [warn] {safe.name}: {ex}"); continue
    st = parse_t(find_text(tl, "SENSING_TIME"))
    sz, sa, vz, va = tile_angles(tl)
    rows.append({
        "scene": safe.name,
        "spacecraft": find_text(prod, "SPACECRAFT_NAME"),
        "relative_orbit": "R%03d" % int(find_text(prod, "SENSING_ORBIT_NUMBER")),
        "date": st.date().isoformat(),
        "sensing_time_utc": st.strftime("%Y-%m-%d %H:%M:%S"),
        "sensing_time_gst": (st + GST).strftime("%Y-%m-%d %H:%M:%S"),
        "datatake_start_utc": parse_t(find_text(prod, "PRODUCT_START_TIME")).strftime("%Y-%m-%d %H:%M:%S"),
        "year": st.year, "month": st.month, "doy": st.timetuple().tm_yday,
        "gulf_season": gulf_season(st.month),
        "cloud_pct_tile": float(find_text(tl, "CLOUDY_PIXEL_PERCENTAGE")),
        "sun_zenith_deg": sz, "sun_azimuth_deg": sa,
        "view_zenith_mean_deg": vz, "view_azimuth_mean_deg": va,
        "acolite_status": aco_status(safe.name),
    })
    if i % 50 == 0: print(f"  ... {i}/{len(safes)}")

rows.sort(key=lambda r: r["sensing_time_utc"])
prev = None
for r in rows:
    t = datetime.strptime(r["sensing_time_utc"], "%Y-%m-%d %H:%M:%S")
    r["days_since_prev"] = round((t - prev).total_seconds() / 86400, 2) if prev else ""
    prev = t

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# summary 
gaps = [r["days_since_prev"] for r in rows if r["days_since_prev"] != ""]
print(f"\n[out] {OUT}  ({len(rows)} rows)")
print(f"[span] {rows[0]['date']} -> {rows[-1]['date']}")
print(f"[gap days] median {statistics.median(gaps):.1f} | mean {statistics.mean(gaps):.1f} | "
      f"min {min(gaps):.1f} | max {max(gaps):.1f}")
for lo, hi in [(0,4),(4,8),(8,15),(15,31),(31,999)]:
    n = sum(lo <= g < hi for g in gaps); print(f"   {lo:>3}-{hi if hi<999 else '+':<3} d : {n}")
from collections import Counter
print("[per year]", dict(sorted(Counter(r["year"] for r in rows).items())))
pm = Counter((r["year"], r["month"]) for r in rows)
print(f"[months with >=1 scene] {len(pm)}  | scenes/month median {statistics.median(pm.values())}")
print("[acolite]", dict(Counter(r["acolite_status"] for r in rows)))
print("[orbit]", dict(Counter((r["spacecraft"][-2:], r["relative_orbit"]) for r in rows)))
tm = [r["sensing_time_gst"][11:16] for r in rows]
print(f"[local time GST] {min(tm)} - {max(tm)}")
