#!/usr/bin/env python3
"""
sst.py - OSTIA daily SST for the AOI, 2019-2023, matched to every scene.

Run on your PC (uses your saved Copernicus Marine login):
    mamba activate C:\\Users\\USER\\.conda\\envs\\acolite
    python E:\\pipeline_out\\code\\06_ancillary\\sst.py

Steps
  1. Download OSTIA reprocessed (REP) for the AOI box; if it stops before 2023-12-31,
     fill the rest from the near-real-time (NRT) product.
  2. Pick ONE all-water OSTIA pixel nearest the AOI centre (0.05 deg grid, ~5 km) and also
     compute the mean of all water pixels in the box.
  3. Write ancillary/sst_daily.csv and add sst_* columns to both AOI2 scene tables.
"""
import csv, sys, datetime as dt
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANC = ROOT / "ancillary"; ANC.mkdir(exist_ok=True)
BOX = dict(minimum_longitude=53.95, maximum_longitude=54.55, minimum_latitude=24.25, maximum_latitude=24.70)
CENTRE = (54.38, 24.36)                      # lon, lat: central lagoon inside the two-orbit footprint
FOOT = (54.22, 54.50, 24.29, 24.50)          # box-mean limited to the analysed (two-orbit) area
START, END = "2019-01-01T00:00:00", "2023-12-31T23:59:59"
REP = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
NRT = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"

def fetch(dataset, start, end, name):
    out = ANC / name
    if out.exists():
        print(f"[skip] {out.name} already downloaded"); return out
    import copernicusmarine as cm
    print(f"[download] {dataset}  {start[:10]} -> {end[:10]}", flush=True)
    cm.subset(dataset_id=dataset, variables=["analysed_sst"], start_datetime=start, end_datetime=end,
              output_directory=str(ANC), output_filename=name, **BOX)
    return out

def read(nc_path):
    import netCDF4 as nc
    d = nc.Dataset(nc_path)
    t = nc.num2date(d.variables["time"][:], d.variables["time"].units, only_use_cftime_datetimes=False)
    days = [dt.date(x.year, x.month, x.day) for x in t]
    sst = np.ma.filled(d.variables["analysed_sst"][:].astype("f8"), np.nan)
    if np.nanmedian(sst) > 200: sst = sst - 273.15          # K -> degC
    return days, d.variables["latitude"][:], d.variables["longitude"][:], sst

def main():
    f1 = fetch(REP, START, END, "ostia_rep.nc")
    days, lat, lon, sst = read(f1)
    print(f"[REP] {days[0]} -> {days[-1]}  ({len(days)} days), grid {len(lat)}x{len(lon)}")
    if days[-1] < dt.date(2023, 12, 31):
        s2 = (days[-1] + dt.timedelta(days=1)).isoformat() + "T00:00:00"
        f2 = fetch(NRT, s2, END, "ostia_nrt_fill.nc")
        d2, lat2, lon2, sst2 = read(f2)
        if sst2.shape[1:] != sst.shape[1:]:
            sys.exit("[error] REP and NRT grids differ - tell Claude")
        days += d2; sst = np.concatenate([sst, sst2]); print(f"[NRT] filled {d2[0]} -> {d2[-1]}")
    water = np.isfinite(sst).all(axis=0)
    LON, LAT = np.meshgrid(lon, lat)
    dist = np.hypot((LON - CENTRE[0]) * np.cos(np.radians(CENTRE[1])), LAT - CENTRE[1]); dist[~water] = np.inf
    iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
    print(f"[pixel] nearest all-water OSTIA pixel: lon {lon[ix]:.3f}, lat {lat[iy]:.3f} "
          f"({dist[iy, ix]*111:.1f} km from AOI centre) | water pixels in box: {water.sum()}")
    infoot = water & (LON >= FOOT[0]) & (LON <= FOOT[1]) & (LAT >= FOOT[2]) & (LAT <= FOOT[3])
    print(f"[box] {infoot.sum()} all-water OSTIA pixels inside the analysed area")
    px = sst[:, iy, ix]; box = np.nanmean(sst[:, infoot], axis=1)
    with open(ANC / "sst_daily.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "sst_pixel_c", "sst_boxmean_c"])
        for d, a, b in zip(days, px, box): w.writerow([d.isoformat(), round(float(a), 3), round(float(b), 3)])
    print(f"[out] {ANC/'sst_daily.csv'}  | pixel SST range {np.nanmin(px):.1f}-{np.nanmax(px):.1f} C")

    lut = {d: (a, b) for d, a, b in zip(days, px, box)}
    mon = {}
    for d, a in zip(days, px): mon.setdefault((d.year, d.month), []).append(a)
    mon = {k: float(np.nanmean(v)) for k, v in mon.items()}
    for tag in ("aoi2_noglint", "aoi2_glint"):
        p = ROOT / f"scene_table_{tag}.csv"
        rows = list(csv.DictReader(open(p))); cols = list(rows[0].keys())
        for c in ("sst_pixel_c", "sst_boxmean_c", "sst_month_mean_c"):
            if c not in cols: cols.append(c)
        miss = 0
        for r in rows:
            d = dt.date.fromisoformat(r["date"][:10]); v = lut.get(d)
            if v is None: miss += 1; v = (np.nan, np.nan)
            r["sst_pixel_c"], r["sst_boxmean_c"] = round(float(v[0]), 3), round(float(v[1]), 3)
            r["sst_month_mean_c"] = round(mon.get((d.year, d.month), np.nan), 3)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print(f"[merge] {p.name}: SST added to {len(rows)-miss}/{len(rows)} scenes")
    print("[done] next: tell Claude - RQ2/RQ3 can now use SST")

if __name__ == "__main__":
    main()
