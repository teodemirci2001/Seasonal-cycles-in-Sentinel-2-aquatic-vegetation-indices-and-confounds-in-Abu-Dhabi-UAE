"""
make_aoi_composite.py - true-colour composite of the ANALYSED area only (two-orbit footprint).
Scenes: clear (no haze flag), glint angle >= 30 deg, 3 per year 2019-2023 (lowest cloud, spread across
seasons), both orbits. Median of rho_w; R=665, G=560, B=492 nm; stretch on analysed water pixels.
Outside the footprint = nodata. Cropped to the footprint's bounding box.
Output: extraction_aoi2_noglint/composite_truecolour_aoi.tif + scenes list.
"""
import glob, numpy as np, pandas as pd, netCDF4 as nc, rasterio
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; SRC = ROOT / "acolite_aoi2_noglint"; OUT = ROOT / "extraction_aoi2_noglint"
PER_YEAR = 3
t = pd.read_csv(ROOT / "scene_table_aoi2_noglint.csv", parse_dates=["date"])
t = t[(t.glint_angle_deg >= 30) & ~t.flag_haze.astype(bool) & t.veg_ndavi.notna()].copy()
t["season"] = t.date.dt.month % 12 // 3                      # 0 DJF, 1 MAM, 2 JJA, 3 SON
pick = []
for y, g in t.groupby(t.date.dt.year):                       # one per season (lowest cloud), first PER_YEAR seasons with data
    for _, s in g.sort_values("cloud_pct_tile").groupby("season"):
        pick.append(s.iloc[0])
    pick = pick[:PER_YEAR * (y - 2018)]
pick = pd.DataFrame(pick).sort_values("date")
def band(ds, wl):
    w = {int(v.split("_")[1]): v for v in ds.variables if v.startswith("Rrs_")}; return w[min(w, key=lambda x: abs(x - wl))]
code = np.load(OUT / "masks.npz")["code"]
foot = np.load(OUT / "mask_cache" / "valid_count.npy") >= 14
rr, cc = np.where(foot); r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
code, foot = code[r0:r1, c0:c1], foot[r0:r1, c0:c1]
water = foot & np.isin(code, (3, 4, 5, 6, 7, 8))
files = [glob.glob(str(SRC / s / "*_L2W.nc"))[0] for s in pick.scene]
rgb = np.empty((3,) + foot.shape, "f4")
for i, wl in enumerate((665, 560, 492)):                   # one band at a time, cropped, strip median (low memory)
    st = np.stack([(np.pi * np.ma.filled(nc.Dataset(f).variables[band(nc.Dataset(f), wl)][r0:r1, c0:c1].astype("f4"), np.nan)).astype("f2") for f in files])
    for a0 in range(0, st.shape[1], 200):
        rgb[i, a0:a0 + 200] = np.nanmedian(st[:, a0:a0 + 200].astype("f4"), 0)
    del st; print(f"  band {wl} done", flush=True)
lo, hi = np.nanpercentile(rgb[:, water], 1, axis=1), np.nanpercentile(rgb[:, water], 99, axis=1)
u8 = np.clip((rgb - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1)
u8 = (np.nan_to_num(u8) ** (1 / 1.4) * 254 + 1).astype("u1"); u8[:, ~foot | ~np.isfinite(rgb).all(0)] = 0
with rasterio.open(OUT / "masks_v5.tif") as ref:
    prof = ref.profile.copy(); tr = ref.transform * rasterio.Affine.translation(c0, r0)
prof.update(count=3, dtype="uint8", nodata=0, photometric="RGB", height=r1 - r0, width=c1 - c0, transform=tr)
out = OUT / "composite_truecolour_aoi.tif"
with rasterio.open(out, "w", **prof) as dst:
    dst.write(u8)
pick[["date", "sensing_time_utc", "relative_orbit", "cloud_pct_tile", "glint_angle_deg"]].to_csv(OUT / "composite_truecolour_aoi_scenes.csv", index=False)
print(f"[out] {out}  ({c1-c0} x {r1-r0} px)\n[scenes] {len(pick)}: " + ", ".join(pick.date.dt.strftime('%Y-%m-%d')))
