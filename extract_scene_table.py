"""
extract_scene_table.py - masks + one-row-per-scene table (AOI2 = full ACOLITE window, 54.01-54.49E 24.29-24.49N).

INPUT
  ACOLITE L2W Rrs_<wl> (sr^-1; the file attribute wrongly says "1"), converted to
  rho_w = pi * Rrs (dimensionless) before any mask or index. Matters for WAVI (+0.5 term);
  NDAVI unaffected; DII only gains a constant (1-k)ln(pi) offset.

MASKS (multi-scene median of N_MASK_SCENES evenly spaced scenes)
  Water           median NDWI(560, 833) > NDWI_MIN; no shore buffer (SHORE_BUFFER_M = 0)
  Reference water darkest REF_PCT % of median rho_w(560) water, > REF_MIN_DIST_M from land,
                  anywhere in the window (AOI is now the full rectangle); excluded from the classes
  Seabed visible  median rho_w(560) > BOTTOM_RATIO x reference-water median (depth proxy)
  Classes         visible-seabed rho_w(560) terciles: dark (vegetation candidate) / mid / bright (sand)
  DII k           Lyzenga variance/covariance of ln(blue), ln(green) over bright (sand) pixels
  Exclusions      optional aoi/exclude_dredged.geojson (digitised in QGIS): dredged channels etc. -> code 8,
                  removed before the terciles are computed
  Footprint       only pixels seen by BOTH relative orbits (R020 + R063) are analysed, so every scene samples
                  the same pixels (R020's swath edge crosses the window); west of it -> code 9
  Codes           1 land, 2 shore buffer, 3 seabed not visible, 4 dark, 5 mid, 6 bright, 7 reference, 8 excluded,
                  9 outside two-orbit footprint

PER-SCENE OUTPUT (median over each mask class: veg=4, sand=6, park_visible=4-6, deep=7)
  NDAVI, WAVI, DII, turbidity (TUR_Nechad2016_665, FNU), rho_w at 492/560/665/833/1610.
  Summary adds glint angle (geometry), haze / low-coverage / high-glint flags, baseline.
  No SWIR subtraction (rejected: pushed NDAVI below -1).

OUTPUTS
  extraction_<tag>/masks.npz, masks_v3.tif, median_green_rhow_v2.tif, masks_info.json, rows/*.json
  scene_table_<tag>.csv   (ancillary columns added later by sst.py / tides.py are preserved)
"""
import argparse, glob, json, math, os, sys, time
from pathlib import Path
import numpy as np

# CONFIG
ROOT = Path(__file__).resolve().parents[2]         

PIX            = 10       # m
NDWI_MIN       = 0.0      # water if median NDWI > this
SHORE_BUFFER_M = 0        # dropped on request (bright land edges handled by NDWI)
REF_PCT        = 5.0      # reference water = darkest % of median green
REF_MIN_DIST_M = 2000     # reference water at least this far from land
BOTTOM_RATIO   = 1.5      # seabed visible if green > ratio x reference
N_MASK_SCENES  = 16
WAVI_L         = 0.5
BANDS          = (492, 560, 665, 833, 1610)
GLINT_HIGH_DEG = 25       # glint angle below this = high-glint scene
HAZE_Z         = 3        # robust z of reference-water blue
MAX_CLOUD_FRAC = 0.01     # L1C cloud mask: drop scene if > 1 % of analysed seabed or reference water is cloud
MIN_PIXELS     = 20       # min valid pixels for a class median
COVER_MISS     = 2        # pixel must be valid in >= N_MASK_SCENES - COVER_MISS sampled scenes (two-orbit footprint)
MASK_TIME_BUDGET_S = float(os.environ.get("MASK_TIME_BUDGET_S", 7200))  # caching time limit (s); rerun continues
CRS            = "EPSG:32640"
MASK_TIF       = "masks_v5.tif"           # v5 = dredged exclusions (mask_v5.gpkg) 

EXCLUDE = ROOT / "aoi" / "exclude_dredged.geojson"   # optional: polygons digitised in QGIS (dredged channels etc.)

CODES = {1: "land", 2: "shore buffer (park)", 3: "park, seabed not visible",
         4: "DARK seabed (vegetation candidate)", 5: "park, mid brightness",
         6: "BRIGHT seabed (sand control)", 7: "deep-water patch", 8: "excluded (dredged / artificial)",
         9: "outside two-orbit footprint (R063 only, not analysed)"}
AREAS = {"veg": (4,), "sand": (6,), "park_visible": (4, 5, 6), "deep": (7,)}   # output column prefixes
ROW_VARS = ("ndavi", "wavi", "dii", "tur", "rho492", "rho560", "rho665", "rho833", "rho1610")


def parse_args():
    ap = argparse.ArgumentParser(description="Masks and per-scene extraction from ACOLITE L2W")
    ap.add_argument("stage", nargs="?", default="summary", choices=("masks", "extract", "summary"))
    ap.add_argument("slice", nargs="*", type=int, help="optional A B scene slice for extract")
    ap.add_argument("--source", default="acolite_aoi2_noglint")
    ap.add_argument("--aoi", default="aoi/aoi_rect.geojson")
    ap.add_argument("--tag", default="aoi2_noglint")
    ap.add_argument("--masks-from", default="aoi2_noglint", help="tag whose masks to reuse")
    return ap.parse_args()


class Paths:
    def __init__(self, a):
        self.aco = ROOT / a.source
        self.out = ROOT / f"extraction_{a.tag}"
        self.rows = self.out / "rows"
        self.cache = self.out / "mask_cache"
        self.masks = ROOT / f"extraction_{a.masks_from}" / "masks.npz"
        self.aoi = ROOT / a.aoi
        self.table = ROOT / f"scene_table_{a.tag}.csv"
        self.rows.mkdir(parents=True, exist_ok=True)


# I/O HELPERS
def l2w_files(P):
    return sorted(glob.glob(str(P.aco / "*/*_L2W.nc")))


def band_var(ds, wl):
    """Nearest Rrs_<wl> variable (S2A/S2B wavelengths differ slightly)."""
    w = {int(v.split("_")[1]): v for v in ds.variables if v.startswith("Rrs_")}
    return w[min(w, key=lambda x: abs(x - wl))]


def read(ds, var, rs=slice(None), cs=slice(None)):
    return np.ma.filled(ds.variables[var][rs, cs].astype("f4"), np.nan)


def rho_w(ds, wl, rs=slice(None), cs=slice(None)):
    """Water reflectance, dimensionless: pi * Rrs."""
    return np.pi * read(ds, band_var(ds, wl), rs, cs)


def glint_angle(sza, saa, vza, vaa):
    d = math.radians
    c = math.cos(d(vza)) * math.cos(d(sza)) - math.sin(d(vza)) * math.sin(d(sza)) * math.cos(d(vaa - saa))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


# STAGE 1: MASKS
def cache_mask_scenes(P, picks):
    """Cache float16 G, S, B, NIR for each sampled scene. Returns False if out of time."""
    import netCDF4 as nc
    P.cache.mkdir(exist_ok=True); t0 = time.time()
    for i, f in enumerate(picks, 1):
        cf, nf = P.cache / f"{i:02d}.npz", P.cache / f"{i:02d}_nir.npy"
        if cf.exists() and nf.exists():
            continue
        if time.time() - t0 > MASK_TIME_BUDGET_S:
            print("[masks] partial - rerun to continue"); return False
        d = nc.Dataset(f)
        np.save(nf, rho_w(d, 833).astype("f2"))
        if not cf.exists():
            np.savez(cf, G=rho_w(d, 560).astype("f2"), S=rho_w(d, 1610).astype("f2"), B=rho_w(d, 492).astype("f2"))
        print(f"  cached {i}/{len(picks)} ({time.time()-t0:.0f}s)", flush=True)
    return True


def strip_median(P, n, load):
    """Low-memory median: float16 stack, median computed in 200-row strips."""
    st = np.stack([load(i) for i in range(1, n + 1)]); out = np.empty(st.shape[1:], "f4")
    for r0 in range(0, st.shape[1], 200):
        out[r0:r0 + 200] = np.nanmedian(st[:, r0:r0 + 200].astype("f4"), 0)
    return out


def aoi_mask(P, lon, lat):
    from matplotlib.path import Path as MPath
    geom = json.load(open(P.aoi))["features"][0]["geometry"]
    poly = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
    return MPath(poly).contains_points(np.c_[lon.ravel(), lat.ravel()]).reshape(lat.shape)


def exclusion_mask(shape, transform):
    """Rasterise EXCLUDE polygons (any CRS QGIS writes to GeoJSON; lon/lat assumed if no crs member)."""
    if not EXCLUDE.exists():
        return np.zeros(shape, bool)
    from rasterio.features import rasterize
    from rasterio.warp import transform_geom
    gj = json.load(open(EXCLUDE))
    src = gj.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    src = "EPSG:" + src.split("EPSG::")[-1] if "EPSG::" in src else ("EPSG:4326" if "CRS84" in src else src)
    geoms = [transform_geom(src, CRS, f["geometry"]) for f in gj["features"] if f.get("geometry")]
    m = rasterize([(g, 1) for g in geoms], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    print(f"[masks] exclusion polygons: {len(geoms)} from {EXCLUDE.name} ({src}) -> {int(m.sum()):,} px")
    return m


def lyzenga_k(B, G):
    lb, lg = np.log(B), np.log(G); ok = np.isfinite(lb) & np.isfinite(lg)
    vb, vg = np.var(lb[ok]), np.var(lg[ok]); cv = np.cov(lb[ok], lg[ok])[0, 1]
    a = (vb - vg) / (2 * cv)
    return float(a + np.sqrt(a * a + 1))


def classify(G, NIR, B, in_aoi, excl, full):
    from scipy import ndimage
    with np.errstate(all="ignore"):
        ndwi = (G - NIR) / (G + NIR)
    land = ~(ndwi > NDWI_MIN) | ~np.isfinite(ndwi)
    water = ~land
    dist = ndimage.distance_transform_edt(water) * PIX

    cand = water & (dist > REF_MIN_DIST_M) & np.isfinite(G) & full
    ref = cand & (G <= np.nanpercentile(G[cand], REF_PCT))
    g_ref = np.nanmedian(G[ref])

    aoi_w = in_aoi & water & (dist > SHORE_BUFFER_M) & ~ref & ~excl & full
    visible = aoi_w & (G > BOTTOM_RATIO * g_ref)
    t1, t2 = np.nanpercentile(G[visible], [33.3, 66.7])
    dark, bright = visible & (G <= t1), visible & (G >= t2)

    code = np.zeros(G.shape, "u1")
    code[land] = 1
    code[in_aoi & water & (dist <= SHORE_BUFFER_M)] = 2
    code[aoi_w & ~visible] = 3
    code[visible] = 5; code[dark] = 4; code[bright] = 6; code[ref] = 7
    code[excl & in_aoi & water & ~ref & full] = 8
    code[water & ~full & (code == 0)] = 9

    k = lyzenga_k(B[bright], G[bright])
    n = lambda m: int(m.sum())
    info = {"park_water_px": n(aoi_w), "bottom_visible_px": n(visible), "dark_px": n(dark), "bright_px": n(bright),
            "deep_px": n(ref), "excluded_px": n(excl & in_aoi & water & full), "outside_footprint_water_px": n(water & ~full), "deep_median_rhow560": float(g_ref), "tercile_thresholds": [float(t1), float(t2)],
            "k_dii": k, "land_frac_in_park": float((in_aoi & land).sum() / in_aoi.sum()),
            "water_rule": f"median NDWI(560,833) > {NDWI_MIN}",
            "deep_patch_rule": f"darkest {REF_PCT}% median rho560, >{REF_MIN_DIST_M} m from land, anywhere in window",
            "codes": {str(c): v for c, v in CODES.items()}}
    return code, k, info


def write_geotiffs(P, code, G, x, y):
    try:
        import rasterio
        from rasterio.transform import from_origin
        tr = from_origin(float(x[0]) - PIX / 2, float(y[0]) + PIX / 2, PIX, PIX)
        for name, arr, dt, nd in ((MASK_TIF, code, "uint8", 0), ("median_green_rhow_v2.tif", G.astype("f4"), "float32", None)):
            if name != MASK_TIF and (P.out / name).exists():
                continue   # AOI-independent; don't overwrite (may be open in QGIS)
            with rasterio.open(P.out / name, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                               count=1, dtype=dt, crs=CRS, transform=tr, nodata=nd) as dst:
                dst.write(arr, 1)
        print(f"[out] {P.out/MASK_TIF} (+ median_green_rhow_v2.tif)")
    except Exception as ex:
        print(f"[warn] GeoTIFF not written ({ex}); masks.npz saved")


def stage_masks(P):
    import netCDF4 as nc
    fs = l2w_files(P)
    picks = [fs[i] for i in np.linspace(0, len(fs) - 1, N_MASK_SCENES).astype(int)]
    ds0 = nc.Dataset(fs[0])
    lat, lon = np.asarray(ds0.variables["lat"][:]), np.asarray(ds0.variables["lon"][:])
    x, y = np.asarray(ds0.variables["x"][:]), np.asarray(ds0.variables["y"][:])
    print(f"[masks] {lat.shape[0]}x{lat.shape[1]} px @{PIX} m | sampling {len(picks)} scenes")
    if not cache_mask_scenes(P, picks):
        return
    n = len(picks)
    valid = np.zeros(lat.shape, "u1")
    for i in range(1, n + 1):
        valid += np.isfinite(np.load(P.cache / f"{i:02d}.npz")["G"])
    full = valid >= n - COVER_MISS                              # seen by both orbits
    print(f"[masks] two-orbit footprint: {full.mean()*100:.0f}% of window")
    G, S, B = (strip_median(P, n, lambda i, k=k: np.load(P.cache / f"{i:02d}.npz")[k]) for k in "GSB")
    NIR = strip_median(P, n, lambda i: np.load(P.cache / f"{i:02d}_nir.npy"))
    print("[masks] medians done", flush=True)

    from rasterio.transform import from_origin
    tr = from_origin(float(x[0]) - PIX / 2, float(y[0]) + PIX / 2, PIX, PIX)
    code, k, info = classify(G, NIR, B, aoi_mask(P, lon, lat), exclusion_mask(G.shape, tr), full)
    np.savez_compressed(P.out / "masks.npz", code=code, rs=[0, code.shape[0]], cs=[0, code.shape[1]], k_dii=k)
    write_geotiffs(P, code, G, x, y)
    json.dump(info, open(P.out / "masks_info.json", "w"), indent=1)
    for kk, v in info.items():
        if kk != "codes":
            print(f"  {kk:24} {v}")


# STAGE 2: EXTRACT
def scene_indices(rho, k):
    b, g, n = rho[492], rho[560], rho[833]
    with np.errstate(all="ignore"):
        ndavi = (n - b) / (n + b)
        wavi = (1 + WAVI_L) * (n - b) / (n + b + WAVI_L)
        dii = np.log(np.where(b > 0, b, np.nan)) - k * np.log(np.where(g > 0, g, np.nan))
    return ndavi, wavi, dii


def reset_if_stale(rows_dir, masks_file):
    """Rows built with an older mask are moved aside (not deleted) so extraction starts fresh."""
    stamp = rows_dir / "_mask_stamp.txt"; now = str(int(masks_file.stat().st_mtime))
    if rows_dir.exists() and any(rows_dir.glob("*.json")) and (not stamp.exists() or stamp.read_text() != now):
        old = rows_dir.with_name(f"{rows_dir.name}_stale_{time.strftime('%Y%m%d_%H%M%S')}")
        rows_dir.rename(old); print(f"[extract] masks changed - old rows moved to {old.name}")
    rows_dir.mkdir(parents=True, exist_ok=True); stamp.write_text(now)


def stage_extract(P, a=None, b=None):
    import netCDF4 as nc
    reset_if_stale(P.rows, P.masks)
    m = np.load(P.masks); code = m["code"]; k = float(m["k_dii"])
    rs, cs = slice(*m["rs"]), slice(*m["cs"])
    areas = {name: np.isin(code, c) for name, c in AREAS.items()}
    fs = l2w_files(P)[a:b] if a is not None else l2w_files(P)
    t0 = time.time(); done = 0
    for f in fs:
        scene = Path(f).parent.name; rowf = P.rows / f"{scene}.json"
        if rowf.exists():
            continue
        d = nc.Dataset(f)
        rho = {wl: rho_w(d, wl, rs, cs) for wl in BANDS}
        ndavi, wavi, dii = scene_indices(rho, k)
        arrays = dict(zip(ROW_VARS, (ndavi, wavi, dii, read(d, "TUR_Nechad2016_665", rs, cs),
                                     rho[492], rho[560], rho[665], rho[833], rho[1610])))
        row = {"scene": scene}
        for an, mask in areas.items():
            tot = int(mask.sum()); valid = mask & np.isfinite(ndavi)
            row[f"{an}_valid_frac"] = round(valid.sum() / tot, 4) if tot else 0
            for vn, arr in arrays.items():
                vals = arr[valid]; vals = vals[np.isfinite(vals)]
                row[f"{an}_{vn}"] = float(np.median(vals)) if vals.size >= MIN_PIXELS else None
        json.dump(row, open(rowf, "w")); done += 1
        print(f"  {scene[:26]}  veg_valid={row['veg_valid_frac']:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    left = len(l2w_files(P)) - len(list(P.rows.glob("S2*.json")))
    print(f"[extract] {done} new | {left} scenes still to do")


# STAGE 3: SUMMARY
def build_table(P):
    import pandas as pd
    df = pd.DataFrame([json.load(open(p)) for p in sorted(P.rows.glob("S2*.json"))])
    log = pd.read_csv(ROOT / "image_log.csv")
    log["scene"] = log["scene"].str.replace(".SAFE", "", regex=False)
    df = log.merge(df, on="scene", how="left")
    geo = ["sun_zenith_deg", "sun_azimuth_deg", "view_zenith_mean_deg", "view_azimuth_mean_deg"]
    df["glint_angle_deg"] = [glint_angle(*r) for r in df[geo].values]
    df["baseline"] = df["scene"].str.split("_").str[3]
    df["log_tur_deep"] = np.log(df["deep_tur"])
    med = df["deep_rho492"].median(); mad = (df["deep_rho492"] - med).abs().median()
    df["haze_z_deep_blue"] = (df["deep_rho492"] - med) / (1.4826 * mad)
    df["flag_haze"] = (df["haze_z_deep_blue"] > HAZE_Z) & (df["glint_angle_deg"] >= GLINT_HIGH_DEG)
    df["flag_low_coverage"] = df["veg_valid_frac"] < 0.5
    df["flag_high_glint"] = df["glint_angle_deg"] < GLINT_HIGH_DEG
    cf = ROOT / "ancillary" / "cloud_fraction_l1c.csv"           # from 06_ancillary/cloud_fraction.py
    if cf.exists():
        df = df.merge(pd.read_csv(cf), on="scene", how="left")
        df["flag_cloud"] = (df["cloud_frac_analysed"] > MAX_CLOUD_FRAC) | (df["cloud_frac_ref"] > MAX_CLOUD_FRAC)
    else:
        print("  WARNING: ancillary/cloud_fraction_l1c.csv missing - run 06_ancillary/cloud_fraction.py; no cloud flag")
        df["flag_cloud"] = False
    df["flag_exclude"] = df["flag_haze"] | df["flag_cloud"]       # scenes left out of every analysis
    return df


def keep_ancillary(P, df):
    """Carry over columns added by sst.py / tides.py so a re-run doesn't drop them."""
    import pandas as pd
    if not P.table.exists():
        return df
    old = pd.read_csv(P.table)
    extra = [c for c in old.columns if c not in df.columns and c not in ("cloud_frac_analysed", "cloud_frac_ref", "flag_cloud", "flag_exclude")]
    if extra:
        df = df.merge(old[["scene"] + extra], on="scene", how="left")
        print(f"  kept ancillary columns: {', '.join(extra)}")
    return df


def print_checks(df):
    print(f"  flags: haze {int(df.flag_haze.sum())} | cloud {int(df.flag_cloud.sum())} | excluded {int(df.flag_exclude.sum())} | low coverage {int(df.flag_low_coverage.sum())} | high glint {int(df.flag_high_glint.sum())}")
    for c in ("veg_ndavi", "veg_wavi", "veg_dii", "deep_tur", "deep_rho1610"):
        s = df[c].dropna()
        print(f"  {c:14} n={len(s):3d}  median={s.median():.4f}  p5={s.quantile(.05):.4f}  p95={s.quantile(.95):.4f}")
    j = df.dropna(subset=["deep_rho560"]).sort_values("sensing_time_utc")
    pre, post = j[j.year == 2021].tail(12)["deep_rho560"], j[j.year == 2022].head(12)["deep_rho560"]
    print(f"  Jan-2022 baseline check (deep rho560, last 12 of 2021 vs first 12 of 2022): {pre.median():.4f} -> {post.median():.4f}")


def stage_summary(P):
    df = keep_ancillary(P, build_table(P))
    df.to_csv(P.table, index=False)
    print(f"[summary] {P.table.name}: {len(df)} rows, {df['veg_ndavi'].notna().sum()} with vegetation values")
    print_checks(df)


# MAIN
if __name__ == "__main__":
    args = parse_args(); P = Paths(args)
    if args.stage == "masks":
        stage_masks(P)
    elif args.stage == "extract":
        stage_extract(P, *args.slice[:2]) if len(args.slice) >= 2 else stage_extract(P)
    else:
        stage_summary(P)
