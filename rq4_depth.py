"""
rq4_depth.py - RQ4: how does the seasonal cycle of each index change with water depth?

Depth comes ONLY from ICESat-2 ATL24 bathymetry photons (independent of Sentinel-2,
so no circularity with the blue/green bands used by DII):
  depth below mean sea level = -ortho_h   (ortho_h = seafloor height above EGM2008 geoid, m)
  -> tide-independent; photons kept if class_ph == 40 (bathymetry) and low_confidence_flag == 0.
Photons are gridded onto the 10 m mask grid (median depth per pixel). Only AOI pixels with
visible seabed are used, then grouped into depth bins.
"""
import glob, json, sys, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]                
ATL24_DIR = ROOT.parent / "atl24_data"                     
MASKDIR = ROOT / "extraction_aoi2_noglint"
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
PIX_NPZ = MASKDIR / "atl24_depth_pixels.npz"

BBOX = (54.00, 24.28, 54.50, 24.50)                         # W S E N 
BEAMS = ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r")
BATHY_CLASS = 40
DEPTH_BINS = [0, 1, 2, 3, 5, 10]                        # m below MSL 
VISIBLE_CODES = (4, 5, 6)
MIN_PIXELS = 20
WAVI_L = 0.5
VERSIONS = ("noglint", "glint")


def bin_labels():
    return [f"{a}-{b} m" for a, b in zip(DEPTH_BINS[:-1], DEPTH_BINS[1:])]


# STAGE 1: PHOTONS -> PIXELS
def stage_photons():
    import h5py, rasterio
    from rasterio.warp import transform
    m = np.load(MASKDIR / "masks.npz"); code = m["code"]
    with rasterio.open(MASKDIR / "masks_v5.tif") as r:
        inv = ~r.transform
    files = sorted(glob.glob(str(ATL24_DIR / "ATL24_*.h5")))
    print(f"[photons] {len(files)} ATL24 granules in {ATL24_DIR}")
    w, s, e, n = BBOX; lats, lons, deps = [], [], []
    for i, f in enumerate(files, 1):
        with h5py.File(f, "r") as h:
            for b in BEAMS:
                if b not in h:
                    continue
                g = h[b]; lat = g["lat_ph"][:]; lon = g["lon_ph"][:]
                k = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n) & (g["class_ph"][:] == BATHY_CLASS)
                if "low_confidence_flag" in g:
                    k &= g["low_confidence_flag"][:] == 0
                if k.any():
                    lats.append(lat[k]); lons.append(lon[k]); deps.append(-g["ortho_h"][:][k])
        if i % 20 == 0:
            print(f"  {i}/{len(files)} granules", flush=True)
    lat, lon, dep = (np.concatenate(a) for a in (lats, lons, deps))
    x, y = transform("EPSG:4326", "EPSG:32640", lon.tolist(), lat.tolist())
    col, row = inv * (np.asarray(x), np.asarray(y)); col, row = col.astype(int), row.astype(int)
    ok = (row >= 0) & (row < code.shape[0]) & (col >= 0) & (col < code.shape[1]) & np.isfinite(dep)
    row, col, dep = row[ok], col[ok], dep[ok]
    flat = row * code.shape[1] + col
    order = np.argsort(flat); flat, dep = flat[order], dep[order]
    uniq, start, cnt = np.unique(flat, return_index=True, return_counts=True)
    med = np.array([np.median(dep[a:a + c]) for a, c in zip(start, cnt)], "f4")
    pr, pc = uniq // code.shape[1], uniq % code.shape[1]
    vis = np.isin(code[pr, pc], VISIBLE_CODES)
    np.savez_compressed(PIX_NPZ, row=pr, col=pc, depth=med, n=cnt, code=code[pr, pc])
    print(f"[photons] {len(dep):,} bathymetry photons -> {len(uniq):,} pixels ({vis.sum():,} on visible seabed)")
    d = med[vis]
    print(f"  depth on visible seabed: p5 {np.percentile(d,5):.1f} | median {np.median(d):.1f} | p95 {np.percentile(d,95):.1f} m")
    h, _ = np.histogram(d, DEPTH_BINS)
    for lab, c in zip(bin_labels(), h):
        print(f"  {lab:8} {c:6,d} px")
    for c_, nm in ((4, "dark/veg"), (5, "mid"), (6, "bright/sand")):
        dd = med[code[pr, pc] == c_]
        if dd.size:
            print(f"  code {c_} {nm:11} n={dd.size:6,d}  median depth {np.median(dd):.1f} m")
    print(f"[out] {PIX_NPZ}")


# STAGE 2: PER-SCENE EXTRACTION
def band_var(ds, wl):
    w = {int(v.split("_")[1]): v for v in ds.variables if v.startswith("Rrs_")}
    return w[min(w, key=lambda x: abs(x - wl))]


def stage_extract():
    import netCDF4 as nc
    P = np.load(PIX_NPZ); m = np.load(MASKDIR / "masks.npz"); k = float(m["k_dii"])
    vis = np.isin(P["code"], VISIBLE_CODES)
    row, col, dep, cde = P["row"][vis], P["col"][vis], P["depth"][vis], P["code"][vis]
    b = np.digitize(dep, DEPTH_BINS) - 1                      # 0..len-2 inside range
    groups = {}
    for i, lab in enumerate(bin_labels()):
        groups[(lab, "all")] = (b == i)
        groups[(lab, "veg")] = (b == i) & (cde == 4)
    for ver in VERSIONS:
        rows_dir = ROOT / f"extraction_aoi2_{ver}" / "rows_depth"
        stamp = rows_dir / "_mask_stamp.txt"; now = str(int(PIX_NPZ.stat().st_mtime))
        if rows_dir.exists() and any(rows_dir.glob("S2*.json")) and (not stamp.exists() or stamp.read_text() != now):
            old = rows_dir.with_name(f"rows_depth_stale_{time.strftime('%Y%m%d_%H%M%S')}"); rows_dir.rename(old)
            print(f"[extract] depth pixels changed - old rows moved to {old.name}")
        rows_dir.mkdir(exist_ok=True); stamp.write_text(now)
        files = sorted(glob.glob(str(ROOT / f"acolite_aoi2_{ver}" / "*/*_L2W.nc")))
        t0 = time.time(); done = 0
        for f in files:
            scene = Path(f).parent.name; rf = rows_dir / f"{scene}.json"
            if rf.exists():
                continue
            d = nc.Dataset(f)
            rho = {wl: np.pi * np.ma.filled(d.variables[band_var(d, wl)][:].astype("f4"), np.nan)[row, col]
                   for wl in (492, 560, 833)}
            B, G, N = rho[492], rho[560], rho[833]
            with np.errstate(all="ignore"):
                idx = {"ndavi": (N - B) / (N + B), "wavi": (1 + WAVI_L) * (N - B) / (N + B + WAVI_L),
                       "dii": np.log(np.where(B > 0, B, np.nan)) - k * np.log(np.where(G > 0, G, np.nan))}
            out = {"scene": scene}
            for (lab, cls), msk in groups.items():
                for nm, arr in idx.items():
                    v = arr[msk]; v = v[np.isfinite(v)]
                    out[f"{cls}|{lab}|{nm}"] = float(np.median(v)) if v.size >= MIN_PIXELS else None
            json.dump(out, open(rf, "w")); done += 1
            if done % 10 == 0:
                print(f"  {ver}: {done} new scenes ({time.time()-t0:.0f}s)", flush=True)
        print(f"[extract] {ver}: {len(list(rows_dir.glob('S2*.json')))}/{len(files)} scenes done")


# STAGE 3: ANALYSIS
def harm(t, y):
    from scipy import stats
    X = np.column_stack([np.ones_like(t, float), np.sin(2 * np.pi * t / 365.25), np.cos(2 * np.pi * t / 365.25)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); f = X @ b
    rss1, rss0, n = np.sum((y - f) ** 2), np.sum((y - y.mean()) ** 2), len(y)
    F = ((rss0 - rss1) / 2) / (rss1 / (n - 3))
    pk = (np.degrees(np.arctan2(b[1], b[2])) % 360) / 360 * 365.25
    amp = np.hypot(b[1], b[2])
    return dict(n=n, R2=1 - rss1 / rss0, p=1 - stats.f.cdf(F, 2, n - 3), amp=amp, amp_over_sd=amp / np.std(y),
                peak_doy=pk, max_rate_per_month=amp * 2 * np.pi / 12)


def stage_analyse():
    import pandas as pd
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    P = np.load(PIX_NPZ); vis = np.isin(P["code"], VISIBLE_CODES)
    npx = {lab: int(((np.digitize(P["depth"][vis], DEPTH_BINS) - 1) == i).sum()) for i, lab in enumerate(bin_labels())}
    res = []
    for ver in VERSIONS:
        st = pd.read_csv(ROOT / f"scene_table_aoi2_{ver}.csv", parse_dates=["date"])
        keep = st.loc[~st.flag_exclude.astype(bool), ["scene", "date"]]
        rows = pd.DataFrame([json.load(open(p)) for p in (ROOT / f"extraction_aoi2_{ver}" / "rows_depth").glob("S2*.json")])
        df = keep.merge(rows, on="scene")
        for c in [c for c in df.columns if "|" in c]:
            cls, lab, ix = c.split("|"); s = df[["date", c]].dropna()
            if len(s) < 30:
                continue
            h = harm(s.date.dt.dayofyear.values, s[c].values)
            res.append(dict(version=ver, pixels=cls, depth_bin=lab, n_pixels_bin=npx[lab], index=ix.upper(), **h,
                            peak_month=(pd.Timestamp("2021-01-01") + pd.Timedelta(days=h["peak_doy"])).strftime("%b")))
    R = pd.DataFrame(res); R.to_csv(OUT / "rq4_depth_aoi2.csv", index=False)
    pd.set_option("display.width", 200)
    v = R[(R.version == "noglint") & (R.pixels == "all")]
    print("[RQ4] no-glint run, all visible-seabed pixels: annual harmonic by depth bin")
    print(v[["index", "depth_bin", "n_pixels_bin", "n", "R2", "amp", "amp_over_sd", "peak_month", "max_rate_per_month"]]
          .round(4).to_string(index=False))
    print("\n[RQ4] amplitude in deepest bin as % of shallowest (lower = more depth-affected)")
    for ver in VERSIONS:
        for ix in ("NDAVI", "WAVI", "DII"):
            a = R[(R.version == ver) & (R.pixels == "all") & (R["index"] == ix)].set_index("depth_bin")["amp"]
            if len(a) >= 2:
                print(f"  {ver:7} {ix:5}  {a.iloc[0]:.4f} -> {a.iloc[-1]:.4f}  ({100*a.iloc[-1]/a.iloc[0]:.0f}%)")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for i, ix in enumerate(("NDAVI", "WAVI", "DII")):
        for ver, ls in (("noglint", "-"), ("glint", "--")):
            for cls, col in (("all", "#2a78d6"), ("veg", "#1b9e77")):
                a = R[(R.version == ver) & (R.pixels == cls) & (R["index"] == ix)]
                ax[i].plot(range(len(a)), a.amp_over_sd, ls=ls, marker="o", color=col,
                           label=f"{'visible seabed' if cls=='all' else 'vegetation candidate'} ({ver})")
                ax[i].set_xticks(range(len(a))); ax[i].set_xticklabels(a.depth_bin)
        ax[i].set_title(ix, loc="left"); ax[i].set_xlabel("ICESat-2 depth below MSL")
        ax[i].grid(color="#e4e4e0", lw=.6); ax[i].spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("Seasonal amplitude / SD"); ax[2].legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "rq4_depth_aoi2.png", dpi=150)
    print(f"\n[out] {OUT/'rq4_depth_aoi2.csv'} | rq4_depth_aoi2.png")


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    stale = PIX_NPZ.exists() and PIX_NPZ.stat().st_mtime < (MASKDIR / "masks.npz").stat().st_mtime
    if st in ("photons", "all") and (st == "photons" or not PIX_NPZ.exists() or stale):
        stage_photons()
    if st in ("extract", "all"):
        stage_extract()
    if st in ("analyse", "all"):
        stage_analyse()
