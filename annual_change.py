"""
annual_change.py - has the seabed / coastline changed 2019-2023 (dredging, reclamation)?
Writes extraction_aoi2_noglint/annual/ (per-year .npy, change GeoTIFF, preview PNG).
"""
import glob, sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ACO = ROOT / "acolite_aoi2_noglint"; OUT = ROOT / "extraction_aoi2_noglint" / "annual"; OUT.mkdir(exist_ok=True)
YEARS = range(2019, 2024); MAX_PER_YEAR = 12; MAX_CLOUD = 1.0;  # tile cloud % (2023-01-01 at 4.8 % left cloud in the median)
DROP = 0.5    # "deepened" = rho560 fell to < 50 % of first-year value


def band(ds, wl):
    w = {int(v.split("_")[1]): v for v in ds.variables if v.startswith("Rrs_")}
    return w[min(w, key=lambda x: abs(x - wl))]


def year_median(y):
    import netCDF4 as nc, pandas as pd
    f = OUT / f"G_{y}.npy"
    if f.exists():
        return
    st = pd.read_csv(ROOT / "scene_table_aoi2_noglint.csv")
    ok = st[(st.year == y) & st.scene.str.contains("R063") & ~st.flag_haze.astype(bool) & ~st.flag_high_glint.astype(bool) & (st.cloud_pct_tile < MAX_CLOUD)].scene
    files = [p for p in sorted(glob.glob(str(ACO / "*/*_L2W.nc"))) if Path(p).parent.name in set(ok)][:MAX_PER_YEAR]
    Gs, Ns = [], []
    for p in files:
        d = nc.Dataset(p)
        Gs.append((np.pi * np.ma.filled(d.variables[band(d, 560)][:].astype("f4"), np.nan)).astype("f2"))
        Ns.append((np.pi * np.ma.filled(d.variables[band(d, 833)][:].astype("f4"), np.nan)).astype("f2"))
    G = np.nanmedian(np.stack(Gs).astype("f4"), 0); N = np.nanmedian(np.stack(Ns).astype("f4"), 0)
    np.save(f, G.astype("f4")); np.save(OUT / f"NDWI_{y}.npy", ((G - N) / (G + N)).astype("f4"))
    print(f"  {y}: {len(files)} scenes", flush=True)


def change():
    import rasterio, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from scipy import ndimage
    G0, G1 = np.load(OUT / "G_2019.npy"), np.load(OUT / "G_2023.npy")
    W0, W1 = np.load(OUT / "NDWI_2019.npy") > 0, np.load(OUT / "NDWI_2023.npy") > 0
    deep = W0 & W1 & (G1 < DROP * G0) & (G0 > 0.06)          # was visibly shallow, now much darker
    shallow = W0 & W1 & (G0 < DROP * G1) & (G1 > 0.06)       # infilled / shoaled
    recl = W0 & ~W1; lost = ~W0 & W1
    clean = lambda m: ndimage.binary_opening(m, iterations=2)  # drop speckle (< ~50 m)
    deep, shallow, recl, lost = map(clean, (deep, shallow, recl, lost))
    code = np.zeros(G0.shape, "u1"); code[deep] = 1; code[shallow] = 2; code[recl] = 3; code[lost] = 4
    with rasterio.open(ROOT / "extraction_aoi2_noglint" / "masks_v5.tif") as r:
        prof = r.profile
    prof.update(nodata=0)
    with rasterio.open(OUT / "change_2019_2023.tif", "w", **prof) as d:
        d.write(code, 1)
    km = lambda m: m.sum() * 1e-4
    print(f"[change 2019->2023] deepened/dredged {km(deep):.2f} km2 | shoaled/infilled {km(shallow):.2f} km2 | "
          f"reclaimed (water->land) {km(recl):.2f} km2 | land->water {km(lost):.2f} km2")
    yrs = [np.load(OUT / f"G_{y}.npy") for y in YEARS]
    fig, ax = plt.subplots(1, 3, figsize=(21, 4.8), constrained_layout=True)
    for i, (y, g) in enumerate(((2019, yrs[0]), (2023, yrs[-1]))):
        ax[i].imshow(np.clip(g[::2, ::2] / 0.3, 0, 1), cmap="gray"); ax[i].set_title(f"median green reflectance {y}")
    cm = matplotlib.colors.ListedColormap(["#ffffff", "#d62728", "#1f77b4", "#ff7f0e", "#2ca02c"])
    ax[2].imshow(np.clip(yrs[-1][::2, ::2] / 0.3, 0, 1), cmap="gray", alpha=.5)
    ax[2].imshow(np.ma.masked_where(code[::2, ::2] == 0, code[::2, ::2]), cmap=cm, vmin=0, vmax=4, interpolation="nearest")
    ax[2].set_title("red = deepened/dredged, blue = infilled, orange = reclaimed, green = land->water")
    for x in ax: x.set_xticks([]); x.set_yticks([])
    fig.savefig(OUT / "change_2019_2023.png", dpi=70)


if __name__ == "__main__":
    for y in YEARS:
        year_median(y)
    change()
