"""
Fig 1 - study area map.
  (a) true-colour composite of the analysed area, Bul Syayeef Ramsar boundary, ICESat-2 ATL24
      depth pixels, area outside the two-orbit footprint shaded, scale bar, north arrow, inset
  (b) mask classes (masks_v5.tif) over the composite, with class areas in the legend

Inputs (all relative to the project folder):
  extraction_aoi2_noglint/composite_truecolour_aoi.tif, masks_v5.tif, atl24_depth_pixels.npz
  aoi/ramsar_bul_syayeef/features_param_idrvis.shp      (Ramsar site boundary, EPSG:4326)
  ancillary/naturalearth_lowres/naturalearth_lowres.shp  (Natural Earth 1:110m countries, for the inset)
Output: figures/fig1_study_area.png (300 dpi) and .pdf

    python make_fig1_study_area.py
"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform, transform_geom
import pyogrio.raw
import shapely
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "extraction_aoi2_noglint"
OUT = ROOT / "figures"; OUT.mkdir(exist_ok=True)
WIDTH = 6.3
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "savefig.dpi": 300})

CLASSES = {  # legend: (label, colour)
    4: ("SAV candidate (dark tercile)", "#1b9e77"),
    5: ("Mid tercile (not analysed)", "#bdbdbd"),
    6: ("Sand control (bright tercile)", "#e6ab02"),
    3: ("Optically deep", "#2c5f8a"),
    7: ("Reference deep water", "#e7298a"),
    8: ("Dredged (excluded)", "#6a3d9a"),
}

# read rasters on the composite extent
with rasterio.open(E / "composite_truecolour_aoi.tif") as c:
    rgb = np.moveaxis(c.read(), 0, -1)
    rgb = np.dstack([rgb, np.where(rgb.max(axis=2) == 0, 0, 255).astype("uint8")])   # no-data transparent
    b = c.bounds
    crs = c.crs
extent = (b.left, b.right, b.bottom, b.top)

with rasterio.open(E / "masks_v5.tif") as m:
    win = from_bounds(*b, m.transform).round_offsets().round_lengths()
    code = m.read(1, window=win)
    mtr = m.transform
r0, c0 = int(win.row_off), int(win.col_off)

# vectors and ATL24 pixels
def read_polys(path, to_crs=None):
    _, _, geoms, _ = pyogrio.raw.read(path)
    out = [shapely.from_wkb(g) for g in geoms]
    if to_crs:
        out = [shapely.geometry.shape(transform_geom("EPSG:4326", to_crs, shapely.geometry.mapping(g))) for g in out]
    return out

ramsar = read_polys(ROOT / "aoi" / "ramsar_bul_syayeef" / "features_param_idrvis.shp", crs)
world = read_polys(ROOT / "ancillary" / "naturalearth_lowres" / "naturalearth_lowres.shp")

px = np.load(E / "atl24_depth_pixels.npz")
xs, ys = rasterio.transform.xy(mtr, px["row"], px["col"])
xs, ys = np.asarray(xs), np.asarray(ys)
inside = (xs >= b.left) & (xs <= b.right) & (ys >= b.bottom) & (ys <= b.top) & np.isin(px["code"], [3, 4, 5, 6])


def outline(ax, geoms, **kw):
    for g in geoms:
        for p in getattr(g, "geoms", [g]):
            x, y = p.exterior.xy
            ax.plot(x, y, **kw)


def lonlat_ticks(ax):
    """Label axes in degrees: transform the frame edges from UTM to lon/lat."""
    xt = np.linspace(b.left, b.right, 200)
    lon, _ = transform(crs, "EPSG:4326", xt, np.full_like(xt, b.bottom))
    yt = np.linspace(b.bottom, b.top, 200)
    _, lat = transform(crs, "EPSG:4326", np.full_like(yt, b.left), yt)
    want_lon = [l for l in (54.25, 54.30, 54.35, 54.40, 54.45) if min(lon) < l < max(lon)]
    want_lat = [l for l in (24.30, 24.35, 24.40, 24.45) if min(lat) < l < max(lat)]
    ax.set_xticks([np.interp(l, lon, xt) for l in want_lon]); ax.set_xticklabels([f"{l:.2f}°E" for l in want_lon])
    ax.set_yticks([np.interp(l, lat, yt) for l in want_lat]); ax.set_yticklabels([f"{l:.2f}°N" for l in want_lat])


def scale_and_north(ax):
    x0, y0, L = b.left + 0.05 * (b.right - b.left), b.bottom + 0.05 * (b.top - b.bottom), 5000
    ax.add_patch(Rectangle((x0, y0), L / 2, 250, fc="black", ec="black"))
    ax.add_patch(Rectangle((x0 + L / 2, y0), L / 2, 250, fc="white", ec="black"))
    ax.text(x0 + L / 2, y0 + 500, "5 km", ha="center", va="bottom", fontsize=7,
            bbox=dict(fc="white", ec="none", alpha=.7, pad=1))
    xn, yn = b.right - 0.06 * (b.right - b.left), b.top - 0.16 * (b.top - b.bottom)
    ax.annotate("N", xy=(xn, yn + 2600), xytext=(xn, yn), ha="center", fontsize=8, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", fc="black", lw=1.2))


fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 9.6), constrained_layout=True)

# ---------------------------------------------------------------- (a) context
ax = axes[0]
ax.imshow(rgb, extent=extent)
out_fp = np.ma.masked_where(code != 9, code)
ax.imshow(out_fp, extent=extent, cmap=ListedColormap(["white"]), alpha=0.6, interpolation="nearest")
ax.scatter(xs[inside], ys[inside], s=0.15, c="#ffd400", lw=0, rasterized=True)
outline(ax, ramsar, color="#d7191c", lw=1.3)
scale_and_north(ax)
lonlat_ticks(ax)
ax.legend(handles=[Line2D([], [], color="#d7191c", lw=1.3, label="Bul Syayeef Ramsar site"),
                   Line2D([], [], color="#ffd400", marker="o", lw=0, ms=4, label="ICESat-2 ATL24 depth pixels"),
                   Patch(fc="white", ec="0.5", alpha=.6, label="Outside two-orbit footprint")],
          loc="upper left", framealpha=.85)
ax.set_title("a", loc="left", fontweight="bold")
ins = ax.inset_axes([0.70, 0.66, 0.29, 0.31])
for g in world:
    for p in getattr(g, "geoms", [g]):
        x, y = p.exterior.xy; ins.fill(x, y, fc="#d9d9d9", ec="#7f7f7f", lw=0.4)
lon_c, lat_c = transform(crs, "EPSG:4326", [b.left, b.right], [b.bottom, b.top])
ins.add_patch(Rectangle((lon_c[0], lat_c[0]), lon_c[1] - lon_c[0], lat_c[1] - lat_c[0], fc="none", ec="#d7191c", lw=1.2))
ins.set_xlim(50, 57); ins.set_ylim(22, 27.5); ins.set_xticks([]); ins.set_yticks([])
ins.set_facecolor("#cfe6f3"); ins.text(55.4, 23.3, "UAE", fontsize=6.5, ha="center")

# (b) mask classes
ax = axes[1]
ax.imshow(rgb, extent=extent, alpha=0.45)
codes = list(CLASSES)
lut = np.full(10, -1); lut[codes] = np.arange(len(codes))
idx = np.ma.masked_less(lut[code], 0)
ax.imshow(idx, extent=extent, cmap=ListedColormap([CLASSES[k][1] for k in codes]), vmin=0, vmax=len(codes) - 1,
          interpolation="nearest")
outline(ax, ramsar, color="#d7191c", lw=1.0, ls="--")
lonlat_ticks(ax)
handles = [Patch(fc=CLASSES[k][1], label=f"{CLASSES[k][0]}: {(code == k).sum() * 1e-4:.1f} km²") for k in codes]
handles.append(Line2D([], [], color="#d7191c", ls="--", lw=1, label="Bul Syayeef Ramsar site"))
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False)
ax.set_title("b", loc="left", fontweight="bold")

for a in axes:
    a.set_xlim(extent[0], extent[1]); a.set_ylim(extent[2], extent[3])
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig1_study_area.{ext}", bbox_inches="tight")
print(f"[fig] fig1_study_area -> {OUT}")
