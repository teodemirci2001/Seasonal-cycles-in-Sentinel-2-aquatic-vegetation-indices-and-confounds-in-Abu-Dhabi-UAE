"""
Dissertation figures and tables, built from the existing analysis outputs.
Nothing is recomputed except the harmonic curves drawn on Figs 4 and 6.

    python make_figures.py          -> figures/fig2..fig7 (.png 300 dpi), figures/table1, table2 (.csv)

Inputs:  scene_table_aoi2_noglint.csv, scene_table_aoi2_glint.csv,
         results/rq1_harmonic_aoi2.csv, rq2_rq3_aoi2.csv, rq3_veg_minus_sand_aoi2.csv, rq4_depth_aoi2.csv
Fig 1 (study-area map) is made in QGIS, not here.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "figures"; OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- style
WIDTH = 6.3                      # inches, A4 text width with 2.5 cm margins
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "savefig.dpi": 300})
VEG, SAND = "#2a78d6", "#eb6834"                         # SAV candidate, bright sand
ORBIT = {"R063": "#333333", "R020": "#a6a6a6"}
CONF = {"turbidity": "#E69F00", "tide": "#56B4E9", "glint": "#009E73", "orbit": "#CC79A7"}   # Okabe-Ito
IDX = ("NDAVI", "WAVI", "DII")
MONTHS = "JFMAMJJASOND"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {name}")


def letter(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=9, ha="right", va="bottom")


def load_scenes(version):
    d = pd.read_csv(ROOT / f"scene_table_aoi2_{version}.csv", parse_dates=["date"]).sort_values("date")
    d = d[~d.flag_exclude.astype(bool) & d.veg_ndavi.notna()]   # same scenes as RQ1 (haze + cloud removed)
    return d.replace([np.inf, -np.inf], np.nan)


def harmonic_fit(doy, y):
    """Annual harmonic y = a + b sin + c cos. Returns a function of day of year."""
    w = 2 * np.pi / 365.25
    X = np.column_stack([np.ones_like(doy, float), np.sin(w * doy), np.cos(w * doy)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return lambda t: b[0] + b[1] * np.sin(w * t) + b[2] * np.cos(w * t)


S = load_scenes("noglint")
print(f"[data] {len(S)} scenes (haze- and cloud-flagged removed)")

# Fig 2: confounds through time
# Methods 2.6: glint, turbidity, tide, sun zenith, SST
panels = [("deep_rho1610", "Glint\n(ref. water ρ$_w$1610)", False),
          ("deep_tur", "Turbidity\n(FNU, ref. water)", True),
          ("tide_m", "Tide at overpass\n(m)", False),
          ("sun_zenith_deg", "Sun zenith\n(°)", False),
          ("sst_pixel_c", "SST\n(°C)", False)]
fig, axes = plt.subplots(len(panels), 1, figsize=(WIDTH, 7), sharex=True)
for ax, (col, label, logy), let in zip(axes, panels, "abcde"):
    for orb, c in ORBIT.items():
        s = S[S.relative_orbit == orb]
        ax.scatter(s.date, s[col], s=6, color=c, lw=0, label=orb)
    monthly = S.set_index("date")[col].resample("MS").mean()
    ax.plot(monthly.index + pd.Timedelta(days=14), monthly.values, color="#2a78d6", lw=0.9, label="Monthly mean")
    if logy: ax.set_yscale("log")
    ax.set_ylabel(label)
    letter(ax, let)
axes[0].legend(ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.35))
axes[-1].set_xlabel("Date")
fig.align_ylabels(axes)
save(fig, "fig2_confounds_timeseries")

# Fig 3: glint angle vs measured glint
s = S[["glint_angle_deg", "deep_rho1610", "relative_orbit"]].dropna()
r = np.corrcoef(s.glint_angle_deg, s.deep_rho1610)[0, 1]
fig, ax = plt.subplots(figsize=(WIDTH * 0.55, 3))
for orb, c in ORBIT.items():
    o = s[s.relative_orbit == orb]
    ax.scatter(o.glint_angle_deg, o.deep_rho1610, s=10, color=c, lw=0, label=orb)
ax.text(0.97, 0.95, f"r = {r:.2f}\nn = {len(s)}", transform=ax.transAxes, ha="right", va="top")
ax.set_xlabel("Glint angle (°)")
ax.set_ylabel("Reference-water ρ$_w$(1610)")
ax.legend(frameon=False, loc="upper center")
save(fig, "fig3_glint_angle_vs_swir")
print(f"       r(glint angle, rho1610) = {r:.3f}")

# Fig 4: RQ1 monthly climatology + harmonic
fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4), constrained_layout=True)
t = np.arange(1, 366)
for ax, ix, let in zip(axes, IDX, "abc"):
    for area, c, name in (("veg", VEG, "SAV candidate"), ("sand", SAND, "Sand")):
        col = f"{area}_{ix.lower()}"
        s = S[["date", col]].dropna()
        clim = s.groupby(s.date.dt.month)[col].agg(["mean", "sem"])
        ax.errorbar(clim.index, clim["mean"], yerr=clim["sem"], fmt="o", ms=3, color=c, capsize=1.5, lw=0.8, label=name)
        f = harmonic_fit(s.date.dt.dayofyear.values, s[col].values)
        ax.plot(t / 365.25 * 12 + 0.5, f(t), color=c, lw=1, alpha=0.8)
    ax.set_title(ix, loc="left")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS); ax.set_xlim(0.5, 12.5)
    letter(ax, let)
axes[0].set_ylabel("Index value\n(monthly mean ± 1 SE)")
axes[0].legend(frameon=False)
save(fig, "fig4_rq1_climatology")

# Table 1: RQ1 harmonic metrics
rq1 = pd.read_csv(RES / "rq1_harmonic_aoi2.csv")
t1 = rq1[(rq1.version == "noglint") & rq1.area.isin(["veg", "sand"])].copy()
t1["area"] = t1.area.map({"veg": "SAV candidate", "sand": "Sand"})
t1 = t1[["index", "area", "n_scenes", "R2_scene", "amp", "amp_over_sd", "max_rate_per_month", "peak", "peak_by_year"]]
t1.columns = ["Index", "Class", "n", "R²", "Amplitude", "Amplitude/SD", "Max rate (per month)", "Peak", "Peak month by year"]
t1.round({"R²": 2, "Amplitude": 3, "Amplitude/SD": 2, "Max rate (per month)": 3}).to_csv(OUT / "table1_rq1_harmonic.csv", index=False)
print("[tab] table1_rq1_harmonic")

# Fig 5: RQ2 LMG shares
rq2 = pd.read_csv(RES / "rq2_rq3_aoi2.csv")
m = rq2[(rq2.version == "noglint") & (rq2.spec == "main")]
labels, bottoms = [], None
fig, ax = plt.subplots(figsize=(WIDTH * 0.7, 2.8))
order = [(ix, a) for ix in IDX for a in ("veg", "sand")]
x = np.arange(len(order))
bottom = np.zeros(len(order))
for name, c in CONF.items():
    vals = np.array([m[(m["index"] == ix) & (m.area == a)][f"LMG_{name}"].iloc[0] for ix, a in order])
    ax.bar(x, vals, bottom=bottom, color=c, width=0.7, label=name.capitalize())
    bottom += vals
for xi, tot in zip(x, bottom):
    ax.text(xi, tot + 0.01, f"{tot:.2f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{ix}\n{'SAV' if a == 'veg' else 'Sand'}" for ix, a in order])
ax.set_ylabel("Share of variance explained (R²)")
ax.set_ylim(0, 1)
ax.grid(axis="x", visible=False)
ax.legend(frameon=False, ncol=4, loc="upper right", bbox_to_anchor=(1, 1.12))
save(fig, "fig5_rq2_lmg_shares")

# Fig 6: RQ3 vegetation minus sand
z = lambda v: (v - v.mean()) / v.std()
fig, axes = plt.subplots(4, 1, figsize=(WIDTH, 6), sharex=True)
for ax, ix, let in zip(axes[:3], ("WAVI", "DII", "NDAVI"), "abc"):
    s = S[["date", f"veg_{ix.lower()}", f"sand_{ix.lower()}"]].dropna()
    diff = s[f"veg_{ix.lower()}"] - s[f"sand_{ix.lower()}"]
    ax.scatter(s.date, diff, s=6, color="#555555", lw=0)
    f = harmonic_fit(s.date.dt.dayofyear.values, diff.values)
    days = pd.date_range(S.date.min(), S.date.max(), freq="D")
    ax.plot(days, f(days.dayofyear.values), color=VEG, lw=1)
    ax.set_ylabel(f"{ix}\nSAV − sand")
    letter(ax, let)
ax = axes[3]
ax.plot(S.date, z(S.sst_pixel_c), color="#d62728", lw=0.8, label="SST")
ax.plot(S.date, z(S.sun_zenith_deg), color="#7f7f7f", lw=0.8, label="Sun zenith")
ax.set_ylabel("Standardised\n(z-score)")
ax.set_xlabel("Date")
ax.legend(frameon=False, ncol=2, loc="upper left")
letter(ax, "d")
fig.align_ylabels(axes)
save(fig, "fig6_rq3_veg_minus_sand")

# Fig 7: RQ4 by depth
rq4 = pd.read_csv(RES / "rq4_depth_aoi2.csv")
d4 = rq4[(rq4.version == "noglint") & (rq4.pixels == "all")]
bins = list(dict.fromkeys(d4.depth_bin))
npix = d4.drop_duplicates("depth_bin").set_index("depth_bin").n_pixels_bin
IXCOL = {"NDAVI": "#1b9e77", "WAVI": "#7570b3", "DII": "#d95f02"}
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), constrained_layout=True)
for ax, col, ylab, let in ((axes[0], "R2", "Harmonic R²", "a"), (axes[1], "amp_over_sd", "Amplitude / SD", "b")):
    for ix in IDX:
        v = d4[d4["index"] == ix].set_index("depth_bin").loc[bins, col]
        ax.plot(range(len(bins)), v.values, marker="o", ms=3.5, color=IXCOL[ix], label=ix)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f"{b}\n(n={npix[b]:,})" for b in bins], fontsize=6.5)
    ax.set_xlabel("ICESat-2 ATL24 depth bin")
    ax.set_ylabel(ylab)
    letter(ax, let)
axes[0].legend(frameon=False)
save(fig, "fig7_rq4_depth")

# glint-correction sensitivity table
rq3 = pd.read_csv(RES / "rq3_veg_minus_sand_aoi2.csv")
rows = []
for ix in IDX:
    row = {"Index": ix}
    for ver, tag in (("noglint", "no corr."), ("glint", "glint corr.")):
        a = rq1[(rq1.version == ver) & (rq1["index"] == ix) & (rq1.area == "veg")].iloc[0]
        b = rq2[(rq2.version == ver) & (rq2.spec == "main") & (rq2["index"] == ix) & (rq2.area == "veg")].iloc[0]
        c = rq3[(rq3.version == ver) & (rq3["index"] == ix)].iloc[0]
        row[f"RQ1 R² ({tag})"] = round(a.R2_scene, 2)
        row[f"RQ1 peak ({tag})"] = a.peak
        row[f"RQ2 confound R² ({tag})"] = round(b.R2_confounds, 2)
        row[f"RQ3 SAV−sand R² ({tag})"] = round(c.cycle_R2, 2)
        row[f"RQ3 p(sun) ({tag})"] = f"{c.p_sun_joint:.3f}"
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "table2_glint_sensitivity.csv", index=False)
print("[tab] table2_glint_sensitivity")
print(f"\n[out] {OUT}")
