"""RQ1 on the AOI scene tables: annual harmonic per index x area x glint version.
Scene-level fit (n~209); max rate of change = 2*pi*amp/12 per month (steepest slope of the fitted cycle) + monthly-mean fit; repeatability = peak month per year."""
import numpy as np, pandas as pd, warnings
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
T = {v: pd.read_csv(ROOT / f"scene_table_aoi2_{v}.csv", parse_dates=["date"]) for v in ("noglint", "glint")}

def harm(t, y):
    X = np.column_stack([np.ones_like(t, float), np.sin(2*np.pi*t/365.25), np.cos(2*np.pi*t/365.25)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); f = X @ b
    rss1, rss0, n = np.sum((y-f)**2), np.sum((y-y.mean())**2), len(y)
    F = ((rss0-rss1)/2)/(rss1/(n-3)); p = 1-stats.f.cdf(F, 2, n-3)
    pk = (np.degrees(np.arctan2(b[1], b[2])) % 360)/360*365.25
    return dict(R2=1-rss1/rss0, p=p, amp=np.hypot(b[1], b[2]), rate=np.hypot(b[1], b[2])*2*np.pi/12, peak=(pd.Timestamp("2021-01-01")+pd.Timedelta(days=pk)).strftime("%b"), n=n)

rows = []
for ver, df in T.items():
    df = df[~df.flag_exclude.astype(bool) & df.veg_ndavi.notna()]
    for ix in ("ndavi", "wavi", "dii"):
        for ar in ("veg", "sand", "deep"):
            s = df[["date", f"{ar}_{ix}"]].dropna(); y = s[f"{ar}_{ix}"].values
            sc = harm(s.date.dt.dayofyear.values, y)
            mo = s.groupby(pd.Grouper(key="date", freq="MS"))[f"{ar}_{ix}"].mean().dropna()
            mm = harm(mo.index.dayofyear.values, mo.values)
            pk = [g.idxmax().month for _, g in mo.groupby(mo.index.year) if len(g) >= 8]
            rows.append(dict(version=ver, index=ix.upper(), area=ar, n_scenes=sc["n"], R2_scene=sc["R2"], p_scene=sc["p"],
                             amp=sc["amp"], amp_over_sd=sc["amp"]/np.std(y), max_rate_per_month=sc["rate"], peak=sc["peak"], R2_monthly=mm["R2"], peak_by_year=pk))
R = pd.DataFrame(rows); R.to_csv(OUT / "rq1_harmonic_aoi2.csv", index=False)
pd.set_option("display.width", 200)
print("[RQ1] annual harmonic, scene level (haze- and cloud-flagged scenes removed)")
print(R.assign(p_scene=R.p_scene.map(lambda v: f"{v:.0e}"), R2_scene=R.R2_scene.round(2), R2_monthly=R.R2_monthly.round(2),
               amp=R.amp.round(4), amp_over_sd=R.amp_over_sd.round(2), max_rate_per_month=R.max_rate_per_month.round(4)).to_string(index=False))
print("\n[key] veg-vs-sand: biological signal should give LARGER amplitude over vegetation than sand")
for ver in ("noglint", "glint"):
    for ix in ("NDAVI", "WAVI", "DII"):
        a = R[(R.version == ver) & (R["index"] == ix)].set_index("area")
        print(f"  {ver:7} {ix:5}  amp veg {a.amp['veg']:.4f} vs sand {a.amp['sand']:.4f}  -> ratio {a.amp['veg']/a.amp['sand']:.2f} | R2 veg {a.R2_scene['veg']:.2f} sand {a.R2_scene['sand']:.2f}")

# figure: monthly climatology, veg (blue) vs sand (orange), solid=no glint, dashed=glint
C = {"veg": "#2a78d6", "sand": "#eb6834"}
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
for i, ix in enumerate(("ndavi", "wavi", "dii")):
    for ver, ls in (("noglint", "-"), ("glint", "--")):
        df = T[ver][~T[ver].flag_exclude.astype(bool)]
        for ar in ("veg", "sand"):
            cl = df.groupby(df.date.dt.month)[f"{ar}_{ix}"].agg(["mean", "sem"])
            ax[i].plot(cl.index, cl["mean"], color=C[ar], ls=ls, lw=2, marker="o" if ls == "-" else None, ms=3.5,
                       label=f"{'Vegetation' if ar=='veg' else 'Sand'} ({'no glint corr.' if ver=='noglint' else 'glint-corrected'})")
            if ls == "-": ax[i].fill_between(cl.index, cl["mean"]-cl["sem"], cl["mean"]+cl["sem"], color=C[ar], alpha=.12, lw=0)
    ax[i].set_title(ix.upper(), loc="left"); ax[i].set_xticks(range(1, 13)); ax[i].set_xticklabels("JFMAMJJASOND")
    ax[i].grid(color="#e4e4e0", lw=.6); ax[i].spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("Mean by calendar month, 2019–2023 (±1 SE)"); ax[2].legend(frameon=False, fontsize=7.5)
fig.suptitle("RQ1 – new AOI: seasonal cycle over dark (vegetation-candidate) vs bright (sand) seabed, with and without ACOLITE glint correction", fontsize=10, x=.01, ha="left")
fig.savefig(OUT / "rq1_seasonal_aoi2.png", dpi=150); print(f"\n[out] {OUT/'rq1_seasonal_aoi2.png'} | rq1_harmonic_aoi2.csv")
