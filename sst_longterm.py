#!/usr/bin/env python3
"""
sst_longterm.py - OSTIA daily SST 2015-2025 (11 years) for the analysed area. Context only:
does NOT touch the scene tables or sst_daily.csv used in RQ2/RQ3.

Run on your PC (uses your saved Copernicus Marine login):
    mamba activate C:\\Users\\USER\\.conda\\envs\\acolite
    python E:\\pipeline_out\\code\\06_ancillary\\sst_longterm.py

Outputs (ancillary/):
    ostia_longterm_rep.nc (+ ostia_longterm_nrt.nc if REP stops early)
    sst_longterm_daily.csv     date, box mean over analysed area, centre pixel
    sst_longterm_annual.csv    per year: mean, max, Jun-Sep mean, days >= thresholds
    sst_longterm.png           daily series + annual means + trend
"""
import csv, sys, datetime as dt
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ANC = ROOT / "ancillary"; ANC.mkdir(exist_ok=True)
BOX = dict(minimum_longitude=53.95, maximum_longitude=54.55, minimum_latitude=24.25, maximum_latitude=24.70)
CENTRE = (54.38, 24.36)
FOOT = (54.22, 54.50, 24.29, 24.50)          # analysed (two-orbit) area
START, END = "2015-01-01T00:00:00", "2025-12-31T23:59:59"
REP = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
NRT = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"
THRESH = {"days_ge_33p7": 33.7,  # site estimate from Marba et al. (2022) regression with mean SST 28.3 C
          "days_ge_34p7": 34.7}  # mean upper thermal limit of tropical/subtropical seagrasses (Marba et al. 2022)


def fetch(dataset, start, end, name):
    out = ANC / name
    if out.exists():
        print(f"[skip] {out.name} already downloaded"); return out
    import copernicusmarine as cm
    print(f"[download] {dataset}  {start[:10]} -> {end[:10]} (a few minutes)", flush=True)
    cm.subset(dataset_id=dataset, variables=["analysed_sst"], start_datetime=start, end_datetime=end,
              output_directory=str(ANC), output_filename=name, **BOX)
    return out


def read(p):
    import netCDF4 as nc
    d = nc.Dataset(p)
    t = nc.num2date(d.variables["time"][:], d.variables["time"].units, only_use_cftime_datetimes=False)
    days = [dt.date(x.year, x.month, x.day) for x in t]
    sst = np.ma.filled(d.variables["analysed_sst"][:].astype("f8"), np.nan)
    if np.nanmedian(sst) > 200: sst -= 273.15
    return days, np.asarray(d.variables["latitude"][:]), np.asarray(d.variables["longitude"][:]), sst


def main():
    days, lat, lon, sst = read(fetch(REP, START, END, "ostia_longterm_rep.nc"))
    src = ["REP"] * len(days)
    print(f"[REP] {days[0]} -> {days[-1]} ({len(days)} days)")
    if days[-1] < dt.date(2025, 12, 31):
        s2 = (days[-1] + dt.timedelta(days=1)).isoformat() + "T00:00:00"
        d2, la2, lo2, s2v = read(fetch(NRT, s2, END, "ostia_longterm_nrt.nc"))
        if s2v.shape[1:] != sst.shape[1:]:
            sys.exit("[error] REP and NRT grids differ - tell Claude")
        days += d2; sst = np.concatenate([sst, s2v]); src += ["NRT"] * len(d2)
        print(f"[NRT] filled {d2[0]} -> {d2[-1]}  (product switch: note it as a caveat)")
    LON, LAT = np.meshgrid(lon, lat)
    water = np.isfinite(sst).all(axis=0)
    foot = water & (LON >= FOOT[0]) & (LON <= FOOT[1]) & (LAT >= FOOT[2]) & (LAT <= FOOT[3])
    dist = np.hypot((LON - CENTRE[0]) * np.cos(np.radians(CENTRE[1])), LAT - CENTRE[1]); dist[~water] = np.inf
    iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
    box, px = np.nanmean(sst[:, foot], axis=1), sst[:, iy, ix]
    print(f"[area] {foot.sum()} OSTIA pixels in analysed area | centre pixel {lon[ix]:.3f}E {lat[iy]:.3f}N")
    with open(ANC / "sst_longterm_daily.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["date", "sst_box_c", "sst_pixel_c", "product"])
        for d, b, p, s in zip(days, box, px, src): w.writerow([d.isoformat(), round(float(b), 3), round(float(p), 3), s])

    years = sorted({d.year for d in days}); rows = []
    yr = np.array([d.year for d in days]); mo = np.array([d.month for d in days])
    for y in years:
        k = yr == y; v = box[k]; summer = box[k & (mo >= 6) & (mo <= 9)]
        r = {"year": y, "n_days": int(k.sum()), "mean_c": round(float(np.nanmean(v)), 2), "max_c": round(float(np.nanmax(v)), 2),
             "jun_sep_mean_c": round(float(np.nanmean(summer)), 2)}
        r.update({n: int(np.sum(v >= t)) for n, t in THRESH.items()}); rows.append(r)
    with open(ANC / "sst_longterm_annual.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    from scipy import stats
    full = [r for r in rows if r["n_days"] >= 360]
    xs, ys = np.array([r["year"] for r in full]), np.array([r["mean_c"] for r in full])
    lr = stats.linregress(xs, ys)
    print("\n year  mean   max  Jun-Sep  days>=33.7  days>=34.7")
    for r in rows: print(f" {r['year']}  {r['mean_c']:5.2f} {r['max_c']:5.2f}  {r['jun_sep_mean_c']:5.2f}   {r['days_ge_33p7']:5d}     {r['days_ge_34p7']:5d}")
    print(f"\n[trend] annual mean {lr.slope*10:+.2f} C/decade (p = {lr.pvalue:.2f}, n = {len(full)} years)"
          f"  - 11 years is short: report as indicative, not a climate trend")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    ax.plot(days, box, lw=.6, color="#9ecae1", label="daily (box mean)")
    ax.plot([dt.date(y, 7, 1) for y in xs], ys, "o-", color="#2b5c8a", label="annual mean")
    ax.plot([dt.date(y, 7, 1) for y in xs], lr.intercept + lr.slope * xs, "--", color="#2b5c8a", lw=1,
            label=f"trend {lr.slope*10:+.2f} °C/decade")
    for t, c in ((34.7, "#d62728"), (33.7, "#e6a06a")): ax.axhline(t, color=c, lw=.8, ls=":")
    ax.text(days[0], 34.85, "34.7 °C mean T_limit, tropical seagrass (Marbà et al., 2022)", fontsize=7, color="#d62728")
    ax.set_ylabel("OSTIA SST (°C)"); ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(ANC / "sst_longterm.png", dpi=150)
    print(f"[out] {ANC/'sst_longterm_daily.csv'} | sst_longterm_annual.csv | sst_longterm.png")


if __name__ == "__main__":
    main()
