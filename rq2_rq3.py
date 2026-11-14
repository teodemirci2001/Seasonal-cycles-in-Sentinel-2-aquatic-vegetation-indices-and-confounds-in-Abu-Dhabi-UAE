"""
RQ2 + RQ3 on the AOI2 scene tables (scene level, haze-flagged scenes removed).

RQ2  index ~ log(turbidity) + tide + measured glint + orbit             (MAIN spec)
     measured glint = reference-water rho(1610) (validated vs glint angle, r = -0.75).
     Sun zenith is NOT included: it is ~a pure annual curve (r = -0.98 with season) and
     would absorb any biological cycle. The old geometry spec (glint angle + sun zenith)
     is kept as a sensitivity run ("geometry").
     -> R2, LMG (Shapley) shares, standardised coefs with HAC (maxlags 4) p, VIF, Durbin-Watson
RQ3  (a) annual harmonic on RQ2 residuals; (b) HAC test: does SST add beyond the confounds?
     (c) MAIN: vegetation-minus-sand difference. Turbidity, tide, glint and sun act on both
         surfaces, so the difference cancels shared confounds without regressing out the season.
         Harmonic on the difference; SST vs sun zenith test (SST lags the sun by ~2 months, so
         with both standardised in one HAC model, the one that carries the signal is identifiable).
"""
import itertools, warnings, numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
SPECS = {"main":     ["log_tur_deep", "tide_m", "deep_rho1610", "orbit_R063"],
         "geometry": ["log_tur_deep", "tide_m", "glint_angle_deg", "sun_zenith_deg", "orbit_R063"]}
NAMES = {"log_tur_deep": "turbidity", "tide_m": "tide", "deep_rho1610": "glint", "glint_angle_deg": "glint",
         "sun_zenith_deg": "sun", "orbit_R063": "orbit"}
HAC = dict(cov_type="HAC", cov_kwds={"maxlags": 4})
z = lambda s: (s - s.mean()) / s.std()


def r2(y, X): return sm.OLS(y, sm.add_constant(X)).fit().rsquared if X.shape[1] else 0.0


def lmg(y, X):
    cols = list(X.columns); k = len(cols); share = dict.fromkeys(cols, 0.0)
    for c in cols:
        others = [o for o in cols if o != c]
        for m in range(k):
            subsets = list(itertools.combinations(others, m))
            for S in subsets:
                share[c] += (r2(y, X[list(S) + [c]]) - r2(y, X[list(S)])) / (k * len(subsets))
    return share


def harm(t, y):
    X = np.column_stack([np.ones_like(t, float), np.sin(2*np.pi*t/365.25), np.cos(2*np.pi*t/365.25)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); f = X @ b; rss1, rss0, n = np.sum((y-f)**2), np.sum((y-y.mean())**2), len(y)
    F = ((rss0-rss1)/2)/(rss1/(n-3)); pk = (np.degrees(np.arctan2(b[1], b[2])) % 360)/360*365.25
    return 1-rss1/rss0, 1-stats.f.cdf(F, 2, n-3), np.hypot(b[1], b[2]), (pd.Timestamp("2021-01-01")+pd.Timedelta(days=pk)).strftime("%b")


def load(ver):
    d = pd.read_csv(ROOT / f"scene_table_aoi2_{ver}.csv", parse_dates=["date"]).sort_values("date")
    d = d[~d.flag_exclude.astype(bool)].copy()
    d["orbit_R063"] = (d.relative_orbit == "R063").astype(float)
    return d.replace([np.inf, -np.inf], np.nan)


rows, diffs = [], []
for ver in ("noglint", "glint"):
    d = load(ver)
    # RQ2 + RQ3a/b
    for spec, CONF in SPECS.items():
        for area in ("veg", "sand"):
            for ix in ("ndavi", "wavi", "dii"):
                yc = f"{area}_{ix}"
                s = d[[yc, "sst_pixel_c", "date"] + CONF].dropna()
                Z = z(s[CONF]); y = z(s[yc])
                m = sm.OLS(y, sm.add_constant(Z)).fit(**HAC)
                sh = lmg(y, Z)
                vif = [variance_inflation_factor(sm.add_constant(Z).values, i + 1) for i in range(len(CONF))]
                rr2, rp, _, rpk = harm(s.date.dt.dayofyear.values, m.resid.values)
                mf = sm.OLS(y, sm.add_constant(Z.assign(sst=z(s.sst_pixel_c)))).fit(**HAC)
                rows.append(dict(version=ver, spec=spec, area=area, index=ix.upper(), n=len(s), R2_confounds=m.rsquared,
                                 **{f"LMG_{NAMES[c]}": sh[c] for c in CONF}, **{f"beta_{NAMES[c]}": m.params[c] for c in CONF},
                                 **{f"p_{NAMES[c]}": m.pvalues[c] for c in CONF}, max_VIF=max(vif), DW=durbin_watson(m.resid),
                                 resid_cycle_R2=rr2, resid_cycle_p=rp, resid_peak=rpk,
                                 dR2_SST=mf.rsquared - m.rsquared, beta_SST=mf.params["sst"], p_SST_HAC=mf.pvalues["sst"]))
    # RQ3c: vegetation minus sand
    for ix in ("ndavi", "wavi", "dii"):
        s = d[["date", f"veg_{ix}", f"sand_{ix}", "sst_pixel_c", "sun_zenith_deg"] + SPECS["main"]].dropna()
        dy = s[f"veg_{ix}"] - s[f"sand_{ix}"]
        hr2, hp, hamp, hpk = harm(s.date.dt.dayofyear.values, dy.values)
        both = sm.OLS(z(dy), sm.add_constant(pd.DataFrame({"sst": z(s.sst_pixel_c), "sun": z(s.sun_zenith_deg)}))).fit(**HAC)
        rem = sm.OLS(z(dy), sm.add_constant(z(s[SPECS["main"]]))).fit(**HAC)        # anything shared that did NOT cancel?
        diffs.append(dict(version=ver, index=ix.upper(), n=len(s), cycle_R2=hr2, cycle_p=hp, amp=hamp, amp_over_sd=hamp / dy.std(),
                          peak=hpk, r_SST=np.corrcoef(dy, s.sst_pixel_c)[0, 1], r_sun=np.corrcoef(dy, s.sun_zenith_deg)[0, 1],
                          beta_SST_joint=both.params["sst"], p_SST_joint=both.pvalues["sst"],
                          beta_sun_joint=both.params["sun"], p_sun_joint=both.pvalues["sun"],
                          R2_confounds_on_diff=rem.rsquared))

R = pd.DataFrame(rows); R.to_csv(OUT / "rq2_rq3_aoi2.csv", index=False)
D = pd.DataFrame(diffs); D.to_csv(OUT / "rq3_veg_minus_sand_aoi2.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
f = lambda X, c: X[c].round(2)
for spec in SPECS:
    r = R[R.spec == spec]
    print(f"\n[RQ2] spec = {spec}: {' + '.join(NAMES[c] for c in SPECS[spec])}  (standardised, HAC SEs)")
    cols = {"ver": r.version, "area": r.area, "idx": r["index"], "n": r.n, "R2": f(r, "R2_confounds")}
    cols.update({NAMES[c]: f(r, f"LMG_{NAMES[c]}") for c in SPECS[spec]})
    cols.update({"maxVIF": r.max_VIF.round(1), "DW": f(r, "DW")})
    print(pd.DataFrame(cols).to_string(index=False))
r = R[R.spec == "main"]
print("\n[RQ3a/b] main spec: what is left after removing confounds")
print(pd.DataFrame({"ver": r.version, "area": r.area, "idx": r["index"], "resid_cycle_R2": f(r, "resid_cycle_R2"),
      "resid_p": r.resid_cycle_p.map(lambda v: f"{v:.1e}"), "resid_peak": r.resid_peak, "dR2_SST": r.dR2_SST.round(3),
      "beta_SST": f(r, "beta_SST"), "p_SST(HAC)": r.p_SST_HAC.map(lambda v: f"{v:.1e}")}).to_string(index=False))
print("\n[RQ3c] vegetation minus sand (shared confounds cancel)")
print(pd.DataFrame({"ver": D.version, "idx": D["index"], "n": D.n, "cycle_R2": f(D, "cycle_R2"), "p_cycle": D.cycle_p.map(lambda v: f"{v:.1e}"),
      "amp/SD": f(D, "amp_over_sd"), "peak": D.peak, "r_SST": f(D, "r_SST"), "r_sun": f(D, "r_sun"),
      "b_SST|sun": f(D, "beta_SST_joint"), "p_SST": D.p_SST_joint.map(lambda v: f"{v:.1e}"),
      "b_sun|SST": f(D, "beta_sun_joint"), "p_sun": D.p_sun_joint.map(lambda v: f"{v:.1e}"),
      "R2_left_confounds": f(D, "R2_confounds_on_diff")}).to_string(index=False))
print(f"\n[out] {OUT/'rq2_rq3_aoi2.csv'} | {OUT/'rq3_veg_minus_sand_aoi2.csv'}")
