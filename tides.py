#!/usr/bin/env python3
"""
tides.py - EOT20 tide height at every scene's capture time (replaces FES/AVISO).

ONE-TIME SETUP (separate env so ACOLITE's env isn't disturbed), Miniforge prompt:
    mamba create -n tides -c conda-forge python=3.11 pip -y
    mamba activate tides
    pip install eo-tides
RUN:
    python E:\\pipeline_out\\code\\06_ancillary\\tides.py

What it does
  1. Downloads EOT20 (open, CC BY 4.0, DGFI-TUM; SEANOE doi:10.17882/79489) if not present,
     unpacks to E:\\pipeline_out\\ancillary\\tide_models\\EOT20\\ocean_tides\\*.nc
  2. Predicts tide height (m, relative to mean sea level) at each scene's sensing_time_utc
     at a water point in the AOI.
  3. Writes ancillary/tide_scene.csv and adds tide_m to both AOI2 scene tables.
"""
import io, sys, zipfile, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ANC = ROOT / "ancillary"; MODELS = ANC / "tide_models"; EOT = MODELS / "EOT20" / "ocean_tides"
URL = "https://www.seanoe.org/data/00683/79489/data/85762.zip"
POINT = (54.38, 24.36)        # lon, lat - central lagoon inside the two-orbit (analysed) area

def setup_eot20():
    if EOT.exists() and len(list(EOT.glob("*_ocean_eot20.nc"))) >= 15:
        print(f"[EOT20] found {len(list(EOT.glob('*.nc')))} constituent files"); return
    EOT.mkdir(parents=True, exist_ok=True); outer = ANC / "eot20_85762.zip"
    if not outer.exists():
        print(f"[EOT20] downloading {URL} (large; a few minutes)...", flush=True)
        with urllib.request.urlopen(URL) as r, open(outer, "wb") as f:
            total = int(r.headers.get("Content-Length", 0)); got = 0
            while chunk := r.read(1 << 22):
                f.write(chunk); got += len(chunk)
                if total: print(f"  {got/1e6:,.0f} / {total/1e6:,.0f} MB", end="\r", flush=True)
        print()
    with zipfile.ZipFile(outer) as z:
        inner = [n for n in z.namelist() if n.endswith("ocean_tides.zip")]
        if not inner: sys.exit(f"[error] ocean_tides.zip not inside {outer.name}: {z.namelist()[:10]}")
        with zipfile.ZipFile(io.BytesIO(z.read(inner[0]))) as z2:
            for n in z2.namelist():
                if n.endswith(".nc"): (EOT / Path(n).name).write_bytes(z2.read(n))
    print(f"[EOT20] unpacked {len(list(EOT.glob('*.nc')))} files -> {EOT}")

def main():
    setup_eot20()
    from eo_tides.model import model_tides
    log = pd.read_csv(ROOT / "image_log.csv")
    t = pd.to_datetime(log["sensing_time_utc"])
    out = model_tides(x=POINT[0], y=POINT[1], time=t.values, model="EOT20", directory=str(MODELS), output_format="wide")
    out = out.reset_index()
    tide = out["EOT20"].values
    res = pd.DataFrame({"scene": log["scene"].str.replace(".SAFE", "", regex=False), "sensing_time_utc": log["sensing_time_utc"], "tide_m": tide.round(3)})
    res.to_csv(ANC / "tide_scene.csv", index=False)
    print(f"[tide] {res.tide_m.notna().sum()}/{len(res)} scenes | range {res.tide_m.min():.2f} to {res.tide_m.max():.2f} m | sd {res.tide_m.std():.2f} m")
    if res.tide_m.isna().all():
        sys.exit("[error] all NaN - point may be on EOT20 land mask; tell Claude")
    for tag in ("aoi2_noglint", "aoi2_glint"):
        p = ROOT / f"scene_table_{tag}.csv"; d = pd.read_csv(p)
        d = d.drop(columns=[c for c in ("tide_m",) if c in d.columns]).merge(res[["scene", "tide_m"]], on="scene", how="left")
        d.to_csv(p, index=False); print(f"[merge] {p.name}: tide added to {d.tide_m.notna().sum()}/{len(d)} scenes")
    print("[done] tell Claude - RQ2 can now include tide")

if __name__ == "__main__":
    main()
