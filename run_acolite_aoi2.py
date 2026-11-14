"""
run_acolite_aoi2.py - ACOLITE over the NEW AOI (clipped to tile T40RBN), two versions:
    noglint : dsf_residual_glint_correction=False
    glint   : but dsf_residual_glint_correction=True
Everything else is identical, so the two can be compared directly.
"""
import argparse, csv, glob, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   
SAFE_DIR = ROOT / "downloaded"
LIMIT = "24.29,54.01,24.49,54.49"     # S,W,N,E  (AOI clipped to T40RBN, +0.01 deg pad)
BANDS = {"S2A": [492, 560, 665, 833, 1614],      # band names as ACOLITE writes them 
         "S2B": [492, 559, 665, 833, 1610]}
BASE = {"limit": LIMIT, "s2_target_res": "10", "l2w_mask": "False",
        "export_geotiff": "False", "export_geotiff_coordinates": "False", "verbosity": "2",
        "l1r_delete_netcdf": "True", "l2r_delete_netcdf": "True"}

def free_gb(p): return shutil.disk_usage(p).free / 1e9
def has_l2w(d): return bool(glob.glob(str(d / "*_L2W.nc")))

def worker(mode, acolite, limit_n, min_free):
    out_root = ROOT / f"acolite_aoi2_{mode}"; out_root.mkdir(exist_ok=True)
    launcher = Path(acolite) / "launch_acolite.py"
    todo = [s for s in sorted(SAFE_DIR.glob("*.SAFE")) if not has_l2w(out_root / s.name[:-5])]
    if limit_n: todo = todo[:limit_n]
    log = ROOT / f"acolite_aoi2_{mode}_log.csv"; new = not log.exists()
    print(f"[{mode}] {len(todo)} scenes to do -> {out_root}", flush=True)
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["scene", "start", "end", "elapsed_s", "status", "free_gb_after", "note"])
        ok = 0
        for i, safe in enumerate(todo, 1):
            if free_gb(ROOT) < min_free:
                print(f"[{mode}] STOP: only {free_gb(ROOT):.0f} GB free"); break
            out = out_root / safe.name[:-5]; out.mkdir(parents=True, exist_ok=True)
            sat = safe.name[:3]
            par = ",".join([f"Rrs_{b}" for b in BANDS[sat]] + ["T_Nechad2016"])
            s = dict(BASE, inputfile=str(safe), output=str(out), l2w_parameters=par,
                     dsf_residual_glint_correction="True" if mode == "glint" else "False")
            sf = out / "settings.txt"; sf.write_text("\n".join(f"{k}={v}" for k, v in s.items()) + "\n")
            t0 = time.time(); st = datetime.now(); note = ""
            try:
                with open(out / "stdout.txt", "w") as so:
                    rc = subprocess.run([sys.executable, str(launcher), "--cli", "--settings", str(sf)],
                                        stdout=so, stderr=subprocess.STDOUT, timeout=45 * 60).returncode
                note = f"exit {rc}"
            except subprocess.TimeoutExpired:
                note = "timeout"
            status = "ok" if has_l2w(out) else "FAILED"; ok += status == "ok"; el = int(time.time() - t0)
            print(f"[{mode}] {i}/{len(todo)} {safe.name[:26]} -> {status} {el//60}m{el%60:02d}s  free {free_gb(ROOT):.0f} GB", flush=True)
            w.writerow([safe.name, st.strftime("%Y-%m-%d %H:%M:%S"), datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        el, status, round(free_gb(ROOT), 1), note]); f.flush()
    print(f"[{mode}] done: {ok} ok this run")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["noglint", "glint", "both"], default="both")
    ap.add_argument("--acolite", default=str(Path.home() / "acolite"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenes (testing)")
    ap.add_argument("--min-free-gb", type=float, default=15)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    safes = sorted(SAFE_DIR.glob("*.SAFE"))
    print(f"[root] {ROOT}\n[acolite] {Path(a.acolite)/'launch_acolite.py'} "
          f"{'OK' if (Path(a.acolite)/'launch_acolite.py').exists() else 'NOT FOUND'}\n[python] {sys.executable}")
    for m in ("noglint", "glint"):
        d = ROOT / f"acolite_aoi2_{m}"
        print(f"[{m}] {sum(not has_l2w(d / s.name[:-5]) for s in safes)} of {len(safes)} scenes still to do")
    print(f"[disk] {free_gb(ROOT):.0f} GB free | est. ~0.30 GB/scene/version -> ~{0.30*len(safes)*2:.0f} GB for both")
    print(f"[limit] {LIMIT} (S,W,N,E)")
    if a.dry_run: return
    if a.mode == "both":   # two workers in parallel, each its own process
        procs = [subprocess.Popen([sys.executable, __file__, "--mode", m, "--acolite", a.acolite,
                                   "--limit", str(a.limit), "--min-free-gb", str(a.min_free_gb)]) for m in ("noglint", "glint")]
        for p in procs: p.wait()
    else:
        worker(a.mode, a.acolite, a.limit, a.min_free_gb)

if __name__ == "__main__":
    main()
