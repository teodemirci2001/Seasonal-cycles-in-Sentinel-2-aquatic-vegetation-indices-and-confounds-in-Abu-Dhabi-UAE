"""
cloud_fraction.py - per-scene cloud cover over the analysed seabed and the reference water.

Uses the Sentinel-2 L1C cloud mask (QI_DATA/MSK_CLASSI_B00.jp2, 60 m: band 1 opaque, band 2 cirrus),
dilated by 2 pixels (120 m) because the mask misses cloud edges, resampled (nearest) onto the 10 m mask grid.
    analysed = mask codes 4, 5, 6 (dark/mid/bright_seabed);  reference = code 7 (deep-water patch)
Writes ancillary/cloud_fraction_l1c.csv. extract_scene_table.py summary turns it into flag_cloud.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd, rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[2]
DILATE_PX = 2

with rasterio.open(ROOT / "extraction_aoi2_noglint" / "masks_v5.tif") as m:
    code, grid, crs = m.read(1), m.transform, m.crs
analysed, reference = np.isin(code, [4, 5, 6]), code == 7

rows = []
for p in sorted(glob.glob(str(ROOT / "downloaded" / "*.SAFE" / "GRANULE" / "*" / "QI_DATA" / "MSK_CLASSI_B00.jp2"))):
    scene = Path(p).parts[-5].replace(".SAFE", "")
    with rasterio.open(p) as r:
        cloud = (r.read(1) > 0) | (r.read(2) > 0)
        cloud = binary_dilation(cloud, iterations=DILATE_PX).astype("u1")
        on_grid = np.zeros(code.shape, "u1")
        reproject(cloud, on_grid, src_transform=r.transform, src_crs=r.crs,
                  dst_transform=grid, dst_crs=crs, resampling=Resampling.nearest)
    rows.append({"scene": scene, "cloud_frac_analysed": on_grid[analysed].mean(), "cloud_frac_ref": on_grid[reference].mean()})

out = ROOT / "ancillary" / "cloud_fraction_l1c.csv"
df = pd.DataFrame(rows); df.to_csv(out, index=False)
print(f"[cloud] {len(df)} scenes | >1 % cloud over analysed seabed: {(df.cloud_frac_analysed > 0.01).sum()} | "
      f"over reference water: {(df.cloud_frac_ref > 0.01).sum()} -> {out}")
