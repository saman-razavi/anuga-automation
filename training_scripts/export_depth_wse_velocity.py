#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Depth + WSE GeoTIFFs from ANUGA .sww, with an optional DEM mask so the output
does NOT "paint" values outside your DEM footprint (i.e., beyond the DEM nodata mask).

Important note:
- A GeoTIFF is always rectangular (rows/cols).
- The goal is: pixels outside the DEM footprint should be NODATA, so in GIS it *looks*
  non-rectangular (only DEM-valid area shows values).

Example (1 m output as DEM resolution, masked by the DEM footprint):
  python export_depth_wse_masked.py \
    --sww 20140702000000_Shellmouth_flood_12_days.sww \
    --dem-mask Terrain_Final.dem_rev1.tif \
    --outdir ./exports_10m_masked \
    --start "2014-07-02 00:00:00" \
    --dates "2014-07-08 00:00:00,2014-07-09 00:00:00,2014-07-10 00:00:00,2014-07-11 00:00:00" \
    --res 10 \
    --epsg 26914 \
    --depth-min 0.05

Notes:
- --depth-min is provided (e.g., 0.05 m) to avoid "all cells are wet".
- If DEM CRS differs from SWW CRS, reproject DEM to EPSG:26914 first.
"""

import argparse
import datetime
import math
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from scipy.spatial import cKDTree

def parse_dt(s: str) -> datetime.datetime:
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

def find_time_index(times_sec: np.ndarray, start_dt_str: str, target_dt_str: str):
    start_dt = parse_dt(start_dt_str)
    target_dt = parse_dt(target_dt_str)
    target_sec = (target_dt - start_dt).total_seconds()
    t = np.asarray(times_sec, dtype=float)
    idx = int(np.argmin(np.abs(t - target_sec)))
    return idx, float(t[idx]), float(target_sec)

def load_sww_minimal(sww_path: Path):
    ds = Dataset(sww_path, "r")
    try:
        x = ds.variables["x"][:].astype(np.float64).copy()
        y = ds.variables["y"][:].astype(np.float64).copy()
        t = ds.variables["time"][:].astype(np.float64).copy()
        elev = ds.variables["elevation"][:].astype(np.float32).copy()
        stage = ds.variables["stage"][:].astype(np.float32)  # time x nodes
        uh = ds.variables["xmomentum"][:].astype(np.float32)
        vh = ds.variables["ymomentum"][:].astype(np.float32)
        return x, y, t, elev, stage, uh, vh
    finally:
        ds.close()

def compute_output_grid_from_dem(dem_path: Path, res: float, epsg: int | None):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(dem_path) as src:
        b = src.bounds
        crs = src.crs
        dem_epsg = crs.to_epsg() if crs is not None else None

    xmin, ymin, xmax, ymax = float(b.left), float(b.bottom), float(b.right), float(b.top)
    width = xmax - xmin
    height = ymax - ymin
    ncols = int(math.ceil(width / res))
    nrows = int(math.ceil(height / res))

    transform = from_origin(xmin, ymax, res, res)
    out_epsg = int(epsg) if epsg is not None else (int(dem_epsg) if dem_epsg is not None else None)

    return xmin, ymin, xmax, ymax, ncols, nrows, transform, out_epsg

def compute_output_grid_from_nodes(x: np.ndarray, y: np.ndarray, res: float):
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    width = xmax - xmin
    height = ymax - ymin
    ncols = int(math.ceil(width / res)) + 1
    nrows = int(math.ceil(height / res)) + 1

    import rasterio
    from rasterio.transform import from_origin
    transform = from_origin(xmin, ymax, res, res)
    return xmin, ymin, xmax, ymax, ncols, nrows, transform

def build_mask_from_dem_to_output(dem_path: Path, out_shape, out_transform, out_crs, resampling="nearest"):
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(dem_path) as src:
        dem = src.read(1, masked=False)
        src_transform = src.transform
        src_crs = src.crs
        nodata = src.nodata

    valid = np.isfinite(dem)
    if nodata is not None and np.isfinite(nodata):
        valid &= (dem != nodata)

    src_mask = valid.astype(np.uint8)
    dst_mask = np.zeros(out_shape, dtype=np.uint8)

    rmap = {"nearest": Resampling.nearest, "bilinear": Resampling.bilinear, "mode": Resampling.mode}
    resamp = rmap.get(resampling, Resampling.nearest)

    reproject(
        source=src_mask,
        destination=dst_mask,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=out_transform,
        dst_crs=out_crs,
        resampling=resamp,
        src_nodata=0,
        dst_nodata=0,
    )
    return dst_mask

def make_profile(ncols, nrows, transform, epsg, nodata=-9999.0, block=512, bigtiff="IF_SAFER"):
    profile = {
        "driver": "GTiff",
        "width": ncols,
        "height": nrows,
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "nodata": np.float32(nodata),
        "tiled": True,
        "blockxsize": int(block),
        "blockysize": int(block),
        "compress": "deflate",
        "predictor": 3,
        "BIGTIFF": bigtiff,
    }
    if epsg is not None:
        profile["crs"] = f"EPSG:{int(epsg)}"
    return profile

def write_raster_blockwise(out_tif: Path, tree: cKDTree, values: np.ndarray,
                           ncols: int, nrows: int, transform, epsg,
                           mask: np.ndarray | None,
                           nodata=-9999.0, block=512, bigtiff="IF_SAFER"):
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import xy

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    profile = make_profile(ncols, nrows, transform, epsg, nodata=nodata, block=block, bigtiff=bigtiff)

    with rasterio.open(out_tif, "w", **profile) as dst:
        for row0 in range(0, nrows, block):
            h = min(block, nrows - row0)
            rows = np.arange(row0, row0 + h)

            for col0 in range(0, ncols, block):
                w = min(block, ncols - col0)
                cols = np.arange(col0, col0 + w)

                RR, CC = np.meshgrid(rows, cols, indexing="ij")
                xs, ys = xy(transform, RR, CC, offset="center")
                pts = np.c_[np.array(xs).ravel(), np.array(ys).ravel()]

                _, idx = tree.query(pts, k=1)
                block_data = values[idx].reshape(h, w).astype(np.float32)

                if mask is not None:
                    m = mask[row0:row0+h, col0:col0+w]
                    block_data = np.where(m == 1, block_data, np.float32(nodata))

                window = Window(col0, row0, w, h)
                dst.write(block_data, 1, window=window)

def main():
    p = argparse.ArgumentParser(description="Export Depth & WSE GeoTIFFs (masked by DEM footprint) from ANUGA .sww")
    p.add_argument("--sww", required=True, help="Path to .sww")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--start", required=True, help='Model start UTC "YYYY-MM-DD HH:MM:SS"')
    p.add_argument("--dates", required=True, help='Comma-separated UTC datetimes: "YYYY-MM-DD HH:MM:SS,..."')
    p.add_argument("--epsg", type=int, default=26914, help="EPSG for outputs (default 26914)")
    p.add_argument("--res", type=float, default=10.0, help="Output GeoTIFF resolution in meters (default 10)")
    p.add_argument("--block", type=int, default=512, help="GeoTIFF tile/block size (default 512)")

    p.add_argument("--dem-mask", default=None, help="DEM to use as mask footprint (pixels outside DEM nodata become NODATA)")
    p.add_argument("--depth-min", type=float, default=0.0, help="Set depth < depth-min to 0 (e.g., 0.05)")

    args = p.parse_args()

    sww_path = Path(args.sww)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Reading SWW (minimal): {sww_path}")
    x, y, t, elev, stage, uh, vh = load_sww_minimal(sww_path)

    print("Building KDTree...")
    tree = cKDTree(np.c_[x, y])

    mask = None
    if args.dem_mask:
        dem_path = Path(args.dem_mask)
        xmin, ymin, xmax, ymax, ncols, nrows, transform, out_epsg = compute_output_grid_from_dem(dem_path, args.res, args.epsg)
        if out_epsg is None:
            raise SystemExit("Could not determine EPSG from DEM; please pass --epsg explicitly.")
        print(f"Output grid FROM DEM bounds @ {args.res} m: cols={ncols}, rows={nrows}")
        print("Building DEM footprint mask at output resolution...")
        mask = build_mask_from_dem_to_output(dem_path, (nrows, ncols), transform, f"EPSG:{out_epsg}", resampling="nearest")
        epsg_use = out_epsg
    else:
        xmin, ymin, xmax, ymax, ncols, nrows, transform = compute_output_grid_from_nodes(x, y, args.res)
        epsg_use = args.epsg
        print(f"Output grid FROM SWW node bounds @ {args.res} m: cols={ncols}, rows={nrows}")

    date_list = [s.strip() for s in args.dates.split(",") if s.strip()]
    if not date_list:
        raise SystemExit("No valid --dates provided")

    for dt_str in date_list:
        ti, t_found, t_target = find_time_index(t, args.start, dt_str)
        tag = dt_str.replace(":", "").replace(" ", "_") + "UTC"
        print(f"\nDate {dt_str} -> target {t_target:.0f}s, picked idx={ti} (t={t_found:.0f}s)")

        wse_nodes = np.asarray(stage[ti, :], dtype=np.float32)
        # Depth at nodes
        depth_nodes = np.maximum(wse_nodes - elev, 0.0).astype(np.float32)

        # Apply depth threshold (dry cells -> 0 depth)
        if args.depth_min > 0:
            depth_nodes = np.where(depth_nodes >= args.depth_min, depth_nodes, 0.0).astype(np.float32)

        # Velocity magnitude at nodes (m/s): speed = sqrt((uh/depth)^2 + (vh/depth)^2)
        # uh, vh are xmomentum/ymomentum (m^2/s). Divide by depth (m) -> m/s
        eps = np.float32(1e-6)
        depth_safe = np.maximum(depth_nodes, eps)

        u_nodes = np.asarray(uh[ti, :], dtype=np.float32) / depth_safe
        v_nodes = np.asarray(vh[ti, :], dtype=np.float32) / depth_safe
        vel_nodes = np.sqrt(u_nodes*u_nodes + v_nodes*v_nodes).astype(np.float32)

        # Keep velocity 0 for dry cells
        vel_nodes = np.where(depth_nodes > 0, vel_nodes, 0.0).astype(np.float32)
        
        wse_tif = outdir / f"WSE_{tag}_res{int(args.res)}m.tif"
        dep_tif = outdir / f"DEPTH_{tag}_res{int(args.res)}m.tif"
        vel_tif = outdir / f"VELOCITY_{tag}_res{int(args.res)}m.tif"
        print(f"Writing {wse_tif.name} ...")
        write_raster_blockwise(wse_tif, tree, wse_nodes, ncols, nrows, transform, epsg_use, mask, block=args.block)

        print(f"Writing {dep_tif.name} ...")
        write_raster_blockwise(dep_tif, tree, depth_nodes, ncols, nrows, transform, epsg_use, mask, block=args.block)

        print(f"Writing {vel_tif.name} ...")
        write_raster_blockwise(vel_tif, tree, vel_nodes, ncols, nrows, transform, epsg_use, mask, block=args.block)
    print("\nDone. Outputs in:", outdir)

if __name__ == "__main__":
    main()
