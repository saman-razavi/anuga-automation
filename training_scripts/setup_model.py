#!/usr/bin/env python
# coding: utf-8

import os
#Detect if code is running without a display (typical on servers/HPC).
#Matplotlib switches to "Agg" to save plots to PNG instead of opening windows.
# headless mode for plotting
HEADLESS = os.environ.get("HEADLESS", "1") == "1" or not os.environ.get("DISPLAY")
if HEADLESS:
    import matplotlib
    matplotlib.use("Agg")

import sys
import argparse
from pathlib import Path

import yaml
import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

#prevents plt.show()
if HEADLESS:
    plt.show = lambda *a, **kw: None

import cmocean
import shutil
from tqdm import tqdm
from utils import data_processing_tools as dpt

import anuga
#MPI runs, each process gets a rank. ISROOT is rank 0 which avoids multiple ranks writing the same outpu files.
MYID, NUMPROCS = anuga.myid, anuga.numprocs
ISROOT = (MYID == 0)


##### The functions are Config loader using YAML file #######
def parse_args():
    ap = argparse.ArgumentParser(description="Setup utilities using config_example.yaml")
    ap.add_argument("--config", default="config_example.yaml", help="Path to yaml config")
    return ap.parse_args()


def load_config(cfg_path: str) -> dict:
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Base dir: config's workshop_dir is relative to config file location
    wk = cfg.get("paths", {}).get("workshop_dir", ".")
    workshop_dir = os.path.abspath(os.path.join(os.path.dirname(cfg_path), wk))
    #relative paths to absolute paths
    def R(p: str) -> str:
        if p is None:
            return None
        p = str(p)
        return p if os.path.isabs(p) else os.path.join(workshop_dir, p)

    cfg["_workshop_dir"] = workshop_dir
    cfg["_R"] = R
    return cfg


def finish_plot(path=None, dpi=150):
    if path:
        plt.savefig(path, bbox_inches="tight", dpi=dpi)
    elif not HEADLESS:
        plt.show()
    plt.close()


# --------------------------
# Read config + define paths
# --------------------------
args = parse_args()
CFG = load_config(args.config)
R = CFG["_R"]
#define/create working directories 
paths = CFG["paths"]
workshop_dir = CFG["_workshop_dir"]
data_dir = R(paths["data_dir"])
model_inputs_dir = R(paths["model_inputs_dir"])
model_outputs_dir = R(paths["model_outputs_dir"])
model_visuals_dir = R(paths["model_visuals_dir"])
model_validation_dir = R(paths["model_validation_dir"])

# Create dirs ONCE (fixes earlier bug where visuals/validation were undefined)
for d in [data_dir, model_inputs_dir, model_outputs_dir, model_visuals_dir, model_validation_dir]:
    Path(d).mkdir(parents=True, exist_ok=True)


# --------------------------
# DEM + background from config
# --------------------------
f_DEM_tif = R(CFG["dem"]["tif"])

DEM_src = rio.open(f_DEM_tif)
DEM = DEM_src.read(1)
resolution = DEM_src.res[0]

extent = [DEM_src.bounds.left, DEM_src.bounds.right,
          DEM_src.bounds.bottom, DEM_src.bounds.top]

# DEM nodata and elevations < 0 automatically masked
DEM = DEM_src.read(1, masked=True).astype("float32")
nd = DEM_src.nodata  # e.g., -32767 or -9999
DEM = np.ma.masked_where((~DEM.mask) & (DEM < 0), DEM)

cmap = cmocean.cm.topo.copy()
cmap.set_bad(alpha=0.0)

fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=200)
im1 = ax.imshow(DEM, cmap=cmap, extent=extent)
plt.colorbar(im1)
ax.set_title("DEM (nodata masked)")

if ISROOT:
    finish_plot(os.path.join(model_visuals_dir, "DEM.png"))

print("Range:", np.nanmin(DEM), np.nanmax(DEM), "| nodata:", nd)


# --------------------------
# Write edited DEM ASC from config
# --------------------------
f_edited_DEM_asc = R(CFG["dem"]["edited_asc"])
nodata_out = float(CFG["dem"].get("nodata_out", -9999.0))

if ISROOT and os.path.exists(f_edited_DEM_asc):
    os.remove(f_edited_DEM_asc)

# Note: all ranks can read, but only root writes (avoid MPI races)
with rio.open(f_DEM_tif) as src:
    arr = src.read(1, masked=True).astype("float32")
    src_nd = src.nodata

A = np.ma.filled(arr, fill_value=nodata_out).astype("float32")
A[A == -32767] = nodata_out  # catch any stray sentinel

profile = {
    "driver": "AAIGrid",
    "dtype": "float32",
    "width": A.shape[1],
    "height": A.shape[0],
    "count": 1,
    "crs": src.crs,
    "transform": src.transform,
    "nodata": nodata_out,
}

if ISROOT:
    with rio.open(f_edited_DEM_asc, "w", **profile) as dst:
        dst.write(A, 1)

    print("Wrote:", f_edited_DEM_asc)
    print("Min/Max (ignoring nodata):",
          np.nanmin(np.where(A == nodata_out, np.nan, A)),
          np.nanmax(np.where(A == nodata_out, np.nan, A)))
    print("Counts -> -32767:", np.sum(A == -32767), f" {nodata_out}:", np.sum(A == nodata_out))


# --------------------------
# US/DS boundary lines from config
# --------------------------
f_US_BC = R(CFG["boundaries"]["us_line_shp"])
f_DS_BC = R(CFG["boundaries"]["ds_line_shp"])

us_gdf = gpd.read_file(f_US_BC)
ds_gdf = gpd.read_file(f_DS_BC)

if us_gdf.crs != DEM_src.crs:
    us_gdf = us_gdf.to_crs(DEM_src.crs)
if ds_gdf.crs != DEM_src.crs:
    ds_gdf = ds_gdf.to_crs(DEM_src.crs)

us_bc_line = np.array(us_gdf.geometry.iloc[0].coords)
ds_bc_line = np.array(ds_gdf.geometry.iloc[0].coords)

fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
im = ax.imshow(DEM, cmap="cmo.topo", extent=extent, zorder=1)

ax.plot(us_bc_line[:, 0], us_bc_line[:, 1], color="red", linewidth=2, label="Upstream BC Line", zorder=2)
ax.plot(ds_bc_line[:, 0], ds_bc_line[:, 1], color="blue", linewidth=2, label="Downstream BC Line", zorder=2)

ax.set_title("DEM with Boundary Lines")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect("equal")
ax.legend(loc="lower right")
plt.colorbar(im, ax=ax, label="Elevation (m)")
plt.tight_layout()

if ISROOT:
    finish_plot(os.path.join(model_visuals_dir, "DEM_with_BC_Lines.png"))


# --------------------------
# (Optional) imports you had later — kept as-is
# --------------------------
from dataretrieval import nwis
import noaa_coops as noaa
from tqdm import notebook
import matplotlib
from anuga import Set_stage, Reflective_boundary
from anuga.structures.inlet_operator import Inlet_operator

from utils.anuga_tools import anuga_tools as at
from utils import data_processing_tools as dpt

import subprocess

is_windows = sys.platform.startswith("win")


# --------------------------
# Mesh extraction from triangle shapefile (from config)
# --------------------------
import math
import fiona
from shapely.geometry import shape

mesh_tri_shp = R(CFG["mesh"]["triangle_shp"])
pts_path = R(CFG["mesh"]["pts_npy"])
tris_path = R(CFG["mesh"]["tris_npy"])
dedup_tol = float(CFG["mesh"].get("dedup_tol", 1e-6))

def decimals_from_tol(tol: float) -> int:
    if tol <= 0:
        return 8
    return max(0, int(round(-math.log10(tol))))

def unique_indexer(decimals: int):
    def keyfun(x, y):
        return (round(float(x), decimals), round(float(y), decimals))
    return keyfun

def add_point_get_index(x, y, keyfun, node_index, nodes):
    k = keyfun(x, y)
    idx = node_index.get(k)
    if idx is None:
        idx = len(nodes)
        nodes.append([float(k[0]), float(k[1])])
        node_index[k] = idx
    return idx

def extract_triangles_from_geometry(geom):
    tris = []
    if geom.geom_type == "Polygon":
        rings = [geom.exterior]
    elif geom.geom_type == "MultiPolygon":
        rings = [g.exterior for g in geom.geoms]
    else:
        return tris

    for ring in rings:
        coords = list(ring.coords)
        if len(coords) >= 2 and (coords[0][0] == coords[-1][0] and coords[0][1] == coords[-1][1]):
            coords = coords[:-1]

        uniq = []
        seen = set()
        for (x, y) in coords:
            p = (x, y)
            if p not in seen:
                uniq.append(p)
                seen.add(p)
            if len(uniq) == 3:
                break

        if len(uniq) == 3:
            tris.append(uniq)
    return tris

decimals = decimals_from_tol(dedup_tol)
keyfun = unique_indexer(decimals)

nodes = []
node_index = {}
tri_indices = []

with fiona.open(mesh_tri_shp) as src:
    crs_info = src.crs_wkt or src.crs
    if ISROOT:
        print("Mesh shapefile CRS:", crs_info)

    n_feat = 0
    n_tri = 0
    for ft in src:
        n_feat += 1
        geom = shape(ft["geometry"]) if ft["geometry"] else None
        if geom is None:
            continue
        tri_rings = extract_triangles_from_geometry(geom)
        for tri in tri_rings:
            i = add_point_get_index(tri[0][0], tri[0][1], keyfun, node_index, nodes)
            j = add_point_get_index(tri[1][0], tri[1][1], keyfun, node_index, nodes)
            k = add_point_get_index(tri[2][0], tri[2][1], keyfun, node_index, nodes)
            tri_indices.append([i, j, k])
            n_tri += 1

if ISROOT:
    print(f"Features read: {n_feat} | Triangles extracted: {n_tri} | Unique nodes: {len(nodes)}")

pts = np.asarray(nodes, dtype=np.float64)
tris = np.asarray(tri_indices, dtype=np.int32)

if pts.size == 0 or tris.size == 0:
    raise RuntimeError("No points/triangles extracted. Ensure the shapefile contains triangle polygons.")

# Only root writes to avoid MPI clobber
if ISROOT:
    np.save(pts_path, pts)
    np.save(tris_path, tris)

    np.savetxt(os.path.splitext(pts_path)[0] + ".csv", pts, delimiter=",", header="x,y", comments="")
    np.savetxt(os.path.splitext(tris_path)[0] + ".csv", tris, delimiter=",", header="i,j,k", fmt="%d", comments="")

    print("Saved:")
    print(" -", pts_path)
    print(" -", tris_path)


# --------------------------
# Mesh-over-DEM overlay QA (config-driven)
# --------------------------
import pyproj
from shapely.geometry import LineString
from shapely.ops import transform as shp_transform
import matplotlib.tri as mtri

def reproject_points_xy(pts_xy: np.ndarray, src_crs, dst_crs):
    if src_crs is None or dst_crs is None:
        raise ValueError("CRS missing for reprojection")
    src = pyproj.CRS.from_user_input(src_crs)
    dst = pyproj.CRS.from_user_input(dst_crs)
    if src == dst:
        return pts_xy
    T = pyproj.Transformer.from_crs(src, dst, always_xy=True).transform
    x, y = T(pts_xy[:, 0], pts_xy[:, 1])
    out = pts_xy.copy()
    out[:, 0] = x
    out[:, 1] = y
    return out

def read_first_linestring_with_crs(path):
    with fiona.open(path) as src:
        crs = src.crs_wkt or src.crs
        for ft in src:
            g = ft["geometry"]
            if not g:
                continue
            if g["type"] == "LineString":
                return LineString(g["coordinates"]), crs
            if g["type"] == "MultiLineString":
                parts = [LineString(c) for c in g["coordinates"]]
                coords = [xy for part in parts for xy in part.coords]
                return LineString(coords), crs
    raise RuntimeError(f"No LineString in {path}")

def reproject_linestring(ls, src_crs, dst_crs):
    src = pyproj.CRS.from_user_input(src_crs)
    dst = pyproj.CRS.from_user_input(dst_crs)
    if src == dst:
        return ls
    T = pyproj.Transformer.from_crs(src, dst, always_xy=True).transform
    return shp_transform(T, ls)

# Load arrays (all ranks can read; only root plots/writes)
pts = np.load(pts_path, allow_pickle=False)
tris = np.load(tris_path, allow_pickle=False)
if not np.issubdtype(tris.dtype, np.integer):
    tris = tris.astype(np.int64, copy=False)

MESH_SRC_CRS = CFG["mesh"].get("src_crs", None)  # from YAML
us_line_shp = R(CFG["boundaries"]["us_line_shp"])
ds_line_shp = R(CFG["boundaries"]["ds_line_shp"])
US_raw, US_crs = read_first_linestring_with_crs(us_line_shp)
DS_raw, DS_crs = read_first_linestring_with_crs(ds_line_shp)

asc_dem = R(CFG["dem"]["edited_asc"])
out_over_dem_png = os.path.join(model_visuals_dir, "mesh_over_dem_preview.png")

try:
    with rio.open(asc_dem) as r:
        dem = r.read(1, masked=True)
        dem_crs = r.crs
        bounds = r.bounds
        dem_extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    pts_plot = reproject_points_xy(pts, MESH_SRC_CRS, dem_crs)
    US_plot = reproject_linestring(US_raw, US_crs, dem_crs)
    DS_plot = reproject_linestring(DS_raw, DS_crs, dem_crs)

    if ISROOT:
        fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
        im = ax.imshow(dem, extent=dem_extent, origin="upper")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Elevation")

        triang = mtri.Triangulation(pts_plot[:, 0], pts_plot[:, 1], triangles=tris)
        ax.triplot(triang, linewidth=0.25)

        ax.plot(*US_plot.xy, lw=1.5, label="US line")
        ax.plot(*DS_plot.xy, lw=1.5, label="DS line")

        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Mesh over DEM (all in DEM CRS)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_over_dem_png, bbox_inches="tight")
        plt.close()

        print(f"Saved: {out_over_dem_png}")

except Exception as e:
    if ISROOT:
        print(f"(Skipping DEM overlay: {e})")

# From here onward, CRS of the DEM is used
