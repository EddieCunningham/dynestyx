"""Build a real Lake Champlain environment raster from public Vermont GIS data.

Two open layers from the Vermont Open Geodata portal supply the geometry the
paper draws on.

  - The lake outline polygon (VT Lake Champlain, extracted from VHDCARTO).
  - The Lake Champlain bathymetry point cloud (DEPTH_FT below the surface).

Both layers are served in EPSG:32145 (Vermont State Plane, metres), which is a
planar frame, so no reprojection is needed. The script rasterises the polygon
into a water mask, interpolates the depth points onto the same grid, and writes
an ``.npz`` that ``champlain_dsx.environment.load_champlain`` reads.

The proprietary survey rasters used in the paper are not public. This script
reconstructs the lake's real shoreline and bathymetry from the open layers,
which is enough to drive the model on genuine Lake Champlain geometry.

Usage:
    uv run python champlain_dsx/scripts/acquire_champlain_map.py --res 200
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.interpolate import griddata

BASE = "https://services1.arcgis.com/BkFxaEFNwHqX3tAw/arcgis/rest/services"
POLYGON_LAYER = f"{BASE}/FS_VCGI_OPENDATA_V_WATER_LKCH5K_POLY_SP_v1/FeatureServer/0"
DEPTH_LAYER = f"{BASE}/FS_VCGI_OPENDATA_Elevation_LKCHDEM_point_SP_v1/FeatureServer/0"
SR = 32145
FT_TO_M = 0.3048


def _get(url: str, params: dict, retries: int = 4) -> dict:
    query = urllib.parse.urlencode(params)
    full = f"{url}?{query}"
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(full, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Request failed after {retries} attempts: {full}\n{last}")


def fetch_polygon_rings() -> list[np.ndarray]:
    """Return the polygon rings as arrays of (x, y) vertices in EPSG:32145."""
    data = _get(
        f"{POLYGON_LAYER}/query",
        {"where": "1=1", "outFields": "", "outSR": SR, "f": "geojson"},
    )
    rings: list[np.ndarray] = []
    for feature in data["features"]:
        geom = feature["geometry"]
        if geom["type"] == "Polygon":
            polys = [geom["coordinates"]]
        elif geom["type"] == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue
        for poly in polys:
            for ring in poly:
                rings.append(np.asarray(ring, dtype=np.float64))
    return rings


def fetch_depth_points() -> tuple[np.ndarray, np.ndarray]:
    """Download the bathymetry points. Returns (xy, depth_m) for wet points."""
    count = _get(
        f"{DEPTH_LAYER}/query",
        {"where": "DEPTH_FT<0", "returnCountOnly": "true", "f": "json"},
    )["count"]
    page = 2000
    xs, ys, depths = [], [], []
    for offset in range(0, count, page):
        data = _get(
            f"{DEPTH_LAYER}/query",
            {
                "where": "DEPTH_FT<0",
                "outFields": "DEPTH_FT",
                "orderByFields": "OBJECTID",
                "resultOffset": offset,
                "resultRecordCount": page,
                "outSR": SR,
                "f": "geojson",
            },
        )
        for feat in data["features"]:
            x, y = feat["geometry"]["coordinates"]
            ft = feat["properties"]["DEPTH_FT"]
            xs.append(x)
            ys.append(y)
            depths.append(-float(ft) * FT_TO_M)
        print(f"  depth points {min(offset + page, count)}/{count}", flush=True)
    xy = np.stack([np.asarray(xs), np.asarray(ys)], axis=-1)
    return xy, np.asarray(depths)


def _rings_to_masks(rings, grid_xy, shape):
    """Rasterise polygon rings with the even-odd rule.

    A cell in open water lies inside the single exterior ring, an odd count. A
    cell on an island lies inside both the exterior and the island ring, an even
    count. Parity therefore carves islands out of the lake without relying on
    ring orientation, which Esri and GeoJSON disagree on.
    """
    parity = np.zeros(grid_xy.shape[0], dtype=np.int32)
    for ring in rings:
        contained = MplPath(ring).contains_points(grid_xy)
        parity += contained.astype(np.int32)
    return ((parity % 2) == 1).reshape(shape)


def build_raster(res: float, pad: float = 1000.0):
    print("Fetching lake polygon ...", flush=True)
    rings = fetch_polygon_rings()
    all_xy = np.concatenate(rings, axis=0)
    x_min = np.floor((all_xy[:, 0].min() - pad) / res) * res
    x_max = np.ceil((all_xy[:, 0].max() + pad) / res) * res
    y_min = np.floor((all_xy[:, 1].min() - pad) / res) * res
    y_max = np.ceil((all_xy[:, 1].max() + pad) / res) * res

    width = int(round((x_max - x_min) / res))
    height = int(round((y_max - y_min) / res))
    cx = x_min + (np.arange(width) + 0.5) * res
    cy = y_max - (np.arange(height) + 0.5) * res
    gx, gy = np.meshgrid(cx, cy)
    grid_xy = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    print(f"Grid {height} x {width} at {res} m ({height * width} cells)", flush=True)

    print("Rasterising shoreline ...", flush=True)
    water_mask = _rings_to_masks(rings, grid_xy, (height, width)).astype(np.float32)

    print("Fetching bathymetry points ...", flush=True)
    pts_xy, pts_depth = fetch_depth_points()
    print(f"Interpolating {len(pts_depth)} depth points ...", flush=True)
    depth = griddata(pts_xy, pts_depth, grid_xy, method="linear")
    nearest = griddata(pts_xy, pts_depth, grid_xy, method="nearest")
    depth = np.where(np.isnan(depth), nearest, depth).reshape(height, width)
    depth = np.where(water_mask > 0, np.clip(depth, 0.0, None), np.nan).astype(np.float32)

    return {
        "water_mask": water_mask,
        "depth": depth,
        "x_min": np.float64(x_min),
        "y_max": np.float64(y_max),
        "res": np.float64(res),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", type=float, default=200.0, help="Cell size in metres.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "champlain_map.npz"),
    )
    args = parser.parse_args()

    raster = build_raster(args.res)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **raster)
    wm = raster["water_mask"]
    print(
        f"Saved {out}\n  water cells: {int(wm.sum())}\n"
        f"  depth range: {np.nanmin(raster['depth']):.1f}-{np.nanmax(raster['depth']):.1f} m"
    )


if __name__ == "__main__":
    main()
