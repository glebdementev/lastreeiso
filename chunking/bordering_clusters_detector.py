from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import re
import numpy as np
import laspy

# Global threshold (in coordinate units; typically same units as las.x / las.y)
BORDER_THRESHOLD_UNITS: float = 1.0

# Directions along which we propagate bordering segments
DIR_POS_X = "pos_x"  # right neighbor (increasing X)
DIR_POS_Y = "pos_y"  # top neighbor (increasing Y)

_TILE_RE = re.compile(r".*_x(?P<xi>\d+)_y(?P<yi>\d+)\.[^.]+$", re.IGNORECASE)


def parse_tile_indices_from_path(tile_path: Path) -> Tuple[int, int]:
    m = _TILE_RE.match(tile_path.name)
    if not m:
        raise ValueError(f"Cannot parse tile indices from name: {tile_path.name}")
    return int(m.group("xi")), int(m.group("yi"))


def get_tile_bounds(las: laspy.LasData) -> Tuple[float, float, float, float]:
    min_x = float(las.header.mins[0])
    max_x = float(las.header.maxs[0])
    min_y = float(las.header.mins[1])
    max_y = float(las.header.maxs[1])
    return min_x, max_x, min_y, max_y


def list_borders_for_tile(
    xi: int,
    yi: int,
    existing_coords: Set[Tuple[int, int]],
) -> Dict[str, Tuple[int, int]]:
    borders: Dict[str, Tuple[int, int]] = {}
    # Right neighbor
    if (xi + 1, yi) in existing_coords:
        borders[DIR_POS_X] = (xi + 1, yi)
    # Top neighbor
    if (xi, yi + 1) in existing_coords:
        borders[DIR_POS_Y] = (xi, yi + 1)
    return borders


def _segments_touching_axis_max(
    las: laspy.LasData,
    axis: str,
    threshold: float,
) -> Set[int]:
    if "final_segs" not in las.point_format.extra_dimension_names:
        raise ValueError("Expected 'final_segs' in processed LAS/LAZ for border detection")
    labels = np.asarray(las.final_segs, dtype=np.int64)
    if labels.size == 0:
        return set()
    xs = np.asarray(las.x, dtype=np.float64)
    ys = np.asarray(las.y, dtype=np.float64)
    _, max_x, _, max_y = get_tile_bounds(las)
    axis_vals = xs if axis == "x" else ys
    boundary_max = max_x if axis == "x" else max_y
    seg_ids = np.unique(labels).tolist()
    seg_to_max: Dict[int, float] = {}
    for sid in seg_ids:
        mask = labels == int(sid)
        if not np.any(mask):
            continue
        seg_to_max[int(sid)] = float(np.max(axis_vals[mask]))
    # Sort by max descending (not used for selection, but available for debugging or future tie-breaks)
    sorted_sids = sorted(seg_to_max.keys(), key=lambda k: seg_to_max[k], reverse=True)
    # Select within threshold of boundary maximum
    selected: Set[int] = set()
    for sid in sorted_sids:
        if (boundary_max - seg_to_max[sid]) <= float(threshold):
            selected.add(int(sid))
    return selected


def detect_border_segments(
    processed_tile_path: Path,
    direction: str,
    threshold: float = BORDER_THRESHOLD_UNITS,
) -> Set[int]:
    las = laspy.read(str(processed_tile_path))
    if direction == DIR_POS_X:
        return _segments_touching_axis_max(las, axis="x", threshold=threshold)
    if direction == DIR_POS_Y:
        return _segments_touching_axis_max(las, axis="y", threshold=threshold)
    raise ValueError(f"Unsupported direction: {direction}")


def border_cache_path(processed_tile_path: Path, direction: str) -> Path:
    stem = processed_tile_path.stem
    return processed_tile_path.with_name(f"{stem}_{direction}_border.laz")


def export_border_points(
    processed_tile_path: Path,
    segment_ids: Set[int],
    output_path: Path,
) -> Path:
    """Write only points belonging to given segment_ids into output_path."""
    if not segment_ids:
        return output_path
    las = laspy.read(str(processed_tile_path))
    if "final_segs" not in las.point_format.extra_dimension_names:
        return output_path
    labels = np.asarray(las.final_segs, dtype=np.int64)
    if labels.size == 0:
        return output_path
    mask = np.isin(labels, np.fromiter((int(s) for s in segment_ids), dtype=np.int64))
    if not np.any(mask):
        return output_path
    out_header = las.header.copy()
    out = laspy.LasData(out_header)
    out.points = las.points[mask]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(output_path))
    return output_path


def _align_points_dtype(points: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """Create a view/copy of points with only fields present in target dtype."""
    out = np.empty(points.shape, dtype=target_dtype)
    for name in target_dtype.names or []:
        if name not in points.dtype.names:
            # If a required field is missing, initialize to zeros
            out[name] = 0
        else:
            out[name] = points[name]
    return out


def augment_tile_with_border_segments(
    base_tile_path: Path,
    neighbor_processed_and_segments: Iterable[Tuple[Path, Set[int]]],
    output_aug_path: Path,
) -> Path:
    """Create an augmented tile combining the base tile with bordering neighbor segments' points.

    - base_tile_path: original tile written by tiler (no 'final_segs')
    - neighbor_processed_and_segments: iterable of (processed_path, segment_id_set)
    - output_aug_path: where to write the augmented tile
    """
    base_las = laspy.read(str(base_tile_path))
    base_header = base_las.header.copy()
    base_points = base_las.points.array
    base_dtype = base_points.dtype

    extra_chunks: List[np.ndarray] = []
    for proc_path, seg_ids in neighbor_processed_and_segments:
        if not seg_ids:
            continue
        nlas = laspy.read(str(proc_path))
        if "final_segs" not in nlas.point_format.extra_dimension_names:
            continue
        labels = np.asarray(nlas.final_segs, dtype=np.int64)
        if labels.size == 0:
            continue
        # Build mask for any of the selected segment ids
        mask = np.isin(labels, np.fromiter((int(s) for s in seg_ids), dtype=np.int64))
        if not np.any(mask):
            continue
        npoints = nlas.points.array[mask]
        # Align neighbor points to base dtype (drop 'final_segs' and any other extras)
        npoints_aligned = _align_points_dtype(npoints, base_dtype)
        extra_chunks.append(npoints_aligned)

    if len(extra_chunks) == 0:
        # No augmentation necessary; copy base as-is
        output_aug_path.parent.mkdir(parents=True, exist_ok=True)
        base_las.write(str(output_aug_path))
        return output_aug_path

    # Concatenate as numpy structured array first
    concat_points = np.concatenate([base_points] + extra_chunks, axis=0)
    # Build a proper PointRecord matching the base header/point_format,
    # then fill it field-by-field from the structured array
    total_points = int(concat_points.shape[0])
    pr = laspy.ScaleAwarePointRecord.zeros(total_points, header=base_header)
    pr_array = pr.array
    for name in (base_dtype.names or ()):
        pr_array[name] = concat_points[name]

    out_las = laspy.LasData(base_header)
    out_las.points = pr
    output_aug_path.parent.mkdir(parents=True, exist_ok=True)
    out_las.write(str(output_aug_path))
    return output_aug_path


