from pathlib import Path
from typing import Tuple

import sys

import laspy
import numpy as np

# Ensure repository root is on path so we can import PythonCpp.treeiso
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PythonCpp.treeiso import process_point_cloud  # type: ignore


def _run_treeiso_on_xyz(xyz: np.ndarray) -> np.ndarray:
    """
    Run the core treeiso pipeline on a (N, 3) XYZ array and return per-point final labels.
    """
    if xyz.size == 0:
        return np.zeros(0, dtype=np.int64)

    _, _, final_labels, dec_inverse_idx, _ = process_point_cloud(xyz)
    # Map final labels back to the original (non-decimated) points
    per_point_labels = final_labels[dec_inverse_idx]
    return per_point_labels.astype(np.int64, copy=False)


def _segment_stats(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute basic stats for positive segment IDs in `labels`.

    Returns (seg_ids, counts, max_seg_id).
    """
    unique_ids, counts = np.unique(labels, return_counts=True)
    positive_mask = unique_ids > 0
    seg_ids = unique_ids[positive_mask].astype(np.int64, copy=False)
    seg_counts = counts[positive_mask].astype(np.int64, copy=False)
    max_seg_id = int(seg_ids.max()) if seg_ids.size > 0 else 0
    return seg_ids, seg_counts, max_seg_id


def refine_segments_with_treeiso(
    point_cloud_path: Path,
    max_single_fraction: float = 0.7,
    min_points_per_segment: int = 500,
) -> None:
    """
    Iteratively refine existing segments in-place by re-running treeiso per segment.

    Process:
      - Read `final_segs` from the given LAS/LAZ file.
      - Sort segments by point count (descending).
      - For each sufficiently large segment, run treeiso again on just the points
        of that segment.
      - If treeiso returns >1 sub-segment, replace that segment with new global
        segment IDs.
      - Repeat until fewer than `max_single_fraction` (default 0.7) of the
        attempted segments result in a single sub-segment, or `max_iters` is reached.

    This function updates only the `final_segs` dimension and writes back to the
    same file (always as .laz).
    """
    las = laspy.read(str(point_cloud_path))
    if "final_segs" not in las.point_format.extra_dimension_names:
        print(f"{point_cloud_path}: no 'final_segs' dimension found, skipping refinement.")
        return

    labels = np.asarray(las.final_segs, dtype=np.int64)
    if labels.size == 0:
        print(f"{point_cloud_path}: no points found, skipping refinement.")
        return

    x = np.asarray(las.x, dtype=float)
    y = np.asarray(las.y, dtype=float)
    z = np.asarray(las.z, dtype=float)

    seg_ids, seg_counts, max_seg_id = _segment_stats(labels)
    if seg_ids.size == 0:
        print(f"{point_cloud_path}: no positive segments to refine.")
        return

    # Work on a copy of labels and maintain a running max ID for new segments.
    current_labels = labels.copy()
    next_global_id = max_seg_id

    # Sort segments by point count (descending: focus on the largest segments first).
    order = np.argsort(-seg_counts)
    seg_ids_sorted = seg_ids[order]

    print(
        f"{point_cloud_path}: starting sequential per-segment refinement on "
        f"{seg_ids_sorted.size} segments (max_seg_id={max_seg_id})."
    )

    # Sliding window over the most recent results (True = single-label, False = split).
    window_size = 15
    recent_results: list[bool] = []
    attempted = 0
    single_result_count = 0
    split_count = 0

    for seg_id in seg_ids_sorted:
        # Extract indices for this segment
        idx = np.nonzero(current_labels == seg_id)[0]
        if idx.size < min_points_per_segment:
            continue

        xyz = np.stack((x[idx], y[idx], z[idx]), axis=1)
        try:
            sub_labels = _run_treeiso_on_xyz(xyz)
        except Exception as exc:
            print(f"{point_cloud_path}: refinement of segment {seg_id} failed with {exc!r}, skipping.")
            continue

        attempted += 1

        unique_sub = np.unique(sub_labels)
        if unique_sub.size <= 1:
            single_result_count += 1
            recent_results.append(True)
        else:
            # We obtained multiple sub-segments; assign new global IDs for each.
            split_count += 1
            sub_mapping = {}
            for sub_id in unique_sub:
                next_global_id += 1
                sub_mapping[int(sub_id)] = next_global_id

            mapped = np.empty_like(sub_labels, dtype=np.int64)
            for i, sid in enumerate(sub_labels):
                mapped[i] = sub_mapping[int(sid)]
            current_labels[idx] = mapped

            recent_results.append(False)

        # Maintain fixed-size sliding window of last `window_size` results.
        if len(recent_results) > window_size:
            recent_results.pop(0)

        if len(recent_results) == window_size:
            single_in_window = sum(1 for v in recent_results if v)
            # Once in the 10-window there are more than 7 single-label results, stop.
            if single_in_window > int(max_single_fraction * window_size):
                print(
                    f"{point_cloud_path}: stopping refinement after reaching a window of "
                    f"{window_size} segments with {single_in_window} single-label results."
                )
                break

    if attempted == 0:
        print(
            f"{point_cloud_path}: no segments with at least {min_points_per_segment} points "
            "were eligible for refinement; nothing changed."
        )
    else:
        overall_single_fraction = single_result_count / float(attempted)
        print(
            f"{point_cloud_path}: refinement summary - attempted={attempted}, "
            f"split={split_count}, single-result={single_result_count} "
            f"({overall_single_fraction:.2%})."
        )

    # Write back updated labels
    las.final_segs = current_labels.astype(np.int32, copy=False)
    out_path = (
        point_cloud_path
        if point_cloud_path.suffix.lower() == ".laz"
        else point_cloud_path.with_suffix(".laz")
    )
    las.write(str(out_path))

    print(f"{point_cloud_path}: finished per-segment refinement (max new ID={next_global_id}).")



