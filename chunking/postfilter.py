from pathlib import Path

import laspy
import numpy as np


def drop_low_segments_in_place(point_cloud_path: Path, z_threshold: float = 5.0) -> None:
    """
    Remove all points belonging to segments whose maximum Z is below the given threshold.

    Operates in-place on the provided LAS/LAZ file (always writing back as .laz).
    Prints the number of dropped segments and remaining segments.
    """
    las = laspy.read(str(point_cloud_path))

    if "final_segs" not in las.point_format.extra_dimension_names:
        print(f"{point_cloud_path}: no 'final_segs' dimension found, skipping low-segment filtering.")
        return

    labels = np.asarray(las.final_segs, dtype=np.int64)
    if labels.size == 0:
        print(f"{point_cloud_path}: no points found, nothing to filter.")
        return

    z = np.asarray(las.z, dtype=float)

    # Group by segment ID and compute max Z per segment efficiently.
    order = np.argsort(labels, kind="mergesort")
    labels_sorted = labels[order]
    z_sorted = z[order]

    unique_labels, idx_start = np.unique(labels_sorted, return_index=True)
    idx_end = np.empty_like(idx_start)
    idx_end[:-1] = idx_start[1:]
    idx_end[-1] = labels_sorted.size

    max_z_per_label = np.empty_like(unique_labels, dtype=float)
    for i, (start, end) in enumerate(zip(idx_start, idx_end)):
        max_z_per_label[i] = float(z_sorted[start:end].max())

    positive_mask = unique_labels > 0
    total_segments = int(positive_mask.sum())

    drop_label_mask = (max_z_per_label < z_threshold) & positive_mask
    labels_to_drop = unique_labels[drop_label_mask]

    num_dropped_segments = int(labels_to_drop.size)
    if num_dropped_segments == 0:
        print(
            f"{point_cloud_path}: all {total_segments} segments have max Z >= {z_threshold} m; "
            "no segments removed."
        )
        return

    labels_to_drop_set = set(int(v) for v in labels_to_drop.tolist())

    total_points_before = int(labels.size)
    keep_mask = ~np.isin(labels, np.fromiter(labels_to_drop_set, dtype=np.int64))
    total_points_after = int(keep_mask.sum())

    las.points = las.points[keep_mask]

    out_path = (
        point_cloud_path
        if point_cloud_path.suffix.lower() == ".laz"
        else point_cloud_path.with_suffix(".laz")
    )
    las.write(str(out_path))

    remaining_segments = total_segments - num_dropped_segments

    print(
        f"{point_cloud_path}: dropped {num_dropped_segments} segments with max Z < {z_threshold} m; "
        f"{remaining_segments} segments remain."
    )
    print(
        f"{point_cloud_path}: points before = {total_points_before}, "
        f"after = {total_points_after}."
    )


