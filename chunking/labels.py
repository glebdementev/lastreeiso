from pathlib import Path

import zlib
import numpy as np
import laspy


def relabel_final_segs_with_offset(processed_path: Path, start_label: int) -> int:
    """
    Relabel `final_segs` in-place so that all positive segment IDs are mapped
    into a contiguous block starting at start_label + 1.

    Returns the new global maximum label after relabeling.
    """
    las = laspy.read(str(processed_path))
    if "final_segs" not in las.point_format.extra_dimension_names:
        return start_label

    labels = np.asarray(las.final_segs, dtype=np.int64)
    if labels.size == 0:
        return start_label

    unique_labels = np.unique(labels)
    # Treat 0 as background / no-seg if present; keep it unchanged.
    seg_ids = [int(v) for v in unique_labels.tolist() if int(v) > 0]
    if not seg_ids:
        return start_label

    seg_ids_sorted = sorted(seg_ids)
    next_label = start_label + 1

    # Build mapping old_id -> new_id
    mapping: dict[int, int] = {}
    for sid in seg_ids_sorted:
        mapping[sid] = next_label
        next_label += 1

    relabeled = labels.copy()
    for old, new in mapping.items():
        relabeled[labels == old] = new

    las.final_segs = relabeled.astype(np.int32, copy=False)
    out_path = processed_path if processed_path.suffix.lower() == ".laz" else processed_path.with_suffix(".laz")
    las.write(str(out_path))

    # Return the new global maximum segment id
    return next_label - 1


def relabel_and_scramble_final_segs_contiguous_in_place(path: Path) -> None:
    """
    On the given LAS/LAZ file, relabel `final_segs` so that all positive IDs
    become a contiguous set {1, ..., N} and then scramble that ID assignment
    deterministically based on the filename.

    This guarantees:
      - No gaps in positive IDs (1..N).
      - A deterministic but shuffled mapping for reproducibility.
      - Label 0 (if present) is preserved as background.
    """
    las = laspy.read(str(path))
    if "final_segs" not in las.point_format.extra_dimension_names:
        return

    labels = np.asarray(las.final_segs, dtype=np.int64)
    if labels.size == 0:
        return

    unique_labels = np.unique(labels)
    seg_ids = [int(v) for v in unique_labels.tolist() if int(v) > 0]
    if not seg_ids:
        return

    seg_ids_sorted = sorted(seg_ids)
    n = len(seg_ids_sorted)

    # Deterministic RNG seeded by file name
    seed = zlib.crc32(path.name.encode("utf-8"))
    rng = np.random.default_rng(seed=seed)

    # Generate a permutation of 1..N
    permuted_ids = rng.permutation(n) + 1  # values in [1, N]

    mapping: dict[int, int] = {}
    for old_id, new_id in zip(seg_ids_sorted, permuted_ids.tolist()):
        mapping[old_id] = int(new_id)

    relabeled = labels.copy()
    for old, new in mapping.items():
        relabeled[labels == old] = new

    las.final_segs = relabeled.astype(np.int32, copy=False)
    out_path = path if path.suffix.lower() == ".laz" else path.with_suffix(".laz")
    las.write(str(out_path))


