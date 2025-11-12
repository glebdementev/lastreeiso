from pathlib import Path

import zlib
import numpy as np
import laspy


def scramble_final_segs_in_place(tile_path: Path) -> None:
    las = laspy.read(str(tile_path))
    if "final_segs" not in las.point_format.extra_dimension_names:
        return
    labels = np.asarray(las.final_segs, dtype=np.int64)
    if labels.size == 0:
        return
    unique_vals, first_idx = np.unique(labels, return_index=True)
    order = np.argsort(first_idx)
    ordered_labels = unique_vals[order]
    seed = zlib.crc32(tile_path.name.encode("utf-8"))
    rng = np.random.default_rng(seed=seed)
    perm = rng.permutation(len(ordered_labels))
    shuffled_labels = ordered_labels[perm]
    remapped = np.empty_like(labels)
    for orig, new in zip(ordered_labels.tolist(), shuffled_labels.tolist()):
        mask = labels == orig
        remapped[mask] = new
    las.final_segs = remapped.astype(np.int32, copy=False)
    out_path = tile_path if tile_path.suffix.lower() == ".laz" else tile_path.with_suffix(".laz")
    las.write(str(out_path))


