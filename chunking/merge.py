from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import laspy
from collections import defaultdict


def processed_tile_path(tile_path: Path) -> Path:
    s = str(tile_path)
    # treeiso consistently writes *_treeiso.laz
    return Path(s[:-4] + "_treeiso.laz")


def merge_tiles(processed_tiles: List[Path]) -> None:
    # Deterministic order defines precedence: earlier tiles win for duplicate coordinates
    tiles_sorted = sorted(processed_tiles, key=lambda p: p.name)

    coord_to_global_seg: Dict[Tuple[int, int, int], int] = {}
    tile_seg_to_global: Dict[Tuple[int, int], int] = {}
    next_global_id = 1

    for tile_index, tile_path in enumerate(tiles_sorted):
        las = laspy.read(str(tile_path))

        # Integer coordinates are stable for exact-duplicate detection
        X = las.X
        Y = las.Y
        Z = las.Z
        local_segs = np.asarray(las.final_segs, dtype=np.int64)

        num_points = int(len(X))
        keep_mask = np.ones(num_points, dtype=bool)

        # First pass: establish mappings and mark duplicates to drop
        for i in range(num_points):
            coord = (int(X[i]), int(Y[i]), int(Z[i]))
            local_seg = int(local_segs[i])
            key_tile_seg = (tile_index, local_seg)

            if coord in coord_to_global_seg:
                # Duplicate point: unify this tile's local segment to existing global id
                keep_mask[i] = False
                existing_gid = coord_to_global_seg[coord]
                tile_seg_to_global[key_tile_seg] = existing_gid
            else:
                if key_tile_seg in tile_seg_to_global:
                    gid = tile_seg_to_global[key_tile_seg]
                else:
                    gid = next_global_id
                    tile_seg_to_global[key_tile_seg] = gid
                    next_global_id = next_global_id + 1
                coord_to_global_seg[coord] = gid

        # Second pass: build remapped labels
        remapped = np.empty(num_points, dtype=np.int64)
        for i in range(num_points):
            local_seg = int(local_segs[i])
            key_tile_seg = (tile_index, local_seg)
            remapped[i] = tile_seg_to_global[key_tile_seg]

        # Apply deduplication and relabel
        if not np.all(keep_mask):
            las.points = las.points[keep_mask]
            remapped = remapped[keep_mask]

        # Ensure extra dim exists and assign
        if "final_segs" not in las.point_format.extra_dimension_names:
            las.add_extra_dim(laspy.ExtraBytesParams(name="final_segs", type="int32", description="final_segs"))
        las.final_segs = remapped.astype(np.int32, copy=False)

        # Always save as .laz
        out_path = tile_path if tile_path.suffix.lower() == ".laz" else tile_path.with_suffix(".laz")
        las.write(str(out_path))



def merge_tiles_to_single(processed_tiles: List[Path], output_path: Path) -> Path:
    tiles_sorted = sorted(processed_tiles, key=lambda p: p.name)
    assert len(tiles_sorted) > 0, "No tiles to merge"

    # Read all tiles; concatenate points and core arrays
    las_list: List[laspy.LasData] = []
    header0 = laspy.read(str(tiles_sorted[0])).header.copy()
    all_points_list: List[np.ndarray] = []
    all_X: List[np.ndarray] = []
    all_Y: List[np.ndarray] = []
    all_Z: List[np.ndarray] = []
    all_segs: List[np.ndarray] = []

    for tile_path in tiles_sorted:
        las = laspy.read(str(tile_path))
        # Ensure final_segs dim exists in data for downstream use
        if "final_segs" not in las.point_format.extra_dimension_names:
            las.add_extra_dim(laspy.ExtraBytesParams(name="final_segs", type="int32", description="final_segs"))
            las.final_segs = np.zeros(len(las.points), dtype=np.int32)
        las_list.append(las)
        all_points_list.append(las.points.array)  # structured numpy structured array
        all_X.append(np.asarray(las.X, dtype=np.int64))
        all_Y.append(np.asarray(las.Y, dtype=np.int64))
        all_Z.append(np.asarray(las.Z, dtype=np.int64))
        all_segs.append(np.asarray(las.final_segs, dtype=np.int64))

    all_points = np.concatenate(all_points_list, axis=0) if len(all_points_list) > 1 else all_points_list[0]
    X = np.concatenate(all_X, axis=0) if len(all_X) > 1 else all_X[0]
    Y = np.concatenate(all_Y, axis=0) if len(all_Y) > 1 else all_Y[0]
    Z = np.concatenate(all_Z, axis=0) if len(all_Z) > 1 else all_Z[0]
    final_segs = np.concatenate(all_segs, axis=0) if len(all_segs) > 1 else all_segs[0]

    num_points_total = int(X.shape[0])
    assert num_points_total == int(Y.shape[0]) == int(Z.shape[0]) == int(final_segs.shape[0]), "Mismatched array sizes"

    # Step 2: find all points which have exactly the same coordinates
    coords = np.stack((X, Y, Z), axis=1)  # shape (N, 3), int64
    uniques, inv_idx, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
    # Duplicate groups: indices where counts[group] > 1
    dup_group_mask = counts > 1
    if np.any(dup_group_mask):
        pass  # duplicates exist; expected for cross-tile overlaps

    # Build index arrays grouped by group id (for groups with duplicates only)
    order = np.argsort(inv_idx)
    inv_sorted = inv_idx[order]
    # boundaries where group id changes
    boundaries = np.flatnonzero(np.diff(inv_sorted)) + 1
    group_slices: List[Tuple[int, int, int]] = []  # (group_id, start, end)
    start = 0
    for b in boundaries.tolist():
        gid = int(inv_sorted[start])
        if dup_group_mask[gid]:
            group_slices.append((gid, start, b))
        start = b
    # last slice
    if order.size > 0:
        gid_last = int(inv_sorted[start])
        if dup_group_mask[gid_last]:
            group_slices.append((gid_last, start, order.size))

    # Step 3: construct pairwise matrix on segments appearing in duplicate coords
    # - diagonal must be zero: no two points with same seg share coords
    # - matrix must be symmetric; we will count both directions and verify
    pair_counts_dir: Dict[Tuple[int, int], int] = {}
    diag_violations: Dict[int, int] = {}

    for gid, s, e in group_slices:
        inds = order[s:e]
        segs = final_segs[inds]
        unique_segs, seg_counts = np.unique(segs, return_counts=True)
        # Check diagonal violation: same seg appears more than once at same coords
        for u_seg, c in zip(unique_segs.tolist(), seg_counts.tolist()):
            if int(c) > 1:
                existing = diag_violations[u_seg] if u_seg in diag_violations else 0
                diag_violations[u_seg] = existing + (c - 1)
        # For pairwise counts, count each unordered pair once per coordinate group
        # Here, if multiple segs are present, increment each pair by 1 (not product)
        seg_list = unique_segs.tolist()
        n_unique = len(seg_list)
        if n_unique > 1:
            for i in range(n_unique - 1):
                a = int(seg_list[i])
                for j in range(i + 1, n_unique):
                    b = int(seg_list[j])
                    # count both directions for explicit symmetry check
                    key_ab = (a, b)
                    key_ba = (b, a)
                    val_ab = pair_counts_dir[key_ab] if key_ab in pair_counts_dir else 0
                    pair_counts_dir[key_ab] = val_ab + 1
                    val_ba = pair_counts_dir[key_ba] if key_ba in pair_counts_dir else 0
                    pair_counts_dir[key_ba] = val_ba + 1

    # Diagonal must be zero
    if len(diag_violations) > 0:
        details = ", ".join([f"segment {k}: {v} duplicate-coord intra-segment points" for k, v in sorted(diag_violations.items())])
        raise ValueError(f"Matrix diagonal should be all zeros; found intra-segment duplicates: {details}")

    # Symmetry check: ensure count(a,b) == count(b,a); we will then keep only upper triangle
    asym: List[Tuple[int, int]] = []
    for (a, b), v in pair_counts_dir.items():
        val_ba = pair_counts_dir[(b, a)] if (b, a) in pair_counts_dir else 0
        if v != val_ba:
            asym.append((a, b))
    if len(asym) > 0:
        pairs_str = ", ".join([f"({a},{b})" for a, b in asym[:10]])
        raise ValueError(f"Pairwise matrix is not symmetric for pairs: {pairs_str}")

    # Drop lower triangle, keep (min,max) only
    pair_counts: Dict[Tuple[int, int], int] = {}
    for (a, b), v in pair_counts_dir.items():
        if a == b:
            continue
        lo = a if a < b else b
        hi = b if b > a else a
        key = (lo, hi)
        prev = pair_counts[key] if key in pair_counts else 0
        # Since we counted both directions, divide by 2 to get true count
        pair_counts[key] = prev if prev > 0 else (v // 2)

    # Step 4: print top-10 matrix cells by point count
    sorted_pairs = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)
    for idx, (key, cnt) in enumerate(sorted_pairs[:10]):
        a, b = key
        print(f"{cnt} points, segment {a} and {b}")

    # Step 5/6: per-segment totals from matrix; remember intersecting ids
    seg_total: Dict[int, int] = {}
    seg_neighbors: Dict[int, Dict[int, int]] = {}
    for (a, b), cnt in pair_counts.items():
        val_a = seg_total[a] if a in seg_total else 0
        seg_total[a] = val_a + cnt
        val_b = seg_total[b] if b in seg_total else 0
        seg_total[b] = val_b + cnt
        if a not in seg_neighbors:
            seg_neighbors[a] = {}
        if b not in seg_neighbors:
            seg_neighbors[b] = {}
        prev_ab = seg_neighbors[a][b] if b in seg_neighbors[a] else 0
        seg_neighbors[a][b] = prev_ab + cnt
        prev_ba = seg_neighbors[b][a] if a in seg_neighbors[b] else 0
        seg_neighbors[b][a] = prev_ba + cnt

    top_seg_totals = sorted(seg_total.items(), key=lambda kv: kv[1], reverse=True)
    for seg_id, total in top_seg_totals[:10]:
        neighbors = seg_neighbors[seg_id] if seg_id in seg_neighbors else {}
        neighbors_sorted = sorted(neighbors.items(), key=lambda kv: kv[1], reverse=True)
        details = ", ".join([f"{cnt} with segment {nbr}" for nbr, cnt in neighbors_sorted])
        print(f"final_seg {seg_id} has a total of {total} common points: {details}")

    # Step 7: merge labels - descending by total; map intersecting segments to the max-total representative
    rep_map: Dict[int, int] = {}
    for seg_id, _ in top_seg_totals:
        # If this seg already assigned, use its representative, else itself
        rep = rep_map[seg_id] if seg_id in rep_map else seg_id
        rep_map[seg_id] = rep
        neighbors = seg_neighbors[seg_id] if seg_id in seg_neighbors else {}
        for nbr, cnt in neighbors.items():
            if cnt <= 0:
                continue
            if nbr in rep_map:
                continue
            rep_map[nbr] = rep

    # Apply mapping to all labels; segments not present in rep_map remain as-is
    new_labels = np.empty_like(final_segs, dtype=np.int64)
    # Build map array-wise: we will vectorize via loop over unique labels to avoid slow Python per-point
    unique_labels = np.unique(final_segs)
    for ul in unique_labels.tolist():
        mapped = rep_map[ul] if ul in rep_map else ul
        new_labels[final_segs == ul] = int(mapped)

    # Step 8: deduplicate points by identical coordinates; keep first occurrence overall; count deletions per segment
    # Reuse inv_idx/counts on coords (same as before)
    # Keep mask retains the first occurrence of each unique coordinate
    first_occurrence_idx = np.zeros(uniques.shape[0], dtype=np.int64)
    # The first index for each group can be found by taking the first index where inv_idx == gid
    # We can compute using the sorted order and group_slices computed earlier
    # Build a map: group_id -> global first index
    for gid, s, e in group_slices:
        first_occurrence_idx[gid] = int(order[s])
    # For non-duplicate groups (count==1), the unique point is the position where inv_idx == gid; we can compute via argmax on a mask
    # Faster: build an array that marks the first occurrence for every gid using a seen mask
    seen = np.zeros(uniques.shape[0], dtype=bool)
    keep_mask = np.zeros(num_points_total, dtype=bool)
    for idx_pt in range(num_points_total):
        g = int(inv_idx[idx_pt])
        if not seen[g]:
            keep_mask[idx_pt] = True
            seen[g] = True

    # Count deletions per segment (after merge labels)
    deleted_counts: Dict[int, int] = {}
    for idx_pt in range(num_points_total):
        if not keep_mask[idx_pt]:
            seg_id = int(new_labels[idx_pt])
            prev = deleted_counts[seg_id] if seg_id in deleted_counts else 0
            deleted_counts[seg_id] = prev + 1
    # Print deletions per segment that have deletions
    for seg_id, cnt in sorted(deleted_counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"Deleted {cnt} duplicate points for segment {seg_id}")

    # Filter points and labels to kept set
    points_filtered = all_points[keep_mask]
    labels_filtered = new_labels[keep_mask]

    # Step 9: reindex labels to contiguous 1..N
    uniq_after = np.unique(labels_filtered)
    uniq_sorted = uniq_after.tolist()
    uniq_sorted.sort()
    reindex_map: Dict[int, int] = {}
    next_id = 1
    for old in uniq_sorted:
        reindex_map[int(old)] = next_id
        next_id = next_id + 1
    labels_reindexed = np.empty_like(labels_filtered, dtype=np.int64)
    for old in uniq_sorted:
        labels_reindexed[labels_filtered == int(old)] = int(reindex_map[int(old)])

    # Ensure header has final_segs extra dim
    if "final_segs" not in header0.point_format.extra_dimension_names:
        header0.add_extra_dims([laspy.ExtraBytesParams(name="final_segs", type="int32", description="final_segs")])

    # Update final_segs in the structured array before writing
    if "final_segs" in points_filtered.dtype.names:
        points_filtered["final_segs"] = labels_reindexed.astype(np.int32, copy=False)
    else:
        raise ValueError("final_segs field missing from points array after processing")

    output_path = output_path.with_suffix(".laz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with laspy.open(str(output_path), mode="w", header=header0) as writer:
        # Build a ScaleAwarePointRecord from the structured numpy array for safe writing
        point_record = laspy.ScaleAwarePointRecord(points_filtered, header0.point_format, header0.scales, header0.offsets)
        writer.write_points(point_record)

    return output_path