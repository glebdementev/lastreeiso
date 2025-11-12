from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import laspy


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

    coord_to_global_seg: Dict[Tuple[int, int, int], int] = {}
    tile_seg_to_global: Dict[Tuple[int, int], int] = {}
    next_global_id = 1

    las_list: List[laspy.LasData] = []
    header0 = laspy.read(str(tiles_sorted[0])).header.copy()

    for tile_index, tile_path in enumerate(tiles_sorted):
        las = laspy.read(str(tile_path))

        X = las.X
        Y = las.Y
        Z = las.Z
        local_segs = np.asarray(las.final_segs, dtype=np.int64)

        num_points = int(len(X))
        keep_mask = np.ones(num_points, dtype=bool)

        for i in range(num_points):
            coord = (int(X[i]), int(Y[i]), int(Z[i]))
            local_seg = int(local_segs[i])
            key_tile_seg = (tile_index, local_seg)

            if coord in coord_to_global_seg:
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

        remapped = np.empty(num_points, dtype=np.int64)
        for i in range(num_points):
            local_seg = int(local_segs[i])
            key_tile_seg = (tile_index, local_seg)
            remapped[i] = tile_seg_to_global[key_tile_seg]

        if not np.all(keep_mask):
            las.points = las.points[keep_mask]
            remapped = remapped[keep_mask]

        if "final_segs" not in las.point_format.extra_dimension_names:
            las.add_extra_dim(laspy.ExtraBytesParams(name="final_segs", type="int32", description="final_segs"))
        las.final_segs = remapped.astype(np.int32, copy=False)

        las_list.append(las)

    output_path = output_path.with_suffix(".laz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with laspy.open(str(output_path), mode="w", header=header0) as writer:
        for las in las_list:
            writer.write_points(las.points)

    return output_path
