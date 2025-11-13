from pathlib import Path
from typing import List, Union

# Ensure repository root is on path so we can import PythonCpp.treeiso when run as a script
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from concurrent.futures import ProcessPoolExecutor
from chunking.config import TilingConfig, DEFAULT_CONFIG
from chunking.tiler import tile_file
from chunking.worker import run_treeiso_on_tile
from chunking.bordering_clusters_detector import (
    BORDER_THRESHOLD_UNITS,
    DIR_POS_X,
    DIR_POS_Y,
    parse_tile_indices_from_path,
    list_borders_for_tile,
    detect_border_segments,
    augment_tile_with_border_segments,
)
from chunking.scramble import scramble_final_segs_in_place
import numpy as np
import laspy


def processed_tile_path(tile_path: Path) -> Path:
    s = str(tile_path)
    return Path(s[:-4] + "_treeiso.laz")


def remove_segments_from_processed_in_place(processed_path: Path, seg_ids_to_remove: set[int]) -> None:
    if not seg_ids_to_remove:
        return
    las = laspy.read(str(processed_path))
    if "final_segs" not in las.point_format.extra_dimension_names:
        return
    labels = np.asarray(las.final_segs, dtype=np.int64)
    if labels.size == 0:
        return
    remove_mask = np.isin(labels, np.fromiter((int(s) for s in seg_ids_to_remove), dtype=np.int64))
    if not np.any(remove_mask):
        return
    keep_mask = ~remove_mask
    las.points = las.points[keep_mask]
    # Write back in-place; always as .laz
    out_path = processed_path if processed_path.suffix.lower() == ".laz" else processed_path.with_suffix(".laz")
    las.write(str(out_path))


def tile_and_process_file(input_path: Union[str, Path], config: TilingConfig = DEFAULT_CONFIG) -> List[Path]:
    input_path_resolved = Path(input_path).resolve()
    input_stem = input_path_resolved.stem
    ext = config.output_format.value
    existing_tiles = sorted(config.output_dir.glob(f"{input_stem}_x*_y*.{ext}"))
    existing_processed = sorted(config.output_dir.glob(f"{input_stem}_x*_y*_treeiso.laz"))

    # Avoid mixing prior results with a new pipeline run
    if len(existing_processed) > 0:
        raise ValueError("Found existing processed tiles; please move or remove them before running the new pipeline.")

    # Ensure we have tiles (no-overlap tiling enforced in tiler)
    tiles: List[Path] = existing_tiles if len(existing_tiles) > 0 else tile_file(input_path_resolved, config)

    if len(tiles) == 0:
        return tiles

    # Index tiles by their (xi, yi)
    coord_to_tile: dict[tuple[int, int], Path] = {}
    coords: List[tuple[int, int]] = []
    for t in tiles:
        xi, yi = parse_tile_indices_from_path(t)
        coord = (xi, yi)
        coord_to_tile[coord] = t
        coords.append(coord)

    existing_coords = set(coords)

    def wavefront_key(c: tuple[int, int]) -> tuple[int, int]:
        x, y = c
        return (x + y, x)

    # Start at (0,0) if available, otherwise minimal wavefront order
    coords_sorted = sorted(coords, key=wavefront_key)
    if (0, 0) in existing_coords:
        # Reorder to ensure (0,0) first while keeping wavefront grouping
        coords_sorted = [(0, 0)] + [c for c in coords_sorted if c != (0, 0)]

    processed_map: dict[tuple[int, int], Path] = {}
    border_sets: dict[tuple[tuple[int, int], str], set[int]] = {}

    for coord in coords_sorted:
        xi, yi = coord
        base_tile_path = coord_to_tile[coord]

        # Augment from left (pos_x of left neighbor) and bottom (pos_y of bottom neighbor), if present
        neighbor_inputs: list[tuple[Path, set[int]]] = []
        left = (xi - 1, yi)
        bottom = (xi, yi - 1)
        if left in processed_map:
            segs = border_sets.get((left, DIR_POS_X), set())
            if len(segs) > 0:
                neighbor_inputs.append((processed_map[left], segs))
        if bottom in processed_map:
            segs = border_sets.get((bottom, DIR_POS_Y), set())
            if len(segs) > 0:
                neighbor_inputs.append((processed_map[bottom], segs))

        if len(neighbor_inputs) > 0:
            aug_path = base_tile_path.with_name(f"{base_tile_path.stem}_aug{base_tile_path.suffix}")
            seg_input = augment_tile_with_border_segments(base_tile_path, neighbor_inputs, aug_path)
        else:
            seg_input = base_tile_path

        # Segment this tile
        if config.call_treeiso:
            run_treeiso_on_tile(str(seg_input))
        proc_path = processed_tile_path(seg_input)

        # Detect bordering segments for neighbors in +X and +Y directions
        borders = list_borders_for_tile(xi, yi, existing_coords)
        segs_to_remove: set[int] = set()
        if DIR_POS_X in borders:
            segs_x = detect_border_segments(proc_path, DIR_POS_X, BORDER_THRESHOLD_UNITS)
            border_sets[(coord, DIR_POS_X)] = segs_x
            segs_to_remove.update(segs_x)
        if DIR_POS_Y in borders:
            segs_y = detect_border_segments(proc_path, DIR_POS_Y, BORDER_THRESHOLD_UNITS)
            border_sets[(coord, DIR_POS_Y)] = segs_y
            segs_to_remove.update(segs_y)

        # Remove those bordering segments from this processed tile
        remove_segments_from_processed_in_place(proc_path, segs_to_remove)
        scramble_final_segs_in_place(proc_path)

        processed_map[coord] = proc_path

    return tiles


def main() -> None:
    path_input = str(input('Please enter path to LAS/LAZ file: '))
    tile_and_process_file(path_input, DEFAULT_CONFIG)


if __name__ == "__main__":
    main()


