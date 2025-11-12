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
from chunking.merge import merge_tiles_to_single, processed_tile_path
from chunking.scramble import scramble_final_segs_in_place
import numpy as np
import laspy


def tile_and_process_file(input_path: Union[str, Path], config: TilingConfig = DEFAULT_CONFIG) -> List[Path]:
    input_path_resolved = Path(input_path).resolve()
    input_stem = input_path_resolved.stem
    merged_out = (config.output_dir / f"{input_stem}_merged.laz").resolve()

    # Refuse to run if merged output already exists
    if merged_out.exists():
        raise ValueError(f"Merged output already exists: {merged_out}")

    ext = config.output_format.value
    existing_tiles = sorted(config.output_dir.glob(f"{input_stem}_x*_y*.{ext}"))
    existing_processed = sorted(config.output_dir.glob(f"{input_stem}_x*_y*_treeiso.laz"))

    # Initialize tile list (do not create yet)
    tiles: List[Path] = existing_tiles if len(existing_tiles) > 0 else []
    processed: List[Path] = []
    just_processed: List[Path] = []

    if len(existing_processed) > 0:
        # Prefer existing segmented tiles; skip processing
        processed = existing_processed
    else:
        # No segmented tiles yet; ensure we have tiles, then process missing
        if len(tiles) == 0:
            tiles = tile_file(input_path_resolved, config)
        if config.call_treeiso and len(tiles) > 0:
            to_process = [p for p in tiles if not processed_tile_path(p).exists()]
            if len(to_process) > 0:
                with ProcessPoolExecutor() as executor:
                    list(executor.map(run_treeiso_on_tile, [str(p) for p in to_process]))
                just_processed = [processed_tile_path(p) for p in to_process if processed_tile_path(p).exists()]
            processed = [processed_tile_path(p) for p in tiles if processed_tile_path(p).exists()]

    # Scramble only the tiles produced in this run (avoid mutating pre-existing ones)
    for proc_path in just_processed:
        scramble_final_segs_in_place(proc_path)

    # Ensure global uniqueness of final_segs across tiles before merging
    if len(processed) > 0:
        ensure_global_unique_final_segs(processed)

    if len(processed) > 0:
        merge_tiles_to_single(processed, merged_out)
        # Scramble final_segs on the merged result as well
        scramble_final_segs_in_place(merged_out)
        # Remove per-tile .laz files so only the merged file remains
        for proc in processed:
            if proc.exists() and proc.suffix.lower() == ".laz":
                proc.unlink()
        for tile in tiles:
            if tile.exists() and tile.suffix.lower() == ".laz":
                tile.unlink()
    return tiles


def ensure_global_unique_final_segs(processed_paths: List[Path]) -> None:
    paths_sorted = sorted(processed_paths, key=lambda p: p.name)
    next_label: int = 1
    # Remap each tile's labels into a disjoint global range
    for tile_path in paths_sorted:
        las = laspy.read(str(tile_path))
        # Require presence of final_segs on processed tiles
        has_final = "final_segs" in las.point_format.extra_dimension_names
        assert has_final, f"Missing final_segs in processed tile: {tile_path}"
        labels = np.asarray(las.final_segs, dtype=np.int64)
        if labels.size == 0:
            # Nothing to do
            continue
        unique_vals = np.unique(labels)
        # Build mapping old -> new contiguous labels
        mapping: dict[int, int] = {}
        for val in unique_vals.tolist():
            mapping[int(val)] = next_label
            next_label = next_label + 1
        remapped = np.empty_like(labels)
        for old, new in mapping.items():
            remapped[labels == int(old)] = int(new)
        las.final_segs = remapped.astype(np.int32, copy=False)
        out_path = tile_path if tile_path.suffix.lower() == ".laz" else tile_path.with_suffix(".laz")
        las.write(str(out_path))
    # Validate disjointness across all tiles
    seen: set[int] = set()
    for tile_path in paths_sorted:
        las = laspy.read(str(tile_path))
        labels = np.asarray(las.final_segs, dtype=np.int64)
        uniques = np.unique(labels).tolist()
        for v in uniques:
            iv = int(v)
            if iv in seen:
                raise ValueError(f"final_segs id collision across tiles detected after remap: {iv} in {tile_path.name}")
            seen.add(iv)

def main() -> None:
    path_input = str(input('Please enter path to LAS/LAZ file: '))
    tile_and_process_file(path_input, DEFAULT_CONFIG)


if __name__ == "__main__":
    main()


