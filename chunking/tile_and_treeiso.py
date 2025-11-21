from pathlib import Path
from typing import List, Union
import shutil

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
    border_cache_path,
    export_border_points,
)
from chunking.scramble import scramble_final_segs_in_place
from chunking.merge import merge_tiles_to_single
from chunking.postfilter import drop_low_segments_in_place
from chunking.refine_segments import refine_segments_with_treeiso
import numpy as np
import laspy

from chunking.preprocess import preprocess_las_file
from chunking.labels import (
    relabel_final_segs_with_offset,
    relabel_and_scramble_final_segs_contiguous_in_place,
)


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

    # Use a per-input working directory inside the configured output_dir to avoid
    # collisions when multiple files are processed in parallel or share the same
    # base name. Final merged *_iso.laz is still written directly into
    # config.output_dir, next to (or under) the desired location.
    work_dir = (config.output_dir / input_stem).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess input (SOR, decimate, dedup) before tiling
    preproc_out = preprocess_las_file(
        input_path_resolved,
        work_dir / f"{input_stem}_preproc.laz",
        sor_k=10,
        sor_std=1.0,
        decimation_res_m=0.02,
    )
    preproc_stem = Path(preproc_out).stem
    existing_tiles = sorted(work_dir.glob(f"{preproc_stem}_x*_y*.{ext}"))
    existing_processed = sorted(work_dir.glob(f"{preproc_stem}_x*_y*_treeiso.laz"))

    # If there are existing processed tiles from a previous run, remove them so we
    # can safely re-run the pipeline without manual cleanup.
    if len(existing_processed) > 0:
        for p in existing_processed:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        existing_processed = []

    # For tiling we write into the per-input working directory.
    if len(existing_tiles) > 0:
        tiles: List[Path] = existing_tiles
    else:
        tile_config = TilingConfig(
            output_dir=work_dir,
            tile_size=config.tile_size,
            output_format=config.output_format,
            call_treeiso=config.call_treeiso,
        )
        tiles = tile_file(preproc_out, tile_config)

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

    # Group coordinates by wavefront (sum index)
    coords_sorted = sorted(coords, key=wavefront_key)
    from collections import defaultdict
    waves: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for c in coords_sorted:
        s = c[0] + c[1]
        waves[s].append(c)
    if 0 in waves and (0, 0) in waves[0]:
        # ensure (0,0) first inside its wave
        w0 = waves[0]
        waves[0] = [(0, 0)] + [c for c in w0 if c != (0, 0)]

    processed_map: dict[tuple[int, int], Path] = {}
    # Map of ((xi, yi), direction) -> (cache_file_path, segment_ids)
    edge_cache_map: dict[tuple[tuple[int, int], str], tuple[Path, set[int]]] = {}
    # Global running maximum segment ID across all processed tiles
    current_max_seg_id: int = 0

    for wave_idx in sorted(waves.keys()):
        wave_coords = waves[wave_idx]
        # Build augmented inputs using cached border points from previous waves
        seg_inputs: list[Path] = []
        coords_for_inputs: list[tuple[int, int]] = []
        for coord in wave_coords:
            xi, yi = coord
            base_tile_path = coord_to_tile[coord]
            neighbor_point_files: list[tuple[Path, set[int]]] = []
            left = (xi - 1, yi)
            bottom = (xi, yi - 1)
            if left in processed_map:
                entry = edge_cache_map.get((left, DIR_POS_X))
                if entry is not None and len(entry[1]) > 0:
                    neighbor_point_files.append(entry)
            if bottom in processed_map:
                entry = edge_cache_map.get((bottom, DIR_POS_Y))
                if entry is not None and len(entry[1]) > 0:
                    neighbor_point_files.append(entry)
            if len(neighbor_point_files) > 0:
                aug_path = base_tile_path.with_name(f"{base_tile_path.stem}_aug{base_tile_path.suffix}")
                seg_input = augment_tile_with_border_segments(
                    base_tile_path,
                    neighbor_point_files,
                    aug_path,
                )
            else:
                seg_input = base_tile_path
            seg_inputs.append(seg_input)
            coords_for_inputs.append(coord)

        # Segment this entire wave in parallel
        if config.call_treeiso and len(seg_inputs) > 0:
            with ProcessPoolExecutor() as executor:
                list(executor.map(run_treeiso_on_tile, [str(p) for p in seg_inputs]))

        # After segmentation, relabel with global offset, detect and export border points,
        # then delete from tile and scramble.
        for coord, seg_input in zip(coords_for_inputs, seg_inputs):
            xi, yi = coord
            proc_path = processed_tile_path(seg_input)

            # Ensure this tile's positive segment IDs occupy a unique, non-overlapping
            # range starting from current_max_seg_id + 1.
            current_max_seg_id = relabel_final_segs_with_offset(proc_path, current_max_seg_id)

            borders = list_borders_for_tile(xi, yi, existing_coords)

            if DIR_POS_X in borders:
                segs_x = detect_border_segments(proc_path, DIR_POS_X, BORDER_THRESHOLD_UNITS)
                cache_x = border_cache_path(proc_path, DIR_POS_X)
                export_border_points(proc_path, segs_x, cache_x)
                edge_cache_map[(coord, DIR_POS_X)] = (cache_x, segs_x)
            if DIR_POS_Y in borders:
                segs_y = detect_border_segments(proc_path, DIR_POS_Y, BORDER_THRESHOLD_UNITS)
                cache_y = border_cache_path(proc_path, DIR_POS_Y)
                export_border_points(proc_path, segs_y, cache_y)
                edge_cache_map[(coord, DIR_POS_Y)] = (cache_y, segs_y)

            # Remove exported bordering segments from this processed tile
            segs_to_remove: set[int] = set()
            if (coord, DIR_POS_X) in edge_cache_map:
                segs_to_remove.update(edge_cache_map[(coord, DIR_POS_X)][1])
            if (coord, DIR_POS_Y) in edge_cache_map:
                segs_to_remove.update(edge_cache_map[(coord, DIR_POS_Y)][1])
            remove_segments_from_processed_in_place(proc_path, segs_to_remove)
            scramble_final_segs_in_place(proc_path)
            processed_map[coord] = proc_path

    # Final merge
    processed_list = [processed_map[c] for c in sorted(processed_map.keys(), key=wavefront_key)]
    merged_out = (config.output_dir / f"{input_stem}_iso.laz").resolve()
    merged_path = merge_tiles_to_single(processed_list, merged_out)

    # Refine existing segments by re-running treeiso on each segment separately,
    # until fewer than ~70% of refined segments produce a single segment as result.
    refine_segments_with_treeiso(merged_path, max_single_fraction=0.7)

    # Drop segments whose maximum Z is below a given threshold (in meters) from the
    # final merged point cloud. Uses default threshold of 5.0 m.
    drop_low_segments_in_place(merged_path, z_threshold=5.0)

    # Ensure final file has globally contiguous labels 1..N (positive IDs only) and
    # scramble those IDs for reproducibility.
    relabel_and_scramble_final_segs_contiguous_in_place(merged_path)

    # Cleanup per-input working directory (tiles, preproc, border caches) so that
    # only the final *_iso.laz remains in the target output directory.
    try:
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        # Best-effort cleanup; failures here should not break the pipeline.
        pass

    return tiles


def main() -> None:
    path_input = str(input('Please enter path to LAS/LAZ file: '))
    tile_and_process_file(path_input, DEFAULT_CONFIG)


if __name__ == "__main__":
    main()


