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


def tile_and_process_file(input_path: Union[str, Path], config: TilingConfig = DEFAULT_CONFIG) -> List[Path]:
    tiles = tile_file(input_path, config)
    if config.call_treeiso and len(tiles) > 0:
        with ProcessPoolExecutor() as executor:
            list(executor.map(run_treeiso_on_tile, [str(p) for p in tiles]))
        processed = []
        for p in tiles:
            out_p = processed_tile_path(p)
            if out_p.exists():
                processed.append(out_p)
        # Scramble final_segs per processed tile before merging
        for proc_path in processed:
            scramble_final_segs_in_place(proc_path)
        if len(processed) > 0:
            input_stem = Path(input_path).resolve().stem
            merged_out = (config.output_dir / f"{input_stem}_merged.laz").resolve()
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


def main() -> None:
    path_input = str(input('Please enter path to LAS/LAZ file: '))
    tile_and_process_file(path_input, DEFAULT_CONFIG)


if __name__ == "__main__":
    main()


