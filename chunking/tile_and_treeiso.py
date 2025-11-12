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


def tile_and_process_file(input_path: Union[str, Path], config: TilingConfig = DEFAULT_CONFIG) -> List[Path]:
    tiles = tile_file(input_path, config)
    if config.call_treeiso and len(tiles) > 0:
        with ProcessPoolExecutor() as executor:
            list(executor.map(run_treeiso_on_tile, [str(p) for p in tiles]))
    return tiles


def main() -> None:
    path_input = str(input('Please enter path to LAS/LAZ file: '))
    tile_and_process_file(path_input, DEFAULT_CONFIG)


if __name__ == "__main__":
    main()


