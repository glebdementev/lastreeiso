import os
import sys
from pathlib import Path
from typing import Union

from chunking.fallback import write_final_segs_ones

def run_treeiso_on_tile(tile_path: Union[str, Path]) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    old_stdout = sys.stdout
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    from PythonCpp.treeiso import process_las_file

    success = True
    try:
        process_las_file(str(tile_path))
    except Exception:
        success = False
    finally:
        sys.stdout = old_stdout
        devnull.close()
    if not success:
        write_final_segs_ones(tile_path)


