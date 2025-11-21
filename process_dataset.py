from pathlib import Path
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from chunking.config import TilingConfig, DEFAULT_CONFIG
from chunking.tile_and_treeiso import tile_and_process_file


def iter_las_files(root: Path) -> Iterable[Path]:
    """
    Recursively yield all LAS/LAZ files under the given root directory.
    """
    exts = (".las", ".laz")
    for ext in exts:
        yield from root.rglob(f"*{ext}")


def _process_single_file(las_path: Path) -> None:
    """
    Process a single LAS/LAZ file, writing outputs next to the input file.
    """
    name_lower = las_path.name.lower()
    # Skip already-segmented outputs to avoid re-processing
    if name_lower.endswith("_iso.laz"):
        return

    output_dir = las_path.parent.resolve()
    cfg = TilingConfig(
        output_dir=output_dir,
        tile_size=DEFAULT_CONFIG.tile_size,
        output_format=DEFAULT_CONFIG.output_format,
        call_treeiso=DEFAULT_CONFIG.call_treeiso,
    )

    print(f"Processing {las_path} ...", flush=True)
    tile_and_process_file(las_path, cfg)
    print(f"Done: {las_path}", flush=True)


def process_dataset(root: Path, max_workers: int | None = None) -> None:
    """
    Process all LAS/LAZ files under the given dataset root.

    For each input file, tiles/temporary files and the final segmented
    *_iso.laz file are written alongside the original file (i.e., in the
    same directory as the input).

    Files are processed in parallel using a thread pool. Each file still
    uses the internal process pool in tile_and_treeiso for per-tile work.
    """
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist or is not a directory: {root}")

    las_files = sorted(iter_las_files(root))
    if not las_files:
        print(f"No LAS/LAZ files found under {root}")
        return

    # Remove already-segmented outputs from the queue
    files_to_process = [p for p in las_files if not p.name.lower().endswith("_iso.laz")]
    if not files_to_process:
        print(f"All LAS/LAZ files under {root} appear to be already segmented.")
        return

    print(f"Found {len(files_to_process)} LAS/LAZ files to process under {root}")

    if max_workers is None:
        # Be conservative to avoid oversubscribing too many inner process pools.
        max_workers = min(4, (os.cpu_count() or 1))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(_process_single_file, p): p for p in files_to_process}
        for fut in as_completed(future_to_path):
            las_path = future_to_path[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"Error processing {las_path}: {e}", flush=True)


def main() -> None:
    # Default dataset path: sibling "dataset" directory next to this repo
    # (C:\Users\Gleb\Work\OpenForest\dev\dataset on your machine).
    default_dataset_root = (Path(__file__).resolve().parents[1] / "dataset").resolve()
    process_dataset(default_dataset_root, max_workers=1)


if __name__ == "__main__":
    main()


