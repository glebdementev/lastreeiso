from pathlib import Path
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from chunking.config import TilingConfig, DEFAULT_CONFIG
from chunking.tile_and_treeiso import tile_and_process_file


def iter_las_files(root: Path) -> Iterable[Path]:
    """
    Recursively yield all LAS/LAZ files under the given root directory.

    Uses suffix-based filtering so that both lower- and upper-case extensions
    (.las, .LAS, .laz, .LAZ) are detected.
    """
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".las", ".laz"):
            yield p


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

    # Select only those LAS/LAZ files that:
    #  1) Are not *_iso.laz themselves
    #  2) Have at least one .geojson file in the same folder
    #  3) Do NOT yet have a corresponding *_iso.laz file next to them
    files_to_process = []
    for p in las_files:
        name_lower = p.name.lower()
        if name_lower.endswith("_iso.laz"):
            continue

        # Condition 2: there must be at least one .geojson file in the same folder
        parent = p.parent
        has_geojson = any(
            child.is_file() and child.suffix.lower() == ".geojson"
            for child in parent.iterdir()
        )
        if not has_geojson:
            continue

        # Condition 3: skip if the expected *_iso.laz already exists
        expected_iso = p.with_name(f"{p.stem}_iso.laz")
        if expected_iso.exists():
            continue

        files_to_process.append(p)

    if not files_to_process:
        print(
            f"No LAS/LAZ files under {root} meet all conditions: "
            f'has sibling .geojson and no corresponding "*_iso.laz" yet.'
        )
        return

    print(f"Found {len(files_to_process)} LAS/LAZ files to process under {root}:")
    for p in files_to_process:
        print(f"  {p}")

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

    # After processing, report any inputs that still do not have a corresponding
    # *_iso.laz output next to them, *for those that satisfy the same preconditions*
    # (have a sibling .geojson). This helps identify files that failed.
    missing_outputs = []
    for p in las_files:
        name_lower = p.name.lower()
        if name_lower.endswith("_iso.laz"):
            continue

        parent = p.parent
        has_geojson = any(
            child.is_file() and child.suffix.lower() == ".geojson"
            for child in parent.iterdir()
        )
        if not has_geojson:
            continue

        expected_iso = p.with_name(f"{p.stem}_iso.laz")
        if not expected_iso.exists():
            missing_outputs.append(p)

    if missing_outputs:
        print("\nWARNING: the following LAS/LAZ files do NOT have a corresponding *_iso.laz output:")
        for p in missing_outputs:
            print(f"  {p}")
    else:
        print("\nAll LAS/LAZ files have corresponding *_iso.laz outputs.")


def main() -> None:
    # Default dataset path: sibling "dataset" directory next to this repo
    # (C:\Users\Gleb\Work\OpenForest\dev\dataset on your machine).
    default_dataset_root = (Path(__file__).resolve().parents[1] / "dataset").resolve()
    process_dataset(default_dataset_root, max_workers=1)


if __name__ == "__main__":
    main()


