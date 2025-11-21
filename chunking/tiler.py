from pathlib import Path
from typing import Generator, Tuple, List, Union

import numpy as np
import laspy

from .config import TilingConfig


def compute_bounds(las: laspy.LasData) -> Tuple[float, float, float, float]:
    return float(las.header.mins[0]), float(las.header.maxs[0]), float(las.header.mins[1]), float(las.header.maxs[1])


def deduplicate_points(las: laspy.LasData) -> laspy.LasData:
    if len(las.points) == 0:
        return las
    X = np.asarray(las.X, dtype=np.int64)
    Y = np.asarray(las.Y, dtype=np.int64)
    Z = np.asarray(las.Z, dtype=np.int64)
    coords = np.stack((X, Y, Z), axis=1)
    _, inv_idx = np.unique(coords, axis=0, return_inverse=True)
    num_unique = int(np.max(inv_idx) + 1)
    seen = np.zeros(num_unique, dtype=bool)
    keep_mask = np.zeros(len(las.points), dtype=bool)
    for idx_pt in range(len(las.points)):
        g = int(inv_idx[idx_pt])
        if not seen[g]:
            keep_mask[idx_pt] = True
            seen[g] = True
    if np.all(keep_mask):
        return las
    num_deleted = int(np.sum(~keep_mask))
    print(f"Deleted {num_deleted} duplicate points (kept {int(np.sum(keep_mask))} unique points)")
    header = las.header.copy()
    deduplicated = laspy.LasData(header)
    deduplicated.points = las.points[keep_mask]
    return deduplicated


def generate_tile_windows(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    tile_size: float,
) -> Generator[Tuple[float, float, float, float, int, int], None, None]:
    step = tile_size
    x_starts: List[float] = []
    y_starts: List[float] = []

    x_current = min_x
    while x_current < max_x:
        x_starts.append(x_current)
        x_current = x_current + step
    if len(x_starts) == 0 or (x_starts[-1] + tile_size) < max_x:
        last_start = max(max_x - tile_size, min_x)
        if len(x_starts) == 0 or abs(last_start - x_starts[-1]) > 1e-9:
            x_starts.append(last_start)

    y_current = min_y
    while y_current < max_y:
        y_starts.append(y_current)
        y_current = y_current + step
    if len(y_starts) == 0 or (y_starts[-1] + tile_size) < max_y:
        last_start = max(max_y - tile_size, min_y)
        if len(y_starts) == 0 or abs(last_start - y_starts[-1]) > 1e-9:
            y_starts.append(last_start)

    for yi, y0 in enumerate(y_starts):
        y1 = y0 + tile_size
        for xi, x0 in enumerate(x_starts):
            x1 = x0 + tile_size
            yield (x0, x1, y0, y1, xi, yi)


def mask_points_in_window(
    xs: np.ndarray,
    ys: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> np.ndarray:
    in_x = (xs >= x0) & (xs < x1)
    in_y = (ys >= y0) & (ys < y1)
    return in_x & in_y


def write_tile(
    source: laspy.LasData,
    mask: np.ndarray,
    out_path: Path,
) -> None:
    header = source.header.copy()
    tile = laspy.LasData(header)
    tile.points = source.points[mask]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tile.write(str(out_path))


def tile_file(input_path: Union[str, Path], config: TilingConfig) -> List[Path]:
    path = Path(input_path).resolve()
    assert path.exists(), "Input file does not exist"
    assert path.suffix.lower() in (".las", ".laz"), "Input must be .las or .laz"
    assert config.tile_size > 0.0, "Tile size must be positive"

    las = laspy.read(str(path))
    las = deduplicate_points(las)
    min_x, max_x, min_y, max_y = compute_bounds(las)
    xs = las.x
    ys = las.y

    input_stem = path.stem
    written: List[Path] = []
    for x0, x1, y0, y1, xi, yi in generate_tile_windows(min_x, max_x, min_y, max_y, config.tile_size):
        mask = mask_points_in_window(xs, ys, x0, x1, y0, y1)
        num_points = int(np.count_nonzero(mask))
        if num_points == 0:
            continue

        out_name = f"{input_stem}_x{xi}_y{yi}.{config.output_format.value}"
        out_path = config.output_dir / out_name
        write_tile(las, mask, out_path)
        written.append(out_path)
    return written


