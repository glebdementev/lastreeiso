from pathlib import Path
from typing import Tuple, Union

import numpy as np
import laspy
from scipy.spatial import cKDTree


def _copy_with_mask(source: laspy.LasData, mask: np.ndarray) -> laspy.LasData:
    header = source.header.copy()
    out = laspy.LasData(header)
    out.points = source.points[mask]
    return out


def statistical_outlier_removal(
    las: laspy.LasData,
    k_neighbors: int = 10,
    std_ratio: float = 1.0,
) -> Tuple[laspy.LasData, int]:
    if len(las.points) == 0:
        return las, 0
    coords = np.stack((las.x, las.y, las.z), axis=1)
    # Query K+1 to include self; we will drop the first column
    tree = cKDTree(coords)
    # Clamp K to available points to avoid degenerate queries
    k_eff = int(min(max(k_neighbors + 1, 1), len(coords)))
    if k_eff <= 1:
        return las, 0
    # SciPy API differences: prefer 'workers', fallback to no parallelism
    try:
        distances, _ = tree.query(coords, k=k_eff, workers=-1)  # SciPy >= 1.6
    except TypeError:
        distances, _ = tree.query(coords, k=k_eff)  # Older SciPy
    # Exclude distance to self at [:, 0]
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
    local_means = np.mean(distances[:, 1:], axis=1)
    mu = float(np.mean(local_means))
    sigma = float(np.std(local_means))
    if sigma == 0.0:
        return las, 0
    keep_mask = local_means <= (mu + std_ratio * sigma)
    removed = int(np.count_nonzero(~keep_mask))
    if removed == 0:
        return las, 0
    return _copy_with_mask(las, keep_mask), removed


def voxel_grid_decimate(
    las: laspy.LasData,
    resolution_m: float = 0.02,
) -> Tuple[laspy.LasData, int]:
    if len(las.points) == 0:
        return las, 0
    x = las.x
    y = las.y
    z = las.z
    min_x = float(np.min(x))
    min_y = float(np.min(y))
    min_z = float(np.min(z))
    # Compute integer voxel coordinates
    ix = np.floor((x - min_x) / resolution_m).astype(np.int64)
    iy = np.floor((y - min_y) / resolution_m).astype(np.int64)
    iz = np.floor((z - min_z) / resolution_m).astype(np.int64)
    voxels = np.stack((ix, iy, iz), axis=1)
    # Keep first occurrence per voxel
    _, first_indices = np.unique(voxels, axis=0, return_index=True)
    keep_mask = np.zeros(len(las.points), dtype=bool)
    keep_mask[first_indices] = True
    removed = int(np.count_nonzero(~keep_mask))
    if removed == 0:
        return las, 0
    return _copy_with_mask(las, keep_mask), removed


def deduplicate_points_exact(las: laspy.LasData) -> Tuple[laspy.LasData, int]:
    if len(las.points) == 0:
        return las, 0
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
        return las, 0
    removed = int(np.count_nonzero(~keep_mask))
    return _copy_with_mask(las, keep_mask), removed


def preprocess_las_file(
    input_path: Union[str, Path],
    out_path: Union[str, Path, None] = None,
    sor_k: int = 10,
    sor_std: float = 1.0,
    decimation_res_m: float = 0.02,
) -> Path:
    """
    Apply SOR (k, std), voxel decimation (meters), and deduplication.
    Writes a .laz file and returns its path. Prints removal stats.
    """
    input_path_resolved = Path(input_path).resolve()
    las = laspy.read(str(input_path_resolved))
    original_count = int(len(las.points))

    # 1) SOR
    las, removed_sor = statistical_outlier_removal(las, k_neighbors=sor_k, std_ratio=sor_std)
    # 2) Decimation
    las, removed_dec = voxel_grid_decimate(las, resolution_m=decimation_res_m)
    # 3) Deduplication
    las, removed_dup = deduplicate_points_exact(las)

    final_count = int(len(las.points))
    total_removed = original_count - final_count

    print(f"SOR removed {removed_sor} points (k={sor_k}, std={sor_std})")
    print(f"Decimation removed {removed_dec} points (res={decimation_res_m} m)")
    print(f"Deduplication removed {removed_dup} points")
    print(f"Total removed {total_removed}; final count {final_count} from original {original_count}")

    # Determine output path
    if out_path is None:
        out_dir = input_path_resolved.parent
        out_name = input_path_resolved.stem + "_preproc.laz"
        out_path_resolved = (out_dir / out_name).resolve()
    else:
        out_path_resolved = Path(out_path).resolve()
        if out_path_resolved.suffix.lower() not in (".laz", ".las"):
            out_path_resolved = out_path_resolved.with_suffix(".laz")
    out_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    # Always write as .laz
    las.write(str(out_path_resolved.with_suffix(".laz")))
    return out_path_resolved.with_suffix(".laz")


