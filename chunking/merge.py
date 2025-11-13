from pathlib import Path
from typing import List

import numpy as np
import laspy


def merge_tiles_to_single(processed_tiles: List[Path], output_path: Path) -> Path:
    assert len(processed_tiles) > 0, "No tiles to merge"
    tiles_sorted = sorted(processed_tiles, key=lambda p: p.name)

    header0 = laspy.read(str(tiles_sorted[0])).header.copy()
    points_list: List[np.ndarray] = []
    for p in tiles_sorted:
        las = laspy.read(str(p))
        if "final_segs" not in las.point_format.extra_dimension_names:
            las.add_extra_dim(laspy.ExtraBytesParams(name="final_segs", type="int32", description="final_segs"))
            las.final_segs = np.zeros(len(las.points), dtype=np.int32)
        points_list.append(las.points.array)

    all_points = points_list[0] if len(points_list) == 1 else np.concatenate(points_list, axis=0)

    if "final_segs" not in header0.point_format.extra_dimension_names:
        header0.add_extra_dims([laspy.ExtraBytesParams(name="final_segs", type="int32", description="final_segs")])
    if "final_segs" not in all_points.dtype.names:
        # If somehow not present, initialize to zeros
        zeros = np.zeros(all_points.shape[0], dtype=np.int32)
        # Create a new dtype with final_segs appended
        new_dtype = all_points.dtype.descr + [("final_segs", "<i4")]
        new_points = np.empty(all_points.shape, dtype=np.dtype(new_dtype))
        for name in all_points.dtype.names or []:
            new_points[name] = all_points[name]
        new_points["final_segs"] = zeros
        all_points = new_points

    output_path = output_path.with_suffix(".laz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with laspy.open(str(output_path), mode="w", header=header0) as writer:
        point_record = laspy.ScaleAwarePointRecord(all_points, header0.point_format, header0.scales, header0.offsets)
        writer.write_points(point_record)
    return output_path


