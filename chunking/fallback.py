from pathlib import Path
from typing import Union

import numpy as np
import laspy


def write_final_segs_ones(tile_path: Union[str, Path]) -> None:
    path = Path(tile_path)
    las = laspy.read(str(path))
    num_points = int(len(las.x))
    final_labels = np.ones(num_points, dtype=np.int32)
    las.add_extra_dim(laspy.ExtraBytesParams(name="final_segs", type="int32", description="final_segs"))
    las.final_segs = final_labels
    out_path = Path(str(path)[:-4] + "_treeiso.laz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_path))

