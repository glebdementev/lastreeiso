from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class OutputFormat(Enum):
    LAS = "las"
    LAZ = "laz"


@dataclass(frozen=True)
class TilingConfig:
    output_dir: Path
    tile_size: float
    overlap: float
    output_format: OutputFormat
    call_treeiso: bool


DEFAULT_CONFIG = TilingConfig(
    output_dir=Path("tiles").resolve(),
    tile_size=40.0,
    overlap=0.0,
    output_format=OutputFormat.LAZ,
    call_treeiso=True,
)


