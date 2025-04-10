from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Frame:
    position: Tuple[float, float, float]  # xyz
    orientation: Tuple[float, float, float, float]  # xyzw

    def __post_init__(self):
        assert len(self.position) == 3, f"len(self.position): {len(self.position)}"
        assert len(self.orientation) == 4, (
            f"len(self.orientation): {len(self.orientation)}"
        )

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(position=data["position"], orientation=data["orientation"])

    def to_dict(self):
        return {"position": self.position, "orientation": self.orientation}


@dataclass
class Trajectory:
    type: str
    mesh_path: Path
    mesh_scale: Tuple[float, float, float]
    frames: List[Frame]

    def __post_init__(self):
        assert self.type in ["mesh"], f"self.type: {self.type}"
        assert self.mesh_path.exists(), f"self.mesh_path: {self.mesh_path}"
        assert len(self.mesh_scale) == 3, (
            f"len(self.mesh_scale): {len(self.mesh_scale)}"
        )

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            type=data["type"],
            mesh_path=Path(data["mesh_path"]),
            mesh_scale=data["mesh_scale"],
            frames=[Frame.from_dict(frame) for frame in data["frames"]],
        )

    def to_dict(self):
        return {
            "type": self.type,
            "mesh_path": str(self.mesh_path),
            "mesh_scale": self.mesh_scale,
            "frames": [frame.to_dict() for frame in self.frames],
        }
