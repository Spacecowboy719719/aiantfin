from typing import Any, Dict, List, Optional
import math

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import HOME_X, HOME_Y, SIMULATION_STATE_VERSION, WORLD_HEIGHT, WORLD_WIDTH


def _finite_or_default(value: Any, default: float) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


class AntStateSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    x: float = Field(default=HOME_X, ge=0.0, le=float(WORLD_WIDTH))
    y: float = Field(default=HOME_Y, ge=0.0, le=float(WORLD_HEIGHT))
    direction: float = 0.0
    speed: float = Field(default=0.0, ge=0.0, le=20.0)
    age: int = Field(default=0, ge=0)
    energy: float = Field(default=100.0, ge=0.0, le=100.0)
    hunger: float = Field(default=0.0, ge=0.0, le=100.0)
    fear: float = Field(default=0.0, ge=0.0, le=100.0)
    mode: str = "wander"

    memory: List[Any] = Field(default_factory=list)
    episodes: List[Dict[str, Any]] = Field(default_factory=list)
    spatialmap: List[List[float]] = Field(default_factory=list)
    visiblefoodcount: int = 0
    target: Optional[Dict[str, Any]] = None
    hazards: List[Dict[str, Any]] = Field(default_factory=list)

    curiosity: float = 0.0
    smoothness: float = 1.0

    dominant_drive: str = "curiosity"
    drives: Dict[str, float] = Field(default_factory=dict)
    decision_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    topo_nodes: int = Field(default=0, ge=0)

    state_version: str = SIMULATION_STATE_VERSION

    @model_validator(mode="before")
    @classmethod
    def clamp_values(cls, data):
        if not isinstance(data, dict):
            return data

        data.setdefault("memory", [])
        data.setdefault("episodes", [])
        data.setdefault("spatialmap", [])
        data.setdefault("hazards", [])
        data.setdefault("drives", {})
        data.setdefault("state_version", SIMULATION_STATE_VERSION)
        data.setdefault("mode", "wander")
        data.setdefault("dominant_drive", "curiosity")

        data["x"] = min(float(WORLD_WIDTH), max(0.0, _finite_or_default(data.get("x", HOME_X), HOME_X)))
        data["y"] = min(float(WORLD_HEIGHT), max(0.0, _finite_or_default(data.get("y", HOME_Y), HOME_Y)))
        data["direction"] = _finite_or_default(data.get("direction", 0.0), 0.0) % 360.0
        data["speed"] = min(20.0, max(0.0, _finite_or_default(data.get("speed", 0.0), 0.0)))
        data["energy"] = min(100.0, max(0.0, _finite_or_default(data.get("energy", 100.0), 100.0)))
        data["hunger"] = min(100.0, max(0.0, _finite_or_default(data.get("hunger", 0.0), 0.0)))
        data["fear"] = min(100.0, max(0.0, _finite_or_default(data.get("fear", 0.0), 0.0)))
        data["curiosity"] = min(100.0, max(0.0, _finite_or_default(data.get("curiosity", 0.0), 0.0)))
        data["smoothness"] = min(1.0, max(0.0, _finite_or_default(data.get("smoothness", 1.0), 1.0)))
        data["decision_confidence"] = min(
            1.0, max(0.0, _finite_or_default(data.get("decision_confidence", 0.0), 0.0))
        )

        try:
            data["age"] = max(0, int(data.get("age", 0)))
        except Exception:
            data["age"] = 0

        try:
            data["visiblefoodcount"] = max(0, int(data.get("visiblefoodcount", 0)))
        except Exception:
            data["visiblefoodcount"] = 0

        try:
            data["topo_nodes"] = max(0, int(data.get("topo_nodes", 0)))
        except Exception:
            data["topo_nodes"] = 0

        if not isinstance(data.get("target"), dict):
            data["target"] = None
        if not isinstance(data.get("drives"), dict):
            data["drives"] = {}

        return data


class SnapshotResponse(AntStateSchema):
    home: Dict[str, float]
    worldw: float
    worldh: float

    foods: List[Dict[str, Any]] = Field(default_factory=list)
    foodclusters: List[Dict[str, Any]] = Field(default_factory=list)
    poisonzones: List[Dict[str, Any]] = Field(default_factory=list)
    walls: List[Dict[str, Any]] = Field(default_factory=list)
    vision: Optional[Dict[str, Any]] = None

    metastuckcounter: int = 0
    meta_stuck_counter: int = 0

    decisionconfidence: float = 0.0
    toponodes: int = 0

    # Дополнительные поля, которые могут приходить из brain/server
    current_goal: str = "WANDER"
    goal_utility: float = 0.0
    personality: Dict[str, float] = Field(default_factory=dict)
    regions: List[Dict[str, Any]] = Field(default_factory=list)
    decorations: List[Dict[str, Any]] = Field(default_factory=list)