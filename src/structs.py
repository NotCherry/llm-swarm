

from dataclasses import dataclass
import hashlib
import time
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

@dataclass
class NetworkMsg:
    msg: str
    data: Dict


@dataclass
class Shard:
    model_id: str
    start_layer: int
    end_layer: int
    n_layers: int
    loaded: bool

    def is_first_layer(self) -> bool:
        return self.start_layer == 0

    def is_last_layer(self) -> bool:
        return self.end_layer == (self.n_layers - 1)

    def get_layer_count(self) -> int:
        return self.end_layer + 1
    def to_dict(self) -> dict:
        return {
        "model_id": self.model_id,
        "start_layer": self.start_layer,
        "end_layer": self.end_layer,
        "n_layers": self.n_layers,
        "loaded": self.loaded
        }


class DeviceSpec(BaseModel):
    device: Literal["cpu", "cuda"]
    name: str
    ram: int
    tflops: float 


class Node(BaseModel):
    ip: str
    spec: Dict[str,DeviceSpec]
    shard: Optional[Shard] = None
    next_node_ip: Optional[str] = None
    loaded_layers: List[str] = []
    saved_layers: Optional[Dict[str, List[str]]] = {}
    def to_tuple(self):
        """Convert Node to a tuple for checksum computation."""
        return (self.ip, self.spec)

@dataclass
class NetworkConfig():
    nodes: Dict[str, Node]
    loading_model: bool = False
    loaded_model: bool = False
    generating: bool = False

class DictChecksumTracker:
    def __init__(self, dictionary: Dict[str, Node]):
        self.dictionary = dictionary
        self._checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute MD5 checksum of the dictionary."""
        # Sort keys to ensure consistent hashing
        sorted_items = sorted(
            (key, node.to_tuple()) for key, node in self.dictionary.items()
        )
        # Convert to a stable string representation
        data = str(sorted_items).encode()
        return hashlib.md5(data).hexdigest()

    def has_changed(self) -> bool:
        """Check if the dictionary has changed."""
        return self._compute_checksum() != self._checksum

    def reset(self):
        """Update the stored checksum to the current state."""
        self._checksum = self._compute_checksum()