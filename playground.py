# with open('model.safetensors', 'rb') as f:
#     print(int.from_bytes(f.read(8), 'little'))
import hashlib
from typing import Dict
from dataclasses import dataclass
from util import log

@dataclass
class Shard:
    model_id: str
    start_layer: int
    end_layer: int
    n_layers: int
    loaded: bool



@dataclass
class Node:
    ip: str
    shard: Shard
    def to_tuple(self):
        """Convert Node to a tuple for checksum computation."""
        return (self.ip, self.shard)

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


a = {
    "a" : Node("1.1.1", shard=Shard("1", 0, 5, 10, False)),
    "b" : Node("2.2.2", shard=Shard("1", 6, 9, 10, False))
} 

ch = DictChecksumTracker(a)._checksum

a = {}

a = {
    "a" : Node("1.1.1", shard=Shard("1", 0, 5, 10, False)),
    "b" : Node("2.2.2", shard=Shard("1", 6, 9, 10, False))
} 

ch2 = DictChecksumTracker(a)._checksum

print(ch, ch2)
assert ch == ch2

b =[('__metadata__', {'format': 'pt'}), ('model.embed_tokens.weight', {'dtype': 'BF16', 'shape': [128256, 2048], 'data_offsets': [0, 525336576]})]

print(isinstance(b, list))

for k, v in a.items():
    log.info(f"{v.shard}")