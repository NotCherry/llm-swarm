import json
from typing import Dict
import psutil
from src.structs import Node
import torch
from src import global_vars
from src.benchmarks import get_flops
from src.structs import DeviceSpec, NetworkConfig




def serialize_network_config(config: NetworkConfig) -> str:
    # Convert NetworkConfig to a dict, ensuring Pydantic models are serialized
    config_dict = {
        "nodes": {key: node.model_dump() for key, node in config.nodes.items()}
    }
    # Serialize to JSON string
    return json.dumps(config_dict)

def deserialize_network_config(json_str: str) -> NetworkConfig:
    # Parse JSON string to dict
    config_dict = json.loads(json_str)
    # Convert nodes dict to Node objects
    nodes = {
        key: Node(**node_data)
        for key, node_data in config_dict["nodes"].items()
    }
    # Create NetworkConfig with deserialized nodes
    return NetworkConfig(nodes=nodes)

def detect_device():
    memory = psutil.virtual_memory()

    spec = {
  
        "cpu" :
        DeviceSpec(
        device="cpu",
        name="cpu",
        ram=memory.available,
        ram_type="RAM",
        tflops=0
    )
    }
    if torch.cuda.is_available():
        spec["gpu"] =  DeviceSpec( 
            device="cuda",
            name=torch.cuda.get_device_name(torch.cuda.current_device()),
            ram=torch.cuda.mem_get_info()[0],
            ram_type="VRAM",
            tflops=get_flops()
        )
    return spec    


def get_model_filename():
    return global_vars.SELECTED_MODEL.replace("/","-")

def separate_nodes():
    # Nodes with GPU devices
    nodes_with_gpu: Dict[str, DeviceSpec] = {
        k: v.spec['gpu'] for k, v in global_vars.NETWORK_TOPOLOGY.nodes.items() if "gpu" in v.spec
    }
    
    # Nodes with CPU devices but no GPU
    nodes_cpu_only: Dict[str, DeviceSpec] = {
        k: v.spec['cpu'] for k, v in global_vars.NETWORK_TOPOLOGY.nodes.items() if "cpu" in v.spec and "gpu" not in v.spec
    }

    return (
        sorted(nodes_with_gpu.items(), key=lambda x: x[1].ram, reverse=True),
        sorted(nodes_cpu_only.items(), key=lambda x: x[1].ram, reverse=True)
    )

def get_last_layer_number(metadata: Dict[str, any]) -> int:
    layer_numbers = [
        int(x.split(".")[2])
        for x in metadata.keys()
        if "layer" in x.lower() and len(x.split(".")) > 2 and x.split(".")[2].isdigit()
    ]
    if not layer_numbers:
        raise ValueError("No valid layer keys found in metadata")
    return max(layer_numbers)

def convert_and_sort_by_offset(metadata):
    # Convert dict to list of (key, value) tuples
    if '__metadata__' in metadata.keys(): del metadata['__metadata__']
    items = [(key, value) for key, value in metadata.items()]
    
    # Sort by the first element of data_offsets
    sorted_items = sorted(items, key=lambda x: x[1]["data_offsets"][0])
    
    return sorted_items

