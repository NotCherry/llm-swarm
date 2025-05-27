import json
from typing import Dict
import psutil
from src.structs import Node
import torch
from src import global_vars
from src.benchmarks import get_flops
from src.structs import DeviceSpec, NetworkConfig
from src.local_logger import log



def serialize_network_config(config: NetworkConfig) -> str:
    if not isinstance(config.nodes, dict):
        raise ValueError("config.nodes must be a dictionary")
    config_dict = {
        "nodes": {key: node.model_dump() for key, node in config.nodes.items()}
    }
    try:
        return json.dumps(config_dict, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Serialization failed: {str(e)}")

import json
from typing import Dict

def deserialize_network_config(json_str: str) -> NetworkConfig:
    # Parse JSON string to dict
    config_dict = json.loads(json_str)
    # Convert nodes dict to Node objects, ensuring saved_layers is a dict
    nodes = {
        key: Node(**{**node_data, "saved_layers": node_data.get("saved_layers") or {}})
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
    return global_vars.SELECTED_MODEL.replace("/","!")

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

def debug_decorator(func):
    def wrapper(*args, **kwargs):
        # Get function name
        func_name = func.__name__
        
        # Helper function to format values
        def format_value(value):
            if isinstance(value, (bytearray, bytes, list)):
                return str(len(value))
            elif isinstance(value, dict):
                return '{' + ', '.join(f"{k}: {format_value(v)}" for k, v in value.items()) + '}'
            return repr(value)
        
        # Format positional arguments
        args_repr = [format_value(arg) for arg in args]
        
        # Format keyword arguments
        kwargs_repr = [f"{k}={format_value(v)}" for k, v in kwargs.items()]
        
        # Combine all arguments
        all_args = args_repr + kwargs_repr
        
        # Print function call details
        log.error(f"Calling {func_name}({', '.join(all_args)})")
        
        # Call the original function
        result = func(*args, **kwargs)
        
        return result
    return wrapper

from urllib.parse import urlparse

def normalize_url(url):
    """
    Normalize a URL or model identifier to a full HuggingFace model URL.
    
    Args:
        url (str): Either a full URL or a model identifier like "Qwen/Qwen3-1.7B"
        
    Returns:
        str: Full URL to the model.safetensors file
        
    Raises:
        ValueError: If URL is not a valid non-empty string
    """
    # Check if url is a valid string
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")
    
    url = url.strip()
    
    # Parse the URL to check for scheme and netloc
    parsed = urlparse(url)
    
    # If it's already a complete URL (has both scheme and netloc), return unchanged
    if parsed.scheme and parsed.netloc:
        return url
    
    # If it's a model identifier (like "Qwen/Qwen3-1.7B") or partial path,
    # convert it to full HuggingFace URL
    # Remove leading/trailing slashes to normalize the identifier
    model_id = url.strip('/')
    return f"https://huggingface.co/{model_id}/resolve/main/model.safetensors"