import json
import os
import re
import struct
from typing import Dict, Union
import requests
import torch
from src.ptcode import LlamaModel
from src.structs import Shard
from accelerate import init_empty_weights
from src.util import debug_decorator, get_model_filename, normalize_url
from src.local_logger import log
from src import global_vars 

def init_model():
    shard: Shard = global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].shard
    with global_vars.MODEL_LOCK:
        if "gpu" in global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].spec.keys():
            global_vars.MODEL = LlamaModel(shard)
            # global_vars.MODEL.to_empty(device = torch.device("cuda"))
        else:
            global_vars.MODEL = LlamaModel(shard)
        


def layers_size(metadata: Union[list, dict]) -> Dict[str, int]:
    sizes = {}
    layer_size = 0
    current_layer = -1

    def process_layer(key: str, size: int):
        nonlocal layer_size, current_layer
        layer_match = re.match(r"model\.layers\.(\d+)", key)
        if layer_match:
            layer_num = int(layer_match.group(1))
            if layer_num != current_layer:
                if current_layer != -1:
                    sizes[f"model.layers.{current_layer}"] = layer_size
                layer_size = size
                current_layer = layer_num
            else:
                layer_size += size
        else:
            sizes[key] = size

    if isinstance(metadata, list):
        for key, layer in metadata:
            size = layer['data_offsets'][1] - layer['data_offsets'][0]
            process_layer(key, size)
    else:
        if '__metadata__' in metadata:
            del metadata['__metadata__']
        for key, layer in metadata.items():
            size = layer['data_offsets'][1] - layer['data_offsets'][0]
            process_layer(key, size)

    # Add the last layer's size
    if current_layer != -1:
        sizes[f"model.layers.{current_layer}"] = layer_size

    return sizes

def get_selected_model_metadata_from_index():
    url = f"https://huggingface.co/{global_vars.SELECTED_MODEL}/resolve/main/model.safetensors.index.json"

    
    layers_names = {}
    headers = {}
    headers["Authorization"] = f"Bearer {os.getenv("HF_TOKEN")}"
    response = requests.get(url, headers=headers)
    if response.status_code not in (206, 200, 404):
        raise Exception(f"Failed to fetch metadata: HTTP {response.status_code}")
    elif response.status_code != 404:
        metadata = response.json()['weight_map']
        
        for k in metadata.keys():
            url_of_safetensors = f"https://huggingface.co/{global_vars.SELECTED_MODEL}/resolve/main/{k}"
            layers_names = {**layers_names, **get_model_metadata(url_of_safetensors)}
    else:
        url_of_safetensors = f"https://huggingface.co/{global_vars.SELECTED_MODEL}/resolve/main/model.safetensors"
        layers_names = {**layers_names, **get_model_metadata(url_of_safetensors)}
    
    if not "lm_heads" in layers_names.keys():
        layers_names["lm_head.weight"] = layers_names["model.embed_tokens.weight"]
    if "__metadata__" in layers_names.keys():
        del layers_names["__metadata__"]
    return layers_names
    

@debug_decorator
def get_model_metadata(url: str, meta_folder: str = "meta") -> dict | None:
    """
    Fetch model metadata from a URL and cache it locally.
    
    Args:
        url (str): URL to fetch metadata from
        meta_folder (str): Directory to store metadata files
    
    Returns:
        dict | None: Metadata dictionary if successful, None otherwise
    """
    url = normalize_url(url)

    def download_metadata(url: str) -> dict | None:
        """Helper function to download metadata from URL."""
        try:
            headers = {'Range': 'bytes=0-7'}
            hf_token = os.getenv('HF_TOKEN', '')
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"

            # Fetch first 8 bytes to get header length
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            length_of_header = struct.unpack('<Q', response.content)[0]

            # Fetch metadata based on header length
            headers['Range'] = f'bytes=8-{7 + length_of_header}'
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse JSON response
            meta = response.json()
            if '__metadata__' in meta:
                del meta['__metadata__']
            return meta

        except (requests.RequestException, ValueError, KeyError) as e:
            log.error(f"Failed to fetch metadata from {url}: {e}")
            # Try fallback URL
            try:
                fallback_url = f"{url}.index.json"
                response = requests.get(fallback_url, timeout=10)
                response.raise_for_status()
                if "__metadata__" in response.json():
                    del response.json()['__metadata__']
                return response.json()
            except requests.RequestException as e:
                log.error(f"Failed to fetch metadata from fallback {fallback_url}: {e}")
                return None

    # Initialize metadata store
    metadata_store_file = "metadata_store.json"
    data = {}

    # Load existing metadata store
    if os.path.exists(metadata_store_file):
        try:
            with open(metadata_store_file, 'r') as f:
                content = f.read().strip()
                if content:  # Check if file is not empty
                    data = json.loads(content)
                else:
                    log.warning(f"{metadata_store_file} is empty")
        except json.JSONDecodeError as e:
            log.error(f"Failed to parse {metadata_store_file}: {e}")
            return None
        except IOError as e:
            log.error(f"Failed to read {metadata_store_file}: {e}")
            return None

    # Check if metadata is already cached
    if url in data:
        try:
            with open(os.path.join(meta_folder, data[url]), 'r') as mf:
                return json.loads(mf.read())
        except (IOError, json.JSONDecodeError) as e:
            log.error(f"Failed to read cached metadata for {url}: {e}")
            # If cached file is corrupted, try downloading again
            meta = download_metadata(url)
            if meta is None:
                return None
    else:
        # Download new metadata
        meta = download_metadata(url)
        if meta is None:
            return None

        # Save metadata to file
        try:
            # Assuming get_model_filename() is defined elsewhere
            fn_meta = f"{get_model_filename()}.json"
            data[url] = fn_meta

            # Create meta_folder if it doesn't exist
            os.makedirs(meta_folder, exist_ok=True)
            log.info(f"Ensured folder exists: {meta_folder}")

            # Write metadata to file
            with open(os.path.join(meta_folder, fn_meta), 'w') as ff:
                json.dump(meta, ff, indent=2)
            
            # Update metadata store
            with open(metadata_store_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return meta

        except (IOError, NameError) as e:
            log.error(f"Failed to save metadata for {url}: {e}")
            return None
            

def get_metadata_size(url):
    meta = get_model_metadata(url)
    json_bytes = json.dumps(meta, separators=(",", ":")).encode('utf-8')
    extra = (8 - len(json_bytes) % 8) % 8
    json_bytes += b" " * extra
    return len(json_bytes)
