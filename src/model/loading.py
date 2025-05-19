

from src.model.model import get_metadata_size, get_model_metadata
from src.local_logger import log
import glob
import os
import re
import torch
from typing import Optional

import requests
from tqdm import tqdm
from src import global_vars

from src.network.process_info import broadcast_layer_to_node, request_layer_from_node
from src.ptcode import safe_load_by_layer, safe_load_metadata_single
from src.structs import NetworkConfig
from src.util import convert_and_sort_by_offset, get_model_filename


def find_key_of_node_with_layer(topology: NetworkConfig, layer: int) -> Optional[str]:
    return next((key for key, node in topology.nodes.items() 
                 if layer >= node.shard.start_layer and  layer <= node.shard.end_layer ), None)

async def download_model():
    # TODO: afer moving layers to dir
    node_key = find_key_of_node_with_layer(global_vars.NETWORK_TOPOLOGY, 0)
    assert node_key == global_vars.LOCAL_ADDRESS

    global_vars.NETWORK_TOPOLOGY.loading_model = True
    global_vars.NETWORK_TOPOLOGY.loaded_model = False
    
    missing_layers = await rearrange_layers_in_nodes()

    if len(missing_layers) <= 0:
        log.info("All layers were loaded from network")
        global_vars.NETWORK_TOPOLOGY.loading_model = False
        global_vars.NETWORK_TOPOLOGY.loaded_model = True
        return
    
    url = f"https://huggingface.co/{global_vars.SELECTED_MODEL}/resolve/main/model.safetensors.index.json"

    headers = {}
    headers["Authorization"] = f"Bearer {os.getenv("HF_TOKEN")}"
    response = requests.get(url, headers=headers)
    if response.status_code not in (206, 200, 404):
        raise Exception(f"Failed to fetch metadata: HTTP {response.status_code}")
    elif  response.status_code != 404:
        
        metadata = response.json()['weight_map']

        layers_in_file = {}
        for l in missing_layers:
            if l not in metadata:
                log.warning(f"Warning: Layer '{l}' not found in metadata")
                continue
            file_key = metadata[l]
            if file_key not in layers_in_file:
                layers_in_file[file_key] = []
            layers_in_file[file_key].append(l)

        for k, v in layers_in_file.items():
            url_of_safetensors = f"https://huggingface.co/{global_vars.SELECTED_MODEL}/resolve/main/{k}"
            load_safetensors_from_network(url_of_safetensors, missing_layers=v, full_metadata=metadata)
    else:
        url_of_safetensors = f"https://huggingface.co/{global_vars.SELECTED_MODEL}/resolve/main/model.safetensors"
        await load_safetensors_from_network(url_of_safetensors, missing_layers=missing_layers)


    global_vars.NETWORK_TOPOLOGY.loading_model = False
    global_vars.NETWORK_TOPOLOGY.loaded_model = True
    log.info(f"Download of missing layers for model {global_vars.SELECTED_MODEL} completed")    


async def load_safetensors_from_network(url, missing_layers, hf_token=os.getenv("HF_TOKEN"), chunk_size=1024*1024*10, full_metadata=None):
    # Set headers with Hugging Face token if provided
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    metadata = get_model_metadata(url)
    header_offset = get_metadata_size(url) + 8
    
    layers_to_download = []
    list_of_layers = convert_and_sort_by_offset(metadata)
    for (l_name, l_info) in list_of_layers:
        for l_num in missing_layers:
            if f"layer.{l_num}." in l_name or f".{l_num}." in l_name:
                layers_to_download.append((l_name, l_info))

    l_names = [ name for (name, i) in layers_to_download ]

    last_layer = f"model.layers.{global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].shard.n_layers - 1}"
    if any(name.startswith(last_layer) for name in l_names):
        layers_to_download.append(("model.norm.weight", metadata["model.norm.weight"]))
    if any(name.startswith("model.layers.0") for name in l_names):
        layers_to_download.append(("model.embed_tokens.weight", metadata["model.embed_tokens.weight"]))
    

    if not layers_to_download:
        log.info("No missing layers to download")
        return

    # Calculate total size to download for progress bar
    total_download_size = sum(
        info['data_offsets'][1] - info['data_offsets'][0]
        for _, info in layers_to_download
    )

    # Initialize buffer and data dictionary
    data_dict = {}
    max_layer_size = max(
        (info['data_offsets'][1] - info['data_offsets'][0])
        for _, info in layers_to_download
    ) if layers_to_download else chunk_size
    buffer = bytearray(max_layer_size)

    # Step 4: Loop over missing layers and download their data ranges
    with tqdm(total=total_download_size, unit="B") as pbar:
        for layer_name, layer_info in layers_to_download:
            log.info(f"Downloading layer : {layer_name} - {layer_info}")
            start_offset, end_offset = layer_info['data_offsets']
            
            # account that data offsets are not absolute but relative after header end
            start_offset += header_offset
            end_offset += header_offset

            range_size = end_offset - start_offset

            # Request specific range for the layer
            headers['Range'] = f'bytes={start_offset}-{end_offset-1}'
            response = requests.get(url, headers=headers)
            if response.status_code not in (206, 200):
                raise Exception(f"Failed to download layer {layer_name}: HTTP {response.status_code}")

            # Get the complete content
            l_data = bytearray(response.content)
            downloaded_size = len(l_data)

            # Verify downloaded size
            if downloaded_size != range_size:
                log.error(f"Downloaded size {downloaded_size} for {layer_name} does not match expected {range_size}")
                continue

            # Store layer data
            log.info(f"Downloaded Layer: {layer_name}, Size: {range_size}")
            data_dict[layer_name] = {"data": l_data, "info": layer_info}
            pbar.update(range_size)
            # Determine node for broadcasting
            node_key = None
            l_num = None
            if "model.layers" in layer_name:
                l_num = int(layer_name.split('.')[2])
                node_key = find_key_of_node_with_layer(global_vars.NETWORK_TOPOLOGY, l_num)
            elif "model.embed_tokens" in layer_name:
                node_key = find_key_of_node_with_layer(global_vars.NETWORK_TOPOLOGY, 0)
            elif "model.norm" in layer_name or "output.weight" in layer_name:
                last_layer = global_vars.NETWORK_TOPOLOGY.nodes[find_key_of_node_with_layer(global_vars.NETWORK_TOPOLOGY, 0)].shard.n_layers - 1
                node_key = find_key_of_node_with_layer(global_vars.NETWORK_TOPOLOGY, last_layer)
            if node_key is None:
                log.error(f"Failed to find node_key for layer: {layer_name}")
                continue

            # Broadcast layer to node
            log.error(data_dict.keys())
            await broadcast_layer_to_node(node_key, data_dict)

            # Handle special case for embed_tokens
            if layer_name == "model.embed_tokens.weight" and ("output.weight" not in metadata or "output.weight" not in (full_metadata if full_metadata is not None else "")):
                data_dict['lm_head.weight'] = data_dict['model.embed_tokens.weight']
                del data_dict['model.embed_tokens.weight']
                last_layer = global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].shard.n_layers - 1
                node_key = find_key_of_node_with_layer(global_vars.NETWORK_TOPOLOGY, last_layer)
                log.error(data_dict.keys())
                await broadcast_layer_to_node(node_key, data_dict)

            # Clear data_dict for next layer
            data_dict = {}
            buffer_pointer = 0



def verify_local_keys(folder_path="."):
    try:
        if not os.path.isdir(folder_path):
            raise ValueError(f"The folder '{folder_path}' does not exist or is not a directory.")

        safetensors_files = glob.glob(os.path.join(folder_path, "*.safetensors"))
        
        if not safetensors_files:
            log.info(f"No *.safetensors files found in '{folder_path}'.")
            return

        log.info(f"Found {len(safetensors_files)} *.safetensors files:")

        layers_on_disk = {}

        for file_path in safetensors_files:
            log.info(f"Processing: {file_path}")
            cleaned_name = file_path.removeprefix("./").removesuffix(".safetensors").replace("!", "/")
            file_handle, data_start, metadata = safe_load_metadata_single(file_path)
            layers_on_disk[cleaned_name] = list(metadata.keys())
            print(layers_on_disk)

        return layers_on_disk
            
    except Exception as e:
        log.error(f"An error occurred: {e}")



async def rearrange_layers_in_nodes():
    """
    The primary goal is to manage a distributed network of nodes, each storing specific model layers 
    (e.g., Node A stores layers 1, 2, 3; Node B stores layers 4, 5, 6) to execute a machine learning model. 
    When the network detects a new node, Node C, which is computationally faster than
    Node A (with Node C > Node A > Node B in terms of speed), the system will copy the layers from the slowest node (Node B in this case) 
    to Node C. This transfer aims to leverage Node C's superior performance to accelerate model execution.

    Future Considerations:

    The system will account for the network card speed and internet connectivity of nodes. For instance, 
    if Node B uses a base-1000 (1 Gbps) network interface and Node C uses a base-10G (10 Gbps) interface, 
    the system will factor in these differences to optimize layer transfers and model performance in future iterations.
    """
    log.debug("Rearrange nodes")
    layers_missing_in_network = []
    with global_vars.NETWORK_LOCK:
        network_layers = {
            k: v.saved_layers[global_vars.SELECTED_MODEL] for k, v in global_vars.NETWORK_TOPOLOGY.nodes.items()
            if global_vars.SELECTED_MODEL in v.saved_layers
        }
        network_shards = {
            k: v.shard for k, v in global_vars.NETWORK_TOPOLOGY.nodes.items()
        }
    for k, v in network_shards.items():
        # Get the range of layers for this shard (as strings)
        layers = [str(i) for i in range(v.start_layer, v.end_layer + 1)]
        last_layer = global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].shard.n_layers - 1

        # Track loaded components
        loaded = {"embed": False, "norm": False, "lm_head": False}

        for layer in layers:
            # Check for matching layer in network_layers
            for layer_name in network_layers.get(k, []):
                if re.search(rf'\.layers\.{layer}\b', layer_name):
                    layer_found = True
                    # Add layer to loaded_layers if not already present
                    with global_vars.NETWORK_LOCK:
                        if layer_name not in global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].loaded_layers:
                            global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].loaded_layers.append(layer_name)
                    
                    # Load and broadcast layer weights
                    weights = safe_load_by_layer(get_model_filename(), layer_prefix=layer_name)
                    if weights:
                        await broadcast_layer_to_node(global_vars.LOCAL_ADDRESS, weights)
                    break  # Found the layer, move to next

            # Handle special layers (embed, norm, lm_head) for specific conditions
            special_layers = [
                (layer == "0" and not loaded["embed"], "model.embed_tokens.weight", "embed"),
                (layer == str(last_layer) and not loaded["norm"], "model.norm.weight", "norm"),
                (layer == str(last_layer) and not loaded["lm_head"], "lm_head.weight", "lm_head")
            ]

            for condition, layer_name, key in special_layers:
                if condition:
                    global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].loaded_layers.append(layer_name)
                    weights = safe_load_by_layer(get_model_filename(), layer_prefix=layer_name)
                    if weights:
                        await broadcast_layer_to_node(global_vars.LOCAL_ADDRESS, weights)
                        loaded[key] = True
                    else:
                        log.error(f"Failed to load {layer_name}")


                
            if layer_found:
                # Layer is already loaded on this node, no action needed
                continue
            else:
                # Layer not found on this node, search other nodes
                target_node = None
                for node_key, saved_layers in network_layers.items():
                    if node_key == k:
                        continue  # Skip the current node
                    for layer_name in saved_layers:
                        if re.search(rf'\.layers\.{layer}\b', layer_name):
                            target_node = node_key
                            # Add layer to loaded_layers if not already present
                            with global_vars.NETWORK_LOCK:
                                if layer_name not in global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].loaded_layers:
                                    global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].loaded_layers.append(layer_name)
                            
                            # Load and broadcast layer weights
                            await request_layer_from_node(target_node, layer_name)

                            # Handle special layers (embed, norm, lm_head) for specific conditions
                            special_layers = [
                                (layer == "0" and not loaded["embed"], "model.embed_tokens.weight", "embed"),
                                (layer == str(last_layer) and not loaded["norm"], "model.norm.weight", "norm"),
                                (layer == str(last_layer) and not loaded["lm_head"], "lm_head.weight", "lm_head")
                            ]

                            for condition, special_layer_name, key in special_layers:
                                if condition:
                                    global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].loaded_layers.append(special_layer_name)
                                    await broadcast_layer_to_node(global_vars.LOCAL_ADDRESS, weights)
                                    loaded[key] = True
                    if target_node:
                        break
                
                if not target_node:                    
                    log.error(f"Layer {layer} not found on any node for model {global_vars.SELECTED_MODEL}.")
                    layers_missing_in_network.append(layer)
    return layers_missing_in_network
