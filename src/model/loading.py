

from src.local_logger import log
import glob
import os
import re
import struct
from typing import Optional

import requests
from tqdm import tqdm
from src import global_vars

from src.network.wss import broadcast_layer_to_node, request_layer_from_node
from src.ptcode import safe_load_by_layer, safe_load_metadata_single
from src.structs import NetworkConfig
from src.util import convert_and_sort_by_offset


def find_key_of_node_with_layer(topology: NetworkConfig, layer: int) -> Optional[str]:
    return next((key for key, node in topology.nodes.items() if layer >= node.shard.start_layer and  layer <= node.shard.end_layer ), None)

async def download_model():
    # TODO: afer moving layers to dir
    node_key = find_key_of_node_with_layer(global_vars.NETWORK_TOPOLOGY, 0)
    assert node_key == global_vars.LOCAL_ADDRESS

    global_vars.NETWORK_TOPOLOGY.loading_model = True
    global_vars.NETWORK_TOPOLOGY.loaded_model = False
    
    missing_layers = await rearrange_layers_in_nodes()

    if len(missing_layers) == None:
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
                print(f"Warning: Layer '{l}' not found in metadata")
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

    # Step 1: Fetch header size (first 8 bytes)
    headers['Range'] = 'bytes=0-7'
    response = requests.get(url, headers=headers)
    if response.status_code not in (206, 200):
        raise Exception(f"Failed to fetch header size: HTTP {response.status_code}")
    
    length_of_header = struct.unpack('<Q', response.content)[0]

    # Step 2: Fetch metadata JSON
    headers['Range'] = f'bytes=8-{7 + length_of_header}'
    response = requests.get(url, headers=headers)
    if response.status_code not in (206, 200):
        raise Exception(f"Failed to fetch metadata: HTTP {response.status_code}")
    
    metadata = response.json()
    if "__metadata__" in metadata:
        del metadata["__metadata__"]

    # Step 3: Filter layers to download based on missing_layers
    layers_to_download = [
        (key, info) for key, info in convert_and_sort_by_offset(metadata)
        if re.match(r"model\.layers\.(\d+)", key) and re.match(r"model\.layers\.(\d+)", key).group(1) in missing_layers    
    ]

    if "model.layers.0" in metadata:
        layers_to_download.append(metadata["model.embed_tokens.weight"])

    last_layer = f"model.layers.{global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].shard.n_layers - 1}"
    if last_layer in layers_to_download:
        if "lm_head.weight" in metadata:
            layers_to_download.append(("lm_head.weight", metadata["lm_head.weight"]))
        elif "output.weight" in metadata:
            layers_to_download.append(("lm_head.weight", metadata["output.weight"]))

        if full_metadata and "model.embed_tokens.weight" in metadata:
            if "lm_head.weight" not in full_metadata and "output.weight" not in full_metadata:
                layers_to_download.append(("lm_head.weight", metadata["model.embed_tokens.weight"]))




    log.error(f"{layers_to_download} == layers_to_download")

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
            start_offset, end_offset = layer_info['data_offsets']
            range_size = end_offset - start_offset

            # Request specific range for the layer
            headers['Range'] = f'bytes={start_offset}-{end_offset-1}'
            response = requests.get(url, headers=headers, stream=True)
            if response.status_code not in (206, 200):
                raise Exception(f"Failed to download layer {layer_name}: HTTP {response.status_code}")

            # Process the response
            buffer_pointer = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    pbar.update(len(chunk))
                    buffer[buffer_pointer:buffer_pointer + len(chunk)] = chunk
                    buffer_pointer += len(chunk)

            # Verify downloaded size
            if buffer_pointer != range_size:
                log.error(f"Downloaded size {buffer_pointer} for {layer_name} does not match expected {range_size}")
                continue

            # Store layer data
            l_data = buffer[:range_size]
            log.info(f"Downloaded Layer: {layer_name}, Size: {range_size}")
            data_dict[layer_name] = {"data": l_data, "info": layer_info}

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
            await broadcast_layer_to_node(node_key, data_dict)

            # Handle special case for embed_tokens
            if layer_name == "model.embed_tokens.weight" and ("output.weight" not in metadata or "output.weight" not in (full_metadata if full_metadata is not None else "")):
                data_dict['lm_head.weight'] = data_dict['model.embed_tokens.weight']
                del data_dict['model.embed_tokens.weight']
                last_layer = global_vars.NETWORK_TOPOLOGY.nodes[find_key_of_node_with_layer(global_vars.NETWORK_TOPOLOGY, 0)].shard.n_layers - 1
                node_key = find_key_of_node_with_layer(global_vars.NETWORK_TOPOLOGY, last_layer)
                await broadcast_layer_to_node(node_key, data_dict)

            # Clear data_dict for next layer
            data_dict = {}
            buffer_pointer = 0



def veryfy_local_keys(folder_path="."):
    try:
        # Ensure the folder exists
        if not os.path.isdir(folder_path):
            raise ValueError(f"The folder '{folder_path}' does not exist or is not a directory.")

        # Get list of all *.safetensors files
        safetensors_files = glob.glob(os.path.join(folder_path, "*.safetensors"))
        
        # Check if any files were found
        if not safetensors_files:
            print(f"No *.safetensors files found in '{folder_path}'.")
            return

        # Loop over each file
        print(f"Found {len(safetensors_files)} *.safetensors files:")

        for file_path in safetensors_files:
            print(f"Processing: {file_path}")
            f, data_start, metadata = safe_load_metadata_single(file_path)
            with global_vars.NETWORK_LOCK:
                # if list(metadata.keys()) != global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].loaded_layers:
                #     list(metadata.keys())
                for l_key in metadata.keys():
                    if l_key not in global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].loaded_layers:
                        safe_load_by_layer(file_path, layer_prefix=l_key)
            f.close()
            
            
    except Exception as e:
        print(f"An error occurred: {e}")



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
            list_of_layers = [str(x) for x in range(v.start_layer, v.end_layer + 1)]
            
            for layer in list_of_layers:
                # Look for the layer in the current node's loaded layers for global_vars.SELECTED_MODEL
                layer_found = False
                layer_name = None
                if k in network_layers:
                    for s in network_layers[k]:
                        # Check if the layer name contains "layers.<layer>." or "layers.<layer>.<something>"
                        if re.search(rf'\.layers\.{layer}\b', s):  # \b ensures word boundary
                            layer_found = True
                            layer_name = s
                            break
                    
                if layer_found:
                    # Layer is already loaded on this node, no action needed
                    continue
                else:
                    # Layer not found on this node, search other nodes
                    target_node = None
                    for node_key, saved_layers in network_layers.items():
                        if node_key == k:
                            continue  # Skip the current node
                        for s in saved_layers:
                            if re.search(rf'\.layers\.{layer}\b', s):
                                target_node = node_key
                                layer_name = s
                                break
                        if target_node:
                            break
                    
                    if target_node:
                        # Call send_requested_layer with the target node and layer name
                        await request_layer_from_node(target_node, layer_name)
                    else:
                        print(f"Layer {layer} not found on any node for model {global_vars.SELECTED_MODEL}.")
                        layers_missing_in_network.append(layer)
    return layers_missing_in_network
