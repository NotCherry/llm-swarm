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
from src.util import get_model_filename
from src.local_logger import log
from src import global_vars 

def init_model():
    shard: Shard = global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].shard
    with global_vars.MODEL_LOCK:
        if "gpu" in global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].spec.keys():
            with init_empty_weights():
                global_vars.MODEL = LlamaModel(shard)
            global_vars.MODEL.to_empty(device = torch.device("cuda"))
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
        

def get_model_metadata(url, meta_folder="meta"):
    # Fetch the first 8 bytes of the file
    meta = None
    def download(url):
        try:
            headers = {'Range': 'bytes=0-7'}
            hf_token = os.getenv('HF_TOKEN')
            if hf_token != "":
                headers["Authorization"] = f"Bearer {hf_token}"

            response = requests.get(url, headers=headers)
            # Interpret the bytes as a little-endian unsigned 64-bit integer
            length_of_header = struct.unpack('<Q', response.content)[0]
            # Fetch length_of_header bytes starting from the 9th byte
            headers['Range'] = f'bytes=8-{7 + length_of_header}'
            response = requests.get(url, headers=headers)
            # Interpret the response as a JSON object
            return response.json()
        except Exception as e:
            log.error(e)
            try:
                url = url + ".index.json"
                response = requests.get(url)
                return response.json() 
            except:
                return None
    
    f_mode = "r+"
    if not os.path.exists("metadata_store.json"):
        f_mode = "w+"

    with open("metadata_store.json", f_mode) as f:
        data = {}
        if f_mode == "r+":
            data = json.loads(f.read())
        # log.debug(f"metadata_store.json: {data}")
        if url not in data.keys():
            meta = download(url)
            if meta != None:
                fn_meta = f"{get_model_filename()}.json"
                data[url] = fn_meta
                f.write(json.dumps(data))
                if not os.path.exists(meta_folder):
                    os.makedirs(meta_folder)
                    log.info(f"Created folder: {meta_folder}")
                with open(f"{meta_folder}/{fn_meta}", 'w') as ff:
                    ff.write(json.dumps(meta))
                return meta    

        else:
            with open(f"{meta_folder}/{data[url]}", "r") as mf:
                return json.loads(mf.read())