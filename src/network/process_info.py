
from io import BytesIO
import json
import torch
import time
from src.ptcode import model_generate_text, str_to_torch_dtype
from src.local_logger import log
from src import global_vars
from src.util import debug_decorator, deserialize_network_config, get_model_filename

from src.model.model import get_model_metadata, get_selected_model_metadata_from_index
from src.ptcode import safe_load_metadata_single
from src import global_vars
from src.structs import Node  
from src.util import detect_device, get_model_filename
from src.local_logger import log
import safetensors

import json

# @debug_decorator
async def process_info(message, client_ip):
    if type(message) != dict:
        message = json.loads(message)
    if message['type'] == "layer_data":
        state_dict = message['data']
        data_to_model(state_dict)

        await save_local_layers(state_dict)                  
    if message['type'] == "layer_request":
        send_requested_layer(message['data']['target_node', message['data']['layer_name']])
    if message['type'] == "llm-decode":
        log.info(f"Chat output: \n {message['data']}")    
    if message['type'] == "gen":
        ## GEN
        global_vars.NETWORK_TOPOLOGY.generating = True
        with global_vars.MODEL_LOCK:
            dev = "cude" if "gpu" == global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].spec.keys() else "cpu"
            if next(global_vars.MODEL.parameters()).device != dev:
                global_vars.MODEL.to(device=dev )
            global_vars.MODEL.eval()

        while global_vars.LOCAL_ADDRESS not in global_vars.NETWORK_TOPOLOGY.nodes.keys():
            time.sleep(0.001)
        n = global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS] 
        first_layer = n.shard.is_first_layer()
        last_layer = n.shard.is_last_layer()            
        with global_vars.MODEL_LOCK:
            result = model_generate_text(
                    global_vars.MODEL, 
                    input_text=message['data']['prompt'] if first_layer else "",
                    first_layer=first_layer,
                    last_layer=last_layer,
                    h=torch.tensor(message['data']['h']) if not first_layer else None,
                    p_ids=torch.tensor(message['data']['p_ids']) if not first_layer else None,
                    att=torch.tensor(message['data']['att']) if not first_layer else None,
                    )
        if isinstance(result, tuple) and len(result) == 3:
            h, p_ids, att = result
        else:
            output = result

        if not last_layer:
            await broadcast_data_to_node(node_ip=n.next_node_ip, data={ "type":"gen", "data": { "h": h.tolist(), "p_ids": p_ids.tolist(), "att": att.tolist() if att is not None else None}})
            return
        
        # if last layer returned is bool teling if continue
        if isinstance(output, bool) and not output:
            global_vars.NETWORK_TOPOLOGY.generating = False
            return
        
        # the loop not ended run next iter
        # TODO: Limit token output
        await broadcast_data_to_node(node_ip=global_vars.LOCAL_ADDRESS, data={ "type":"gen", "data": { "prompt" : output }})

        # retun user the respone
        if global_vars.LOCAL_ADDRESS != global_vars.MASTER_NODE_IP:
            await broadcast_data_to_node(node_ip=global_vars.MASTER_NODE_IP, data={"type": "llm-decode", "data": output})
        else:
            log.info(output)
    if message['type'] == "shard_update":
        global_vars.NETWORK_TOPOLOGY.nodes[client_ip].shard.loaded = True
    if message['type'] == "node_layer_loaded":
        # FIX ME: No way this works - LOL i start to not remember how this works XD
        log.error(message)
        with global_vars.NETWORK_LOCK:          
            global_vars.NETWORK_TOPOLOGY.nodes[client_ip].loaded_layers.extend(message["data"])
        await bordcast_net_config()

    if message['type'] == "local_layers":
        with global_vars.NETWORK_LOCK:
            model_name = get_model_filename().replace('!', '/')
            log.error(model_name)
            if client_ip not in global_vars.NETWORK_TOPOLOGY.nodes.keys():
                global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS] = Node(ip=global_vars.LOCAL_ADDRESS, spec=detect_device())
            if message["data"] is not None:    
                for k, v in message["data"].items():    
                    if k not in global_vars.NETWORK_TOPOLOGY.nodes[client_ip].saved_layers.keys():
                        global_vars.NETWORK_TOPOLOGY.nodes[client_ip].saved_layers[k] = []
                    if v is not None:                
                        global_vars.NETWORK_TOPOLOGY.nodes[client_ip].saved_layers[k].extend(v)
        await bordcast_net_config()
    if message['type']  == "net_config":
        with global_vars.NETWORK_LOCK:
            global_vars.NETWORK_TOPOLOGY.nodes = deserialize_network_config(message["data"]).nodes

from websockets import connect
import websockets
import json
from src.local_logger import log
import asyncio
from src import global_vars
from src.util import serialize_network_config



async def bordcast_net_config():
    net_config = {"type": "net_config", "data": serialize_network_config(global_vars.NETWORK_TOPOLOGY)}
    for ip in global_vars.ACTIVE_HOSTS:
        await broadcast_data_to_node(ip, net_config)


async def broadcast_data_to_node(node_ip, data):
    if node_ip == global_vars.LOCAL_ADDRESS:
        await process_info(message=data, client_ip=node_ip)
        return
    try:
        con_string = f"ws://{node_ip}:{global_vars.WS_PORT}"
        # log.error(con_string)
        async with connect(con_string) as websocket:
            await websocket.send(json.dumps(data))
            return True
    except (websockets.exceptions.ConnectionClosedError, 
            websockets.exceptions.InvalidURI, 
            asyncio.TimeoutError, 
            ConnectionRefusedError, 
            OSError) as e:
        log.error(f"Failed to connect or send data to {node_ip}: {e}")
        return False 
    

import copy
import json
import os
import re

from src.ptcode import safe_load_metadata_single
from src import global_vars
from src.util import get_model_filename
from src.local_logger import log
import safetensors

def tensor_to_bytes(v, k):
    t = v.cpu() if v.device.type == "cuda" else v
    t = t.contiguous()
    return safetensors.torch._tobytes(t, k)

async def save_local_layers(state_dict):
    with global_vars.MODEL_LOCK:
        node_layer_loaded = {"type": "node_layer_loaded", "data": list(state_dict.keys())}
        await broadcast_data_to_node(global_vars.MASTER_NODE_IP, node_layer_loaded)

        r = [str(x) for x in range(global_vars.MODEL.shard.start_layer, global_vars.MODEL.shard.end_layer + 1)]
        condition1 = all(any(re.search(rf'\.layers\.{layer}\.', s) for s in global_vars.MODEL.loaded_keys) for layer in r)
        condition2 = all( layer in global_vars.MODEL.loaded_keys for layer in  ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"] )
        if condition1 and condition2:
            log.info('Saving node weights')
            fn = f"{get_model_filename()}.safetensors"

            mt = get_selected_model_metadata_from_index()
            if not os.path.exists(fn):
                keys_in_order = global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].loaded_layers
                new_meta = {}
                # get from mt all valid keys
                for k in mt.keys():
                    if k in keys_in_order:
                        new_meta[k] = mt[k]
                dummy_layer = {"dtype": "BF16", "shape": [1000000000000000000], "data_offsets": [1000000000000000000, 1000000000000000000]}
                # Add missing keys from mt to st with the dummy layer to preallocat metadata to match all layers
                for i, key in enumerate(mt.keys()):
                    if key not in new_meta.keys():
                        new_meta[f"dummy_layer_{i}"] = dummy_layer

                # loop over new metadata and rewrite the data offsets
                start_offset = 0
                for i, key in enumerate(keys_in_order):
                    layer_size = new_meta[key]['data_offsets'][1] - new_meta[key]['data_offsets'][0]
                    new_meta[key]['data_offsets'] = [start_offset, start_offset + layer_size]
                    start_offset += layer_size

                with open(fn, "wb") as f:
                    meta_str = json.dumps(new_meta).encode('utf-8')
                    extra = (8 - len(meta_str) % 8) % 8
                    meta_str += b" " * extra
                    size = len(meta_str)
                    f.seek(0)
                    f.write(size.to_bytes(8, byteorder='little', signed=False))
                    f.write(meta_str)
                    for k in keys_in_order:
                        t = tensor_to_bytes(global_vars.MODEL.state_dict()[k].to(dtype=str_to_torch_dtype(new_meta[k]['dtype'])), k)
                        f.write(t)
                    
                return
            

            # metadata_local - local file metadata
            # local metadata has dummy_layers
            f, data_start, local_metadata = safe_load_metadata_single(fn)
            new_local_metadata = copy.deepcopy(local_metadata)
            last_data_offset = [v for k, v in new_local_metadata.items() if "dummy" not in k and "metadata" not in k][-1]
             
            keys_to_add = [key for key in mt if key not in local_metadata]
            if len(keys_to_add) == 0:
                log.info("No new keys to add safetensors file")
                f.close()
                return
            log.info(f"Keys to add: {keys_to_add}")

            # Find dummy keys to remove (up to the number of keys to add)
            keys_to_remove = [key for key in new_local_metadata if "dummy" in key][:len(keys_to_add)]

            # Update new_local_metadata: add new keys, remove dummy keys
            new_local_metadata.update({key: mt[key] for key in keys_to_add})
            for key in keys_to_remove:
                del new_local_metadata[key]

            # Update data offsets of the new keys:
            s = 0
            for k in keys_to_add:
                s = new_local_metadata[k]['data_offsets'][1] - new_local_metadata[k]['data_offsets'][0]
                new_local_metadata[k]['data_offsets'] = [last_data_offset, last_data_offset + s]
                last_data_offset += s


            json_bytes = json.dumps(new_local_metadata).encode('utf-8')
            json_size = len(json_bytes)
            padding_size = data_start - json_size

            if padding_size < 0:
                raise ValueError(f"JSON content is too large to fit in {json_size} bytes without newlines.")

            # Create padding
            padding = b' ' * padding_size

            # Final output
            final_output = json_bytes + padding

            f.seek(8)
            f.write(final_output)
            f.seek(keys_to_add[0]['data_offset'][0])
            for k in keys_to_add:
                t = tensor_to_bytes(global_vars.MODEL.state_dict()[k].to(dtype=str_to_torch_dtype(new_meta[k]['dtype'])), k)
                f.write(t)            
            f.close()

import os
from typing import Dict

from src.model.model import init_model
from src.ptcode import safe_load_by_layer, safe_load_layer
from src.util import get_model_filename
from src import global_vars

async def data_to_model(state_dict):

    if global_vars.MODEL is None: 
        init_model()
    global_vars.MODEL.load_state_dict(state_dict, strict=False)
    log.info(f"Loading layers to model: {state_dict.keys()}")


def is_dict_str_tensor(d):
    if not isinstance(d, dict):
        return False
    return all(isinstance(key, str) and isinstance(value, torch.Tensor) 
               and not isinstance(value, bytes) 
               for key, value in d.items())

# @debug_decorator
async def broadcast_layer_to_node(node, layer_data: Dict[str, bytes]):
    if not is_dict_str_tensor(layer_data):
        state_dict = {}
        for layer_name, layer_info in layer_data.items():
            state_dict[layer_name] = safe_load_layer(layer_name, layer_dtype=layer_info['info']['dtype'], layer_shape=layer_info['info']['shape'], data=layer_info['data'])[layer_name]
            log.error(f"Layer {layer_name} loaded with shape {state_dict[layer_name].shape} and dtype {state_dict[layer_name].dtype}")
    else:
        state_dict = layer_data
    log.warning(f"{node} - {global_vars.LOCAL_ADDRESS}")
    if node != global_vars.LOCAL_ADDRESS:
        await broadcast_data_to_node(global_vars.NETWORK_TOPOLOGY.nodes[node].ip, {"type":"layer_data", "data":state_dict})
        return
   

    await data_to_model(state_dict)


    await save_local_layers(state_dict)

@debug_decorator
async def request_layer_from_node(target_node, dst_node, layer_name):
    await broadcast_data_to_node(global_vars.NETWORK_TOPOLOGY.nodes[target_node].ip, {"type":"layer_request", "data": {"layer_name": layer_name, "target_node": dst_node}})
    
@debug_decorator
def send_requested_layer(target_node, layer_name):
    if layer_name in global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].loaded_layers:
       broadcast_layer_to_node(target_node, global_vars.MODEL.model.state_dict()[layer_name])

    fn = f"{get_model_filename()}.safetensors"
    
    if not os.path.exists(fn):
        log.error("Config is invalig brodcasting sync request")
        return

    weights = safe_load_by_layer(fn, layer_prefix=layer_name)
    broadcast_layer_to_node(target_node, weights)
