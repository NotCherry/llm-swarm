
import json
import os
from typing import Dict

from websockets import connect

from src.model.model import init_model
from src.model.saving import save_local_layers
from src.ptcode import safe_load_by_layer, safe_load_layer
from src.util import get_model_filename
from src.local_logger import log
from src import global_vars

async def broadcast_layer_to_node(node, layer_data: Dict[str, bytes]):
    state_dict = {}
    for layer_name, layer_data in layer_data.items():
        state_dict[layer_name] = safe_load_layer(layer_name, layer_dtype=layer_data['info']['dtype'], layer_shape=layer_data['info']['shape'], data=layer_data['data'])[layer_name]
    
    log.warning(f"{node} - {global_vars.LOCAL_ADDRESS}")
    log.info(f"Loading Layers to model {state_dict.keys()}")    
    if node != global_vars.LOCAL_ADDRESS:
        await brodcast_data_to_node(global_vars.NETWORK_TOPOLOGY.nodes[node].ip, {"type":"layer_data", "data":state_dict})
        return
    if global_vars.MODEL is None: 
        init_model()

    save_local_layers(state_dict)

async def request_layer_from_node(target_node, layer_name):
    await brodcast_data_to_node(global_vars.NETWORK_TOPOLOGY.nodes[target_node].ip, {"type":"layer_request", "data": layer_name})
    

def send_requested_layer(target_node, layer_name):
    if layer_name in global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].loaded_layers:
       broadcast_layer_to_node(target_node, global_vars.MODEL.model.state_dict()[layer_name])

    fn = f"{get_model_filename()}.safetensors"
    
    if not os.path.exists(fn):
        log.error("Config is invalig brodcasting sync request")
        # TODO: send brodcast
        return

    weights = safe_load_by_layer(fn, layer_prefix=layer_name)
    broadcast_layer_to_node(target_node, weights)

async def brodcast_data_to_node(node_ip, data):
    async with connect(f"ws://{node_ip}:{global_vars.WS_PORT}") as websocket:
        await websocket.send(json.dumps(data))
