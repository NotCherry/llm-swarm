import asyncio
from dataclasses import dataclass
import json
import time
from typing import Dict

import torch
from websockets import serve

from src import global_vars
from src.model.saving import save_local_layers
from src.network.wss import brodcast_data_to_node, send_requested_layer
from src.ptcode import model_generate_text
from src.structs import DeviceSpec, NetworkMsg, Node
from src.util import deserialize_network_config, get_model_filename, serialize_network_config
from src.local_logger import log

def listen(sock):    
    while True:
        data, addr = sock.recvfrom(1024)
        data:NetworkMsg = json.loads(data.decode('utf-8'))
        
        if data['msg']  == "connect":
            # spec =  DeviceSpec(**json.loads(data['data']))
            data_dict = json.loads(data["data"])

            # Step 2: Reconstruct DeviceSpec objects
            spec: dict[str, DeviceSpec] = {
                key: DeviceSpec.model_validate(value) for key, value in data_dict.items()
            }
            if addr[0] not in global_vars.ACTIVE_HOSTS:
                global_vars.ACTIVE_HOSTS.append(addr[0])
            if addr[0] not in global_vars.NETWORK_TOPOLOGY.nodes.keys():
                global_vars.NETWORK_TOPOLOGY.nodes[addr[0]] = Node(ip=addr[0], spec=spec)
                log.error(f"Node INIT {global_vars.NETWORK_TOPOLOGY.nodes[addr[0]]}")

            elif spec !=  global_vars.NETWORK_TOPOLOGY.nodes[addr[0]].spec:
                global_vars.NETWORK_TOPOLOGY.nodes[addr[0]].spec = spec
        if data['msg'] == "shard_update":
            global_vars.NETWORK_TOPOLOGY.nodes[addr[0]].shard.loaded = True
        if data['msg'] == "node_layer_loaded":
            # FIX ME: No way this works - LOL i start to not remember how this works XD
            with global_vars.NETWORK_LOCK:
                if get_model_filename() not in global_vars.NETWORK_TOPOLOGY.nodes[addr[0]].saved_layers.keys():
                    global_vars.NETWORK_TOPOLOGY.nodes[addr[0]].saved_layers = []
                global_vars.NETWORK_TOPOLOGY.nodes[addr[0]].saved_layers[get_model_filename()].extend(json.loads(data["data"]))            
                global_vars.NETWORK_TOPOLOGY.nodes[addr[0]].loaded_layers.extend(json.loads(data["data"]))
                net_config = {"msg": "net_config", "data": serialize_network_config(global_vars.NETWORK_TOPOLOGY)}
                net_config_bytes = json.dumps(net_config).encode('utf-8')

                for node_id, node in global_vars.NETWORK_TOPOLOGY.nodes.items():
                    sock.sendto(net_config_bytes, (node.ip, global_vars.PEER_PORT))
                    log.info(f"Node: {node_id} - {node.shard}")
        if data['msg'] == "MASTERNODE" and ((global_vars.START_TIME + global_vars.WAIT_TIME) > time.perf_counter()):
            log.info(f"Master node address: {addr[0]}")
            # maby we can just append it to the node struct?
            global_vars.MASTER_NODE = False
            global_vars.MASTER_NODE_IP = addr[0]
            # TODO Handle case when masternode Disapear
        if data['msg']  == "net_config":
            with global_vars.NETWORK_LOCK:
                global_vars.NETWORK_TOPOLOGY.nodes = deserialize_network_config(data["data"]).nodes
                log.error(f"config recived {global_vars.NETWORK_TOPOLOGY.nodes}")

                with open("net.cfg", "w") as f:
                    f.write(serialize_network_config(global_vars.NETWORK_TOPOLOGY))

# Struct like
# type: str
# data: {
# h_state
# att
# pid
# input_prompt
# }
#

async def comunicate(websocket):
    async for message in websocket:
        message = json.loads(message)
        if message['type'] == "layer_data":
            state_dict = message['data']
            save_local_layers(state_dict)                  
        if message['type'] == "layer_request":
            send_requested_layer(message['data'])
        if message['type'] == "llm-decode":
            log.info(f"Chat output: \n {message['data']}")    
        if message['type'] == "gen":
            ## GEN
            global_vars.NETWORK_TOPOLOGY.generating = True
            with global_vars.MODEL_LOCK:
                global_vars.MODEL.to(device= "cude" if "gpu" == global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].spec.keys() else "cpu")
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
                await brodcast_data_to_node(node_ip=n.next_node_ip, data={ "type":"gen", "data": { "h": h.tolist(), "p_ids": p_ids.tolist(), "att": att.tolist() if att is not None else None}})
                return
            
            # if last layer returned is bool teling if continue
            if isinstance(output, bool) and not output:
                global_vars.NETWORK_TOPOLOGY.generating = False
                return
            
            # the loop not ended run next iter
            # TODO: Limit token output
            await brodcast_data_to_node(node_ip=global_vars.LOCAL_ADDRESS, data={ "type":"gen", "data": { "prompt" : output }})

            # retun user the respone
            if global_vars.LOCAL_ADDRESS != global_vars.MASTER_NODE_IP:
                await brodcast_data_to_node(node_ip=global_vars.MASTER_NODE_IP, data={"type": "llm-decode", "data": output})
            else:
                log.info(output)


async def run_ws_server():
    async with serve(comunicate, global_vars.LISTEN_IP, global_vars.WS_PORT) as server:
        await server.serve_forever()

def wsserver():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_ws_server())
    loop.close()