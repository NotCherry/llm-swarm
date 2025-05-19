
# Start listener thread
import asyncio
import json
import re
import threading
import time
from typing import Dict

from src.model.loading import download_model 
from src import global_vars 
from .. import global_vars
from src.model.model import get_model_metadata, layers_size
from src.structs import DictChecksumTracker, Shard
from src.util import detect_device, get_last_layer_number, separate_nodes, serialize_network_config
from src.local_logger import log

def update_network():
    dev_spec = detect_device()
    connect_msg = {
    "msg": "connect",
    "data": json.dumps({key: spec.model_dump() for key, spec in dev_spec.items()})  # Convert dataclass to dict
    }
    connect_bytes = json.dumps(connect_msg).encode('utf-8')

    master_node_msg = {
        "msg": "MASTERNODE",    
    }
    master_node_bytes = json.dumps(master_node_msg).encode('utf-8')

    def detect():
        log.info(f"Discovering hosts on: {".".join(global_vars.LOCAL_ADDRESS.split(".")[:3])}.0")
        for host in global_vars.SEARCH_IP_RANGE:
            global_vars.SOCK_UDP.sendto(connect_bytes, (host, global_vars.PEER_PORT))
        time.sleep(4)
        log.info("Discovery end hosts found:")
        log.info(global_vars.ACTIVE_HOSTS)    
    
    def broadcast_masternode():
        if global_vars.START_TIME + global_vars.WAIT_TIME < time.perf_counter() and global_vars.MASTER_NODE:
            log.debug("Brodcasting Master Node")
            for host in global_vars.ACTIVE_HOSTS:
                global_vars.SOCK_UDP.sendto(master_node_bytes, (host, global_vars.PEER_PORT))
            global_vars.MASTER_NODE_IP = global_vars.LOCAL_ADDRESS                
            if not global_vars.SHARDING_SERVICE:
                global_vars.SHARDING_SERVICE = True
                threading.Thread(target=plan_network, args=()).start()
            
    
    time.sleep(5)
    time_start = None
    while True:
        time_start = time.perf_counter()
        broadcast_masternode()
        log.info(f"Node Type {'Master' if global_vars.MASTER_NODE else 'Worker'}")
        global_vars.ACTIVE_HOSTS = []
        detect()
        with global_vars.NETWORK_LOCK:
            keys = list(global_vars.NETWORK_TOPOLOGY.nodes.keys())
            for key in keys:
                if key not in global_vars.ACTIVE_HOSTS:
                    del global_vars.NETWORK_TOPOLOGY.nodes[key]
        time.sleep(7 - (time.perf_counter() - time_start) if (time.perf_counter() - time_start) > 0 else 0)


async def shard_planner():    
    while True:
        time.sleep(5)
        current_checksum = DictChecksumTracker(global_vars.NETWORK_TOPOLOGY.nodes)._checksum
        if global_vars.NETWORK_CHECKSUM is not None or global_vars.NETWORK_CHECKSUM == current_checksum:
            return
        if global_vars.NETWORK_TOPOLOGY.loading_model or global_vars.NETWORK_TOPOLOGY.generating:
            return
        
        url = f"https://huggingface.co/{global_vars.SELECTED_MODEL}/resolve/main/model.safetensors"
        metadata = get_model_metadata(url)


        if metadata is None:
            log.info("No Metadata to be found in repo")
            return
        

        nodes_with_gpu, nodes_cpu_only = separate_nodes()

        log.debug("Calculating total memory")
        log.debug(f"GPU nodes: {nodes_with_gpu}")
        log.debug(f"CPU nodes: {nodes_cpu_only}")

        # TODO: asser if current node have moemory to be master node if not set masternode to false and wait for other node to take over
        # log.error("{}, {}, {}".format(global_vars.NETWORK_TOPOLOGY.nodes[local_addres].spec['cpu'].ram, (global_vars.PROGRAM_MINIMAL_SPACE + global_vars.MASTER_NODE_BUFFER), global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].spec['cpu'].ram > (global_vars.PROGRAM_MINIMAL_SPACE + global_vars.MASTER_NODE_BUFFER)))
        if global_vars.MASTER_NODE and (global_vars.PROGRAM_MINIMAL_SPACE + global_vars.MASTER_NODE_BUFFER) > global_vars.NETWORK_TOPOLOGY.nodes[global_vars.LOCAL_ADDRESS].spec['cpu'].ram:
            log.error("Not enough memory available for master node stepping down to worker")
            global_vars.MASTER_NODE = False    
            break

        total_memory = 0
        for (key, v) in nodes_with_gpu:
            total_memory += (v.ram - global_vars.PROGRAM_MINIMAL_SPACE - (global_vars.MASTER_NODE_BUFFER if key == global_vars.LOCAL_ADDRESS and global_vars.MASTER_NODE else 0))
        for (key, v) in nodes_cpu_only:
            log.debug(f"CPU node {key}: {v}")
            total_memory += (v.ram - global_vars.PROGRAM_MINIMAL_SPACE - (global_vars.MASTER_NODE_BUFFER if key == global_vars.LOCAL_ADDRESS and global_vars.MASTER_NODE else 0))

        log.debug(f"Total memory: {total_memory}")
            
        layers_dict = {}

        async def run_it(layers_dict):
            end_layer = max([int(k.split(".")[2]) for k in layers_dict.keys() if "model.layer" in k])
            await plan_network_from_layers(layers_dict,  end_layer)

            NETWORK_CHECKSUM = DictChecksumTracker(global_vars.NETWORK_TOPOLOGY.nodes)._checksum
            # await download_file_with_metadata(url=url, hf_token=os.getenv('HF_TOKEN'))
            log.info("Loading of the models begins")
            await download_model()

        if isinstance(metadata, list) or 'format' in metadata['__metadata__'].keys():
            del metadata["__metadata__"]
            
            # add lm_head does not exist in smaller model account for loading embed layer as it 
            if "output" not in metadata.keys() and "lm_head" not in metadata.keys():
                metadata["lm_head.weight"] = metadata['model.embed_tokens.weight']
            layers_dict = layers_size(metadata)
            del metadata["lm_head.weight"]


            model_size = sum([ v for k, v in layers_dict.items()])
            log.debug(f"Model size: {model_size}")

            if model_size > total_memory:
                log.error("It is never enough MEMEORY!!!!")
                log.error(f"Model size: {model_size} -  Available: {total_memory}")
                return
            
            await run_it(layers_dict)
            return
            
        if metadata['metadata']['total_size'] > total_memory:
            log.info("It is never enough MEMEORY!!!!")
            return
        
        if metadata['metadata']['total_size'] is not None:
            files = set([v for key, v in metadata['weight_map']])

            for fn in files:
                url = "/".join(url.split("/")[:-1]) + "/" + fn
                fn_metadat = get_model_metadata(url)
                del fn_metadat['__metadata__']
                layers_dict.update(fn_metadat)
            layers_dict = layers_size(layers_dict)
            await run_it(layers_dict)


async def plan_network_from_layers(layer_dict: Dict[str, int], n_layers):
    # Get model metadata and set buffer
    url = f"https://huggingface.co/{global_vars.SELECTED_MODEL}/resolve/main/model.safetensors"
    metadata = get_model_metadata(url)
    last_layer_number = get_last_layer_number(metadata)
    global_vars.MASTER_NODE_BUFFER = max(layers_size(metadata).values())

    # Remove unnecessary layers
    for key in ["lm_head.weight", "model.norm.weight"]:
        layer_dict.pop(key, None)

    nodes_with_gpu, nodes_cpu_only = separate_nodes()
    current_layer = start_layer = 0
    last_node_ip = None

    def process_nodes(nodes):
        nonlocal current_layer, start_layer, last_node_ip
        for node_id, spec in nodes:
            if not layer_dict:
                break

            # Check if node has enough memory
            if spec.ram <= global_vars.PROGRAM_MINIMAL_SPACE:
                continue
            if node_id == global_vars.LOCAL_ADDRESS and global_vars.MASTER_NODE and spec.ram < global_vars.MASTER_NODE_BUFFER + global_vars.PROGRAM_MINIMAL_SPACE:
                continue

            mem = spec.ram - (global_vars.MASTER_NODE_BUFFER + global_vars.PROGRAM_MINIMAL_SPACE if node_id == global_vars.LOCAL_ADDRESS else 0)

            # Handle embedding layer
            if current_layer == 0 and layer_dict.get("model.embed_tokens.weight", 0) <= mem:
                mem -= layer_dict.pop("model.embed_tokens.weight")

            # Process model layers
            keys_to_remove = []
            sorted_layers = sorted(layer_dict.items(), key=lambda x: int(x[0].split(".")[2]))
            for key, size in sorted_layers:
                if "layer" not in key:
                    continue
                if f"layer.{last_layer_number}" in key and \
                   mem - size - sum(layer_dict.get(k, 0) for k in ["lm_head.weight", "model.norm.weight"]) <= 0:
                    break
                if mem > size:
                    current_layer += 1
                    keys_to_remove.append(key)
                    mem -= size
                else:
                    break

            # Create shard if layers were assigned
            if current_layer > start_layer:
                if last_node_ip:
                    global_vars.NETWORK_TOPOLOGY.nodes[node_id].next_node_ip = last_node_ip
                shard = Shard(global_vars.SELECTED_MODEL, start_layer, current_layer-1, n_layers + 1, False)
                global_vars.NETWORK_TOPOLOGY.nodes[node_id].shard = shard
                start_layer = current_layer
                last_node_ip = node_id

            # Remove processed layers
            for key in keys_to_remove:
                layer_dict.pop(key)

    # Process GPU and CPU nodes
    process_nodes(nodes_with_gpu)
    process_nodes(nodes_cpu_only)

    # Send network configuration
    net_config = {"msg": "net_config", "data": serialize_network_config(global_vars.NETWORK_TOPOLOGY)}
    net_config_bytes = json.dumps(net_config).encode('utf-8')
    
    log.debug("Network planning complete")
    for node_id, node in global_vars.NETWORK_TOPOLOGY.nodes.items():
        global_vars.SOCK_UDP.sendto(net_config_bytes, (node.ip, global_vars.PEER_PORT))
        log.info(f"Node: {node_id} - {node.shard}")



def plan_network():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(shard_planner())
    finally:
        loop.close()

