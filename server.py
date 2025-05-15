import json
import socket
import struct
import threading
import time
import asyncio
from typing import Dict, List, Literal, Optional, Union
import time
from dataclasses import asdict, dataclass 
import requests
from ptcode import Shard, build_transformer, safe_load_layer, LlamaModel, model_generate_text
import psutil
import torch
from benchmarks import get_flops
from util import log
from tqdm.rich import tqdm
import os
from websockets.asyncio.server import serve
from websockets.asyncio.client import connect
import hashlib
from pydantic import BaseModel
from dotenv import load_dotenv
from safetensors.torch import save_file
from accelerate import init_empty_weights
import re
load_dotenv()

from util import SELECTED_MODEL

def get_outbound_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to actually send data; just triggers routing logic
        s.connect(('1.1.1.1', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = None
    finally:
        s.close()

    log.info(f"IP with access to internet: {ip}")    
    return ip


LISTEN_IP = '0.0.0.0'     # Listen on all interfaces
LISTEN_PORT = 5005        # Port to listen on
WS_PORT = 8543

local_address = get_outbound_ip()
PROGRAM_MINIMAL_SPACE = (500 * 1024 * 1024)
MASTER_NODE_BUFFER = (600 * 1024 * 1024)

SEARCH_IP_RANGE = [f"{".".join(local_address.split(".")[:3])}.{i}" for i in range(1,254) ] # Change to the peer's IP
PEER_PORT = 5005          # Change to the peer's port

ACTIVE_HOSTS = []
MASTER_NODE = True
MASTER_NODE_IP: str = None


START_TIME = time.perf_counter()
WAIT_TIME = 10

SHARDING_SERVICE = False
METADATA_STORE: Dict[str, str] = {}

MODEL: LlamaModel = None
MODEL_LOCK = threading.Lock()


class DeviceSpec(BaseModel):
    device: Literal["cpu", "cuda"]
    name: str
    ram: int
    tflops: float 


class Node(BaseModel):
    ip: str
    spec: Dict[str,DeviceSpec]
    shard: Optional[Shard] = None
    next_node_ip: Optional[str] = None
    def to_tuple(self):
        """Convert Node to a tuple for checksum computation."""
        return (self.ip, self.spec)

@dataclass
class NetworkConfig():
    nodes: Dict[str, Node]
    loading_model: bool = False
    loaded_model: bool = False
    generating: bool = False


NETWORK_TOPOLOGY = NetworkConfig(nodes={})
NETWORK_LOCK = threading.Lock()
NETWORK_CHECKSUM = None

def serialize_network_config(config: NetworkConfig) -> str:
    # Convert NetworkConfig to a dict, ensuring Pydantic models are serialized
    config_dict = {
        "nodes": {key: node.model_dump() for key, node in config.nodes.items()}
    }
    # Serialize to JSON string
    return json.dumps(config_dict)

# Function to deserialize JSON string back to NetworkConfig
def deserialize_network_config(json_str: str) -> NetworkConfig:
    # Parse JSON string to dict
    config_dict = json.loads(json_str)
    # Convert node dicts back to Node objects
    nodes = {key: Node(**node_data) for key, node_data in config_dict["nodes"].items()}
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

@dataclass
class UDPMsg:
    msg: str
    data: Dict

def listen(sock):
    global ACTIVE_HOSTS, START_TIME, MASTER_NODE, MASTER_NODE_IP, NETWORK_TOPOLOGY
    
    while True:
        data, addr = sock.recvfrom(1024)
        data:UDPMsg = json.loads(data.decode('utf-8'))
        
        if data['msg']  == "connect":
            # spec =  DeviceSpec(**json.loads(data['data']))
            data_dict = json.loads(data["data"])

            # Step 2: Reconstruct DeviceSpec objects
            spec: dict[str, DeviceSpec] = {
                key: DeviceSpec.model_validate(value) for key, value in data_dict.items()
            }
            if addr[0] not in ACTIVE_HOSTS:
                ACTIVE_HOSTS.append(addr[0])
            if addr[0] not in NETWORK_TOPOLOGY.nodes.keys():
                NETWORK_TOPOLOGY.nodes[addr[0]] = Node(ip=addr[0], spec=spec)
                log.error(f"Node INIT {NETWORK_TOPOLOGY.nodes[addr[0]]}")
            elif spec !=  NETWORK_TOPOLOGY.nodes[addr[0]].spec:
                NETWORK_TOPOLOGY.nodes[addr[0]].spec = spec
        if data['msg'] == "MASTERNODE" and ((START_TIME + WAIT_TIME) > time.perf_counter()):
            log.info(f"Master node address: {addr[0]}")
            # maby we can just append it to the node struct?
            MASTER_NODE = False
            MASTER_NODE_IP = addr[0]
            # TODO Handle case when masternode Disapear
        if data['msg']  == "net_config":
            with NETWORK_LOCK:
                NETWORK_TOPOLOGY.nodes =  deserialize_network_config(data["data"]).nodes
                log.error(f"config recived {NETWORK_TOPOLOGY.nodes}")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)
sock.bind((LISTEN_IP, LISTEN_PORT))

class DictChecksumTracker:
    def __init__(self, dictionary: Dict[str, Node]):
        self.dictionary = dictionary
        self._checksum = self._compute_checksum()

    def _compute_checksum(self) -> str:
        """Compute MD5 checksum of the dictionary."""
        # Sort keys to ensure consistent hashing
        sorted_items = sorted(
            (key, node.to_tuple()) for key, node in self.dictionary.items()
        )
        # Convert to a stable string representation
        data = str(sorted_items).encode()
        return hashlib.md5(data).hexdigest()

    def has_changed(self) -> bool:
        """Check if the dictionary has changed."""
        return self._compute_checksum() != self._checksum

    def reset(self):
        """Update the stored checksum to the current state."""
        self._checksum = self._compute_checksum()

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
        
def separate_nodes():
    global NETWORK_TOPOLOGY
    # Nodes with GPU devices
    nodes_with_gpu: Dict[str, DeviceSpec] = {
        k: v.spec['gpu'] for k, v in NETWORK_TOPOLOGY.nodes.items() if "gpu" in v.spec
    }
    
    # Nodes with CPU devices but no GPU
    nodes_cpu_only: Dict[str, DeviceSpec] = {
        k: v.spec['cpu'] for k, v in NETWORK_TOPOLOGY.nodes.items() if "cpu" in v.spec and "gpu" not in v.spec
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


async def plan_network_from_layers(layer_dict: Dict[str, int], n_layers):
    global NETWORK_TOPOLOGY, local_address, MASTER_NODE_BUFFER

    # Get model metadata and set buffer
    url = f"https://huggingface.co/{SELECTED_MODEL}/resolve/main/model.safetensors"
    metadata = get_model_metadata(url)
    last_layer_number = get_last_layer_number(metadata)
    MASTER_NODE_BUFFER = max(layers_size(metadata).values())

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
            if spec.ram <= PROGRAM_MINIMAL_SPACE:
                continue
            if node_id == local_address and MASTER_NODE and spec.ram < MASTER_NODE_BUFFER + PROGRAM_MINIMAL_SPACE:
                continue

            mem = spec.ram - (MASTER_NODE_BUFFER + PROGRAM_MINIMAL_SPACE if node_id == local_address else 0)

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
                    NETWORK_TOPOLOGY.nodes[node_id].next_node_ip = last_node_ip
                shard = Shard(SELECTED_MODEL, start_layer, current_layer-1, n_layers + 1, False)
                NETWORK_TOPOLOGY.nodes[node_id].shard = shard
                start_layer = current_layer
                last_node_ip = node_id

            # Remove processed layers
            for key in keys_to_remove:
                layer_dict.pop(key)

    # Process GPU and CPU nodes
    process_nodes(nodes_with_gpu)
    process_nodes(nodes_cpu_only)

    # Send network configuration
    net_config = {"msg": "net_config", "data": serialize_network_config(NETWORK_TOPOLOGY)}
    net_config_bytes = json.dumps(net_config).encode('utf-8')
    
    log.debug("Network planning complete")
    for node_id, node in NETWORK_TOPOLOGY.nodes.items():
        sock.sendto(net_config_bytes, (node.ip, PEER_PORT))
        log.info(f"Node: {node_id} - {node.shard}")

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
                fn_meta = f"{SELECTED_MODEL.replace("/","-")}.json"
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

def plan_network():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(shard_planner())
    finally:
        loop.close()

async def shard_planner():
    global NETWORK_TOPOLOGY, SELECTED_MODEL, NETWORK_TOPOLOGY, NETWORK_CHECKSUM
    
    while True:
        time.sleep(5)
        current_checksum = DictChecksumTracker(NETWORK_TOPOLOGY.nodes)._checksum
        if NETWORK_CHECKSUM is not None or NETWORK_CHECKSUM == current_checksum:
            return
        if NETWORK_TOPOLOGY.loading_model or NETWORK_TOPOLOGY.generating:
            return
        
        url = f"https://huggingface.co/{SELECTED_MODEL}/resolve/main/model.safetensors"
        metadata = get_model_metadata(url)


        if metadata is None:
            log.info("No Metadata to be found in repo")
            return
        

        nodes_with_gpu, nodes_cpu_only = separate_nodes()

        log.debug("Calculating total memory")
        log.debug(f"GPU nodes: {nodes_with_gpu}")
        log.debug(f"CPU nodes: {nodes_cpu_only}")

        total_memory = 0
        for (key, v) in nodes_with_gpu:
            total_memory += (v.ram - PROGRAM_MINIMAL_SPACE - (MASTER_NODE_BUFFER if key == local_address and MASTER_NODE else 0))
        for (key, v) in nodes_cpu_only:
            log.debug(f"CPU node {key}: {v}")
            total_memory += (v.ram - PROGRAM_MINIMAL_SPACE - (MASTER_NODE_BUFFER if key == local_address and MASTER_NODE else 0))

        log.debug(f"Total memory: {total_memory}")
            
        layers_dict = {}

        async def run_it(layers_dict):
            global NETWORK_CHECKSUM

            end_layer = max([int(k.split(".")[2]) for k in layers_dict.keys() if "model.layer" in k])
            await plan_network_from_layers(layers_dict,  end_layer)

            NETWORK_CHECKSUM = DictChecksumTracker(NETWORK_TOPOLOGY.nodes)._checksum
            await download_file_with_metadata(url=url, hf_token=os.getenv('HF_TOKEN'))

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


# Start listener thread
def update_network():
    global ACTIVE_HOSTS, MASTER_NODE, SHARDING_SERVICE, NETWORK_TOPOLOGY

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
        log.info(f"Discovering hosts on: {local_address}.0")
        for host in SEARCH_IP_RANGE:
            sock.sendto(connect_bytes, (host, PEER_PORT))
        time.sleep(4)
        log.info("Discovery end hosts found:")
        log.info(ACTIVE_HOSTS)    
    
    def master_node():
        global SHARDING_SERVICE, MASTER_NODE_IP
        if START_TIME + WAIT_TIME < time.perf_counter() and MASTER_NODE:
            log.debug("Brodcasting Master Node")
            for host in ACTIVE_HOSTS:
                sock.sendto(master_node_bytes, (host, PEER_PORT))
            MASTER_NODE_IP = local_address                
            if not SHARDING_SERVICE:
                SHARDING_SERVICE = True
    
    time.sleep(5)
    time_start = None
    while True:
        time_start = time.perf_counter()
        master_node()
        log.info(f"Node Type {'Master' if MASTER_NODE else 'Worker'}")
        ACTIVE_HOSTS = []
        detect()
        with NETWORK_LOCK:
            keys = list(NETWORK_TOPOLOGY.nodes.keys())
            for key in keys:
                if key not in ACTIVE_HOSTS:
                    del NETWORK_TOPOLOGY.nodes[key]
        time.sleep(7 - (time.perf_counter() - time_start) if (time.perf_counter() - time_start) > 0 else 0)

def convert_and_sort_by_offset(metadata):
    # Convert dict to list of (key, value) tuples
    if '__metadata__' in metadata.keys(): del metadata['__metadata__']
    items = [(key, value) for key, value in metadata.items()]
    
    # Sort by the first element of data_offsets
    sorted_items = sorted(items, key=lambda x: x[1]["data_offsets"][0])
    
    return sorted_items

def init_model():
    global MODEL, MODEL_LOCK, NETWORK_TOPOLOGY, local_address
    shard: Shard = NETWORK_TOPOLOGY.nodes[local_address].shard
    with MODEL_LOCK:
        if "gpu" in NETWORK_TOPOLOGY.nodes[local_address].spec.keys():
            with init_empty_weights():
                MODEL = LlamaModel(shard)
            MODEL.to_empty(device = torch.device("cuda"))
        else:
            MODEL = LlamaModel(shard)
        

def find_key_of_node_with_layer(topology: NetworkConfig, layer: int) -> Optional[str]:
    return next((key for key, node in topology.nodes.items() if layer >= node.shard.start_layer and  layer <= node.shard.end_layer ), None)

async def brodcast_data_to_node(node_ip, data):
    async with connect(f"ws://{node_ip}:{WS_PORT}") as websocket:
        await websocket.send(json.dumps(data))
    

async def broadcast_layer_to_node(node, layer_data: Dict[str, bytes]):
    global NETWORK_TOPOLOGY, local_address, MODEL, MODEL_LOCK
    state_dict = {}
    for layer_name, layer_data in layer_data.items():
        state_dict[layer_name] = safe_load_layer(layer_name, layer_dtype=layer_data['info']['dtype'], layer_shape=layer_data['info']['shape'], data=layer_data['data'])[layer_name]
    
    log.warning(f"{node} - {local_address}")
    log.info(f"Loading Layers to model {state_dict.keys()}")    
    if node != local_address:
        await brodcast_data_to_node(NETWORK_TOPOLOGY.nodes[node].ip, {"type":"layer_data", "data":state_dict})
        return
    if MODEL is None: 
        init_model()

    with MODEL_LOCK:
        MODEL.load_state_dict(state_dict, strict=False)


async def download_file_with_metadata(url, hf_token=None, chunk_size=1024*1024*10):
    global NETWORK_TOPOLOGY, local_address

    node_key = find_key_of_node_with_layer(NETWORK_TOPOLOGY, 0)
    
    assert node_key == local_address

    NETWORK_TOPOLOGY.loading_model = True
    NETWORK_TOPOLOGY.loaded_model = False
    # Set headers with Hugging Face token if provided
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    # Get file size for progress bar (if available)
    response = requests.head(url, headers=headers, allow_redirects=True)
    file_size = int(response.headers.get("content-length", 0))

    # Stream the download
    response = requests.get(url, stream=True, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download: HTTP {response.status_code}")

    # Variables for metadata parsing
    header_size = None
    buffer = bytearray(chunk_size)
    last_buffer_pointer = 0
    buffer_pointer = 0
    data_dict = {}

    metadata = None
    layers_to_dwonload = None
    file_size = os.path.getsize("model.safetensors")

    with tqdm(total=file_size, unit="B") as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # Filter out keep-alive chunks
                pbar.update(len(chunk))

                buffer_pointer += len(chunk)
                buffer[last_buffer_pointer:buffer_pointer] = chunk
                last_buffer_pointer += len(chunk)
                # Accumulate buffer for metadata parsing
                # Read header size (first 8 bytes)
                if header_size is None and len(buffer) >= 8:
                    header_size = struct.unpack("<Q", buffer[:8])[0]

                # Read JSON header once we have enough data
                elif header_size is not None and buffer_pointer >= 8 + header_size and metadata is None:
                    b = buffer[8:8 + header_size]
                    header_data = b.decode()
                    metadata = json.loads(header_data)
                    del metadata["__metadata__"]
                
                    max_layer_size = max([ s for k, s in  layers_size(metadata).items() ])
                    layers_to_dwonload = convert_and_sort_by_offset(metadata)

                    buffer_copy = buffer[8 + header_size:]
                    buffer = bytearray(max_layer_size)
                    buffer[:len(buffer_copy)] = buffer_copy

                    buffer_pointer = len(buffer_copy)
                    last_buffer_pointer = len(buffer_copy)

                elif metadata is not None:
                    # Ensure buffer is a bytearray for mutability
                    if not isinstance(buffer, bytearray):
                        buffer = bytearray(buffer)

                    while layers_to_dwonload:
                        # Calculate layer length
                        layer_info = layers_to_dwonload[0][1]
                        next_layer_lenght = layer_info['data_offsets'][1] - layer_info['data_offsets'][0]
                        
                        # Check if buffer has enough data
                        if buffer_pointer < next_layer_lenght:
                            break

                        # Process layer
                        layer_name = layers_to_dwonload[0][0]
                        node_key = None
                        l_num = None
                        if "model.layers" in layer_name:
                            l_num = int(layer_name.split('.')[2])
                            node_key = find_key_of_node_with_layer(NETWORK_TOPOLOGY, l_num)
                        elif "model.embed_tokens" in layer_name:
                            node_key = find_key_of_node_with_layer(NETWORK_TOPOLOGY, 0)
                        elif "model.norm" in layer_name or "output.weight" in layer_name:
                            last_layer = NETWORK_TOPOLOGY.nodes[find_key_of_node_with_layer(NETWORK_TOPOLOGY, 0)].shard.n_layers - 1
                            node_key = find_key_of_node_with_layer(NETWORK_TOPOLOGY, last_layer)
                        
                        if node_key is None:
                            log.error(f"Failed to find node_key for layer: {layer_name}")
                            break

                        # Extract layer data
                        l_data = buffer[:next_layer_lenght]
                        log.info(f"Downloaded Layer for node: {node_key}, Name: {layer_name}")
                        data_dict[layer_name] = {"data": l_data, "info": layer_info}

                        # Remove processed layer
                        del layers_to_dwonload[0]

                        # Shift buffer left
                        buffer[:last_buffer_pointer - next_layer_lenght] = buffer[next_layer_lenght:last_buffer_pointer]
                        # Zero out the remaining space
                        buffer[last_buffer_pointer - next_layer_lenght:last_buffer_pointer] = bytes(last_buffer_pointer - next_layer_lenght)

                        # Update pointers
                        last_buffer_pointer -= next_layer_lenght
                        buffer_pointer -= next_layer_lenght

                        # Optional: Check if we need to skip further processing
                        layers_number_in_queue = set(int(x[0].split(".")[2]) for x in layers_to_dwonload if "model.layers" in x[0])
                        if l_num is not None and l_num in layers_number_in_queue:
                            continue
                        await broadcast_layer_to_node(node_key, data_dict)
                        if "model.embed_tokens.weight" == layer_name and "output.weight" not in metadata.keys():
                            data_dict['lm_head.weight'] = data_dict['model.embed_tokens.weight']
                            del data_dict['model.embed_tokens.weight']
                            last_layer = NETWORK_TOPOLOGY.nodes[find_key_of_node_with_layer(NETWORK_TOPOLOGY, 0)].shard.n_layers - 1
                            node_key = find_key_of_node_with_layer(NETWORK_TOPOLOGY, last_layer)
                            await broadcast_layer_to_node(node_key, data_dict)

                        data_dict = {}
                        
    NETWORK_TOPOLOGY.loading_model = False                
    NETWORK_TOPOLOGY.loaded_model = True
    log.info(f"Download of model {SELECTED_MODEL} completed")
    


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
    global MODEL, NETWORK_TOPOLOGY, local_address, MASTER_NODE_IP, MODEL_LOCK
    async for message in websocket:
        message = json.loads(message)
        if message['type'] == "layer_data":
            with MODEL_LOCK:
                MODEL.load_state_dict(message['data'], strict=False)
        if message['type'] == "llm-decode":
            log.info(f"Chat output: \n {message['data']}")    
        if message['type'] == "gen":
            ## GEN
            NETWORK_TOPOLOGY.generating = True
            with MODEL_LOCK:
                # MODEL.to(device= "cude" if "gpu" NETWORK_TOPOLOGY.nodes[local_address].spec.keys() else "cpu")
                MODEL.eval()

            n = NETWORK_TOPOLOGY.nodes[local_address] 
            first_layer = n.shard.is_first_layer()
            last_layer = n.shard.is_last_layer()            
            with MODEL_LOCK:
                result = model_generate_text(
                        MODEL, 
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
                NETWORK_TOPOLOGY.generating = False
                return
            
            # the loop not ended run next iter
            # TODO: Limit token output
            await brodcast_data_to_node(node_ip=local_address, data={ "type":"gen", "data": { "prompt" : output }})

            # retun user the respone
            if local_address != MASTER_NODE_IP:
                await brodcast_data_to_node(node_ip=MASTER_NODE_IP, data={"type": "llm-decode", "data": output})
            else:
                log.info(output)

            
            



async def WebSocketDataStream():
    async with serve(comunicate, LISTEN_IP, WS_PORT) as server:
        await server.serve_forever()


def wsserver():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(WebSocketDataStream())
    finally:
        loop.close()

async def swarm_discover(sock):
    global MODEL, MODEL_LOCK, NETWORK_TOPOLOGY, SHARDING_SERVICE

    threading.Thread(target=listen, args=(sock,)).start()
    threading.Thread(target=update_network).start()
    threading.Thread(target=wsserver).start()
    
    
    while True:
        if SHARDING_SERVICE:
            threading.Thread(target=plan_network, args=()).start()
            break
        time.sleep(1)
    while True:
        if MASTER_NODE and NETWORK_TOPOLOGY.loaded_model and not NETWORK_TOPOLOGY.generating:
            await brodcast_data_to_node(node_ip=local_address, data={ "type":"gen", "data": { "prompt" : " Hi my name is bryan" }})
            break
        time.sleep(1)

async def main():
    await swarm_discover(sock)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
