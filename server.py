import json
import socket
import struct
import threading
from multiprocessing import Process
import time
import asyncio
from typing import Dict, List, Literal, Optional
import time
from dataclasses import asdict, dataclass 
import requests
import io
from ptcode import Shard, build_transformer, safe_load_layer, LlamaModel, model_generate_text
import psutil
import torch
from benchmarks import get_flops
from util import log
from tqdm.rich import tqdm
import pickle
import os
from websockets.asyncio.server import serve
from websockets.asyncio.client import connect
import hashlib
from pydantic import BaseModel

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

SEARCH_IP_RANGE = [f"{".".join(local_address.split(".")[:3])}.{i}" for i in range(1,254) ] # Change to the peer's IP
PEER_PORT = 5005          # Change to the peer's port

ACTIVE_HOSTS = []
MASTER_NODE = True
MASTER_NODE_IP: str = None
SELECTED_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

START_TIME = time.perf_counter()
WAIT_TIME = 10

SHARDING_SERVICE = False
METADATA_STORE: Dict[str, str] = {}


MODEL: LlamaModel = None

class DeviceSpec(BaseModel):
    device: Literal["CPU", "GPU"]
    name: str
    ram: int
    ram_type: Literal["RAM", "VRAM"]
    tflops: float 


class Node(BaseModel):
    ip: str
    spec: DeviceSpec
    shard: Optional[Shard] = None
    next_node_ip: Optional[str] = None
    def to_tuple(self):
        """Convert Node to a tuple for checksum computation."""
        return (self.ip, self.spec)

@dataclass
class NetworkConfig():
    nodes: Dict[str, Node]
    lading_model: bool = False
    loaded_model: bool = False


NETWORK_TOPOLOGY = NetworkConfig(nodes={})
network_topology_lock = threading.Lock()
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
    if torch.cuda.is_available() and torch.backends.cudnn.enabled:
        return DeviceSpec(
            device="GPU",
            name=torch.cuda.get_device_name(torch.cuda.current_device()),
            ram=torch.cuda.mem_get_info()[0],
            ram_type="VRAM",
            tflops=get_flops()
        )
    
    memory = psutil.virtual_memory()
    return DeviceSpec(
        device="CPU",
        name="CPU",
        ram=memory.available,
        ram_type="RAM",
        tflops=get_flops()[0]
    )

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
            spec = DeviceSpec(**json.loads(data['data']))
            if addr[0] not in ACTIVE_HOSTS:
                ACTIVE_HOSTS.append(addr[0])
            if addr[0] not in NETWORK_TOPOLOGY.nodes.keys():
                NETWORK_TOPOLOGY.nodes[addr[0]] = Node(ip=addr[0], spec=spec)
            elif spec !=  NETWORK_TOPOLOGY.nodes[addr[0]].spec:
                NETWORK_TOPOLOGY.nodes[addr[0]].spec = spec
        if data['msg'] == "MASTERNODE" and ((START_TIME + WAIT_TIME) > time.perf_counter()):
            log.info(f"Master node address: {addr[0]}")
            # maby we can just append it to the node struct?
            MASTER_NODE = False
            MASTER_NODE_IP = addr[0]
            # TODO Handle case when masternode Disapear
        if data['msg']  == "net_config":
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

def layers_size(metadata) -> Dict[str, int]:
    d = {}
    # TODO: sum layer begining with model.layers.{i}
    layer_size = 0
    layer_number = 0

    if isinstance(metadata, list):
        for layer in metadata:
            s = layer[1]['data_offsets'][1] - layer[1]['data_offsets'][0]
            if "model.layers" in layer[0]:
                l_number = int(layer[1].split('.')[2])
                if layer_number != l_number:
                    d[f"model.layers.{l_number}"] = layer_size
                    layer_size = 0
                    layer_number = l_number
                else:
                    layer_size += s
                continue
            d[layer[0]] = s             
        return d
    else:
        for key, layer in metadata.items():

            s = layer['data_offsets'][1] - layer['data_offsets'][0]
            if "model.layers" in key:
                l_number = int(key.split('.')[2])
                if layer_number != l_number:
                    d[f"model.layers.{l_number}"] = layer_size
                    layer_size = 0
                    layer_number = l_number
                else:
                    layer_size += s
                continue
            d[key] = s         
        return d
        

async def plan_network_from_layers(layer_dict: Dict[str, int], n_layers):
    global NETWORK_TOPOLOGY
    nodes = sorted(NETWORK_TOPOLOGY.nodes.items(), key=lambda x: x[1].spec.ram)
    # n_layers = len(set([ l.split(".")[2] for l in layer_dict.keys() if "layer"]))
    start_layer = 0
    current_layer = 0

    # if the network has capasity to load the entire model then te last layer do not need to be accounted
    if "lm_head.weight" in layer_dict.keys(): del layer_dict['lm_head.weight']
    if "model.norm.weight" in layer_dict.keys(): del layer_dict['model.norm.weight']

    last_node_ip = None
    
    for key, node in nodes:
        if len(layer_dict.keys()) <= 0:
            break
        
        mem = node.spec.ram
        if current_layer == 0:
            mem -= layer_dict['model.embed_tokens.weight']
            del layer_dict['model.embed_tokens.weight']
        
        keys_to_remove = []
        log.debug(f"Node memory:{mem}")
        sorted_layers = sorted(layer_dict.items(), key=lambda x: int(x[0].split(".")[2]))
        for k, layer_size in sorted_layers:
            if "layer" in k and mem - layer_size > 0:
                current_layer += 1
                keys_to_remove.append(k)
            else:
                # Create shard for previous layers
                if last_node_ip != None:
                     NETWORK_TOPOLOGY.nodes[key].next_node_ip = last_node_ip
                shard = Shard(SELECTED_MODEL, start_layer, current_layer, n_layers, False)
                NETWORK_TOPOLOGY.nodes[key].shard = shard
                start_layer = current_layer
                last_node_ip = key
                break

        # Handle case where all layers fit in memory
        if current_layer > start_layer and sorted_layers and "layer" in sorted_layers[-1][0]:
            shard = Shard(SELECTED_MODEL, start_layer, current_layer, n_layers, False)
            NETWORK_TOPOLOGY.nodes[key].shard = shard

        for k in keys_to_remove:
            del layer_dict[k]

    net_config_msg = {
        "msg": "net_config",
        "data": serialize_network_config(NETWORK_TOPOLOGY)
    }
    net_config_bytes = json.dumps(net_config_msg).encode('utf-8')
        
    log.debug("Planing network compleate")
    for k, n in NETWORK_TOPOLOGY.nodes.items():
        if n.ip != local_address:
            sock.sendto(net_config_bytes, (n.ip, PEER_PORT))
        log.info(f"Node: {k} - {n.shard}")

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
            print("con", response.content)
            # Interpret the bytes as a little-endian unsigned 64-bit integer
            length_of_header = struct.unpack('<Q', response.content)[0]
            print(length_of_header)
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
                    print(f"Created folder: {meta_folder}")
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
        if NETWORK_TOPOLOGY.lading_model:
            return
        
        url = f"https://huggingface.co/{SELECTED_MODEL}/resolve/main/model.safetensors"
        metadata = get_model_metadata(url)


        if metadata is None:
            log.info("No Metadata to be found in repo")
            return
        
        
        total_memory = sum([ v.spec.ram for key, v in NETWORK_TOPOLOGY.nodes.items() ])
        layers_dict = {}

        async def run_it(layers_dict):
            global NETWORK_CHECKSUM

            # TODO: Reair MAX not working
            end_layer = max([int(k.split(".")[2]) for k in layers_dict.keys() if "model.layer" in k])
            await plan_network_from_layers(layers_dict,  end_layer)

            NETWORK_CHECKSUM = DictChecksumTracker(NETWORK_TOPOLOGY.nodes)._checksum
            # important uncoment after testing
            # await download_file_with_metadata(url=url, hf_token=os.getenv('HF_TOKEN'))

        if isinstance(metadata, list) or 'format' in metadata['__metadata__'].keys():
            del metadata["__metadata__"]
            layers_dict = layers_size(metadata)
            model_size = sum([ v for k, v in layers_dict.items()])
            log.debug(f"Model size: {model_size}")

            if model_size > total_memory:
                log.error("It is never enough MEMEORY!!!!")
                log.error(f"{model_size}-{total_memory}")
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
    "data": dev_spec.model_dump_json()  # Convert dataclass to dict
    }
    connect_bytes = json.dumps(connect_msg).encode('utf-8')

    dev_spec = detect_device()
    master_node_msg = {
        "msg": "MASTERNODE",    
    }
    master_node_bytes = json.dumps(master_node_msg).encode('utf-8')

    def detect():
        log.info(f"Discovering hosts on: {local_address}.0")
        for host in SEARCH_IP_RANGE:
            sock.sendto(connect_bytes, (host, PEER_PORT))
        log.info("Discovery end hosts found:")
        log.info(ACTIVE_HOSTS)    
    
    def master_node():
        global SHARDING_SERVICE
        if START_TIME + WAIT_TIME < time.perf_counter() and MASTER_NODE:
            log.debug("Brodcasting Master Node")
            for host in ACTIVE_HOSTS:
                sock.sendto(master_node_bytes, (host, PEER_PORT))
            if not SHARDING_SERVICE:
                Process(target=plan_network, args=()).start()
                SHARDING_SERVICE = True
    
    time.sleep(5)
    while True:
        master_node()
        log.info(f"Node Type {'Master' if MASTER_NODE else 'Worker'}")
        ACTIVE_HOSTS = []
        NETWORK_TOPOLOGY.nodes = {}
        detect()
        time.sleep(5)

def convert_and_sort_by_offset(metadata):
    # Convert dict to list of (key, value) tuples
    if '__metadata__' in metadata.keys(): del metadata['__metadata__']
    items = [(key, value) for key, value in metadata.items()]
    
    # Sort by the first element of data_offsets
    sorted_items = sorted(items, key=lambda x: x[1]["data_offsets"][0])
    
    return sorted_items

def init_model():
    global MODEL, NETWORK_TOPOLOGY
    shard: Shard = NETWORK_TOPOLOGY.nodes[local_address].shard
    MODEL = LlamaModel(shard)
    

def find_key_of_node_with_layer(topology: NetworkConfig, layer: int) -> Optional[str]:
    return next((key for key, node in topology.nodes.items() if layer >= node.shard.start_layer and  layer <= node.shard.end_layer ), None)

async def brodcast_data_to_node(node_ip, data):
    async with connect(f"ws://{node_ip}:{WS_PORT}") as websocket:
        await websocket.send(json.dumps(data))
    

async def broadcast_layer_to_node(node, layer_data: Dict[str, bytes]):
    global NETWORK_TOPOLOGY, local_address, MODEL
    state_dict = {}
    for layer_name, layer_data in layer_data.items():
        state_dict = safe_load_layer(layer_name, layer_dtype=layer_data['info']['dtype'], layer_shape=layer_data['info']['shape'], data=layer_data['data'])
    
    log.warning(f"{node} - {local_address}")
    log.info(f"Loading Layers to model {state_dict.keys()}")    
    if node != local_address:
        await brodcast_data_to_node(NETWORK_TOPOLOGY.nodes[node].ip, {"type":"layer_data", "data":state_dict})
        return
    if MODEL is None: 
        init_model()
    MODEL.load_state_dict(state_dict, strict=False)

# co wtedy gdy podczas pobierania dla danej konfiguracji node zostanie wyłączony
# co gdy pojawią się nowę podczas pobierania ? dodatakowo przy wspieraniu FSDP?

async def download_file_with_metadata(url, hf_token=None, chunk_size=1024*1024*10):
    global NETWORK_TOPOLOGY, local_address

    NETWORK_TOPOLOGY.lading_model = True
    NETWORK_TOPOLOGY.loaded_model = False
    # Set headers with Hugging Face token if provided
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    
    # Get file size for progress bar (if available)
    response = requests.head(url, headers=headers)
    file_size = int(response.headers.get("content-length", 0))

    # Stream the download
    response = requests.get(url, stream=True, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download: HTTP {response.status_code}")

    # Variables for metadata parsing
    header_size = None
    buffer = bytearray(40 * 1024)
    last_buffer_pointer = 0
    buffer_pointer = 0
    data_dict = {}

    metadata = None

    layers_to_dwonload = None

    node_key = find_key_of_node_with_layer(NETWORK_TOPOLOGY, 0)
    print(node_key, local_address)
    assert node_key == local_address

    # Open file for writing in binary mode
    with tqdm(total=file_size, unit="B") as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:  # Filter out keep-alive chunks
                # Write chunk to file
                pbar.update(len(chunk))

                buffer_pointer += len(chunk)
                buffer[last_buffer_pointer:buffer_pointer] = chunk
                last_buffer_pointer += len(chunk)
                # Accumulate buffer for metadata parsing
                # Read header size (first 8 bytes)
                if header_size is None and len(buffer) >= 8:
                    header_size = struct.unpack("<Q", buffer[:8])[0]
                    log.debug(header_size)

                # Read JSON header once we have enough data
                elif header_size is not None and buffer_pointer >= 8 + header_size and metadata is None:
                    b = buffer[8:8 + header_size]
                    header_data = b.decode()
                    metadata = json.loads(header_data)
                    del metadata["__metadata__"]
                    max_layer_size = max([ s for k, s in  layers_size(metadata).items() ])
                    layers_to_dwonload = convert_and_sort_by_offset(metadata)

                    buffer = bytearray(max_layer_size)
                    buffer_pointer = 0
                    last_buffer_pointer = 0

                elif metadata is not None:
                    next_layer_lenght = layers_to_dwonload[0][1]['data_offsets'][1] - layers_to_dwonload[0][1]['data_offsets'][0]
                    if buffer_pointer > next_layer_lenght:
                        node_key = None
                        layer_name = layers_to_dwonload[0][0]
                        l_num = None
                        if "model.layers" in layer_name:
                            l_num = int(layer_name.split('.')[2])
                            node_key = find_key_of_node_with_layer(NETWORK_TOPOLOGY, l_num)
                        elif "model.embed_tokens" in layer_name:
                            node_key = find_key_of_node_with_layer(NETWORK_TOPOLOGY, 0)
                        elif "model.norm" in layer_name or "output.weight" in layer_name:
                            last_layer = NETWORK_TOPOLOGY.nodes[find_key_of_node_with_layer(NETWORK_TOPOLOGY, 0)].shard.n_layers - 1
                            node_key = find_key_of_node_with_layer(NETWORK_TOPOLOGY, last_layer)
                        l_data = buffer[:next_layer_lenght]
                        log.info(f"Downloaded Layer for node: {node_key}, Name: {layer_name}")
                        

                        data_dict[layer_name] = { "data": l_data, "info": layers_to_dwonload[0][1] }
                        del layers_to_dwonload[0]
                        layers_number_in_queue = set([int(x[0].split(".")[2]) for x in layers_to_dwonload if "model.layers" in x[0]])
                        if l_num is not None and l_num in layers_number_in_queue:
                            continue

                        await broadcast_layer_to_node(node_key, data_dict)

                        data_dict = {}
                        buffer[:last_buffer_pointer] = bytes(last_buffer_pointer)
                        last_buffer_pointer = 0
                        buffer_pointer = 0
            
    NETWORK_TOPOLOGY.lading_model = False                
    NETWORK_TOPOLOGY.loaded_model = True
    log.info(f"Download of model {SELECTED_MODEL} completed")
    await brodcast_data_to_node(node_ip=local_address, data={ "type":"gen", "data": { "prompt" : " Hi my name is bryan" }})
    


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
    global MODEL, NETWORK_TOPOLOGY, local_address, MASTER_NODE_IP
    async for message in websocket:
        message = json.loads(message)
        if message['type'] == "layer_data":
            MODEL.load_state_dict(message.data, strict=False)
        if message['type'] == "llm-decode":
            log.info(f"Chat output: \n {message['data']}")    
        if message['type'] == "gen":
            ## GEN
            print(NETWORK_TOPOLOGY)
            n = NETWORK_TOPOLOGY.nodes[local_address] 
            first_layer = n.shard.is_first_layer()
            last_layer = n.shard.is_last_layer()

            h, p_ids, att = model_generate_text(
                    MODEL, 
                    input_text=message.data['prompt'] if first_layer else "",
                    first_layer=first_layer,
                    last_layer=last_layer,
                    h=message.data['h'] if not first_layer else None,
                    p_ids=message.data['p_ids'] if not first_layer else None,
                    att=message.data['att'] if not first_layer else None,
                    )
            

            if not last_layer:
                await brodcast_data_to_node(node_ip=local_address, data={ "type":"gen", "data": { "h": h, "p_ids": p_ids, "att": att }})
                return
            
            # if last layer returned is bool teling if continue
            if isinstance(h, bool) and not h:
                return
            
            # retun user the respone
            await brodcast_data_to_node(node_ip=MASTER_NODE_IP, data={"type": "llm-decode", "data": h})
            # the loop not ended run next iter
            # TODO: Limit token output
            await brodcast_data_to_node(node_ip=local_address, data={ "type":"gen", "data": { "prompt" : h }})
            
            
            



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
    global MODEL, NETWORK_TOPOLOGY
    threading.Thread(target=listen, args=(sock,)).start()
    threading.Thread(target=update_network).start()
    threading.Thread(target=wsserver).start()
    
    while True:
        time.sleep(1)
        if local_address in NETWORK_TOPOLOGY.nodes.keys():
            # log.error(f"{NETWORK_TOPOLOGY.nodes[local_address].shard}")
            if NETWORK_TOPOLOGY.nodes[local_address].shard is not None:
                break
    # shard1 = Shard("LLAMA-3.2-1B", 0, 15, 16, True)
    NETWORK_TOPOLOGY.nodes[local_address].shard.loaded = False
    log.error(f'{NETWORK_TOPOLOGY.nodes[local_address].shard}')
    MODEL = build_transformer(".", NETWORK_TOPOLOGY.nodes[local_address].shard)

    await brodcast_data_to_node(node_ip=local_address, data={ "type":"gen", "data": { "prompt" : " Hi my name is bryan" }})


   

async def main():
    await swarm_discover(sock)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
