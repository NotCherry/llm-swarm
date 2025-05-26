import asyncio
import json
import select
import socket
import time
from typing import List, Tuple
from websockets import serve

from src.network.process_info import process_info, save_local_layers
from src import global_vars


from src.ptcode import model_generate_text
from src.structs import DeviceSpec, NetworkMsg, Node
from src.local_logger import log

def listen(sock):    
    while True:
        data, addr = sock.recvfrom(10*1024)
        data:NetworkMsg = json.loads(data.decode('utf-8'))
        
        if data['msg']  == "connect":
            data_dict = json.loads(data["data"])

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
        if data['msg'] == "MASTERNODE" and ((global_vars.START_TIME + global_vars.WAIT_TIME) > time.perf_counter()):
            log.info(f"Master node address: {addr[0]}")
            global_vars.MASTER_NODE = False
            global_vars.MASTER_NODE_IP = addr[0]

async def comunicate(websocket):
    client_ip, client_port = websocket.remote_address
    async for message in websocket:
        if type(message) != dict:
            message = json.loads(message)
        process_info(message, client_ip)
        

async def run_ws_server():
    async with serve(comunicate, global_vars.LISTEN_IP, global_vars.WS_PORT) as server:
        global_vars.WSS_SERVER_READY.set()
        await server.serve_forever()

def wsserver():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_ws_server())
    loop.close()