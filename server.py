import threading
import asyncio
import time
from src.model.loading import verify_local_keys
from src.network.listeners import listen, wsserver
from src.network.process_info import broadcast_data_to_node
from src import global_vars 
from src.scheduler.plan_network import network_service
from src.local_logger import log
from src import global_vars

import os
from dotenv import load_dotenv

from src.structs import Node


load_dotenv()

async def swarm_discover(sock):
    threading.Thread(target=wsserver).start()
    threading.Thread(target=listen, args=(sock,)).start()
    global_vars.WSS_SERVER_READY.wait()
    time.sleep(3)
    await broadcast_data_to_node(global_vars.MASTER_NODE_IP if global_vars.MASTER_NODE_IP != None else global_vars.LOCAL_ADDRESS, {"type": "local_layers" ,"data": verify_local_keys()})

    # Start background threads
    threading.Thread(target=network_service).start()

    # Wait for conditions to broadcast
    while global_vars.MASTER_NODE and not (global_vars.NETWORK_TOPOLOGY.loaded_model and not global_vars.NETWORK_TOPOLOGY.generating):
        await asyncio.sleep(1)
    await broadcast_data_to_node(node_ip=global_vars.LOCAL_ADDRESS, data={"type": "gen", "data": {"prompt": "Hi my name is Bryan"}})

async def main():
    await swarm_discover(global_vars.SOCK_UDP)

if __name__ == "__main__":
    log.info("Starting Swarm Process")
    asyncio.run(main())