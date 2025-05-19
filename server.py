import threading
import asyncio
from src.network.listeners import listen, wsserver
from src.network.wss import brodcast_data_to_node
from src import global_vars 
from src.scheduler.plan_network import update_network
from src.util import deserialize_network_config
from src.local_logger import log
from src import global_vars

import os
from dotenv import load_dotenv


load_dotenv()

async def swarm_discover(sock):

    # Restore last network config if it exists
    net_cfg_path = "net.cfg"
    if os.path.exists(net_cfg_path):
        with open(net_cfg_path, "r") as f:
            with global_vars.NETWORK_LOCK:
                global_vars.NETWORK_TOPOLOGY = deserialize_network_config(f.read())

    # Start background threads
    threading.Thread(target=listen, args=(sock,)).start()
    threading.Thread(target=update_network).start()
    threading.Thread(target=wsserver).start()

    # Wait for conditions to broadcast
    while global_vars.MASTER_NODE and not (global_vars.NETWORK_TOPOLOGY.loaded_model and not global_vars.NETWORK_TOPOLOGY.generating):
        await asyncio.sleep(1)
    await brodcast_data_to_node(node_ip=global_vars.LOCAL_ADDRESS, data={"type": "gen", "data": {"prompt": "Hi my name is Bryan"}})

async def main():
    await swarm_discover(global_vars.SOCK_UDP)

if __name__ == "__main__":
    asyncio.run(main())