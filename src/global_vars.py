import socket
import threading
import time
from typing import Dict
from src.structs import NetworkConfig
from src.local_logger import log
from src import global_vars
def get_outbound_ip():
    if global_vars.LOCAL_ADDRESS is not None:
        log.info(f"Using existing LOCAL_ADDRESS: {global_vars.LOCAL_ADDRESS}")
        return global_vars.LOCAL_ADDRESS


    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('1.1.1.1', 80))
        ip = s.getsockname()[0]
    except Exception as e:
        assert False, f"Failed to get outbound IP: {e} cannot continue without internet access."
    finally:
        s.close()

    log.info(f"IP with access to internet: {ip}")    
    return ip


LISTEN_IP = '0.0.0.0'
LISTEN_PORT = 5005
LISTEN_PORT_TCP = 5006
WS_PORT = 8543

TCP_CONNECTIONS = []

LOCAL_ADDRESS = get_outbound_ip()
PROGRAM_MINIMAL_SPACE = (500 * 1024 * 1024)
MASTER_NODE_BUFFER = (600 * 1024 * 1024)

SEARCH_IP_RANGE = [f"{".".join(LOCAL_ADDRESS.split(".")[:3])}.{i}" for i in range(1,254) ]
PEER_PORT = 5005

ACTIVE_HOSTS = []
MASTER_NODE = True
MASTER_NODE_IP: str = None


START_TIME = time.perf_counter()
WAIT_TIME = 10

SHARDING_SERVICE = False
METADATA_STORE: Dict[str, str] = {}

MODEL = None
MODEL_LOCK = threading.Lock()

NETWORK_TOPOLOGY = NetworkConfig(nodes={})
NETWORK_LOCK = threading.Lock()
NETWORK_CHECKSUM = None

SELECTED_MODEL = "Qwen/Qwen3-1.7B"
# SELECTED_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

WSS_SERVER_READY = threading.Event()

SOCK_UDP = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
SOCK_UDP.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)
SOCK_UDP.bind((LISTEN_IP, LISTEN_PORT))



