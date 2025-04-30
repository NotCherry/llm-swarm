import socket
import threading
from multiprocessing import Process
import time
import asyncio
import logging
from rich.logging import RichHandler
import tinygrad
import transformers
import time
from dataclasses import dataclass 
import requests
import io



FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")

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

local_address = ".".join(get_outbound_ip().split(".")[:3])

SEARCH_IP_RANGE = [f"{local_address}.{i}" for i in range(1,254) ] # Change to the peer's IP
PEER_PORT = 5005          # Change to the peer's port

ACTICE_HOSTS = []
MASTER_NODE = True

START_TIME = time.perf_counter()
WAIT_TIME = 10



def listen(sock):
    global ACTICE_HOSTS, START_TIME, MASTER_NODE
    
    while True:
        data, addr = sock.recvfrom(1024)
        text = data.decode()
        if text  == "connect":
            ACTICE_HOSTS.append(addr[0])
  
        if ((START_TIME + WAIT_TIME) > time.perf_counter()) and text == "MASTERNODE":
            log.info(f"Master node address: {addr[0]}")
            MASTER_NODE = False


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)  # 1 MB buffer
sock.bind((LISTEN_IP, LISTEN_PORT))


# Start listener thread
def update_network():
    global ACTICE_HOSTS, MASTER_NODE

    CONNECT = "connect".encode()
    MASTER_BRODCAST = "MASTERNODE".encode()

    def detect():
        log.info(f"Discovering hosts on: {local_address}.0")
        for host in SEARCH_IP_RANGE:
            sock.sendto(CONNECT, (host, PEER_PORT))
        log.info("Discovery end hosts found:")
        log.info(ACTICE_HOSTS)    
    
    while True:
        if START_TIME + WAIT_TIME < time.perf_counter():
            log.debug("Brodcasting Master Node")
            for host in SEARCH_IP_RANGE:
                sock.sendto(MASTER_BRODCAST, (host, PEER_PORT))
        log.info(f"Node Type {'Master' if MASTER_NODE else 'Worker'}")
        ACTICE_HOSTS = []
        detect()
        time.sleep(5)


def swarm_discover(sock):
    threading.Thread(target=listen, args=(sock,)).start()
    threading.Thread(target=update_network).start()
    

def model_downloader_to_ram():
    url = "https://huggingface.co/Qwen/Qwen2-1.5B/resolve/main/model.safetensors"
    # url = "https://huggingface.co/Qwen/Qwen2-1.5B/raw/main/merges.txt"
    metadata_end = 0
    size_read = False

    buffer = io.BytesIO()
    with open('downloaded_file.txt', 'wb') as f:
        # Make a GET request to the server with streaming enabled
        response = requests.get(url, stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=1024):
                print(int.from_bytes(chunk[:8]))
                buffer.write(chunk)
                if not size_read:
                    metadata_end = int.from_bytes(buffer.read(8))
                    print(metadata_end)
                break
            print("Download completed successfully.")
        else:
            print("Failed to retrieve the file. HTTP Status Code:", response.status_code)


async def main():
    # Process(target=swarm_discover, args=(sock,)).start()
    # Process(target=)
    model_downloader_to_ram()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
