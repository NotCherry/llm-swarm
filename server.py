import socket
import threading
from multiprocessing import Process
import time
import asyncio
import logging
from rich.logging import RichHandler



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



# Set your own listening port and the peer's IP/port
LISTEN_IP = '0.0.0.0'     # Listen on all interfaces
LISTEN_PORT = 5005        # Port to listen on

local_address = ".".join(get_outbound_ip().split(".")[:3])

SEARCH_IP_RANGE = [f"{local_address}.{i}" for i in range(1,254) ] # Change to the peer's IP
PEER_PORT = 5005          # Change to the peer's port
CONNECT = "connect".encode()

ACTICE_HOSTS = []

def listen(sock):
    global ACTICE_HOSTS
    while True:
        data, addr = sock.recvfrom(1024)
        if data.decode() == "connect":
            ACTICE_HOSTS.append(addr[0])

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1048576)  # 1 MB buffer
sock.bind((LISTEN_IP, LISTEN_PORT))


# Start listener thread
def update_network():
    global ACTICE_HOSTS

    def detect():
        log.info(f"Discovering hosts on: {local_address}.0")
        for host in SEARCH_IP_RANGE:
            sock.sendto(CONNECT, (host, PEER_PORT))
        log.info("Discovery end hosts found:")
        log.info(ACTICE_HOSTS)    
    
    while True:
        ACTICE_HOSTS = []
        detect()
        time.sleep(5)


def swarm_discover(sock):
    threading.Thread(target=listen, args=(sock,)).start()
    threading.Thread(target=update_network).start()
    

async def main():
    Process(target=swarm_discover, args=(sock,)).start()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
