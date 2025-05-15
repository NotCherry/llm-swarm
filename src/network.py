from dataclasses import dataclass
import json
from typing import Dict

@dataclass
class NetworkMsg:
    msg: str
    data: Dict


def listen(sock):
    global ACTIVE_HOSTS, START_TIME, MASTER_NODE, MASTER_NODE_IP, NETWORK_TOPOLOGY, SELECTED_MODEL
    
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
            if addr[0] not in ACTIVE_HOSTS:
                ACTIVE_HOSTS.append(addr[0])
            if addr[0] not in NETWORK_TOPOLOGY.nodes.keys():
                NETWORK_TOPOLOGY.nodes[addr[0]] = Node(ip=addr[0], spec=spec)
                log.error(f"Node INIT {NETWORK_TOPOLOGY.nodes[addr[0]]}")

            elif spec !=  NETWORK_TOPOLOGY.nodes[addr[0]].spec:
                NETWORK_TOPOLOGY.nodes[addr[0]].spec = spec
        if data['msg'] == "shard_update":
            NETWORK_TOPOLOGY.nodes[addr[0]].shard.loaded = True
        if data['msg'] == "node_layer_loaded":
            # FIX ME: No way this works - LOL i start to not remember how this works XD
            with NETWORK_LOCK:
                if get_model_filename() not in NETWORK_TOPOLOGY.nodes[addr[0]].saved_layers.keys():
                    NETWORK_TOPOLOGY.nodes[addr[0]].saved_layers = []
                NETWORK_TOPOLOGY.nodes[addr[0]].saved_layers[get_model_filename()].extend(json.loads(data["data"]))            
                NETWORK_TOPOLOGY.nodes[addr[0]].loaded_layers.extend(json.loads(data["data"]))
                net_config = {"msg": "net_config", "data": serialize_network_config(NETWORK_TOPOLOGY)}
                net_config_bytes = json.dumps(net_config).encode('utf-8')

                for node_id, node in NETWORK_TOPOLOGY.nodes.items():
                    sock.sendto(net_config_bytes, (node.ip, PEER_PORT))
                    log.info(f"Node: {node_id} - {node.shard}")
        if data['msg'] == "MASTERNODE" and ((START_TIME + WAIT_TIME) > time.perf_counter()):
            log.info(f"Master node address: {addr[0]}")
            # maby we can just append it to the node struct?
            MASTER_NODE = False
            MASTER_NODE_IP = addr[0]
            # TODO Handle case when masternode Disapear
        if data['msg']  == "net_config":
            with NETWORK_LOCK:
                NETWORK_TOPOLOGY.nodes = deserialize_network_config(data["data"]).nodes
                log.error(f"config recived {NETWORK_TOPOLOGY.nodes}")

                with open("net.cfg", "w") as f:
                    f.write(serialize_network_config(NETWORK_TOPOLOGY))