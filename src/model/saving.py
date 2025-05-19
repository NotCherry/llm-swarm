
import copy
import json
import os
import re

from src.model.model import get_model_metadata
from src.ptcode import safe_load_metadata_single
from src import global_vars  
from ..util import get_model_filename
from src.local_logger import log
from safetensors.torch import save_file

def save_local_layers(state_dict):    
    with global_vars.MODEL_LOCK:
        global_vars.MODEL.load_state_dict(state_dict, strict=False)
        node_layer_loaded = {"msg": "node_layer_loaded", "data": list(state_dict.keys())}
        node_layer_loaded_bytes = json.dumps(node_layer_loaded).encode('utf-8')
        global_vars.SOCK_UDP.sendto(node_layer_loaded_bytes, (global_vars.MASTER_NODE_IP, global_vars.PEER_PORT))
        r = [str(x) for x in range(global_vars.MODEL.shard.start_layer, global_vars.MODEL.shard.end_layer + 1)]
        condition1 = all(any(re.search(rf'\.layers\.{layer}\.', s) for s in global_vars.MODEL.loaded_keys) for layer in r)
        condition2 = all( layer in global_vars.MODEL.loaded_keys for layer in  ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"] )
        if condition1 and condition2:
            log.info('Saving node weights')
            fn = f"{get_model_filename}.safetensors"
            # FIXME
            mt = get_model_metadata()
            if not os.path.exists(fn):
                st = global_vars.MODEL.state_dict()
                dummy_layer = {"dtype": "BF16", "shape": [1000000000000000000], "data_offsets": [1000000000000000000, 1000000000000000000]}
                # Add missing keys from mt to st with the dummy layer to preallocat metadata to match all layers
                for i, key in enumerate(mt.keys()):
                    if key not in st.keys():
                        st[f"dummy_layer_{i}"] = dummy_layer
                save_file(st, fn)
                return
            

            # mt loaded model metadata | metadata_local - local file metadata
            # local metadata has dummy_layers
            mt = get_model_metadata()
            f, data_start, local_metadata = safe_load_metadata_single(fn)
            new_local_metadata = copy.deepcopy(local_metadata)
            last_data_offset = [v for k, v in new_local_metadata.items() if "dummy" not in k and "metadata" not in k][-1]
             
            keys_to_add = [key for key in mt if key not in local_metadata]

            # Find dummy keys to remove (up to the number of keys to add)
            keys_to_remove = [key for key in new_local_metadata if "dummy" in key][:len(keys_to_add)]

            # Update new_local_metadata: add new keys, remove dummy keys
            new_local_metadata.update({key: mt[key] for key in keys_to_add})
            for key in keys_to_remove:
                del new_local_metadata[key]
            

            # Update data offsets of the new keys:
            s = 0
            for k in keys_to_add:
                s = new_local_metadata[k]['data_offset'][1] - new_local_metadata[k]['data_offset'][0]
                new_local_metadata[k]['data_offset'] = [last_data_offset, last_data_offset + s]
                last_data_offset += s


            json_bytes = json.dumps(new_local_metadata).encode('utf-8')
            json_size = len(json_bytes)
            padding_size = data_start - json_size

            if padding_size < 0:
                raise ValueError(f"JSON content is too large to fit in {json_size} bytes without newlines.")

            # Create padding
            padding = b' ' * padding_size

            # Final output
            final_output = json_bytes + padding


            f.seek(8)
            f.write(final_output)
