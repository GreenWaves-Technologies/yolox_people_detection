import sys
import os
import pickle
from pathlib import Path
from loguru import logger
from nntool.api import NNGraph

def build_graph(onnx_path):

    graph = NNGraph.load_graph(onnx_path)

    ## fix the order of the last layer to match that in pytorch 
    graph[-1].fixed_order = True

    graph.adjust_order()
    graph.add_dimensions()
    graph.fusions('scaled_match_group')
    graph.fusions('expression_matcher')
    
    logger.info("LAST LAYER ORDER", graph[-1].fixed_order)
    return graph

if len(sys.argv) < 3:
    raise ValueError("Provide model path and calibration directory path")


cur_dir = sys.argv[1]
model_name = sys.argv[2]
stats_name = sys.argv[3]
at_model_path = sys.argv[4]

at_model_file = os.path.split(at_model_path)[-1]
at_model_dir = os.path.split(at_model_path)[0]

model_path = Path(cur_dir).parent / "weights" / model_name
stats_path = Path(cur_dir).parent / "weights" / stats_name

print("Model path: ", str(model_path))
print("Calibration path: ", str(stats_path))
print("AT model path: ", at_model_path)


G = build_graph(str(model_path))

# reate stats from pickle file 
with open(str(stats_path), "rb") as f:
    stats = pickle.load(f)

G.quantize(
    stats,
    graph_options={
        "bits": 8,
        "quantize_dimension": "channel",
        "use_ne16": True,
        "hwc": True
    },
    node_options= {
        'output_1' : 
            {
                'scheme' : 'float', 
                'float_type': 'float32'
            },
    },
)

G.name = "main"
print(G.qshow([G[0]]))
G[0].allocate = True

G.generate(
    write_constants=True,
    settings={
        "tensor_directory": f"{at_model_dir}/tensors",
        "model_directory": at_model_dir,
        "model_file": at_model_file,

        "l1_size": 128000,
        "l2_size": 1000000,

        "graph_monitor_cycles": True,
        "graph_produce_node_names": True,
        "graph_produce_operinfos": True,
        "graph_const_exec_from_flash": True,

        "l3_ram_device": "AT_MEM_L3_DEFAULTRAM",
        # "l3_flash_device": "AT_MEM_L3_MRAMFLASH", #"AT_MEM_L3_DEFAULTFLASH",
        "l3_flash_device": "AT_MEM_L3_DEFAULTFLASH",
    }
)