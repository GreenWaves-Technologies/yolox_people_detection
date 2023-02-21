import sys
import os
import copy
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

def clip_stats(stats, n_std):
    clipped_stats = copy.deepcopy(stats)
    if n_std:
        for layer, stat in stats.items():
            min_, max_ = stat['range_out'][0]['min'], stat['range_out'][0]['max']
            mean, std = stat['range_out'][0]['mean'], stat['range_out'][0]['std']
            clipped_stats[layer]['range_out'][0]['min'] = max(min_, mean - n_std * std)
            clipped_stats[layer]['range_out'][0]['max'] = min(max_, mean + n_std * std)
    return clipped_stats

if len(sys.argv) < 3:
    raise ValueError("Provide model path and calibration directory path")

cur_dir = sys.argv[1]
model_name = sys.argv[2]
stats_name = sys.argv[3]
at_model_path = sys.argv[4]
weights_path = sys.argv[5]

at_model_file = os.path.split(at_model_path)[-1]
at_model_dir = os.path.split(at_model_path)[0]

model_path = Path(cur_dir).parent / "weights" / model_name
stats_path = Path(cur_dir).parent / "weights" / stats_name

logger.info(f"Model path: {str(model_path)}")
logger.info(f"Calibration path: {str(stats_path)}")
logger.info(f"AT model path: {at_model_path}")

logger.info("Building graph")
G = build_graph(str(model_path))

# reate stats from pickle file 
logger.info(f"Loading stats from {str(stats_path)}")
with open(str(stats_path), "rb") as f:
    stats = pickle.load(f)

# clipping stats. std = 3 worked the best 
logger.info(f"Clipping stats with std {3}")
stats = clip_stats(stats, n_std = 3)

logger.info("Quantizing graph")
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
        "tensor_directory": weights_path,
        "model_directory": at_model_dir,
        "model_file": at_model_file,

        "l1_size": 128000,
        "l2_size": 1000000,

        "graph_monitor_cycles": True,
        "graph_produce_node_names": True,
        "graph_produce_operinfos": True,
        "graph_const_exec_from_flash": True,

        "l3_ram_device": "AT_MEM_L3_DEFAULTRAM",
        "l3_flash_device": "AT_MEM_L3_DEFAULTFLASH", # "AT_MEM_L3_MRAMFLASH"
        
    }
)