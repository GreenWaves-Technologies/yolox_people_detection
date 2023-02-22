import argparse
import argcomplete
import sys
import os
import copy
import pickle
from pathlib import Path
from loguru import logger
from nntool.api import NNGraph

def create_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='train')

    parser.add_argument('--trained_model', default=None, required=True,
                        help='Output - Trained model in tflite format')
    parser.add_argument('--tensors_dir', default="tensors",
                        help="Where nntool stores the weights/bias tensors dir (only used in generate and performance mode)")
    parser.add_argument('--at_model_path', default=None,
                        help="Path to the C autotiler model file to generate (only used in generate mode)")
    parser.add_argument('--stats_path', default=None,
                        help="Path to pickle file for statistics")
    parser.add_argument('--ram_type', default="AT_MEM_L3_DEFAULTRAM", choices=['AT_MEM_L3_HRAM', 'AT_MEM_L3_QSPIRAM', 'AT_MEM_L3_OSPIRAM', 'AT_MEM_L3_DEFAULTRAM'],
                        help="Ram type to use during inference on platform (only used in generate and performance mode)")
    parser.add_argument('--flash_type', default="AT_MEM_L3_DEFAULTFLASH", choices=['AT_MEM_L3_HFLASH', 'AT_MEM_L3_QSPIFLASH', 'AT_MEM_L3_OSPIFLASH', 'AT_MEM_L3_MRAMFLASH', 'AT_MEM_L3_DEFAULTFLASH'],
                        help="Flash type to use during inference (only used in generate and performance mode)")
    return parser

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


if __name__ == '__main__':
 
    parser = create_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    logger.info(f"Model path: {str(args.trained_model)}")
    logger.info(f"Calibration path: {str(args.stats_path)}")
    logger.info(f"AT model path: {args.at_model_path}")

    logger.info("Building graph")
    G = build_graph(str(args.trained_model))

    # reate stats from pickle file 
    logger.info(f"Loading stats from {str(args.stats_path)}")
    with open(str(args.stats_path), "rb") as f:
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
            "tensor_directory": args.tensors_dir,
            "model_directory": os.path.split(args.at_model_path)[0] if args.at_model_path else "",
            "model_file": os.path.split(args.at_model_path)[1] if args.at_model_path else "ATmodel.c",

            "l1_size": 128000,
            "l2_size": 1000000,

            "graph_monitor_cycles": True,
            "graph_produce_node_names": True,
            "graph_produce_operinfos": True,
            "graph_const_exec_from_flash": True,

            "l3_ram_device": args.ram_type,
            "l3_flash_device": args.flash_type, #"AT_MEM_L3_DEFAULTFLASH",
        }
    )