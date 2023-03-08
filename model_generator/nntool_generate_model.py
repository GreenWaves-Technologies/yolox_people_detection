import argparse
import argcomplete
import sys
import os
import copy
import pickle
from pathlib import Path
#from loguru import logger
from nntool.api import NNGraph
from nntool.utils.stats_funcs import qsnr
from PIL import Image
import numpy as np
print(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from quantization.utils import hwc_slice
from decoding_layer.decoding import decode_C_style

def create_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='train')

    parser.add_argument('--trained_model', default=None, required=True,
                        help='Output - Trained model in tflite format')
    parser.add_argument('--mode', default="generate", choices=['generate', 'inference'],
                        help="generate: generate at model, inference: run inference with image provided and save gt file")
    parser.add_argument('--stats_path', default=None, required=True,
                        help="Path to pickle file for statistics")

    # Generate mode options
    parser.add_argument('--tensors_dir', default="tensors",
                        help="Where nntool stores the weights/bias tensors dir (only used in generate and performance mode)")
    parser.add_argument('--at_model_path', default=None,
                        help="Path to the C autotiler model file to generate (only used in generate mode)")
    parser.add_argument('--ram_type', default="AT_MEM_L3_DEFAULTRAM", choices=['AT_MEM_L3_HRAM', 'AT_MEM_L3_QSPIRAM', 'AT_MEM_L3_OSPIRAM', 'AT_MEM_L3_DEFAULTRAM'],
                        help="Ram type to use during inference on platform (only used in generate and performance mode)")
    parser.add_argument('--flash_type', default="AT_MEM_L3_DEFAULTFLASH", choices=['AT_MEM_L3_HFLASH', 'AT_MEM_L3_QSPIFLASH', 'AT_MEM_L3_OSPIFLASH', 'AT_MEM_L3_MRAMFLASH', 'AT_MEM_L3_DEFAULTFLASH'],
                        help="Flash type to use during inference (only used in generate and performance mode)")
    # inference mode options
    parser.add_argument('--input_image', default=None,
                        help="Image to run inference on")
    parser.add_argument('--gt_file', default=None,
                        help="File to save ground truth output")
    parser.add_argument('--conf_thresh', default=0.01,
                        help="Confidence threshold")

    return parser

def build_graph(onnx_path):

    graph = NNGraph.load_graph(onnx_path)

    ## fix the order of the last layer to match that in pytorch 
    graph[-1].fixed_order = True

    graph.adjust_order()
    graph.add_dimensions()
    graph.fusions('scaled_match_group')
    graph.fusions('expression_matcher')
    
    print("LAST LAYER ORDER", graph[-1].fixed_order)
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

    print(f"Model path: {str(args.trained_model)}")
    print(f"Calibration path: {str(args.stats_path)}")
    print(f"AT model path: {args.at_model_path}")

    print("Building graph")
    G = build_graph(str(args.trained_model))

    # reate stats from pickle file 
    print(f"Loading stats from {str(args.stats_path)}")
    with open(str(args.stats_path), "rb") as f:
        stats = pickle.load(f)

    # clipping stats. std = 3 worked the best 
    print(f"Clipping stats with std {3}")
    stats = clip_stats(stats, n_std = 3)

    print("Quantizing graph")
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
    print(G.show([G[0]]))
    print(G.qshow([G[0]]))

    if args.mode == "inference":
        #G.draw(filepath="draw")
        input_tensor = hwc_slice(np.array(Image.open(args.input_image)))
        print(input_tensor.shape)
        flout = G.execute([input_tensor])
        dqout = G.execute([input_tensor], dequantize=True)
        qqout = G.execute([input_tensor], dequantize=False, quantize=True)


        onode = G["output_1"]

        float_boxes = flout[onode.step_idx][0]
        quant_boxes = dqout[onode.step_idx][0]
        strides = [8, 16, 32]
        feature_map_sizes = [(30, 40), (15, 20), (8, 10)]
        decoded_float_boxes = decode_C_style(float_boxes.flatten(), feature_map_sizes, strides)
        decoded_quant_boxes = decode_C_style(quant_boxes.flatten(), feature_map_sizes, strides)

        print(G.show([G[onode.step_idx]]))
        print(G.qshow([G[onode.step_idx]]))
        print(decoded_float_boxes)
        print(decoded_quant_boxes)
        print(qsnr(float_boxes, quant_boxes))
        print(qsnr(decoded_float_boxes, decoded_quant_boxes))

    else:
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