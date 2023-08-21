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
from PIL import Image, ImageDraw
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from quantization.utils import hwc_slice
from decoding_layer.decoding import decode_C_style_imp

def create_parser():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='train')

    parser.add_argument('--trained_model', default=None, required=True,
                        help='Output - Trained model in tflite format')
    parser.add_argument('--mode', default="generate", choices=['generate', 'inference', 'test'],
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
    parser.add_argument('--conf_thresh', default=0.35, type=float,
                        help="Confidence threshold")

    return parser

def build_graph(onnx_path, stats_path):

    graph = NNGraph.load_graph(onnx_path)

    ## fix the order of the last layer to match that in pytorch 
    graph[-1].fixed_order = True

    graph.adjust_order()
    graph.add_dimensions()
    graph.fusions('scaled_match_group')
    graph.fusions('expression_matcher')
    
    print("LAST LAYER ORDER", graph[-1].fixed_order)

    # create stats from pickle file 
    print(f"Loading stats from {str(stats_path)}")
    with open(str(stats_path), "rb") as f:
        stats = pickle.load(f)

    # clipping stats. std = 3 worked the best 
    print(f"Clipping stats with std {3}")
    stats = clip_stats(stats, n_std = 3)

    print("Quantizing graph")
    graph.quantize(
        stats,
        graph_options={
            "bits": 8,
            "quantize_dimension": "channel",
            "use_ne16": True,
            "hwc": True,
        },
        node_options= {
            'output_1' : 
                {
                    'scheme' : 'float', 
                    'float_type': 'float32'
                },
        },
    )

    graph.name = "main"
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

# boxes in form of xcnt, ycnt, w, h
def iou(box1, box2):
    ax1, ay1, ax2, ay2, _, _, _ = box1
    bx1, by1, bx2, by2, _, _, _ = box2
    x_left, x_right = max(ax1, bx1), min(ax2, bx2)
    y_top, y_bot = max(ay1, by1), min(ay2, by2)
    if x_right < x_left or y_bot < y_top:
        return 0
    intersection_area = (x_right - x_left) * (y_bot - y_top)
    box1_area = (ax2 - ax1) * (ay2 - ay1)
    box2_area = (bx2 - bx1) * (by2 - by1)
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou


def nms(boxes, nms_thresh=0.65, max_boxes=400):
    n_valid_boxes = min(len(boxes), max_boxes)
    # set everything to alive
    in_boxes = [np.concatenate((box, np.array([True]))) for box in boxes[:n_valid_boxes]]
    # np.set_printoptions(linewidth=120)
    # print(in_boxes)
    out_boxes = []
    for i in range(n_valid_boxes):
        if in_boxes[i][-1]:
            # print("box1 ", in_boxes[i])
            for j in range(n_valid_boxes):
                if i != j and in_boxes[j][-1]:
                    # print("\tbox2 ", in_boxes[j])
                    if iou(in_boxes[i], in_boxes[j]) >= nms_thresh:
                        if in_boxes[i][4] > in_boxes[j][4]:
                            in_boxes[j][-1] = 0
                        else:
                            in_boxes[i][-1] = 0
            if in_boxes[i][-1]:
                # print(f"append: {in_boxes[i]}")
                out_boxes.append(in_boxes[i])
    return out_boxes

def print_boxes(boxes):
    for i, box in enumerate(boxes):
        print(f"[{i}] {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f} {box[4]:.2f} {box[5]:.2f}")
        

if __name__ == '__main__':
 
    parser = create_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    print(f"Model path: {str(args.trained_model)}")
    print(f"Calibration path: {str(args.stats_path)}")
    print(f"AT model path: {args.at_model_path}")

    print("Building graph")
    G = build_graph(str(args.trained_model), args.stats_path)

    print(G.show([G[0]]))
    print(G.qshow([G[0]]))
    at_settings={
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

    if args.mode in ["inference", "test"]:
        if not args.input_image or not os.path.exists(args.input_image):
            raise ValueError("Provide an image file with --input_image in inference mode")
        print(f"Image file {args.input_image}")
        image = Image.open(args.input_image)
        input_tensor = hwc_slice(np.array(image))
        print(input_tensor.shape)
        flout = G.execute([input_tensor])
        dqout = G.execute([input_tensor], dequantize=True)
        qqout = G.execute([input_tensor], dequantize=False, quantize=True)
        if args.mode == "test":
            res = G.execute_on_target(
                input_tensors=qqout[0],
                directory="/tmp/test_yolox_people_detection",
                settings=at_settings,
                check_on_target=True,
                output_tensors=4,
                print_output=True
            )

        else:
            onode = G["output_1"]

            float_boxes = flout[onode.step_idx][0]
            quant_boxes = dqout[onode.step_idx][0]
            strides = [8, 16, 32]
            feature_map_sizes = [(30, 40), (15, 20), (8, 10)]
            decoded_float_boxes = decode_C_style_imp(float_boxes.flatten(), feature_map_sizes, strides)
            decoded_quant_boxes = decode_C_style_imp(quant_boxes.flatten(), feature_map_sizes, strides)
            # print("Before filter")
            # print_boxes(quant_boxes)

            print(G.show([G[onode.step_idx]]))
            print(G.qshow([G[onode.step_idx]]))
            print(f"QSNR between float and quantized execution of the bounding boxes: {qsnr(float_boxes, quant_boxes)}")
            print(f"QSNR between float and quantized execution of the bounding boxes (after decoding): {qsnr(decoded_float_boxes, decoded_quant_boxes)}")

            boxes = decoded_quant_boxes.reshape(-1, 6)
            # print_boxes(boxes)
            # filter boxes
            boxes = boxes[(boxes[:,4]*boxes[:,5]) > args.conf_thresh]
            for i, box in enumerate(boxes):
                w, h = box[2] / 2, box[3] / 2
                x1, y1, x2, y2 = box[0]-w, box[1]-h, box[0]+w, box[1]+h
                boxes[i] = np.array([x1, y1, x2, y2, box[4], box[5]])

            # print("After filter")
            # print_boxes(boxes)

            # nms
            nms_boxes = nms(boxes, 0.65, 400)
            print("Bounding boxes after nms:")
            print_boxes(nms_boxes)

            if args.gt_file:
                # remove the alive bit
                np.array(nms_boxes)[:, :-1].flatten().astype(np.float32).tofile(args.gt_file)
            else:
                draw=ImageDraw.Draw(image)
                for box in nms_boxes:
                    draw.rectangle([(box[0],box[1]),(box[2],box[3])],outline="white")
                image.show()

    else:
        G[0].allocate = True

        G.generate(
            write_constants=True,
            settings=at_settings
        )