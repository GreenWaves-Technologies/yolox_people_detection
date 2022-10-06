from re import T
import numpy as np
import pickle, cv2
from loguru import logger
from utils import CustomCOCODaset
from utils import build_graph, make_parser, onnx_layer_output, clip_stats, check_input_dims

from nntool.utils.stats_funcs import qsnr
from nntool.graph.types import ConstantInputNode
from collections import OrderedDict

if __name__ == "__main__":

    np.random.seed(10)
    np.set_printoptions(suppress=True)

    # get arguments 
    args = make_parser().parse_args()

    for name in args.__dict__:
        logger.info(f"{name}: {args.__dict__[name]}")

    #create graph
    graph_fp16 = build_graph(args.path)
    graph_int8 = build_graph(args.path)

    # graph_fp16.draw()

    # calculate statistics
    dataset = CustomCOCODaset(
        args.coco_path,
        args.ann_pickle,
        max_size=1000,
        input_size=args.input_size)   

    path = "./images/000000001296.jpg"
    sample = cv2.imread(path)
    sample = CustomCOCODaset.preproc(sample, input_size=(256, 320))
    sample = CustomCOCODaset.slicing(sample)

    check_input_dims(graph_fp16, sample)

    with open(f"./data/stats_val.pickle", "rb") as f:
        # stats = pickle.load(f)
        stats = graph_fp16.collect_statistics([sample])

    # clip stats
    if args.clip_stats:
        logger.info("\t\t *** Clipping statistics ***")
        stats = clip_stats(stats, args.clip_stats)

    # quantize graph
    node_options_int8 = {

        # 'Conv_0': {'scheme' : 'float', 'float_type': 'float16'},
        # 'Sigmoid_1': {'scheme' : 'float', 'float_type': 'float16'},
        # 'Mul_2': {'scheme' : 'float', 'float_type': 'float16'},

        # 'Conv_3': {'scheme' : 'float', 'float_type': 'float16'},
        # 'Sigmoid_4': {'scheme' : 'float', 'float_type': 'float16'},
        # 'Mul_5': {'scheme' : 'float', 'float_type': 'float16'},

        # 'Conv_6': {'scheme' : 'float', 'float_type': 'float16'},
        # 'Sigmoid_7': {'scheme' : 'float', 'float_type': 'float16'},
        # 'Mul_8': {'scheme' : 'float', 'float_type': 'float16'},

        # 'Conv_9': {'scheme' : 'float', 'float_type': 'float16'},
        # 'Sigmoid_10': {'scheme' : 'float', 'float_type': 'float16'},
        # 'Mul_11': {'scheme' : 'float', 'float_type': 'float16'},

        # 'Conv_12': {'scheme' : 'float', 'float_type': 'float16'},
        # 'Sigmoid_13': {'scheme' : 'float', 'float_type': 'float16'},
        # 'Mul_14': {'scheme' : 'float', 'float_type': 'float16'},

        # 'Conv_78': {'scheme' : 'float', 'float_type': 'float32'},
        # 'Sigmoid_79': {'scheme' : 'float', 'float_type': 'float32'},
        # 'Mul_80': {'scheme' : 'float', 'float_type': 'float32'},

        'output_1' : {'scheme' : 'float', 'float_type': 'float16'},
    }

    node_options_fp16 = {
        'output_1' : {'scheme' : 'float', 'float_type': 'float16'},
    }

    graph_int8.quantize(
                stats,
                graph_options={
                    'bits': 8,
                    'quantized_dimension': 'channel',
                    'use_ne16': False,
                    'hwc': False,
                },
                node_options=node_options_int8,
            )
    
    graph_fp16.quantize(
                stats,
                graph_options={
                    "scheme": "float",
                    "float_type": "ieee16"
                },
                node_options=node_options_fp16,
            )


    ## run onnx, nntool float and  nntool quantized
    outputs_nntool_float = graph_fp16.execute([sample], quantize=True, dequantize=True)
    outputs_nntool_int8 = graph_int8.execute([sample], quantize=True, dequantize=True)

    
    float_names = OrderedDict()
    for i, output in enumerate(outputs_nntool_float):
        float_names[graph_fp16[i].name] = output[0]

    int8_names = OrderedDict() 
    for i, output in enumerate(outputs_nntool_int8):
        int8_names[graph_int8[i].name] = output[0]

    # intersection of names in both graphs
    names = float_names.keys() & int8_names.keys()

    # calucate qsnr for each layer
    for name in float_names:
        if name in int8_names :
            if "constant" in name or "weights" in name:
                continue
            if float_names[name].shape == int8_names[name].shape:
                qsnr_ = qsnr(float_names[name], int8_names[name])
                logger.info(f"{name}:" + f"\t\t {qsnr_}") 
            else:
                logger.info(f"{name}: \t\t SHAPE MISMATCH")
        else:
            logger.info(f"{name} NOT FOUND IN INT8 GRAPH")

 
    print(outputs_nntool_float[-1][0].shape)