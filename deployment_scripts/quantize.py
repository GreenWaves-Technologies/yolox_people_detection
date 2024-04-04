import pickle
import argparse
import numpy as np
from tqdm import tqdm

from dataset_generators import CostomCOCODaset, get_annotations
from nntool.utils.stats_funcs import qsnr
from nntool_utils import build_graph, clip_stats, check_input_dims, quantize_graph


def make_parser():
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument(
        "--onnx_path", 
        type=str, 
        default="./weights/yolox_nano_MVGA_less_chan.onnx", 
        help="path to onnx model"
    )
    parser.add_argument(
        "--coco_path", 
        type=str, 
        default="~/data/coco/train2017_MVGA_v2", 
        help="path to coco dataset images"
    ) 
    parser.add_argument(
        "--img_type", 
        type=str, 
        choices=["rgb", "bayer"],
        default="rgb",
        help="type of images to quantize with"
    ) 
    parser.add_argument(
        "--coco_annotations_path", 
        type=str, 
        default="~/data/coco/annotations/instances_train2017_MVGA.json", 
        help="path to coco dataset annotations"
    ) 
    parser.add_argument(
        "--quant_dataset_size", 
        type=int, 
        default=1000, 
        help="size f dataset for quantization"
    )
    parser.add_argument(
        "--input_size", 
        type=int, 
        nargs='+',
        default=(240, 320), 
        help="input size"
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=3,
        help="input channels"
    )
    parser.add_argument(
        "--stats", 
        type=str, 
        default=None, 
        help="path to precalculated statistics"
    )
    parser.add_argument(
        "--clip_stats", 
        type=int, 
        default=None, 
        help="value with which to clip the statistics"
    )
    return parser


def main():

    np.random.seed(10)
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=np.inf)

    # get arguments 
    args = make_parser().parse_args()

    print(tuple(args.input_size))
    #create graph
    graph = build_graph(args.onnx_path, quantize=False)

    # get class person annotations
    annotations = get_annotations(args.coco_annotations_path)

    # calculate statistics
    quant_dataset = CostomCOCODaset(
        image_folder     = args.coco_path,
        annotations      = annotations,
        max_size         = args.quant_dataset_size,
        input_size       = tuple(args.input_size),
        transpose_to_chw = True,
        input_channels   = args.input_channels,
        img_type         = args.img_type,
    )

    # prepare input
    sample = next(iter(quant_dataset))

    # check input dims
    check_input_dims(graph, sample)

    # get statistics
    if args.stats:
        print(f"Using statistics from file {args.stats}")
        with open(args.stats, "rb") as f:
            stats = pickle.load(f)
    else:
        print(f"Calculating statistics from {args.quant_dataset_size} images")
        stats = graph.collect_statistics(tqdm(quant_dataset))

        print(f"Saving statistics to precalculated_stats.pickle")
        with open("precalculated_stats.pickle", "wb") as f:
            pickle.dump(stats, f)

    if args.clip_stats:
        print(f"Clipping statistics using stats: {args.clip_stats}")
        stats = clip_stats(stats, args.clip_stats)

    print(f"Start quantization process !!!")
    quantize_graph(graph, stats)
    print("finished quantization process !!!")


    # get quantized outputs
    sample = sample.transpose(0, 2, 3, 1)

    # check input dims
    check_input_dims(graph, sample)

    qfout = graph.execute([sample], quantize=True, dequantize=True)
    fout  = graph.execute([sample], quantize=False, dequantize=False)

    for i, fp32, fp16 in zip(range(len(graph)), fout, qfout):
        print(f"Graph[{i:3}] -> {graph[i].name:>40}:\t{qsnr(fp32[0], fp16[0])}")

    # # make inference in gvsoc
    # print("Generating GVSOC inference tamplete. This might take a couple of minutes !!!")

    # graph.name = "main"
    # qout = graph.execute([sample], quantize=True, dequantize=False)

    # graph[0].allocate = 1
    # res = graph.execute_on_target(
    #     pmsis_os='freertos',
    #     directory="./GVSOC_INFERENCE_TEMPLATE",
    #     input_tensors=[qout[0][0]],
    #     output_tensors=6,
    #     dont_run=False,
    #     do_clean=False,
    #     cmake=True,
    #     at_loglevel=1, 
    #     platform = "gvsoc", # 'board'
    #     settings={
    #         'l1_size': 128000,
    #         'l2_size': 1000000,
    #         'tensor_directory': './weights_tensors',
    #         'graph_const_exec_from_flash': True,

    #         'l3_ram_device': 'AT_MEM_L3_DEFAULTRAM',
    #         # "l3_flash_device": "AT_MEM_L3_MRAMFLASH",
    #         'l3_flash_device':  'AT_MEM_L3_DEFAULTFLASH',
    #     }
    # ) 
    # print("Finished generating GVSOC inference tamplete !!!")
    
    # qsnrs = graph.qsnrs(qout, res.output_tensors)
    # for i, el in enumerate(qsnrs):
    #     print(f"QSNR {i}: {el}")
    #     if el is not None:
    #         print(f"graph[{i:3}] -> {graph[i].name:>25}: {el}")

    # print("Finished !!!")    


if __name__ == "__main__":

    main()
