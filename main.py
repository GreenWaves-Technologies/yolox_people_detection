import numpy as np
import os, cv2
import pickle
from utils import CustomCOCODaset, build_graph, make_parser, clip_stats
from loguru import logger

if __name__ == "__main__":
    np.random.seed(10)
    np.set_printoptions(suppress=True)

    # get arguments 
    args = make_parser().parse_args()

    #create graph
    graph = build_graph(args.path)

    # calculate statistics
    dataset = CustomCOCODaset(
        args.coco_path,
        args.ann_pickle,
        max_size=1000,
        input_size=args.input_size,
    )   

    # get statistics
    if args.stats:
        split = "train" if "train" in args.ann_pickle else "val"
        with open(f"./data/stats_{split}.pickle", "rb") as f:
            stats = pickle.load(f)
    else:
        logger.info(f"Calculating statistics using {args.ann_pickle}")
        stats = graph.collect_statistics(dataset)
        save_name = "train" if "train" in args.ann_pickle else "val"
        with open(f"./data/stats_{save_name}.pickle", "wb") as f:
            pickle.dump(stats, f)

    if args.clip_stats:
        print("\t\t *** Clipping Stats ***")
        stats = clip_stats(stats, args.clip_stats)



    graph.quantize(
                stats,

                # graph_options={
                #     'bits': 8,
                #     'quantized_dimension': 'channel',
                #     'use_ne16': False,
                #     'hwc': False,
                # },

                graph_options={
                    "scheme": "float",
                    "float_type": "ieee16"
                },

                # node_options = node_options,
            )

    # prepare input
    sample = next(iter(dataset))
    
    path = "./images/000000001296.jpg"
    sample = cv2.imread(path)
    sample = CustomCOCODaset.preproc(sample, input_size=(256, 320))
    sample = CustomCOCODaset.slicing(sample)

    qout = graph.execute([sample], quantize=True, dequantize=False)

    # make inference in gvsoc
    graph[0].allocate = 1
    res = graph.execute_on_target(
        # pmsis_os='pulpos',
        pmsis_os='freertos',
        directory="./inference_validataion_fp16",
        pretty=True,
        input_tensors=[qout[0][0].astype(np.float16)],
        output_tensors=6,
        dont_run=False,
        do_clean=False,
        settings={
            'l1_size': 128000,
            'l2_size': 1000000,
            'tensor_directory': './weights_tensors',
        }
    )
    
    # print(res.stdout.split("\n"))
    # print(type(res))
    OUTPUT = res.stdout.split("\n")


    for (i, r) in enumerate(OUTPUT):
        print(f"Output {i}: \t {r}")

    # qsnrs = graph.qsnrs(qout, res.output_tensors)
    # for i, el in enumerate(qsnrs):
    #     if el is not None:
    #         print(f"graph[{i:3}] -> {graph[i].name:>25}: {el}")
    
    print(" \t \t *** Conversion Complieted ***")
    # print(res.stdout.split("\n"))








    ### save useful code lines for later use

    # dum_inp = np.random.randint(low=0, high=255, size=(3, 256, 320))
    # dum_inp = np.random.randint(low=0, high=255, size=(256, 320, 3))
    # with open(os.path.join("./inference_gvsoc_hwc/", f"Input_1_Unsliced.bin"), "wb") as in_f:
    #     dum_inp.astype(np.uint8, order='C', casting='unsafe', copy=True).tofile(in_f)

    # # slice the input
    # dum_inp_chw = chw_slice(dum_inp.copy())
    # # dum_inp_hwc = hwc_slice(np.moveaxis(dum_inp.copy(), 0, -1))
    # dum_inp_hwc = hwc_slice(dum_inp)
    
    # # print(dum_inp_chw[0, :2, :2])
    # print(dum_inp_hwc[:2, :2, 0])
    # print(dum_inp.shape)
    # # print(dum_inp_chw.shape)
    # print(dum_inp_hwc.shape)

    # with open(os.path.join("./inference_gvsoc_hwc/", f"Input_1_Python_sliced.bin"), "wb") as in_f:
    #     # dum_inp.astype(np.uint8, order='C', casting='unsafe', copy=True).tofile(in_f)
    #     dum_inp_hwc.astype(np.uint8, order='C', casting='unsafe', copy=True).tofile(in_f)

    # # with open(os.path.join("./", f"Input_1_sliced.bin"), "wb") as in_f:
    # #     dum_inp.astype(np.uint8, order='C', casting='unsafe', copy=True).tofile(in_f)