import argparse
import argcomplete
import os
from nntool.api.utils import qsnr, model_settings
from PIL import Image, ImageDraw
import numpy as np
from nntool_utils import build_graph
from detection_utils import nms, decode_np, hwc_slice, print_boxes

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
    parser.add_argument('--remove_focus_layer', action="store_true",
                        help="Pass argument if you want to remove the focus layer")
    parser.add_argument('--isp',   action='store_true',
                        help="Add ISP layers in front of the network")
    

    return parser


if __name__ == '__main__':
 
    parser = create_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    print(f"Model path: {str(args.trained_model)}")
    print(f"Calibration path: {str(args.stats_path)}")
    print(f"AT model path: {args.at_model_path}")

    print("Building graph")
    G = build_graph(
        str(args.trained_model),
        args.stats_path
    )

 
    print(G.qshow([G[0]]))
    at_settings=model_settings(
        tensor_directory=args.tensors_dir,
        model_directory=os.path.split(args.at_model_path)[0] if args.at_model_path else "",
        model_file=os.path.split(args.at_model_path)[1] if args.at_model_path else "ATmodel.c",

        l1_size=128000,
        l2_size=1000000,
        graph_trace_exec=False,
        graph_monitor_cycles=True,
        graph_produce_node_names=True,
        graph_produce_operinfos=True,
        graph_const_exec_from_flash=True,

        l3_ram_device=args.ram_type,
        l3_flash_device=args.flash_type,
    )

    if args.mode in ["inference", "test"]:
        if not args.input_image or not os.path.exists(args.input_image):
            raise ValueError("Provide an image file with --input_image in inference mode")
        print(f"Image file {args.input_image}")
        image = Image.open(args.input_image)
        input_tensor = np.array(image)
        if args.remove_focus_layer:
            input_tensor = hwc_slice(input_tensor)
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
                print_output=True
            )

        else:
            onode = G["output_1"]

            float_boxes = flout[onode.step_idx][0]
            quant_boxes = dqout[onode.step_idx][0]
            strides = [8, 16, 32]
            feature_map_sizes = [(30, 40), (15, 20), (8, 10)]
            decoded_float_boxes = decode_np(float_boxes.flatten(), feature_map_sizes, strides)
            decoded_quant_boxes = decode_np(quant_boxes.flatten(), feature_map_sizes, strides)
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

            draw=ImageDraw.Draw(image)
            for box in nms_boxes:
                draw.rectangle([(box[0],box[1]),(box[2],box[3])],outline="white")
            if args.gt_file:
                # remove the alive bit
                num_boxes = np.array([len(nms_boxes)]).astype(np.float32)
                nms_boxes = np.array(nms_boxes)[:, :-1].flatten().astype(np.float32)
                gt = np.concatenate([num_boxes, nms_boxes])
                gt.tofile(args.gt_file)
                image.save(f"{os.path.split(args.gt_file)[0]}/gt.jpg")
            else:
                image.show()

    else:
        print("HELLLO")
        print(args.isp)
        if args.isp:
            print("CIAO MAMMA!!!!")
            #Add ISP Nodes directly in NNTool to get generated in the AT model
            # To add a resizer that goes to a different shape of the input (i.e. if network input is not QVGA)
            # There is not need to express the "to shape" since NNTool knows already the input shape of network.
            # resizer = G.insert_resizer(
            #     "input_1",
            #     from_shape=(240, 320),
            #     resize_op="bilinear",
            #     spatial_axes=(0, 1)
            #     )

            #We start from white balance since it is attached to the input
            wb_node = G.insert_isp(
                "input_1",
                isp_node_type="white_balance",   
                percentile=95,
                spatial_axes=(0, 1)
                )
            demosaic_node = G.insert_isp(
                "input_1",
                isp_node_type="demosaic_and_downsample",
                spatial_axes=(0, 1)
                )
            # output of Demosaiced and white balanced image can be saved to show results with these options"
            new_out_node = G.insert_output(wb_node)
            new_out_node.at_options.allocate = 1
            # force this node to output and put it in L3
            # these options does not work
            #new_out_node.at_options.out_home_mem_loc = "AT_MEM_L3_DEFAULTRAM"
            #new_out_node.at_options.out_exec_mem_loc = "AT_MEM_L3_DEFAULTRAM"
        G[0].allocate = True 
        G.gen_at_model(
            write_constants=True,
            directory=os.path.split(args.at_model_path)[0],
            settings=at_settings
        )
