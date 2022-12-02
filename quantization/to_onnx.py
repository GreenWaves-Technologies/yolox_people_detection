import os
import torch
import argparse
import importlib

from torch import nn
from loguru import logger

from yolox.utils.data_augment import preproc
from yolox.models.network_blocks import SiLU
from yolox.utils.model_utils import replace_module


def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    parser.add_argument(
        "-on",
        "--output-name", 
        type=str, 
        default="yolox.onnx", 
        help="output name of models"
    )
    parser.add_argument(
        "--input", 
        default="images", 
        type=str, 
        help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", 
        default="output", 
        type=str, 
        help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", 
        "--opset", 
        default=11, 
        type=int, 
        help="onnx opset version"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=1, 
        help="batch size"
    )
    parser.add_argument(
        "--dynamic", 
        action="store_true", 
        help="whether the input shape should be dynamic or not"
    )
    parser.add_argument(
        "--no-onnxsim", 
        action="store_true", 
        help="use onnxsim or not"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument(
        "-expn",
        "--experiment-name",
        type=str,
        default=None
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default='yolox-nano',
        help="model name"
    )
    parser.add_argument(
        "-c",
        "--ckpt",
        default="weights/model.pth",
        type=str,
        help="ckpt path"
    )
    parser.add_argument(
        "-iw",
        "--input_width",
        type=int,
        help="input images width"
    )
    parser.add_argument(
        "-ih",
        "--input_height",
        type=int,
        help="input images hight"
    )
    parser.add_argument(
        "-ic",
        "--input_channels",
        type=int,
        help="input channels (3 for RGB images, 1 for BAYER"
    )
    parser.add_argument(
        "-it",
        "--input_type",
        choices=["rgb", "bayer"],
        default="rgb",
        help="input image type: RGB or BAYER"
    )
    parser.add_argument(
        "--decode",
        action="store_true",
        help="create onnx model with decoding inside"
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="create onnx model with postprocessing inside"
    )

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    exp_name = args.name.replace("-", "_")  # convert string like "yolox-s" to "yolox_s"
    module_name = ".".join(["yolox", "exp", exp_name])
    exp = importlib.import_module(module_name).Exp()

    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    if args.input_width and args.input_height:
        exp.input_size = (args.input_height, args.input_width)
    if args.input_channels:
        exp.input_channels = args.input_channels
    if args.input_type:
        exp.image_type = args.input_type
    if args.postprocess:
        exp.postprocess_in_forward = args.postprocess

    # switch to_onnx flat on since we are in onnx export mode
    model = exp.get_model(to_onnx=True)

    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    model.eval()
    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = args.decode

    logger.info("Loading checkpoint done!")
    
    # create an input sample and slice it 
    dummy_input = torch.randn(args.batch_size, exp.input_channels, exp.input_size[0], exp.input_size[1])
    patch_top_left = dummy_input[..., ::2, ::2]
    patch_top_right = dummy_input[..., ::2, 1::2]
    patch_bot_left = dummy_input[..., 1::2, ::2]
    patch_bot_right = dummy_input[..., 1::2, 1::2]
    dummy_input = torch.cat(
        (
            patch_top_left,
            patch_bot_left,
            patch_top_right,
            patch_bot_right,
        ),
        dim=1,
    )

    logger.info("Converting the model to onnx...")
    # export the model
    torch.onnx._export(
        model,
        dummy_input,
        args.output_name,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes={args.input: {0: 'batch'},
                      args.output: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset,
    )
    logger.info("Generated onnx model named {}".format(args.output_name))

    if not args.no_onnxsim:
        import onnx
        from onnxsim import simplify

        input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model,
                                     dynamic_input_shape=args.dynamic,
                                     input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name)
        logger.info("Generated simplified onnx model named {}".format(args.output_name))


if __name__ == "__main__":
    main()
