import argparse
from utils import GvsocInputGeneratorCOCO

def make_parser():
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument(
        "--image_folder", 
        type=str,
        default="./data/coco/val2017", 
        help="path to onnx model"
    )
    parser.add_argument(
        "--annotations", 
        type=str, 
        default="/data/coco/val2017/annotations/instances_val2017.json", 
        help="path to coco dataset images"
    ) 
    parser.add_argument(
        "--gvsoc_inputs",
        type=str, 
        default="./gvsoc_inputs", 
        help="path to coco dataset annotations"
    ) 
    parser.add_argument(
        "--input_size", 
        type=tuple, 
        default=(240,320), 
        help="input size"
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=3,
        help="input channels"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["rgb", "bayer"],
        default="rgb",
        help="model type to validate"
    )
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()

    gvsoc_input_generator = GvsocInputGeneratorCOCO(
        image_folder = args.image_folder,
        annotations = args.annotations,
        gvsoc_inputs_folder = args.gvsoc_inputs,
        input_size = args.input_size,
        model_type = args.model_type,
        input_channels = args.input_channels,
    )
    gvsoc_input_generator.generate_and_save()


if __name__ == "__main__":
    main()
