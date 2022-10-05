import argparse
import pickle
import os
import cv2
import numpy as np 
import copy
from nntool.importer.importer import create_graph

import onnx
import onnxruntime as ort
import collections

class CustomCOCODaset():
    def __init__(self, image_folder, ann_pickle, input_size, max_size=500, transpose_to_chw=True):

        np.random.seed(10)
        with open(ann_pickle, "rb") as f:
            self.annotations = sorted(pickle.load(f), key=lambda x: x[3])
            # self.annotations = pickle.load(f)

        self._idx = 0 
        assert self._idx < max_size < len(self.annotations), f"Choose max_size between {self._idx} and {len(self.annotations)}"
        self.max_idx = max_size 
        self.data_dir = image_folder
        self.input_size = input_size

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):

        if self._idx > self.max_idx:
            raise StopIteration()

        filename = self.annotations[self._idx][3]
        img_file = os.path.join(self.data_dir, filename)

        image = cv2.imread(img_file)
        image = self.preproc(image, self.input_size)
        image = self.slicing(image)

        self._idx += 1
        return image 
    
    def __len__(self):
        return self.max_idx

    @staticmethod
    def preproc(img, input_size, swap=(2, 0, 1), input_channels=3):
        if len(img.shape) == 3:
            padded_img = np.ones(
                (input_size[0], input_size[1], input_channels),
                dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)

        if len(resized_img.shape) == 2:
            resized_img = resized_img[..., None]

        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img 

    @staticmethod  
    def slicing(image):
        patch_top_left = image[..., ::2, ::2]
        patch_top_right = image[..., ::2, 1::2]
        patch_bot_left = image[..., 1::2, ::2]
        patch_bot_right = image[..., 1::2, 1::2]
        image = np.concatenate(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            axis=0,
        )
        return image

def chw_slice(array):
    patch_top_left = array[..., ::2, ::2]
    patch_top_right = array[..., ::2, 1::2]
    patch_bot_left = array[..., 1::2, ::2]
    patch_bot_right = array[..., 1::2, 1::2]
    array = np.concatenate(
        (
            patch_top_left,
            patch_bot_left,
            patch_top_right,
            patch_bot_right,
        ),
        axis=0,
    )
    return array
    
def hwc_slice(array):

    patch_top_left = array[::2, ::2, ...]
    patch_top_right = array[::2, 1::2, ...]
    patch_bot_left = array[1::2, ::2, ...]
    patch_bot_right = array[1::2, 1::2, ...]
    array = np.concatenate(
        (
            patch_top_left,
            patch_bot_left,
            patch_top_right,
            patch_bot_right,
        ),
        axis=-1,
    )
    return array

def build_graph(onnx_path, stats=None):

    graph = create_graph(onnx_path, opts={})

    ## fix the order of the last layer to match that in pytorch 
    graph[-1].fixed_order = True

    graph.adjust_order()
    graph.add_dimensions()
    graph.fusions('expression_matcher')
    graph.fusions('scaled_match_group')
    
    print("LAST LAYER ORDER", graph[-1].fixed_order)

    return graph



def onnx_layer_output(onnx_model_path, dummy_input):
    ort_session = ort.InferenceSession(onnx_model_path)
    org_outputs = [x.name for x in ort_session.get_outputs()]
    model = onnx.load(onnx_model_path)
    for node in model.graph.node:
        for output in node.output:
            if output not in org_outputs:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])

    # excute onnx
    ort_session = ort.InferenceSession(model.SerializeToString())
    outputs = [x.name for x in ort_session.get_outputs()]
    nodes = org_outputs + [node.name for node in model.graph.node]
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_outs = ort_session.run(outputs, ort_inputs)
    outputs = ort_outs = collections.OrderedDict(zip(nodes, ort_outs))
    return outputs

def clip_stats(stats, n_std):
    clipped_stats = copy.deepcopy(stats)
    if n_std:
        for layer, stat in stats.items():
            min_, max_ = stat['range_out'][0]['min'], stat['range_out'][0]['max']
            mean, std = stat['range_out'][0]['mean'], stat['range_out'][0]['std']
            clipped_stats[layer]['range_out'][0]['min'] = max(min_, mean - n_std * std)
            clipped_stats[layer]['range_out'][0]['max'] = min(max_, mean + n_std * std)
    return clipped_stats


def make_parser():
    parser = argparse.ArgumentParser("Parser")
    parser.add_argument("--path", type=str, default="./onnx_weights/yolox_nano_rgb_256x320_coco.onnx" )
    parser.add_argument("--coco_path", type=str, default="/home/abduragim/data/coco/val2017") 
    # parser.add_argument("--coco_path", type=str, default="/home/abduragim/data/coco/train2017") 
    # parser.add_argument("--ann_pickle", type=str, default="./data/coco_train_dataset_person_annotations.pickle")
    parser.add_argument("--ann_pickle", type=str, default="./data/coco_val_dataset_person_annotations.pickle")
    parser.add_argument("--stats", type=int, default=1, help="to use precalculated statistics")
    parser.add_argument("--clip_stats", type=int, default=None, help="value with which to clip the statistics")
    parser.add_argument("--input_size", type=tuple, default=(256, 320), help="input size")

    return parser

