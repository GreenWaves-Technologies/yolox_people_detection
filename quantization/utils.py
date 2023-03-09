import copy
import onnx
import collections
import numpy as np 
import onnxruntime as ort
from nntool.api import NNGraph
from pycocotools.coco import COCO

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

def check_input_dims(G, arr):
    assert G.inputs_dim[0] == list(arr.shape), \
        f"Incorrect input size for {G.name} model. Got: {arr.shape} expected:{G.inputs_dim[0]}"

def get_annotations(coco_annotations_path):

    print("Loading annotations for class person...")
    cocoGt = COCO(coco_annotations_path)
    class_ids = cocoGt.getCatIds("person")

    img_ids = []
    for cls in class_ids:
        img_ids.extend(cocoGt.getImgIds(catIds=cls))

    annotations = cocoGt.loadImgs(img_ids) 
    return annotations