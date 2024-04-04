import copy
import os
import pickle
from nntool.api import NNGraph
from nntool.api.utils import quantization_options

def quantize_graph(graph: NNGraph, stats):
    """Util function to quantize the graph"""
    graph.quantize(
        stats,
        graph_options=quantization_options(
            use_ne16=True,
            hwc=True,
        ),
        node_options= {
            out_node : quantization_options(
                scheme='float', 
                float_type='float32',
            ) for out_node in ["output_1", "output_2", "output_3"]
        },
    )

def build_graph(onnx_path, stats_path=None, quantize=True, remove_focus=False):
    """Function to prepare the model for the deployment

    Parameters:
    ===========

            - onnx_path:
                path to the onnx to open
            - stats_path:
                path to the statistics file storing all the information (required if quantize=True)
            - quantize:
                if False it will return the model as is without any quantization scheme applied
    """
    graph = NNGraph.load_graph(onnx_path)

    if remove_focus:
        # Remove the strided slice implementing the focus layer
        graph.remove_nodes(graph["_backbone_backbone_stem_Concat"], leave=False, up=True)
    # separate the last concat in multiple outputs, this helps the quantization because it doesn't quantize the
    # boxes with the same scheme as the scores and classes which tipycally have different statistics
    # graph.remove_nodes(graph["_head_Concat"], leave=False, up=False)
    # graph.remove_nodes(graph["_head_Concat_1"], leave=False, up=False)
    # graph.remove_nodes(graph["_head_Concat_2"], leave=False, up=False)
    graph.adjust_order()
    graph.fusions('scaled_match_group')

    print("LAST LAYER ORDER", graph[-1].fixed_order)

    if quantize:
        if not os.path.exists(str(stats_path)):
            raise ValueError("You need to provide a valid stats file path")

        # create stats from pickle file 
        print(f"Loading stats from {str(stats_path)}")
        with open(str(stats_path), "rb") as f:
            stats = pickle.load(f)

        # clipping stats. std = 3 worked the best 
        print(f"Clipping stats with std {3}")
        stats = clip_stats(stats, n_std = 3)

        print("Quantizing graph")
        quantize_graph(graph, stats)

    graph.name = "yolox"
    return graph


def clip_stats(stats, n_std):
    """This function will clip the statistics of the graph to n_std instead of min/max"""
    clipped_stats = copy.deepcopy(stats)
    if n_std:
        for layer, stat in stats.items():
            min_, max_ = stat['range_out'][0]['min'], stat['range_out'][0]['max']
            mean, std = stat['range_out'][0]['mean'], stat['range_out'][0]['std']
            clipped_stats[layer]['range_out'][0]['min'] = max(min_, mean - n_std * std)
            clipped_stats[layer]['range_out'][0]['max'] = min(max_, mean + n_std * std)
    return clipped_stats

def check_input_dims(G, arr):
    """Check that the input shape of the graph is matching the arr shape"""
    assert G.inputs_dim[0] == list(arr.shape), \
        f"Incorrect input size for {G.name} model. Got: {arr.shape} expected:{G.inputs_dim[0]}"
