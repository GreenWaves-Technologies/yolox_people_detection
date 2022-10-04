import os
import numpy as np 
from post_processing import xywh_to_xyxy, filter_boxes, nms, to_bboxes

def test_xywhtoxyxy(data, file_name):
    print("testing xywhtoxyxy", end="\t")
    data = data.reshape(-1, 6)
    with open(f"postprocessing_layer/post_processing_output1_bin/{file_name}", "rb") as f:
        output = np.fromfile(f, dtype=np.float32)
    
    output = output.reshape(-1, 6)
    assert np.allclose(data, output, rtol=1e-05, atol=1e-08), "xywhxyxy Output is not correct"


def test_filter_boxes(data, file_name):
    print("testing filter_boxes", end="\t") 
    data = np.array(data)
    data = data.reshape(-1, 7)
    with open(f"postprocessing_layer/post_processing_output_filtered_bin/{file_name}", "rb") as f:
        output = np.fromfile(f, dtype=np.float32)
    
    output = output.reshape(-1, 7)
    assert np.allclose(data, output, rtol=1e-05, atol=1e-08), "filtered Output is not correct"


def test_nms(data, file_name):

    print("testing nms", end="\t")
    data = np.array(data)
    data = data.reshape(-1, 7)
    print(data.shape, end="\t")
    data = data[data[:, 5].argsort()]
    print(data)
    with open(f"postprocessing_layer/post_processing_output_nms_bin/{file_name}", "rb") as f:
        output = np.fromfile(f, dtype=np.float32)

    output = output.reshape(-1, 7)
    print(output.shape, end="\t")
    # output = output[output[:, 5].argsort()]
    print(output)
    assert np.allclose(data, output, rtol=1e-05, atol=1e-08), "nms Output is not correct"

    # print("Checking nms output C", end="\t")
    # with open(f"postprocessing_layer/post_processing_output_nms_C/{file_name}", "rb") as f:
    #     output = np.fromfile(f, dtype=np.float32)
    #     output = np.array(output)
    
    # output = output.reshape(-1, 7)
    # print(output.shape, end="\t")
    # output = sorted(output, key=lambda x: x[4] * x[5], reverse=True)
    # print(data.shape)
    # assert np.allclose(data, output, rtol=1e-05, atol=1e-08), "nms Output C is not correct"

def postprocessing_test(data, file_name):

    data = xywh_to_xyxy(data)
    test_xywhtoxyxy(data, file_name)

    data, val_boxes = filter_boxes(data, conf_thresh=0.3)
    test_filter_boxes(data, file_name)

    data = to_bboxes(data, val_boxes)

    data = nms(data, val_boxes, nms_thresh=0.3)
    test_nms(data, file_name)

    print("ok")
    return data



if __name__ == "__main__":

    print("Postprocessing Test")

    for file_name in os.listdir("./postprocessing_layer/post_processing_input_bin"):
        with open(f"./postprocessing_layer/post_processing_input_bin/{file_name}", "rb") as f:
            data = np.fromfile(f, dtype=np.float32)

        postprocessing_test(data, file_name)
