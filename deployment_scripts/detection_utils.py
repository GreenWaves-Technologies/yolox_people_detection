
import numpy as np
from nntool.utils.fast_float import np_fastexp

def chw_slice(array):
    """Input slice operation for yolo NN working on CHW input arrays"""
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
    """Input slice operation for yolo NN working on HWC input arrays"""
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


def decode_np(inputs, hw, strides, use_fastexp=True):
    """Decode function to decode the boxes predicted by the backbone before the NMS"""
    exp_fn = np_fastexp if use_fastexp else np.exp

    i = 0
    for (height, width), stride in zip(hw, strides):
        # height, width = hw[j]
        for h in range(0, height):
            for w in range(0, width):
                inputs[i]     = (inputs[i    ] + w) * stride
                inputs[i + 1] = (inputs[i + 1] + h) * stride
                inputs[i + 2] = exp_fn(inputs[i + 2]) * stride
                inputs[i + 3] = exp_fn(inputs[i + 3]) * stride
                # print(f"[{i}] {inputs[i]:.2f}, {inputs[i+1]:.2f}, {inputs[i+2]:.2f}, {inputs[i+3]:.2f} {inputs[i+4]:.2f} {inputs[i+5]:.2f}")
                i += 6
    return inputs



def iou(box1, box2):
    """Intersect over union - boxes in form of xcnt, ycnt, w, h"""
    ax1, ay1, ax2, ay2, _, _, _ = box1
    bx1, by1, bx2, by2, _, _, _ = box2
    x_left, x_right = max(ax1, bx1), min(ax2, bx2)
    y_top, y_bot = max(ay1, by1), min(ay2, by2)
    if x_right < x_left or y_bot < y_top:
        return 0
    intersection_area = (x_right - x_left) * (y_bot - y_top)
    box1_area = (ax2 - ax1) * (ay2 - ay1)
    box2_area = (bx2 - bx1) * (by2 - by1)
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou


def nms(boxes, nms_thresh=0.65, max_boxes=400):
    """NMS function"""
    n_valid_boxes = min(len(boxes), max_boxes)
    # set everything to alive
    in_boxes = [np.concatenate((box, np.array([True]))) for box in boxes[:n_valid_boxes]]
    # np.set_printoptions(linewidth=120)
    # print(in_boxes)
    out_boxes = []
    for i in range(n_valid_boxes):
        if in_boxes[i][-1]:
            # print("box1 ", in_boxes[i])
            for j in range(n_valid_boxes):
                if i != j and in_boxes[j][-1]:
                    # print("\tbox2 ", in_boxes[j])
                    if iou(in_boxes[i], in_boxes[j]) >= nms_thresh:
                        if in_boxes[i][4] > in_boxes[j][4]:
                            in_boxes[j][-1] = 0
                        else:
                            in_boxes[i][-1] = 0
            if in_boxes[i][-1]:
                # print(f"append: {in_boxes[i]}")
                out_boxes.append(in_boxes[i])
    return out_boxes


def print_boxes(boxes):
    for i, box in enumerate(boxes):
        print(f"[{i}] {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f} {box[4]:.2f} {box[5]:.2f}")
