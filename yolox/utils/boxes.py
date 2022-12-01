#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import numpy as np

import torch
import torchvision

__all__ = [
    "postprocess",
    "bboxes_iou",
]

def decode_outputs(hw, strides, outputs, dtype):
    grids = []
    new_strides = []
    for (hsize, wsize), stride in zip(hw, strides):
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing="ij")
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        new_strides.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1).type(dtype)
    new_strides = torch.cat(new_strides, dim=1).type(dtype)

    outputs[..., :2] = (outputs[..., :2] + grids) * new_strides
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * new_strides
    return outputs

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in range(prediction.shape[0])]
    image_pred = prediction[0]
    # for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
    # if not image_pred.size(0):
    #     continue
    # Get score and class with highest confidence
    class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

    conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
    detections = detections[conf_mask]
    # if not detections.size(0):
    #     continue

    if class_agnostic:
        nms_out_index = torchvision.ops.nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            nms_thre,
        )
    else:
        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )

    detections = detections[nms_out_index]
    if output[0] is None:
        output[0] = detections
    else:
        output[0] = torch.cat((output[0], detections))

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def convert_to_coco_format(img_size, output, info_imgs, img_id):
    data_list = []
    img_h, img_w = info_imgs
    if output is None:
        return data_list
    if type(output) == torch.Tensor:
        output = output.cpu()
        output = output.numpy()

    bboxes = output[:, 0:4]

    # preprocessing: resize
    scale = min(
        img_size[0] / float(img_h), img_size[1] / float(img_w)
    )
    bboxes /= scale
    bboxes = xyxy2xywh(bboxes)

    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    for ind in range(bboxes.shape[0]):
        label = 1
        pred_data = {
            "image_id": int(img_id),
            "category_id": label,
            "bbox": bboxes[ind].tolist(),
            "score": scores[ind].item(),
            "segmentation": [],
        }  # COCO json format
        data_list.append(pred_data)
    return data_list