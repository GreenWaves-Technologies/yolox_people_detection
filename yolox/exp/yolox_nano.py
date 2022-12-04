#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch
import torch.nn as nn

from abc import ABCMeta

class Exp(metaclass=ABCMeta):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (240, 320)
        self.test_size = self.input_size 
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ----- model config ------
        self.class_names = ('person',)
        self.num_classes = len(self.class_names)
        self.input_channels = 1
        self.act = "relu6"
        # ---------------- dataloader config ---------------- #
        # checkpoint saving directory 
        self.output_dir = ("./YOLOX_outputs")

        self.postprocess_in_forward = False
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65

        self.strides = [8, 16, 32]

        self.data_dir =  "/media/data/datasets/coco/"
        self.train_dir = "train2017"
        self.val_dir = "val2017"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        # self.image_type = 'RGB'
        self.image_type = 'BAYER'
        self.data_num_workers = 1

    def get_model(self, sublinear=False, to_onnx=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            backbone = YOLOPAFPN(
                self.depth, 
                self.width, 
                in_channels=in_channels,
                act=self.act, 
                depthwise=True, 
                input_channels=self.input_channels,
                to_onnx=to_onnx,
            )
            head = YOLOXHead(
                self.num_classes, 
                self.width, 
                in_channels=in_channels, 
                strides=self.strides,
                act=self.act, 
                depthwise=True
            )
            self.model = YOLOX(backbone, 
                                head, 
                                postprocess_in_forward=self.postprocess_in_forward
                            )

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
    
    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            # only update value with same key
            if hasattr(self, k):
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)
    