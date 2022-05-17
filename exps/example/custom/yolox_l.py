#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/COCO"
        self.train_ann = "/home/bizon/YOLOX/datasets/COCO/annotations/instances_train2017.json"
        self.val_ann = "/home/bizon/YOLOX/datasets/COCO/annotations/instances_val2017.json"
        self.test_ann = "/home/bizon/YOLOX/datasets/COCO/annotations/instances_test2017.json"

        self.num_classes = 10

        self.max_epoch = 10
        self.data_num_workers = 4
        self.eval_interval = 1
        self.input_size = (1056,1056)
        self.flip_prob = 1
        self.momentum = 0.949
        self.test_size = (1056,1056)
        self.no_aug_epochs = 5

        # lr = basic_lr_per_img * batch_size
        # for lr to be 0.00065 with batch size 8 -> 0.000008125
        # 0.00125
        self.basic_lr_per_img = 0.000008125



