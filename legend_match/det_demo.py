#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Project: CHART-DeMatch
@Author: sol@JinGroup
@File: det_demo.py
@Time: 6/9/21 1:16 PM
@E-mail: hesuozhang@gmail.com
'''

import argparse

import cv2
import torch
import os
from pycocotools.coco import COCO
import shutil
import mmcv
import numpy as np
from mmdet.apis import inference_detector, init_detector
import matplotlib.pyplot as plt
import json


def save_det(json_path, result, score_thr):
    L = []
    d = {"legend": [], "legend_bounding": []}
    for i in range(2):
        for line in result[0][i]:
            box = line[:4].astype(np.int).tolist()
            score = line[4]
            if score < score_thr:
                continue
            L.append("%d,%d,%d,%d,%f\n" % (box[0], box[1], box[2], box[3], score))
            if i == 0:
                d["legend"].append(box)
            else:
                d["legend_bounding"].append(box)
    with open(json_path, "w") as f:
        f.write(json.dumps(d))


def main():
    config = "configs/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py"
    checkpoint = "work_dirs/hrnet_0115/epoch_21.pth"
    image_root = "/path/to/images"
    cache_json_root = "/path/to/save path"

    if not os.path.exists(cache_json_root):
        os.mkdir(cache_json_root)

    model = init_detector(
        config, checkpoint, device=torch.device('cuda', 0))

    all_fns = os.listdir(image_root)
    import random;random.shuffle(all_fns)
    for i, fn in enumerate(all_fns):
        image_path = os.path.join(image_root, fn)
        json_path = os.path.join(cache_json_root, "%s.json" % fn[:-4])
        if not os.path.exists(image_path):
            continue
        print(i, image_path, save_path)
        img = cv2.imread(image_path)
        result = inference_detector(model, img)
        save_det(json_path, result, score_thr=0.6)

        if hasattr(model, 'module'):
            model = model.module
        img = model.show_result(img, result, score_thr=0.5, show=False)


if __name__ == '__main__':
    main()