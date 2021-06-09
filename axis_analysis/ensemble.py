#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Project: work5_20201215
@Author: sol@JinGroup
@File: ensemble.py
@Time: 1/29/21 3:19 PM
@E-mail: hesuozhang@gmail.com
'''

import torch
from det.utils.coord import transfer_target
import numpy as np
from demo import vis_heatmap
import cv2
import os.path as osp
import os
from PIL import Image
import det.dataset.transform as T


def extract():
    real = True
    img_dir = "/path/to/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/" \
              "splits_with_GT/split_3/images"
    save_dir = "/path/to/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/" \
               "splits_with_GT/split_3/"
    save_model_root = "/path/to/save"
    transform = T.Compose([
        T.FixSize((480, 640)),
        T.ToTensor(),
    ])
    for img_path in os.listdir(img_dir):
        print(img_path)
        img = Image.open(osp.join(img_dir, img_path)).convert('RGB')
        tr_img, _ = transform(img, target=None)

        ori_w, ori_h = img.size
        tr_h, tr_w = tr_img.shape[1:]

        sum_pred = []
        for i in range(1, 11):
            save_model = os.path.join(save_model_root, "m%d" % i,
                                      "%s%s" % (os.path.basename(img_path), ".pth"))
            pred = torch.load(save_model)
            sum_pred.append(pred)
        pred = torch.mean(torch.cat(sum_pred), dim=0).unsqueeze(0)


        coord = transfer_target(pred, thresh=0.2, n_points=None)[0]
        stride = 1

        if len(coord):
            cur_res = np.zeros_like(np.array(coord))
            cur_res[:, 0] = np.array(coord)[:, 0] / tr_w * ori_w * stride
            cur_res[:, 1] = np.array(coord)[:, 1] / tr_h * ori_h * stride
            cur_res = cur_res.astype(np.int32)
        else:
            cur_res = []
        if real:
            ext = ".jpg"
        else:
            ext = ".png"
        f = open(osp.join(save_dir, 'pred', img_path.replace(ext, ".txt")), "w")
        # f = open(osp.join(save_dir, 'pred', img_path.replace("jpg", "txt")), "w")
        for val in cur_res:
            f.write(str(int(val[0])) + "," + str(int(val[1])) + "\n")
        f.close()


if __name__ == "__main__":
    extract()
