#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@Project: CHART_COM
@Author: sol@JinGroup
@File: find_final_result.py
@Time: 9/19/20 11:19 AM
@E-mail: hesuozhang@gmail.com
'''

import json
import os
import os.path as osp
import cv2
import numpy as np


def compute_iou(aa, bb):
    ax1, ay1, ax2, ay2 = aa
    bx1, by1, bx2, by2 = bb

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter

    iou = inter / float(union) if union else 0
    return iou


def search_legend(pred_json_path, bb, margin=5):
    xmin, ymin, xmax, ymax = bb
    with open(pred_json_path, "r") as f:
        tmp = json.loads(f.read())
    if not tmp["legend_bounding"] or not tmp["legend"]:
        return []
    for bbox in tmp["legend_bounding"]:
        bx1, by1, bx2, by2 = bbox
        bx1, by1, bx2, by2 = bx1 - margin, by1 - margin, bx2 + margin, by2 + margin
        if (xmin > bx1) and (ymin > by1) and (xmax < bx2) and (ymax < by2):
            for box in tmp["legend"]:
                x1, y1, x2, y2 = box
                if (x1 > bx1) and (y1 > by1) and (x2 < bx2) and (y2 < by2):
                    return [x1, y1, x2, y2]


def match_contain(margin=5, anno_json_dir="", image_dir="", pred_json_dir="", final_json_dir="", final_json_vis_dir="",
                  data_type=""):
    img_suffix = os.listdir(image_dir)[0][-4:]
    for i, fn in enumerate(os.listdir(image_dir)):
        print(i)
        anno_json_path = osp.join(anno_json_dir, fn.replace(img_suffix, ".json"))
        pred_json_path = osp.join(pred_json_dir, fn.replace(img_suffix, ".json"))
        final_json_path = osp.join(final_json_dir, fn.replace(img_suffix, ".json"))
        final_json_vis_path = osp.join(final_json_vis_dir, fn)
        image_path = osp.join(image_dir, fn)
        im = cv2.imread(image_path)
        final_d = {"task5": {"output": {"legend_pairs": []}}}
        with open(anno_json_path, "r") as f:
            for text_bb in json.loads(f.read())["task5"]["input"]["task2_output"]["text_blocks"]:
                text_id = text_bb["id"]
                if data_type == "real":
                    xmin = min(text_bb["polygon"]["x0"], text_bb["polygon"]["x1"], text_bb["polygon"]["x2"],
                               text_bb["polygon"]["x3"])
                    xmax = max(text_bb["polygon"]["x0"], text_bb["polygon"]["x1"], text_bb["polygon"]["x2"],
                               text_bb["polygon"]["x3"])
                    ymin = min(text_bb["polygon"]["y0"], text_bb["polygon"]["y1"], text_bb["polygon"]["y2"],
                               text_bb["polygon"]["y3"])
                    ymax = max(text_bb["polygon"]["y0"], text_bb["polygon"]["y1"], text_bb["polygon"]["y2"],
                               text_bb["polygon"]["y3"])
                elif data_type == "synthetic":
                    xmin = min(text_bb["bb"]["x0"], text_bb["bb"]["x2"])
                    ymin = min(text_bb["bb"]["y0"], text_bb["bb"]["y2"])
                    xmax = max(text_bb["bb"]["x0"], text_bb["bb"]["x2"])
                    ymax = max(text_bb["bb"]["y0"], text_bb["bb"]["y2"])
                else:
                    xmin = text_bb["bb"]["x0"]
                    ymin = text_bb["bb"]["y0"]
                    xmax = text_bb["bb"]["x0"] + text_bb["bb"]["width"] - 1
                    ymax = text_bb["bb"]["y0"] + text_bb["bb"]["height"] - 1
                legend_bb = search_legend(pred_json_path, [xmin, ymin, xmax, ymax], margin=margin)
                if legend_bb:
                    x1, y1, x2, y2 = legend_bb
                    final_d["task5"]["output"]["legend_pairs"].append(
                        {"bb": {
                            "height": y2 - y1 + 1,
                            "width": x2 - x1 + 1,
                            "x0": x1,
                            "y0": y1
                        },
                            "id": text_id}
                    )
                    cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color=(255, 0, 0))
                    cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 255, 0))
                    cv2.line(im, (x2, y2), (xmin, ymin), color=(0, 0, 255))
        with open(final_json_path, "w") as f:
            f.write(json.dumps(final_d))
        cv2.imwrite(final_json_vis_path, im)


def match_distance(anno_json_dir="", image_dir="", pred_json_dir="", final_json_dir="", final_json_vis_dir="",
                   data_type=""):
    img_suffix = os.listdir(image_dir)[0][-4:]
    for fn in os.listdir(image_dir):
        anno_json_path = osp.join(anno_json_dir, fn.replace(img_suffix, ".json"))
        pred_json_path = osp.join(pred_json_dir, fn.replace(img_suffix, ".json"))
        final_json_path = osp.join(final_json_dir, fn.replace(img_suffix, ".json"))
        final_json_vis_path = osp.join(final_json_vis_dir, fn)
        image_path = osp.join(image_dir, fn)
        im = cv2.imread(image_path)
        final_d = {"task5": {"output": {"legend_pairs": []}}}
        with open(pred_json_path, "r") as f:
            tmp = json.loads(f.read())
        if not tmp["legend"]:
            continue
        with open(anno_json_path, "r") as f:
            all_text = json.loads(f.read())["task5"]["input"]["task2_output"]["text_blocks"]
        for box in tmp["legend"]:
            x1, y1, x2, y2 = box
            target_text_id = None
            tmp_d = 100000
            for text_bb in all_text:
                print(text_bb)
                text_id = text_bb["id"]
                if data_type == "real":
                    xmin = min(text_bb["polygon"]["x0"], text_bb["polygon"]["x1"], text_bb["polygon"]["x2"],
                               text_bb["polygon"]["x3"])
                    xmax = max(text_bb["polygon"]["x0"], text_bb["polygon"]["x1"], text_bb["polygon"]["x2"],
                               text_bb["polygon"]["x3"])
                    ymin = min(text_bb["polygon"]["y0"], text_bb["polygon"]["y1"], text_bb["polygon"]["y2"],
                               text_bb["polygon"]["y3"])
                    ymax = max(text_bb["polygon"]["y0"], text_bb["polygon"]["y1"], text_bb["polygon"]["y2"],
                               text_bb["polygon"]["y3"])
                elif data_type == "synthetic":
                    xmin = min(text_bb["bb"]["x0"], text_bb["bb"]["x2"])
                    ymin = min(text_bb["bb"]["y0"], text_bb["bb"]["y2"])
                    xmax = max(text_bb["bb"]["x0"], text_bb["bb"]["x2"])
                    ymax = max(text_bb["bb"]["y0"], text_bb["bb"]["y2"])
                else:
                    xmin = text_bb["bb"]["x0"]
                    ymin = text_bb["bb"]["y0"]
                    xmax = text_bb["bb"]["x0"] + text_bb["bb"]["width"] - 1
                    ymax = text_bb["bb"]["y0"] + text_bb["bb"]["height"] - 1
                center_x = (xmax + xmin) / 2
                center_y = (ymax + ymin) / 2
                d = abs(max(x1, x2) - center_x) + abs(max(y1, y2) - center_y)
                if d < tmp_d:
                    target_text_id = text_id
                    tmp_d = d

            final_d["task5"]["output"]["legend_pairs"].append(
                {"bb": {
                    "height": y2 - y1 + 1,
                    "width": x2 - x1 + 1,
                    "x0": x1,
                    "y0": y1
                },
                    "id": target_text_id}
            )
            # cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color=(255, 0, 0))
            # cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 255, 0))
            # cv2.line(im, (x2,  y2), (xmin, ymin), color=(0, 0, 255))
        with open(final_json_path, "w") as f:
            f.write(json.dumps(final_d))
        # cv2.imwrite(final_json_vis_path, im)


def match_iou(anno_json_dir="", image_dir="", pred_json_dir="", final_json_dir="", final_json_vis_dir="", data_type=""):
    img_suffix = os.listdir(image_dir)[0][-4:]
    for fn in os.listdir(image_dir):
        anno_json_path = osp.join(anno_json_dir, fn.replace(img_suffix, ".json"))
        if not os.path.exists(anno_json_path):
            breakpoint()
        pred_json_path = osp.join(pred_json_dir, fn.replace(img_suffix, ".json"))
        final_json_path = osp.join(final_json_dir, fn.replace(img_suffix, ".json"))
        final_json_vis_path = osp.join(final_json_vis_dir, fn)
        image_path = osp.join(image_dir, fn)
        im = cv2.imread(image_path)
        final_d = {"task5": {"output": {"legend_pairs": []}}}
        with open(anno_json_path, "r") as f:
            d = json.loads(f.read())
            all_text_bbs = d["task5"]["input"]["task2_output"]["text_blocks"]
            chart_type = d["task5"]["input"]["task1_output"]["chart_type"]
        with open(pred_json_path, "r") as f:
            pred_box = json.loads(f.read())
        if not pred_box["legend_bounding"] or not pred_box["legend"]:
            with open(final_json_path, "w") as f:
                f.write(json.dumps(final_d))
            cv2.imwrite(final_json_vis_path, im)
            continue
        text_lbb_iou_matrix = []
        for i, text_bb in enumerate(all_text_bbs):
            text_id = text_bb["id"]
            if data_type == "real":
                xmin = min(text_bb["polygon"]["x0"], text_bb["polygon"]["x1"], text_bb["polygon"]["x2"],
                           text_bb["polygon"]["x3"])
                xmax = max(text_bb["polygon"]["x0"], text_bb["polygon"]["x1"], text_bb["polygon"]["x2"],
                           text_bb["polygon"]["x3"])
                ymin = min(text_bb["polygon"]["y0"], text_bb["polygon"]["y1"], text_bb["polygon"]["y2"],
                           text_bb["polygon"]["y3"])
                ymax = max(text_bb["polygon"]["y0"], text_bb["polygon"]["y1"], text_bb["polygon"]["y2"],
                           text_bb["polygon"]["y3"])
            elif data_type == "synthetic":
                xmin = min(text_bb["bb"]["x0"], text_bb["bb"]["x2"])
                ymin = min(text_bb["bb"]["y0"], text_bb["bb"]["y2"])
                xmax = max(text_bb["bb"]["x0"], text_bb["bb"]["x2"])
                ymax = max(text_bb["bb"]["y0"], text_bb["bb"]["y2"])
            else:
                xmin = text_bb["bb"]["x0"]
                ymin = text_bb["bb"]["y0"]
                xmax = text_bb["bb"]["x0"] + text_bb["bb"]["width"] - 1
                ymax = text_bb["bb"]["y0"] + text_bb["bb"]["height"] - 1

            for j, lbb in enumerate(pred_box["legend_bounding"]):
                text_lbb_iou_matrix.append(
                    [compute_iou([xmin, ymin, xmax, ymax], lbb), [i, j], [xmin, ymin, xmax, ymax]])

        text_lbb_iou_matrix_sorted = sorted(text_lbb_iou_matrix, key=lambda x: x[0], reverse=True)

        pairs = []
        text_indexes, lbb_indexes = [], []
        for item in text_lbb_iou_matrix_sorted:
            if item[0] < 0.001:
                break
            if item[1][0] not in text_indexes and item[1][1] not in lbb_indexes and len(lbb_indexes) < len(
                    pred_box["legend_bounding"]):
                pairs.append([item[1][0], item[1][1], item[2]])
                text_indexes.append(item[1][0])
                lbb_indexes.append(item[1][1])
        legend_lbb_iou_matrix = []
        for pair in pairs:
            text_index, lbb_index, text_bb = pair
            for k, lb in enumerate(pred_box["legend"]):
                legend_lbb_iou_matrix.append([compute_iou(pred_box["legend_bounding"][lbb_index], lb),
                                              [text_index, k, lbb_index], text_bb])
        legend_lbb_iou_matrix_sorted = sorted(legend_lbb_iou_matrix, key=lambda x: x[0], reverse=True)
        cache_lbb_indexes = []
        for item in legend_lbb_iou_matrix_sorted:
            if item[0] < 0.001:
                break
            lbb_index = item[1][2]
            if lbb_index in cache_lbb_indexes:
                continue
            cache_lbb_indexes.append(lbb_index)
            x1, y1, x2, y2 = pred_box["legend"][item[1][1]]
            text_id = all_text_bbs[item[1][0]]["id"]
            xmin, ymin, xmax, ymax = item[2]
            final_d["task5"]["output"]["legend_pairs"].append(
                {"bb": {
                    "height": y2 - y1 + 1,
                    "width": x2 - x1 + 1,
                    "x0": x1,
                    "y0": y1
                },
                    "id": text_id}
            )
            cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color=(255, 0, 0))
            cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 255, 0))
            cv2.line(im, (x2, y2), (xmin, ymin), color=(0, 0, 255))
        with open(final_json_path, "w") as f:
            f.write(json.dumps(final_d))
        cv2.imwrite(final_json_vis_path, im)


def match(margin=5, anno_json_dir="", image_dir="", pred_json_dir="", final_json_dir="", final_json_vis_dir="",
          data_type="real", mode="iou"):
    if mode == "contain":
        match_contain(margin=margin, anno_json_dir=anno_json_dir, image_dir=image_dir, pred_json_dir=pred_json_dir,
                      final_json_dir=final_json_dir, final_json_vis_dir=final_json_vis_dir, data_type=data_type)
    elif mode == "distance":
        match_distance(anno_json_dir=anno_json_dir, image_dir=image_dir, pred_json_dir=pred_json_dir,
                       final_json_dir=final_json_dir, final_json_vis_dir=final_json_vis_dir, data_type=data_type)
    else:
        match_iou(anno_json_dir=anno_json_dir, image_dir=image_dir, pred_json_dir=pred_json_dir,
                  final_json_dir=final_json_dir, final_json_vis_dir=final_json_vis_dir, data_type=data_type)


if __name__ == "__main__":
    # data_type = "synthetic"
    # data_type = "real"
    data_type = "ic19"
    mode = "contain"
    # mode = "iou"
    # mode = "distance"
    if data_type == "synthetic":
        root = "/path/to/saved path"
        # input
        anno_json_dir = os.path.join(root, "gt_json")  # input json, to get task5's input(task2's text)
        image_dir = os.path.join(root, "test_images")  # to visualization
        pred_json_dir = os.path.join(root, "pred/json_1012")  # output from det_demo.py

        # output
        final_json_dir = os.path.join(root, "pred/final_json_1012")  # match result
        final_json_vis_dir = os.path.join(root, "pred/final_json_1012_vis")  # visualization image saved path
        if not os.path.exists(final_json_dir):
            os.mkdir(final_json_dir)
        if not os.path.exists(final_json_vis_dir):
            os.mkdir(final_json_vis_dir)

        match(anno_json_dir=anno_json_dir, image_dir=image_dir, pred_json_dir=pred_json_dir,
              final_json_dir=final_json_dir, final_json_vis_dir=final_json_vis_dir, data_type=data_type, mode=mode)
    elif data_type == "ic19":
        root = "/path/to/saved path"
        # input
        anno_json_dir = os.path.join(root, "gts/gt_json")  # input json, to get task5's input(task2's text)
        image_dir = os.path.join(root, "png")  # to visualization
        pred_json_dir = os.path.join(root, "pred/pred_json_0923")  # output from det_demo.py

        # output
        final_json_dir = os.path.join(root, "pred/final_json_0923")  # match result
        final_json_vis_dir = os.path.join(root, "pred/final_json_0923_vis")  # visualization image saved path
        if not os.path.exists(final_json_dir):
            os.mkdir(final_json_dir)
        if not os.path.exists(final_json_vis_dir):
            os.mkdir(final_json_vis_dir)

        match(anno_json_dir=anno_json_dir, image_dir=image_dir, pred_json_dir=pred_json_dir,
              final_json_dir=final_json_dir, final_json_vis_dir=final_json_vis_dir, data_type=data_type, mode=mode)
    else:
        root_real = "/path/to/saved path"
        anno_json_dir = os.path.join(root_real, "test_anns")
        image_dir = os.path.join(root_real, "test_images")
        pred_json_dir = os.path.join(root_real, "preds/pred_json_1119")

        # output
        final_json_dir = os.path.join(root_real, "preds/final_json_1119")
        final_json_vis_dir = os.path.join(root_real, "preds/final_json_vis_1119")
        if not os.path.exists(final_json_dir):
            os.mkdir(final_json_dir)
        if not os.path.exists(final_json_vis_dir):
            os.mkdir(final_json_vis_dir)

        match(anno_json_dir=anno_json_dir, image_dir=image_dir, pred_json_dir=pred_json_dir,
              final_json_dir=final_json_dir, final_json_vis_dir=final_json_vis_dir, data_type=data_type, mode=mode)

    os.system("python metric5_2.py %s %s %s" % (
        final_json_dir, anno_json_dir, False))
