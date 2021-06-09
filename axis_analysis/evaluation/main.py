import os, json, shutil
import torch
import os.path as osp
import numpy as np
from evaluation.metric4_pmc import eval_task4

IOU_THRESH = 0.5


def rect_iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h
    iou = area / (s1 + s2 - area)
    return iou


def load_gt_plotbb(read_dir):
    gt_polygons = {}
    for file in os.listdir(read_dir):
        data = json.load(open(osp.join(read_dir, file)))
        plotbb = data['task4']['output']['_plot_bb']
        bbox = [plotbb['x0'], plotbb['y0'], plotbb['x0'] + plotbb['width'], plotbb['y0'] + plotbb['height']]
        gt_polygons[file] = [bbox]
    return gt_polygons


def load_pred_plotbb(pth_path):
    output_res = {}
    preds = torch.load(pth_path)

    for idx, det in enumerate(preds):
        # fname = det['file_name']
        fname = det['file_name'].replace(det['file_name'].split(".")[-1], "json")
        bbox = det['instances'][0]['bbox']
        l = int(bbox[0]);
        t = int(bbox[1])
        right = int(bbox[0] + bbox[2]);
        bt = int(bbox[1] + bbox[3])
        output_res[fname] = [[l, t, right, bt]]
    return output_res


def eval_bb(gt_dict, dt_dict):
    all_dt_match = []
    all_dt_scores = []

    n_gt = 0
    num_gt = len(gt_dict)
    num_dt = len(dt_dict)
    print("Number of gt files", num_gt)
    print("Number of dt files", num_dt)

    for idx, (dt_file, dt) in enumerate(dt_dict.items()):
        dt_polys = dt
        gt_polys = gt_dict[dt_file]
        n_gt += len(gt_polys)

        dt_match = []
        gt_match = [False for i in range(len(gt_polys))]
        for idx1, dt_poly in enumerate(dt_polys):
            match = False
            for idx2, gt_poly in enumerate(gt_polys):
                if gt_match[idx2]:
                    continue
                iou = rect_iou(dt_poly, gt_poly)
                if iou >= IOU_THRESH:
                    match = True
                    gt_match[idx2] = True

            dt_match.append(match)
        all_dt_match.extend(dt_match)

        # calculate scores and append to list
        dt_scores = [1.] * len(dt)
        all_dt_scores.extend(dt_scores)

    # calculate precision, recall and f-measure at all thresholds
    all_dt_match = np.array(all_dt_match, dtype=np.bool).astype(np.int)
    all_dt_scores = np.array(all_dt_scores)
    sort_idx = np.argsort(all_dt_scores)[::-1]  # sort in descending order
    all_dt_match = all_dt_match[sort_idx]
    all_dt_scores = all_dt_scores[sort_idx]

    n_pos = np.cumsum(all_dt_match)
    n_dt = np.arange(1, len(all_dt_match) + 1)
    precision = n_pos.astype(np.float) / n_dt.astype(np.float)
    recall = n_pos.astype(np.float) / float(n_gt)
    eps = 1e-9
    fmeasure = 2.0 / ((1.0 / (precision + eps)) + (1.0 / (recall + eps)))

    # find maximum fmeasure
    max_idx = np.argmax(fmeasure)

    eval_results = {
        'fmeasure': fmeasure[max_idx],
        'precision': precision[max_idx],
        'recall': recall[max_idx],
        'threshold': all_dt_scores[max_idx],
        'all_precisions': precision,
        'all_recalls': recall
    }

    # evaluation summary
    print('Maximum f-measure: %f' % eval_results['fmeasure'])
    print('    |-- precision: %f' % eval_results['precision'])
    print('    |-- recall:    %f' % eval_results['recall'])
    print('    |-- threshold: %f' % eval_results['threshold'])
    print('=================================================================')
    return


def analyse_error(file, vis_dir, save_dir):
    data = open(file).read().splitlines()
    for line in data:
        path = line.split(" ")[2]
        shutil.copy(osp.join(vis_dir, path.replace("json", "jpg")), osp.join(save_dir, path.replace("json", "jpg")))
    return


if __name__ == "__main__":
    # real
    gt_folder = "/path/to/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/" \
                "ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_3/annotations_JSON"
    result_folder = "/path/to/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/" \
                    "ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_3/prediction"
    img_folder = "/path/to/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/" \
                 "ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_3/images"
    eval_task4(gt_folder, result_folder, img_folder)

