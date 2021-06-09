'''
v2 version
Same script to process synthetic, onedrive, real dataset
'''
import os, torch, json, math
import os.path as osp
import numpy as np
from collections import Counter
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint


def write(bb_dir, tick_dir, input_dir, save_dir, coco_json_path, real):
    bb_output = torch.load(bb_dir)

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    search = dict()
    with open(coco_json_path, "r") as f:
        d = json.loads(f.read())
    for item in d["images"]:
        search[item["id"]] = os.path.basename(item["file_name"])

    for idx, det in enumerate(bb_output):
        fname = search[det["image_id"]]
        # if fname!="PMC3323430___1471-2458-12-218-1.jpg":
        #     continue
        print("fname: ", fname)

        pred = {};
        pred['task4'] = {}
        pred['task4']['output'] = {}
        pred['task4']['output']['_plot_bb'] = {}

        # write task4 plotbb
        bbox = det['instances'][0]['bbox']
        pred['task4']['output']['_plot_bb']['height'] = int(bbox[3])
        pred['task4']['output']['_plot_bb']['width'] = int(bbox[2])
        pred['task4']['output']['_plot_bb']['x0'] = int(bbox[0])
        pred['task4']['output']['_plot_bb']['y0'] = int(bbox[1])

        # write task4 axis
        pred['task4']['output']['axes'] = {}
        pred['task4']['output']['axes']['x-axis'] = []
        pred['task4']['output']['axes']['x-tick-type'] = None
        pred['task4']['output']['axes']['x-axis-2'] = []
        pred['task4']['output']['axes']['x2-tick-type'] = None

        pred['task4']['output']['axes']['y-axis'] = []
        pred['task4']['output']['axes']['y-tick-type'] = None
        pred['task4']['output']['axes']['y-axis-2'] = []
        pred['task4']['output']['axes']['y2-tick-type'] = None

        # task1, task2 output
        json_path = fname.replace(fname.split(".")[-1], "json")
        input_data = json.load(open(osp.join(input_dir, json_path)))
        cls_ = input_data['task4']['input']['task1_output']['chart_type']
        task2_box = input_data['task4']['input']['task2_output']['text_blocks']
        task2_boxes_id = {}
        for block in task2_box:
            # polygon format
            polygon = block['bb'] if not real else block['polygon']
            poly = []
            for val in ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3']:
                poly.append(polygon[val])
            task2_boxes_id[block['id']] = poly

        # use plotbb to filter inside boxes
        plotbb = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
        task2_boxes = np.array([v for k, v in task2_boxes_id.items()])
        boxes_ct = np.zeros((task2_boxes.shape[0], 2))
        boxes_ct[:, 0] = np.sum(task2_boxes[:, [0, 2, 4, 6]], axis=1) / 4
        boxes_ct[:, 1] = np.sum(task2_boxes[:, [1, 3, 5, 7]], axis=1) / 4
        invalid = (plotbb[0] < boxes_ct[:, 0]) * (boxes_ct[:, 0] < plotbb[2]) * (plotbb[1] < boxes_ct[:, 1]) * (
                    boxes_ct[:, 1] < plotbb[3])
        valid = np.logical_not(invalid)
        task2_boxes_id = {k: v for idx, (k, v) in enumerate(task2_boxes_id.items()) if valid[idx]}

        if ("horizontal" in cls_) or ("Horizontal" in cls_):
            x_axis_is_vert = True
        else:
            x_axis_is_vert = False

        if not real:
            tick_pred_path = osp.join(tick_dir, fname.replace("png", "txt"))
        else:
            tick_pred_path = osp.join(tick_dir, fname.replace("jpg", "txt"))

        if not osp.exists(tick_pred_path):
            print("filename not found:", fname)
            continue
        tick_pred_data = open(tick_pred_path).read().splitlines()
        tick_preds = [list(map(int, val.split(","))) for val in tick_pred_data]
        hor_ticks, vert_ticks = find_dir_tick(tick_preds, task2_boxes_id, plotbb, x_axis_is_vert, real)

        if x_axis_is_vert:
            write_x = vert_ticks
            write_y = hor_ticks
        else:
            write_x = hor_ticks
            write_y = vert_ticks

        for idx, write_ in enumerate(write_x):
            for val in write_:
                dict_ = {}
                dict_['id'] = val['id']
                dict_['tick_pt'] = {'x': int(val['pt'][0]), 'y': int(val['pt'][1])}

                # OneDrive dataset
                # pred['task4']['output']['axes']['x-axis'].append(dict_)
                if idx == 0:
                    pred['task4']['output']['axes']['x-axis'].append(dict_)
                elif idx == 1:
                    pred['task4']['output']['axes']['x-axis-2'].append(dict_)

        for idx, write_ in enumerate(write_y):
            for val in write_:
                dict_ = {}
                dict_['id'] = val['id']
                dict_['tick_pt'] = {'x': int(val['pt'][0]), 'y': int(val['pt'][1])}

                # OneDrive dataset
                # pred['task4']['output']['axes']['y-axis'].append(dict_)
                if idx == 0:
                    pred['task4']['output']['axes']['y-axis'].append(dict_)
                elif idx == 1:
                    pred['task4']['output']['axes']['y-axis-2'].append(dict_)

        json_file = osp.join(save_dir, fname.replace(fname.split(".")[-1], "json"))
        json.dump(pred, open(json_file, 'w'), indent=4)
    return


def find_dir_tick(tick_preds, task2_boxes_id, plotbb, x_axis_is_vert, real):
    vertical_tick = []
    horizontal_tick = []
    tick_preds = np.array(tick_preds)

    x_val = tick_preds[:, 0];
    y_val = tick_preds[:, 1]
    min_left = min(x_val);
    max_right = max(x_val)
    min_top = min(y_val);
    max_bt = max(y_val)

    cluster_pt_1, cluster_idx_1 = cluster(min_left, x_val, tick_preds)  # vertical direction, left,     type==1
    cluster_pt_2, cluster_idx_2 = cluster(max_right, x_val, tick_preds)  # vertical direction, right,    type==2
    cluster_pt_3, cluster_idx_3 = cluster(min_top, y_val, tick_preds)  # horizontal direction, top,    type==3
    cluster_pt_4, cluster_idx_4 = cluster(max_bt, y_val, tick_preds)  # horizontal direction, bottom, type==4

    cluster_idx_1 = cluster_idx_1.tolist()
    cluster_idx_2 = cluster_idx_2.tolist()
    cluster_idx_3 = cluster_idx_3.tolist()
    cluster_idx_4 = cluster_idx_4.tolist()

    if max_right - min_left < 10 or len(cluster_pt_2) == 1:
        cluster_pt_2 = [];
        cluster_idx_2 = []
    if max_bt - min_top < 10 or len(cluster_pt_3) == 1:
        cluster_pt_3 = [];
        cluster_idx_3 = []

    # when no y axis or x axis
    if len(cluster_pt_1) == 1 and (
            abs(cluster_pt_1[0, 1] - plotbb[1]) < 10 or abs(cluster_pt_1[0, 1] - plotbb[3]) < 10):
        cluster_pt_1 = [];
        cluster_idx_1 = []
    if len(cluster_pt_2) == 1 and (
            abs(cluster_pt_2[0, 1] - plotbb[1]) < 10 or abs(cluster_pt_2[0, 1] - plotbb[3]) < 10):
        cluster_pt_2 = [];
        cluster_idx_2 = []
    if len(cluster_pt_3) == 1 and (
            abs(cluster_pt_3[0, 0] - plotbb[0]) < 10 or abs(cluster_pt_3[0, 0] - plotbb[2]) < 10):
        cluster_pt_3 = [];
        cluster_idx_3 = []
    if len(cluster_pt_4) == 1 and (
            abs(cluster_pt_4[0, 0] - plotbb[0]) < 10 or abs(cluster_pt_4[0, 0] - plotbb[2]) < 10):
        cluster_pt_4 = [];
        cluster_idx_4 = []

    pts = [cluster_pt_1, cluster_pt_2, cluster_pt_3, cluster_pt_4]
    ids = [cluster_idx_1, cluster_idx_2, cluster_idx_3, cluster_idx_4]
    # pts, ids = del_inter_pt(pts, ids, x_axis_is_vert)

    for idx, clus in enumerate(pts):
        matched_tick = find_match_id(real, clus, task2_boxes_id, plotbb, tick_type=idx)
        if idx == 0 or idx == 1:
            vertical_tick.append(matched_tick)
        if idx == 2 or idx == 3:
            horizontal_tick.append(matched_tick)

    # remove tick not belong to same id
    DEBUG = True
    if DEBUG:
        hor_rm_idx = {};
        ver_rm_idx = {}
        for idx1_1, axis_tick_1 in enumerate(horizontal_tick):
            if not len(axis_tick_1):
                continue
            for idx1_2, h_tick in enumerate(axis_tick_1):
                for idx2_1, axis_tick_2 in enumerate(vertical_tick):
                    if not len(axis_tick_2):
                        continue
                    axis2_ids = [val['id'] for val in axis_tick_2]
                    if h_tick['id'] in axis2_ids:
                        for idx2_2, v_tick in enumerate(axis_tick_2):
                            if h_tick['id'] == v_tick['id'] and (h_tick['pt'] == v_tick['pt']).all():
                                if x_axis_is_vert:
                                    if idx2_1 not in ver_rm_idx.keys():
                                        ver_rm_idx[idx2_1] = []
                                    ver_rm_idx[idx2_1].append(idx2_2)
                                else:
                                    if idx1_1 not in hor_rm_idx.keys():
                                        hor_rm_idx[idx1_1] = []
                                    hor_rm_idx[idx1_1].append(idx1_2)

        for k, v in hor_rm_idx.items():
            indices = v
            for i in sorted(indices, reverse=True):
                del horizontal_tick[k][i]

        for k, v in ver_rm_idx.items():
            indices = v
            for i in sorted(indices, reverse=True):
                del vertical_tick[k][i]

    return horizontal_tick, vertical_tick


def find_match_id(real, tick_cluster, task2_boxes_id, plotbb, tick_type):
    output = []
    # ids = []

    if not real:
        tick_radius = 30  # onedrive
    else:
        tick_radius = 10  # real

    task2_ids = [k for k, v in task2_boxes_id.items()]
    boxes = np.array([v for k, v in task2_boxes_id.items()])
    boxes_poly = [Polygon(box.reshape(4, 2)).convex_hull for box in boxes]

    for idx, tick in enumerate(tick_cluster):

        # bp = False
        # if (tick==np.array([52, 448])).all():
        #     bp = True

        try_times = 0
        Flag = True

        while Flag:
            Flag = False
            if tick_type == 0:  # expand to left
                tick_box = np.array([tick[0] - tick_radius, tick[1] - tick_radius / 2,
                                     tick[0], tick[1] + tick_radius / 2])
            elif tick_type == 1:  # expand to right
                tick_box = np.array([tick[0], tick[1] - tick_radius / 2,
                                     tick[0] + tick_radius, tick[1] + tick_radius / 2])
            elif tick_type == 2:  # expand to top
                tick_box = np.array([tick[0] - tick_radius / 2, tick[1] - tick_radius,
                                     tick[0] + tick_radius / 2, tick[1]])
            elif tick_type == 3:  # expand to bottom
                tick_box = np.array([tick[0] - tick_radius / 2, tick[1],
                                     tick[0] + tick_radius / 2, tick[1] + tick_radius])

            tick_poly_pts = tick_box[[0, 1, 2, 1, 2, 3, 0, 3]].reshape(4, 2)
            tick_poly = Polygon(tick_poly_pts).convex_hull

            valid_idx = calc_inter(boxes_poly, tick_poly)
            relative = calc_relative(boxes, tick, valid_idx)

            if 1 == len(valid_idx):
                output.append({'id': task2_ids[valid_idx[0]], 'pt': tick})
                # ids.append(task2_ids[valid_idx[0]])
            elif 2 <= len(valid_idx) <= 3:
                if tick_type == 0 or tick_type == 1:
                    delta = np.where(relative[:, 1] < 10)[0].tolist()
                    if len(delta) == 1:
                        output.append({'id': task2_ids[valid_idx[delta[0]]], 'pt': tick})
                        # ids.append(task2_ids[valid_idx[delta[0]]])
                    elif len(delta) > 1:
                        Flag = True
                        tick_radius -= 3;
                        try_times += 1

                if tick_type == 2 or tick_type == 3:
                    delta = np.where(relative[:, 0] < 10)[0].tolist()
                    if len(delta) == 1:
                        output.append({'id': task2_ids[valid_idx[delta[0]]], 'pt': tick})
                        # ids.append(task2_ids[valid_idx[delta[0]]])
                    elif len(delta) > 1:
                        Flag = True
                        tick_radius -= 3;
                        try_times += 1

            elif len(valid_idx) > 3:
                Flag = True
                tick_radius -= 3;
                try_times += 1
            elif len(valid_idx) == 0:
                Flag = True
                tick_radius += 3;
                try_times += 1

            if try_times > 10:
                break

    if len(output):
        output = post_proc(output, boxes, task2_ids, tick_type, real=True)

    return output


def del_inter_pt(pts, ids, x_axis_is_vert):
    cluster_pt_1, cluster_pt_2, cluster_pt_3, cluster_pt_4 = pts
    cluster_idx_1, cluster_idx_2, cluster_idx_3, cluster_idx_4 = ids
    output = []

    inter_1_3 = list(set(cluster_idx_1).intersection(set(cluster_idx_3)))  # left and top
    inter_1_4 = list(set(cluster_idx_1).intersection(set(cluster_idx_4)))  # left and bottom
    inter_2_3 = list(set(cluster_idx_2).intersection(set(cluster_idx_3)))  # right and top
    inter_2_4 = list(set(cluster_idx_2).intersection(set(cluster_idx_4)))  # right and bottom

    inters = [inter_1_3, inter_1_4, inter_2_3, inter_2_4]
    use_ids = [(1, 3), (1, 4), (2, 3), (2, 4)]
    for (val1, val2) in zip(inters, use_ids):
        if len(val1) == 1 and x_axis_is_vert:
            # x_axis is vertical, del pt in vertical direction
            chose_to_del = val2[0] - 1
            cluster_pt = pts[chose_to_del];
            cluster_id = ids[chose_to_del]
            idx = cluster_id.index(val1[0])
            cluster_id.remove(cluster_id[idx])
            cluster_pt = np.delete(cluster_pt, idx, axis=0)
            # update
            ids[chose_to_del] = cluster_id
            pts[chose_to_del] = cluster_pt

        if len(val1) == 1 and not x_axis_is_vert:
            # x_axis is horizontal, del pt in horizontal direction
            chose_to_del = val2[1] - 1
            cluster_pt = pts[chose_to_del];
            cluster_id = ids[chose_to_del]
            idx = cluster_id.index(val1[0])
            cluster_id.remove(cluster_id[idx])
            cluster_pt = np.delete(cluster_pt, idx, axis=0)
            # update
            ids[chose_to_del] = cluster_id
            pts[chose_to_del] = cluster_pt
    return pts, ids


def calc_inter(boxes_poly, tick_poly):
    areas = []
    for box_poly in boxes_poly:
        inter_area = box_poly.intersection(tick_poly).area
        areas.append(inter_area)
    areas = np.array(areas)
    idx = np.where(areas)[0]
    return idx


def calc_relative(boxes, tick, valid_idx):
    cand_boxes = boxes[valid_idx]
    cand_x_ct = ((cand_boxes[:, 0] + cand_boxes[:, 2]) / 2).reshape(-1, 1)
    cand_y_ct = ((cand_boxes[:, 1] + cand_boxes[:, 5]) / 2).reshape(-1, 1)
    cand_ct = np.concatenate((cand_x_ct, cand_y_ct), axis=1)
    relative = np.abs(cand_ct - tick)
    return relative


def cluster(val, cand, tick_preds):
    '''
    return idx belong to same cluster
    '''
    thresh = 10
    chose = np.abs(val - cand) < thresh
    idx = np.where(chose)[0]
    cluster_preds = tick_preds[idx]
    return cluster_preds, idx


def post_proc(output, boxes, task2_ids, tick_type, real):
    # 1.filter when multi pt point towards same text block
    # 2.filter when one point towards outer most
    ids = [val['id'] for val in output]
    id_pts = {}
    for id_ in list(set(ids)):
        id_pts[id_] = []
        for val in output:
            if id_ == val['id']:
                id_pts[id_].append(val['pt'])

    # 1.
    final_output = []
    if len(ids) != len(list(set(ids))):
        for id_ in list(set(ids)):
            cand_pt = np.array(id_pts[id_])
            cur = {}
            cur['id'] = id_
            if len(cand_pt) == 1:
                cur['pt'] = cand_pt[0]
            else:
                task2_box = boxes[task2_ids.index(id_)]
                x_ct = task2_box[[0, 2, 4, 6]].sum() / 4
                y_ct = task2_box[[1, 3, 5, 7]].sum() / 4
                task2_box_ct = np.array([x_ct, y_ct]).astype(np.int32)
                if tick_type == 0 or tick_type == 1:
                    relative = np.abs(task2_box_ct[1] - cand_pt[:, 1])
                    cur['pt'] = cand_pt[np.argsort(relative)[0]]
                elif tick_type == 2 or tick_type == 3:
                    relative = np.abs(task2_box_ct[0] - cand_pt[:, 0])
                    cur['pt'] = cand_pt[np.argsort(relative)[0]]
            final_output.append(cur)
    else:
        final_output = output

    # 2.
    # chose_task2_boxes = []
    # ids = [val['id'] for val in final_output]
    # for id_ in ids:
    #     chose_task2_boxes.append(boxes[task2_ids.index(id_)])
    # chose_task2_boxes = np.array(chose_task2_boxes)
    # chose_boxes_ct = np.zeros((chose_task2_boxes.shape[0], 2))
    # chose_boxes_ct[:,0] = np.sum(chose_task2_boxes[:, [0,2,4,6]], axis=1)/4
    # chose_boxes_ct[:,1] = np.sum(chose_task2_boxes[:, [1,3,5,7]], axis=1)/4

    # if tick_type==0 or tick_type==1:
    #     mean = np.mean(chose_boxes_ct[:,0])
    #     std  = np.std( chose_boxes_ct[:,0])
    #     rule = np.abs((chose_boxes_ct[:,0]-mean)/std)
    #     final_output_2 = [val for idx, val in enumerate(final_output) if rule[idx]<1.5]
    #     import pdb; pdb.set_trace()
    # elif tick_type==2 or tick_type==3:
    #     mean = np.mean(chose_boxes_ct[:,1])
    #     std  = np.std( chose_boxes_ct[:,1])
    #     rule = np.abs((chose_boxes_ct[:,1]-mean)/std)
    #     final_output_2 = [val for idx, val in enumerate(final_output) if rule[idx]<1.5]
    #     import pdb; pdb.set_trace()
    # return final_output_2
    return final_output


if __name__ == "__main__":
    # OneDrive dataset
    bb_dir = "/home/sol/Project/detectron2/projects/onedrive/inference/instances_predictions.pth"
    tick_dir = "/home/sol/Project/CHART_COM/work5_20201215/demo/onedrive_sol_baseline_0107/pred"
    input_dir = "/home/sol/Project/CHART_COM/work5_20201215/evaluation/OneDrive/task4_json"
    save_dir = "/home/sol/Project/CHART_COM/work5_20201215/evaluation/OneDrive/prediction2"

    coco_json_path = "/home/sol/Project/CHART_COM/work5_20201215/dataset/annotations/onedrive_plot_bb_test2020.json"

    write(bb_dir, tick_dir, input_dir, save_dir, coco_json_path, real=False)

    # Real dataset
    # bb_dir = "/home/pci/disk1/maweihong/experiment/exp3_ICPR_CHART/detectron2/projects/ICPR/output/exp1_task4_real_cascade/inference/ICPR_task4_real_test/instances_predictions.pth"
    # tick_dir  = "/home/pci/disk1/maweihong/experiment/exp3_ICPR_CHART/Task4/work1/checkpoints/debug0_real_data/inference_46_0.4"
    # input_dir = "/home/pci/disk1/maweihong/experiment/exp3_ICPR_CHART/Task4/evaluation/real/1.21_task4"
    # save_dir  = "real/prediction"

    # write(bb_dir, tick_dir, input_dir, save_dir, real=True)
