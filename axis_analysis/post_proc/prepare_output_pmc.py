'''
v4 version
script to process real test dataset
'''

import os, torch, json, math
import os.path as osp
import numpy as np
from collections import Counter
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint


def write(bb_dir, tick_dir, input_dir, save_dir, coco_json_path, task7_test):
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
        # if fname!='PMC3112134___1471-2458-11-327-3.txt':
        #     continue
        print("fname: ", fname)


        pred = {};
        pred['task4'] = {}
        pred['task4']['input'] = {}
        pred['task4']['output'] = {}
        pred['task4']['output']['_plot_bb'] = {}

        # write task4 plotbb
        if det['instances']:
            bbox = det['instances'][0]['bbox']
            pred['task4']['output']['_plot_bb']['height'] = int(bbox[3])
            pred['task4']['output']['_plot_bb']['width'] = int(bbox[2])
            pred['task4']['output']['_plot_bb']['x0'] = int(bbox[0])
            pred['task4']['output']['_plot_bb']['y0'] = int(bbox[1])
            # use plotbb to filter inside boxes
            plotbb = [int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
        else:
            plotbb = []


        # write task4 axis
        pred['task4']['output']['axes'] = {}
        # pred['task4']['output']['axes']['x-tick-type'] = None
        # pred['task4']['output']['axes']['x2-tick-type'] = None
        # pred['task4']['output']['axes']['y-tick-type'] = None
        # pred['task4']['output']['axes']['y2-tick-type'] = None

        pred['task4']['output']['axes']['x-axis'] = []
        pred['task4']['output']['axes']['x-axis-2'] = []
        pred['task4']['output']['axes']['y-axis'] = []
        pred['task4']['output']['axes']['y-axis-2'] = []

        # task1, task2 output
        json_path = fname.replace(fname.split(".")[-1], "json")
        input_data = json.load(open(osp.join(input_dir, json_path)))

        if not task7_test:
            pred['task4']['input']['task1_output'] = input_data['task4']['input']['task1_output']
            pred['task4']['input']['task2_output'] = input_data['task4']['input']['task2_output']
            cls_ = input_data['task4']['input']['task1_output']['chart_type']
            task2_box = input_data['task4']['input']['task2_output']['text_blocks']
        else:
            pred['task4']['input']['task1_output'] = input_data['task5']['input']['task1_output']
            pred['task4']['input']['task2_output'] = input_data['task5']['input']['task2_output']
            cls_ = input_data['task5']['input']['task1_output']['chart_type']
            task2_box = input_data['task5']['input']['task2_output']['text_blocks']

        task2_boxes_id = {}
        for block in task2_box:
            # polygon format
            polygon = block['polygon']
            poly = []
            for val in ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3']:
                poly.append(polygon[val])
            task2_boxes_id[block['id']] = poly

        task2_boxes = np.array([v for k, v in task2_boxes_id.items()])
        boxes_ct = np.zeros((task2_boxes.shape[0], 2))
        boxes_ct[:, 0] = np.sum(task2_boxes[:, [0, 2, 4, 6]], axis=1) / 4
        boxes_ct[:, 1] = np.sum(task2_boxes[:, [1, 3, 5, 7]], axis=1) / 4
        if plotbb:
            invalid = (plotbb[0] < boxes_ct[:, 0]) * (boxes_ct[:, 0] < plotbb[2]) * (plotbb[1] < boxes_ct[:, 1]) * (
                    boxes_ct[:, 1] < plotbb[3])
            valid = np.logical_not(invalid)
            if len(valid) < len(task2_boxes_id.keys()):
                task2_boxes_id = {k: v for idx, (k, v) in enumerate(task2_boxes_id.items()) if valid[idx]}
        if not task2_boxes_id:
            breakpoint()

        if ("horizontal" in cls_) or ("Horizontal" in cls_):
            x_axis_is_vert = True
        else:
            x_axis_is_vert = False

        tick_pred_path = osp.join(tick_dir, fname.replace("jpg", "txt"))
        if not osp.exists(tick_pred_path):
            print("filename not found:", fname)
            continue

        tick_pred_data = open(tick_pred_path).read().splitlines()
        tick_preds = [list(map(int, val.split(","))) for val in tick_pred_data]
        hor_ticks, vert_ticks = find_dir_tick(tick_preds, task2_boxes_id, plotbb, x_axis_is_vert)

        if x_axis_is_vert:
            write_x = vert_ticks
            write_y = hor_ticks
        else:
            write_x = hor_ticks
            write_y = vert_ticks

        len_1 = np.argsort((-1) * np.array([len(val) for val in write_x])).tolist()
        write_x = [write_x[idx] for idx in len_1]
        len_2 = np.argsort((-1) * np.array([len(val) for val in write_y])).tolist()
        write_y = [write_y[idx] for idx in len_2]

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


def find_dir_tick(tick_preds, task2_boxes_id, plotbb, x_axis_is_vert):
    vertical_tick = []
    horizontal_tick = []
    tick_preds = np.array(tick_preds)
    clus_thresh = 13

    try:
        x_val = tick_preds[:, 0]
        y_val = tick_preds[:, 1]
    except:
        return [], []
    min_left = min(x_val);
    max_right = max(x_val)
    min_top = min(y_val);
    max_bt = max(y_val)

    cluster_pt_1, cluster_idx_1 = cluster(min_left, x_val, tick_preds,
                                          clus_thresh)  # vertical direction, left,     type==1
    cluster_pt_2, cluster_idx_2 = cluster(max_right, x_val, tick_preds,
                                          clus_thresh)  # vertical direction, right,    type==2
    cluster_pt_3, cluster_idx_3 = cluster(min_top, y_val, tick_preds,
                                          clus_thresh)  # horizontal direction, top,    type==3
    cluster_pt_4, cluster_idx_4 = cluster(max_bt, y_val, tick_preds,
                                          clus_thresh)  # horizontal direction, bottom, type==4

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

    if plotbb:
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

    for idx, clus in enumerate(pts):  # 遍历四个方向
        matched_tick = find_match_id(clus, task2_boxes_id, plotbb, tick_type=idx)
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

                                bb = task2_boxes_id[h_tick['id']]
                                bb_ct = [np.array(bb)[[0, 2, 4, 6]].sum() / 4, np.array(bb)[[1, 3, 5, 7]].sum() / 4]
                                tick_pt = h_tick['pt']
                                if abs(tick_pt[0] - bb_ct[0]) > abs(tick_pt[1] - bb_ct[1]):
                                    if idx1_1 not in hor_rm_idx.keys():
                                        hor_rm_idx[idx1_1] = []
                                    hor_rm_idx[idx1_1].append(idx1_2)
                                else:
                                    if idx2_1 not in ver_rm_idx.keys():
                                        ver_rm_idx[idx2_1] = []
                                    ver_rm_idx[idx2_1].append(idx2_2)

                                # if x_axis_is_vert:
                                #     if idx2_1 not in ver_rm_idx.keys():
                                #         ver_rm_idx[idx2_1]=[]
                                #     ver_rm_idx[idx2_1].append(idx2_2)
                                # else:
                                #     if idx1_1 not in hor_rm_idx.keys():
                                #         hor_rm_idx[idx1_1]=[]
                                #     hor_rm_idx[idx1_1].append(idx1_2)

        for k, v in hor_rm_idx.items():
            indices = v
            for i in sorted(indices, reverse=True):
                del horizontal_tick[k][i]

        for k, v in ver_rm_idx.items():
            indices = v
            for i in sorted(indices, reverse=True):
                del vertical_tick[k][i]

    return horizontal_tick, vertical_tick


def find_match_id(tick_cluster, task2_boxes_id, plotbb, tick_type):
    output = []

    ratio = 2
    tick_radius = 15  # real

    task2_ids = [k for k, v in task2_boxes_id.items()]
    boxes = np.array([v for k, v in task2_boxes_id.items()])
    boxes_ct = np.zeros((boxes.shape[0], 2))

    boxes_ct[:, 0] = np.sum(boxes[:, [0, 2, 4, 6]], axis=1) / 4
    boxes_ct[:, 1] = np.sum(boxes[:, [1, 3, 5, 7]], axis=1) / 4
    boxes_poly = [Polygon(box.reshape(4, 2)).convex_hull for box in boxes]

    # sort tick cluster
    if len(tick_cluster):
        if tick_type == 0 or tick_type == 1:
            tick_cluster = tick_cluster[np.argsort(tick_cluster[:, 1])]
        elif tick_type == 2 or tick_type == 3:
            tick_cluster = tick_cluster[np.argsort(tick_cluster[:, 0])]

    for idx, tick in enumerate(tick_cluster):
        # bp = False
        # if (tick==np.array([335, 271])).all():
        #     bp = True
        #     import pdb; pdb.set_trace()

        try_times = 0
        Flag = True

        while Flag:
            Flag = False
            if tick_type == 0:  # expand to left
                tick_box = np.array([tick[0] - tick_radius, tick[1] - tick_radius / ratio,
                                     tick[0], tick[1] + tick_radius / ratio])
            elif tick_type == 1:  # expand to right
                tick_box = np.array([tick[0], tick[1] - tick_radius / ratio,
                                     tick[0] + tick_radius, tick[1] + tick_radius / ratio])
            elif tick_type == 2:  # expand to top
                tick_box = np.array([tick[0] - tick_radius / ratio, tick[1] - tick_radius,
                                     tick[0] + tick_radius / ratio, tick[1]])
            elif tick_type == 3:  # expand to bottom
                tick_box = np.array([tick[0] - tick_radius / ratio, tick[1],
                                     tick[0] + tick_radius / ratio, tick[1] + tick_radius])

            tick_poly_pts = tick_box[[0, 1, 2, 1, 2, 3, 0, 3]].reshape(4, 2)
            tick_poly = Polygon(tick_poly_pts).convex_hull

            valid_idx = calc_inter(boxes_poly, tick_poly)
            relative = calc_relative(boxes, tick, valid_idx)

            if 1 == len(valid_idx):
                output.append({'id': task2_ids[valid_idx[0]], 'pt': tick})
            # elif len(valid_idx)==2:
            #     bb_1 = task2_boxes_id[task2_ids[valid_idx[0]]]
            #     bb_1_ct = [np.array(bb_1)[[0,2,4,6]].sum()/4, np.array(bb_1)[[1,3,5,7]].sum()/4]
            #     bb_2 = task2_boxes_id[task2_ids[valid_idx[1]]]
            #     bb_2_ct = [np.array(bb_2)[[0,2,4,6]].sum()/4, np.array(bb_2)[[1,3,5,7]].sum()/4]

            #     if (tick_type==0 or tick_type==1):
            #         if abs(bb_1_ct[0]-tick[0])>abs(bb_2_ct[0]-tick[0]):
            #             output.append({'id': task2_ids[valid_idx[0]], 'pt': tick})
            #         else:
            #             output.append({'id': task2_ids[valid_idx[1]], 'pt': tick})
            #     if (tick_type==2 or tick_type==3):
            #         if abs(bb_1_ct[1]-tick[1])>abs(bb_2_ct[1]-tick[1]):
            #             output.append({'id': task2_ids[valid_idx[0]], 'pt': tick})
            #         else:
            #             output.append({'id': task2_ids[valid_idx[1]], 'pt': tick})

            elif len(valid_idx) == 2 and (tick_type == 0 or tick_type == 1) and (
                    idx == 0 or idx == len(tick_cluster) - 1):
                bb_1 = task2_boxes_id[task2_ids[valid_idx[0]]]
                bb_1_ct = [np.array(bb_1)[[0, 2, 4, 6]].sum() / 4, np.array(bb_1)[[1, 3, 5, 7]].sum() / 4]
                bb_2 = task2_boxes_id[task2_ids[valid_idx[1]]]
                bb_2_ct = [np.array(bb_2)[[0, 2, 4, 6]].sum() / 4, np.array(bb_2)[[1, 3, 5, 7]].sum() / 4]

                if abs(bb_1_ct[0] - tick[0]) > abs(bb_2_ct[0] - tick[0]):
                    output.append({'id': task2_ids[valid_idx[0]], 'pt': tick})
                else:
                    output.append({'id': task2_ids[valid_idx[1]], 'pt': tick})

            elif len(valid_idx) == 2 and (tick_type == 2 or tick_type == 3) and (
                    idx == 0 or idx == len(tick_cluster) - 1):
                bb_1 = task2_boxes_id[task2_ids[valid_idx[0]]]
                bb_1_ct = [np.array(bb_1)[[0, 2, 4, 6]].sum() / 4, np.array(bb_1)[[1, 3, 5, 7]].sum() / 4]
                bb_2 = task2_boxes_id[task2_ids[valid_idx[1]]]
                bb_2_ct = [np.array(bb_2)[[0, 2, 4, 6]].sum() / 4, np.array(bb_2)[[1, 3, 5, 7]].sum() / 4]

                if abs(bb_1_ct[1] - tick[1]) > abs(bb_2_ct[1] - tick[1]):
                    output.append({'id': task2_ids[valid_idx[0]], 'pt': tick})
                else:
                    output.append({'id': task2_ids[valid_idx[1]], 'pt': tick})

            elif 2 <= len(valid_idx) <= 3:
                if tick_type == 0 or tick_type == 1:
                    delta = np.where(relative[:, 1] < 10)[0].tolist()
                    if len(delta) == 1:
                        output.append({'id': task2_ids[valid_idx[delta[0]]], 'pt': tick})
                    elif len(delta) > 1:
                        Flag = True
                        tick_radius -= 3;
                        try_times += 1
                if tick_type == 2 or tick_type == 3:
                    delta = np.where(relative[:, 0] < 10)[0].tolist()
                    if len(delta) == 1:
                        output.append({'id': task2_ids[valid_idx[delta[0]]], 'pt': tick})
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
        output, num = post_proc(output, boxes, task2_ids, tick_type)
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


def cluster(val, cand, tick_preds, thresh):
    '''
    return idx belong to same cluster
    '''
    chose = np.abs(val - cand) < thresh
    idx = np.where(chose)[0]
    cluster_preds = tick_preds[idx]
    return cluster_preds, idx


def post_proc(output, boxes, task2_ids, tick_type):
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
    num = 0  # num of pt to drop
    if len(ids) != len(list(set(ids))):
        for id_ in list(set(ids)):
            cand_pt = np.array(id_pts[id_])
            cur = {}
            cur['id'] = id_
            if len(cand_pt) == 1:
                cur['pt'] = cand_pt[0]
            else:
                num += cand_pt.shape[0] - 1
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
    return final_output, num


if __name__ == "__main__":
    # Real dataset
    # bb_dir = "real_test/plotbb/pred"
    # tick_dir  = "/home/pci/disk1/maweihong/experiment/exp3_ICPR_CHART/Task4/work1/demo/debug1_real_alldata_add/pred"
    # input_dir = "/home/pci/disk1/maweihong/dataset/ICPR2020_CHART/2020ICPR_UB_PMC_TRAIN/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/split_3/annotations"
    # save_dir  = "real_test/prediction"
    # write(bb_dir, tick_dir, input_dir, save_dir, task7_test=False)

    # Real task7 dataset
    # bb_dir = "/home/sol/Project/detectron2/projects/pmc/inference/instances_predictions.pth"
    bb_dir = "/home/sol/Project/detectron2/projects/pmc_icpr_testset2/inference/instances_predictions.pth"
    # tick_dir = "/home/sol/Project/CHART_COM/work5_20201215/demo/pmc_sol_baseline_1229/pred"
    # input_dir = "/home/sol/Project/CHART_COM/work5_20201215/evaluation/real/annotations"
    # save_dir = "/home/sol/Project/CHART_COM/work5_20201215/evaluation/real/prediction2"
    tick_dir = "/home/sol/data/Datasets/CHART/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/" \
               "splits_with_GT/split_3/pred"
    input_dir = "/home/sol/data/Datasets/CHART/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/" \
                "splits_with_GT/split_3/annotations_JSON"
    save_dir = "/home/sol/data/Datasets/CHART/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/" \
               "splits_with_GT/split_3/prediction2"
    coco_json_path = "/home/sol/Project/CHART_COM/work5_20201215/dataset/annotations/pmc_plot_bb_test2020_icpr.json"
    write(bb_dir, tick_dir, input_dir, save_dir, coco_json_path, task7_test=True)
