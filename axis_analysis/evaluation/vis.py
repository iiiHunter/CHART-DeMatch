'''
Visualize prediction folder
'''
import os, cv2, json
import os.path as osp
import numpy as np

# for onedrive
# read_dir       = "../post_proc/OneDrive/prediction"
# input_json_dir = "OneDrive/task4_json"
# img_dir        = "OneDrive/task4_img"
# save_dir       = "../post_proc/OneDrive/vis"

# for real
# read_dir       = "../post_proc/real/prediction"
# input_json_dir = "real/1.21_task4"
# img_dir        = "/home/pci/disk1/maweihong/dataset/ICPR2020_CHART/2020ICPR_UB_PMC_TRAIN/2020ICPR_UB_PMC_TRAIN_v1.21/task4/images"
# save_dir       = "../post_proc/real/vis"

# for onedrive test
# read_dir       = "../post_proc/OneDrive_test/prediction"
# input_json_dir = "/home/pci/disk1/maweihong/dataset/ICPR2020_CHART/2020_Add_Synthetic_OneDrive/Adobe Synth Test Dataset/task_3_4_5/Inputs"
# img_dir        = "/home/pci/disk1/maweihong/dataset/ICPR2020_CHART/2020_Add_Synthetic_OneDrive/Adobe Synth Test Dataset/task_3_4_5/Charts"
# save_dir       = "../post_proc/OneDrive_test/vis"
# real = False

# for onedrive task7 test
# read_dir       = "../post_proc/OneDrive_task7_test/prediction"
# input_json_dir = "/home/pci/disk1/maweihong/experiment/exp3_ICPR_CHART/Task7/OneDrive/output_task1-2"
# img_dir        = "/home/pci/disk1/maweihong/dataset/ICPR2020_CHART/2020_Add_Synthetic_OneDrive/Adobe Synth Test Dataset/task_7/Charts"
# save_dir       = "../post_proc/OneDrive_task7_test/vis"
# real           = False

# for real test
# read_dir       = "../post_proc/real_test/prediction"
# input_json_dir = "/home/pci/disk1/maweihong/dataset/ICPR2020_CHART/2020ICPR_UB_PMC_TRAIN/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/split_3/annotations"
# img_dir        = "/home/pci/disk1/maweihong/dataset/ICPR2020_CHART/2020ICPR_UB_PMC_TRAIN/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/split_3/images"
# save_dir       = "../post_proc/real_test/vis"
# real = True

# for real task7 test
input_json_dir = "/home/sol/Project/CHART_COM/work5_20201215/evaluation/real/annotations"
read_dir = "/home/sol/Project/CHART_COM/work5_20201215/evaluation/real/prediction"
img_dir = "/home/sol/Project/CHART_COM/work5_20201215/evaluation/real/images"
save_dir = "/home/sol/Project/CHART_COM/work5_20201215/evaluation/real/vis"
real = True

if not osp.exists(save_dir):
    os.makedirs(save_dir)


def find_by_id(_id, objs):
    for obj in objs:
        if obj['id'] == _id:
            return obj
    return None


for file in os.listdir(read_dir):
    # if file!='PMC3039587___1471-2458-11-52-3.json':
    #     continue
    if not real:
        im = cv2.imread(osp.join(img_dir, file.replace(file.split('.')[-1], 'png')))
    else:
        im = cv2.imread(osp.join(img_dir, file.replace(file.split('.')[-1], 'jpg')))
    input_data = json.load(open(osp.join(input_json_dir, file)))

    # input_bb = input_data['task4']['input']['task2_output']['text_blocks']
    # input_bb = input_data['task2_output']['text_blocks']
    # for task7 test
    input_bb = input_data['task5']['input']['task2_output']['text_blocks']

    id_bb = {}
    for val in input_bb:
        # for synthetic
        # id_bb[val['id']] = [val['bb']['x0'], val['bb']['y0'], val['bb']['x0']+val['bb']['width'], val['bb']['y0']+val['bb']['height']]

        if not real:
            # for onedrive
            id_bb[val['id']] = [val['bb']['x0'], val['bb']['y0']]
        else:
            # for real
            id_bb[val['id']] = [val['polygon']['x0'], val['polygon']['y0'],
                                val['polygon']['x1'], val['polygon']['y1'],
                                val['polygon']['x2'], val['polygon']['y2'],
                                val['polygon']['x3'], val['polygon']['y3']]

    in_obj = json.load(open(osp.join(read_dir, file)))

    try:
        bb = in_obj['task4']['output']['_plot_bb']
        p1 = (int(bb['x0']), int(bb['y0']))
        p2 = (int(bb['x0'] + bb['width']), int(bb['y0'] + bb['height']))
        cv2.rectangle(im, p1, p2, (0, 255, 0), thickness=2)
    except:
        pass

    # for axis, color in [('x-axis', (255, 0, 0)), ('y-axis', (255, 0, 255))]:
    for axis, color in [('x-axis', (255, 255, 0)), ('x-axis-2', (255, 255, 0)), ('y-axis', (255, 0, 255)),
                        ('y-axis-2', (255, 0, 255))]:
        for tick_obj in in_obj['task4']['output']['axes']['%s' % axis]:
            _id = tick_obj['id']
            pt = tick_obj['tick_pt']
            x = pt['x']
            y = pt['y']
            cv2.circle(im, (x, y), 4, color, thickness=-1)

            p = (int((id_bb[_id][0] + id_bb[_id][2] + id_bb[_id][4] + id_bb[_id][6]) / 4),
                 int((id_bb[_id][1] + id_bb[_id][3] + id_bb[_id][5] + id_bb[_id][7]) / 4))

            poly = np.array(id_bb[_id])
            cv2.polylines(im, [poly.reshape(-1, 2)], isClosed=True, color=(255, 0, 0), thickness=2)
            cv2.line(im, (x, y), p, color, thickness=2)

    if real:
        cv2.imwrite(osp.join(save_dir, file.replace("json", "jpg")), im)
    else:
        cv2.imwrite(osp.join(save_dir, file.replace("json", "png")), im)
