'''
Visualize circles
'''

import os, cv2
import os.path as osp


def vis_onedrive(test_txt):
    img_dir  = "/home/pci/disk1/maweihong/dataset/ICPR2020_CHART/2020_Add_Synthetic_OneDrive/OneDrive-2020/ICPR/Charts"
    pred_dir = "checkpoints/debug8_onedrive_v1/inference"
    save_dir = pred_dir.replace("inference", "vis")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    data = open(test_txt).read().splitlines()
    for line in data:
        filename = "/".join(line.split("/")[-2:])
        img  = cv2.imread(osp.join(img_dir, filename.replace("json", "png")))
        coors = open(osp.join(pred_dir, filename.split("/")[-1].replace("json", "txt"))).read().splitlines()
        for coor in coors:
            x, y = list(map(int, coor.split(",")))
            cv2.circle(img, (int(x), int(y)), 4, (255, 0, 0), thickness=-1)
        cv2.imwrite(osp.join(save_dir, filename.split("/")[-1].replace("json", "png")), img)
    return


def vis_UB_pmc():
    img_dir  = ""
    pred_dir = ""
    save_dir = ""
    if not osp.exists(save_dir):
        os.makedirs(save_dir)        
    return


if __name__=="__main__":
    # test_txt = "/home/pci/disk1/maweihong/dataset/ICPR2020_CHART/2020_Add_Synthetic_OneDrive/OneDrive-2020/Task4/test.txt"
    # vis_onedrive(test_txt)

    import shutil
    to_annotate = "/home/sol/data/Datasets/CHART/to_annotate/all"
    to_annotate2 = "/home/sol/data/Datasets/CHART/to_annotate/left"
    annotated = "/home/sol/data/Datasets/CHART/icpr_extra_data_v2/image"
    annotated_files = os.listdir(annotated)
    for item in os.listdir(to_annotate):
        if not item in annotated_files:
            shutil.copy(os.path.join(to_annotate, item), os.path.join(to_annotate2, item))
