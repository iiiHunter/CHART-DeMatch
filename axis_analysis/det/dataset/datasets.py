import glob
import os, json
import os.path as osp

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from . import transform as T
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from ..utils.utils import draw_umich_gaussian



class RealDataset(Dataset):
    def __init__(self, is_train):
        super(RealDataset, self).__init__()

        if is_train:
            img_dir = "/path/to/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.2.1/images"
            gt_dir = "/path/to/CHART/ICPR2020_CHARTINFO_UB_PMC_TRAIN_v1.2.1/annotations"
            files = os.listdir(gt_dir)
            self.label_files = [osp.join(gt_dir, file) for file in files]
            self.img_files = [osp.join(img_dir, file.replace(file.split(".")[-1], "jpg")) for file in files]
        else:
            icpr_test_json = "/path/to/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_3/annotations_JSON"
            icpr_test_images = "/path/to/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/splits_with_GT/split_3/images"
            icpr_test_anns = os.listdir(icpr_test_json)
            self.label_files = [osp.join(icpr_test_json, file) for file in icpr_test_anns]
            self.img_files = [osp.join(icpr_test_images, file.replace(file.split(".")[-1], "jpg")) for file in icpr_test_anns]

        if is_train:
            self.transforms = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                # T.RandomHorizontalFlip(prob=0.5),
                # T.RandomVerticalFlip(prob=0.5),
                # T.RandomRotate(prob=0.5),
                T.FixSize((480, 640)),  # height, width
                T.ToTensor(),
            ])
        else:
            self.transforms = T.Compose([
                T.FixSize((480, 640)),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        label_path = self.label_files[idx]
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert('RGB')

        key_pt = []
        data = json.load(open(label_path))
        x_axis_1 = data['task4']['output']['axes']['x-axis']
        x_axis_2 = data['task4']['output']['axes']['x-axis-2']
        y_axis_1 = data['task4']['output']['axes']['y-axis']
        y_axis_2 = data['task4']['output']['axes']['y-axis-2']
        vals = [x_axis_1, x_axis_2, y_axis_1, y_axis_2]
        for axis in vals:
            if not len(axis):
                continue
            for pt in axis:
                x = pt['tick_pt']['x'];
                y = pt['tick_pt']['y']
                key_pt.append([x, y])

        w, h = img.size
        target = {"size": (w, h), "point": key_pt}
        img, target = self.transforms(img, target)
        key_pt = target['point']

        # Visualize
        # vis = (img*255).numpy().transpose(1,2,0).copy()
        # pts = target['point']
        # for pt in pts:
        #     cv2.circle(vis, (pt[0], pt[1]), 4, color=(0,0,255), thickness=-1)
        # cv2.imwrite("1.jpg", vis)
        # import pdb; pdb.set_trace()

        heatmap = generate_target(img, key_pt, stride=1)
        img_info = {"img_path": img_path, 'size': (w, h)}
        return img, torch.tensor(heatmap).unsqueeze(0).float(), img_info

    def __len__(self):
        return len(self.label_files)


class SynDataset(Dataset):
    def __init__(self, is_train):
        super(SynDataset, self).__init__()

        if is_train:
            img_dir = ""
            gt_dir = ""
            self.label_files, self.img_files = [], []

            files = os.listdir(img_dir)
            for item in files:
                fn = item.replace(".png", "")
                im_path = os.path.join(img_dir, item)
                gt_path = os.path.join(gt_dir, "%s.json" % fn)
                with open(gt_path, "r") as f:
                    dd = json.loads(f.read())
                    if not dd["task4"]:
                        continue
                    if not dd["task4"]["output"]:
                        continue
                    self.img_files.append(im_path)
                    self.label_files.append(gt_path)

        else:
            img_dir = ""
            gt_dir = ""
            files = os.listdir(gt_dir)
            self.label_files = [osp.join(gt_dir, file) for file in files]
            self.img_files = [osp.join(img_dir, file.replace(".json", ".png")) for file in files]

        if is_train:
            self.transforms = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                T.RandomHorizontalFlip(prob=0.5),
                T.RandomVerticalFlip(prob=0.5),
                # T.RandomRotate(prob=0.5),
                T.FixSize((480, 640)),  # height, width
                T.ToTensor(),
            ])
        else:
            self.transforms = T.Compose([
                T.FixSize((480, 640)),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        label_path = self.label_files[idx]
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert('RGB')
        data = json.load(open(label_path))

        key_pt = []
        for k, v in data['task4']['output']["axes"].items():
            for pt in v:
                cur_x = pt['tick_pt']['x']
                cur_y = pt['tick_pt']['y']
                key_pt.append([cur_x, cur_y])

        w, h = img.size
        target = {"size": (w, h), "point": key_pt}
        img, target = self.transforms(img, target)
        key_pt = target['point']

        # Visualize
        # vis = (img*255).numpy().transpose(1,2,0).copy()
        # pts = target['point']
        # for pt in pts:
        #     cv2.circle(vis, (pt[0], pt[1]), 4, color=(0,0,255), thickness=-1)
        # cv2.imwrite("1.jpg", vis)
        # import pdb; pdb.set_trace()

        heatmap = generate_target(img, key_pt, stride=1)
        img_info = {"img_path": img_path, 'size': (w, h)}
        return img, torch.tensor(heatmap).unsqueeze(0).float(), img_info

    def __len__(self):
        return len(self.label_files)



class Adobe2020Dataset(Dataset):
    def __init__(self, is_train):
        super(Adobe2020Dataset, self).__init__()

        train_files = "/path/to/train.txt"
        test_files = "/path/to/test.txt"


        train_data = open(train_files).read().splitlines()
        test_data = open(test_files).read().splitlines()

        if is_train:
            # train_data.extend(test_data)
            files = train_data
        else:
            files = test_data

        self.label_files = files
        self.img_files = [file.replace("onedrive_new_jsons", "Charts").replace("json", "png") for file in files]

        if is_train:
            self.transforms = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                T.RandomHorizontalFlip(prob=0.5),
                T.RandomVerticalFlip(prob=0.5),
                # T.RandomRotate(prob=0.5),
                T.FixSize((480, 640)),  # height, width
                T.ToTensor(),
            ])
        else:
            self.transforms = T.Compose([
                T.FixSize((480, 640)),
                T.ToTensor(),
            ])

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        img = Image.open(img_path).convert('RGB')

        key_pt = []
        data = json.load(open(label_path))
        for k, v in data['task4']['output']["axes"].items():
            for pt in v:
                cur_x = pt['tick_pt']['x']
                cur_y = pt['tick_pt']['y']
                key_pt.append([cur_x, cur_y])

        w, h = img.size
        target = {"size": (w, h), "point": key_pt}
        img, target = self.transforms(img, target)
        key_pt = target['point']

        heatmap = generate_target(img, key_pt, stride=1)
        # heatmap = generate_target(img, key_pt, stride=4)
        img_info = {"img_path": img_path, 'size': (w, h)}
        return img, torch.tensor(heatmap).unsqueeze(0).float(), img_info

    def __len__(self):
        return len(self.label_files)


def generate_target(img, key_pt, stride):
    stride = 1
    heatmap = np.zeros((int(img.shape[1] / stride), int(img.shape[2] / stride)))
    for (x, y) in key_pt:
        draw_umich_gaussian(heatmap, (int(x / stride), int(y / stride)), 10)
        # draw_umich_gaussian(heatmap, (int(x/stride), int(y/stride)), 2)
    return heatmap
