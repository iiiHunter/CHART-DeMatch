import random

import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon
from shapely import affinity


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, hue=None):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue, )

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class RandomRotate(object):
    def __init__(self, prob=0.5, max_theta=10):
        self.prob = prob
        self.max_theta = max_theta

    def __call__(self, image, target):
        if random.random() < self.prob:
            delta = random.uniform(-1 * self.max_theta, self.max_theta)
            width, height = image.size
            img_box = [[[0, 0], [width, 0], [width, height], [0, height]]]

            rotated_img_box = self._quad2minrect(self._rotate_polygons(img_box, delta, (width / 2, height / 2)))
            r_height = int(
                max(rotated_img_box[0][3], rotated_img_box[0][1]) - min(rotated_img_box[0][3], rotated_img_box[0][1]))
            r_width = int(
                max(rotated_img_box[0][2], rotated_img_box[0][0]) - min(rotated_img_box[0][2], rotated_img_box[0][0]))
            r_height = max(r_height, height + 1)
            r_width = max(r_width, width + 1)

            # padding im
            im_padding = np.zeros((r_height, r_width, 3))
            start_h, start_w = int((r_height - height) / 2.0), int((r_width - width) / 2.0)
            end_h, end_w = start_h + height, start_w + width
            im_padding[start_h:end_h, start_w:end_w, :] = image

            M = cv2.getRotationMatrix2D((r_width / 2, r_height / 2), delta, 1)
            im = cv2.warpAffine(im_padding, M, (r_width, r_height))
            im = Image.fromarray(im.astype(np.uint8))

            target['size'] = im.size
            target['point'] = self.rotate(target, -delta, (r_width / 2, r_height / 2), start_h, start_w)

        return im, target

    def _quad2minrect(self, boxes):
        return np.hstack((boxes[:, ::2].min(axis=1).reshape((-1, 1)), boxes[:, 1::2].min(axis=1).reshape((-1, 1)),
                          boxes[:, ::2].max(axis=1).reshape((-1, 1)), boxes[:, 1::2].max(axis=1).reshape((-1, 1))))

    def _rotate_polygons(self, polygons, angle, r_c):
        ## polygons: N*8
        ## r_x: rotate center x
        ## r_y: rotate center y
        ## angle: -15~15
        rotate_boxes_list = []
        for poly in polygons:
            box = Polygon(poly)
            rbox = affinity.rotate(box, angle, r_c)
            if len(list(rbox.exterior.coords)) < 5:
                print('img_box_ori:', poly)
                print('img_box_rotated:', rbox)
            # assert(len(list(rbox.exterior.coords))>=5)
            rotate_boxes_list.append(rbox.boundary.coords[:-1])
        res = self._boxlist2quads(rotate_boxes_list)
        return res

    def _boxlist2quads(self, boxlist):
        res = np.zeros((len(boxlist), 8))
        for i, box in enumerate(boxlist):
            # print(box)
            res[i] = np.array([box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1]])
        return res

    def rotate(self, target, angle, r_c, start_h, start_w):
        tr_pts = []
        pts = target['point']
        pts = np.array(pts)
        pts[:, 0] += start_w
        pts[:, 1] += start_h

        polys = Polygon(pts)
        r_polys = list(affinity.rotate(polys, angle, r_c).boundary.coords[:-1])
        tr_pts.extend(np.array(r_polys).reshape(-1, 2).astype(np.int32).tolist())
        return tr_pts


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            w, h = target['size']
            point = target['point']
            tr_point = []
            for val in point:
                tr_point.append([w - val[0], val[1]])
            target['point'] = tr_point
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.vflip(image)
            w, h = target['size']
            point = target['point']
            tr_point = []
            for val in point:
                tr_point.append([val[0], h - val[1]])
            target['point'] = tr_point
        return image, target


class FixSize(object):
    def __init__(self, size):
        h, w = size
        self.size = (h, w)

    def __call__(self, image, target):
        ori_size = image.size
        image = F.resize(image, self.size)
        if target is not None:
            tr_target = self.resize(target, image.size, ori_size)
            return image, tr_target
        return image, target

    def resize(self, target, tr_size, ori_size):
        target['size'] = tr_size
        pts = np.array(target['point'])
        if pts.any():
            pts[:, 0] = (pts[:, 0] / ori_size[0] * tr_size[0]).astype(np.int32)
            pts[:, 1] = (pts[:, 1] / ori_size[1] * tr_size[1]).astype(np.int32)
        target['point'] = pts.tolist()
        return target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target
