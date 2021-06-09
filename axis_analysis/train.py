import os, cv2
import os.path as osp
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from yacs.config import CfgNode as CN

from det.dataset import RealDataset, SynDataset, Adobe2020Dataset, ConcatDataset, MixDataset, collect_fn
from det.model import Custom_model, get_pose_resnet, get_pose_hrnet
from det.model.hehe_seg import SegModel
from det.model import cfg

from det.utils.utils import Visualizer, compute_loss
from det.utils.common import AverageMeter, ProgressMeter, LogSummary
from det.utils.coord import transfer_target
from post_proc.prepare_output_syn import write as write_onedrive
from post_proc.prepare_output_pmc import write as write_pmc
from evaluation.metric4_synthetic import eval_task4 as eval_task4_synth
from evaluation.metric4_pmc import eval_task4 as eval_task4_pmc


def train(model, epoch, data_loader, optimizer, criterion, device, project_dir):
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(len(data_loader), [losses], prefix="Epoch: [{}]".format(epoch))

    model.train()

    for it, (img, target, _) in enumerate(data_loader):
        batches_done = len(data_loader) * epoch + it

        img = img.to(device)
        target = target.to(device)
        prediction = model(img)

        loss = criterion(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img.size(0))

        if it % 10 == 0:
            progress.display(it)
            logger.write_scalars({
                'loss': losses.avg,
            }, tag='train', n_iter=batches_done)

        torch.save(model.state_dict(), project_dir + "/model_%d.pth" % epoch)


def inference(model, data_loader, save_dir, stride, input_dir, img_dir, result_dir="", bb_dir="", coco_json_path=""):
    '''

    :param model:
    :param data_loader:
    :param save_dir: save txt result of tick points
    :param stride:
    :param input_dir: path of input json
    :param img_dir: path of image path
    :param result_dir: path of json format result
    :param bb_dir: main_plot_region bounding box prediction .pth file (default from detectron2)
    :param coco_json_path: coco json format file that helps to determine the original filename
    :return:
    '''
    # model.eval()
    with torch.no_grad():
        for it, (imgs, targets, img_infos) in enumerate(data_loader):
            imgs = imgs.to(device)
            predictions = model(imgs)

            # Extract coord
            tr_h = imgs.size(-2)
            tr_w = imgs.size(-1)
            coords = transfer_target(predictions, thresh=0.1, n_points=None)
            output_res = []

            for idx, (img_info, coord) in enumerate(zip(img_infos, coords)):
                txt_path = img_info['img_path'].split("/")[-1].replace(".jpg", ".txt").replace(".png", ".txt")
                ori_w, ori_h = img_info['size']
                if len(coord):
                    cur_res = np.zeros_like(np.array(coord))
                    cur_res[:, 0] = np.array(coord)[:, 0] / tr_w * ori_w * stride
                    cur_res[:, 1] = np.array(coord)[:, 1] / tr_h * ori_h * stride
                    cur_res = cur_res.astype(np.int32)
                else:
                    cur_res = []
                f = open(osp.join(save_dir, txt_path), "w")
                for val in cur_res:
                    f.write(str(int(val[0])) + "," + str(int(val[1])) + "\n")
                f.close()
    write_pmc(bb_dir, save_dir, input_dir, result_dir, coco_json_path, task7_test=False)
    total_recall, total_precision, f_measure = eval_task4_pmc(input_dir, result_dir, img_dir)
    return total_recall, total_precision, f_measure


if __name__ == "__main__":
    base_dir = "work_dirs"
    log_dir = "logs"
    exp_name = "base"
    # resume_path = osp.join("checkpoints", exp_name, "model_48.pth")  # inference mode
    resume_path = None

    project_dir = osp.join(base_dir, exp_name)
    log_path = osp.join(log_dir, exp_name)

    if not osp.exists(log_path):
        os.makedirs(log_path)

    if not osp.exists(project_dir):
        os.makedirs(project_dir)

    logger = LogSummary(log_path)

    epochs = 100
    last_epoch = 0
    bs = 8  # 8
    num_workers = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_layers = 18
    model_config = {'DECONV_WITH_BIAS': False,
                    'NUM_JOINTS': 1,
                    'FINAL_CONV_KERNEL': 1,
                    'NUM_DECONV_LAYERS': 5,
                    'NUM_DECONV_FILTERS': [256, 128, 128, 64, 32],
                    'NUM_DECONV_KERNELS': [4, 4, 4, 4, 4]}
    cfg = {'backbone': 'resnet18', 'out_nx_infos': {'1x': 32}, 'pretrain': True}
    model = SegModel(cfg).cuda()
    stride = 1

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [5, 10], 0.1, last_epoch - 1)

    # compute_loss
    criterion = torch.nn.MSELoss()

    # train_set = SynDataset(is_train=True)
    train_set = RealDataset(is_train=True)
    # train_set = Adobe2020Dataset(is_train=True)
    # train_set = MixDataset([Adobe2020Dataset(is_train=True), RealDataset(is_train=True)], ratios=[0.5, 0.5])

    train_loader = DataLoader(train_set,
                              shuffle=True,
                              batch_size=bs,
                              num_workers=num_workers,
                              collate_fn=collect_fn)

    # test_set = SynDataset(is_train=False)
    test_set = RealDataset(is_train=False)
    # test_set = Adobe2020Dataset(is_train=False)
    test_loader = DataLoader(test_set,
                             shuffle=False,
                             batch_size=4,
                             num_workers=num_workers,
                             collate_fn=collect_fn)

    img_dir = "/path/to/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/" \
               "splits_with_GT/split_3/images"
    input_dir = "/path/to/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/" \
                "splits_with_GT/split_3/annotations_JSON"
    bb_dir = "/path/to/detectron2/projects/pmc/inference/instances_predictions.pth"
    coco_json_path = "dataset/annotations/pmc_plot_bb_test2020_icpr.json"
    result_dir = osp.join(project_dir, "prediction")
    if resume_path:
        model.load_state_dict(torch.load(resume_path))
        save_dir = osp.join(project_dir, "inference")
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        inference(model, test_loader, save_dir, stride, input_dir, img_dir, result_dir, bb_dir, coco_json_path)
    else:
        # resume_path = osp.join("checkpoints", exp_name, "model_35.pth")
        # model.load_state_dict(torch.load(resume_path))
        save_dir = osp.join(project_dir, "inference")
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        for ep in range(epochs):
            print("Training epoch: ", ep)
            total_recall, total_precision, f_measure = inference(model, test_loader, save_dir, stride, input_dir,
                                                                 img_dir, result_dir, bb_dir, coco_json_path)
            logger.write_scalars({
                'recall': total_recall,
            }, tag='test', n_iter=ep)
            logger.write_scalars({
                'precision': total_precision,
            }, tag='test', n_iter=ep)
            logger.write_scalars({
                'f_measure': f_measure,
            }, tag='test', n_iter=ep)

            train(model, ep, train_loader, optimizer, criterion, device, project_dir)
            lr_scheduler.step()
