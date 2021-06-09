import os, cv2, torch
import numpy as np
import os.path as osp
from PIL import Image

from det.model import Custom_model, get_pose_resnet, get_pose_hrnet, cfg
from det.model.hehe_seg import SegModel
from det.utils.coord import transfer_target
import det.dataset.transform as T


def vis_heatmap(img, heatmap, scale=False):
    '''
    img:     numpy array (h,w,3)
    heatmap: numpy array (h,w)
    '''
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap)
    max_val = np.max(heatmap);
    min_val = np.min(heatmap)
    nor_heat = heatmap - min_val / (max_val - min_val)

    hm = np.uint8(255 * nor_heat)
    hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    if scale:
        hm = cv2.resize(hm, (0, 0), fx=img.shape[1] / hm.shape[1], fy=img.shape[0] / hm.shape[0],
                        interpolation=cv2.INTER_NEAREST)
    composed = hm * 0.5 + img * 0.5
    return composed


if __name__ == '__main__':
    img_dir = "/path/to/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/" \
              "splits_with_GT/split_3/images"
    save_dir = "/path/to/release_ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/ICPR2020_CHARTINFO_UB_PMC_TEST_v1.0/" \
               "splits_with_GT/split_3/"
    real = True

    if not osp.exists(osp.join(save_dir, 'pred')):
        os.makedirs(osp.join(save_dir, 'pred'))
    if not osp.exists(osp.join(save_dir, 'vis')):
        os.makedirs(osp.join(save_dir, 'vis'))

    model_path = "work_dirs/base/model_22.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    # v2 version
    stride = 1
    num_layers = 18
    model_config = {'DECONV_WITH_BIAS': False,
                    'NUM_JOINTS': 1,
                    'FINAL_CONV_KERNEL': 1,
                    'NUM_DECONV_LAYERS': 5,
                    'NUM_DECONV_FILTERS': [256, 128, 128, 64, 32],
                    'NUM_DECONV_KERNELS': [4, 4, 4, 4, 4]}

    # model = get_pose_resnet(num_layers, model_config, is_train=False).to(device)
    cfg = {'backbone': 'resnet18', 'out_nx_infos': {'1x': 32}, 'pretrain': True}
    model = SegModel(cfg).cuda()
    model.eval()
    model.load_state_dict(torch.load(model_path))

    transform = T.Compose([
        T.FixSize((480, 640)),
        T.ToTensor(),
    ])
    for img_path in os.listdir(img_dir):
        print(img_path)
        img = Image.open(osp.join(img_dir, img_path)).convert('RGB')
        tr_img, _ = transform(img, target=None)
        input_ = tr_img.to(device)

        ori_w, ori_h = img.size
        tr_h, tr_w = tr_img.shape[1:]

        pred = model(input_.unsqueeze(0))
        coord = transfer_target(pred, thresh=0.2, n_points=None)[0]

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

        # Visualize
        vis = input_.cpu() * torch.tensor([0.2479, 0.2475, 0.2485])[None, :, None, None] + torch.tensor(
            [0.4372, 0.4372, 0.4373])[None, :, None, None]
        vis = (vis * 255).numpy()
        for idx2, (val1, val2, val3) in enumerate(zip(vis, pred, coord)):
            val1 = val1.transpose(1, 2, 0).copy()
            val2 = val2[0].cpu().detach().numpy()
            composed = vis_heatmap(val1, val2)
            cv2.imwrite(osp.join(save_dir, 'vis', img_path.split('/')[-1]), composed)
