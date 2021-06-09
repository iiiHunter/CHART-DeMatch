import torch

def collect_fn(batch):
    imgs, labels, img_infos = zip(*batch)
    return torch.stack(imgs, 0), torch.stack(labels, 0), img_infos