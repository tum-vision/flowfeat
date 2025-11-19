"""
Copyright (c) 2024 TU Munich
Author: Nikita Araslanov <nikita.araslanov@tum.de>
License: Apache License 2.0
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import numpy as np

import random
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.distributed as dist
import torch.multiprocessing as mp

import torchvision.transforms as tf

from tqdm.auto import tqdm

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from davis_utils.davis2017 import evaluate_semi

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

######################
#      Utils         #
######################
def pca_feats(ff):

    N, C, H, W = ff.shape

    pca = PCA(
        n_components=3,
        svd_solver='auto',
        whiten=True
    )

    ff = ff.movedim(1, -1)
    ff = ff.reshape(-1, C).detach().cpu().numpy()

    x = torch.Tensor(pca.fit_transform(ff))
    x = x.view(N, H, W, 3)
    x = x.movedim(-1, 1)

    # normalising
    x = (x - x.min()) / (x.max() - x.min())

    return x

######################
#      Dataset       #
######################
from dataloader import DataVideo

######################
#       Model        #
######################
from model import FlowFeatTrain

######################
#      Training      #
######################

def train_epoch(cfg, net, net_simple, optim, loader, epoch, vis=False):

    if is_main():
        loader = tqdm(loader)

    all_losses = {}
    for n, batch in enumerate(loader):
        net.zero_grad()

        frames, frameref, tf1, tf2 = [x.cuda(non_blocking=True) for x in batch]

        losses, outs = net(frames, frameref, tf1, tf2, epoch=epoch)

        losses["total"].backward()
        optim.step()

        net_simple.update_ema()

        if is_main():
            discr = ""
            discr += "Total = {:4.3f}".format(losses["total"])
            discr += " | FlowRes = {:4.3f}".format(losses["flowres"])
            discr += " | FlowBdr = {:4.3f}".format(losses["flowbdr"])
            loader.set_description(discr, refresh=True)

        for loss_key, loss_val in losses.items():
            if not loss_key in all_losses:
                all_losses[loss_key] = []
            all_losses[loss_key] += [loss_val.item()]

    if is_main():
        for loss_key, loss_val in all_losses.items():
            all_losses[loss_key] = np.mean(all_losses[loss_key])

        wandb.log(all_losses)

        if vis:
            visualise_train(frames, outs["crop1"], outs["crop2"], outs)

@torch.no_grad()
def val_epoch(cfg, net, loader, epoch, vis=False):

    all_losses = {}
    for n, batch in enumerate(loader):

        frames, frameref, tf1, tf2 = [x.cuda() for x in batch]
        losses, outs = net(frames, frameref, tf1, tf2, epoch=epoch)

        for loss_key, loss_val in losses.items():
            if not loss_key in all_losses:
                all_losses[f"val_{loss_key}"] = []
            all_losses[f"val_{loss_key}"] += [loss_val.cpu()]

    if is_main():
        for loss_key, loss_val in all_losses.items():
            all_losses[loss_key] = np.mean(all_losses[loss_key])

        wandb.log(all_losses)
        if vis:
            visualise_train(frames, outs["crop1"], outs["crop2"], outs, val=True)


######################
#   Visualisation    #
######################

from util.flow_vis import flow_to_image
from torchvision.utils import make_grid

def visualise_train(frames, crop1, crop2, outs, val=False):

    frames_input, frames_next = frames[:, 0], frames[:, 1]

    denorm = tf.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                           std=[1/0.229, 1/0.224, 1/0.225])

    resize = lambda x: F.interpolate(x, frames_input.shape[-2:], mode="bilinear", align_corners=False)

    def prep_rgb(x):
        return 255. * denorm(x.detach().cpu()).clamp(0, 1)

    def prep_rgb_pred(x):
        return 255. * x.detach().cpu().clamp(0, 1)

    def prep_depth(x):
        x = -x.detach().cpu().numpy()
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = np.array([plt.cm.plasma(image) for image in x])
        x = torch.from_numpy(x)[:, 0].movedim(-1, 1)
        return 255. * x[:, :3]

    frame_input = prep_rgb(frames_input)
    frame_next = prep_rgb(frames_next)
    crop1 = prep_rgb(crop1)
    crop2 = prep_rgb(crop2)

    flow_rgb = torch.stack([torch.from_numpy(flow_to_image(x.numpy())) \
                                for x in outs["flow_0"].detach().cpu()], 0)
    flow_rgb = F.interpolate(flow_rgb.movedim(-1, 1), frame_input.shape[-2:], mode="nearest")

    tflow_rgb = torch.stack([torch.from_numpy(flow_to_image(x.numpy())) \
                                for x in outs["t_flow_0"].detach().cpu()], 0)
    tflow_rgb = F.interpolate(tflow_rgb.movedim(-1, 1), frame_input.shape[-2:], mode="nearest")

    feats0_rgb = torch.cat([pca_feats(ff) for ff in outs["features"].split(1, 0)], 0)
    feats0_rgb = prep_rgb_pred(resize(feats0_rgb))

    image_big = torch.cat([frame_input, frame_next, crop1, crop2, \
                           feats0_rgb, tflow_rgb, flow_rgb], -1)

    image_grid = make_grid(image_big[::10], nrow=1)
    tag = "val" if val else "train"

    wandb.log({f"{tag}-reconstruction-flow": wandb.Image(image_grid, caption="reconstruction with flow")})

def visualise_test(tag, outs):

    denorm = tf.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                           std=[1/0.229, 1/0.224, 1/0.225])

    def prep_rgb(x):
        return 255. * denorm(x.detach().cpu()).clamp(0, 1)

    def prep_mask(m):
        m = m.movedim(1, 2).flatten(2, 3)
        return 255. * m[:, None].expand(-1, 3, -1, -1)
    
    def prep_feats(features):
        return 255. * torch.cat([pca_feats(ff) \
                    for ff in outs["features"].split(1, 0)], 0)

    masks_pred = torch.cat([prep_rgb(outs["input"]),
                            prep_feats(outs["features"]),
                            prep_mask(outs["mask"])], -1)

    image_grid = make_grid(masks_pred[::5], nrow=1)

    wandb.log({f"{tag}-mask-propagation": wandb.Image(image_grid, caption="mask propagation")})


######################
#     Validation     #
######################
def run_validation(cfg, net, dataset, vis=False):

    all_metrics = {"Jaccard": [], "F-score": []}

    for tag, video in enumerate(dataset.videos):
        metrics = validate_one(f"{tag:02d}", cfg, net, video, vis)

        # accumulating
        for key, vals in metrics.items():
            all_metrics[key] += vals

    # summary -- computing the mean over metrics
    for key, vals in all_metrics.items():
        all_metrics[key] = np.mean(vals)

    print("*mean* Jaccard: ", all_metrics["Jaccard"])
    print("*mean* F-score: ", all_metrics["F-score"])
    return all_metrics

def validate_one(tag, cfg, net, video, vis):

    min_size = min(cfg.train.input_size)
    tf_val = tf.Compose([tf.Resize(min_size), tf.CenterCrop(cfg.train.input_size)])
    tf_val_mask = tf.Compose([tf.Resize(min_size), tf.CenterCrop(cfg.train.input_size)])
    tf_val_net =  tf.Compose([tf.ToTensor(),
                              tf.Normalize(mean=[0.485, 0.456, 0.406], \
                                           std =[0.229, 0.224, 0.225])])

    def mask2tensor(mask, num_cls):
        h,w = mask.shape
        ones = torch.ones(1, mask.shape[0], mask.shape[1])
        zeros = torch.zeros(num_cls, h, w)
        return zeros.scatter(0, mask[None, ...], ones)

    images = []
    masks_gt = []
    num_cls = 1
    for n in range(video["len"]):
        image_pil = Image.open(video["images"][n]).convert('RGB')
        image_pil = tf_val(image_pil)

        mask = tf_val_mask(Image.open(video["masks"][n]))
        mask = torch.from_numpy(np.array(mask, np.int64, copy=False))
        masks_gt.append(mask)
        num_cls = max(num_cls, mask.max().item() + 1)

        im = tf_val_net(image_pil)
        images.append(im)


    # initialising the mask
    mask_init = masks_gt[0][None, ...].cuda()

    outs = {"input": [], "mask": [], "features": []}

    # binary classifier
    mlp_cls = nn.Sequential(nn.Conv2d(net.fdim, net.fdim, 1),
                            nn.ReLU(True),
                            nn.Conv2d(net.fdim, num_cls, 1)).cuda()
    cls_optim = torch.optim.Adam(mlp_cls.parameters(), weight_decay=0.0005, lr=0.005)

    frames = torch.stack(images[:cfg.train.temp_win], 0)
    # fitting a classifier to the first image
    with torch.no_grad():
        frames = frames[None, ...].cuda()
        _, features = net.forward_up(frames[:, 0])

    # we will overwrite this later
    pbar = tqdm(range(300))
    for itr in pbar:
        cls_optim.zero_grad()

        logits = mlp_cls(features)
        loss = F.cross_entropy(logits, mask_init)

        loss.backward()
        cls_optim.step()

        pbar.set_description("Loss = {:4.3f}".format(loss.item()), refresh=True)

    images = images + images[-cfg.train.temp_win:]
    for n in range(video["len"]):

        frames = images[n:n+cfg.train.temp_win]

        with torch.no_grad():
            frames = torch.stack(frames, 0)[None, ...].cuda()
            _, features = net.forward_up(frames[:, 0])
            outs["input"].append(frames[:, 0].cpu())

            logits = mlp_cls(features)
            outs["mask"].append(mask2tensor(logits.argmax(1)[0].cpu(), num_cls).cpu())
            outs["features"].append(features.cpu())

    masks = torch.stack(outs["mask"], 1).numpy()
    masks_gt = torch.stack([mask2tensor(mask, num_cls) for mask in masks_gt], 1).numpy()
    metrics = evaluate_semi((masks_gt, ), (masks, ))

    # visualisation
    outs["input"] = torch.cat(outs["input"], 0)
    outs["mask"] = torch.stack(outs["mask"], 0)
    outs["features"] = torch.cat(outs["features"], 0)

    if vis and is_main():
        visualise_test(tag, outs)

    print("Jaccard: ", metrics["J"]["M"][1:])
    print("F-score: ", metrics["F"]["M"][1:])

    return {"Jaccard": metrics["J"]["M"][1:], 
            "F-score": metrics["F"]["M"][1:]}

def setup_loader(cfg, split_file, world_size, rank, **kwargs):

    dataset = DataVideo(cfg, split_file)

    if world_size > 1:

        if "shuffle" in kwargs:
            del kwargs["shuffle"]

        sampler = data.DistributedSampler(dataset, \
                                          num_replicas=world_size, \
                                          rank=rank, shuffle=True)
    else:
        sampler = None

    loader = data.DataLoader(dataset, sampler=sampler, **kwargs)
    return loader, sampler


### Main ###
def main(rank, cfg: DictConfig, world_size, port):

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['LOCAL_RANK'] = str(rank)

    print(">>> Local rank ", rank)
    #torch.set_default_device(f'cuda:{rank}')

    if world_size > 1:
        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        
        # fixing the seed
        rank = dist.get_rank()

    global is_main
    is_main = lambda: rank == 0

    seed = cfg.run.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    # num of gpus
    gpu_here = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(gpu_here)

    print(cfg)
    if is_main():
        wandb.login()
        wandb.init(entity=cfg.run.wandb_entity,
                   project=cfg.run.wandb_project,
                   group=cfg.run.group,
                   name=cfg.run.tag, 
                   notes="", 
                   dir=cfg.run.wandb_dir)

    assert cfg.train.batch_size % world_size == 0, \
            "Batch size not divisible by # of GPUs"

    batch_per_gpu = cfg.train.batch_size // world_size

    loader, sampler_train = setup_loader(cfg, cfg.data.train_split, world_size, rank,
                            batch_size=batch_per_gpu,
                            num_workers=cfg.run.num_workers,
                            pin_memory=True,
                            drop_last=True,
                            shuffle=True)

    valloader, sampler_val = setup_loader(cfg, cfg.data.val_split, world_size, rank,
                             batch_size=batch_per_gpu,
                             num_workers=cfg.run.num_workers,
                             pin_memory=True,
                             drop_last=True)

    # cfg.model.patch_size should match the patch size of the encoder
    net = FlowFeatTrain(cfg.model)
    net.cuda()
    net_simple = net

    if world_size > 1:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu_here], find_unused_parameters=True)
        net_simple = net.module

    optim = torch.optim.AdamW(net_simple.parameter_groups(), \
                              betas=cfg.train.opt.betas, \
                              weight_decay=cfg.train.opt.weight_decay, \
                              lr=cfg.train.opt.lr)

    if os.path.isfile(cfg.model.dec_snapshot):
        params = torch.load(cfg.model.dec_snapshot)
        missing, unexpected = net_simple.load_state_dict(params["model"], strict=False)
        print("Loading decoder parameters: ", cfg.model.dec_snapshot)
        print(missing, unexpected)

    best_val = 0.
    snapshots = []
    num_save_best = 5
    for epoch in range(cfg.train.epochs):

        vis_train = epoch % cfg.run.vis_train_every == 0
        vis_val = epoch % cfg.run.vis_val_every == 0

        if world_size > 1:
            sampler_train.set_epoch(epoch)
            sampler_val.set_epoch(epoch)

        net.train()
        train_epoch(cfg, net, net_simple, optim, loader, epoch, vis_train)
        val_epoch(cfg, net, valloader, epoch, vis_train)

        if is_main() and epoch % cfg.run.val_every == 0:
            net.eval()
            val_metrics = run_validation(cfg, net_simple, valloader.dataset, vis_val)

            mean_metric = 0.5 * (val_metrics["F-score"] + val_metrics["Jaccard"])
            if mean_metric > best_val:
                print(f"Best J&F: {mean_metric}")
                best_val = mean_metric
                # saving the snapshot
                save_dir = f"runs/{cfg.run.series}/{cfg.run.tag}"
                os.makedirs(save_dir, exist_ok=True)
                filename = os.path.join(save_dir, f"snapshot_{epoch:03d}_{best_val:4.3f}.pth")
                torch.save({"model": net_simple.state_dict(), "optim": optim.state_dict()}, filename)
                if len(snapshots) == num_save_best: # reached the maximum
                    if os.path.isfile(snapshots[0]):
                        os.remove(snapshots[0])
                    del snapshots[0]
                snapshots.append(filename)

            wandb.log(val_metrics)
        
    if world_size > 1:
        dist.destroy_process_group()

@hydra.main(version_base=None, config_path="config")
def main_cfg(cfg: DictConfig):
    port = random.randint(12900, 12999)
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(main,
                 args=(cfg, world_size, port),
                 nprocs=world_size,
                 join=True)
    else:
        main(0, cfg, world_size, port)

if __name__ == "__main__":
    main_cfg()
