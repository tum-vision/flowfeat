"""
Copyright (c) 2024 TU Munich
Author: Nikita Araslanov <nikita.araslanov@tum.de>
License: Apache License 2.0
"""

import os
import sys
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as tf

from hydra.utils import instantiate
from util.ema_pytorch import EMA

#######################################
#       RAFT  & SEARAFT & SMURF       #
#######################################
import importlib

def try_import(module_name, class_path=None, alias=None):
    """
    Try to import a module or class dynamically if its directory exists.
    Returns the imported module/class, or None if not available.
    
    Args:
        module_name (str): Folder or top-level package name, e.g. 'SEA-RAFT'
        class_path (str): Optional full import path, e.g. 'SEA-RAFT.core.raft.RAFT'
        alias (str): Optional short alias for printing
    
    Example:
        SEARAFT = try_import("SEA-RAFT", "SEA-RAFT.core.raft.RAFT", alias="SEA-RAFT")
    """
    module_dir = os.path.join(os.path.dirname(__file__), module_name)
    display_name = alias or module_name

    if not os.path.isdir(module_dir):
        print(f"[Warning] {display_name} directory not found. Skipping import.")
        return None

    try:
        if class_path:
            # Replace slashes/hyphens for valid Python import
            class_path = class_path.replace("/", ".").replace("-", "_")
            components = class_path.split(".")
            mod = importlib.import_module(".".join(components[:-1]))
            return getattr(mod, components[-1])
        else:
            mod_name = module_name.replace("-", "_")
            return importlib.import_module(mod_name)
    except Exception as e:
        print(f"[Warning] {display_name} import failed: {e}")
        return None

# --------------------------------------------------------------------
# Trying to import
# --------------------------------------------------------------------
# Path to submodule root (e.g., SEARAFT)

# SUBMODULES
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SEARAFT/core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RAFT/core")) # either RAFT or SEARAFT
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "SMURF"))

# --- RAFT ---
RAFT = try_import("RAFT", "RAFT.core.raft.RAFT", alias="RAFT")
InputPadder = try_import("RAFT", "RAFT.core.utils.utils.InputPadder", alias="RAFT InputPadder")

# --- SEA-RAFT ---
#SEARAFT = try_import("SEARAFT", "SEARAFT.core.raft.RAFT", alias="SEARAFT")

# --- SMURF ---
SMURF_RAFT = try_import("SMURF", "SMURF.smurf.raft_smurf", alias="SMURF")

class RAFT_Args:
    
    model = "models/raft-sintel.pth"
    step = 1
    small = False
    mixed_precision = False
    alternate_corr = False
    
    def __contains__(self, value):
        return hasattr(self, value)

class RaftFlow:

    def __init__(self, denorm_func):
        if RAFT is None:
            raise ImportError(
                "RAFT not available. Ensure `RAFT/` exists and is importable."
            )

        args = RAFT_Args()
        model = nn.DataParallel(RAFT(args))
        model.load_state_dict(torch.load(args.model))
        self.model = model.module.eval()
        self.denorm = denorm_func

    @torch.no_grad()
    def __call__(self, image1, image2):
        image1_255 = self.denorm(image1) * 255.
        image2_255 = self.denorm(image2) * 255.

        padder = InputPadder(image1_255.shape)
        image1_pad, image2_pad = padder.pad(image1_255, image2_255)
        
        _, flow = self.model(image1_pad, image2_pad, iters=20, test_mode=True)

        flow = padder.unpad(flow)
        flow[:, 0, :, :] *= 2 / flow.shape[3]
        flow[:, 1, :, :] *= 2 / flow.shape[2]
        
        return flow


class SeaRAFTArgs:
    model_url = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
    name =  "spring-M"
    dataset= "spring"
    gpus = [0, 1, 2, 3, 4, 5, 6, 7]

    use_var= True
    var_min= 0
    var_max= 10
    pretrain= "resnet34"
    initial_dim= 64
    block_dims =[64, 128, 256]
    radius= 4
    dim= 128
    num_blocks= 2
    iters= 4

    def __contains__(self, value):
        return hasattr(self, value)

class SeaFlow:

    def __init__(self, denorm_func):
        if SEARAFT is None:
            raise ImportError(
                "SEARAFT not available. Ensure `SEARAFT/` exists and is importable."
            )

        print('Initializing Sea-RAFT')
        args = SeaRAFTArgs()
        model = SEARAFT.from_pretrained(args.model_url, args=args, force_download=True)  # Use Hugging Face model repository
        self.model = model.eval()
        self.denorm = denorm_func

    @torch.no_grad()
    def __call__(self, image1, image2):
        image1_255 = self.denorm(image1) * 255.
        image2_255 = self.denorm(image2) * 255.

        padder = InputPadder(image1_255.shape)
        image1_pad, image2_pad = padder.pad(image1_255, image2_255)

        output = self.model(image1_pad, image2_pad, iters=SeaRAFTArgs.iters, test_mode=True)
        flow = output['flow'][-1]

        flow = padder.unpad(flow)

        flow[:, 0, :, :] *= 2 / flow.shape[3]
        flow[:, 1, :, :] *= 2 / flow.shape[2]

        return flow

class SMURFArgs:

    #checkpoint = "SMURF/models/smurf-sintel/smurf-sintel.pt"
    #checkpoint = "SMURF/models/smurf-chairs/smurf-chairs.pt"
    checkpoint = "SMURF/models/smurf-kitti/smurf-kitti.pt"

    def __contains__(self, value):
        return hasattr(self, value)

class SMURF:

    def __init__(self, denorm_func):

        if SMURF_RAFT is None:
            raise ImportError(
                "SMURF not available. Ensure `SMURF/` exists and is importable."
            )

        args = SMURFArgs()
        print(f'Initialize SMURF / {args.checkpoint}')
        model = SMURF_RAFT(checkpoint=args.checkpoint)

        self.model = model.eval()
        self.denorm = denorm_func

    @torch.no_grad()
    def __call__(self, image1, image2):
        image1_norm = 2.0 * self.denorm(image1) - 1.0
        image2_norm = 2.0 * self.denorm(image2) - 1.0

        #padder = InputPadder(image1_255.shape)
        #image1_pad, image2_pad = padder.pad(image1_255, image2_255)

        output = self.model(image1_norm, image2_norm)

        # last iteration
        flow = output[-1]

        flow[:, 0, :, :] *= 2 / flow.shape[3]
        flow[:, 1, :, :] *= 2 / flow.shape[2]

        return flow

######################
#      Models        #
######################

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed

class MAEEncoder(nn.Module):
    """ Masked Autoencoder (encoder only) with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, imgs): 
        B, nc, w, h = imgs.shape

        # embed patches
        x = self.patch_embed(imgs)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :] #self.interpolate_pos_encoding(imgs, w, h)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

def mae_vit_base_encoder(cfg):
    from timm.layers import resample_abs_pos_embed

    # by default
    model = MAEEncoder(img_size=cfg.input_size, patch_size=cfg.patch_size,
                       embed_dim=768, depth=12, num_heads=12,
                       mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))


    print("MAE: output token grid size", model.patch_embed.grid_size)
    model_weights = torch.load(cfg.enc_snapshot)["model"]
    model_weights["pos_embed"] = resample_abs_pos_embed(model_weights["pos_embed"],
                                                        new_size=model.patch_embed.grid_size,
                                                        num_prefix_tokens=1,
                                                        interpolation='bicubic',
                                                        antialias=True,
                                                        verbose=True)

    model.load_state_dict(model_weights, strict=True)
    return model

def mae_vit_large_encoder(cfg):
    from timm.layers import resample_abs_pos_embed

    model = MAEEncoder(img_size=cfg.input_size, patch_size=cfg.patch_size,
                       embed_dim=1024, depth=24, num_heads=16,
                       mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6))

    print("MAE: output token grid size", model.patch_embed.grid_size)
    model_weights = torch.load(cfg.enc_snapshot)["model"]
    model_weights["pos_embed"] = resample_abs_pos_embed(model_weights["pos_embed"],
                                                        new_size=model.patch_embed.grid_size,
                                                        num_prefix_tokens=1,
                                                        interpolation='bicubic',
                                                        antialias=True,
                                                        verbose=True)

    model.load_state_dict(model_weights, strict=True)
    return model

class CfgDPT():

    def __init__(self, patch_size, features, fdim, hooks=[2, 5, 8, 11]):
        self.patch_size = patch_size
        self.features = features
        self.vit_features = fdim
        self.hooks = hooks

def load_encoder(cfg):

    if cfg.enc_snapshot.startswith("dino_"):
        enc = torch.hub.load('facebookresearch/dino:main', cfg.enc_snapshot)

        if cfg.enc_snapshot.endswith("vits16"):
            return enc, CfgDPT(16, 4*[384], 384)
        elif cfg.enc_snapshot.endswith("vitb16"):
            return enc, CfgDPT(16, 4*[768], 768)
        elif cfg.enc_snapshot.endswith("vitl16"):
            return enc, CfgDPT(16, 4*[1024], 1024)
        else:
            return enc, None

        print("Did not find DINOv1 ", cfg.enc_snapshot)

    elif cfg.enc_snapshot.startswith("dinov2_"):
        enc = torch.hub.load('facebookresearch/dinov2', cfg.enc_snapshot)

        if cfg.enc_snapshot.endswith("vits14"):
            return enc, CfgDPT(14, 4*[384], 384)
        elif cfg.enc_snapshot.endswith("vitb14"):
            return enc, CfgDPT(14, 4*[768], 768)
        elif cfg.enc_snapshot.endswith("vitl14"):
            return enc, CfgDPT(14, 4*[1024], 1024, [5, 11, 17, 23])
        
        print("Did not find DINOv2 ", cfg.enc_snapshot)

    elif cfg.enc_snapshot.endswith("mae_pretrain_vit_base.pth"):

        print("Using MAE ViT-B")
        return mae_vit_base_encoder(cfg), \
                CfgDPT(16, [96, 192, 384, 768], 768)

    elif cfg.enc_snapshot.endswith("mae_pretrain_vit_large.pth"):

        print("Using MAE ViT-L")
        return mae_vit_large_encoder(cfg), \
                CfgDPT(16, [256, 512, 1024, 1024], 1024, [5, 11, 17, 23])

    raise NotImplemented


######################
#      DPT           #
######################

class SkipCLS(nn.Module):

    def __init__(self, start_index=1):
        super(SkipCLS, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x

class Postprocess(nn.Module):

    def __init__(self, patch_size, *conv_layers):
        super().__init__()
        self.patch_size = patch_size
        self.pre = nn.Sequential(SkipCLS(), Transpose(1, 2))
        self.post = nn.Sequential(*conv_layers)

    def forward(self, x, hw):
        x = self.pre(x)
        x = x.unflatten(2, (hw[0] // self.patch_size, \
                            hw[1] // self.patch_size))
        return self.post(x)

def save_activation(model, name):

    assert not hasattr(model, name), f"Model already has attribute {name}"

    def hook(module, input, output):
        setattr(model, name, output)

    return hook

def dpt_wrapper(model, hooks = [2, 5, 8, 11]):

    # adding hooks
    model.blocks[hooks[0]].register_forward_hook(save_activation(model, "layer_1"))
    model.blocks[hooks[1]].register_forward_hook(save_activation(model, "layer_2"))
    model.blocks[hooks[2]].register_forward_hook(save_activation(model, "layer_3"))
    model.blocks[hooks[3]].register_forward_hook(save_activation(model, "layer_4"))

from util.dpt_blocks import (
    _make_scratch,
    FeatureFusionBlock,
    Interpolate,
    LayerNormBCHW,
)

class DecodeDPT(nn.Module):
    """Network for monocular depth estimation."""

    def __init__(self, cfg, features_out=256, features_final=128, non_negative=True):
        super(DecodeDPT, self).__init__()

        self.scratch = _make_scratch(cfg.features, features_out)

        self.act_postprocess1 = Postprocess(cfg.patch_size,
            nn.Conv2d(
                in_channels=cfg.vit_features,
                out_channels=cfg.features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=cfg.features[0],
                out_channels=cfg.features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            )
        )

        self.act_postprocess2 = Postprocess(cfg.patch_size,
            nn.Conv2d(
                in_channels=cfg.vit_features,
                out_channels=cfg.features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=cfg.features[1],
                out_channels=cfg.features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            )
        )

        self.act_postprocess3 = Postprocess(cfg.patch_size,
            nn.Conv2d(
                in_channels=cfg.vit_features,
                out_channels=cfg.features[2],
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        self.act_postprocess4 = Postprocess(cfg.patch_size,
            nn.Conv2d(
                in_channels=cfg.vit_features,
                out_channels=cfg.features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=cfg.features[3],
                out_channels=cfg.features[3],
                kernel_size=3,
                stride=2,
                padding=1,
            )
        )

        self.scratch.refinenet4 = FeatureFusionBlock(features_out)
        self.scratch.refinenet3 = FeatureFusionBlock(features_out)
        self.scratch.refinenet2 = FeatureFusionBlock(features_out)
        self.scratch.refinenet1 = FeatureFusionBlock(features_out)

        self.scratch.output_conv0 = nn.Conv2d(features_out, features_out, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv1 = nn.Sequential(
            nn.Conv2d(features_out, features_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features_out, features_final, kernel_size=1, stride=1, padding=0)
        )

        self.norm = LayerNormBCHW(features_final)

    def forward(self, enc, hw, with_norm=True):

        layer_1 = self.act_postprocess1(enc.layer_1, hw)
        layer_2 = self.act_postprocess2(enc.layer_2, hw)
        layer_3 = self.act_postprocess3(enc.layer_3, hw)
        layer_4 = self.act_postprocess4(enc.layer_4, hw)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv0(path_1)
        out = F.interpolate(out, hw, mode="bilinear", align_corners=False)
        out = self.scratch.output_conv1(out)
        #out = torch.squeeze(out, dim=1)

        if with_norm:
            out = self.norm(out)

        return out

######################
#    FlowFeat        #
######################

class FlowFeat(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.encoder, dpt_cfg = load_encoder(cfg)

        # encoder
        dpt_wrapper(self.encoder, dpt_cfg.hooks)

        # decoder
        self.decoder = DecodeDPT(dpt_cfg, cfg.fdim, cfg.fdim)

    @torch.no_grad()
    def forward_up(self, x):
        b,c,h,w = x.shape

        hh = h // self.cfg.patch_size
        ww = w // self.cfg.patch_size
        
        # hooks
        self.encoder(x)

        x_enc = self.encoder.layer_4[:, 1:] # skipping cls token
        x_enc = self.encoder.norm(x_enc)
        x_enc = x_enc.movedim(1, -1).view(b, -1, hh, ww)

        x = self.decoder(self.encoder, (h, w))
        return x_enc, x

    def forward(self, x):
        return self.forward_up(x)

class FlowFeatTrain(FlowFeat):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.fdim = cfg.fdim

        self.ridge_alpha = cfg.ridge_alpha
        self.input_size = cfg.input_size

        assert cfg.input_size[0] % cfg.patch_size == 0, "Height is not divisible by patch size"
        assert cfg.input_size[1] % cfg.patch_size == 0, "Wideht is not divisible by patch size"

        self.flow_loss = getattr(self, "flow_" + cfg.flow_loss)
        self.edge_loss = getattr(self, "edge_" + cfg.edge_loss)

        # EMA decoder
        self.decoder_ema = EMA(self.decoder, 
                               beta = cfg.decoder_momentum,
                               update_after_step = 1,
                               update_every = cfg.decoder_update_every)

        self.denorm = tf.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                    std=[1/0.229, 1/0.224, 1/0.225])

        self.norm = tf.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])

        self.flow = globals()[cfg.flownet](self.denorm)

    def parameter_groups(self):
        return [{"name": "decoder", "params": self.decoder.parameters()}]

    def flow_l1(self, pred_flow, teach_flow, **kwargs):
        l1_dist = torch.abs(pred_flow - teach_flow).sum(1, keepdim=True)
        return l1_dist.mean()

    def flow_l2(self, pred_flow, teach_flow, **kwargs):
        return F.mse_loss(pred_flow, teach_flow)

    def flow_l1smooth(self, pred_flow, teach_flow, beta=1., **kwargs):
        loss = F.smooth_l1_loss(pred_flow, teach_flow, beta=beta)
        return loss.mean()

    def flow_l1huber(self, pred_flow, teach_flow, beta=1., **kwargs):
        loss = F.huber_loss(pred_flow, teach_flow, delta=beta)
        return loss.mean()

    def edge_l1(self, x, y, sigma=1.0):
        return (1. - torch.exp(-y / sigma)) * torch.abs(x - y)

    def edge_l1norm(self, x, y, sigma=1.0):
        w = 1. - torch.exp(-y / sigma)
        w = w / w.sum((-1, -2), keepdim=True)
        return (w * torch.abs(x - y)).sum((-1, -2))

    def edge_l2norm(self, x, y, sigma=1.0):
        w = 1. - torch.exp(-y / sigma)
        w = w / w.sum((-1, -2), keepdim=True)
        return (w * (x - y)**2).sum((-1, -2))

    def edge_l1smooth(self, x, y, sigma=1.0):
        return F.smooth_l1_loss(x, y, beta=sigma)

    def edge_l1huber(self, x, y, sigma=1.0):
        return F.huber_loss(x, y, delta=sigma)

    def flow_boundary_loss(self, gt_flow, pred_flow, sigma, eps=1e-5):
        """Computes flow boundary loss"""
        grad_x = lambda x: torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        grad_y = lambda x: torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])

        pred_dx = grad_x(pred_flow)
        pred_dy = grad_y(pred_flow)

        gt_dx = grad_x(gt_flow)
        gt_dy = grad_y(gt_flow)

        loss_dx = self.edge_loss(pred_dx, gt_dx, sigma)
        loss_dy = self.edge_loss(pred_dy, gt_dy, sigma)

        return loss_dx.mean() + loss_dy.mean()

    def update_ema(self):
        self.decoder_ema.update()

    def forward_flow(self, flow, rX_ema, rX, alpha=0.1, mask_ratio=0.75):
        to_mat = lambda x: x.flatten(2, 3).movedim(1, -1)
        add_one = lambda x: torch.cat([x, torch.ones_like(x[..., :1])], -1)

        b,_,H,W = rX.shape

        flow_mat = to_mat(flow)
        d = flow_mat.shape[-1]

        X_ema = add_one(to_mat(rX_ema))
        f = X_ema.shape[-1]

        lhs = X_ema.transpose(-1, -2) @ X_ema / X_ema.shape[1]

        if alpha > 0.:
            lhs += alpha * torch.eye(f)[None, ...].expand(b, -1, -1).type_as(lhs)

        rhs = X_ema.transpose(-1, -2) @ flow_mat / X_ema.shape[1]

        A, res, rank, svs = torch.linalg.lstsq(lhs, rhs)

        pred_flow = add_one(to_mat(rX)) @ A
        pred_flow = pred_flow.movedim(1, -1).view(b, d, H, W)

        return pred_flow, A

    def forward_enc(self, crop1, crop2):
        b,c,h,w = crop1.shape

        with torch.no_grad():
            self.encoder(crop1)
            xT = self.decoder_ema(self.encoder, (h, w))

        # student
        with torch.no_grad():
            self.encoder(crop2)

        xS = self.decoder(self.encoder, (h, w))

        return xS, xT

    def crop_view(self, frame, params, input_size):
        b = frame.shape[0]
        affine_grid = F.affine_grid(params, (b, 1, input_size[0], input_size[1]), align_corners=False)
        frame_crop = F.grid_sample(frame, affine_grid, align_corners=False)
        return frame_crop, affine_grid

    def forward(self, frames, frame0, params1, params2, epoch=0.):
        """
        frames: [B, T, 3, H, W]
        """

        ### compute teacher flow
        # flow -> crop1 and crop2
        teacher_flow = self.flow(frames[:, 0], frames[:, 1])

        # normalising the flow
        if self.cfg.norm_flow:
            flow_mean = teacher_flow.mean((2, 3), keepdim=True)
            flow_std = teacher_flow.std((2, 3), keepdim=True)
            teacher_flow = (teacher_flow - flow_mean) / (flow_std + 1e-5)

        ### main ###

        b,T = frames.shape[:2]

        crop1, affine_grid1 = self.crop_view(frame0, params1, self.cfg.input_size)
        crop2, affine_grid2 = self.crop_view(frame0, params2, self.cfg.input_size)

        features, features_ema = self.forward_enc(crop1, crop2)

        outs = {}
        losses = {}

        teacher_flow1 = F.grid_sample(teacher_flow, affine_grid1, align_corners=False)
        teacher_flow2 = F.grid_sample(teacher_flow, affine_grid2, align_corners=False)

        # student flow
        student_flow, A = self.forward_flow(teacher_flow1, features_ema, features, self.ridge_alpha)

        losses["flowres"] = self.flow_loss(student_flow, teacher_flow2, beta=self.cfg.flow_beta)
        losses["flowbdr"] = self.flow_boundary_loss(teacher_flow2, student_flow, self.cfg.flow_edge_sigma)

        losses["total"] = 0.
        losses["total"] += self.cfg.flow_weight * losses["flowres"]
        losses["total"] += self.cfg.flow_edge_weight * losses["flowbdr"]

        tag = "flow_0"

        outs[tag] = student_flow.movedim(1, -1)[..., :2]
        outs["t_" + tag] = teacher_flow2.movedim(1, -1)
        outs["features"] = features
        outs["crop1"] = crop1
        outs["crop2"] = crop2

        return losses, outs
