import cv2
import torch
import torch.nn.functional as F

def warp_w_opflow(mask, flow):
    theta = torch.Tensor([[1, 0, 0], [0, 1, 0]]).to(mask.device)
    theta = theta.view(-1, 2, 3)
    h, w = mask.size()
    flow[:, :, 0] = flow[:, :, 0] / (w-1)
    flow[:, :, 1] = flow[:, :, 1] / (h-1)
    flow = flow.unsqueeze(0)
    grid = F.affine_grid(theta, (1, 1, h, w))
    grid = grid + 2*flow

    mask = mask.unsqueeze(0).unsqueeze(0)
    # align corner True or False shows similar results
    mask = F.grid_sample(mask, grid, mode='bilinear', align_corners=True)
    return mask.squeeze()

def rescale_flow(flow, scale_factor=60, renormalize=True):
    u, v = cv2.resize(flow, (scale_factor, scale_factor)).transpose(2, 0, 1)
    if renormalize:
        u = u*(flow.shape[1] / scale_factor)
        v = v*(flow.shape[0] / scale_factor)
    return u, v