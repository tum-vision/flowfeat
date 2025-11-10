"""
Copyright (c) 2024 TU Munich
Author: Nikita Araslanov <nikita.araslanov@tum.de>
License: Apache License 2.0
"""

import pickle
import sys
import os
import math
import random
import glob
import tqdm

import torch
import torch.utils.data as data
import torchvision.transforms as tf_func
import torchvision as tv

from PIL import Image
import torch.nn.functional as F
import io

class MaskRandScaleCrop(object):

    def __init__(self, scale_range):
        self.scale_from, self.scale_to = scale_range

    def get_params(self, h, w):
        # generating random crop
        # preserves aspect ratio
        new_scale = random.uniform(self.scale_from, self.scale_to)

        new_h = int(new_scale * h)
        new_w = int(new_scale * w)

        # generating 
        if new_scale < 1.:
            assert w >= new_w, "{} vs. {} | {} / {}".format(w, new_w, h, new_h)
            i = random.randint(0, h - new_h)
            j = random.randint(0, w - new_w)
        else:
            assert w <= new_w, "{} vs. {} | {} / {}".format(w, new_w, h, new_h)
            i = random.randint(h - new_h, 0)
            j = random.randint(w - new_w, 0)

        return i, j, new_h, new_w, new_scale
    
    def get_affine_inv(self, affine, params, crop_size):

        aspect_ratio = crop_size[0] / crop_size[1]

        affine_inv = affine.clone()
        affine_inv[0,1] = affine[1,0] * aspect_ratio**2
        affine_inv[1,0] = affine[0,1] / aspect_ratio**2
        affine_inv[0,2] = -1 * (affine_inv[0,0] * affine[0,2] + affine_inv[0,1] * affine[1,2])
        affine_inv[1,2] = -1 * (affine_inv[1,0] * affine[0,2] + affine_inv[1,1] * affine[1,2])

        # scaling
        affine_inv /= torch.Tensor(params)[3].view(1,1)**2

        return affine_inv


    def get_affine(self, params, crop_size):
        # construct affine operator
        affine = torch.zeros(2, 3)

        aspect_ratio = crop_size[0] / crop_size[1] # float
        
        dy, dx, alpha, scale, flip = params

        # R inverse
        sin = math.sin(alpha * math.pi / 180.)
        cos = math.cos(alpha * math.pi / 180.)

        # inverse, note how flipping is incorporated
        affine[0,0], affine[0,1] = flip * cos, sin * aspect_ratio
        affine[1,0], affine[1,1] = -sin / aspect_ratio, cos

        # T inverse Rinv * t == R^T * t
        affine[0,2] = -1. * (cos * dx + sin * dy)
        affine[1,2] = -1. * (-sin * dx + cos * dy)

        # T
        affine[0,2] /= crop_size[1] // 2 # integer
        affine[1,2] /= crop_size[0] // 2 # integer

        # scaling
        affine *= scale
        
        affine = self.get_affine_inv(affine, params, crop_size)

        return affine

    def __call__(self, WH):

        affine = [0.,0.,0.,1.,1.]

        W, H = WH

        i2 = H / 2
        j2 = W / 2

        ii, jj, h, w, s = self.get_params(H, W)
        assert s < 1. and ii >= 0 and jj >= 0

        # displacement of the centre
        dy = ii + h / 2 - i2
        dx = jj + w / 2 - j2

        affine[0] = dy
        affine[1] = dx
        affine[3] = 1 / s # scale

        return self.get_affine(affine, (H, W))

class DataVideo(data.Dataset):

    def __init__(self, cfg, split, min_num_iter=10):
        super().__init__()

        self.id = random.random()
        print("My ID = ", self.id)

        self.cfg = cfg

        # train/val/test splits are pre-cut
        split_fn = split + ".txt"
        pickle_fn = split + ".pickle"
        assert os.path.isfile(split_fn), "File {} not found".format(split_fn)
        num_frames = 0

        if os.path.isfile(pickle_fn):
            with open(pickle_fn, 'rb') as handle:
                self.videos = pickle.load(handle)
        else:
            def check_dir(path):
                full_path = os.path.join(cfg.data.root, path.lstrip('/'))
                #assert os.path.isdir(full_path), '%s not found' % full_path
                return full_path

            def load_filenames(path):
                image_fns = sorted(glob.glob(path + "/*.jpeg") + \
                                   glob.glob(path + "/*.jpg"))
                return image_fns

            def load_masks(path):
                if not path is None:
                    return sorted(glob.glob(path + "/*.png"))
                return None

            self.videos = []

            with open(split_fn, "r") as lines:

                for n, line in tqdm.tqdm(enumerate(lines)):
                    #print(line)
                    paths = line.strip("\n").split(' ') + [None, None]
                    image_dir, mask_dir = paths[:2]
                    image_dir_path = check_dir(image_dir)

                    if not mask_dir is None:
                        mask_dir = check_dir(mask_dir)

                    images = load_filenames(image_dir_path)
                    masks = load_masks(mask_dir)

                    if len(images) < 10:
                        continue

                    self.videos.append({"images": images, "masks": masks, 
                                        "image_dir": image_dir,
                                        "has_masks": not mask_dir is None,
                                        "len": len(images)})

            with open(pickle_fn, 'wb') as handle:
                pickle.dump(self.videos, handle, protocol=pickle.HIGHEST_PROTOCOL)

        total_num_frames = sum([len(v["images"]) for v in self.videos])
        print("Loaded {} sequences | Total frames {}".format(len(self.videos), total_num_frames))
        self.num_iter = max(len(self.videos), min_num_iter * cfg.train.batch_size)

        self.tf_affine_crop1 = MaskRandScaleCrop(cfg.train.crop_range)
        self.tf_affine_crop2 = MaskRandScaleCrop(cfg.train.crop_range)
         
        self.tf = tf_func.Compose([tf_func.Resize(cfg.train.input_size),
                                   tf_func.ToTensor(),
                                   tf_func.Normalize(mean=[0.485, 0.456, 0.406], \
                                                     std =[0.229, 0.224, 0.225]) 
                                   ])

        self.tf_norm = tf_func.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])

    def __len__(self):
        return self.num_iter

    def __getitem__(self, index):

        index = index % len(self.videos)
        images = self.videos[index]["images"]

        gap = self.cfg.train.gap # 1 = next/prev frame; 2 can jump over frames, etc.
        timeflip = int(self.cfg.train.timeflip) # flipping time direction
        temp_win = self.cfg.train.temp_win


        frame_idx0 = random.randint(timeflip * gap * (temp_win - 1), \
                                    len(images) - 1 - (temp_win - 1) * gap)
        frame_idx1 = frame_idx0 + random.randint(1, gap) * (1 - timeflip * random.choice([0, 2]))

        image0 = Image.open(images[frame_idx0]).convert('RGB')
        image0 = tf_func.Resize(448)(image0)

        image0_ctr = tf_func.CenterCrop(min(image0.size[0], image0.size[1]))(image0)
        frame0 = self.tf(image0_ctr)

        image1 = Image.open(images[frame_idx1]).convert('RGB')
        image1 = tf_func.Resize(448)(image1)

        image1_ctr = tf_func.CenterCrop(min(image1.size[0], image1.size[1]))(image1)
        frame1 = self.tf(image1_ctr)

        affine_params1 = self.tf_affine_crop1(image0_ctr.size)
        affine_params2 = self.tf_affine_crop2(image0_ctr.size)

        image0_ctr = self.tf_norm(tf_func.ToTensor()(image0_ctr))

        return torch.stack([frame0, frame1], 0), image0_ctr, affine_params1, affine_params2
