# FlowFeat: Pixel-Dense Embedding of Motion Profiles
**Nikita Araslanov**, **Anna Sonnweber**, **Daniel Cremers**  
*NeurIPS 2025 (Spotlight)*

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

[[Paper]](https://cvg.cit.tum.de/_media/research/flowfeat/camera_ready.pdf) | [[Supplemental Material]](https://cvg.cit.tum.de/_media/research/flowfeat/camera_ready_supp.zip)

---

<p align="center">
  <img src="./assets/preview.gif" alt="Preview video" width="80%">
  <br>
  FlowFeat is a pixel-level feature representation learned from optical flow.
</p>




---

## Overview

This repository contains the official implementation of our NeurIPS 2025 Spotlight paper.

It includes code for model training, pretrained checkpoints, and a demo notebook.

---

## 🚀 Usage

You can load and run **FlowFeat** directly via **PyTorch Hub** or from a local clone of this repository.

🔹 Load from PyTorch Hub
```
import torch

# Load a pretrained FlowFeat model from PyTorch Hub
model = torch.hub.load(
    "tum-vision/flowfeat",
    "flowfeat",
    name="dinov2_vits14_yt",   # model variant
    pretrained=True
)

model.eval()
```

🔹 Load from a local clone
```bash
model = torch.hub.load(
    "./flowfeat",              # path to local repo clone
    "flowfeat",
    name="dinov2_vits14_yt",
    pretrained=True
)
```

🔹 Supported model variants
```
dino_vits16_yt
dino_vitb16_yt
dino_vitb16_kt
dinov2_vits14_yt
dinov2_vitb14_yt
dinov2_vitb14_kt
```

🔹 Example inference
```
import torch

x = torch.randn(1, 3, 224, 224)  # example input
with torch.no_grad():
    y_enc, y_dec = model(x)

print(y_enc.shape) # encoder features, e.g. (1,384,16,16)
print(y_dec.shape) # decoder features, e.g. (1,128,224,224)
```

## 🧰 Pre-trained Models
| **Model Name (`name`)** | **Backbone**    | **Train Dataset**  |  **Feature Dim** | **Checkpoint**                                                                          |
| :---------------------- | :-------------- | :----------- | :-------------: | :-------------------------------------------------------------------------------------- |
| `dino_vits16_yt`        | DINO ViT-S/16   | YouTube-VOS  |       128       | [Download](https://cvg.cit.tum.de/webshare/g/papers/flowfeat/dino_s16_flowfeat_yt.pth)  |
| `dino_vitb16_yt`        | DINO ViT-B/16   | YouTube-VOS  |       128       | [Download](https://cvg.cit.tum.de/webshare/g/papers/flowfeat/dino_b16_flowfeat_yt.pth)  |
| `dino_vitb16_kt`        | DINO ViT-B/16   | Kinetics |       128       | [Download](https://cvg.cit.tum.de/webshare/g/papers/flowfeat/dino_b16_flowfeat_kt.pth)  |
| `mae_vitb16_kt`        | DINO ViT-B/16   | Kinetics |       128       | [Download](https://cvg.cit.tum.de/webshare/g/papers/flowfeat/mae_b16_flowfeat_kt.pth)  |
| `dinov2_vits14_yt`      | DINOv2 ViT-S/14 | YouTube-VOS  |       128       | [Download](https://cvg.cit.tum.de/webshare/g/papers/flowfeat/dino2_s14_flowfeat_yt.pth) |
| `dinov2_vitb14_yt`      | DINOv2 ViT-B/14 | YouTube-VOS  |       128       | [Download](https://cvg.cit.tum.de/webshare/g/papers/flowfeat/dino2_b14_flowfeat_yt.pth) |
| `dinov2_vitb14_kt`      | DINOv2 ViT-B/14 | Kinetics |       128       | [Download](https://cvg.cit.tum.de/webshare/g/papers/flowfeat/dino2_b14_flowfeat_kt.pth) |


> 🔐 Note: Model weights are released under the same license as the codebase. Please cite the paper if you use these in your work.

## 🏋️‍♀️ Training Instructions

### Step 0. Clone the repository
Fetch the repository with the submodules (optical flow networks):
```bash
git clone --recurse-submodules https://github.com/tum-vision/flowfeat.git
```

### Step 1. Set up the environment
Create and activate a virtual environment, then install dependencies:
```
python -m venv flowfeat
source flowfeat/bin/activate  # (use `env\Scripts\activate` on Windows)
pip install -r requirements.txt
```

### Step 2. Set up the data

Follow the data setup instructions to link the required datasets (e.g., DAVIS2017, YouTube-VOS).

Download snapshots of an optical flow network into `models/`.
For example, for RAFT pre-trained on Sintel:
```bash
mkdir -p models && wget -O models/raft-sintel.pth <URL-RAFT-Sintel>
```
Follow the download links here:

| **Flow Network**                                                                 | **Link**                                                                                  |
| :------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------- |
| **RAFT**     | [README](https://github.com/princeton-vl/RAFT)    |
| **SEA-RAFT** | [README](https://github.com/princeton-vl/SEA-RAFT)     |
| **SMURF**  | [README](https://github.com/ChristophReich1996/SMURF) |

### Step 3. Set up Weights & Biases (wandb) and run training
1. Create a free wandb account. Create an entity and a project.
2. Run

```bash
python train.py --config-name=ytvos.yaml run.wandb_entity="<your_entity>" run.wandb_project="<your_wandb_project>"
```

Logs, metrics, and checkpoints will be automatically uploaded to your wandb workspace. By default, we evaluate the model on a subset of videos from DAVIS-2017.


## 🧪 Evaluation

We provide an reference implementation of the attention probe in `probe/attention.py`.
For full evaluation and benchmarking, we used [AnyProbe](https://openreview.net/pdf?id=q9r2DzPgiM) (coming soon).


## 📚 Citation
If you use this code or pretrained models, please cite our paper:

```
@inproceedings{Araslanov:2025:FlowFeat,
  author = {Araslanov, Nikita and Sonnweber, Anna and Cremers, Daniel},
  title = {{FlowFeat}: Pixel-Dense Embedding of Motion Profiles},
  booktitle = {NeurIPS},
  year = {2025},
}
```

<hr>
<sub>
<b>Acknowledgements:</b>
This work was supported by the ERC Advanced Grant SIMULACRON and DFG project CR 250/26-1 ``4D-YouTube''. We thank the open-source community for tools such as PyTorch and NumPy that made this work possible.
</sub>
