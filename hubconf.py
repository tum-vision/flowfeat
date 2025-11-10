import torch
from torch.hub import load_state_dict_from_url

# Edit import path to your actual module
from model import FlowFeat


# Optional: declarative dependencies shown by `torch.hub.help`
dependencies = ["torch", "numpy", "tqdm", "Pillow"]

MODEL_CONFIG = {
    "dino_vits16_yt":   {"fdim": 128, "patch_size": 16, "url": "https://cvg.cit.tum.de/webshare/g/papers/flowfeat/dino_s16_flowfeat_yt.pth"},
    "dino_vitb16_yt":   {"fdim": 128, "patch_size": 16, "url": "https://cvg.cit.tum.de/webshare/g/papers/flowfeat/dino_b16_flowfeat_yt.pth"},
    "dino_vitb16_kt":   {"fdim": 128, "patch_size": 16, "url": "https://cvg.cit.tum.de/webshare/g/papers/flowfeat/dino_b16_flowfeat_kt.pth"},
    "mae_vitb16_kt":    {"fdim": 128, "patch_size": 16, "url": "https://cvg.cit.tum.de/webshare/g/papers/flowfeat/mae_b16_flowfeat_kt.pth"},
    "dinov2_vits14_yt": {"fdim": 128, "patch_size": 14, "url": "https://cvg.cit.tum.de/webshare/g/papers/flowfeat/dino2_s14_flowfeat_yt.pth"},
    "dinov2_vitb14_yt": {"fdim": 128, "patch_size": 14, "url": "https://cvg.cit.tum.de/webshare/g/papers/flowfeat/dino2_b14_flowfeat_yt.pth"},
    "dinov2_vitb14_kt": {"fdim": 128, "patch_size": 14, "url": "https://cvg.cit.tum.de/webshare/g/papers/flowfeat/dino2_b14_flowfeat_kt.pth"}
}

class Dict2Obj:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = Dict2Obj(v)
            setattr(self, k, v)


def flowfeat(name: str = "dinov2_vits14_yt",
             pretrained: bool = True,
             map_location=None,
             strict: bool = True,
             **kwargs):
    """
    Returns an instance of encoder-decoder FlowFeat.
    Args:
        name: (see above)
        pretrained: if True, loads weights from GitHub Release
        map_location: e.g. 'cpu' or torch.device('cuda:0')
        strict: passed to load_state_dict
        **kwargs: forwarded to MyModel constructor
    """
    if name not in MODEL_CONFIG:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Available: {list(MODEL_CONFIG.keys())}")

    cfg = MODEL_CONFIG[name]
    cfg["enc_snapshot"] = name[:-3]
    cfg.update(kwargs)
    cfg = Dict2Obj(cfg)

    model = FlowFeat(cfg)

    if pretrained:
        state_dict = load_state_dict_from_url(
            MODEL_CONFIG[name]["url"],
            map_location=map_location,
            check_hash=True,
            progress=True,
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            print(f"[hub] load_state_dict: missing={missing}, unexpected={unexpected}")

    model.eval()  # typical default for inference
    return model

