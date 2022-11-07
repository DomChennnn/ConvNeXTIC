import sys
import torch

sys.path.append("..")
from models.cnexTIC import cnexTIC
from utils import get_config
from torch.hub import load_state_dict_from_url

from .pretrained import load_pretrained

__all__ = [
    "cnexTIC",
]

model_architectures = {
    "cnexTIC": cnexTIC,
}

models = {
    "cnexTIC": cnexTIC,
}

root_url = "./ckpts"
model_urls = {
    "cnexTIC": {
        "mse": {
            1: f"{root_url}/1/best.pt",
            2: f"{root_url}/2/best.pt",
            # 3: f"{root_url}/3/best.pt",
            # 3: f"/workspace/dmc/ConvNextIC/results/VCIP_channel192to320/3/snapshots/best.pt",
            3: f"/workspace/dmc/ConvNextIC/results/VCIP_VR/3/snapshots/060.pt",
            # 3: f"/workspace/dmc/ConvNextIC/results/VCIP_VR_VCIPfinetune/3/snapshots/150.pt",
            4: f"{root_url}/4/best.pt",
            5: f"{root_url}/5/best.pt",
            6: f"{root_url}/6/best.pt",
            7: f"{root_url}/7/best.pt",
            8: f"{root_url}/8/best.pt",
        },
        "ms-ssim": {
            1: f"{root_url}/",
            2: f"{root_url}/",
            3: f"{root_url}/",
            4: f"{root_url}/",
            5: f"{root_url}/",
            6: f"{root_url}/",
            7: f"{root_url}/",
            8: f"{root_url}/",
        },
    },
}

cfgs = {
    "cnexTIC": {
        1: (128, 192),
        2: (128, 192),
        3: (192, 320),
        4: (128, 192),
        5: (192, 320),
        6: (192, 320),
        7: (192, 320),
        8: (192, 320),
    },
}


def _load_model(
        architecture, metric, quality, pretrained=False, progress=True, **kwargs
):
    if architecture not in model_architectures:
        raise ValueError(f'Invalid architecture name "{architecture}"')

    if quality not in cfgs[architecture]:
        raise ValueError(f'Invalid quality value "{quality}"')

    if pretrained:
        if (
                architecture not in model_urls
                or metric not in model_urls[architecture]
                or quality not in model_urls[architecture][metric]
        ):
            raise RuntimeError("Pre-trained model not yet available")

        url = model_urls[architecture][metric][quality]
        print("Loading Ckpts From:", url)
        # state_dict = load_state_dict_from_url(url, progress=progress)
        state_dict = torch.load(url)
        # state_dict = load_pretrained(state_dict)

        config = get_config("config.yaml")
        config['embed_dim'] = cfgs[architecture][quality][0]
        config['latent_dim'] = cfgs[architecture][quality][1]
        model = model_architectures[architecture](config)
        model.load_state_dict(state_dict['model'])

        # TODO: should be put in traning loop
        model.update()

        # model = model_architectures[architecture].from_state_dict(state_dict)

    # model = model_architectures[architecture](*cfgs[architecture][quality], **kwargs)
    return model


def cnexTIC(quality, metric="mse", pretrained=False, progress=True, **kwargs):
    r"""
        Neural image compression framework from Lu, Ming and Guo, Peiyao and Shi, Huiqing and Cao, Chuntong and Ma, Zhan:
        `"Transformer-based Image Compression" <https://arxiv.org/abs/2111.06707>`, (DCC 2021).
    Args:
        quality (int): Quality levels (1: lowest, highest: 8)
        metric (str): Optimized metric, choose from ('mse')
        pretrained (bool): If True, returns a pre-trained model
    """
    if metric not in ("mse",):
        raise ValueError(f'Invalid metric "{metric}"')

    if quality < 1 or quality > 8:
        raise ValueError(f'Invalid quality "{quality}", should be between (1, 8)')

    return _load_model("cnexTIC", metric, quality, pretrained, progress, **kwargs)