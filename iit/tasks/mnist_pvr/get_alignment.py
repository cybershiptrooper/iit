from transformer_lens.hook_points import HookedRootModule

from .pvr_hl import get_corr, MNIST_PVR_HL
from .pvr_check_leaky_hl import get_corr as get_corr_leaky, MNIST_PVR_Leaky_HL
import torchvision
from iit.utils.config import DEVICE
import torch as t
from iit.utils.wrapper import HookedModuleWrapper
from iit.utils.correspondence import Correspondence


def get_alignment(config: dict, task: str) -> tuple[HookedModuleWrapper, HookedRootModule, Correspondence]:
    if config["model"] == "resnet18":
        resnet18 = torchvision.models.resnet18().to(DEVICE)  # 11M parameters
        resnet18.fc = t.nn.Linear(512, 10).to(DEVICE)
        ll_model = HookedModuleWrapper(
            resnet18, name="resnet18", recursive=True, get_hook_self=False
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model {config['model']}")

    if task == "mnist_pvr":
        hl_model = MNIST_PVR_HL().to(DEVICE)
        corr = get_corr(
            config["mode"], config["hook_point"], ll_model, config["input_shape"]
        )
    elif task == "pvr_leaky":
        hl_model = MNIST_PVR_Leaky_HL().to(DEVICE)
        corr = get_corr_leaky(
            config["mode"], config["hook_point"], ll_model, config["input_shape"]
        )
    else:
        raise ValueError(f"Unknown task {task}")
    return ll_model, hl_model, corr
