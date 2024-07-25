from .mnist_pvr.dataset import ImagePVRDataset
from .mnist_pvr.utils import make_mnist_dataset
from .mnist_pvr.get_alignment import get_alignment as get_mnist_pvr_corr
from iit.utils.iit_dataset import IITDataset
from iit.utils.correspondence import Correspondence


def get_dataset(
    task: str, dataset_config: dict
) -> tuple[IITDataset, IITDataset]:
    if "pvr" in task:
        default_dataset_args = {
            "pad_size": 7,
            "train_size": 60000,
            "test_size": 10000,
        }
        default_dataset_args.update(dataset_config)
        if task == "mnist_pvr":
            unique_per_quad = False
        elif task == "pvr_leaky":
            unique_per_quad = False
        else:
            raise ValueError(f"Unknown task {task}")
        mnist_train, mnist_test = make_mnist_dataset()
        train_set = ImagePVRDataset(
            mnist_train,
            length=default_dataset_args["train_size"],
            pad_size=default_dataset_args["pad_size"],
            unique_per_quad=unique_per_quad,
        )
        test_set = ImagePVRDataset(
            mnist_test,
            length=default_dataset_args["test_size"],
            pad_size=default_dataset_args["pad_size"],
            unique_per_quad=unique_per_quad,
        )
    else:
        raise ValueError(f"Unknown task {task}")
    return IITDataset(train_set, train_set), IITDataset(test_set, test_set)


def get_alignment(task: str, config: dict = {}) -> Correspondence:
    if "pvr" in task:
        default_config = {
            "mode": "q",
            "hook_point": "mod.layer3.mod.1.mod.conv2.hook_point",
            "model": "resnet18",
            "pad_size": 7,
        }
        default_config.update(config)
        return get_mnist_pvr_corr(default_config, task)
    if "ioi" in task:
        from .ioi import corr
        return corr
    raise ValueError(f"Unknown task {task}")

def get_default_corr(task: str) -> dict:
    if "pvr" in task:
        return get_alignment(task)[-1]
    elif "ioi" in task:
        from .ioi import corr_dict
        return corr_dict
    raise ValueError(f"Unknown task {task}")
