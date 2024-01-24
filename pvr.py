# %%
"""
The MNIST-PVR task. We need to download the MNIST dataset and process it.
"""

import numpy as np
import torchvision.datasets as datasets
import torch as t
import torchvision
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from typing import Optional
from transformer_lens.hook_points import HookedRootModule, HookPoint

DEVICE = t.device('cuda' if t.cuda.is_available() else 'cpu')

# %%

mnist_train = datasets.MNIST("./data", download=True)
mnist_test = datasets.MNIST("./data", train=False, download=True)
# %%

MNIST_CLASS_MAP = {k: [1, 1, 1, 1, 2, 2, 2, 3, 3, 3][k] for k in range(10)}

class ImagePVRDataset(Dataset):
    """
    Turns the regular dataset into a PVR dataset.
    Images are concatenated into a 2x2 square.
    The label is the class of the image in position class_map[label of top left].
    """
    def __init__(self, base_dataset, class_map:dict[int, int]=MNIST_CLASS_MAP, seed=0, use_cache=True, length=200000, iid=True, pad_size=0):
        self.base_dataset = base_dataset
        self.class_map = class_map
        self.seed=seed
        self.rng = np.random.default_rng(seed)
        assert all(v in {1, 2, 3} for v in class_map.values())
        self.cache = {}
        self.use_cache = False
        self.length = length
        self.iid = iid
        self.pad_size = pad_size
        if use_cache:
            self.use_cache = True
        if not self.iid:
            print("WARNING: using non-iid mode")
            assert len(self.base_dataset) >= 4*self.length, "Dataset is too small for non-iid mode"

    @staticmethod
    def concatenate_2x2(images):
        """
        Concatenates four PIL.Image.Image objects into a 2x2 square.
        """
        assert len(images) == 4, "Need exactly four images"
        width, height = images[0].size
        new_image = Image.new('RGB', (width * 2, height * 2))

        new_image.paste(images[0], (0, 0))
        new_image.paste(images[1], (width, 0))
        new_image.paste(images[2], (0, height))
        new_image.paste(images[3], (width, height))

        return new_image
    
    def __getitem__(self, index):
        if index in self.cache and self.use_cache:
            return self.cache[index]
        if self.iid:
            self.rng = np.random.default_rng(self.seed * self.length + index)
            base_items = [self.base_dataset[self.rng.integers(0, len(self.base_dataset))] for i in range(4)]
        else:
            base_items = [self.base_dataset[i] for i in range(index * 4, index * 4 + 4)]
        images = [base_item[0] for base_item in base_items]
        if self.pad_size > 0:
            images = [ImageOps.expand(image, border=self.pad_size, fill='black') for image in images]
            # print(f"Padding images by {self.pad_size}")
        new_image = self.concatenate_2x2(images)
        new_image = torchvision.transforms.functional.to_tensor(new_image)

        base_label = base_items[0][1]
        pointer = self.class_map[base_label]
        new_label = t.tensor(base_items[pointer][1])
        intermediate_vars = t.tensor([base_items[i][1] for i in range(4)], dtype=t.long)
        ret = new_image, new_label, intermediate_vars
        if self.use_cache:
            self.cache[index] = ret
        return ret
    
    def __len__(self):
        return self.length


# %%

mnist_pvr_train = ImagePVRDataset(mnist_train, length=200000, pad_size=7) # because first conv layer is 7
mnist_pvr_test = ImagePVRDataset(mnist_test, length=20000, pad_size=7)
# %%

class MNIST_PVR_HL(HookedRootModule):
    """
    A high-level implementation of the algorithm used for MNIST_PVR
    """
    def __init__(self, class_map=MNIST_CLASS_MAP, device=DEVICE):
        super().__init__()
        self.hook_tl = HookPoint()
        self.hook_tr = HookPoint()
        self.hook_bl = HookPoint()
        self.hook_br = HookPoint()
        self.class_map = t.tensor([class_map[i] for i in range(len(class_map))], dtype=t.long, device=device)
        self.setup()

    def forward(self, args):
        input, label, intermediate_data = args
        # print([a.shape for a in args])
        tl, tr, bl, br = [intermediate_data[:, i] for i in range(4)]
        # print(f"intermediate_data is a {type(intermediate_data)}; tl is a {type(tl)}")
        tl = self.hook_tl(tl)
        tr = self.hook_tr(tr)
        bl = self.hook_bl(bl)
        br = self.hook_br(br)
        pointer = self.class_map[(tl,)] - 1
        # TODO fix to support batching
        tr_bl_br = t.stack([tr, bl, br], dim=0)
        return tr_bl_br[pointer, range(len(pointer))]

# %%
hl = MNIST_PVR_HL()
hl.hook_dict
# %%

def visualize_datapoint(dataset, index):
    image, label, intermediate_vars = dataset[index]
    print(f"Label: {label}")
    print(f"Intermediate vars: {intermediate_vars}")
    print(f"Image shape: {image.shape}")
    image = torchvision.transforms.functional.to_pil_image(image)
    image.show()

# visualize_datapoint(mnist_pvr_train, 3)
# %%