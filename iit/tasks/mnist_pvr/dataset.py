import torch as t
import torchvision
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import numpy as np
from .utils import *
from iit.utils.index import Ix, Index
from iit.utils.nodes import HLNode


class ImagePVRDataset(Dataset):
    """
    Turns the regular dataset into a PVR dataset.
    Images are concatenated into a 2x2 square.
    The label is the class of the image in position class_map[label of top left].
    """

    def __init__(
        self,
        base_dataset,
        class_map: dict[int, int] = MNIST_CLASS_MAP,
        seed=0,
        use_cache=True,
        length=200000,
        iid=True,
        pad_size=0,
        unique_per_quad=False,
    ):
        self.base_dataset = base_dataset
        self.class_map = class_map
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        assert all(v in {1, 2, 3} for v in class_map.values())
        self.cache = {}
        self.use_cache = False
        self.length = length
        self.iid = iid
        self.pad_size = pad_size
        self.use_cache = use_cache
        self.unique_per_quad = unique_per_quad
        if not self.iid:
            print("WARNING: using non-iid mode")
            assert (
                len(self.base_dataset) >= 4 * self.length
            ), "Dataset is too small for non-iid mode"
        self.input_shape = None
        self.set_input_shape(self[0][0].unsqueeze(0).shape)

    def set_input_shape(self, shape):
        self.input_shape = shape

    def get_input_shape(self):
        return self.input_shape

    @staticmethod
    def concatenate_2x2(images):
        """
        Concatenates four PIL.Image.Image objects into a 2x2 square.
        """
        assert len(images) == 4, "Need exactly four images"
        width, height = images[0].size
        new_image = Image.new("RGB", (width * 2, height * 2))

        new_image.paste(images[0], (0, 0))
        new_image.paste(images[1], (width, 0))
        new_image.paste(images[2], (0, height))
        new_image.paste(images[3], (width, height))

        return new_image

    def make_label_from_intermediate(self, intermediate_vars):
        """
        Returns the label for the new image based on the intermediate variables.
        """
        pointer = self.class_map[intermediate_vars[0].item()]
        new_label = t.tensor(intermediate_vars[pointer].item())
        return new_label

    def __getitem__(self, index):
        if index in self.cache and self.use_cache:
            return self.cache[index]
        if self.iid:
            self.rng = np.random.default_rng(self.seed * self.length + index)
            base_items = [
                self.base_dataset[self.rng.integers(0, len(self.base_dataset))]
                for i in range(4)
            ]
        else:
            base_items = [self.base_dataset[i] for i in range(index * 4, index * 4 + 4)]

        if self.unique_per_quad:
            # keep sampling until we get 4 different classes
            while len(set([item[1] for item in base_items])) < 4:
                base_items = [
                    self.base_dataset[self.rng.integers(0, len(self.base_dataset))]
                    for i in range(4)
                ]

        images = [base_item[0] for base_item in base_items]
        if self.pad_size > 0:
            images = [
                ImageOps.expand(image, border=self.pad_size, fill="black")
                for image in images
            ]
            # print(f"Padding images by {self.pad_size}")
        new_image = self.concatenate_2x2(images)
        new_image = torchvision.transforms.functional.to_tensor(new_image)

        base_label = base_items[0][1]
        pointer = self.class_map[base_label]
        new_label = t.tensor(base_items[pointer][1])
        intermediate_vars = t.tensor([base_items[i][1] for i in range(4)], dtype=t.long)
        new_label_from_func = self.make_label_from_intermediate(intermediate_vars)
        assert (
            new_label == new_label_from_func
        ), f"new_label: {new_label}; new_label_from_func: {new_label_from_func}"
        ret = new_image, new_label, intermediate_vars
        if self.use_cache:
            self.cache[index] = ret
        return ret

    def __len__(self):
        return self.length

    def patch_at_hl_idx(
        self,
        input: t.Tensor,
        intermediate_var: t.Tensor,
        idx: Index,
        idx_to_intermediate: int,
    ):
        """
        Patches the input and label to be compatible with the PVR model.
        """
        # sample new image until image at pointer is different from the original
        new_input = input.clone().detach()
        while True:
            quad_items = self.base_dataset[self.rng.integers(0, len(self.base_dataset))]
            quad_image = quad_items[0]
            if self.pad_size > 0:
                quad_image = ImageOps.expand(
                    quad_image, border=self.pad_size, fill="black"
                )
            quad_image = torchvision.transforms.functional.to_tensor(quad_image).to(
                input.device
            )
            if quad_image.shape[0] == 1:
                quad_image = quad_image.repeat(3, 1, 1)
            quad_label = quad_items[1]
            if quad_label != intermediate_var[idx_to_intermediate]:
                new_input[idx.as_index] = quad_image
                new_intermediate_var = intermediate_var.clone().detach()
                new_intermediate_var[idx_to_intermediate] = quad_label
                new_label = self.make_label_from_intermediate(new_intermediate_var)
                break

        return new_input, new_intermediate_var, new_label

    def patch_batch_at_hl(
        self, batch: list, intermediate_vars: list, hl_node: HLNode
    ):
        """
        Patches the input and label to be compatible with the PVR model.
        """
        idx, idx_to_intermediate = self.get_idx_and_intermediate(hl_node)
        new_batch = []
        new_intermediate_vars = []
        new_labels = []
        for i in range(len(batch)):
            new_in, new_vars, new_label = self.patch_at_hl_idx(
                batch[i], intermediate_vars[i], idx, idx_to_intermediate
            )
            new_batch.append(new_in)
            new_intermediate_vars.append(new_vars)
            new_labels.append(new_label)
        return new_batch, new_labels, new_intermediate_vars

    def get_idx_and_intermediate(self, hl_node: HLNode):
        input_shape = self.get_input_shape()
        width, height = input_shape[2], input_shape[3]
        if "hook_tl" in hl_node.name:
            idx = Ix[None, : width // 2, : height // 2]
            idx_to_intermediate = 0
        elif "hook_tr" in hl_node.name:
            idx = Ix[None, : width // 2, height // 2 : height]
            idx_to_intermediate = 1
        elif "hook_bl" in hl_node.name:
            idx = Ix[None, width // 2 : width, : height // 2]
            idx_to_intermediate = 2
        elif "hook_br" in hl_node.name:
            idx = Ix[None, width // 2 : width, height // 2 : height]
            idx_to_intermediate = 3
        else:
            raise ValueError(f"Hook name {hl_node.name} not recognised")
        return idx, idx_to_intermediate
