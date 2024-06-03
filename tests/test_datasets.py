"""
TODO: Add tests for the datasets module.

1. Create a small mnist dataset with 10 images
2. Test patch quadrant for each quadrant by patching and then dividing the image into 4 quadrants, then finding the exact match in the mnist dataset
"""

import numpy as np
import torch as t
from torch.utils.data import Dataset
from iit.tasks.mnist_pvr.dataset import ImagePVRDataset
from iit.tasks.mnist_pvr.pvr_check_leaky_hl import MNIST_PVR_Leaky_HL
from iit.utils.index import Ix
from iit.tasks.mnist_pvr.utils import make_mnist_dataset


class SmallMNIST(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def create_small_mnist():
    mnist_train, _= make_mnist_dataset()
    images = []
    labels = []
    np.random.seed(0)
    while len(images) < 10:
        image, label = mnist_train[np.random.randint(0, len(mnist_train))]
        if label not in labels:
            images.append(image)
            labels.append(label)
    return SmallMNIST(images, labels)


def test_get_input_shape():
    small_mnist = create_small_mnist()
    dataset = ImagePVRDataset(small_mnist, length=1, pad_size=3)
    assert dataset.get_input_shape() == (1, 3, (28 + 3 * 2) * 2, (28 + 3 * 2) * 2)


def test_patch_quadrant():
    np.random.seed(1)
    small_mnist = create_small_mnist()
    dataset = ImagePVRDataset(small_mnist, length=1, pad_size=0)
    image, label, intermediate_vars = dataset[0]
    hl_model = MNIST_PVR_Leaky_HL()
    patch_tl_out = dataset.patch_batch_at_hl(
        [image], [intermediate_vars], hl_model.hook_tl
    )
    patch_tr_out = dataset.patch_batch_at_hl(
        [image], [intermediate_vars], hl_model.hook_tr
    )
    patch_bl_out = dataset.patch_batch_at_hl(
        [image], [intermediate_vars], hl_model.hook_bl
    )
    patch_br_out = dataset.patch_batch_at_hl(
        [image], [intermediate_vars], hl_model.hook_br
    )

    # check if all other indices of patched images are unchanged
    _, _, width, height = dataset.get_input_shape()
    tl_idx = Ix[None, : width // 2, : height // 2].as_index
    tr_idx = Ix[None, : width // 2, height // 2 : height].as_index
    bl_idx = Ix[None, width // 2 : width, : height // 2].as_index
    br_idx = Ix[None, width // 2 : width, height // 2 : height].as_index

    assert t.all(patch_tl_out[0][0][tr_idx] == image[tr_idx])  # top right
    assert t.all(patch_tl_out[0][0][bl_idx] == image[bl_idx])  # bottom left
    assert t.all(patch_tl_out[0][0][br_idx] == image[br_idx])  # bottom right

    assert t.all(patch_tr_out[0][0][tl_idx] == image[tl_idx])  # top left
    assert t.all(patch_tr_out[0][0][bl_idx] == image[bl_idx])  # bottom left
    assert t.all(patch_tr_out[0][0][br_idx] == image[br_idx])  # bottom right

    assert t.all(patch_bl_out[0][0][tl_idx] == image[tl_idx])  # top left
    assert t.all(patch_bl_out[0][0][tr_idx] == image[tr_idx])  # top right
    assert t.all(patch_bl_out[0][0][br_idx] == image[br_idx])  # bottom right

    assert t.all(patch_br_out[0][0][tl_idx] == image[tl_idx])  # top left
    assert t.all(patch_br_out[0][0][tr_idx] == image[tr_idx])  # top right
    assert t.all(patch_br_out[0][0][bl_idx] == image[bl_idx])  # bottom left
