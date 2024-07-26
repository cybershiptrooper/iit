# import everything relevant
from typing import Optional
import numpy as np
from torch.utils.data import Dataset
from iit.utils.config import DEVICE
from torch.utils.data import DataLoader
import torch as t
from torch import Tensor


class IITDataset(Dataset):
    """
    Each thing is randomly sampled from a pair of datasets.
    """

    def __init__(
        self, 
        base_data: Dataset, 
        ablation_data: Dataset, 
        seed: int = 0, 
        every_combination: bool = False, 
        device: t.device = DEVICE
    ):
        # For vanilla IIT, base_data and ablation_data are the same
        self.base_data = base_data
        self.ablation_data = ablation_data
        self.seed = seed
        self.every_combination = every_combination
        self.device = device

    def __getitem__(self, index: int) -> tuple:
        if self.every_combination:
            base_index = index // len(self.ablation_data)
            ablation_index = index % len(self.ablation_data)
            base_input = self.base_data[base_index]
            ablation_input = self.ablation_data[ablation_index]
            return base_input, ablation_input

        # sample based on seed
        rng = np.random.default_rng(self.seed * 1000000 + index)
        base_index = rng.choice(len(self.base_data))
        ablation_index = rng.choice(len(self.ablation_data))

        base_input = self.base_data[base_index]
        ablation_input = self.ablation_data[ablation_index]

        return base_input, ablation_input

    def __len__(self) -> int:
        if self.every_combination:
            return len(self.base_data) * len(self.ablation_data)
        return len(self.base_data)

    @staticmethod
    def get_encoded_input_from_torch_input(
        xy: tuple, 
        device: t.device = DEVICE
        ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        zipped_data = tuple(zip(*xy))
        x, y = zipped_data[0:2]
        x = t.stack([x_i.to(device) for x_i in x])
        y = t.stack([y_i.to(device) for y_i in y])

        if len(zipped_data) == 3:
            int_vars = zipped_data[2]
            int_vars = t.stack([iv.to(device) for iv in int_vars])
            return x, y, int_vars
        else:
            return x, y

    @staticmethod
    def collate_fn(batch: list[Tensor] | Tensor, device: t.device = DEVICE) -> tuple[tuple, tuple]:
        if not isinstance(batch, list):
            # if batch is a single element, because batch_size was 1 or None, it is a tuple instead of a list
            batch = [batch]

        base_input, ablation_input = zip(*batch)
        return IITDataset.get_encoded_input_from_torch_input(
            base_input, device
        ), IITDataset.get_encoded_input_from_torch_input(ablation_input, device)

    def make_loader(
        self,
        batch_size: int,
        num_workers: int,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=lambda x: self.collate_fn(x, self.device),
        )


def train_test_split(
        dataset: Dataset, 
        test_size: float = 0.2, 
        random_state: Optional[int] = None
        ) -> list[t.utils.data.Subset]:
    n = len(dataset)
    split = int(n * test_size)
    if random_state is None:
        return t.utils.data.random_split(dataset, [n - split, split])
    return t.utils.data.random_split(
        dataset,
        [n - split, split],
        generator=t.Generator().manual_seed(random_state),
    )
