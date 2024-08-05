# import everything relevant
from typing import Optional, cast, Sized, Callable
import numpy as np
from torch.utils.data import Dataset
from iit.utils.config import DEVICE
from torch.utils.data import DataLoader
import torch as t
from torch import Tensor

dataset_len: Callable[[Dataset], int] = lambda dataset: len(cast(Sized, dataset))

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
            base_index = index // dataset_len(self.ablation_data)
            ablation_index = index % dataset_len(self.ablation_data)
            base_input = self.base_data[base_index]
            ablation_input = self.ablation_data[ablation_index]
            return base_input, ablation_input

        # sample based on seed
        rng = np.random.default_rng(self.seed * 1000000 + index)
        base_index = rng.choice(dataset_len(self.base_data))
        ablation_index = rng.choice(dataset_len(self.ablation_data))

        base_input = self.base_data[base_index]
        ablation_input = self.ablation_data[ablation_index]

        return base_input, ablation_input

    def __len__(self) -> int:
        if self.every_combination:
            return dataset_len(self.base_data) * dataset_len(self.ablation_data)
        return dataset_len(self.base_data)

    @staticmethod
    def get_encoded_input_from_torch_input(
        xy: tuple, 
        device: t.device = DEVICE
        ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        zipped_data = tuple(zip(*xy))
        x_in, y_in = zipped_data[0:2]
        x = t.stack([x_i.to(device) for x_i in x_in])
        y = t.stack([y_i.to(device) for y_i in y_in])

        if len(zipped_data) == 3:
            int_vars_in = zipped_data[2]
            int_vars = t.stack([iv.to(device) for iv in int_vars_in])
            return x, y, int_vars
        else:
            return x, y

    @staticmethod
    def collate_fn(
        batch: list[Tensor] | Tensor, 
        device: t.device = DEVICE
        ) -> tuple[tuple, tuple]:
        if not isinstance(batch, list):
            # if batch is a single element, because batch_size was 1 or None, it is a tuple instead of a list
            batch_list = [batch]
        else:
            batch_list = batch
        
        base_input_list, ablation_input_list = zip(*batch_list)
        return IITDataset.get_encoded_input_from_torch_input(
            base_input_list, device
        ), IITDataset.get_encoded_input_from_torch_input(ablation_input_list, device)

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
    
    def get_input_shape(self) -> t.Size:
        return self.base_data.get_input_shape() # type: ignore

def train_test_split(
        dataset: Dataset, 
        test_size: float = 0.2, 
        random_state: Optional[int] = None
        ) -> list[t.utils.data.Subset]:
    n = dataset_len(dataset)
    split = int(n * test_size)
    if random_state is None:
        return t.utils.data.random_split(dataset, [n - split, split])
    return t.utils.data.random_split(
        dataset,
        [n - split, split],
        generator=t.Generator().manual_seed(random_state),
    )
