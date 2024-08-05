from typing import Any, cast, Sized


import torch as t
from torch import Tensor
from torch.utils.data import Dataset

from iit.utils.config import DEVICE
from iit.utils.iit_dataset import IITDataset

class IITUniqueDataset(IITDataset):
    def __init__(self, base_data: Dataset, ablation_data: Dataset, seed: int = 0, every_combination: bool = False, device: t.device = DEVICE) -> None:
        super().__init__(base_data, ablation_data, seed, every_combination, device)

    def __getitem__(self, index: int) -> Any:
        return self.base_data[index]

    def __len__(self) -> int:
        return len(cast(Sized, self.base_data))
    
    @staticmethod
    def collate_fn(batch: tuple, device: t.device = DEVICE) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]: #type: ignore
        return IITDataset.get_encoded_input_from_torch_input(batch, device)

