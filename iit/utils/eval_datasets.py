import torch as t
from torch import Tensor

from iit.utils.config import DEVICE
from iit.utils.iit_dataset import IITDataset

class IITUniqueDataset(IITDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        return self.base_data[index]

    def __len__(self) -> int:
        return len(self.base_data)

    @staticmethod
    def collate_fn(batch: tuple, device: t.device = DEVICE) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        return IITDataset.get_encoded_input_from_torch_input(batch, device)

