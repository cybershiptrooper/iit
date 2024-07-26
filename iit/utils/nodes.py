import dataclasses
from dataclasses import dataclass
import torch as t
from typing import Optional
from iit.utils.index import Ix, TorchIndex

HookName = str
HLCache = dict[HookName, t.Tensor]

@dataclass
class HLNode:
    name: HookName
    num_classes: int
    index: TorchIndex = Ix[[None]]

    def __post_init__(self) -> None:
        if self.index is None:
            self.index = Ix[[None]]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, HLNode):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        return False

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


@dataclass
class LLNode:
    name: HookName
    index: TorchIndex
    subspace: Optional[t.Tensor] = None

    def __post_init__(self) -> None:
        if self.index is None:
            self.index = Ix[[None]]

    def __eq__(self, other) -> bool:
        return isinstance(other, LLNode) and dataclasses.astuple(
            self
        ) == dataclasses.astuple(other)

    def __hash__(self) -> int:
        return hash(dataclasses.astuple(self))

    def get_index(self) -> tuple[slice]:
        return self.index.as_index