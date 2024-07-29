from typing import Callable
import torch as t
from torch import Tensor
from transformer_lens.hook_points import HookedRootModule, HookPoint
from iit.utils.config import DEVICE
from .utils import MNIST_CLASS_MAP
from iit.utils.index import Ix
from iit.utils.nodes import HLNode, LLNode, HookName
from iit.utils.correspondence import Correspondence


class MNIST_PVR_Leaky_HL(HookedRootModule):
    def __init__(self, class_map: dict = MNIST_CLASS_MAP, device: t.device = DEVICE):
        super().__init__()
        hook_str = """hook_{}_leaked_to_{}"""
        self.leaky_hooks = {}
        self.hook_tl = HookPoint()
        self.hook_tr = HookPoint()
        self.hook_bl = HookPoint()
        self.hook_br = HookPoint()

        for i in ["tl", "tr", "bl", "br"]:
            for j in ["tl", "tr", "bl", "br"]:
                if i != j:
                    hl_node = HLNode(hook_str.format(i, j), 10, Ix[[None]])
                    self.leaky_hooks[hl_node] = HookPoint()
                    setattr(
                        self, hl_node.name, self.leaky_hooks[hl_node]
                    )  # needed as pytorch only checks  variables for named modules
        self.class_map = t.tensor(
            [class_map[i] for i in range(len(class_map))], dtype=t.long, device=device
        )
        self.setup()

    def get_idx_to_intermediate(self, name: str) -> Callable[[Tensor], Tensor]:
        if "hook_tl" in name:
            return lambda intermediate_vars: intermediate_vars[:, 0]
        if "hook_tr" in name:
            return lambda intermediate_vars: intermediate_vars[:, 1]
        if "hook_bl" in name:
            return lambda intermediate_vars: intermediate_vars[:, 2]
        if "hook_br" in name:
            return lambda intermediate_vars: intermediate_vars[:, 3]
        else:
            raise ValueError(f"Hook name {name} not recognised")

    def forward(self, args: tuple[t.Any, t.Any, Tensor]) -> Tensor:
        _, _, intermediate_data = args
        # print([a.shape for a in args])
        tl, tr, bl, br = [intermediate_data[:, i] for i in range(4)]
        # print(f"intermediate_data is a {type(intermediate_data)}; tl is a {type(tl)}")
        tl = self.hook_tl(tl)  # used while ablating
        tr = self.hook_tr(tr)
        bl = self.hook_bl(bl)
        br = self.hook_br(br)
        for k, v in self.leaky_hooks.items():
            if "hook_tl" in k.name:
                tl = v(tl)
            elif "hook_tr" in k.name:
                tr = v(tr)
            elif "hook_bl" in k.name:
                bl = v(bl)
            elif "hook_br" in k.name:
                br = v(br)
        pointer = self.class_map[(tl,)] - 1
        # TODO fix to support batching
        tr_bl_br = t.stack([tr, bl, br], dim=0)
        return tr_bl_br[pointer, range(len(pointer))]


hl = MNIST_PVR_Leaky_HL().to(DEVICE)


def get_corr(mode: str, hook_point: str, model: HookedRootModule, input_shape: tuple[int, int, int, int]) -> Correspondence:
    with t.no_grad():
        out, cache = model.run_with_cache(t.zeros(input_shape, device=DEVICE))
        input_shape = cache[hook_point].shape
        channel_size = input_shape[1]
        dim_at_hook = input_shape[2]
        assert input_shape[2] == input_shape[3], "Input shape is not square"

    if mode == "q":
        quadrant_size = dim_at_hook // 2
        tl_idx = Ix[None, None, :quadrant_size, :quadrant_size]
        tr_idx = Ix[None, None, :quadrant_size, quadrant_size : quadrant_size * 2]
        bl_idx = Ix[None, None, quadrant_size : quadrant_size * 2, :quadrant_size]
        br_idx = Ix[
            None,
            None,
            quadrant_size : quadrant_size * 2,
            quadrant_size : quadrant_size * 2,
        ]
        corr = {}
        for k in hl.leaky_hooks.keys():
            if "to_tl" in k.name:
                corr[k] = {
                    LLNode(
                        name=hook_point,
                        index=tl_idx,
                    )
                }
            elif "to_tr" in k.name:
                corr[k] = {LLNode(name=hook_point, index=tr_idx)}
            elif "to_bl" in k.name:
                corr[k] = {LLNode(name=hook_point, index=bl_idx)}
            elif "to_br" in k.name:
                corr[k] = {LLNode(name=hook_point, index=br_idx)}
            else:
                print(f"!!!!!! Skipping {k}")
        return Correspondence(corr)
    raise NotImplementedError(mode)
