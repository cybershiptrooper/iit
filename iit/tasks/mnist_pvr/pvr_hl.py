from iit.utils.nodes import HookName
import torch as t
from transformer_lens.hook_points import HookedRootModule, HookPoint
from iit.utils.config import DEVICE
from iit.utils.nodes import HLNode, LLNode
from iit.utils.index import Ix
from .utils import *
from iit.tasks.hl_model import HLModel

class MNIST_PVR_HL(HookedRootModule, HLModel):
    """
    A high-level implementation of the algorithm used for MNIST_PVR
    """

    def __init__(self, class_map=MNIST_CLASS_MAP, device=DEVICE):
        super().__init__()
        self.hook_tl = HookPoint()
        self.hook_tr = HookPoint()
        self.hook_bl = HookPoint()
        self.hook_br = HookPoint()
        self.class_map = t.tensor(
            [class_map[i] for i in range(len(class_map))], dtype=t.long, device=device
        )
        self.setup()

    def uses_intermediate_variables(self):
        return True

    def is_categorical(self):
        return True

    def get_idx_to_intermediate(self, name: HookName):
        """
        Returns a function that takes in a list of intermediate variables and returns the index of the one to use.
        """
        if name == "hook_tl":
            return lambda intermediate_vars: intermediate_vars[:, 0]
        elif name == "hook_tr":
            return lambda intermediate_vars: intermediate_vars[:, 1]
        elif name == "hook_bl":
            return lambda intermediate_vars: intermediate_vars[:, 2]
        elif name == "hook_br":
            return lambda intermediate_vars: intermediate_vars[:, 3]
        else:
            raise NotImplementedError(name)

    def forward(self, args):
        input, label, intermediate_data = args
        # print([a.shape for a in args])
        tl, tr, bl, br = [intermediate_data[:, i] for i in range(4)]
        # print(f"intermediate_data is a {type(intermediate_data)}; tl is a {type(tl)}")
        tl = self.hook_tl(tl)  # used while ablating
        tr = self.hook_tr(tr)
        bl = self.hook_bl(bl)
        br = self.hook_br(br)
        pointer = self.class_map[(tl,)] - 1
        # TODO fix to support batching
        tr_bl_br = t.stack([tr, bl, br], dim=0)
        return tr_bl_br[pointer, range(len(pointer))]


# %%
hl_nodes = {
    "hook_tl": HLNode("hook_tl", 10, None),
    "hook_tr": HLNode("hook_tr", 10, None),
    "hook_bl": HLNode("hook_bl", 10, None),
    "hook_br": HLNode("hook_br", 10, None),
}


def get_corr(mode, hook_point, model: HookedRootModule, input_shape):
    with t.no_grad():
        out, cache = model.run_with_cache(t.zeros(input_shape).to(DEVICE))
        # print(out.shape)
        output_shape = cache[hook_point].shape
        channel_size = output_shape[1]
        dim_at_hook = output_shape[2]
        assert (
            output_shape[2] == output_shape[3]
        ), f"Input shape is not square, got {output_shape}"

    if mode == "c":
        channel_stride = channel_size // 4
        corr = {
            hl_nodes["hook_tl"]: {
                LLNode(hook_point, Ix[None, :channel_stride, None, None])
            },
            hl_nodes["hook_tr"]: {
                LLNode(
                    hook_point,
                    Ix[None, channel_stride : channel_stride * 2, None, None],
                )
            },
            hl_nodes["hook_bl"]: {
                LLNode(
                    hook_point,
                    Ix[None, channel_stride * 2 : channel_stride * 3, None, None],
                )
            },
            hl_nodes["hook_br"]: {
                LLNode(
                    hook_point,
                    Ix[None, channel_stride * 3 : channel_stride * 4, None, None],
                )
            },
        }
    elif mode == "q":
        quadrant_size = (dim_at_hook) // 2  # conv has stride 2
        corr = {
            hl_nodes["hook_tl"]: {
                LLNode(hook_point, Ix[None, None, :quadrant_size, :quadrant_size])
            },
            hl_nodes["hook_tr"]: {
                LLNode(
                    hook_point,
                    Ix[None, None, :quadrant_size, quadrant_size : quadrant_size * 2],
                )
            },
            hl_nodes["hook_bl"]: {
                LLNode(
                    hook_point,
                    Ix[None, None, quadrant_size : quadrant_size * 2, :quadrant_size],
                )
            },
            hl_nodes["hook_br"]: {
                LLNode(
                    hook_point,
                    Ix[
                        None,
                        None,
                        quadrant_size : quadrant_size * 2,
                        quadrant_size : quadrant_size * 2,
                    ],
                )
            },
        }
    return corr
