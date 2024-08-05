from typing import Callable

import torch as t
from torch import Tensor
from transformer_lens.hook_points import HookedRootModule, HookPoint


class HookedModuleWrapper(HookedRootModule):
    """
    Wraps any module, adding a hook after the output.
    """

    def __init__(
        self,
        mod: t.nn.Module,
        name: str = "model",
        recursive: bool = False,
        get_hook_self: bool = True,
        get_hook_pre: bool = False,
    ):
        super().__init__()
        self.mod = mod  # deepcopy(mod)
        if get_hook_pre:
            self.hook_pre = HookPoint()
            self.hook_pre.name = name + "pre"
        else:
            self.hook_pre = None
        if get_hook_self:
            hook_point = HookPoint()
            hook_point.name = name
            self.hook_point = hook_point
        else:
            self.hook_point = None
        if recursive:
            self.wrap_hookpoints_recursively()
        self.setup()

    def wrap_hookpoints_recursively(self, verbose: bool = False) -> None:
        show: Callable[[t.Any], None] = lambda *args: print(*args) if verbose else None
        for key, submod in list(self.mod._modules.items()):
            if isinstance(submod, HookedModuleWrapper):
                show(f"SKIPPING {key}:{type(submod)}")
                continue
            if key in ["intermediate_value_head", "value_head"]:  # these return tuples
                show(f"SKIPPING {key}:{type(submod)}")
                continue
            if isinstance(submod, t.nn.ModuleList):
                show(f"INDIVIDUALLY WRAPPING {key}:{type(submod)}")
                for i, subsubmod in enumerate(submod):
                    new_submod = HookedModuleWrapper(
                        subsubmod, name=f"{key}.{i}", recursive=True
                    )
                    submod[i] = new_submod
                continue

            if isinstance(submod, t.nn.Module):
                new_submod = HookedModuleWrapper(
                    submod, name=key, recursive=True
                )
                self.mod.__setattr__(key, new_submod)

    def forward(self, *args, **kwargs) -> Tensor: #type: ignore
        if self.hook_pre:
            result = self.mod.forward(self.hook_pre(*args, **kwargs))
        else:
            result = self.mod.forward(*args, **kwargs)
        if not self.hook_self:
            return result
        assert isinstance(result, Tensor)
        return self.hook_point(result)

def get_hook_points(model: HookedRootModule) -> list[str]:
    return [k for k in list(model.hook_dict.keys()) if "conv" in k]
