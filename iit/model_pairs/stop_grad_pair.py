from typing import Any, Callable

from torch import Tensor
import torch
from transformer_lens.hook_points import HookPoint, HookedRootModule
from transformer_lens import HookedTransformer

from iit.model_pairs.freeze_model_pair import FreezedModelPair
from iit.model_pairs.ll_model import LLModel
from iit.utils.nodes import LLNode
import iit.utils.node_picker as node_picker
from iit.utils.correspondence import Correspondence


class StopGradHookedModel:
    def __init__(
        self,
        model: HookedTransformer,
        params_not_in_circuit: list[LLNode],
        nodes_not_in_circuit: list[LLNode],
        post_nodes_not_in_circuit: list[LLNode],
        scale: float = 1e6,
        use_forward_hooks: bool = True,
    ):
        self.model = model
        self.params_not_in_circuit = params_not_in_circuit
        self.nodes_not_in_circuit = nodes_not_in_circuit
        self.post_nodes_not_in_circuit = post_nodes_not_in_circuit
        self.scale = scale
        self.use_forward_hooks = use_forward_hooks
        # for params in self.params_not_in_circuit:
        #     for param_name, param in self.model.named_parameters():
        #         if params.name == param_name:
        #             param = param/1e6 # make the parameter very small

    def __getattr__(self, __name: str) -> Any:
        if hasattr(self.model, __name):
            return getattr(self.model, __name)
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{__name}'"
            )

    @staticmethod
    def make_ln_hook(ll_node: LLNode, scale: float) -> Callable[[Tensor, HookPoint], Tensor]:
        def hook_fn(hook_point_out: Tensor, hook: HookPoint) -> Tensor:
            return (
                hook_point_out / scale
            )  # TODO: this won't work when individual heads are switched on/off

        return hook_fn

    @staticmethod
    def make_detached_hook(ll_node: LLNode) -> Callable[[Tensor, HookPoint], Tensor]:
        print(f"Attaching hook to {ll_node.name}")

        def hook_fn(hook_point_out: Tensor, hook: HookPoint) -> Tensor:
            act_idx = ll_node.get_index()
            hook_point_out[act_idx] = (
                hook_point_out[act_idx].clone().detach()
            )  # do I need to clone?
            # check if the gradient is detached
            # assert hook_point_out[act_idx].requires_grad == False, f"hook_point_out[act_idx].requires_grad is {hook_point_out[act_idx].requires_grad}, expected False"
            # print(ll_node, hook_point_out[act_idx].requires_grad, hook_point_out.shape)
            return hook_point_out

        return hook_fn

    def make_zero_grad_hook(self, ll_node: LLNode) -> Callable[[Tensor, HookPoint], Tensor]:
        def hook_fn(grad: Tensor, hook: HookPoint) -> Tensor:
            act_idx = ll_node.get_index()
            ori_grad_shape = grad.shape
            grad[act_idx] = torch.zeros_like(grad[act_idx])
            assert (
                grad.shape == ori_grad_shape
            ), f"grad.shape is {grad.shape}, expected {ori_grad_shape}"
            assert (
                grad[act_idx].abs().sum() == 0
            ), f"grad[act_idx].abs().sum() is {grad[act_idx].abs().sum()}, expected 0"
            return [grad]

        return hook_fn

    def forward(self, x: Tensor) -> Tensor:
        self.model.reset_hooks()
        if self.use_forward_hooks:
            return self.model.run_with_hooks(
                x,
                fwd_hooks=[
                    (ll_node.name, self.make_ln_hook(ll_node, scale=self.scale))
                    for ll_node in self.post_nodes_not_in_circuit
                ],
                bwd_hooks=[
                    (ll_node.name, self.make_zero_grad_hook(ll_node))
                    for ll_node in self.nodes_not_in_circuit
                ],
                reset_hooks_end=False,
            )
        else:
            return self.model.run_with_hooks(
                x,
                bwd_hooks=[
                    (ll_node.name, self.make_zero_grad_hook(ll_node))
                    for ll_node in self.nodes_not_in_circuit
                ],
                reset_hooks_end=False,
            )

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class StopGradModelPair(FreezedModelPair):
    def __init__(
            self, 
            hl_model: HookedRootModule, 
            ll_model: HookedRootModule | LLModel, 
            corr: Correspondence, 
            training_args: dict = {}
            ):
        default_training_args = {
            "batch_size": 256,
            "lr": 0.001,
            "num_workers": 0,
            "use_single_loss": False,
            "iit_weight": 1.0,
            "behavior_weight": 1.0,
            "scale": 1e6,
            "use_ln_hooks": True,
        }
        training_args = {**default_training_args, **training_args}
        super().__init__(hl_model, ll_model, corr=corr, training_args=training_args)
        params_not_in_circuit = node_picker.get_params_not_in_circuit(corr, ll_model)
        nodes_not_in_circuit = node_picker.get_nodes_not_in_circuit(ll_model, corr)
        post_nodes_not_in_circuit = node_picker.get_post_nodes_not_in_circuit(
            ll_model, corr
        )
        # self.make_backward_hooks()
        self.ll_model = StopGradHookedModel(
            ll_model,
            params_not_in_circuit,
            nodes_not_in_circuit,
            post_nodes_not_in_circuit,
            scale=training_args["scale"],
            use_forward_hooks=training_args["use_ln_hooks"],
        )
        self.wandb_method = "stop grads"

        # TODO: test another part of the model and see if the gradient changes after registering the hook

    # @staticmethod
    # def make_detached_hook(ll_node: LLNode):
    #     def hook_fn(grad):
    #         act_idx = node_picker.get_activation_idx(ll_node)
    #         print(grad.shape, act_idx)
    #         grad[act_idx.as_index] = t.zeros_like(grad[act_idx.as_index])
    #     return hook_fn

    # def make_backward_hooks(self):
    #     for ll_node in self.params_not_in_circuit:
    #         for name, param in self.ll_model.named_parameters():
    #             if ll_node.name == name:
    #                 hook = self.make_detached_hook(ll_node)
    #                 print(f"Attaching hook to {name}")
    #                 param.register_hook(hook)
