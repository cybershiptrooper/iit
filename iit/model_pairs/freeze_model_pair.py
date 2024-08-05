
import torch as t
from transformer_lens.hook_points import HookedRootModule #type: ignore

from iit.model_pairs.iit_behavior_model_pair import IITBehaviorModelPair
from iit.model_pairs.ll_model import LLModel
from iit.utils.correspondence import Correspondence
import iit.utils.node_picker as node_picker

class FreezedModelPair(IITBehaviorModelPair):

    def __init__(
            self, 
            hl_model: HookedRootModule, 
            ll_model: LLModel, 
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
        }
        training_args = {**default_training_args, **training_args}
        super().__init__(hl_model, ll_model, corr=corr, training_args=training_args)
        self.params_not_in_circuit = node_picker.get_params_not_in_circuit(corr, ll_model)
        self.wandb_method = "freeze_unwanted"

    def zero_grad_for_not_in_circuit(self) -> None:
        for ll_node in self.params_not_in_circuit:
            for name, param in self.ll_model.named_parameters():
                if ll_node.name == name:
                    param_idx = ll_node.index
                    param[param_idx.as_index].grad = t.zeros_like(param[param_idx.as_index])
                    # check other gradients are not affected/not zeroed
                    assert param.grad is not None 
                    # assert (param.grad.abs().sum() != 0) or (param_idx.as_index == index.Ix[[None]].as_index), f"got {param.grad.abs().sum()} and {param_idx.as_index} and {index.Ix[[None]].as_index}"
                    
    def step_on_loss(self, loss: t.Tensor, optimizer: t.optim.Optimizer) -> None:
        optimizer.zero_grad()
        loss.backward() # type: ignore
        self.zero_grad_for_not_in_circuit() # else, no need as we do it via hooks
        self.clip_grad_fn()
        optimizer.step()
