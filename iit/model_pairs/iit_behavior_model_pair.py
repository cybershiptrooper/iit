from typing import Callable

import torch as t
from torch import Tensor
from transformer_lens.hook_points import HookedRootModule #type: ignore

from iit.model_pairs.iit_model_pair import IITModelPair
from iit.model_pairs.ll_model import LLModel
from iit.utils.correspondence import Correspondence
from iit.utils.metric import MetricStore, MetricStoreCollection, MetricType


class IITBehaviorModelPair(IITModelPair):
    def __init__(
            self,
            hl_model: HookedRootModule, 
            ll_model: LLModel, 
            corr: Correspondence, 
            training_args: dict = {}
            ):
        default_training_args = {
            "lr": 0.001,
            "atol": 5e-2,
            "early_stop": True,
            "use_single_loss": False,
            "iit_weight": 1.0,
            "behavior_weight": 1.0,
        }
        training_args = {**default_training_args, **training_args}
        super().__init__(hl_model, ll_model, corr=corr, training_args=training_args)
        self.wandb_method = "iit_and_behavior"

    @staticmethod
    def make_train_metrics() -> MetricStoreCollection:
        return MetricStoreCollection(
            [
                MetricStore("train/iit_loss", MetricType.LOSS),
                MetricStore("train/behavior_loss", MetricType.LOSS),
            ]
        )

    @staticmethod
    def make_test_metrics() -> MetricStoreCollection:
        return MetricStoreCollection(
            [
                MetricStore("val/iit_loss", MetricType.LOSS),
                MetricStore("val/IIA", MetricType.ACCURACY),
                MetricStore("val/accuracy", MetricType.ACCURACY),
            ]
        )

    def get_behaviour_loss_over_batch(
            self, 
            base_input: tuple[Tensor, Tensor, Tensor], 
            loss_fn: Callable[[Tensor, Tensor], Tensor]
            ) -> Tensor:
        base_x, base_y = base_input[0:2]
        output = self.ll_model(base_x)
        behavior_loss = loss_fn(output, base_y)
        return behavior_loss

    def step_on_loss(self, loss: Tensor, optimizer: t.optim.Optimizer) -> None:
        optimizer.zero_grad()
        loss.backward() # type: ignore
        self.clip_grad_fn()
        optimizer.step()

    def run_train_step(
        self,
        base_input: tuple[Tensor, Tensor, Tensor],
        ablation_input: tuple[Tensor, Tensor, Tensor],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: t.optim.Optimizer,
    ) -> dict:
        use_single_loss = self.training_args["use_single_loss"]

        hl_node = self.sample_hl_name()  # sample a high-level variable to ablate
        iit_loss = (
            self.get_IIT_loss_over_batch(base_input, ablation_input, hl_node, loss_fn)
            * self.training_args["iit_weight"]
        )
        if not use_single_loss:
            self.step_on_loss(iit_loss, optimizer)

        behavior_loss = (
            self.get_behaviour_loss_over_batch(base_input, loss_fn)
            * self.training_args["behavior_weight"]
        )
        if not use_single_loss:
            self.step_on_loss(behavior_loss, optimizer)

        if use_single_loss:
            total_loss = iit_loss + behavior_loss
            self.step_on_loss(total_loss, optimizer)

        return {
            "train/iit_loss": iit_loss.item(),
            "train/behavior_loss": behavior_loss.item(),
        }

    def run_eval_step(
        self,
        base_input: tuple[Tensor, Tensor, Tensor],
        ablation_input: tuple[Tensor, Tensor, Tensor],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> dict:
        atol = self.training_args["atol"]

        # compute IIT loss and accuracy
        label_idx = self.get_label_idxs()
        hl_node = self.sample_hl_name()
        hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
        hl_output.to(ll_output.device)
        hl_output = hl_output[label_idx.as_index]
        ll_output = ll_output[label_idx.as_index]
        if self.hl_model.is_categorical():
            loss = loss_fn(ll_output, hl_output)
            if ll_output.shape == hl_output.shape:
                # To handle the case when labels are one-hot
                hl_output = t.argmax(hl_output, dim=-1)
            top1 = t.argmax(ll_output, dim=-1)
            accuracy = (top1 == hl_output).float().mean()
            IIA = accuracy.item()
        else:
            loss = loss_fn(ll_output, hl_output)
            IIA = ((ll_output - hl_output).abs() < atol).float().mean().item()

        # compute behavioral accuracy
        base_x, base_y = base_input[0:2]
        output = self.ll_model(base_x)
        if self.hl_model.is_categorical():
            top1 = t.argmax(output, dim=-1)
            if output.shape == base_y.shape:
                # To handle the case when labels are one-hot
                # TODO: is there a better way?
                base_y = t.argmax(base_y, dim=-1)
            accuracy = (top1 == base_y).float().mean()
        else:
            accuracy = ((output - base_y).abs() < atol).float().mean()
        return {
            "val/iit_loss": loss.item(),
            "val/IIA": IIA,
            "val/accuracy": accuracy.item(),
        }
    

    def _check_early_stop_condition(self, test_metrics: MetricStoreCollection) -> bool:
        if self.training_args["iit_weight"] == 0:
            for metric in test_metrics:
                if metric.get_name() == "val/accuracy":
                    return metric.get_value() == 100
        else:
            return super()._check_early_stop_condition(test_metrics)
        return False