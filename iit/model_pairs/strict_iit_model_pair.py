import numpy as np
import torch as t
from torch import Tensor
from transformer_lens.hook_points import HookedRootModule #type: ignore

from iit.model_pairs.base_model_pair import Callable, Tensor
from iit.model_pairs.iit_behavior_model_pair import IITBehaviorModelPair
from iit.model_pairs.ll_model import LLModel
import iit.utils.node_picker as node_picker
from iit.utils.nodes import LLNode
from iit.utils.metric import MetricStore, MetricStoreCollection, MetricType
from iit.utils.correspondence import Correspondence


class StrictIITModelPair(IITBehaviorModelPair):
    def __init__(
            self, 
            hl_model: HookedRootModule, 
            ll_model: HookedRootModule | LLModel, 
            corr: Correspondence, 
            training_args: dict = {}
            ):
        default_training_args = {
            "strict_weight": 1.0,
            "strict_weight_schedule" : lambda s, i: s,
            "siit_sampling" : "individual", # individual, sample_all, all
        }
        training_args = {**default_training_args, **training_args}
        super().__init__(hl_model, ll_model, corr=corr, training_args=training_args)
        self.nodes_not_in_circuit = node_picker.get_nodes_not_in_circuit(self.ll_model, self.corr)
        assert (
            self.training_args["iit_weight"] > 0
            or self.training_args["behavior_weight"] > 0
            or self.training_args["strict_weight"] > 0
        ), ValueError("At least one of the losses should be non-zero")

    @staticmethod
    def make_train_metrics() -> MetricStoreCollection:
        return MetricStoreCollection(
            [
                MetricStore("train/iit_loss", MetricType.LOSS),
                MetricStore("train/behavior_loss", MetricType.LOSS),
                MetricStore("train/strict_loss", MetricType.LOSS),
            ]
        )

    @staticmethod
    def make_test_metrics() -> MetricStoreCollection:
        return MetricStoreCollection(
            IITBehaviorModelPair.make_test_metrics().metrics
            + [MetricStore("val/strict_accuracy", MetricType.ACCURACY)],
        )

    def sample_ll_nodes(self) -> list[LLNode]:
        if self.training_args['siit_sampling'] == 'individual':
            ll_nodes = [self.rng.choice(np.array(self.nodes_not_in_circuit, dtype=object)),]
        elif self.training_args['siit_sampling'] == 'sample_all':
            importance = t.randint(0, 2, (len(self.nodes_not_in_circuit),)).to(bool).tolist()
            ll_nodes = [node for node, imp in zip(self.nodes_not_in_circuit, importance) if imp]
        elif self.training_args['siit_sampling'] == 'all':
            ll_nodes = self.nodes_not_in_circuit
        else:
            raise ValueError(f"Unexpected SIIT sampling mode: {self.training_args['siit_sampling']}")
        return ll_nodes
    
    def get_SIIT_loss_over_batch(
            self,
            base_input: tuple[Tensor, Tensor, Tensor],
            ablation_input: tuple[Tensor, Tensor, Tensor],
            loss_fn: Callable[[Tensor, Tensor], Tensor]
    ) -> Tensor:
        base_x, base_y = base_input[0:2]
        ablation_x, _ = ablation_input[0:2]
        ll_nodes = self.sample_ll_nodes()
        _, cache = self.ll_model.run_with_cache(ablation_x)
        self.ll_cache = cache
        hooks = []
        for ll_node in ll_nodes:
            hooks.append((ll_node.name, self.make_ll_ablation_hook(ll_node)))
        out = self.ll_model.run_with_hooks(
            base_x, fwd_hooks=hooks
        )
        # print(out.shape, base_y.shape)
        label_idx = self.get_label_idxs()
        siit_loss = (
            loss_fn(out[label_idx.as_index], base_y[label_idx.as_index].to(self.ll_model.cfg.device))
            * self.training_args["strict_weight"]
        ) # do this only for the tokens that we care about for IIT
        return siit_loss

    def run_train_step(
        self,
        base_input: tuple[Tensor, Tensor, Tensor],
        ablation_input: tuple[Tensor, Tensor, Tensor],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: t.optim.Optimizer,
    ) -> dict:
        use_single_loss = self.training_args["use_single_loss"]

        iit_loss = 0
        ll_loss = 0
        behavior_loss = 0

        if self.training_args["iit_weight"] > 0:
            hl_node = self.sample_hl_name()  # sample a high-level variable to ablate
            iit_loss = (
                self.get_IIT_loss_over_batch(base_input, ablation_input, hl_node, loss_fn)
                * self.training_args["iit_weight"]
            )
            if not use_single_loss:
                self.step_on_loss(iit_loss, optimizer)

        # loss for nodes that are not in the circuit
        # should not have causal effect on the high-level output
        if self.training_args["strict_weight"] > 0:
            siit_loss = (
            self.get_SIIT_loss_over_batch(base_input, ablation_input, loss_fn)
            * self.training_args["strict_weight"]
        )
            if not use_single_loss:
                self.step_on_loss(ll_loss, optimizer)

        if self.training_args["behavior_weight"] > 0:
            behavior_loss = (
                self.get_behaviour_loss_over_batch(base_input, loss_fn)
                * self.training_args["behavior_weight"]
            )
            if not use_single_loss:
                self.step_on_loss(behavior_loss, optimizer)

        if use_single_loss:
            total_loss = iit_loss + behavior_loss + siit_loss
            self.step_on_loss(total_loss, optimizer)

        return {
            "train/iit_loss": iit_loss.item() if isinstance(iit_loss, Tensor) else iit_loss,
            "train/behavior_loss": behavior_loss.item() if isinstance(behavior_loss, Tensor) else behavior_loss,
            "train/strict_loss": siit_loss.item() if isinstance(siit_loss, Tensor) else siit_loss,
        }

    def run_eval_step(
            self, 
            base_input: tuple[Tensor, Tensor, Tensor],
            ablation_input: tuple[Tensor, Tensor, Tensor],
            loss_fn: Callable[[Tensor, Tensor], Tensor]
            ) -> dict:
        eval_returns = super().run_eval_step(base_input, ablation_input, loss_fn)
        base_x, base_y = base_input[0:2]
        ablation_x, ablation_y = ablation_input[0:2]

        _, cache = self.ll_model.run_with_cache(ablation_x)
        label_idx = self.get_label_idxs()
        base_y = base_y[label_idx.as_index].to(self.ll_model.cfg.device)
        self.ll_cache = cache
        accuracies = []
        for node in self.nodes_not_in_circuit:
            out = self.ll_model.run_with_hooks(
                base_x, fwd_hooks=[(node.name, self.make_ll_ablation_hook(node))]
            )
            ll_output = out[label_idx.as_index]
            if self.hl_model.is_categorical():
                if ll_output.shape == base_y.shape:
                    base_y = t.argmax(base_y, dim=-1)
                top1 = t.argmax(ll_output, dim=-1)
                accuracy = (top1 == base_y).float().mean().item()
            else:
                accuracy = (
                    ((ll_output - base_y).abs() < self.training_args["atol"]).float().mean().item()
                )
            accuracies.append(accuracy)

        if len(accuracies) > 0:
            accuracy = float(np.mean(accuracies))
        else:
            accuracy = 1.0

        eval_returns["val/strict_accuracy"] = accuracy
        return eval_returns


    def _check_early_stop_condition(self, test_metrics: MetricStoreCollection) -> bool:
        metrics_to_check = []
        for metric in test_metrics:
            if (
                metric.get_name() == "val/strict_accuracy"
                and self.training_args["strict_weight"] > 0
            ):
                metrics_to_check.append(metric)
            if metric.get_name() == "val/accuracy" and self.training_args["behavior_weight"] > 0:
                metrics_to_check.append(metric)
            if metric.get_name() == "val/IIA" and self.training_args["iit_weight"] > 0:
                metrics_to_check.append(metric)
        return super()._check_early_stop_condition(MetricStoreCollection(metrics_to_check))
    

    def _run_epoch_extras(self, epoch_number: int) -> None:
        super()._run_epoch_extras(epoch_number)
        self.training_args['strict_weight'] = self.training_args['strict_weight_schedule'](self.training_args['strict_weight'], epoch_number)