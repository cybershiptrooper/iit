import numpy as np
import torch as t

import iit.utils.node_picker as node_picker
from iit.model_pairs.base_model_pair import Callable, Tensor
from iit.model_pairs.iit_behavior_model_pair import IITBehaviorModelPair
from iit.model_pairs.nodes import LLNode
from iit.utils.metric import MetricStore, MetricStoreCollection, MetricType


class StrictIITModelPair(IITBehaviorModelPair):
    def __init__(self, hl_model, ll_model, corr, training_args={}):
        default_training_args = {
            "batch_size": 256,
            "lr": 0.001,
            "num_workers": 0,
            "use_single_loss": False,
            "iit_weight": 1.0,
            "behavior_weight": 1.0,
            "strict_weight": 1.0,
            "clip_grad_norm": 1.0,
        }
        training_args = {**default_training_args, **training_args}
        super().__init__(hl_model, ll_model, corr=corr, training_args=training_args)
        self.nodes_not_in_circuit = node_picker.get_nodes_not_in_circuit(
            self.ll_model, self.corr
        )

    @staticmethod
    def make_train_metrics():
        return MetricStoreCollection(
            [
                MetricStore("train/iit_loss", MetricType.LOSS),
                MetricStore("train/behavior_loss", MetricType.LOSS),
                MetricStore("train/strict_loss", MetricType.LOSS),
            ]
        )
    
    @staticmethod
    def make_test_metrics():
        return MetricStoreCollection(
            IITBehaviorModelPair.make_test_metrics().metrics + [MetricStore("val/strict_accuracy", MetricType.ACCURACY)],
        )

    def sample_ll_node(self) -> LLNode:
        return self.rng.choice(self.nodes_not_in_circuit)

    def run_train_step(
        self,
        base_input,
        ablation_input,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: t.optim.Optimizer,
    ):
        use_single_loss = self.training_args["use_single_loss"]

        iit_loss = 0
        ll_loss = 0
        behavior_loss = 0

        hl_node = self.sample_hl_name()  # sample a high-level variable to ablate
        iit_loss = (
            self.get_IIT_loss_over_batch(base_input, ablation_input, hl_node, loss_fn)
            * self.training_args["iit_weight"]
        )
        if not use_single_loss:
            self.step_on_loss(iit_loss, optimizer)

        # loss for nodes that are not in the circuit
        # should not have causal effect on the high-level output
        base_x, base_y = base_input[0:2]
        ablation_x, ablation_y = ablation_input[0:2]
        ll_node = self.sample_ll_node()
        _, cache = self.ll_model.run_with_cache(ablation_x)
        self.ll_cache = cache
        out = self.ll_model.run_with_hooks(
            base_x, fwd_hooks=[(ll_node.name, self.make_ll_ablation_hook(ll_node))]
        )
        # print(out.shape, base_y.shape)
        label_idx = self.get_label_idxs()
        ll_loss = (
            loss_fn(out[label_idx.as_index], base_y[label_idx.as_index].to(self.ll_model.cfg.device))
            * self.training_args["strict_weight"]
        ) # do this only for the tokens that we care about for IIT
        if not use_single_loss:
            self.step_on_loss(ll_loss, optimizer)

        behavior_loss = (
            self.get_behaviour_loss_over_batch(base_input, loss_fn)
            * self.training_args["behavior_weight"]
        )
        if not use_single_loss:
            self.step_on_loss(behavior_loss, optimizer)

        if use_single_loss:
            total_loss = iit_loss + behavior_loss + ll_loss
            self.step_on_loss(total_loss, optimizer)

        return {
            "train/iit_loss": iit_loss.item(),
            "train/behavior_loss": behavior_loss.item(),
            "train/strict_loss": ll_loss.item(),
        }

    def run_eval_step(self, base_input, ablation_input, loss_fn: Callable[[Tensor, Tensor], Tensor]):
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
                accuracy = ((ll_output - base_y).abs() < self.training_args["atol"]).float().mean().item()
            accuracies.append(accuracy)

        if len(accuracies) > 0:
            accuracy = np.mean(accuracies)
        else:
            accuracy = 1.0

        eval_returns["val/strict_accuracy"] = accuracy
        return eval_returns


    def _check_early_stop_condition(self, test_metrics: list[MetricStore]):
        metrics_to_check = []
        for metric in test_metrics:
            if metric.get_name() == "val/strict_accuracy" and self.training_args["strict_weight"] > 0:
                metrics_to_check.append(metric)
            if metric.get_name() == "val/accuracy" and self.training_args["behavior_weight"] > 0:
                metrics_to_check.append(metric)
            if metric.get_name() == "val/IIA" and self.training_args["iit_weight"] > 0:
                metrics_to_check.append(metric)
        return super()._check_early_stop_condition(metrics_to_check)