from typing import Callable

import numpy as np
import torch as t
from torch import Tensor
from transformer_lens.hook_points import HookedRootModule #type: ignore

from iit.model_pairs.base_model_pair import BaseModelPair
from iit.utils.correspondence import Correspondence
from iit.utils.metric import MetricStore, MetricStoreCollection, MetricType
from iit.model_pairs.ll_model import LLModel


class IITModelPair(BaseModelPair):
    def __init__(
        self,
        hl_model: HookedRootModule,
        ll_model: HookedRootModule | LLModel,
        corr: Correspondence,
        training_args: dict = {},
    ):
        self.hl_model = hl_model
        self.hl_model.requires_grad_(False)

        self.corr = corr
        assert all([str(k) in self.hl_model.hook_dict for k in self.corr.keys()])
        default_training_args = {
            "batch_size": 256,
            "num_workers": 0,
            "early_stop": True,
            "early_stop_accuracy_threshold": 99.5,
            "lr_scheduler": None,
            "scheduler_val_metric": ["val/accuracy", "val/IIA"],
            "scheduler_mode": "max",
            "scheduler_kwargs": {},
            "clip_grad_norm": 1.0,
            "seed": 0,
            "lr": 0.001,
            "detach_while_caching": True,
            "optimizer_cls": t.optim.Adam,
            "optimizer_kwargs" : {
                "betas": (0.9, 0.9)
            },
        }
        training_args = {**default_training_args, **training_args}
        
        if isinstance(ll_model, HookedRootModule):
            ll_model = LLModel.make_from_hooked_transformer(
                ll_model, detach_while_caching=training_args["detach_while_caching"]
            )
        self.ll_model = ll_model
        self.rng = np.random.default_rng(training_args.get("seed", 0))
        self.training_args = training_args
        self.wandb_method = "iit"

    @property
    def loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        # TODO: make this more general
        def class_loss(output: Tensor, target: Tensor) -> Tensor:
            # convert to (N, C, ...) if necessary
            if len(target.shape) == len(output.shape) and len(output.shape) > 2:
                # convert target to float if necessary
                if target.dtype not in [t.float32, t.float64]:
                    target = target.float()
                return t.nn.functional.cross_entropy(
                    output.transpose(-1, 1), target.transpose(-1, 1)
                )
            elif len(output.shape) > 2:
                return t.nn.functional.cross_entropy(output.transpose(-1, 1), target)
            assert len(output.shape) == 2  # N, C
            assert (
                len(target.shape) == 1 or target.shape == output.shape
            )  # argmax, or class probabilities
            return t.nn.functional.cross_entropy(output, target)

        try:
            if self.hl_model.is_categorical():
                return class_loss
            else:
                return t.nn.MSELoss()
        except AttributeError:
            print("WARNING: using default categorical loss function.")
            return class_loss

    @staticmethod
    def make_train_metrics() -> MetricStoreCollection:
        return MetricStoreCollection(
            [
                MetricStore("train/iit_loss", MetricType.LOSS),
            ]
        )

    @staticmethod
    def make_test_metrics() -> MetricStoreCollection:
        return MetricStoreCollection(
            [
                MetricStore("val/iit_loss", MetricType.LOSS),
                MetricStore("val/accuracy", MetricType.ACCURACY),
            ]
        )

    def run_eval_step(
        self,
        base_input: tuple[Tensor, Tensor, Tensor],
        ablation_input: tuple[Tensor, Tensor, Tensor],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> dict:
        hl_node = self.sample_hl_name()  # sample a high-level variable to ablate
        hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
        loss = loss_fn(ll_output, hl_output)
        top1 = t.argmax(ll_output, dim=-1)
        accuracy = (top1 == hl_output).float().mean()
        return {
            "val/iit_loss": loss.item(),
            "val/accuracy": accuracy.item(),
        }

    def run_train_step(
        self,
        base_input: tuple[Tensor, Tensor, Tensor],
        ablation_input: tuple[Tensor, Tensor, Tensor],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: t.optim.Optimizer,
    ) -> dict:
        optimizer.zero_grad()
        hl_node = self.sample_hl_name()  # sample a high-level variable to ablate
        loss = self.get_IIT_loss_over_batch(
            base_input, ablation_input, hl_node, loss_fn
        )
        loss.backward() # type: ignore
        optimizer.step()
        return {"train/iit_loss": loss.item()}
