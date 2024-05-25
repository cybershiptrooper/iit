from iit.model_pairs.base_model_pair import *
from iit.utils.metric import *
from typing import final


class IITModelPair(BaseModelPair):
    def __init__(
        self,
        hl_model: HookedRootModule = None,
        ll_model: HookedRootModule = None,
        hl_graph=None,
        corr: dict[HLNode, set[LLNode]] = {},
        seed=0,
        training_args={},
    ):
        # TODO change to construct hl_model from graph?
        if hl_model is None:
            assert hl_graph is not None
            hl_model = self.make_hl_model(hl_graph)

        self.hl_model = hl_model
        self.ll_model = ll_model
        self.hl_model.requires_grad_(False)

        self.corr: dict[HLNode, set[LLNode]] = corr
        print(self.hl_model.hook_dict)
        print(self.corr.keys())
        assert all([str(k) in self.hl_model.hook_dict for k in self.corr.keys()])
        self.rng = np.random.default_rng(seed)
        default_training_args = {
            "batch_size": 256,
            "lr": 0.001,
            "num_workers": 0,
            "early_stop": True,
            "lr_scheduler": None,
            "scheduler_val_metric": ["val/accuracy", "val/IIA"],
            "scheduler_mode": "max",
            "clip_grad_norm": 1.0,
        }
        training_args = {**default_training_args, **training_args}
        self.training_args = training_args
        self.wandb_method = "iit"

    @property
    def loss_fn(self):
        return t.nn.CrossEntropyLoss()
    
    @loss_fn.setter
    def loss_fn(self, value):
        self._loss_fn = value

    @staticmethod
    def make_train_metrics():
        return MetricStoreCollection(
            [
                MetricStore("train/iit_loss", MetricType.LOSS),
            ]
        )

    @staticmethod
    def make_test_metrics():
        return MetricStoreCollection(
            [
                MetricStore("val/iit_loss", MetricType.LOSS),
                MetricStore("val/accuracy", MetricType.ACCURACY),
            ]
        )

    def run_eval_step(
        self,
        base_input,
        ablation_input,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ):
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
        base_input,
        ablation_input,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: t.optim.Optimizer,
    ):
        optimizer.zero_grad()
        hl_node = self.sample_hl_name()  # sample a high-level variable to ablate
        loss = self.get_IIT_loss_over_batch(
            base_input, ablation_input, hl_node, loss_fn
        )
        loss.backward()
        optimizer.step()
        return {"train/iit_loss": loss.item()}
