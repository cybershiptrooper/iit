import os
from abc import ABC, abstractmethod
from typing import Any, Callable, final, Type, Optional

import numpy as np
import torch as t
import transformer_lens as tl # type: ignore
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm # type: ignore
from transformer_lens.hook_points import HookedRootModule, HookPoint # type: ignore
from IPython.display import clear_output

import wandb # type: ignore
from iit.model_pairs.ll_model import LLModel
from iit.utils.nodes import HLNode, LLNode
from iit.utils.config import WANDB_ENTITY
from iit.utils.correspondence import Correspondence
from iit.utils.iit_dataset import IITDataset
from iit.utils.index import Ix, TorchIndex
from iit.utils.metric import MetricStoreCollection, MetricType

def in_notebook() -> bool:
    try:
        # This will only work in Jupyter notebooks
        shell = get_ipython().__class__.__name__ # type: ignore
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other types of interactive shells
    except NameError:
        return False  # Probably standard Python interpreter
    
if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class BaseModelPair(ABC):
    hl_model: HookedRootModule
    ll_model: 'LLModel' # see iit/model_pairs/ll_model.py
    hl_cache: tl.ActivationCache
    ll_cache: tl.ActivationCache
    corr: 'Correspondence' # hl_model -> ll_model activation correspondence. Capital Pi in paper
    training_args: dict[str, Any]
    wandb_method: str
    rng: np.random.Generator
    dataset_class: 'IITDataset'

    ##########################################
    # Abstract methods you need to implement #
    ##########################################
    @property
    @abstractmethod
    def loss_fn(self) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    @abstractmethod
    def make_train_metrics() -> MetricStoreCollection:
        pass

    @staticmethod
    @abstractmethod
    def make_test_metrics() -> MetricStoreCollection:
        pass

    @abstractmethod
    def run_train_step(
        self,
        base_input: tuple[Tensor, Tensor, Tensor],
        ablation_input: tuple[Tensor, Tensor, Tensor],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: t.optim.Optimizer,
    ) -> dict:
        pass

    @abstractmethod
    def run_eval_step(
        self,
        base_input: tuple[Tensor, Tensor, Tensor],
        ablation_input: tuple[Tensor, Tensor, Tensor],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> dict:
        pass

    ###########################################
    ##### Mutable methods you can override ####
    ###########################################
    def do_intervention(
        self,
        base_input: tuple[Tensor, Tensor, Tensor],
        ablation_input: tuple[Tensor, Tensor, Tensor],
        hl_node: HLNode,
        verbose: bool = False
    ) -> tuple[Tensor, Tensor]:
        ablation_x, ablation_y = ablation_input[0:2]
        base_x, base_y = base_input[0:2]

        hl_ablation_output, self.hl_cache = self.hl_model.run_with_cache(ablation_input)
        ll_ablation_output, self.ll_cache = self.ll_model.run_with_cache(ablation_x)
        ll_nodes = self.corr[hl_node]

        hl_output = self.hl_model.run_with_hooks(
            base_input, fwd_hooks=[(hl_node.name, self.make_hl_ablation_hook(hl_node))]
        )
        ll_output = self.ll_model.run_with_hooks(
            base_x,
            fwd_hooks=[
                (ll_node.name, self.make_ll_ablation_hook(ll_node))
                for ll_node in ll_nodes
            ],
        )

        if verbose:
            print(f"{base_x=}, {base_y.item()=}")
            print(f"{hl_ablation_output=}, {ll_ablation_output.shape=}")
            print(f"{hl_node=}, {ll_nodes=}")
            print(f"{hl_output=}")
        return hl_output, ll_output

    @staticmethod
    def get_label_idxs() -> TorchIndex:
        '''
        Returns the index of the label for which the IIT loss is computed. 
        NOT to be used for computing the behavior loss.
        '''
        return Ix[[None]]

    def set_corr(self, corr: Correspondence) -> None:
        self.corr = corr

    def sample_hl_name(self) -> HLNode:
        return self.rng.choice(np.array(list(self.corr.keys())))

    def make_hl_ablation_hook(self, hl_node: HLNode) -> Callable[[Tensor, HookPoint], Tensor]:
        assert isinstance(hl_node, HLNode), ValueError(
            f"hl_node is not an instance of HLNode, but {type(hl_node)}"
        )

        def hl_ablation_hook(hook_point_out: Tensor, hook: HookPoint) -> Tensor:
            out = hook_point_out.clone()

            if isinstance(out, float) or isinstance(out, int):
                assert (
                    hl_node.index is Ix[[None]] or hl_node.index is None
                ), "scalars cannot be indexed"
                return self.hl_cache[hook.name]

            out[hl_node.index.as_index] = self.hl_cache[hook.name][
                hl_node.index.as_index
            ]
            return out

        if hl_node.index is not None:
            return hl_ablation_hook
        else:
            return self.hl_ablation_hook

    def hl_ablation_hook(
        self, 
        hook_point_out: Tensor, 
        hook: HookPoint
    ) -> Tensor:  # TODO: remove this
        out = self.hl_cache[hook.name]
        return out

    # TODO extend to position and subspace...
    def make_ll_ablation_hook(
        self, ll_node: LLNode
    ) -> Callable[[Tensor, HookPoint], Tensor]:
        if ll_node.subspace is not None:
            raise NotImplementedError

        def ll_ablation_hook(hook_point_out: Tensor, hook: HookPoint) -> Tensor:
            # This works because out is being used in a computation that autograd can track later on
            # So the clone is still connected to the original tensor's computation graph
            # For why is a cloned tensor part of the computation graph, 
            # see here: https://discuss.pytorch.org/t/why-is-the-clone-operation-part-of-the-computation-graph-is-it-even-differentiable/67054/4
            out = hook_point_out.clone()
            index = ll_node.index if ll_node.index is not None else Ix[[None]]
            out[index.as_index] = self.ll_cache[hook.name][index.as_index]
            return out

        return ll_ablation_hook

    def get_IIT_loss_over_batch(
        self,
        base_input: tuple[Tensor, Tensor, Tensor],
        ablation_input: tuple[Tensor, Tensor, Tensor],
        hl_node: HLNode,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
    ) -> Tensor:
        hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
        label_idx = self.get_label_idxs()
        # IIT loss is only computed on the tokens we care about
        loss = loss_fn(ll_output[label_idx.as_index].to(hl_output.device), hl_output[label_idx.as_index])
        return loss

    def clip_grad_fn(self) -> None:
        if self.training_args["clip_grad_norm"]:
            t.nn.utils.clip_grad_norm_(
                self.ll_model.parameters(), self.training_args["clip_grad_norm"]
            )

    def step_scheduler(
            self, 
            lr_scheduler: t.optim.lr_scheduler.LRScheduler, 
            test_metrics: MetricStoreCollection
            ) -> None:
        if isinstance(lr_scheduler, t.optim.lr_scheduler.ReduceLROnPlateau):
            accuracy_metric = 0
            for metric in self.training_args.get("scheduler_val_metric", ["val/accuracy"]):
                try:
                    accuracy_metric += test_metrics.to_dict()[metric]
                except KeyError:
                    raise ValueError(
                        f"val_metric {metric} not found in test_metrics {test_metrics}"
                    )
            try:
                lr_scheduler.step(accuracy_metric)
                return
            except Exception as e:
                raise ValueError(
                    f"WARNING: Could not step lr_scheduler {lr_scheduler} with exception {e}"
                )
        try:
            lr_scheduler.step()
        except Exception as e:
            print(
                f"WARNING: Could not step lr_scheduler {lr_scheduler} with exception {e}"
            )

    def train(
        self,
        train_set: IITDataset,
        test_set: IITDataset,
        epochs: int = 1000,
        use_wandb: bool = False,
        wandb_name_suffix: str = "",
    ) -> None:
        training_args = self.training_args
        print(f"{training_args=}")

        assert isinstance(train_set, IITDataset), ValueError(
            f"train_set is not an instance of IITDataset, but {type(train_set)}"
        )
        assert isinstance(test_set, IITDataset), ValueError(
            f"test_set is not an instance of IITDataset, but {type(test_set)}"
        )
        train_loader, test_loader = self.make_loaders(
            train_set,
            test_set,
            training_args["batch_size"],
            training_args["num_workers"],
        )

        early_stop = training_args["early_stop"]

        optimizer = training_args['optimizer_cls'](self.ll_model.parameters(), **training_args['optimizer_kwargs'])
        loss_fn = self.loss_fn
        scheduler_cls = training_args.get("lr_scheduler", None)
        scheduler_kwargs = training_args.get("scheduler_kwargs", {})
        if scheduler_cls == t.optim.lr_scheduler.ReduceLROnPlateau:
            mode = training_args.get("scheduler_mode", "max")
            if 'patience' not in scheduler_kwargs:
                scheduler_kwargs['patience'] = 10
            if 'factor' not in scheduler_kwargs:
                scheduler_kwargs['factor'] = 0.1
            lr_scheduler = scheduler_cls(optimizer, mode=mode, **scheduler_kwargs)
        elif scheduler_cls:
            lr_scheduler = scheduler_cls(optimizer, **scheduler_kwargs)

        if use_wandb and not wandb.run:
            wandb.init(project="iit", name=wandb_name_suffix, 
                       entity=WANDB_ENTITY)

        if use_wandb:
            wandb.config.update(training_args)
            wandb.config.update({"method": self.wandb_method})
            wandb.run.log_code() # type: ignore

    
        # Set seed before iterating on loaders for reproduceablility.
        t.manual_seed(training_args["seed"])
        with tqdm(range(epochs), desc="Training Epochs") as epoch_pbar:
            with tqdm(total=len(train_loader), desc="Training Batches") as batch_pbar:
                for epoch in range(epochs):
                    batch_pbar.reset()

                    train_metrics = self._run_train_epoch(train_loader, loss_fn, optimizer, batch_pbar)
                    test_metrics = self._run_eval_epoch(test_loader, loss_fn)
                    if scheduler_cls:
                        self.step_scheduler(lr_scheduler, test_metrics)
                    self.test_metrics = test_metrics
                    self.train_metrics = train_metrics
                    current_epoch_log = self._print_and_log_metrics(
                        epoch=epoch, 
                        metrics=MetricStoreCollection(train_metrics.metrics + test_metrics.metrics), 
                        optimizer=optimizer, 
                        use_wandb=use_wandb,
                    )

                    epoch_pbar.update(1)
                    epoch_pbar.set_postfix_str(current_epoch_log.strip(', '))
                    epoch_pbar.set_description(f"Epoch {epoch + 1}/{epochs}")

                    if early_stop and self._check_early_stop_condition(test_metrics):
                        break
                
                    self._run_epoch_extras(epoch_number=epoch+1)

        if use_wandb:
            wandb.log({"final epoch": epoch})

    #########################################
    # Immutable methods- might change later #
    #########################################
    @final
    @staticmethod
    def make_loaders(
        dataset: IITDataset,
        test_dataset: IITDataset,
        batch_size : int,
        num_workers : int,
    ) -> tuple[DataLoader, DataLoader]:
        loader = dataset.make_loader(batch_size, num_workers)
        test_loader = test_dataset.make_loader(batch_size, num_workers)
        return loader, test_loader

    @final
    def _run_train_epoch(
        self, 
        loader: DataLoader, 
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: t.optim.Optimizer,
        pbar: tqdm
        ) -> MetricStoreCollection:
        self.ll_model.train()
        train_metrics = self.make_train_metrics()
        for i, (base_input, ablation_input) in enumerate(loader):
            train_metrics.update(
                self.run_train_step(base_input, ablation_input, loss_fn, optimizer)
            )
            pbar.update(1)
        return train_metrics

    @final
    def _run_eval_epoch(
        self, 
        loader: DataLoader, 
        loss_fn: Callable[[Tensor, Tensor], Tensor]
        ) -> MetricStoreCollection:
        self.ll_model.eval()
        test_metrics = self.make_test_metrics()
        with t.no_grad():
            for i, (base_input, ablation_input) in enumerate(loader):
                test_metrics.update(
                    self.run_eval_step(base_input, ablation_input, loss_fn)
                )
        return test_metrics

    def _check_early_stop_condition(self, test_metrics: MetricStoreCollection) -> bool:
        """
        Returns True if all types of accuracy metrics reach 100%
        """
        got_accuracy_metric = False
        for metric in test_metrics:
            if metric.type == MetricType.ACCURACY:
                got_accuracy_metric = True
                val = metric.get_value()
                if isinstance(val, float) and val < 100:
                    return False
        if not got_accuracy_metric:
            raise ValueError("No accuracy metric found in test_metrics!")
        return True

    @final
    def _print_and_log_metrics(
        self,
        epoch: int, 
        metrics: MetricStoreCollection, 
        optimizer: t.optim.Optimizer,
        use_wandb: bool = False,
        print_metrics: bool = True,
        ) -> str:
        
        # Print the current epoch's metrics
        current_epoch_log = f"lr: {optimizer.param_groups[0]['lr']:.2e}, "
        for k in self.training_args.keys():
            if 'weight' in k and 'schedule' not in k:
                current_epoch_log += f"{k}: {self.training_args[k]:.2e}, "
        if use_wandb:
            wandb.log({"epoch": epoch})
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})
        
        for metric in metrics:
            if metric.type == MetricType.ACCURACY:
                current_epoch_log += f"{metric.get_name()}: {metric.get_value():.2f}, "
            else:
                current_epoch_log += f"{metric.get_name()}: {metric.get_value():.2e}, "
            if use_wandb:
                wandb.log({metric.get_name(): metric.get_value()})
        if print_metrics:
            tqdm.write(f'Epoch {epoch+1}: {current_epoch_log.strip(", ")}')
        
        return current_epoch_log

    def _run_epoch_extras(self, epoch_number: int) -> None:
        """ Optional method for running extra code at the end of each epoch """
        pass
