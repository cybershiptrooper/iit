from typing import Callable

import wandb
import numpy as np
import torch as t
from torch import Tensor
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule

from iit.model_pairs.iit_model_pair import IITModelPair
from iit.utils.nodes import HLNode, LLNode
from iit.utils.config import WANDB_ENTITY
from iit.utils.probes import construct_probes
from iit.utils.correspondence import Correspondence
from iit.utils.iit_dataset import IITDataset
from iit.model_pairs.ll_model import LLModel


class IITProbeSequentialPair(IITModelPair):
    def __init__(
        self,
        hl_model: HookedRootModule,
        ll_model: HookedRootModule | LLModel,
        corr: Correspondence,
        training_args: dict = {},
    ):
        default_training_args = {
            "probe_weight": 1.0,
        }
        training_args = {**default_training_args, **training_args}
        super().__init__(hl_model, ll_model, corr, training_args)

    def run_train_step(
        self,
        base_input: tuple[Tensor, Tensor, Tensor],
        ablation_input: tuple[Tensor, Tensor, Tensor],
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        optimizer: t.optim.Optimizer,
        probes: dict,
        probe_optimizer: t.optim.Optimizer,
        training_args: dict,
    ) -> dict:
        ablation_loss = super().run_train_step(
            base_input, ablation_input, loss_fn, optimizer
        )["iit_loss"]
        # !!! Second forward pass
        # add probe losses and behavior loss
        probe_losses = []
        probe_optimizer.zero_grad()
        for p in probes.values():
            p.train()

        base_x, base_y, base_intermediate_vars = base_input
        out, cache = self.ll_model.run_with_cache(base_x)
        probe_loss = 0
        for hl_node_name in probes.keys():
            gt = self.hl_model.get_idx_to_intermediate(hl_node_name)(
                base_intermediate_vars
            )
            ll_nodes = self.corr[hl_node_name]
            if len(ll_nodes) > 1:
                raise NotImplementedError
            for ll_node in ll_nodes:
                probe_in_shape = probes[hl_node_name].weight.shape[1:]
                probe_out = probes[hl_node_name](
                    cache[ll_node.name][ll_node.index.as_index].reshape(
                        -1, *probe_in_shape
                    )
                )
                probe_loss += loss_fn(probe_out, gt)

        behavior_loss = loss_fn(out, base_y)
        loss = behavior_loss + training_args["probe_weight"] * probe_loss
        loss.backward()
        probe_optimizer.step()

        return {
            "ablation_loss": ablation_loss,
            "probe_loss": probe_loss.item(),
            "behavior_loss": behavior_loss.item(),
        }

    def train(
        self,
        dataset: IITDataset,
        test_dataset: IITDataset,
        epochs: int = 1000,
        use_wandb: bool = False,
    ) -> None:
        training_args = self.training_args
        print(f"{training_args=}")

        # add to make probes
        input_shape = (dataset[0][0][0]).unsqueeze(0).shape
        with t.no_grad():
            probes = construct_probes(self, input_shape)
            print("made probes", [(k, p.weight.shape) for k, p in probes.items()])

        loader, test_loader = self.make_loaders(dataset, test_dataset)
        params = list(self.ll_model.parameters())
        for p in probes.values():
            params += list(p.parameters())
        probe_optimizer = t.optim.Adam(params, lr=training_args["lr"])

        optimizer = t.optim.Adam(self.ll_model.parameters(), lr=training_args["lr"])
        loss_fn = t.nn.CrossEntropyLoss()

        if use_wandb and not wandb.run:
            wandb.init(project="iit", entity=WANDB_ENTITY)

        if use_wandb:
            wandb.config.update(training_args)
            wandb.config.update({"method": "IIT + Probes (Sequential)"})

        for epoch in tqdm(range(epochs)):
            iit_losses = []
            probe_losses = []
            behavior_losses = []
            self.ll_model.train()
            for i, (base_input, ablation_input) in tqdm(
                enumerate(loader), total=len(loader)
            ):
                train_losses = self.run_train_step(
                    base_input,
                    ablation_input,
                    loss_fn,
                    optimizer,
                    probes,
                    probe_optimizer,
                    training_args,
                )
                iit_losses.append(train_losses["ablation_loss"])
                probe_losses.append(train_losses["probe_loss"])
                behavior_losses.append(train_losses["behavior_loss"])

            # now calculate test loss
            test_losses = []
            accuracies = []
            test_probe_losses = []
            probe_accuracies = []
            test_behavior_losses = []
            behavior_accuracies = []

            self.ll_model.eval()
            for p in probes.values():
                p.eval()
            self.hl_model.requires_grad_(False)
            with t.no_grad():
                for i, (base_input, ablation_input) in enumerate(test_loader):
                    output = self.run_eval_step(
                        base_input, ablation_input, loss_fn
                    )
                    accuracies.append(output["accuracy"])
                    test_losses.append(output["iit_loss"])

                    # !!! Second forward pass
                    # add probe losses and accuracies
                    base_x, base_y, base_intermediate_vars = base_input
                    out, cache = self.ll_model.run_with_cache(base_x)
                    behavior_loss = loss_fn(out, base_y)
                    top1_behavior = t.argmax(out, dim=1)
                    behavior_accuracy = (top1_behavior == base_y).float().mean()
                    test_behavior_losses.append(behavior_loss.item())
                    behavior_accuracies.append(behavior_accuracy.item())

                    for hl_node_name in probes.keys():
                        gt = self.hl_model.get_idx_to_intermediate(hl_node_name)(
                            base_intermediate_vars
                        )
                        ll_nodes = self.corr[hl_node_name]
                        if len(ll_nodes) > 1:
                            raise NotImplementedError
                        for ll_node in ll_nodes:
                            probe_in_shape = probes[hl_node_name].weight.shape[1:]
                            probe_out = probes[hl_node_name](
                                cache[ll_node.name][ll_node.index.as_index].reshape(
                                    -1, *probe_in_shape
                                )
                            )
                            probe_loss += loss_fn(probe_out, gt)
                            top1 = t.argmax(probe_out, dim=1)
                            probe_accuracy = (top1 == gt).float().mean()
                    test_probe_losses.append(probe_loss.item() / len(probes))
                    probe_accuracies.append(probe_accuracy.item())

            print(
                f"Epoch {epoch}: {np.mean(iit_losses):.4f}, \n Test: {np.mean(test_losses):.4f}, {np.mean(accuracies)*100:.4f}%, \nProbe: {np.mean(probe_accuracies)*100:.4f}%, {np.mean(test_probe_losses):.4f}, \nBehavior: {np.mean(behavior_accuracies)*100:.4f}%, {np.mean(test_behavior_losses):.4f}"
            )

            if use_wandb:
                wandb.log(
                    {
                        "train IIT loss": np.mean(iit_losses),
                        "train probe loss": np.mean(probe_losses),
                        "train behavior loss": np.mean(behavior_losses),
                        "test loss": np.mean(test_losses),
                        "accuracy": np.mean(accuracies),
                        "epoch": epoch,
                        "probe loss": np.mean(test_probe_losses),
                        "probe accuracy": np.mean(probe_accuracies),
                        "behavior loss": behavior_loss.item(),
                        "behavior accuracy": behavior_accuracy.item(),
                    }
                )
