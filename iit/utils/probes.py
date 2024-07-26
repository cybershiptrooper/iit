from typing import Callable

import torch as t
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from iit.model_pairs.base_model_pair import HLNode, LLNode, BaseModelPair
from iit.utils.config import DEVICE


def construct_probe(
    high_level_node: HLNode,
    ll_nodes: set[LLNode],
    dummy_cache: dict[str, t.Tensor],
    bias=False,
) -> nn.Linear:
    """
    Makes a probe for a given high-level node, given the low-level model and nodes.
    """
    if len(ll_nodes) > 1:
        raise NotImplementedError  # raising as unsure about summing over multiple nodes
    _get_hook_out_size = (
        lambda dummy_cache, ll_node: dummy_cache[ll_node.name][ll_node.index.as_index]
        .flatten()
        .shape[0]
    )
    size = sum(
        [_get_hook_out_size(dummy_cache, ll_node) for ll_node in ll_nodes]
    )  # assuming everything is flattened
    return nn.Linear(size, high_level_node.num_classes, bias=bias).to(DEVICE)


def construct_probes(model_pair: BaseModelPair, input_shape: tuple[int], bias: bool = False) -> dict[HLNode, nn.Linear]:
    probes = {}
    _, dummy_cache = model_pair.ll_model.run_with_cache(
        t.zeros(input_shape).to(DEVICE)
    )
    for hl_node, ll_nodes in model_pair.corr.items():
        probe = construct_probe(hl_node, ll_nodes, dummy_cache, bias=bias)
        probes[hl_node] = probe
    return probes


def train_probes_on_model_pair(
    model_pair: BaseModelPair,
    input_shape: str,
    train_set: t.utils.data.Dataset,
    training_args: dict,
) -> dict[str, dict]:
    probes = construct_probes(model_pair, input_shape=input_shape)
    params = []
    for p in probes.values():
        p.train()
        params += list(p.parameters())

    probe_optimizer = t.optim.Adam(params, lr=training_args["lr"])
    criterion = nn.CrossEntropyLoss()
    probe_losses = {k: [] for k in probes.keys()}
    probe_accuracies = {k: [] for k in probes.keys()}
    loader = t.utils.data.DataLoader(
        train_set,
        batch_size=training_args["batch_size"],
        shuffle=True,
        num_workers=training_args["num_workers"],
    )
    for _ in tqdm(range(training_args["epochs"])):
        probe_accuracy_run = {k: 0 for k in probes.keys()}
        probe_loss_run = {k: 0 for k in probes.keys()}
        for x, y, int_vars in loader:
            probe_optimizer.zero_grad()
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            out, cache = model_pair.ll_model.run_with_cache(x)
            probe_loss = 0
            for hl_node_name, probe in probes.items():
                ll_nodes = model_pair.corr[hl_node_name]
                gt = model_pair.hl_model.get_idx_to_intermediate(hl_node_name)(
                    int_vars
                ).to(DEVICE)
                if len(ll_nodes) > 1:
                    raise NotImplementedError
                for ll_node in ll_nodes:
                    probe_in_shape = probe.weight.shape[1:]
                    probe_out = probes[hl_node_name](
                        cache[ll_node.name][ll_node.index.as_index].reshape(
                            -1, *probe_in_shape
                        )
                    )
                    probe_loss += criterion(probe_out, gt)
                    probe_loss_run[hl_node_name] += probe_loss.item()
                    probe_accuracy_run[hl_node_name] += (
                        (probe_out.argmax(1) == gt).float().mean().item()
                    )
            probe_loss.backward()
            probe_optimizer.step()
        for k in probe_losses.keys():
            probe_losses[k].append(probe_loss_run[k] / len(loader))
            probe_accuracies[k].append(probe_accuracy_run[k] / len(loader))
    return {"probes": probes, "loss": probe_losses, "accuracy": probe_accuracies}


def evaluate_probe(
        probes: dict[str, nn.Linear],
        model_pair: BaseModelPair, 
        test_set: t.utils.data.Dataset, 
        criterion: Callable[[Tensor, Tensor], Tensor]
        ) -> dict[str, dict]:
    probe_stats = {}
    probe_stats["test loss"] = {}
    probe_stats["test accuracy"] = {}
    for hl_node_name, probe in tqdm(probes.items(), desc="Evaluating probes"):
        probe.eval()
        probe_loss = 0
        probe_accuracy = 0
        loader = t.utils.data.DataLoader(
            test_set, batch_size=256, shuffle=True, num_workers=0
        )
        with t.no_grad():
            for x, y, int_vars in loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                _, cache = model_pair.ll_model.run_with_cache(x)
                ll_nodes = model_pair.corr[hl_node_name]
                gt = model_pair.hl_model.get_idx_to_intermediate(hl_node_name)(
                    int_vars
                ).to(DEVICE)
                if len(ll_nodes) > 1:
                    raise NotImplementedError
                for ll_node in ll_nodes:
                    probe_in_shape = probe.weight.shape[1:]
                    probe_out = probes[hl_node_name](
                        cache[ll_node.name][ll_node.index.as_index].reshape(
                            -1, *probe_in_shape
                        )
                    )
                    probe_loss += criterion(probe_out, gt)
                    probe_accuracy += (probe_out.argmax(1) == gt).float().mean().item()
        probe_stats["test loss"][hl_node_name] = probe_loss.item() / len(loader)
        probe_stats["test accuracy"][hl_node_name] = probe_accuracy / len(loader)
    return probe_stats
