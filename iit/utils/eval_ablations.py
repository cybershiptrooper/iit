import dataframe_image as dfi
import os
import pandas as pd
from enum import Enum
from tqdm import tqdm
from transformer_lens.HookedTransformer import HookPoint
from typing import Dict, List, Literal

from iit.model_pairs.base_model_pair import BaseModelPair
from iit.model_pairs.iit_model_pair import IITModelPair
from iit.utils.eval_metrics import *
from iit.utils.iit_dataset import IITDataset
from iit.utils.node_picker import *


class Categorical_Metric(Enum):
    ACCURACY = "accuracy"
    KL = "kl_div"
    KL_SELF = "kl_div_self"


def do_intervention(
    model_pair: BaseModelPair,
    base_input,
    ablation_input,
    node: LLNode,
    hooker: callable,
):
    _, cache = model_pair.ll_model.run_with_cache(ablation_input)
    model_pair.ll_cache = cache  # TODO: make this better when converting to script
    out = model_pair.ll_model.run_with_hooks(
        base_input, fwd_hooks=[(node.name, hooker)]
    )
    return out


def resample_ablate_node(
    model_pair: IITModelPair,
    base_in: tuple[t.Tensor, t.Tensor, t.Tensor],
    ablation_in: tuple[t.Tensor, t.Tensor, t.Tensor],
    node: LLNode,
    hooker: callable,
    atol=5e-2,
    verbose=False,
    categorical_metric: Categorical_Metric = Categorical_Metric.ACCURACY,
):  # TODO: change name to reflect that it's not just for resampling
    base_x, base_y = base_in[0:2]
    ablation_x, ablation_y = ablation_in[0:2]
    ll_out = do_intervention(model_pair, base_x, ablation_x, node, hooker)
    base_ll_out = model_pair.ll_model(base_x).squeeze()  # not used for result
    base_hl_out = model_pair.hl_model(base_in).squeeze()
    if verbose:
        print(node)

    if model_pair.hl_model.is_categorical():
        label_idx = model_pair.get_label_idxs()
        base_label = t.argmax(base_y, dim=-1)[label_idx.as_index]
        ablation_label = t.argmax(ablation_y, dim=-1)[label_idx.as_index]
        label_unchanged = base_label == ablation_label

        if categorical_metric == Categorical_Metric.KL:
            kl = kl_div(ll_out, base_hl_out, label_idx)
            corrupted_output = model_pair.ll_model(ablation_in[0]).squeeze()
            kl_div_clean = kl_div(base_ll_out, base_hl_out, label_idx)
            kl_div_corrupted = kl_div(corrupted_output, base_hl_out, label_idx)
            # normalize by the kl divergence of the corrupted output
            kl = (kl - kl_div_clean) / (kl_div_corrupted - kl_div_clean + 1e-12)
            result = (kl * (~label_unchanged).float()).sum().item() / (
                (~label_unchanged).float().sum().item() + 1e-12
            )

            if verbose:
                kl_old_vs_new = kl_div(ll_out, base_ll_out, label_idx)
                print("kl base_hl vs ll_out: ", kl.mean().item())
                print("kl base_ll vs ll_out: ", kl_old_vs_new.mean().item())
                # check if ll_out and base_ll_out are the same
                print(
                    "ll_out == base_ll_out:",
                    t.isclose(ll_out, base_ll_out, atol=atol).float().mean(),
                )
                print(
                    "fraction of labels changed:",
                    (~label_unchanged).float().mean(),
                )
                print()

        elif categorical_metric == Categorical_Metric.KL_SELF:
            kl = kl_div(ll_out, base_ll_out, label_idx)
            corrupted_output = model_pair.ll_model(ablation_x).squeeze()
            kl_div_corrupted = kl_div(corrupted_output, base_ll_out, label_idx)
            kl = kl / (
                kl_div_corrupted + 1e-12
            )  # normalize by the kl divergence of the corrupted output
            result = (kl * (~label_unchanged).float()).sum().item() / (
                (~label_unchanged).float().sum().item() + 1e-12
            )
        elif categorical_metric == Categorical_Metric.ACCURACY:
            # TODO: Move to a function
            # take argmax of everything
            ll_out = t.argmax(ll_out, dim=-1)[label_idx.as_index]
            base_hl_out = t.argmax(base_hl_out, dim=-1)[label_idx.as_index]
            base_ll_out = t.argmax(base_ll_out, dim=-1)[label_idx.as_index]

            # calculate metrics
            ll_unchanged = ll_out == base_label
            ll_out_unchanged = ll_out == base_ll_out  # not used for result
            accuracy = base_ll_out == base_hl_out  # not used for result
            changed_result = (~label_unchanged).cpu().float() * (
                ~ll_unchanged
            ).cpu().float()
            result = changed_result.sum().item() / (
                (~label_unchanged).float().sum().item() + 1e-12
            )

            if verbose:
                print(
                    "label: ",
                    (~label_unchanged).sum().item() / len(label_unchanged),
                )
                print("ll_vs_hl", (~ll_unchanged).sum().item() / len(ll_unchanged))
                print(
                    "ll_vs_ll",
                    (~ll_out_unchanged).sum().item() / len(ll_out_unchanged),
                )
                print("accuracy", accuracy.sum().item() / len(accuracy))
    else:
        label_unchanged = base_y == ablation_y
        ll_unchanged = t.isclose(
            ll_out.float().squeeze(),
            base_hl_out.float().to(ll_out.device).squeeze(),
            atol=atol,
        )
        label_unchanged = label_unchanged.reshape(ll_unchanged.shape)
        changed_result = (~label_unchanged).cpu().float() * (
            ~ll_unchanged
        ).cpu().float()
        result = changed_result.sum().item() / (
            (~label_unchanged).float().sum().item() + 1e-12
        )

        if verbose:
            print(
                "\nlabel changed:",
                (~label_unchanged).float().mean(),
                "\nouts_changed:",
                (~ll_unchanged).float().mean(),
                "\ndot product:",
                changed_result.mean(),
                "\ndifference:",
                (ll_out.float().squeeze() - base_y.float().to(ll_out.device)).mean(),
                "\nfinal:",
                result,
            )
    return result


def check_causal_effect(
    model_pair: BaseModelPair,
    dataset: IITDataset,
    batch_size: int = 256,
    node_type: Literal["a", "c", "n", "individual_c"] = "a",
    categorical_metric: Categorical_Metric = Categorical_Metric.ACCURACY,
    hook_maker: callable = None,
    verbose: bool = False,
):
    assert node_type in [
        "a",
        "c",
        "n",
        "individual_c",
    ], "type must be one of 'a', 'c', 'n', or 'individual_c'"
    hookers = {}
    results = {}
    all_nodes = (
        get_nodes_not_in_circuit(model_pair.ll_model, model_pair.corr)
        if node_type == "n"
        else (
            get_all_nodes(model_pair.ll_model)
            if node_type == "a"
            else (
                get_all_individual_nodes_in_circuit(
                    model_pair.ll_model, model_pair.corr
                )
                if node_type == "individual_c"
                else get_nodes_in_circuit(model_pair.corr)
            )
        )
    )

    for node in all_nodes:
        if hook_maker is not None:
            hookers[node] = hook_maker(node)
        else:
            hookers[node] = model_pair.make_ll_ablation_hook(node)
        results[node] = 0

    loader = dataset.make_loader(batch_size=batch_size, num_workers=0)
    for base_in, ablation_in in tqdm(loader):
        for node, hooker in hookers.items():
            result = resample_ablate_node(
                model_pair,
                base_in,
                ablation_in,
                node,
                hooker,
                categorical_metric=categorical_metric,
                verbose=verbose,
            )
            results[node] += result / len(loader)
    return results


def get_mean_cache(model, dataset: IITDataset, batch_size=8):
    loader = dataset.make_loader(batch_size=batch_size, num_workers=0)
    mean_cache = {}
    for batch in tqdm(loader):
        if isinstance(model, BaseModelPair):
            _, cache = model.ll_model.run_with_cache(batch[0])
        elif isinstance(model, HookedTransformer):
            _, cache = model.run_with_cache(batch[0])
        else:
            raise ValueError(
                f"model must be of type BaseModelPair or HookedTransformer, got {type(model)}"
            )
        for node, tensor in cache.items():
            if node not in mean_cache:
                mean_cache[node] = t.zeros_like(tensor[0].unsqueeze(0))
            mean_cache[node] += tensor.mean(dim=0).unsqueeze(0) / len(loader)
    return mean_cache


def make_ablation_hook(
    node: LLNode,
    mean_cache: dict[str, t.Tensor],
    use_mean_cache: bool = True,
) -> callable:
    if node.subspace is not None:
        raise NotImplementedError("Subspace not supported yet.")

    def zero_hook(hook_point_out: t.Tensor, hook: HookPoint) -> t.Tensor:
        hook_point_out[node.index.as_index] = 0
        return hook_point_out

    def mean_hook(hook_point_out: t.Tensor, hook: HookPoint) -> t.Tensor:
        cached_tensor = mean_cache[node.name]
        hook_point_out[node.index.as_index] = cached_tensor[node.index.as_index]
        return hook_point_out

    if use_mean_cache:
        return mean_hook
    return zero_hook


def ablate_nodes(
    model_pair: IITModelPair,
    base_input: tuple[t.Tensor, t.Tensor, t.Tensor],
    fwd_hooks: List[tuple[str, callable]],
    atol=5e-2,
    relative_change=True,
    verbose=False,
):
    """
    Returns 1 - accuracy of the model after ablating the nodes in fwd_hooks.
    Args:
        model_pair: IITModelPair
        base_input: input to the model
        fwd_hooks: list of tuples of (str, callable)
        atol: float (default: 5e-2)
        use_single_input: bool (default: False)
        The model takes a single input instead of a tuple of inputs
        relative_change: bool (default: True)
        If relative_change is True, the accuracy is normalized wrt to the accuracy of the model before ablation.
        i.e., we return 1 - accuracy(after ablation | accuracy(before ablation) = 1)
    """
    base_x = base_input[0]
    ll_out = model_pair.ll_model.run_with_hooks(base_x, fwd_hooks=fwd_hooks)

    if model_pair.hl_model.is_categorical():
        # TODO: add other metrics here
        base_hl_out = model_pair.hl_model(base_input).squeeze()
        base_ll_out = model_pair.ll_model(base_x).squeeze()
        label_idx = model_pair.get_label_idxs()
        ll_out = t.argmax(ll_out, dim=-1)[label_idx.as_index]
        base_hl_out = t.argmax(base_hl_out, dim=-1)[label_idx.as_index]
        base_ll_out = t.argmax(base_ll_out, dim=-1)[label_idx.as_index]
        ll_unchanged = (
            ll_out == base_hl_out
        )  # output of ll model is same as hl model after ablation
        accuracy = (
            base_ll_out == base_hl_out
        )  # output of ll model is same as hl model before ablation
        # calculate output output of ll model is different after ablation,
        # given that it was the same before ablation
        changed_result = (~ll_unchanged).cpu().float() * accuracy.cpu().float()
    else:
        base_hl_out = model_pair.hl_model(base_input).squeeze()
        base_ll_out = model_pair.ll_model(base_x).squeeze()
        ll_unchanged = t.isclose(
            ll_out.float().squeeze(),
            base_hl_out.float().to(ll_out.device),
            atol=atol,
        )
        accuracy = (
            t.isclose(base_ll_out.float(), base_hl_out.float(), atol=atol).cpu().float()
        )
        changed_result = (~ll_unchanged).cpu().float() * accuracy
    if relative_change:
        return changed_result.sum().item() / (accuracy.float().sum().item() + 1e-6)

    return (~ll_unchanged).cpu().float().mean()


def get_causal_effects_for_all_nodes(
    model_pair,
    uni_test_set,
    batch_size=256,
    use_mean_cache=True,
    categorical_metric=Categorical_Metric.ACCURACY,
    individual_nodes=True,
):
    mean_cache = None
    if use_mean_cache:
        mean_cache = get_mean_cache(model_pair, uni_test_set, batch_size=batch_size)
    za_result_not_in_circuit = check_causal_effect_on_ablation(
        model_pair,
        uni_test_set,
        node_type="n",
        verbose=False,
        mean_cache=mean_cache,
        categorical_metric=categorical_metric,
    )
    za_result_in_circuit = check_causal_effect_on_ablation(
        model_pair,
        uni_test_set,
        node_type="c" if not individual_nodes else "individual_c",
        verbose=False,
        mean_cache=mean_cache,
        categorical_metric=categorical_metric,
    )
    return za_result_not_in_circuit, za_result_in_circuit


def check_causal_effect_on_ablation(
    model_pair: BaseModelPair,
    dataset: IITDataset,
    batch_size: int = 256,
    node_type: str = "a",
    mean_cache: dict[str, t.Tensor] = None,
    categorical_metric: Categorical_Metric = Categorical_Metric.ACCURACY,
    verbose: bool = False,
):
    use_mean_cache = True if mean_cache else False
    assert node_type in [
        "a",
        "c",
        "n",
        "individual_c",
    ], "type must be one of 'a', 'c', 'n', or 'individual_c'"
    hookers = {}
    results = {}
    all_nodes = (
        get_nodes_not_in_circuit(model_pair.ll_model, model_pair.corr)
        if node_type == "n"
        else (
            get_all_nodes(model_pair.ll_model, model_pair.corr.get_suffixes())
            if node_type == "a"
            else (
                get_all_individual_nodes_in_circuit(
                    model_pair.ll_model, model_pair.corr
            )
                if node_type == "individual_c"
                else get_nodes_in_circuit(model_pair.corr)
            )
        )
    )

    for node in all_nodes:
        hookers[node] = make_ablation_hook(node, mean_cache, use_mean_cache)
        results[node] = 0

    loader = dataset.make_loader(batch_size=batch_size, num_workers=0)
    for base_in in tqdm(loader):
        for node, hooker in hookers.items():
            results[node] += ablate_nodes(model_pair, base_in, [(node.name, hooker)])

    for node, result in results.items():
        results[node] = result / len(loader)
    return results


def make_dataframe_of_results(result_not_in_circuit, result_in_circuit):
    def create_name(node):
        if "mlp" in node.name:
            return node.name
        if node.index is not None and node.index != index.Ix[[None]]:
            return f"{node.name}, head {str(node.index).split(',')[-2]}"
        else:
            return f"{node.name}, head [:]"

    df = pd.DataFrame(
        {
            "node": [create_name(node) for node in result_not_in_circuit.keys()]
            + [create_name(node) for node in result_in_circuit.keys()],
            "status": ["not_in_circuit"] * len(result_not_in_circuit)
            + ["in_circuit"] * len(result_in_circuit),
            "causal effect": list(result_not_in_circuit.values())
            + list(result_in_circuit.values()),
        }
    )
    df = df.sort_values("status", ascending=False)
    return df


def make_combined_dataframe_of_results(
    result_not_in_circuit,
    result_in_circuit,
    za_result_not_in_circuit,
    za_result_in_circuit,
    use_mean_cache: bool = False,
):
    df = make_dataframe_of_results(result_not_in_circuit, result_in_circuit)
    df2 = make_dataframe_of_results(za_result_not_in_circuit, za_result_in_circuit)
    df2_causal_effect = df2.pop("causal effect")
    # rename the columns
    df["resample_ablate_effect"] = df.pop("causal effect")
    if use_mean_cache:
        df["mean_ablate_effect"] = df2_causal_effect
    else:
        df["zero_ablate_effect"] = df2_causal_effect

    return df


def get_circuit_score(
    model_pair: BaseModelPair,
    dataset: IITDataset,
    nodes_to_ablate: List[LLNode],
    mean_cache: Dict[str, t.Tensor] = None,
    batch_size: int = 256,
    use_mean_cache: bool = False,
    relative_change: bool = True,
    verbose: bool = False,
):
    """
    Returns the accuracy of the model after ablating the nodes in nodes_to_ablate.
    Defaults to zero ablation.
    see ablate_nodes for more details
    """
    if use_mean_cache and mean_cache is None:
        mean_cache = get_mean_cache(model_pair, dataset, batch_size=batch_size)
    fwd_hooks = []

    for node in nodes_to_ablate:
        fwd_hooks.append(
            (node.name, make_ablation_hook(node, mean_cache, use_mean_cache))
        )
    loader = dataset.make_loader(batch_size=batch_size, num_workers=0)
    result = 0
    with torch.no_grad():
        for base_in in tqdm(loader):
            result += 1 - ablate_nodes(
                model_pair,
                base_in,
                fwd_hooks,
                verbose=verbose,
                relative_change=relative_change,
            )
    return result / len(loader)


def save_result(
    df: pd.DataFrame, save_dir: str, model_pair: BaseModelPair = None, suffix=""
):
    os.makedirs(save_dir, exist_ok=True)
    try:
        dfi.export(df, f"{save_dir}/results{suffix}.png")
    except Exception as e:
        print(f"Error exporting dataframe to image: {e}")
    df.to_csv(f"{save_dir}/results{suffix}.csv")
    print("Results saved to", save_dir)
    if model_pair is None:
        return
    training_args = model_pair.training_args
    with open(f"{save_dir}/train_args.log", "w") as f:
        f.write(str(training_args))
    print("Training args saved to", save_dir)
