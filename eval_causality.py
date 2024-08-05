import os
from datetime import datetime

import torch as t
from torch import Tensor
import wandb
from tqdm import tqdm

from iit.model_pairs.probed_sequential_pair import IITProbeSequentialPair
from iit.tasks.task_loader import get_alignment, get_dataset
from iit.utils.config import DEVICE
from iit.utils.plotter import plot_ablation_stats
from iit.utils.wrapper import get_hook_points
from iit.tasks.mnist_pvr.dataset import ImagePVRDataset


def evaluate_model_on_ablations(
    ll_model: t.nn.Module,
    task: str,
    test_set: t.utils.data.Dataset,
    eval_args: dict,
    verbose: bool = False,
) -> dict:
    print("reached evaluate_model!")
    stats_per_layer = {}
    for hook_point in tqdm(get_hook_points(ll_model), desc="Hook points"):
        _, hl_model, corr = get_alignment(
            task,
            config={
                "hook_point": hook_point,
                "input_shape": test_set.get_input_shape(), # type: ignore
            },
        )
        model_pair = IITProbeSequentialPair(
            ll_model=ll_model, hl_model=hl_model, corr=corr
        )
        dataloader = t.utils.data.DataLoader(
            test_set,
            batch_size=eval_args["batch_size"],
            num_workers=eval_args["num_workers"],
        )
        # set up stats
        hookpoint_stats = {}
        for hl_node, _ in model_pair.corr.items():
            hookpoint_stats[hl_node] = t.zeros(1)
        # find test accuracy
        with t.no_grad():
            for base_input_lists in tqdm(dataloader, desc=f"Ablations on {hook_point}"):
                base_input: tuple[Tensor, Tensor, Tensor] = (x.to(DEVICE) for x in base_input_lists) # type: ignore
                for hl_node, ll_nodes in model_pair.corr.items():
                    if isinstance(test_set, ImagePVRDataset):
                        ablated_input_pre = test_set.patch_batch_at_hl(
                            list(base_input),
                            list(base_input_lists),
                            hl_node,
                        )
                        ablated_input = (
                            t.stack(ablated_input_pre[0]).to(DEVICE),  # input
                            t.stack(ablated_input_pre[1]).to(DEVICE),  # label
                            t.stack(ablated_input_pre[2]).to(DEVICE),
                        )  # intermediate_data
                    else:
                        raise ValueError(f"patch_batch_at_hl not implemented for this dataset type: {type(test_set)}")
                    
                    # unsqueeze if single element
                    if ablated_input[1].shape == ():
                        assert (
                            eval_args["batch_size"] == 1
                        ), "Logic error! If batch_size is not 1, then labels should not be a scalar"
                        ablated_input = (
                            ablated_input[0].unsqueeze(0),
                            ablated_input[1].unsqueeze(0),
                            ablated_input[2].unsqueeze(0),
                        )
                    hl_output, ll_output = model_pair.do_intervention(
                        base_input, ablated_input, hl_node
                    )
                    ablated_y = ablated_input[1]
                    base_y = base_input[1]
                    changed = (ablated_y != base_y).float()
                    # assert t.all(hl_output == hl_base_output), f"hl_output: {hl_output}; hl_base_output: {hl_base_output}"
                    # find accuracy
                    top1 = t.argmax(ll_output, dim=1)
                    accuracy = (top1 == hl_output).float()
                    accuracy = accuracy * changed
                    # clip to 0 if accuracy is less than unchanged
                    changed_len = changed.sum()
                    # mean accuracy
                    hookpoint_stats[hl_node] += accuracy.sum() / (changed_len + 1e-10)
        for k, v in hookpoint_stats.items():
            hookpoint_stats[k] = v / len(dataloader)
            assert (
                0 <= hookpoint_stats[k] <= 1
            ), f"hookpoint_stats[hl_node]: {hookpoint_stats[hl_node]}"
        stats_per_layer[hook_point] = hookpoint_stats
        # hookpoint_stats = {k: v / len(dataloader) for k, v in hookpoint_stats.items()}
        if verbose:
            print(f"hook_point: {hook_point}")
            print(f"hookpoint_stats: {hookpoint_stats}")

    if verbose:
        print(f"stats_per_layer: {stats_per_layer}")
    return stats_per_layer


if __name__ == "__main__":
    task = "mnist_pvr"
    leaky_task = "pvr_leaky"
    training_args = {"batch_size": 512, "lr": 0.001, "num_workers": 0, "epochs": 5}
    eval_args = {
        "batch_size": 1024,
        "num_workers": 0,
    }
    save_weights = False
    use_wandb = True
    verbose = False
    train = False
    #####################################
    train_set, test_set = get_dataset(task, dataset_config={})
    ll_model, hl_model, corr = get_alignment(
        task, config={"input_shape": test_set.get_input_shape()}
    )
    assert ll_model is not None
    assert hl_model is not None
    model_pair = IITProbeSequentialPair(
        ll_model=ll_model, hl_model=hl_model, corr=corr, training_args=training_args
    )
    if train:
        model_pair.train(
            train_set,
            test_set,
            epochs=int(training_args["epochs"]),
            use_wandb=use_wandb,
        )
    else:
        try:
            model = t.load(f"weights/ll_model/{task}.pt")
            model_pair.ll_model.load_state_dict(model)
        except:
            raise ValueError(f"Could not load model from weights/ll_model/{task}.pt")

    if use_wandb:
        wandb.finish()
    print(f"done training\n------------------------------------")
    print(f"evaluating model")
    leaky_train_set, leaky_test_set = get_dataset(leaky_task, dataset_config={})
    ll_model.eval()

    if save_weights:
        if not os.path.exists(f"weights/ll_model"):
            os.makedirs(f"weights/ll_model")
        t.save(ll_model.state_dict(), f"weights/ll_model/{task}.pt")
    print("evaluating on leakyness")

    if use_wandb:
        wandb.init(project="iit")
        wandb.run.name = f"{leaky_task}_ablation" # type: ignore
        wandb.run.save() # type: ignore
        wandb.config.update(eval_args)

    leaky_stats_per_layer = evaluate_model_on_ablations(
        ll_model=ll_model,
        task=leaky_task,
        eval_args=eval_args,
        test_set=leaky_test_set.base_data,
        verbose=verbose,
    )

    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    plot_ablation_stats(leaky_stats_per_layer, prefix=f"{time}", use_wandb=use_wandb)
    wandb.finish()
    print("done evaluating\n------------------------------------")
