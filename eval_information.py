from torch import nn

from iit.model_pairs.probed_sequential_pair import IITProbeSequentialPair
from iit.utils.wrapper import get_hook_points
from iit.utils.probes import train_probes_on_model_pair, evaluate_probe
from iit.tasks.task_loader import get_alignment, get_dataset
import torch as t
from tqdm import tqdm
from iit.utils.plotter import plot_probe_stats
import os
import wandb
from datetime import datetime


def evaluate_model_on_probes(
    ll_model: t.nn.Module,
    task: str,
    probe_training_args: dict,
    train_set: t.utils.data.Dataset,
    test_set: t.utils.data.Dataset,
    use_wandb: bool = False,
    verbose: bool = False,
    save_probes: bool = False,
):
    print("reached evaluate_model!")
    probe_stats_per_layer = {}
    log_stats_per_layer = {}
    if use_wandb:
        wandb.init(project="iit")
        wandb.run.name = f"{task}_probes"
        wandb.run.save()
        # add training args to wandb config
        wandb.config.update(probe_training_args)

    for hook_point in tqdm(get_hook_points(ll_model), desc="Hook points"):
        _, hl_model, corr = get_alignment(
            task,
            config={
                "hook_point": hook_point,
                "input_shape": test_set.get_input_shape(),
            },
        )
        model_pair = IITProbeSequentialPair(
            ll_model=ll_model,
            hl_model=hl_model,
            corr=corr,
            training_args=probe_training_args,
        )

        input_shape = train_set.get_input_shape()
        trainer_out = train_probes_on_model_pair(
            model_pair, input_shape, train_set, probe_training_args
        )
        # get everything but probes from trainer_out
        log_stats_per_layer[hook_point] = {
            k: v for k, v in trainer_out.items() if k != "probes"
        }
        probe_stats_per_layer[hook_point] = trainer_out
        # save probe model
        if save_probes:
            for k, v in trainer_out["probes"].items():
                if not os.path.exists(f"weights/probes/{task}/{hook_point}"):
                    os.makedirs(f"weights/probes/{task}/{hook_point}")
                t.save(v, f"weights/probes/{task}/{hook_point}/{k}.pt")

        if use_wandb:
            wandb.log({"accuracy": trainer_out["accuracy"]})
            wandb.log({"loss": trainer_out["loss"]})

        # find test accuracy
        evals_out = evaluate_probe(
            trainer_out["probes"], model_pair, test_set, nn.CrossEntropyLoss()
        )
        if verbose:
            print(f"hook_point: {hook_point}")
            print(f"accuracy: {trainer_out['accuracy']}")
            print(f"loss: {trainer_out['loss']}")
            print(f"evals_out: {evals_out}")
        probe_stats_per_layer[hook_point].update(evals_out)
        if use_wandb:
            wandb.log(log_stats_per_layer[hook_point])

    if verbose:
        print(f"probe_stats_per_layer: {probe_stats_per_layer}")
    return probe_stats_per_layer


if __name__ == "__main__":
    task = "mnist_pvr"
    leaky_task = "pvr_leaky"
    training_args = {"batch_size": 512, "lr": 0.001, "num_workers": 0, "epochs": 1}
    save_weights = False
    probe_training_args = {
        "batch_size": 1024,
        "lr": 0.001,
        "num_workers": 0,
        "epochs": 1,
    }
    use_wandb = False
    reduction = "max"
    #####################################
    train_set, test_set = get_dataset(task, dataset_config={})
    ll_model, hl_model, corr = get_alignment(
        task, config={"input_shape": test_set.get_input_shape()}
    )
    model_pair = IITProbeSequentialPair(
        ll_model=ll_model, hl_model=hl_model, corr=corr, training_args=training_args
    )
    model_pair.train(
        train_set,
        test_set,
        epochs=training_args["epochs"],
        use_wandb=use_wandb,
    )
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
    print("evaluating on correctness")
    correctness_stats_per_layer = evaluate_model_on_probes(
        ll_model=ll_model,
        task=task,
        train_set=train_set.base_data,
        test_set=test_set.base_data,
        probe_training_args=probe_training_args,
        use_wandb=use_wandb,
        save_probes=save_weights,
    )
    print("evaluating on leakyness")
    leaky_stats_per_layer = evaluate_model_on_probes(
        ll_model=ll_model,
        task=leaky_task,
        train_set=leaky_train_set.base_data,
        test_set=leaky_test_set.base_data,
        probe_training_args=probe_training_args,
        use_wandb=use_wandb,
        save_probes=save_weights,
    )
    time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    plot_probe_stats(
        correctness_stats_per_layer,
        leaky_stats_per_layer,
        reduction=reduction,
        prefix=f"{time}",
        use_wandb=use_wandb,
    )
    print("done evaluating\n------------------------------------")
