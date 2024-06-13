import os
import json
import torch
import wandb
from iit.tasks.task_loader import get_default_corr


def save_model(model_pair, args, task):
    """
    Folder structure:
    -ll_models
        - task
            ll_model_{weights}.pth
            corr_{weights}.json
            - results_{weights}
                - metrics.log
                - ll_model_cfg.json
                - training_args.json
    """
    training_args = model_pair.training_args
    ll_model = model_pair.ll_model
    epochs = args.epochs

    # make save dirs
    next_token_str = "_next_token" if training_args["next_token"] else ""
    save_dir = os.path.join(args.output_dir, "ll_models", f"{task}{next_token_str}")
    model_suffix = f"{int(100*args.b)}_{int(100*args.iit)}_{int(100*args.s)}"
    results_dir = os.path.join(save_dir, f"results_{model_suffix}")
    os.makedirs(results_dir, exist_ok=True)

    # save model
    torch.save(ll_model.state_dict(), f"{save_dir}/ll_model_{model_suffix}.pth")

    # save training args
    training_args_file = os.path.join(results_dir, "training_args.json")
    with open(training_args_file, "w") as f:
        json.dump(training_args, f)

    # dump model cfg
    cfg = ll_model.cfg.to_dict()
    cfg_file = os.path.join(results_dir, "ll_model_cfg.json")
    with open(cfg_file, "w") as f:
        f.write(str(cfg))

    # log metrics
    metrics_file = os.path.join(results_dir, "metrics.log")
    with open(metrics_file, "w") as f:
        f.write(f"Epochs: {epochs}\n")
        early_stop_condition = model_pair._check_early_stop_condition(
            model_pair.test_metrics.metrics
        )
        f.write(f"Early stop: {early_stop_condition}\n")
        f.write("\n\n--------------------------------\n\n")
        f.write("Training metrics:\n")
        f.write(str(model_pair.train_metrics.metrics))
        f.write("\n\n--------------------------------\n\n")
        f.write("Test metrics:\n")
        f.write(str(model_pair.test_metrics.metrics))

    # save corr dict
    corr_file = os.path.join(save_dir, f"corr_{model_suffix}.json")
    with open(corr_file, "w") as f:
        json.dump(get_default_corr(task), f)

    if args.save_to_wandb:
        wandb.init(
            project="iit_models",
            group="without_mlp" if not args.include_mlp else "with_mlp",
            name=f"{task}{next_token_str}_{model_suffix}",
        )
        wandb.save(f"{save_dir}/ll_model_{model_suffix}.pth", base_path=args.output_dir)
        wandb.save(corr_file, base_path=args.output_dir)
        wandb.save(training_args_file, base_path=args.output_dir)


def load_files_from_wandb(
    task, weights, next_token, files_to_download, base_path, include_mlp=True
):
    api = wandb.Api()
    runs = api.runs("iit_models")
    next_token_str = "_next_token" if next_token else ""
    for run in runs:
        if (
            (task in run.name)
            and (weights in run.name)
            and (next_token_str in run.name)
            and (include_mlp == ("with_mlp" in run.group))
        ):
            files = run.files()
            for file in files:
                if any([file.name.endswith(f) for f in files_to_download]):
                    file.download(replace=True, root=base_path)
                    print(f"Downloaded {file.name}")
