import torch
import os
import transformer_lens
from iit.tasks.ioi import (
    make_ioi_dataset_and_hl,
    NAMES,
    ioi_cfg,
    corr,
    suffixes,
)
from iit.utils.correspondence import Correspondence
from iit.utils.eval_ablations import *
import numpy as np
from iit.utils.iit_dataset import IITDataset
from iit.utils.eval_datasets import IITUniqueDataset
import json
from iit.utils.io_scripts import load_files_from_wandb


def eval_ioi(args):
    weights = args.weights
    use_mean_cache = args.mean
    device = args.device
    save_dir = os.path.join(args.output_dir, f"ioi")
    results_dir = os.path.join(save_dir, f"results_{weights}")
    batch_size = args.batch_size
    num_samples = args.num_samples
    # load model
    ll_cfg = transformer_lens.HookedTransformer.from_pretrained("gpt2").cfg.to_dict()
    ll_cfg.update(ioi_cfg)

    ll_model = transformer_lens.HookedTransformer(ll_cfg).to(device)
    if args.load_from_wandb:
        load_files_from_wandb(
            "ioi",
            weights,
            args.next_token,
            [f"ll_model_{weights}.pth", f"corr_{weights}.json"],
            save_dir,
        )
    try:
        ll_model.load_state_dict(torch.load(f"{save_dir}/ll_model_{weights}.pth"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at {save_dir}")

    # load corr
    corr_file = os.path.join(save_dir, f"corr_{weights}.json")
    if os.path.exists(corr_file):
        with open(corr_file, "r") as f:
            corr_dict = json.load(f)
        corr = Correspondence.make_corr_from_dict(corr_dict, suffixes)
    else:
        print(f"WARNING: {corr_file} not found, using default corr_dict")

    # load dataset
    ioi_dataset, hl_model = make_ioi_dataset_and_hl(
        num_samples, ll_model, NAMES, verbose=True
    )

    model_pair = mp.IOI_ModelPair(ll_model=ll_model, hl_model=hl_model, corr=corr)

    test_set = IITDataset(ioi_dataset, ioi_dataset, seed=0)

    np.random.seed(0)
    t.manual_seed(0)
    result_not_in_circuit = check_causal_effect(
        model_pair, test_set, node_type="n", verbose=False
    )
    result_in_circuit = check_causal_effect(
        model_pair, test_set, node_type="c", verbose=False
    )

    metric_collection = model_pair._run_eval_epoch(
        test_set.make_loader(batch_size, 0), model_pair.loss_fn
    )

    # zero/mean ablation
    uni_test_set = IITUniqueDataset(ioi_dataset, ioi_dataset, seed=0)
    za_result_not_in_circuit, za_result_in_circuit = get_causal_effects_for_all_nodes(
        model_pair,
        uni_test_set,
        batch_size=batch_size,
        use_mean_cache=use_mean_cache,
    )

    df = make_combined_dataframe_of_results(
        result_not_in_circuit,
        result_in_circuit,
        za_result_not_in_circuit,
        za_result_in_circuit,
        use_mean_cache=use_mean_cache,
    )

    save_result(df, results_dir)
    with open(f"{results_dir}/metric_collection.log", "w") as f:
        f.write(str(metric_collection))
        print("Results saved at", save_dir)
        print(metric_collection)