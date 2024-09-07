from transformer_lens import HookedTransformer

from iit.model_pairs.ioi_model_pair import IOI_ModelPair
from iit.utils.iit_dataset import train_test_split
from iit.model_pairs.base_model_pair import *
from iit.utils.metric import *
from iit.tasks.ioi import (
    NAMES,
    make_ioi_dataset_and_hl,
    make_corr_dict,
    ioi_cfg,
    suffixes
)
from iit.utils.correspondence import Correspondence
from iit.utils.argparsing import IOIArgParseNamespace


def train_ioi(args: IOIArgParseNamespace) -> IOI_ModelPair:
    device = t.device(args.device)
    num_samples = args.num_samples
    epochs = args.epochs
    use_wandb = args.use_wandb

    training_args = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "iit_weight": args.iit,
        "behavior_weight": args.b,
        "strict_weight": args.s,
        "next_token": args.next_token,
        "lr_scheduler": None,
        "clip_grad_norm": args.clip_grad_norm,
        "early_stop": True,
        "use_single_loss": args.use_single_loss,
    }
    t.manual_seed(0)
    np.random.seed(0)

    ll_cfg = HookedTransformer.from_pretrained(
        "gpt2"
    ).cfg.to_dict()
    ll_cfg.update(ioi_cfg)

    ll_cfg["init_weights"] = True
    ll_model = HookedTransformer(ll_cfg).to(device)
    print("making ioi dataset and hl")
    ioi_dataset, hl_model = make_ioi_dataset_and_hl(
        num_samples, ll_model, device=device, verbose=True
    )
    print("making IIT dataset")
    train_ioi_dataset, test_ioi_dataset = train_test_split(
        ioi_dataset, test_size=0.2, random_state=42
    )
    train_set = IITDataset(train_ioi_dataset, train_ioi_dataset, seed=0)
    test_set = IITDataset(test_ioi_dataset, test_ioi_dataset, seed=0)
    print("making ioi model pair")
    corr_dict = make_corr_dict(include_mlp=args.include_mlp)
    corr = Correspondence.make_corr_from_dict(corr_dict, suffixes=suffixes)
    model_pair = IOI_ModelPair(
        ll_model=ll_model,
        hl_model=hl_model,
        corr=corr,
        training_args=training_args,
    )
    print("training ioi model pair")
    model_pair.train(train_set, test_set, epochs=epochs, use_wandb=use_wandb)
    print(f"done training")

    if use_wandb:
        wandb.finish()
    return model_pair
