from dataclasses import dataclass

import torch as t

@dataclass
class IOIArgParseNamespace:
    # global
    output_dir: str = "./results"
    include_mlp: bool = False
    use_wandb: bool = False
    num_samples: int = 18000
    device: str = "cuda" if t.cuda.is_available() else "cpu"
    batch_size: int = 512
    next_token: bool = False

    # eval 
    weights: str = "100_100_40"
    mean: bool = True
    load_from_wandb: bool = False

    # train
    epochs: int = 1000
    lr: float = 1e-3
    iit: float = 1.0 # iit loss weight
    b: float = 1.0 # baseline loss weight
    s: float = 0.4 # siit loss weight
    clip_grad_norm: float = 1.0
    use_single_loss: bool = False
    save_to_wandb: bool = False
   