from .utils import make_ioi_dataset_and_hl
from .ioi_config import NAMES
from .ioi_hl import IOI_HL
from .ioi_dataset_tl import IOIDataset, IOIDatasetWrapper
from iit.model_pairs.base_model_pair import HLNode, LLNode
from iit.utils.correspondence import Correspondence

n_layers = 6
n_heads = 4
d_model = 64
d_head = d_model // n_heads
ioi_cfg = {
    "n_layers": n_layers,
    "n_heads": n_heads,
    "d_model": d_model,
    "d_head": d_head,
}
all_attns = [f"blocks.{i}.attn.hook_z" for i in range(ioi_cfg["n_layers"])]
all_mlps = [f"blocks.{i}.mlp.hook_post" for i in range(ioi_cfg["n_layers"])]
corr_dict = {
    "hook_duplicate": [all_attns[0]],
    # "hook_previous": ["blocks.1.attn.hook_result"],
    "hook_s_inhibition": [all_attns[2]],
    "hook_name_mover": [all_attns[4]],
    # "all_nodes_hook": [*all_mlps[:2]],
}
# corr_dict = {
#     "all_nodes_hook": [*all_mlps[:2], *all_attns[:4]]
# }

suffixes = {
    "attn": "attn.hook_z",
    "mlp": "mlp.hook_post",
}

corr = Correspondence.make_corr_from_dict(corr_dict, suffixes=suffixes, make_suffixes_from_corr=False)
