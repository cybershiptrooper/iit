from .utils import make_ioi_dataset_and_hl
from .ioi_config import NAMES
from .ioi_hl import IOI_HL
from .ioi_dataset_tl import IOIDataset, IOIDatasetWrapper
from iit.model_pairs.base_model_pair import HLNode, LLNode
from iit.utils.correspondence import Correspondence
from iit.utils.index import Ix

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


def make_corr_dict(include_mlp=False, eval=False, use_pos_embed=False):
    all_attns = [f"blocks.{i}.attn.hook_z" for i in range(ioi_cfg["n_layers"])]
    all_mlps = [f"blocks.{i}.mlp.hook_post" for i in range(ioi_cfg["n_layers"])]
    if eval:
        all_nodes_hook = "blocks.0.hook_resid_pre" if not use_pos_embed else "blocks.0.hook_pos_embed"
        return {
            "hook_duplicate": [all_attns[0]],
            # "hook_previous": ["blocks.1.attn.hook_result"],
            "hook_s_inhibition": [all_attns[2]],
            "hook_name_mover": [all_attns[4]],
            "all_nodes_hook": (
                [all_nodes_hook, all_mlps[0]]
                if include_mlp
                else [all_nodes_hook]
            ),
            "hook_out": [f"blocks.{n_layers-1}.hook_resid_post"],
        }
    ans = {
        "hook_duplicate": [all_attns[0]],
        # "hook_previous": ["blocks.1.attn.hook_result"],
        "hook_s_inhibition": [all_attns[2]],
        "hook_name_mover": [all_attns[4]],
    }
    if include_mlp:
        ans["all_nodes_hook"] = [all_mlps[0]]
    return ans
    


corr_dict = make_corr_dict(include_mlp=False)

edges = [
    ("all_nodes_hook", "hook_duplicate"),
    ("all_nodes_hook", "hook_s_inhibition"),
    ("all_nodes_hook", "hook_name_mover"),
    ("hook_duplicate", "hook_s_inhibition"),
    ("hook_s_inhibition", "hook_name_mover"),
    ("hook_name_mover", "hook_out"),
]

suffixes = {
    "attn": "attn.hook_z",
    "mlp": "mlp.hook_post",
}


corr = Correspondence.make_corr_from_dict(
    corr_dict, suffixes=suffixes, make_suffixes_from_corr=False
)


def make_ll_edges(corr: Correspondence):
    def expand_nodes(ll_node: LLNode):
        ll_nodes_expanded = []
        for head_index in range(n_heads):
            idx = Ix[:, :, head_index, :]
            if ll_node.index.intersects(idx):
                new_node = LLNode(ll_node.name, idx)
                ll_nodes_expanded.append(new_node)
        return ll_nodes_expanded

    ll_edges = []
    for edge in edges:
        hl_node_from = HLNode(edge[0], -1)
        hl_node_to = HLNode(edge[1], -1)
        ll_nodes_from = corr[hl_node_from]  # set of LLNodes
        ll_nodes_to = corr[hl_node_to]
        additional_from_nodes = set()
        remove_from_nodes = set()
        for ll_node_from in ll_nodes_from:
            if "attn" in ll_node_from.name:
                ll_nodes_from_expanded = expand_nodes(ll_node_from)
                # remove the original node
                remove_from_nodes.add(ll_node_from)
                additional_from_nodes = additional_from_nodes | set(
                    ll_nodes_from_expanded
                )
        ll_nodes_from = ll_nodes_from | additional_from_nodes
        ll_nodes_from = ll_nodes_from - remove_from_nodes

        additional_to_nodes = set()
        remove_to_nodes = set()
        for ll_node_to in ll_nodes_to:
            if "attn" in ll_node_to.name:
                ll_nodes_to_expanded = expand_nodes(ll_node_to)
                # remove the original node
                remove_to_nodes.add(ll_node_to)
                additional_to_nodes = additional_to_nodes | set(ll_nodes_to_expanded)
        ll_nodes_to = ll_nodes_to | additional_to_nodes
        ll_nodes_to = ll_nodes_to - remove_to_nodes

        for ll_node_from in ll_nodes_from:
            for ll_node_to in ll_nodes_to:
                ll_edges.append((ll_node_from, ll_node_to))
    return ll_edges
