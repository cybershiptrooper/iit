from iit.utils.node_picker import get_all_nodes, get_params_in_circuit
from transformer_lens import HookedTransformer
from iit.utils.nodes import LLNode
from iit.utils.index import Ix
import iit.utils.index as index
from iit.utils.correspondence import Correspondence

def test_get_all_nodes():
    cfg = {
        "n_layers": 2,
        "n_heads": 4,
        "d_model": 8,
        "d_head": 2,
        "d_mlp": 16,
        "n_ctx": 16,
        "act_fn": "gelu",
        "d_vocab": 21
    }
    model = HookedTransformer(cfg)
    assert get_all_nodes(model) == [
        LLNode("blocks.0.attn.hook_result", Ix[:, :, 0, :]),
        LLNode("blocks.0.attn.hook_result", Ix[:, :, 1, :]),
        LLNode("blocks.0.attn.hook_result", Ix[:, :, 2, :]),
        LLNode("blocks.0.attn.hook_result", Ix[:, :, 3, :]),
        LLNode("blocks.0.mlp.hook_post", Ix[[None]]),
        LLNode("blocks.1.attn.hook_result", Ix[:, :, 0, :]),
        LLNode("blocks.1.attn.hook_result", Ix[:, :, 1, :]),
        LLNode("blocks.1.attn.hook_result", Ix[:, :, 2, :]),
        LLNode("blocks.1.attn.hook_result", Ix[:, :, 3, :]),
        LLNode("blocks.1.mlp.hook_post", Ix[[None]]),
    ]

def test_get_params_in_circuit():
    ll_cfg = {
        "n_layers": 2,
        "n_heads": 4,
        "d_head": 3,
        "d_model": 12,
        "d_mlp": 16,
        'act_fn': 'relu',
        'd_vocab': 6,
        'n_ctx': 5,
    }

    ll_model = HookedTransformer(ll_cfg)

    hl_ll_corr = {
        "blocks.0.mlp.hook_post": {
            LLNode(name='blocks.0.mlp.hook_post', index=index.Ix[[None]], subspace=None)
            },
        "blocks.1.attn.hook_result": {
            LLNode(name='blocks.1.attn.hook_result', index=index.Ix[:, :, :2, :], subspace=None)
            }
        }
    def lists_match(a, b):
        # check if two lists of LLNodes are equal, they may be in different order
        if len(a) != len(b):
            return False
        for node in a:
            if node not in b:
                return False
        for node in b:
            if node not in a:
                return False
        return True

    to_match_list = [
        LLNode(name='blocks.0.mlp.W_in', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.0.mlp.b_in', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.0.mlp.W_out', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.0.mlp.b_out', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.W_Q', index=Ix[:2], subspace=None),
        LLNode(name='blocks.1.attn.W_K', index=Ix[:2], subspace=None),
        LLNode(name='blocks.1.attn.W_V', index=Ix[:2], subspace=None),
        LLNode(name='blocks.1.attn.W_O', index=Ix[:2], subspace=None),
        LLNode(name='blocks.1.attn.b_Q', index=Ix[:2], subspace=None),
        LLNode(name='blocks.1.attn.b_K', index=Ix[:2], subspace=None),
        LLNode(name='blocks.1.attn.b_V', index=Ix[:2], subspace=None),
        LLNode(name='blocks.1.attn.b_O', index=Ix[[None]], subspace=None)
    ]

    assert lists_match(get_params_in_circuit(hl_ll_corr, ll_model), to_match_list)

    hl_ll_corr = {
        "blocks.0.mlp.hook_post": {
            LLNode(name='blocks.0.mlp.hook_post', index=index.Ix[[None]], subspace=None)
            },
        "blocks.1.attn.hook_result": {
            LLNode(name='blocks.1.attn.hook_result', index=index.Ix[[None]], subspace=None)
            }
        }
    
    to_match_list = [
        LLNode(name='blocks.0.mlp.W_in', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.0.mlp.b_in', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.0.mlp.W_out', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.0.mlp.b_out', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.W_Q', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.W_K', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.W_V', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.W_O', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.b_Q', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.b_K', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.b_V', index=Ix[[None]], subspace=None),
        LLNode(name='blocks.1.attn.b_O', index=Ix[[None]], subspace=None)
    ]

    assert lists_match(get_params_in_circuit(hl_ll_corr, ll_model), to_match_list)

def test_suffix_maker():
    n_layers = 6
    all_attns = [LLNode(f"blocks.{i}.hook_attn_out", index=None) for i in range(n_layers)]
    all_mlps = [LLNode(f"blocks.{i}.mlp.hook_post", index=None) for i in range(n_layers)]

    corr_dict = {
        "all_nodes_hook": [*all_mlps[:2], *all_attns[:4]]
    }

    hook_suffixes = Correspondence.get_hook_suffix(corr_dict)
    assert hook_suffixes == {
        "attn": "hook_attn_out",
        "mlp": "mlp.hook_post"
    }

    all_attns = [(f"blocks.{i}.attn.hook_result", Ix[[None]], None) for i in range(n_layers)]
    all_mlps = [(f"blocks.{i}.mlp.hook_post", Ix[[None]], None) for i in range(n_layers)]

    corr_dict = {
        "all_nodes_hook": [all_mlps[3], all_attns[0]]
    }

    corr = Correspondence.make_corr_from_dict(corr_dict, make_suffixes_from_corr=True)
    hook_suffixes = corr.get_suffixes()
    assert hook_suffixes == {
        "attn": "attn.hook_result",
        "mlp": "mlp.hook_post"
    }
