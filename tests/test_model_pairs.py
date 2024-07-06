from tests.test_utils.caching_model_pair import CachingModelPair
from transformer_lens import HookedTransformer
from iit.utils.correspondence import Correspondence
from iit.model_pairs.nodes import HLNode, LLNode
import iit.utils.index as index
from iit.model_pairs.ll_model import LLModel
import torch

def test_model_pair_gradients():
    ll_model = LLModel(cfg={
        'n_layers': 4,
        'd_model': 32, 
        'n_ctx': 10,
        'd_head': 8,
        'act_fn': 'gelu',
        'd_vocab': 10,
    },
        detach_while_caching=False
    )

    hl_model = HookedTransformer(cfg={
        'n_layers': 4,
        'd_model': 32, 
        'n_ctx': 10,
        'd_head': 8,
        'act_fn': 'gelu',
        'd_vocab': 10,
    })

    corr = Correspondence()
    hook_point = 'blocks.1.attn.hook_z'
    hook_idx = index.Ix[:, :, 0]
    hook_idx_complement = index.Ix[:, :, 1:]
    prev_hooks = ['blocks.0.attn.hook_z', 'blocks.0.mlp.hook_post']
    next_hooks = ['blocks.2.attn.hook_z', 'blocks.3.attn.hook_z', 'blocks.1.mlp.hook_post', 'blocks.2.mlp.hook_post', 'blocks.3.mlp.hook_post']
    corr.update({
            HLNode(hook_point, -1, index=hook_idx) : [LLNode(hook_point, index=hook_idx)],
        }
    )

    model_pair = CachingModelPair(ll_model=ll_model, hl_model=hl_model, corr=corr)

    # model_pair = CachingModelPair(ll_model, hl_model, corr=corr)
    for n, p in model_pair.ll_model.named_parameters():
        assert p.requires_grad, f'grads not enabled for ll_model at {n}'
    
    hl_output, ll_output = model_pair.do_intervention(
        base_input=(torch.tensor([1, 2, 3]), None, None ), 
        ablation_input=(torch.tensor([1, 2, 3]), None, None ),
        hl_node=model_pair.sample_hl_name(),
        verbose=False
    )

    loss = model_pair.loss_fn(hl_output, ll_output)
    loss.backward()


    for prev_hook in prev_hooks:
        assert model_pair.ll_cache[prev_hook].grad is not None, f'{prev_hook} has no grad, but should'

    for next_hook in next_hooks:
        assert model_pair.ll_cache[next_hook].grad is None, f'{next_hook} has grad, but should not'
    model_pair.ll_cache[hook_point].grad[hook_idx.as_index] 
    assert (model_pair.ll_cache[hook_point].grad[hook_idx_complement.as_index] == 0).all()
    assert (model_pair.ll_grad_cache[hook_point].grad[hook_idx.as_index] != 0).all()