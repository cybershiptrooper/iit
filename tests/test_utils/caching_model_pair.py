from iit.model_pairs.strict_iit_model_pair import StrictIITModelPair

class CachingModelPair(StrictIITModelPair):
    """
    This is a util model pair that caches the activations of the ll_model while doing interventions.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ll_grad_cache = {}
    
    def do_intervention(
        self, base_input, ablation_input, hl_node, verbose=False
    ):
        ablation_x, ablation_y = ablation_input[0:2]
        base_x, base_y = base_input[0:2]

        hl_ablation_output, self.hl_cache = self.hl_model.run_with_cache(ablation_x)
        ll_ablation_output, self.ll_cache = self.ll_model.run_with_cache(ablation_x)

        ll_nodes = self.corr[hl_node]

        hl_output = self.hl_model.run_with_hooks(
            base_x, fwd_hooks=[(hl_node.name, self.make_hl_ablation_hook(hl_node))]
        )
        ll_fwd_hooks = [
                (ll_node.name, self.make_ll_ablation_hook(ll_node))
                for ll_node in ll_nodes
            ]
        self.ll_grad_cache, fwd_hooks, _ = self.ll_model.get_caching_hooks()
        ll_fwd_hooks += fwd_hooks
        ll_output = self.ll_model.run_with_hooks(
            base_x,
            fwd_hooks=ll_fwd_hooks
        )

        return hl_output, ll_output