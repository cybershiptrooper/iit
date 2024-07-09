import torch as t
from transformer_lens import HookedTransformer
from typing import Optional, Tuple
from transformer_lens.hook_points import NamesFilter, HookPoint, HookedRootModule
from transformer_lens.ActivationCache import ActivationCache


class LLModel:
    """
    A wrapper around a HookedRootModule that allows for retaining gradients while calling run_with_cache.
    """
    def __init__(self, 
                 model: HookedRootModule = None,
                 cfg: dict = None,
                 detach_while_caching=True):
        assert model is not None or cfg is not None, "Either model or cfg must be provided."
        if model is None:
            model = HookedTransformer(cfg=cfg)
        self.model = model
        self.detach_while_caching = detach_while_caching
    
    def get_caching_hooks(
        self,
        names_filter: NamesFilter = None,
        incl_bwd: bool = False,
        device=None,
        remove_batch_dim: bool = False,
        cache: Optional[dict] = None,
    ) -> Tuple[dict, list, list]:
        """Creates hooks to cache activations. Note: It does not add the hooks to the model.

        Args:
            names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
            incl_bwd (bool, optional): Whether to also do backwards hooks. Defaults to False.
            device (_type_, optional): The device to store on. Keeps on the same device as the layer if None.
            remove_batch_dim (bool, optional): Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
            cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

        Returns:
            cache (dict): The cache where activations will be stored.
            fwd_hooks (list): The forward hooks.
            bwd_hooks (list): The backward hooks. Empty if incl_bwd is False.
        """
        if cache is None:
            cache = {}

        if names_filter is None:
            names_filter = lambda name: True
        elif type(names_filter) == str:
            filter_str = names_filter
            names_filter = lambda name: name == filter_str
        elif type(names_filter) == list:
            filter_list = names_filter
            names_filter = lambda name: name in filter_list
        self.is_caching = True

        def save_hook(tensor: t.Tensor, hook: HookPoint):
            if self.detach_while_caching or (not (tensor.requires_grad and self.model.training)):
                # detach if the tensor requires grad and the model is not training
                tensor_to_cache = tensor.detach()
            else:
                # don't detach if the tensor requires grad and the model is training
                # retain grad if required for logs or further processing later (not memory efficient though)
                # Important Note: We are NOT cloning the tensor here. 
                # Autograd cannot track it back to the original tensor if we do that.
                # This is because we do not use the 'pointer' tensor_to_cache 
                # in any computation that autograd can track while resample ablating.
                tensor_to_cache = tensor
                tensor.retain_grad()

            if remove_batch_dim:
                cache[hook.name] = tensor_to_cache.to(device)[0]
            else:
                cache[hook.name] = tensor_to_cache.to(device)

        def save_hook_back(tensor, hook):
            # we always detach here as loss.backward() was already called 
            # and will throw an error if we don't do this
            tensor_to_cache = tensor.detach() 
            if remove_batch_dim:
                cache[hook.name + "_grad"] = tensor_to_cache.to(device)[0]
            else:
                cache[hook.name + "_grad"] = tensor_to_cache.to(device)

        fwd_hooks = []
        bwd_hooks = []
        for name, hp in self.hook_dict.items():
            if names_filter(name):
                fwd_hooks.append((name, save_hook))
                if incl_bwd:
                    bwd_hooks.append((name, save_hook_back))

        return cache, fwd_hooks, bwd_hooks
    
    @classmethod
    def make_from_hooked_transformer(cls, hooked_transformer: HookedTransformer, detach_while_caching):
        ll_model = cls(hooked_transformer, detach_while_caching=detach_while_caching)
        ll_model.load_state_dict(hooked_transformer.state_dict())
        return ll_model
    
    def run_with_cache(
        self,
        *model_args,
        names_filter: NamesFilter = None,
        device=None,
        remove_batch_dim=False,
        incl_bwd=False,
        reset_hooks_end=True,
        clear_contexts=False,
        **model_kwargs,
    ):
        """
        Runs the model and returns the model output and a Cache object.

        Args:
            *model_args: Positional arguments for the model.
            names_filter (NamesFilter, optional): A filter for which activations to cache. Accepts None, str,
                list of str, or a function that takes a string and returns a bool. Defaults to None, which
                means cache everything.
            device (str or torch.Device, optional): The device to cache activations on. Defaults to the
                model device. WARNING: Setting a different device than the one used by the model leads to
                significant performance degradation.
            remove_batch_dim (bool, optional): If True, removes the batch dimension when caching. Only
                makes sense with batch_size=1 inputs. Defaults to False.
            incl_bwd (bool, optional): If True, calls backward on the model output and caches gradients
                as well. Assumes that the model outputs a scalar (e.g., return_type="loss"). Custom loss
                functions are not supported. Defaults to False.
            reset_hooks_end (bool, optional): If True, removes all hooks added by this function at the
                end of the run. Defaults to True.
            clear_contexts (bool, optional): If True, clears hook contexts whenever hooks are reset.
                Defaults to False.
            **model_kwargs: Keyword arguments for the model.

        Returns:
            tuple: A tuple containing the model output and a Cache object.

        """
        cache_dict, fwd, bwd = self.get_caching_hooks(
            names_filter, incl_bwd, device, remove_batch_dim=remove_batch_dim
        )

        with self.model.hooks(
            fwd_hooks=fwd,
            bwd_hooks=bwd,
            reset_hooks_end=reset_hooks_end,
            clear_contexts=clear_contexts,
        ):
            model_out = self.model(*model_args, **model_kwargs)
            if incl_bwd:
                model_out.backward()
        cache_dict = ActivationCache(
                cache_dict, self, has_batch_dim=not remove_batch_dim
        )
        return model_out, cache_dict
    
    def __getattr__(self, name):
        if name == "run_with_cache":
            return self.run_with_cache
        elif name == "get_caching_hooks":
            return self.get_caching_hooks
        return getattr(self.model, name)
    
    def __call__(self, *args: t.Any, **kwds: t.Any) -> t.Any:
        return self.model(*args, **kwds)
    
    def __repr__(self):
        return self.model.__repr__()
    
    def __str__(self):
        return self.model.__str__()