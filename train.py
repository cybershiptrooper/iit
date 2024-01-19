# %%
import dataclasses
from dataclasses import dataclass
import numpy as np
import torch as t
from torch import Tensor
from torch.utils.data import Dataset
import torchvision
import transformer_lens as tl
from transformer_lens.hook_points import HookedRootModule, HookPoint
import networkx as nx
from wrapper import HookedModuleWrapper
from typing import Callable, Optional
import wandb
from tqdm import tqdm

from pvr import mnist_pvr_train, mnist_pvr_test, MNIST_PVR_HL
from index import TorchIndex, Ix


# %%
"""
Things to write:
- (Ivan is writing TL representation of tracr models)
- Correspondence object, mapping hl variables to subspaces in the model
    - High-level causal structure object
        - HookedRootModule
        - Generate from NetworkX graph as used by tracr compiler intermediate step
        - Functions for computing thing from parent
    - Dictionary mapping graph nodes to TL units (HookPoint objects)
    - (Maybe for future) tau: LL values -> HL values
- Training loop
"""

HookName = str
HLCache = dict

@dataclass
class HLNode():
    name: HookName
    index: Optional[int]

@dataclass
class LLNode():
    name: HookName
    index: Optional[int]
    subspace: Optional[t.Tensor]=None

    def __eq__(self, other):
        return isinstance(other, LLNode) and dataclasses.astuple(self) == dataclasses.astuple(other)

    def __hash__(self):
        return hash(dataclasses.astuple(self))

class IITDataset(Dataset):
    """
    Each thing is randomly sampled from a pair of datasets.
    """
    def __init__(self, base_data, ablation_data, seed=0):
        # For vanilla IIT, base_data and ablation_data are the same
        self.base_data = base_data
        self.ablation_data = ablation_data
        self.seed = seed

    def __getitem__(self, index):
        # sample based on seed
        rng = np.random.default_rng(self.seed * 1000000 + index)
        base_index = rng.choice(len(self.base_data))
        ablation_index = rng.choice(len(self.ablation_data))

        base_input = self.base_data[base_index]
        ablation_input = self.ablation_data[ablation_index]
        return base_input, ablation_input

    def __len__(self):
        return len(self.base_data)


class IITModelPair():
    hl_model: HookedRootModule
    ll_model: HookedRootModule
    hl_cache: tl.ActivationCache
    ll_cache: tl.ActivationCache
    hl_graph: nx.DiGraph
    corr: dict[HookName, set[HookName]] # high -> low correspondence. Capital Pi in paper

    def __init__(self, hl_model:HookedRootModule=None, ll_model:HookedRootModule=None,
                 hl_graph=None, corr:dict[HLNode, set[LLNode]]={}, seed=0, training_args={}):
        # TODO change to construct hl_model from graph?
        if hl_model is None:
            assert hl_graph is not None
            hl_model = self.make_hl_model(hl_graph)

        self.hl_model = hl_model
        self.ll_model = ll_model

        self.corr:dict[HLNode, set[LLNode]] = corr
        assert all([k in self.hl_model.hook_dict for k in self.corr.keys()])
        self.rng = np.random.default_rng(seed)
        self.training_args = training_args

    def make_hl_model(self, hl_graph):
        raise NotImplementedError

    def set_corr(self, corr):
        self.corr = corr

    def sample_hl_name(self) -> str:
        # return a `str` rather than `numpy.str_`
        return str(self.rng.choice(list(self.corr.keys())))

    def hl_ablation_hook(self,hook_point_out:Tensor, hook:HookPoint) -> Tensor:
        out = self.hl_cache[hook.name]
        return out
    
    # TODO extend to position and subspace...
    def ll_ablation_hook(self,hook_point_out:Tensor, hook:HookPoint) -> Tensor:
        out = self.ll_cache[hook.name]
        return out

    def do_intervention(self, base_input, ablation_input, hl_node:HookName):
        ablation_x, ablation_y, ablation_intermediate_vars = ablation_input
        base_x, base_y, base_intermediate_vars = base_input
        hl_ablation_output, self.hl_cache = self.hl_model.run_with_cache(ablation_input)
        ll_ablation_output, self.ll_cache = self.ll_model.run_with_cache(ablation_x)

        ll_nodes = self.corr[hl_node]

        hl_output = self.hl_model.run_with_hooks(base_input, fwd_hooks=[(hl_node, self.hl_ablation_hook)])
        ll_output = self.ll_model.run_with_hooks(base_x, fwd_hooks=[(ll_node.name, self.ll_ablation_hook) for ll_node in ll_nodes])

        return hl_output, ll_output

    def train(self, base_data, ablation_data, epochs=1000, use_wandb=False):
        dataset = IITDataset(base_data, ablation_data)
        loader = t.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        optimizer = t.optim.Adam(self.ll_model.parameters(), lr=self.training_args['lr'])
        loss_fn = t.nn.CrossEntropyLoss()

        if use_wandb:
            raise NotImplementedError

        for epoch in range(epochs):
            losses = []
            for i, (base_input, ablation_input) in tqdm(enumerate(loader)):
                optimizer.zero_grad()
                self.hl_model.requires_grad_(False)
                self.ll_model.train()

                # sample a high-level variable to ablate
                hl_node = self.sample_hl_name()
                hl_output, ll_output = self.do_intervention(base_input, ablation_input, hl_node)
                loss = loss_fn(ll_output, hl_output)
                loss.backward()
                # print(f"{ll_output=}, {hl_output=}")
                losses.append(loss.item())
                optimizer.step()
            print(f"Epoch {epoch}: {np.mean(losses)}")
            if use_wandb:
                wandb.log({'loss': np.mean(losses), 'epoch': epoch})


# %%

hl_model = MNIST_PVR_HL()

resnet18 = torchvision.models.resnet18(pretrained=False) # 11M parameters
wrapped_r18 = HookedModuleWrapper(resnet18, name='resnet18', recursive=True, hook_self=False)

# %%

training_args = {
    'lr': 0.001
}

corr = {
    'hook_tl': {LLNode('mod.maxpool.hook_point', Ix[None, None, :28, :28])},
}


model_pair = IITModelPair(hl_model, ll_model=wrapped_r18, corr=corr, seed=0, training_args=training_args)
model_pair.train(mnist_pvr_train, mnist_pvr_train, epochs=10)

print(f"done training")
# %%

wrapped_r18(t.randn(1, 3, 56, 56)).shape
# %%
mnist_pvr_train[0]
# %%
