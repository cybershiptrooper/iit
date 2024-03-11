# %%
"""
This is one file for now; will eventually split into multiple files.
"""

import torch as t
import transformer_lens
from iit.tasks.ioi.ioi_dataset_tl import IOIDataset, IOIDatasetWrapper
from iit.utils.iit_dataset import IITDataset, train_test_split
import iit.model_pairs as mp
from iit.tasks.ioi.ioi_hl import IOI_HL

from iit.model_pairs.base_model_pair import *
from iit.utils.metric import *
from iit.tasks.ioi.ioi_config import NAMES

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

training_args = {
    'batch_size': 128,
    'lr': 0.01,
    'num_workers': 0,
    'iit_weight': 0.0,
    'behavior_weight': 1.0,
}

ll_cfg = transformer_lens.HookedTransformer.from_pretrained("gpt2").cfg
ll_cfg.n_layers = 6
ll_cfg.n_heads = 4
ll_cfg.d_model = 64
ll_cfg.d_head = 64 // ll_cfg.n_heads

ll_cfg.init_weights = True
ll_model = transformer_lens.HookedTransformer(ll_cfg).to(DEVICE)

# TODO specify names, nouns, samples
ioi_dataset_tl = IOIDataset(
    num_samples=9000,
    tokenizer=ll_model.tokenizer,
    names=NAMES,
)

ioi_names = t.tensor(list(set([ioi_dataset_tl[i]['IO'].item() for i in range(len(ioi_dataset_tl))]))).to(DEVICE)
hl_model = IOI_HL(d_vocab=ll_model.cfg.d_vocab_out,
                  names=ioi_names).to(DEVICE)

ioi_dataset = IOIDatasetWrapper(
    num_samples=9000,
    tokenizer=ll_model.tokenizer,
    names=NAMES,
)

HookName = str
HLCache = dict

corr = {
    'hook_duplicate': {'blocks.0.attn.hook_result'},
    'hook_previous': {'blocks.1.attn.hook_result'},
    'hook_s_inhibition': {'blocks.2.attn.hook_result'},
    'hook_name_mover': {'blocks.3.attn.hook_result'},
}
corr = {HLNode(k, -1): {LLNode(name=name, index=None) for name in v} for k, v in corr.items()}

train_ioi_dataset, test_ioi_dataset = train_test_split(ioi_dataset, test_size=0.2, random_state=42)
train_set = IITDataset(train_ioi_dataset, train_ioi_dataset, seed=0)
test_set = IITDataset(test_ioi_dataset, test_ioi_dataset, seed=0)

model_pair = mp.IOI_ModelPair(ll_model=ll_model, hl_model=hl_model,
                          corr = corr,
                          training_args=training_args)
sentence = ioi_dataset_tl[0]['prompt']
detokenised = [ll_model.tokenizer.decode(i, clean_up_tokenization_spaces=True) for i in sentence]
print(sentence, detokenised)

model_pair.train(train_set, test_set, epochs=1000, use_wandb=False)

print(f"done training")
print([ll_model.tokenizer.decode(ioi_dataset_tl[i]['prompt']) for i in range(5)])
# %%