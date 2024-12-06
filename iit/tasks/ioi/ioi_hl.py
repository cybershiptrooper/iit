from argparse import Namespace
from typing import Callable

import torch as t
from torch import Tensor
from transformer_lens.hook_points import HookedRootModule, HookPoint

from iit.tasks.hl_model import HLModel


class DuplicateHead(t.nn.Module):
    def forward(self, tokens : Tensor) -> Tensor:
        # Write the last previous position of any duplicated token (used at S2)
        positions = (tokens[..., None, :] == tokens[..., :, None]) # batch seq1 seq2
        positions = t.triu(positions, diagonal=1) # only consider positions before this one
        indices = positions.nonzero(as_tuple=True)
        ret = t.full_like(tokens, -1)
        ret[indices[0], indices[2]] = indices[1]
        return ret
    
class PreviousHead(t.nn.Module):
    def forward(self, tokens: Tensor) -> Tensor:
        # copy token S1 to token S1+1 (used at S1+1)
        ret = t.full_like(tokens, -1)
        ret[..., 1:] = tokens[..., :-1]
        return ret

class InductionHead(t.nn.Module):
    """Induction heads omitted because they're redundant with duplicate heads in IOI"""
    

class SInhibitionHead(t.nn.Module):
    def forward(self, tokens: Tensor, duplicate: Tensor) -> Tensor:
        """
        when duplicate is not -1, 
        output a flag to the name mover head to NOT copy this name
        flag is -1 if no duplicate name here, and name token for the name to inhibit
        """
        ret = t.zeros_like(tokens)

        # if duplicate is -1, we don't inhibit
        ret[duplicate == -1] = -1

        # extract token positions we care about from duplicate
        duplicate_pos_at_duplicates = t.where(duplicate != -1)
        duplicate_pos_at_tokens = duplicate[duplicate_pos_at_duplicates[0], duplicate_pos_at_duplicates[1]]
        duplicate_pos_at_tokens_tup = (duplicate_pos_at_duplicates[0], duplicate_pos_at_tokens)
        duplicate_tokens = tokens[duplicate_pos_at_tokens_tup]
        assert ret[duplicate_pos_at_duplicates].abs().sum() == 0 # sanity check, to make sure we're not overwriting anything
        # replace ret with the duplicated tokens
        ret[duplicate_pos_at_duplicates] = duplicate_tokens
        
        return ret
    
class NameMoverHead(t.nn.Module):
    def __init__(self, names: Tensor, d_vocab : int=40):
        super().__init__()
        self.d_vocab_out = d_vocab
        self.names = names

    def forward(self, tokens: Tensor, s_inhibition: Tensor) -> Tensor:
        """
        increase logit of all names in the sentence, except those flagged by s_inhibition
        """
        batch, seq = tokens.shape
        logits = t.zeros((batch, seq, self.d_vocab_out), device=tokens.device) # batch seq d_vocab
        # we want every name to increase its corresponding logit after it appears
        self.names = self.names.to(tokens.device)
        name_mask = t.isin(tokens, self.names)
        
        batch_indices, seq_indices = t.meshgrid(t.arange(batch), t.arange(seq), indexing='ij')
        logits[batch_indices, seq_indices, tokens] = 10 * name_mask.float()
        # now decrease the logit of the names that are inhibited
        logits[batch_indices, seq_indices, s_inhibition] += -15 * s_inhibition.ne(-1).float()
        logits = t.cumsum(logits, dim=1)
        return logits
    
# since 0, 3 contains 20, we write
# a 1 to position 0, 3, 20 of logits
        
# %%
        
class IOI_HL(HookedRootModule, HLModel):
    """
    Components:
    - Duplicate token heads: write the previous position of any duplicated token
    - Previous token heads: copy token S1 to token S1+1
    - Induction heads (omitted): Attend to position written by duplicate token heads
    - S-inhibition heads: Inhibit attention of Name Mover Heads to S1 and S2 tokens
    - Name mover heads: Copy all previous names in the sentence
    """
    def __init__(
            self, 
            d_vocab: int, 
            names: Tensor, 
            device: t.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
        ):
        super().__init__()
        assert isinstance(names, Tensor), ValueError(f"Expected a tensor, got {type(names)}")
        self.all_nodes_hook = HookPoint()
        self.duplicate_head = DuplicateHead()
        self.hook_duplicate = HookPoint()
        # self.previous_head = PreviousHead()
        self.hook_previous = HookPoint()
        self.s_inhibition_head = SInhibitionHead()
        self.hook_s_inhibition = HookPoint()
        self.name_mover_head = NameMoverHead(names, d_vocab)
        self.hook_name_mover = HookPoint()

        self.d_vocab = d_vocab
        self.cfg = Namespace(
            d_vocab=d_vocab,
            d_vocab_out=d_vocab,
            device=device
        )
        self.setup()
    
    @property
    def device(self) -> t.device:
        return self.cfg.device

    def is_categorical(self) -> bool:
        return True
    
    def forward(self, args: Tensor | tuple, verbose: bool = False) -> Tensor:
        show: Callable[[t.Any], None] = lambda *args, **kwargs: print(*args, **kwargs) if verbose else None
        if isinstance(args, Tensor):
            input = args
        elif isinstance(args, tuple):
            input = args[0]
        else:
            raise ValueError(f"Expected a tensor or tuple, got {type(args)}")
        batched = True
        if len(input.shape) == 1:
            batched = False
            input = input[None, ...]
        # print([a.shape for a in args])
        # duplicate, previous, induction, s_inhibition, name_mover = [intermediate_data[:, i] for i in range(5)]
        # print(f"intermediate_data is a {type(intermediate_data)}; duplicate is a {type(duplicate)}")
        input = self.all_nodes_hook(input)
        duplicate = self.duplicate_head(input)
        assert duplicate.shape == input.shape
        duplicate = self.hook_duplicate(duplicate)
        show(f"duplicate: {duplicate}")
        # previous = self.previous_head(input)
        # assert previous.shape == input.shape
        # previous = self.hook_previous(previous)
        # show(f"previous: {previous}")
        s_inhibition = self.s_inhibition_head(input, duplicate)
        assert s_inhibition.shape == input.shape
        s_inhibition = self.hook_s_inhibition(s_inhibition)
        show(f"s_inhibition: {s_inhibition}")
        out = self.name_mover_head(input, s_inhibition)
        assert out.shape == input.shape + (self.d_vocab,)
        out = self.hook_name_mover(out)
        show(f"out: {t.argmax(out, dim=-1)}")
        if not batched:
            out = out[0]
        return out
# %%
