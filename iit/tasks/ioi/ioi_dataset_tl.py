"""Evaluation Helpers.

This module contains some rough evals for models, but you are likely better off using the
HuggingFace evaluate library if you want to do anything properly. This is however here if you want
it and want to eg cheaply and roughly compare models you've trained to baselines.
"""

import random
from typing import Dict, List, Optional

import einops
import torch as t
from torch import Tensor
import tqdm.auto as tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from transformer_lens import utils, HookedTransformer
from iit.utils.config import DEVICE
from iit.utils.iit_dataset import dataset_len


# %%
def sanity_check(model: HookedTransformer) -> Tensor:
    """
    Very basic eval - just feeds a string into the model (in this case, the first paragraph of Circuits: Zoom In), and returns the loss. It's a rough and quick sanity check - if the loss is <5 the model is probably OK, if the loss is >7 something's gone wrong.

    Note that this is a very basic eval, and doesn't really tell you much about the model's performance.
    """

    text = "Many important transition points in the history of science have been moments when science 'zoomed in.' At these points, we develop a visualization or tool that allows us to see the world in a new level of detail, and a new field of science develops to study the world through this lens."

    return model(text, return_type="loss")


# %%
def make_wiki_data_loader(tokenizer: AutoTokenizer, batch_size: int = 8) -> DataLoader:
    """
    Evaluate on Wikitext 2, a dump of Wikipedia articles. (Using the train set because it's larger, I don't really expect anyone to bother with quarantining the validation set nowadays.)

    Note there's likely to be dataset leakage into training data (though I believe GPT-2 was explicitly trained on non-Wikipedia data)
    """
    wiki_data = load_dataset("wikitext", "wikitext-2-v1", split="train")
    print(len(wiki_data))
    dataset = utils.tokenize_and_concatenate(wiki_data, tokenizer)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return data_loader


def make_owt_data_loader(tokenizer: AutoTokenizer, batch_size: int = 8) -> DataLoader:
    """
    Evaluate on OpenWebText an open source replication of the GPT-2 training corpus (Reddit links with >3 karma)

    I think the Mistral models were trained on this dataset, so they get very good performance.
    """
    owt_data = load_dataset("stas/openwebtext-10k", split="train")
    print(len(owt_data))
    dataset = utils.tokenize_and_concatenate(owt_data, tokenizer)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return data_loader


def make_pile_data_loader(tokenizer: AutoTokenizer, batch_size: int = 8) -> DataLoader:
    """
    Evaluate on the first 10k texts from The Pile.

    The Pile is EleutherAI's general-purpose english dataset, made of 22 subsets
    including academic papers, books, internet content...
    """
    pile_data = load_dataset("NeelNanda/pile-10k", split="train")
    print(len(pile_data))
    dataset = utils.tokenize_and_concatenate(pile_data, tokenizer)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return data_loader


def make_code_data_loader(tokenizer: AutoTokenizer, batch_size: int = 8) -> DataLoader:
    """
    Evaluate on the CodeParrot dataset, a dump of Python code.

    All models seem to get significantly lower loss here (even non-code trained models like GPT-2),
    presumably code is much easier to predict than natural language?
    """
    code_data = load_dataset("codeparrot/codeparrot-valid-v2-near-dedup", split="train")
    print(len(code_data))
    dataset = utils.tokenize_and_concatenate(
        code_data, tokenizer, column_name="content"
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return data_loader


DATASET_NAMES = ["wiki", "owt", "pile", "code"]
DATASET_LOADERS = [
    make_wiki_data_loader,
    make_owt_data_loader,
    make_pile_data_loader,
    make_code_data_loader,
]


# %%
@t.inference_mode()
def evaluate_on_dataset(
    model: HookedTransformer, 
    data_loader: DataLoader, 
    truncate: int = 100, 
    device: str = "cuda"
    ) -> float:
    running_loss = 0
    total = 0
    for batch in tqdm.tqdm(data_loader):
        loss = model(batch["tokens"].to(device), return_type="loss").mean()
        running_loss += loss.item()
        total += 1
        if total > truncate:
            break
    return running_loss / total


# %%
@t.inference_mode()
def induction_loss(
    model: HookedTransformer, 
    tokenizer: Optional[AutoTokenizer] = None,
    batch_size: int = 4, 
    subseq_len: int = 384, 
    prepend_bos: Optional[bool] = None, 
    device: str = "cuda"
) -> Tensor:
    """
    Generates a batch of random sequences repeated twice, and measures model performance on the second half. Tests whether a model has induction heads.

    By default, prepends a beginning of string token (when prepend_bos flag defaults to None, model.cfg.default_prepend_bos is used
    whose default is True unless specified otherwise), which is useful to give models a resting position, and sometimes models were trained with this.
    """
    # Make the repeated sequence
    first_half_tokens = t.randint(100, 20000, (batch_size, subseq_len)).to(device)
    repeated_tokens = einops.repeat(first_half_tokens, "b p -> b (2 p)")

    # Use the provided prepend_bos as an override if it's not None;
    # otherwise use model.cfg.default_prepend_bos (defaults to True)
    prepend_bos = utils.override_or_use_default_value(
        model.cfg.default_prepend_bos, override=prepend_bos
    )

    # Prepend a Beginning Of String token
    if prepend_bos:
        if tokenizer is None:
            tokenizer = model.tokenizer
        repeated_tokens[:, 0] = tokenizer.bos_token_id
    # Run the model, and extract the per token correct log prob
    logits = model(repeated_tokens, return_type="logits")
    correct_log_probs = utils.lm_cross_entropy_loss(
        logits, repeated_tokens, per_token=True
    )
    # Take the loss over the second half of the sequence
    return correct_log_probs[:, subseq_len + 1 :].mean()


# %%
@t.inference_mode()
def evaluate(
    model: HookedTransformer, 
    truncate: int = 100, 
    batch_size: int = 8,
    tokenizer: Optional[AutoTokenizer] = None
    ) -> Dict[str, float]:
    if tokenizer is None:
        tokenizer = model.tokenizer
    losses = {}
    for data_name, data_loader_fn in zip(DATASET_NAMES, DATASET_LOADERS):
        data_loader = data_loader_fn(tokenizer=tokenizer, batch_size=batch_size)
        loss = evaluate_on_dataset(model, data_loader, truncate=truncate)
        print(f"{data_name}: {loss}")
        losses[f"{data_name}_loss"] = loss
    return losses


# %%
class IOIDataset(Dataset):
    """
    Dataset for Indirect Object Identification tasks.
    Paper: https://arxiv.org/pdf/2211.00593.pdf

    Example:

    .. code-block:: python

        >>> from transformer_lens.evals import ioi_eval, IOIDataset
        >>> from transformer_lens.HookedTransformer import HookedTransformer

        >>> model = HookedTransformer.from_pretrained('gpt2-small')
        Loaded pretrained model gpt2-small into HookedTransformer

        >>> # Evaluate like this, printing the logit difference
        >>> print(round(ioi_eval(model, num_samples=100)["Logit Difference"], 3))
        5.476

        >>> # Can use custom dataset
        >>> ds = IOIDataset(
        ...     tokenizer=model.tokenizer,
        ...     num_samples=100,
        ...     templates=['[A] met with [B]. [B] gave the [OBJECT] to [A]'],
        ...     names=['Alice', 'Bob', 'Charlie'],
        ...     nouns={'OBJECT': ['ball', 'book']},
        ... )
        >>> print(round(ioi_eval(model, dataset=ds)["Logit Difference"], 3))
        5.397
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        templates: Optional[List[str]] = None,
        names: Optional[List[str]] = None,
        nouns: Optional[Dict[str, List[str]]] = None,
        num_samples: int = 1000,
        symmetric: bool = False,
        prepend_bos: bool = True,
        seed: int = 42,
        device: t.device = DEVICE,
    ):
        self.tokenizer = tokenizer
        self.prepend_bos = prepend_bos
        self.device = device

        self.templates = (
            templates if templates is not None else self.get_default_templates()
        )
        # self.max_sentence_length = max(len(self.tokenizer.encode(t)) for t in self.templates)
        self.max_sentence_length = 17
        self.names = names if names is not None else self.get_default_names()
        self.nouns = nouns if nouns is not None else self.get_default_nouns()

        self.samples = []
        random.seed(seed)
        for _ in range(num_samples // 2 if symmetric else num_samples):
            # If symmetric, get_sample will return two samples
            self.samples.extend(self.get_sample(symmetric=symmetric))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int, pad_token: bool = False) -> Dict[str, Tensor]:
        sample = self.samples[idx]
        prompt = self.tokenizer.encode(sample["text"])
        if self.prepend_bos:
            prompt = [self.tokenizer.bos_token_id] + prompt
        pad_token = self.tokenizer.pad_token_id
        if pad_token:
            prompt += [pad_token] * (self.max_sentence_length - len(prompt))
        idx_to_ablate = len(prompt) - 2

        return {
            "prompt": t.LongTensor(prompt),
            "IO": t.LongTensor(self.tokenizer.encode(sample["IO"])),
            "S": t.LongTensor(self.tokenizer.encode(sample["S"])),
            "idx_to_ablate": t.LongTensor((idx_to_ablate,)),
        }

    def get_sample(self, symmetric: bool = False) -> List[Dict[str, str]]:
        template: str = random.choice(self.templates)
        for noun_type, noun_list in self.nouns.items():
            template = template.replace(f"[{noun_type}]", random.choice(noun_list))

        samples: List[Dict[str, str]] = []

        # Sample two names without replacement
        names = random.sample(self.names, 2)
        sample = template.replace("[A]", names[0])
        sample = sample.replace("[B]", names[1])
        # Prepend spaces to IO and S so that the target is e.g. " Mary" and not "Mary"
        samples.append({"text": sample, "IO": " " + names[0], "S": " " + names[1]})

        if symmetric:
            sample_2 = template.replace("[A]", names[1])
            sample_2 = sample_2.replace("[B]", names[0])
            samples.append(
                {"text": sample_2, "IO": " " + names[1], "S": " " + names[0]}
            )

        return samples

    @staticmethod
    def get_default_names() -> List[str]:
        return ["John", "Mary"]

    @staticmethod
    def get_default_templates() -> List[str]:
        return [
            "Then, [B] and [A] went to the [LOCATION]. [A] gave the [OBJECT] to [B]",
            "Then, [A] and [B] went to the [LOCATION]. [B] gave the [OBJECT] to [A]",
            "Then, [A] and [B] went to the [LOCATION]. [A] gave the [OBJECT] to [B]",
            "Then, [B] and [A] went to the [LOCATION]. [B] gave the [OBJECT] to [A]",
        ]

    @staticmethod
    def get_default_nouns() -> Dict[str, List[str]]:
        return {
            "LOCATION": ["store", "market"],
            "OBJECT": ["milk", "eggs", "bread"],
        }


@t.inference_mode()
def ioi_eval(
    model: HookedTransformer, 
    dataset: Optional[Dataset] = None, 
    batch_size: int = 8, 
    num_samples: int = 1000, 
    tokenizer: Optional[AutoTokenizer] = None, 
    symmetric: bool = False
) -> Dict[str, float]:
    """Evaluate the Model on the Indirect Object Identification Task.

    Args:
        model: HookedTransformer model.
        dataset: PyTorch Dataset that returns a dict with keys "prompt", "IO", and "S".
        batch_size: Batch size to use.
        num_samples: Number of samples to use.
        tokenizer: Tokenizer to use.
        symmetric: Whether to use the symmetric version of the task.

    Returns:
        Average logit difference and accuracy.
    """
    if tokenizer is None:
        tokenizer = model.tokenizer

    if dataset is None:
        dataset = IOIDataset(tokenizer, num_samples=num_samples, symmetric=symmetric)

    def collate(samples: list[dict]) -> dict: #type: ignore 
        prompts = [sample["prompt"] for sample in samples]
        padded_prompts = t.nn.utils.rnn.pad_sequence(prompts, batch_first=True)
        return {
            "prompt": padded_prompts,
            "IO": [sample["IO"] for sample in samples],
            "S": [sample["S"] for sample in samples],
            "prompt_length": [p.shape[0] for p in prompts],
        }

    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate
    )

    total_correct = 0
    total_logit_diff = 0
    for batch in tqdm.tqdm(data_loader):
        batch_logits = model(batch["prompt"], return_type="logits")

        for i in range(batch_logits.shape[0]):
            io = batch["IO"][i]
            s = batch["S"][i]
            prefix_length = batch["prompt_length"][i] - io.shape[0]

            # Trim io and s to the same length
            min_len = min(io.shape[0], s.shape[0])
            io = io[:min_len]
            s = s[:min_len]

            # Remove identical prefixes
            start_idx = t.where(io != s)[0][0]
            io = io[start_idx]
            s = s[start_idx]
            logit_idx = prefix_length + start_idx - 1

            # Get the logits for the tokens we care about
            logits = batch_logits[i, logit_idx]
            correct_logit = logits[io]
            incorrect_logit = logits[s]

            # Compute stats
            logit_diff = correct_logit - incorrect_logit
            correct = logit_diff > 0
            total_correct += correct.item()
            total_logit_diff += logit_diff.item()

    return {
        "Logit Difference": total_logit_diff / dataset_len(dataset),
        "Accuracy": total_correct / dataset_len(dataset),
    }

class IOIDatasetWrapper(IOIDataset):
        
    def get_inputs(self) -> Tensor:
        items = [self.__getitem__(i) for i in range(len(self))]
        inputs = [item[0] for item in items]
        inputs_tensor = t.stack(inputs)
        return inputs_tensor
    
    def get_targets(self) -> list[Tensor]:
        items = [self.__getitem__(i) for i in range(len(self))]
        targets = [item[1] for item in items]
        return targets
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]: # type: ignore 
        x = super().__getitem__(idx)
        prompt = x['prompt']
        y_list = list(prompt[1:])
        y = t.nn.functional.one_hot(t.tensor(y_list), num_classes=self.tokenizer.vocab_size).float()
        return (x['prompt'][:-1].to(self.device), (y).to(self.device), (x['IO']).to(self.device))
    