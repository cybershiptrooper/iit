import torch as t

from iit.model_pairs.ll_model import LLModel
from iit.utils.config import DEVICE
from .ioi_hl import IOI_HL
from .ioi_dataset_tl import IOIDataset, IOIDatasetWrapper

def make_ioi_dataset_and_hl(
        num_samples: int, 
        ll_model: LLModel, 
        NAMES: list[str], 
        device: t.device = DEVICE, 
        verbose: bool = False
        ) -> tuple[IOIDatasetWrapper, IOI_HL]:
    ioi_dataset_tl = IOIDataset(
    num_samples=num_samples,
    tokenizer=ll_model.tokenizer,
    names=NAMES,
    )

    ioi_names = t.tensor(
        list(set([ioi_dataset_tl[i]["IO"].item() for i in range(len(ioi_dataset_tl))]))
    ).to(device)
    hl_model = IOI_HL(d_vocab=ll_model.cfg.d_vocab_out, names=ioi_names).to(device)

    ioi_dataset = IOIDatasetWrapper(
        num_samples=num_samples,
        tokenizer=ll_model.tokenizer,
        names=NAMES,
        device=device
    )

    if verbose:
        sentence = ioi_dataset_tl[0]["prompt"]
        detokenised = [
            ll_model.tokenizer.decode(i, clean_up_tokenization_spaces=True) for i in sentence
        ]
        print(sentence, detokenised)

    return ioi_dataset, hl_model