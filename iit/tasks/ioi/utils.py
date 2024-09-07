import torch as t

from iit.model_pairs.ll_model import LLModel
from iit.utils.config import DEVICE

from .ioi_config import ALL_TEMPLATES, NAMES, OBJECTS, PLACES
from .ioi_dataset_tl import IOIDataset, IOIDatasetWrapper
from .ioi_hl import IOI_HL


def make_ioi_dataset_and_hl(
    num_samples: int,
    ll_model: LLModel,
    names: list[str] = NAMES,
    device: t.device = DEVICE,
    verbose: bool = False,
    nouns_dict: dict[str, list[str]] = {
        "LOCATION": PLACES,
        "OBJECT": OBJECTS,
        "PLACE": PLACES,
    },
    templates: list[str] = ALL_TEMPLATES,
) -> tuple[IOIDatasetWrapper, IOI_HL]:
    ioi_dataset_tl = IOIDataset(
        num_samples=num_samples,
        tokenizer=ll_model.tokenizer,
        names=names,
    )

    ioi_names = t.tensor(
        [ll_model.tokenizer.encode(" " + name) for name in ioi_dataset_tl.names]
    ).flatten()
    hl_model = IOI_HL(d_vocab=ll_model.cfg.d_vocab_out, names=ioi_names, device=device)

    ioi_dataset = IOIDatasetWrapper(
        num_samples=num_samples,
        tokenizer=ll_model.tokenizer,
        names=names,
        device=device,
        nouns=nouns_dict,
        templates=templates,
    )

    if verbose:
        sentence = ioi_dataset_tl[0]["prompt"]
        detokenised = [
            ll_model.tokenizer.decode(i, clean_up_tokenization_spaces=True)
            for i in sentence
        ]
        print(sentence, detokenised)

    return ioi_dataset, hl_model
