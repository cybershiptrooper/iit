from .ioi_dataset_tl import *
from .ioi_hl import *

def make_ioi_dataset_and_hl(num_samples, ll_model, NAMES, device=DEVICE, verbose=False):
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