from iit.tasks.ioi.ioi_dataset_tl import IOIDatasetWrapper
import transformer_lens as tl
from iit.tasks.ioi.ioi_config import NAMES, PLACES, OBJECTS
from iit.tasks.ioi.ioi_config import ALL_TEMPLATES

NOUNS_DICT = {"LOCATION": PLACES, "OBJECT": OBJECTS, "PLACE": PLACES}

def make_ioi_test_dataset(num_samples=1000) -> IOIDatasetWrapper:
    tokenizer = tl.HookedTransformer.from_pretrained("gpt2").tokenizer
    dataset = IOIDatasetWrapper(
        tokenizer=tokenizer, templates=ALL_TEMPLATES, names=NAMES, nouns=NOUNS_DICT, num_samples=num_samples
    )
    return dataset