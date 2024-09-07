import numpy as np
from tests.test_utils.ioi_utils import make_ioi_test_dataset


def test_ioi_dataset() -> None:
    dataset = make_ioi_test_dataset()
    tokenizer = dataset.tokenizer
    # assert all sentences end with "to" after padding and encoding
    for i in dataset:
        postfix = tokenizer.decode(i[0])[-2:]
        assert postfix == "to"
    encoded_names = [tokenizer.encode(" " + i) for i in dataset.names]

    # assert all io names are actually correctly encoded
    ios = [i[-1].item() for i in dataset]
    assert all([io in np.ravel(encoded_names) for io in ios])

    # check max length
    max_len = len(dataset[0][0])
    for num, i in enumerate(dataset):
        assert len(i[0]) == max_len, f"Error in {num}"
