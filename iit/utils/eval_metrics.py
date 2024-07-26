import torch as t
from torch import Tensor
import iit.utils.index as index

def kl_div(
        a: Tensor,
        b: Tensor,
        label_idx: index.TorchIndex
        ) -> Tensor:
    a_pmf = a[label_idx.as_index]
    b_pmf = b[label_idx.as_index]
    # check if b is ints
    if b_pmf.dtype in [t.int32, t.int64, t.long, t.int]:
        if b.shape == a.shape[:-1]:
            b_pmf = t.nn.functional.one_hot(b_pmf, num_classes=a_pmf.shape[-1]).float()
        b_pmf = b_pmf.float()
    pmf_checker = lambda x: t.allclose(
        x.sum(dim=-1), t.ones_like(x.sum(dim=-1))
    )
    if not pmf_checker(a_pmf):
        a_pmf = t.nn.functional.log_softmax(a_pmf, dim=-1)
    else:
        a_pmf = t.log(a_pmf)
    if not pmf_checker(b_pmf):
        b_pmf = t.nn.functional.softmax(b_pmf, dim=-1)

    return t.nn.functional.kl_div(
        a_pmf, b_pmf, reduction="none", log_target=False
    ).sum(dim=-1)

def accuracy_affected(
             a: Tensor,
             b: Tensor,
             label_unchanged: Tensor,
             label_idx: index.TorchIndex
             ) -> Tensor:
    a_lab = t.argmax(a[label_idx.as_index], dim=-1)
    b_lab = t.argmax(b[label_idx.as_index], dim=-1)

    out_unchanged = t.eq(a_lab, b_lab)
    changed_result = (~out_unchanged).cpu().float() * (~label_unchanged).cpu().float()
    return changed_result.sum() / (~label_unchanged).sum()

