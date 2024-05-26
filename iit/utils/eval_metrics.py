import torch
import iit.utils.index as index

def kl_div(a: torch.Tensor,
           b: torch.Tensor,
           label_idx: index.TorchIndex):
    a_pmf = a[label_idx.as_index]
    b_pmf = b[label_idx.as_index]
    # check if b is ints
    if b_pmf.dtype in [torch.int32, torch.int64, torch.long, torch.int]:
        assert b.shape == a.shape[:-1], ValueError(f"Got unknown shapes for a and b: {a.shape} and {b.shape}")
        b_pmf = torch.nn.functional.one_hot(b_pmf, num_classes=a_pmf.shape[-1]).float()
    pmf_checker = lambda x: torch.allclose(
        x.sum(dim=-1), torch.ones_like(x.sum(dim=-1))
    )
    if not pmf_checker(a_pmf):
        a_pmf = torch.nn.functional.softmax(a_pmf, dim=-1)
    if not pmf_checker(b_pmf):
        b_pmf = torch.nn.functional.softmax(b_pmf, dim=-1)

    return torch.nn.functional.kl_div(
        a_pmf.log(), b_pmf, reduction="none", log_target=False
    ).sum(dim=-1)

def accuracy_affected(
             a: torch.Tensor,
             b: torch.Tensor,
             label_unchanged: torch.Tensor,
             label_idx: index.TorchIndex):
    a_lab = torch.argmax(a[label_idx.as_index], dim=-1)
    b_lab = torch.argmax(b[label_idx.as_index], dim=-1)

    out_unchanged = torch.eq(a_lab, b_lab)
    changed_result = (~out_unchanged).cpu().float() * (~label_unchanged).cpu().float()
    return changed_result.sum() / (~label_unchanged).sum()

