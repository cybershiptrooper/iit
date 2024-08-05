from typing import Callable
import torch as t
from torch import Tensor
from .ioi_hl import DuplicateHead, PreviousHead, SInhibitionHead, NameMoverHead, IOI_HL
from transformer_lens.hook_points import HookPoint
from transformer_lens import ActivationCache
import numpy as np

IOI_TEST_NAMES = t.tensor([10, 20, 30])


def nonzero_values(a: Tensor) -> Tensor:
    return t.cat((a.nonzero(), a[a != 0][:, None]), dim=-1)


def make_hook(corrupted_cache: ActivationCache, hook_name: str) -> Callable[[Tensor, HookPoint], Tensor]:
    def hook_fn(hook_point_out: Tensor, hook: HookPoint) -> Tensor:
        out = hook_point_out.clone()
        out = corrupted_cache[hook_name]
        return out

    return hook_fn


def test_duplicate_head() -> None:
    a = DuplicateHead()(t.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5]]))
    assert a.equal(t.tensor([[-1, -1, -1, 1, -1, -1, -1, -1, 4]]))


def test_previous_head() -> None:
    a = PreviousHead()(t.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5]]))
    assert a.equal(t.tensor([[-1, 3, 1, 4, 1, 5, 9, 2, 6]]))


def test_s_inhibition_head() -> None:
    a = SInhibitionHead()(
        t.tensor([[3, 1, 4, 1, 5, 9, 2, 6, 5]]),
        t.tensor([[-1, -1, -1, 1, -1, -1, -1, -1, 4]]),
    )
    assert a.equal(t.tensor([[-1, -1, -1, 1, -1, -1, -1, -1, 5]]))


def test_name_mover_head() -> None:
    a = NameMoverHead(IOI_TEST_NAMES, d_vocab=21)(
        t.tensor([[1, 2, 10, 20]]), t.tensor([[-1, 20, 10, -1]])
    )

    assert nonzero_values(a[0]).equal(
        t.tensor(
            [
                [1.0, 20.0, -15.0],
                [2.0, 10.0, -5.0],
                [2.0, 20.0, -15.0],
                [3.0, 10.0, -5.0],
                [3.0, 20.0, -5.0],
            ]
        )
    )


def test_ioi_hl() -> None:
    a = IOI_HL(d_vocab=21, names=IOI_TEST_NAMES)(
        (t.tensor([[3, 10, 4, 10, 5, 9, 2, 6, 5]]), None, None)
    )
    assert nonzero_values(a[0]).equal(
        t.tensor(
            [
                [1.0, 10.0, 10.0],
                [2.0, 10.0, 10.0],
                [3.0, 10.0, 5.0],
                [4.0, 10.0, 5.0],
                [5.0, 10.0, 5.0],
                [6.0, 10.0, 5.0],
                [7.0, 10.0, 5.0],
                [8.0, 5.0, -15.0],
                [8.0, 10.0, 5.0],
            ]
        )
    )


def test_duplicate_head_patching() -> None:
    test_names = t.tensor(range(10, 60, 1))
    hl_model = IOI_HL(d_vocab=61, names=test_names)

    aba = t.tensor([[1, 2, -1, 3, 4, -2, 5, 6, -1]])
    abb = t.tensor([[1, 2, -1, 3, 4, -2, 5, 6, -2]])
    baa = t.tensor([[1, 2, -2, 3, 4, -1, 5, 6, -1]])
    bab = t.tensor([[1, 2, -2, 3, 4, -1, 5, 6, -2]])
    all_prompts = [aba, abb, baa, bab]
    prompt_type = ["aba", "abb", "baa", "bab"]
    same_combinations = [("bab", "aba"), ("abb", "baa"), ("aba", "bab"), ("baa", "abb")]
    for i, p_clean_ in enumerate(all_prompts):
        # test all permutations of prompts
        for j, p_corrupted_ in enumerate(all_prompts):
            if i == j:
                continue
            # sample 2 names from IOI_TEST_NAMES
            name_idxs = np.random.choice(len(test_names), 4, replace=False)
            name_a = test_names[name_idxs[0]]
            name_b = test_names[name_idxs[1]]
            p_clean = p_clean_.clone()
            p_clean[p_clean == -1] = name_a
            p_clean[p_clean == -2] = name_b

            name_idxs = np.random.choice(len(test_names), 4, replace=False)
            name_a = test_names[name_idxs[0]]
            name_b = test_names[name_idxs[1]]
            p_corrupted = p_corrupted_.clone()
            p_corrupted[p_corrupted == -1] = name_a
            p_corrupted[p_corrupted == -2] = name_b
            _, model_corrupted_cache = hl_model.run_with_cache(
                (p_corrupted, None, None)
            )
            model_clean_out = hl_model((p_clean, None, None))

            model_patch_out = hl_model.run_with_hooks(
                (p_clean, None, None),
                fwd_hooks=[
                    (
                        "hook_duplicate",
                        make_hook(model_corrupted_cache, "hook_duplicate"),
                    )
                ],
            )
            if not t.equal(model_patch_out, model_clean_out):
                combination = (prompt_type[j], prompt_type[i])
                assert (
                    combination not in same_combinations
                ), f"Expected unchanged outputs for {prompt_type[j]} -> {prompt_type[i]} but got different outputs."
            else:
                combination = (prompt_type[j], prompt_type[i])
                assert (
                    combination in same_combinations
                ), f"Expected different outputs for {prompt_type[j]} -> {prompt_type[i]} but got unchanged outputs."
    


def test_all_nodes_patching() -> None:
    hl_model = IOI_HL(d_vocab=21, names=IOI_TEST_NAMES)
    p_clean = t.tensor(
        [[1, 2, IOI_TEST_NAMES[0], 3, 4, IOI_TEST_NAMES[1], 5, 6, IOI_TEST_NAMES[0]]]
    )
    p_corrupted = t.tensor(
        [[1, 2, IOI_TEST_NAMES[0], 3, 4, IOI_TEST_NAMES[1], 5, 6, IOI_TEST_NAMES[1]]]
    )
    model_corrupted_out, model_corrupted_cache = hl_model.run_with_cache(
        (p_corrupted, None, None)
    )
    model_patch_out = hl_model.run_with_hooks(
        (p_clean, None, None),
        fwd_hooks=[
            ("all_nodes_hook", make_hook(model_corrupted_cache, "all_nodes_hook"))
        ],
    )
    assert t.equal(model_patch_out, model_corrupted_out)



def test_s_inhibition_head_patching() -> None:
    return
    # Not implemented yet
    test_names = t.tensor(range(10, 60, 1))
    hl_model = IOI_HL(d_vocab=61, names=test_names)

    aba = t.tensor([[1, 2, -1, 3, 4, -2, 5, 6, -1]])
    abb = t.tensor([[1, 2, -1, 3, 4, -2, 5, 6, -2]])
    baa = t.tensor([[1, 2, -2, 3, 4, -1, 5, 6, -1]])
    bab = t.tensor([[1, 2, -2, 3, 4, -1, 5, 6, -2]])
    all_prompts = [aba, abb, baa, bab]
    prompt_type = ["aba", "abb", "baa", "bab"]
    # same_combinations = [("bab", "aba"), ("abb", "baa"), ("aba", "bab"), ("baa", "abb")]
    for i, p_clean_ in enumerate(all_prompts):
        # test all permutations of prompts
        for j, p_corrupted_ in enumerate(all_prompts):
            if i == j:
                continue
            # sample 2 names from IOI_TEST_NAMES
            name_idxs = np.random.choice(len(test_names), 4, replace=False)
            name_a = test_names[name_idxs[0]]
            name_b = test_names[name_idxs[1]]
            p_clean = p_clean_.clone()
            p_clean[p_clean == -1] = name_a
            p_clean[p_clean == -2] = name_b

            name_idxs = np.random.choice(len(test_names), 4, replace=False)
            name_a = test_names[name_idxs[0]]
            name_b = test_names[name_idxs[1]]
            p_corrupted = p_corrupted_.clone()
            p_corrupted[p_corrupted == -1] = name_a
            p_corrupted[p_corrupted == -2] = name_b
            model_corrupted_out, model_corrupted_cache = hl_model.run_with_cache(
                (p_corrupted, None, None)
            )
            model_clean_out = hl_model((p_clean, None, None))

            model_patch_out = hl_model.run_with_hooks(
                (p_clean, None, None),
                fwd_hooks=[
                    (
                        "hook_s_inhibition",
                        make_hook(model_corrupted_cache, "hook_s_inhibition"),
                    )
                ],
            )
            if not t.equal(model_patch_out, model_clean_out):
                print("Different for ", prompt_type[j], " -> ", prompt_type[i])
            else:
                print("Same for ", prompt_type[j], " -> ", prompt_type[i])
    
