from typing import Optional
import pickle

from iit.utils.nodes import HLNode, LLNode
from iit.utils.index import Ix

class Correspondence(dict[HLNode, set[LLNode]]):
    def __init__( # type: ignore
        self,
        *args,
        suffixes: dict = {"attn": "attn.hook_result", "mlp": "mlp.hook_post"},
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.suffixes = suffixes

    def __setattr__(self, key: HLNode | str, value: set[LLNode] | dict[str, str]) -> None: # type: ignore
        if isinstance(key, str):
            if key != "suffixes":
                raise ValueError(f"Key must be an HLNode or 'suffixes', got {key}")
            assert isinstance(value, dict), ValueError(
                f"__value is not a dict, but {type(value)}"
            )
            assert isinstance(value, dict), ValueError(f"suffixes value is not a dict, but {type(value)}")
        else:
            assert isinstance(key, HLNode), "key must be of type HLNode, got %s" % type(
                key
            )
            assert isinstance(value, set), ValueError(
                f"__value is not a set, but {type(value)}"
            )
            assert all(isinstance(v, LLNode) for v in value), ValueError(
                "__value contains non-LLNode elements"
            )
        # print(self.keys(), self.values())
        super().__setattr__(key, value) # type: ignore

    def get_suffixes(self) -> dict:
        return self.suffixes

    @staticmethod
    def get_hook_suffix(corr: dict[HLNode, set[LLNode]]) -> dict[str, str]:
        suffixes: dict[str, str] = {}
        for _, ll_nodes in corr.items():
            for ll_node in ll_nodes:
                # add everything after 'blocks.<layer>.' to the set
                suffix_pieces = ll_node.name.split(".")[2:]
                suffix = ".".join(suffix_pieces)
                if "attn" in ll_node.name:
                    if "attn" in suffixes and suffixes["attn"] != suffix:
                        raise ValueError(
                            f"Multiple attn suffixes found: {suffixes['attn']} and {suffix}, multiple attn hook locations are not supported yet."
                        )
                    suffixes["attn"] = suffix
                elif "mlp" in ll_node.name:
                    if "mlp" in suffixes and suffixes["mlp"] != suffix:
                        raise ValueError(
                            f"Multiple mlp suffixes found: {suffixes['mlp']} and {suffix}, multiple mlp hook locations are not supported yet."
                        )
                    suffixes["mlp"] = suffix
                else:
                    raise ValueError(f"Unknown node type {ll_node.name}")

        return suffixes


    @classmethod
    def make_corr_from_dict(
        cls, 
        d: dict, 
        suffixes: Optional[dict[str, str]] = None, 
        make_suffixes_from_corr: bool = False
        ) -> "Correspondence":
        if make_suffixes_from_corr:
            suffixes = Correspondence.get_hook_suffix(d)
        
        input_dict = {
            HLNode(k, -1): {LLNode(name=node_name, index=Ix[[None]]) for node_name in v}
            for k, v in d.items()
        }
        if suffixes is not None:
            return cls(input_dict, suffixes=suffixes)
        else:
            return cls(input_dict)

    def save(self, filename: str) -> None:
        pickle.dump(self, open(filename, "wb"))
