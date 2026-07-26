"""`cerebras.pytorch` — only `load`, which is all the scripts use.

`cstorch.load` returns a lazy state dict: keys are tensor names, values are deferred
tensors that materialise on access (see `torch.Tensor` in these stubs). `dump.py` reads
only `.dtype`/`.shape`/`.element_size()` off them, which is why nothing is fetched for a
metadata read.

`save` is declared so that a script referring to it would still type-check, and NOT used:
the read-only guarantee in `remote.rs` is that these scripts never call it.
"""

from collections.abc import ItemsView, Iterator, KeysView
from typing import Any

from torch import Tensor

class StateDict:
    """The mapping `load` returns. Values may be tensors or plain metadata."""

    def keys(self) -> KeysView[str]: ...
    def items(self) -> ItemsView[str, Any]: ...
    def __getitem__(self, key: str) -> Any: ...
    def __contains__(self, key: str) -> bool: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...

def load(ckpt_path: str, map_location: Any | None = ..., **kwargs: Any) -> StateDict: ...
def save(state_dict: Any, ckpt_path: str, **kwargs: Any) -> None: ...

__all__ = ["StateDict", "Tensor", "load", "save"]
