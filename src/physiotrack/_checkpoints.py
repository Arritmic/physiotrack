"""Checkpoint loading helpers.

Several vendored checkpoints were saved from a model wrapped in
``torch.nn.DataParallel``, which prefixes every key with ``module.``. Whether that
prefix matches at load time then depends on whether the loading code happens to wrap
the model the same way — which in practice meant it matched on CUDA and not on CPU.

That made a checkpoint's loadability a function of the host's hardware:

- with ``strict=True`` the load raised a wall of missing/unexpected keys on CPU;
- with ``strict=False`` it was worse — every key was silently discarded, the model kept
  its random initialisation, and inference returned confident nonsense with no error.

Normalising the prefix away makes loading independent of both the device and how the
checkpoint happened to be saved, so ``strict=True`` becomes safe to insist on.
"""
from typing import Any, Dict

__all__ = ["strip_data_parallel_prefix"]

_PREFIX = "module."


def strip_data_parallel_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove a leading ``module.`` from every key of a ``DataParallel`` state dict.

    Args:
        state_dict (dict): A model state dict, possibly saved from a
            ``torch.nn.DataParallel``-wrapped model.

    Returns:
        dict: The state dict with the prefix removed. Returned unchanged when no key
            carries it, so calling this on an ordinary checkpoint is a no-op.

    Note:
        Only stripped when **every** key carries the prefix. A partially prefixed dict
        is left alone rather than half-rewritten, since that indicates a checkpoint
        this function does not understand.
    """
    if not state_dict or not all(k.startswith(_PREFIX) for k in state_dict):
        return state_dict
    return {k[len(_PREFIX):]: v for k, v in state_dict.items()}
