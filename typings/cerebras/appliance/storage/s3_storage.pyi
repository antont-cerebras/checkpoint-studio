"""`cerebras.appliance.storage.s3_storage` — the one internal cstorch class we patch.

`cstorch_fast.py` memoizes `S3Reader.stats` because on the S3 reader it is a property
that re-issues `head_object` on every access (see that script's docstring). This is the
only place we reach into cstorch internals, so it is the one stub entry that is *not* a
public API and could move between cstorch versions — the patch is written to fail soft,
and this stub is where you'd look first if the read got slow again.

`stats` is typed `Any` rather than a property: on the class it is a `property` object
(which `cstorch_fast.py` inspects with `isinstance`) and on an instance it is the stat
value, and a stub can only declare one of those.
"""

from typing import Any

class S3Reader:
    path: str
    stats: Any

    # NOT part of cstorch's API — the idempotency marker `cstorch_fast.py` sets on the
    # class so a second exec of the prelude doesn't wrap the getter twice. Declared here
    # only so that assignment type-checks; ignore it when auditing the cstorch surface.
    _ckpt_studio_memoized: bool

__all__ = ["S3Reader"]
