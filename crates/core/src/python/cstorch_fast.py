"""Make opening an `s3://…` cstorch checkpoint cheap.

Prepended by `remote.rs` to every script that calls `cstorch.load` (there's no
importable module on the proxy — each script is one stdin stream), so it patches at
exec time and the scripts below it need no cooperation.

**The problem.** `cerebras.pytorch.storage.serializers.DeferredTorchTensor.__init__`
records `self._stat = self.deferred._reader.stats`, and on the S3 reader `stats` is a
*property* that issues a fresh `head_object` every access — of the checkpoint's single
`__METADATA__` object, the same key every time. `cstorch.load` builds one deferred
tensor per entry, so opening a checkpoint costs one sequential HTTPS round trip per
tensor: measured 1155 HEADs of one object taking 4.97s of a 7.3s open.

**The fix.** Memoize the property per reader path. The value is only a path plus an
mtime, and it's consulted for staleness in `__getstate__` / `__setstate__` — pickling
paths our read-only scripts never take. With this, the same open takes 0.09s.

Deliberately best-effort: if cstorch's internals move, the `except` leaves the
unpatched (correct, slow) behaviour in place rather than failing the read.

No `from __future__ import annotations` here on purpose: this text is spliced in
*after* the host script's own future import, and Python allows only one, at the top.
Annotations are therefore evaluated at runtime — so keep them to PEP 585 builtin
generics, which our 3.9 floor accepts.
"""

from typing import Any


def _memoize_s3_reader_stats() -> None:
    try:
        from cerebras.appliance.storage.s3_storage import S3Reader
    except Exception:
        return

    original = getattr(S3Reader, "stats", None)
    # Only patch what we recognise: a property whose getter we can still call.
    if not isinstance(original, property) or original.fget is None:
        return
    if getattr(S3Reader, "_ckpt_studio_memoized", False):
        return

    getter = original.fget
    cache: dict[Any, Any] = {}

    def stats(self: Any) -> Any:
        key = getattr(self, "path", id(self))
        if key not in cache:
            cache[key] = getter(self)
        return cache[key]

    S3Reader.stats = property(stats)
    S3Reader._ckpt_studio_memoized = True


_memoize_s3_reader_stats()
