"""List the objects under an `s3://…` checkpoint prefix (one paginated call).

Driven by `remote.rs::list_script`. Emits `{objects:[[rel_key,size],…]}` as a single
sentinel-tagged JSON line, with keys made prefix-relative so the file browser can
show them s3-natively. List only — no per-object HEAD (that's `dump.py`'s S3 phase).

Read-only: `list_objects_v2` is a read; this never writes.
"""
# Annotations are lazy (PEP 563) so they never execute at import time on the
# cluster's interpreter, and modern syntax works on our 3.9 floor.
from __future__ import annotations

import json
import sys
from typing import Any

# Parameters from the Rust caller: the single `__PARAMS__` slot is replaced with a
# JSON object (see `remote.rs::with_params`). One substitution point keeps the rest
# of this file ordinary Python that ruff and pyright can check.
PARAMS = json.loads("__PARAMS__")
SRC = PARAMS["uri"]
S = PARAMS["sentinel"]
def emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(S + json.dumps(obj) + "\n"); sys.stdout.flush()
try:
    from urllib.parse import urlparse

    import boto3
except Exception as e:
    emit({"error": "import boto3 failed: %r" % (e,)}); sys.exit(0)
try:
    u = urlparse(SRC)
    bucket, prefix = u.netloc, u.path.lstrip("/")
    cli = boto3.client("s3")
    objects, tok = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if tok: kw["ContinuationToken"] = tok
        resp = cli.list_objects_v2(**kw)
        for it in resp.get("Contents", []):
            k = it["Key"]
            rel = k[len(prefix):].lstrip("/") if k.startswith(prefix) else k
            if rel:
                objects.append([rel, int(it.get("Size", 0))])
        if resp.get("IsTruncated"): tok = resp.get("NextContinuationToken")
        else: break
except Exception as e:
    emit({"error": "s3 list failed: %r" % (e,)}); sys.exit(0)
emit({"objects": objects})
