"""Read an `s3://…` cstorch checkpoint's tensor metadata (names/dtypes/shapes).

Runs on the ssh proxy inside the cstorch venv, driven by `remote.rs::dump_script`.
`cstorch.load` is lazy, so no tensor data is read; each tensor is emitted as one
sentinel-tagged JSON line, optionally followed by per-object S3 metadata.

Read-only: this only *loads* (lazily) and writes to stdout — it never opens a file
for writing, calls `cstorch.save`/`torch.save`, or otherwise mutates the checkpoint.
"""
# Annotations are lazy (PEP 563) so they never execute at import time on the
# cluster's interpreter, and modern syntax works on our 3.9 floor.
from __future__ import annotations

import sys
import json
from typing import Any

# Parameters from the Rust caller: the single `__PARAMS__` slot is replaced with a
# JSON object (see `remote.rs::with_params`). One substitution point keeps the rest
# of this file ordinary Python that ruff and pyright can check.
PARAMS = json.loads("__PARAMS__")
SRC = PARAMS["uri"]
WANT_S3 = PARAMS["want_s3"]
S = PARAMS["sentinel"]
P = PARAMS["progress"]
def emit(obj: dict[str, Any]) -> None:
    sys.stdout.write(S + json.dumps(obj) + "\n")
    sys.stdout.flush()
def prog(done: int, total: int, unit: str | None = None) -> None:
    tail = ("/" + unit) if unit else ""
    sys.stdout.write("%s%d/%d%s\n" % (P, done, total, tail))
    sys.stdout.flush()
def probe_s3(src: str) -> dict[str, Any]:
    # Best-effort, LIST-ONLY diagnosis of a load failure: enumerate the objects
    # under the prefix and report how many are empty (0 bytes) plus whether the
    # cstorch metadata object itself is empty, so the caller can explain *why*
    # cstorch.load failed (empty checkpoint / missing metadata / wrong prefix)
    # instead of surfacing only the raw dill traceback. Read-only; never mutates.
    if not src.startswith("s3://"):
        return {}
    try:
        import boto3
        from urllib.parse import urlparse
        u = urlparse(src)
        bucket, prefix = u.netloc, u.path.lstrip("/")
        cli = boto3.client("s3")
        total = empty = nbytes = 0
        meta_key = None; meta_empty = False; sample = []; tok = None
        while True:
            kw = {"Bucket": bucket, "Prefix": prefix}
            if tok: kw["ContinuationToken"] = tok
            resp = cli.list_objects_v2(**kw)
            for it in resp.get("Contents", []):
                k = it["Key"]; sz = int(it.get("Size", 0))
                rel = k[len(prefix):].lstrip("/") if k.startswith(prefix) else k
                if not rel:
                    continue
                total += 1; nbytes += sz
                if sz == 0: empty += 1
                if rel.rsplit("/", 1)[-1] == "__METADATA__":
                    meta_key = rel; meta_empty = (sz == 0)
                if len(sample) < 12: sample.append([rel, sz])
            if resp.get("IsTruncated"): tok = resp.get("NextContinuationToken")
            else: break
        return {"s3_probe": {"prefix": prefix, "total": total, "empty": empty,
                             "bytes": nbytes, "metadata_key": meta_key,
                             "metadata_empty": meta_empty, "sample": sample}}
    except Exception as e:
        return {"s3_probe_error": "%r" % (e,)}
try:
    import cerebras.pytorch as cstorch
except Exception as e:
    emit({"error": "import cerebras.pytorch failed: %r" % (e,)}); sys.exit(0)
try:
    sd = cstorch.load(SRC, map_location=None)   # lazy: metadata only, no data
except Exception as e:
    emit(dict({"error": "cstorch.load failed: %r" % (e,)}, **probe_s3(SRC))); sys.exit(0)
keys = list(sd.keys())
total = len(keys)
prog(0, total)                                  # total known → bar goes determinate
step = max(1, total // 100)                     # ~100 updates, not one per tensor
tensors = []
for i, name in enumerate(keys):
    try:
        t = sd[name]
        it = int(t.element_size()) if hasattr(t, "element_size") else 0
        tensors.append({"name": str(name), "dtype": str(getattr(t, "dtype", "")), "shape": [int(d) for d in t.shape], "itemsize": it})
    except Exception:
        pass
    if (i + 1) % step == 0 or i + 1 == total:
        prog(i + 1, total)
# S3 object metadata (best-effort, read-only): list the objects under the prefix
# and HEAD each for size/etag/checksum/last-modified/user-metadata; tags need a
# separate (often ungranted) permission, so they're tried per object and a single
# warning is emitted if denied. Any failure degrades to fewer objects + a warning.
s3_objects = []
s3_warnings = []
if WANT_S3:
    try:
        import boto3
        from urllib.parse import urlparse
        u = urlparse(SRC)
        bucket, prefix = u.netloc, u.path.lstrip("/")
        cli = boto3.client("s3")
        keys, tok = [], None
        while True:
            kw = {"Bucket": bucket, "Prefix": prefix}
            if tok: kw["ContinuationToken"] = tok
            resp = cli.list_objects_v2(**kw)
            keys.extend(it["Key"] for it in resp.get("Contents", []))
            if resp.get("IsTruncated"): tok = resp.get("NextContinuationToken")
            else: break
        # Second progress phase: HEADing each object is the slow part, so drive the
        # bar off it (relabelled "S3 objects") instead of sitting at 100% tensors.
        nkeys = len(keys)
        s3_step = max(1, nkeys // 100)
        prog(0, nkeys, "s3")
        tags_denied = False
        for i, k in enumerate(keys):
            if i % s3_step == 0:
                prog(i, nkeys, "s3")
            try:
                h = cli.head_object(Bucket=bucket, Key=k, ChecksumMode="ENABLED")
            except Exception as e:
                s3_warnings.append("head_object failed for %s: %r" % (k, e)); continue
            rel = k[len(prefix):].lstrip("/") if k.startswith(prefix) else k
            lm = h.get("LastModified")
            obj = {"key": rel, "size": int(h.get("ContentLength", 0)),
                   "etag": str(h.get("ETag", "")).strip('"'),
                   "last_modified": lm.isoformat() if lm else "",
                   "metadata": {mk: str(mv) for mk, mv in (h.get("Metadata") or {}).items()}}
            for algo in ("CRC32", "CRC32C", "SHA1", "SHA256"):
                cv = h.get("Checksum" + algo)
                if cv:
                    obj["checksum"] = [algo.lower(), str(cv)]; break
            try:
                tg = cli.get_object_tagging(Bucket=bucket, Key=k)
                obj["tags"] = {t["Key"]: t["Value"] for t in tg.get("TagSet", [])}
            except Exception as e:
                if not tags_denied:
                    s3_warnings.append("tags unavailable (needs s3:GetObjectTagging): %r" % (e,))
                    tags_denied = True
            s3_objects.append(obj)
        prog(nkeys, nkeys, "s3")
    except Exception as e:
        s3_warnings.append("s3 metadata unavailable: %r" % (e,))
emit({"tensors": tensors, "metadata": [], "s3_objects": s3_objects, "s3_warnings": s3_warnings})
