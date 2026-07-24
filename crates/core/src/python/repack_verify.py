"""Verify two `s3://` cstorch checkpoints hold the same weights in different packings.

Driven by `remote.rs::repack_verify_script`. Decodes shape-folded / sparse-packed
expert tensors (N-bit indices plus their codebook and qscale siblings) on the proxy
and reports whether the unpacked values agree. Only verdicts cross ssh.

Read-only: loads and compares; never writes to either checkpoint.
"""
import sys
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Parameters from the Rust caller: the single `__PARAMS__` slot is replaced with a
# JSON object (see `remote.rs::with_params`). One substitution point keeps the rest
# of this file ordinary Python that ruff and pyright can check.
PARAMS = json.loads("__PARAMS__")
OLD = PARAMS["old"]
NEW = PARAMS["new"]
PAIRS = PARAMS["pairs"]
BITS = PARAMS["bits"]
JOBS = PARAMS["jobs"]
AUTO = PARAMS["auto"]      # sparse<->sparse auto (--values): same shape both sides, fold 1
DL = 16 << 20        # S3 download chunk
CMP = 4000000        # compare column block
EMIT = 64 << 20      # byte-progress emit granularity
S = PARAMS["sentinel"]
ST = PARAMS["status"]
_lock = threading.Lock()
def emit(o):
    with _lock:
        sys.stdout.write(S + json.dumps(o) + "\n"); sys.stdout.flush()
def stat(s):
    with _lock:
        sys.stdout.write(ST + s + "\n"); sys.stdout.flush()
try:
    import cerebras.pytorch as cstorch
except Exception as e:
    emit({"error": "import cerebras.pytorch failed: %r" % (e,)}); sys.exit(0)
try:
    import numpy as np
    import torch
    import zstandard
except Exception as e:
    emit({"error": "import numpy/torch/zstandard failed: %r" % (e,)}); sys.exit(0)
try:
    stat("load\told")
    old_sd = cstorch.load(OLD, map_location=None)
    stat("load\tnew")
    new_sd = cstorch.load(NEW, map_location=None)
except Exception as e:
    emit({"error": "cstorch.load failed: %r" % (e,)}); sys.exit(0)
mask = np.uint16((1 << BITS) - 1)
# Resolve a lazy tensor to its backing S3 object (thread-safe client + key) so we
# can stream it ourselves with progress, rather than cstorch's one-shot read.
def resolve(dt):
    do = getattr(dt, "deferred", None)
    if do is None:
        return None
    r = do._reader
    return (r.s3_client, r.bucket, "%s/%s" % (r.key, do._key))
def head(loc):
    client, bucket, key = loc
    h = client.head_object(Bucket=bucket, Key=key)
    md = h.get("Metadata") or {}
    try:
        meta = json.loads(md.get("metadata") or md.get("Metadata") or "{}")
    except Exception:
        meta = {}
    return int(h.get("ContentLength", 0)), bool(meta.get("compressed")), ("__NUMPY__" in meta)
def download_raw(loc, on):
    # Just the network read (with byte progress). Decompression is deferred so the
    # bar reaches 100% before the slow decode, not during it.
    client, bucket, key = loc
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    buf = bytearray()
    while True:
        c = body.read(DL)
        if not c:
            break
        buf += c
        on(len(buf))
    return buf
def decode_u16(buf, compressed):
    if compressed:
        return np.frombuffer(zstandard.decompress(bytes(buf)), dtype=np.uint16)
    return np.frombuffer(buf, dtype=np.uint16)     # raw 16-bit words, zero-copy
def to_u16(dt):   # fallback via cstorch (no byte progress)
    return dt.to("cpu").contiguous().view(torch.int16).numpy().view(np.uint16)
NPF = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
def aux_meta(loc):
    # (compressed, numpy_dtype, shape, size) for a numpy-float-stored sibling, else None.
    client, bucket, key = loc
    h = client.head_object(Bucket=bucket, Key=key)
    md = h.get("Metadata") or {}
    try:
        meta = json.loads(md.get("metadata") or md.get("Metadata") or "{}")
    except Exception:
        return None
    nm = meta.get("__NUMPY__")
    if not nm or nm.get("dtype") not in NPF:
        return None
    return (bool(meta.get("compressed")), NPF[nm["dtype"]], [int(x) for x in nm["shape"]], int(h.get("ContentLength", 0)))
def stream_aux(sd, name, on_side):
    # Read a sibling float tensor (codebook / scale) over S3 with byte progress → f64.
    # Falls back to cstorch materialise (no progress) if it isn't numpy-float-stored.
    dt = sd[name]
    loc = resolve(dt)
    am = aux_meta(loc) if loc is not None else None
    if loc is None or am is None:
        on_side(0)
        return dt.to("cpu").to(torch.float64).numpy(), 0
    comp, npdt, shape, size = am
    buf = download_raw(loc, on_side)
    raw = zstandard.decompress(bytes(buf)) if comp else buf
    return np.frombuffer(raw, dtype=npdt).reshape(shape).astype(np.float64), size
def aux_size(sd, name):
    # (old_or_new) S3 byte size of a sibling float tensor, 0 if not resolvable — used
    # to size its bar UP FRONT (before it streams), so it never shows an unsized bar.
    if name not in sd:
        return 0
    loc = resolve(sd[name])
    am = aux_meta(loc) if loc is not None else None
    return am[3] if am else 0
def cmp_aux(oname_a, nname_a, osz, nsz):
    # Streams both sides over S3 into this sibling's own bar (keyed by the new name,
    # already sized by the caller), records the NAMES tried, and — like the weight —
    # flags a `comparing` phase before the compare and finishes the bar after it.
    out = {"old_name": oname_a, "new_name": nname_a,
           "old_present": oname_a in old_sd, "new_present": nname_a in new_sd}
    if not out["old_present"] or not out["new_present"]:
        return out
    od = [0]; nd = [0]; last = [0]
    def bump():
        s = od[0] + nd[0]
        if s - last[0] >= EMIT:
            last[0] = s; stat("bytes\t%s\t%d\t%d" % (nname_a, od[0], nd[0]))
    a, _ = stream_aux(old_sd, oname_a, lambda x: (od.__setitem__(0, x), bump()))
    b, _ = stream_aux(new_sd, nname_a, lambda x: (nd.__setitem__(0, x), bump()))
    stat("bytes\t%s\t%d\t%d" % (nname_a, osz, nsz))
    stat("phase\t%s\tcomparing" % (nname_a,))   # decode + compare below
    with agg_lock:
        read_bytes[0] += osz + nsz
    out["shape_old"] = [int(x) for x in a.shape]; out["shape_new"] = [int(x) for x in b.shape]
    if list(a.shape) == list(b.shape):
        d = np.abs(a - b)
        out.update({"elements": int(a.size), "differing": int(np.count_nonzero(a != b)),
                    "max_abs": float(d.max()) if d.size else 0.0, "mean_abs": float(d.mean()) if d.size else 0.0})
    stat("done\t%s" % (nname_a,))   # finish this bar (✓) after the compare
    return out
total = len(PAIRS)
t0 = time.time()
agg_lock = threading.Lock()
read_bytes = [0]
compared = [0]
def work(idx):
    oname = PAIRS[idx][0]; nname = PAIRS[idx][1]
    stat("start\t%d\t%d\t%s" % (idx + 1, total, nname))
    res = {"name": nname}
    # Size the sibling codebook/qscale UP FRONT — before the weight even streams — so
    # their bars show `0/size` immediately instead of an unsized sweep while the (big)
    # weight downloads. Reused by cmp_aux below (no second HEAD).
    aux_sz = {}   # new sibling name -> (old_size, new_size)
    if oname.endswith(".weight") and nname.endswith(".weight"):
        op = oname[:-7]; npx = nname[:-7]
        for kind in ("codebook", "qscale"):
            oa = op + "." + kind; na = npx + "." + kind
            if oa in old_sd and na in new_sd:
                osz = aux_size(old_sd, oa); nsz = aux_size(new_sd, na)
                aux_sz[na] = (osz, nsz)
                stat("size\t%s\t%d\t%d" % (na, osz, nsz))
    try:
        da = old_sd[oname]; db = new_sd[nname]
        ashape = [int(x) for x in da.shape]; bshape = [int(x) for x in db.shape]
        E = ashape[0] if ashape else 0
        W = bshape[0] if bshape else 0
        if AUTO:
            # Sparse<->sparse: same shape both sides, one index per word (fold 1).
            if not ashape or ashape != bshape:
                res["error"] = "sparse compare needs equal shapes (%r vs %r)" % (ashape, bshape); emit(res); return
            fold = 1
        else:
            if not ashape or not bshape or ashape[1:] != bshape[1:] or W <= 0 or E <= W:
                res["error"] = "not a fold pair (shapes %r vs %r)" % (ashape, bshape); emit(res); return
            fold = (E + W - 1) // W
            if W != (E + fold - 1) // fold:
                res["error"] = "fold mismatch (E=%d W=%d -> fold=%d)" % (E, W, fold); emit(res); return
        if fold * BITS > 16:
            res["error"] = "fold*bits=%d exceeds the 16-bit word" % (fold * BITS); emit(res); return
        lo = resolve(da); ln = resolve(db)
        odone = [0]; ndone = [0]; last = [0]
        def bump():
            s = odone[0] + ndone[0]
            if s - last[0] >= EMIT:
                last[0] = s
                stat("bytes\t%s\t%d\t%d" % (nname, odone[0], ndone[0]))
        streamed = None   # (raw_a, comp_a, raw_b, comp_b) when read over the network
        if lo is not None and ln is not None:
            osz, ocomp, onp = head(lo); nsz, ncomp, nnp = head(ln)
            stat("size\t%s\t%d\t%d" % (nname, osz, nsz))
            if onp and nnp:
                def oon(n): odone[0] = n; bump()
                def non(n): ndone[0] = n; bump()
                ra = download_raw(lo, oon)
                rb = download_raw(ln, non)
                stat("bytes\t%s\t%d\t%d" % (nname, osz, nsz))
                streamed = (ra, ocomp, rb, ncomp)
                tb = osz + nsz
            else:
                ao = to_u16(da); bo = to_u16(db)   # not numpy-stored: no byte progress
                tb = int(ao.nbytes) + int(bo.nbytes)
        else:
            ao = to_u16(da); bo = to_u16(db)
            tb = int(ao.nbytes) + int(bo.nbytes)
            stat("size\t%s\t%d\t%d" % (nname, int(ao.nbytes), int(bo.nbytes)))
        # Bytes are all read (bar at 100%); the decompress + decode below is the slow
        # part, so flag "comparing" now — not after it.
        stat("phase\t%s\tcomparing" % (nname,))
        if streamed is not None:
            ao = decode_u16(streamed[0], streamed[1]); bo = decode_u16(streamed[2], streamed[3])
        ao = ao.reshape(E, -1); bo = bo.reshape(W, -1)
        N = ao.shape[1]
        we = (np.arange(E) // fold)
        se = ((np.arange(E) % fold) * BITS).astype(np.uint16)
        # Format checks: old words' bits above BITS, new words' bits above fold*BITS,
        # must all be zero (else the packing assumption is wrong). When fold*BITS==16
        # (e.g. 4-bit ×4) there are no unused high bits — and shifting a uint16 by 16
        # is undefined — so skip the dense check.
        sparse_bad = int(np.count_nonzero(ao >> np.uint16(BITS)))
        dshift = fold * BITS
        dense_bad = 0 if dshift >= 16 else int(np.count_nonzero(bo >> np.uint16(dshift)))
        differing = 0; first = None; maxdelta = 0; big = 0
        sum_abs = 0; sum_old = 0; sum_new = 0; zeros = 0
        blk = max(1, CMP // max(1, E))
        for n0 in range(0, N, blk):
            n1 = min(n0 + blk, N)
            o = ao[:, n0:n1] & mask
            nd = (bo[we, n0:n1] >> se[:, None]) & mask
            # Aggregate |Δ| and per-side sums (for mean |Δ|/parameter + mean index).
            dd = o.astype(np.int64) - nd.astype(np.int64)
            ad = np.abs(dd)
            sum_abs += int(ad.sum())
            sum_old += int(o.sum(dtype=np.uint64))
            sum_new += int(nd.sum(dtype=np.uint64))
            zeros += int(np.count_nonzero(o == 0)) + int(np.count_nonzero(nd == 0))  # both sides
            ne = o != nd
            cnt = int(np.count_nonzero(ne))
            differing += cnt
            if cnt:
                m = int(ad.max())
                if m > maxdelta: maxdelta = m
                big += int(np.count_nonzero(ad > 1))   # differ by more than ±1
            if first is None and cnt:
                p = np.argwhere(ne)[0]; e = int(p[0]); col = n0 + int(p[1])
                first = [e, col, int(o[p[0], p[1]]), int(nd[p[0], p[1]])]
        elems = E * N
        res.update({"elements": elems, "differing": differing, "sparse_bad": sparse_bad, "dense_bad": dense_bad, "fold": fold, "bits": BITS, "bytes": tb, "maxdelta": maxdelta, "big": big,
                    "sum_abs": int(sum_abs),
                    "mean_abs": (sum_abs / elems) if elems else 0.0,
                    "mean_old": (sum_old / elems) if elems else 0.0,
                    "mean_new": (sum_new / elems) if elems else 0.0,
                    "zero_frac": (zeros / (2.0 * elems)) if elems else 0.0})
        # Auto (--values) fallback: the top-bits check failed, so these aren't packed
        # indices — compare the raw words as stored floats instead (same bytes, viewed
        # as float16), so a mis-detected tensor is still meaningfully diffed.
        if AUTO and (sparse_bad > 0 or dense_bad > 0):
            dts = str(db.dtype).replace("torch.", "")
            # Compare in the real stored dtype: F16 words as floats, otherwise the
            # raw 16-bit integers (dense-packed U16, etc.).
            if dts == "float16":
                fa = ao.view(np.float16).astype(np.float64); fb = bo.view(np.float16).astype(np.float64)
            else:
                fa = ao.astype(np.float64); fb = bo.astype(np.float64)
            fd = np.abs(fa - fb)
            fin = np.isfinite(fd)   # F16 bits can be NaN/Inf → JSON-unsafe; mask them
            label = {"float16": "F16", "bfloat16": "BF16", "uint16": "U16", "int16": "I16"}.get(dts, dts.upper())
            res["fallback"] = {"dtype": label,
                               "elements": int(ao.size),
                               # Exact bit inequality (NaN-safe; identical bits ⇒ 0).
                               "differing": int(np.count_nonzero(ao != bo)),
                               "max_abs": float(fd[fin].max()) if bool(fin.any()) else 0.0,
                               "mean_abs": float(fd[fin].mean()) if bool(fin.any()) else 0.0}
        if first is not None:
            res["first"] = first
        # A decoded window (experts × inner-offset), centred on the first mismatch
        # (or the top-left corner), so the caller can SHOW old vs new and see
        # where/how they diverge.
        se0 = max(0, first[0] - 6) if first else 0
        so0 = max(0, first[1] - 16) if first else 0
        se1 = min(E, se0 + 16); so1 = min(N, so0 + 48)
        ew = np.arange(se0, se1)
        sold = (ao[se0:se1, so0:so1] & mask).astype(np.uint16).tolist()
        snew = ((bo[ew // fold][:, so0:so1] >> ((ew % fold) * BITS).astype(np.uint16)[:, None]) & mask).astype(np.uint16).tolist()
        res["sample"] = {"e0": int(se0), "off0": int(so0), "cols": int(so1 - so0), "old": sold, "new": snew}
        # Also diff the sibling codebook + scale tensors (float centroids/scales),
        # since these have the same shape/dtype on both sides and so don't show up in
        # the structural diff — but a codebook difference explains index differences
        # (the same weights quantised against a different codebook).
        if oname.endswith(".weight") and nname.endswith(".weight"):
            op = oname[:-7]; npx = nname[:-7]
            for kind in ("codebook", "qscale"):
                na = npx + "." + kind
                osz, nsz = aux_sz.get(na, (0, 0))
                res[kind] = cmp_aux(op + "." + kind, na, osz, nsz)
        with agg_lock:
            read_bytes[0] += tb; compared[0] += 1
    except Exception as e:
        res["error"] = "%r" % (e,)
    emit(res)
if JOBS <= 1:
    for i in range(total):
        work(i)
else:
    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        list(ex.map(work, range(total)))
emit({"summary": {"tensors": total, "compared": compared[0], "bytes": int(read_bytes[0]), "elapsed_s": time.time() - t0}})
