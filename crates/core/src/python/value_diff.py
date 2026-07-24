"""Compare two `s3://` cstorch checkpoints' tensor VALUES on the proxy.

Driven by `remote.rs::value_diff_script`. Both checkpoints are read on the remote
(which holds the S3 access) and only the per-tensor verdicts cross the ssh link —
never tensor data. Emits progress lines plus one sentinel-tagged JSON result per
pair.

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
WANT_VALUES = PARAMS["want_values"]
WANT_HIST = PARAMS["want_hist"]
FULL_HIST = PARAMS["full_hist"]
BINS = PARAMS["bins"]
JOBS = PARAMS["jobs"]
CHUNK = 4000000
DL = 16 << 20        # S3 download chunk
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
    emit({"error": "import numpy/torch/zstandard failed (needed to compare values): %r" % (e,)}); sys.exit(0)
try:
    stat("load\told")
    old_sd = cstorch.load(OLD, map_location=None)
    stat("load\tnew")
    new_sd = cstorch.load(NEW, map_location=None)
except Exception as e:
    emit({"error": "cstorch.load failed: %r" % (e,)}); sys.exit(0)
def to_f64(rt, i, j):
    sl = rt if i is None else rt[i:j]
    return sl.to(torch.float64).numpy().reshape(-1)
# Map a numpy dtype name (from an object's __NUMPY__ metadata) to a torch dtype, so
# we can stream the S3 object ourselves (chunked, with progress) rather than letting
# cstorch materialise it in one shot with no progress. Comparison stays on the proxy.
NP2T = {}
for _n in ("float16","float32","float64","bfloat16","uint8","int8","int16","uint16","int32","uint32","int64","uint64","bool"):
    _t = getattr(torch, _n, None)
    if _t is not None: NP2T[_n] = _t
def prep(dt):
    do = getattr(dt, "deferred", None)
    if do is None: return None
    r = do._reader
    key = "%s/%s" % (r.key, do._key)
    try:
        head = r.s3_client.head_object(Bucket=r.bucket, Key=key)
        md = head.get("Metadata") or {}
        meta = json.loads(md.get("metadata") or md.get("Metadata") or "{}")
    except Exception:
        return None
    nm = meta.get("__NUMPY__")
    td = NP2T.get(nm.get("dtype")) if nm else None
    if td is None: return None
    return (r.s3_client, r.bucket, key, int(head.get("ContentLength", 0)), bool(meta.get("compressed")), td, [int(x) for x in nm["shape"]])
def stream_raw(p, on):
    # Just the network read (with byte progress). Decompression + reshape is deferred
    # (see decode_tensor) so the bar reaches 100% before the slow decode, not during it.
    client, bucket, key, total, comp, td, shape = p
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    buf = bytearray()
    while True:
        c = body.read(DL)
        if not c: break
        buf += c; on(len(buf))
    return buf
def decode_tensor(buf, p):
    client, bucket, key, total, comp, td, shape = p
    raw = zstandard.decompress(bytes(buf)) if comp else buf
    return torch.frombuffer(bytearray(raw), dtype=td).reshape(shape)
total = len(PAIRS)
t0 = time.time()
read_bytes = 0
compared = 0
def work(idx):
    oname = PAIRS[idx][0]; nname = PAIRS[idx][1]
    stat("start\t%d\t%d\t%s" % (idx + 1, total, nname))
    res = {"name": nname}
    tb = 0
    try:
        a = old_sd[oname]; b = new_sd[nname]
        ashape = [int(d) for d in a.shape]; bshape = [int(d) for d in b.shape]
        if ashape != bshape:
            res["error"] = "shapes differ"; emit(res); return 0
        # Stream each side's S3 object on the proxy (chunked, with byte progress); the
        # per-chunk float64 conversions below work off the in-memory copy. Values are
        # compared here — only progress + the small result cross ssh. Fall back to
        # cstorch materialise (no byte progress) for a non-numpy-stored tensor.
        pa = prep(a); pb = prep(b)
        streamed = None   # (raw_a, raw_b) when read over the network
        if pa is not None and pb is not None:
            osz = pa[3]; nsz = pb[3]
            stat("size\t%s\t%d\t%d" % (nname, osz, nsz))
            od = [0]; nd = [0]; last = [0]
            def bump():
                s = od[0] + nd[0]
                if s - last[0] >= EMIT:
                    last[0] = s; stat("bytes\t%s\t%d\t%d" % (nname, od[0], nd[0]))
            rawa = stream_raw(pa, lambda n: (od.__setitem__(0, n), bump()))
            rawb = stream_raw(pb, lambda n: (nd.__setitem__(0, n), bump()))
            stat("bytes\t%s\t%d\t%d" % (nname, osz, nsz))
            streamed = (rawa, rawb)
            tb = osz + nsz
        else:
            ra = a.to("cpu"); rb = b.to("cpu")
            tb = ra.element_size() * ra.nelement() + rb.element_size() * rb.nelement()
            stat("size\t%s\t%d\t%d" % (nname, int(ra.element_size() * ra.nelement()), int(rb.element_size() * rb.nelement())))
        # Bytes are all read (bar at 100%); decompress + decode below is the slow part,
        # so flag "comparing" now — not after it.
        stat("phase\t%s\tcomparing" % (nname,))
        if streamed is not None:
            ra = decode_tensor(streamed[0], pa); rb = decode_tensor(streamed[1], pb)
        res["bytes"] = int(tb)
        scalar = (len(ashape) == 0)
        d0 = 0 if scalar else ashape[0]
        inner = 1
        for d in ashape[1:]: inner *= d
        rows = max(1, CHUNK // max(1, inner))
        spans = [(None, None)] if scalar else [(i, min(i + rows, d0)) for i in range(0, d0, rows)]
        if not spans: spans = [(None, None)]
        elements = 0; differing = 0; nfm = 0; max_abs = 0.0; sum_abs = 0.0
        omin = omax = nmin = nmax = None
        for (i, j) in spans:
            av = to_f64(ra, i, j); bv = to_f64(rb, i, j)
            if WANT_VALUES:
                both_nan = np.isnan(av) & np.isnan(bv)
                dmask = ~((av == bv) | both_nan)
                elements += int(av.size)
                differing += int(np.count_nonzero(dmask))
                bothfin = np.isfinite(av) & np.isfinite(bv)
                fdm = dmask & bothfin
                if bool(np.any(fdm)):
                    dd = np.abs(av[fdm] - bv[fdm])
                    m = float(dd.max())
                    if m > max_abs: max_abs = m
                    sum_abs += float(dd.sum())
                nfm += int(np.count_nonzero(dmask & ~bothfin))
            if WANT_HIST:
                af = av[np.isfinite(av)]; bf = bv[np.isfinite(bv)]
                if af.size:
                    lo = float(af.min()); hi = float(af.max())
                    omin = lo if omin is None else min(omin, lo)
                    omax = hi if omax is None else max(omax, hi)
                if bf.size:
                    lo = float(bf.min()); hi = float(bf.max())
                    nmin = lo if nmin is None else min(nmin, lo)
                    nmax = hi if nmax is None else max(nmax, hi)
        if WANT_VALUES:
            mean_abs = (sum_abs / elements) if elements > 0 else 0.0
            res["values"] = {"elements": elements, "differing": differing, "max_abs": max_abs, "mean_abs": mean_abs, "nonfinite_mismatch": nfm}
        if WANT_HIST:
            mins = [x for x in (omin, nmin) if x is not None]
            maxs = [x for x in (omax, nmax) if x is not None]
            n = BINS if BINS else 40
            lo_c = min(mins) if mins else 0.0
            hi_c = max(maxs) if maxs else 0.0
            if hi_c <= lo_c: n = 1
            oc = np.zeros(n, dtype=np.int64); nc = np.zeros(n, dtype=np.int64)
            onf = 0; nnf = 0
            for (i, j) in spans:
                av = to_f64(ra, i, j); bv = to_f64(rb, i, j)
                afin = np.isfinite(av); bfin = np.isfinite(bv)
                onf += int(av.size) - int(np.count_nonzero(afin))
                nnf += int(bv.size) - int(np.count_nonzero(bfin))
                af = av[afin]; bf = bv[bfin]
                if hi_c <= lo_c:
                    oc[0] += int(af.size); nc[0] += int(bf.size)
                else:
                    hc, _ = np.histogram(af, bins=n, range=(lo_c, hi_c)); oc += hc.astype(np.int64)
                    hc, _ = np.histogram(bf, bins=n, range=(lo_c, hi_c)); nc += hc.astype(np.int64)
            ot = int(oc.sum()); nt = int(nc.sum())
            if ot == 0 and nt == 0: tvd = 0.0
            elif ot == 0 or nt == 0: tvd = 1.0
            else: tvd = 0.5 * float(np.abs(oc.astype(np.float64) / ot - nc.astype(np.float64) / nt).sum())
            h = {"tvd": tvd, "n": int(n)}
            if FULL_HIST:
                h["lo"] = lo_c; h["hi"] = hi_c
                h["old"] = [int(x) for x in oc.tolist()]; h["new"] = [int(x) for x in nc.tolist()]
                h["old_total"] = ot; h["new_total"] = nt; h["old_nonfinite"] = onf; h["new_nonfinite"] = nnf
            res["histogram"] = h
    except Exception as e:
        res["error"] = "%r" % (e,)
    emit(res)
    return tb
# Reading each tensor's S3 object is latency-bound, so overlap JOBS of them; results
# are order-independent. JOBS<=1 stays sequential (the safe fallback). Byte + count
# tallies happen in this main thread as results land, so no locking is needed there.
if JOBS <= 1:
    tbs = (work(i) for i in range(total))
else:
    ex = ThreadPoolExecutor(max_workers=JOBS)
    tbs = ex.map(work, range(total))
for tb in tbs:
    if tb > 0:
        read_bytes += tb; compared += 1
emit({"summary": {"tensors": total, "compared": compared, "bytes": int(read_bytes), "elapsed_s": time.time() - t0}})
