"""Compare two checkpoints' tensor VALUES on the proxy.

Driven by `remote.rs::value_diff_script`. Each side is either an `s3://` cstorch
checkpoint (loaded with cstorch, which holds the S3 access) or a safetensors file /
directory of shards **on this host** (read with `safetensors`) — the proxy can reach
both, which is what makes comparing them possible without moving tensor data
anywhere. Only the per-tensor verdicts cross the ssh link. Emits progress lines plus
one sentinel-tagged JSON result per pair.

Read-only: loads and compares; never writes to either checkpoint.
"""
# Annotations are lazy (PEP 563) so they never execute at import time on the
# cluster's interpreter, and modern syntax works on our 3.9 floor.
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, NamedTuple

# Parameters from the Rust caller: the single `__PARAMS__` slot is replaced with a
# JSON object (see `remote.rs::with_params`). One substitution point keeps the rest
# of this file ordinary Python that ruff and pyright can check.
PARAMS = json.loads("__PARAMS__")
OLD = PARAMS["old"]      # {"kind": "s3"|"safetensors", "path": …}
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
def emit(o: dict[str, Any]) -> None:
    with _lock:
        sys.stdout.write(S + json.dumps(o) + "\n"); sys.stdout.flush()
def stat(s: str) -> None:
    with _lock:
        sys.stdout.write(ST + s + "\n"); sys.stdout.flush()
try:
    import numpy as np
    import torch
except Exception as e:
    emit({"error": "import numpy/torch failed (needed to compare values): %r" % (e,)}); sys.exit(0)
KINDS = {OLD["kind"], NEW["kind"]}
cstorch = None
if "s3" in KINDS:
    try:
        import cerebras.pytorch as cstorch
        import zstandard
    except Exception as e:
        emit({"error": "import cerebras.pytorch/zstandard failed (needed for an s3:// side): %r" % (e,)}); sys.exit(0)
safe_open = None
if "safetensors" in KINDS:
    try:
        from safetensors import safe_open
    except Exception as e:
        emit({"error": "import safetensors failed (needed for a safetensors side): %r" % (e,)}); sys.exit(0)


class SafeSide:
    """A safetensors file or directory of shards on this host, keyed by tensor name.

    Reads a whole tensor at a time, like the s3 path's in-memory copy: the comparison
    below streams *within* the tensor, so peak memory is one tensor per side per
    worker either way. The header scan that builds the map is a few kilobytes per
    shard.
    """

    def __init__(self, path: str) -> None:
        from pathlib import Path

        p = Path(path)
        files = sorted(p.glob("*.safetensors")) if p.is_dir() else [p]
        if not files:
            raise FileNotFoundError("no .safetensors files under %s" % (path,))
        # Which shard holds each tensor. Header reads only — a few kilobytes per shard — and the first
        # shard to claim a name wins, matching how the rest of the app resolves a duplicate.
        self.where: dict[str, str] = {}
        for f in files:
            with safe_open(str(f), framework="pt") as h:
                for k in h.keys():  # noqa: SIM118  (safe_open is not a dict; `in` is not `keys()`)
                    self.where.setdefault(k, str(f))

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def shape(self, name: str) -> list[int]:
        """From the header alone — `get_slice` reads no data, so a shape mismatch is free."""
        with safe_open(self.where[name], framework="pt") as h:
            return [int(d) for d in h.get_slice(name).get_shape()]

    def read(self, name: str) -> torch.Tensor:
        with safe_open(self.where[name], framework="pt") as h:
            return h.get_tensor(name)


def open_side(side: dict[str, Any], which: str) -> Any:
    stat("load\t%s" % (which,))
    if side["kind"] == "s3":
        return cstorch.load(side["path"], map_location=None)
    return SafeSide(side["path"])


try:
    old_sd = open_side(OLD, "old")
    new_sd = open_side(NEW, "new")
except Exception as e:
    emit({"error": "opening a checkpoint failed: %r" % (e,)}); sys.exit(0)
def to_f64(rt: torch.Tensor, i: int | None, j: int | None) -> np.ndarray[Any, Any]:
    sl = rt if i is None else rt[i:j]
    return sl.to(torch.float64).numpy().reshape(-1)
def span_range(rt: torch.Tensor, i: int | None, j: int | None) -> tuple[float, float] | None:
    """Finite min/max of one span WITHOUT widening it to float64.

    The histogram needs the global range before it can bin, which used to cost a whole
    extra float64 pass over the data. Reducing in the stored dtype is ~4x less memory
    traffic for f16, and exact: f16/bf16 min-max convert to f64 losslessly, and an
    integer min/max rounds to f64 identically either way. `None` if nothing is finite.
    """
    sl = rt if i is None else rt[i:j]
    if not sl.is_floating_point():
        return float(sl.min()), float(sl.max())     # integers are always finite
    fin = torch.isfinite(sl)
    if bool(fin.all()):
        return float(sl.min()), float(sl.max())     # common case: no copy at all
    if not bool(fin.any()):
        return None
    v = torch.masked_select(sl, fin)                # native dtype, not float64
    return float(v.min()), float(v.max())
# Map a numpy dtype name (from an object's __NUMPY__ metadata) to a torch dtype, so
# we can stream the S3 object ourselves (chunked, with progress) rather than letting
# cstorch materialise it in one shot with no progress. Comparison stays on the proxy.
NP2T = {}
for _n in ("float16","float32","float64","bfloat16","uint8","int8","int16","uint16","int32","uint32","int64","uint64","bool"):
    _t = getattr(torch, _n, None)
    if _t is not None: NP2T[_n] = _t
class Obj(NamedTuple):
    """One tensor's backing S3 object: where it lives plus how to decode it."""

    client: Any        # boto3 S3 client (thread-safe)
    bucket: str
    key: str
    size: int          # ContentLength, for sizing the progress bar up front
    compressed: bool
    dtype: Any         # torch dtype
    shape: list[int]
def prep(dt: Any) -> Obj | None:
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
    return Obj(r.s3_client, r.bucket, key, int(head.get("ContentLength", 0)),
               bool(meta.get("compressed")), td, [int(x) for x in nm["shape"]])
def stream_raw(p: Obj, on: Callable[[int], object]) -> bytearray:
    # Just the network read (with byte progress). Decompression + reshape is deferred
    # (see decode_tensor) so the bar reaches 100% before the slow decode, not during it.
    body = p.client.get_object(Bucket=p.bucket, Key=p.key)["Body"]
    buf = bytearray()
    while True:
        c = body.read(DL)
        if not c: break
        buf += c; on(len(buf))
    return buf
def decode_tensor(buf: bytearray, p: Obj) -> torch.Tensor:
    raw = zstandard.decompress(bytes(buf)) if p.compressed else buf
    return torch.frombuffer(bytearray(raw), dtype=p.dtype).reshape(p.shape)
def read_whole(sd: Any, side: dict[str, Any], name: str, handle: Any) -> torch.Tensor:
    """One tensor as a CPU torch tensor, whichever kind of side it came from."""
    if side["kind"] == "safetensors":
        return sd.read(name)
    return handle.to("cpu")


total = len(PAIRS)
t0 = time.time()
read_bytes = 0
compared = 0
def work(idx: int) -> int:
    oname = PAIRS[idx][0]; nname = PAIRS[idx][1]
    stat("start\t%d\t%d\t%s" % (idx + 1, total, nname))
    res: dict[str, Any] = {"name": nname}
    tb = 0
    try:
        if oname not in old_sd:
            res["error"] = "not in the baseline"; emit(res); return 0
        if nname not in new_sd:
            res["error"] = "not in the candidate"; emit(res); return 0
        # Shapes from the header / the lazy handle — no tensor data read, so a mismatch costs nothing.
        a = None if OLD["kind"] == "safetensors" else old_sd[oname]
        b = None if NEW["kind"] == "safetensors" else new_sd[nname]
        ashape = old_sd.shape(oname) if a is None else [int(d) for d in a.shape]
        bshape = new_sd.shape(nname) if b is None else [int(d) for d in b.shape]
        if ashape != bshape:
            res["error"] = "shapes differ"; emit(res); return 0
        # Stream each side's S3 object on the proxy (chunked, with byte progress); the
        # per-chunk float64 conversions below work off the in-memory copy. Values are
        # compared here — only progress + the small result cross ssh. Fall back to
        # cstorch materialise (no byte progress) for a non-numpy-stored tensor.
        def comparing() -> None:
            # Bytes are all read (bar at 100%); decompress + decode is the slow part, so
            # flag the phase BEFORE it, not after.
            stat("phase\t%s\tcomparing" % (nname,))
        pa = prep(a) if a is not None else None
        pb = prep(b) if b is not None else None
        # Each branch decodes its own tensors, so `ra`/`rb` are bound on every path.
        if a is None or b is None:
            # At least one side is safetensors on this host: read each side whole (the s3 path does the
            # same — it downloads the whole object), then the span loop below streams within them. A
            # local read needs no byte bar of its own, so the sizes are announced once and completed.
            ra = read_whole(old_sd, OLD, oname, a)
            rb = read_whole(new_sd, NEW, nname, b)
            osz = int(ra.element_size() * ra.nelement()); nsz = int(rb.element_size() * rb.nelement())
            tb = osz + nsz
            stat("size\t%s\t%d\t%d" % (nname, osz, nsz))
            stat("bytes\t%s\t%d\t%d" % (nname, osz, nsz))
            comparing()
        elif pa is not None and pb is not None:
            osz = pa.size; nsz = pb.size
            stat("size\t%s\t%d\t%d" % (nname, osz, nsz))
            od = [0]; nd = [0]; last = [0]
            def bump() -> None:
                s = od[0] + nd[0]
                if s - last[0] >= EMIT:
                    last[0] = s; stat("bytes\t%s\t%d\t%d" % (nname, od[0], nd[0]))
            rawa = stream_raw(pa, lambda n: (od.__setitem__(0, n), bump()))
            rawb = stream_raw(pb, lambda n: (nd.__setitem__(0, n), bump()))
            stat("bytes\t%s\t%d\t%d" % (nname, osz, nsz))
            tb = osz + nsz
            comparing()
            ra = decode_tensor(rawa, pa); rb = decode_tensor(rawb, pb)
        else:
            ra = a.to("cpu"); rb = b.to("cpu")
            tb = ra.element_size() * ra.nelement() + rb.element_size() * rb.nelement()
            stat("size\t%s\t%d\t%d" % (nname, int(ra.element_size() * ra.nelement()), int(rb.element_size() * rb.nelement())))
            comparing()
        res["bytes"] = int(tb)
        scalar = (len(ashape) == 0)
        d0 = 0 if scalar else ashape[0]
        inner = 1
        for d in ashape[1:]: inner *= d
        rows = max(1, CHUNK // max(1, inner))
        spans = [(None, None)] if scalar else [(i, min(i + rows, d0)) for i in range(0, d0, rows)]
        if not spans: spans = [(None, None)]
        elements = 0; differing = 0; nfm = 0; max_abs = 0.0; sum_abs = 0.0
        # The bin range is taken from a cheap native-dtype scan (see span_range) so the
        # single float64 pass below can do BOTH the value diff and the binning. It used
        # to be two float64 passes over the whole tensor: one to diff and find the
        # range, another to bin — twice the widening and twice the memory traffic on
        # every `--values --histogram` run.
        n = BINS or 40
        lo_c = hi_c = 0.0
        oc = nc = None
        onf = nnf = 0
        if WANT_HIST:
            bounds = [r for rt in (ra, rb) for r in (span_range(rt, i, j) for (i, j) in spans) if r]
            if bounds:
                lo_c = min(b[0] for b in bounds); hi_c = max(b[1] for b in bounds)
            if hi_c <= lo_c: n = 1
            oc = np.zeros(n, dtype=np.int64); nc = np.zeros(n, dtype=np.int64)
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
                    max_abs = max(max_abs, m)
                    sum_abs += float(dd.sum())
                nfm += int(np.count_nonzero(dmask & ~bothfin))
            if WANT_HIST and oc is not None and nc is not None:
                afin = np.isfinite(av); bfin = np.isfinite(bv)
                onf += int(av.size) - int(np.count_nonzero(afin))
                nnf += int(bv.size) - int(np.count_nonzero(bfin))
                af = av[afin]; bf = bv[bfin]
                if hi_c <= lo_c:
                    oc[0] += int(af.size); nc[0] += int(bf.size)
                else:
                    hc, _ = np.histogram(af, bins=n, range=(lo_c, hi_c)); oc += hc.astype(np.int64)
                    hc, _ = np.histogram(bf, bins=n, range=(lo_c, hi_c)); nc += hc.astype(np.int64)
        if WANT_VALUES:
            mean_abs = (sum_abs / elements) if elements > 0 else 0.0
            res["values"] = {"elements": elements, "differing": differing, "max_abs": max_abs, "mean_abs": mean_abs, "nonfinite_mismatch": nfm}
        if WANT_HIST and oc is not None and nc is not None:
            ot = int(oc.sum()); nt = int(nc.sum())
            if ot == 0 and nt == 0: tvd = 0.0
            elif ot == 0 or nt == 0: tvd = 1.0
            else: tvd = 0.5 * float(np.abs(oc.astype(np.float64) / ot - nc.astype(np.float64) / nt).sum())
            h: dict[str, Any] = {"tvd": tvd, "n": int(n)}
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
