"""Run the embedded remote scripts locally, against fakes, and check what they print.

The scripts in `crates/core/src/python/` execute on the ssh proxy inside the cstorch
venv, so nothing here can import what they import — `cerebras.pytorch`, `boto3`, torch.
Until now that meant their *control flow* had no test at all: `remote.rs`'s replay tests
check that recorded output parses, which pins the protocol but never runs a line of
Python. A typo in a branch that only fires on a big checkpoint would surface minutes
into a user's job.

So: stub the imports in `sys.modules`, exec the real script text with real `__PARAMS__`,
and assert on the sentinel-tagged lines it writes. The fakes are deliberately dumb —
they only have to be shaped enough for the script's own logic to run.

`dump.py` and `list_objects.py` are covered here: their work is protocol and
orchestration (progress phases, the parallel S3 metadata pass, pagination, error
reporting), which fakes model faithfully. `value_diff.py` and `repack_verify.py` are
NOT: their bodies are numpy/torch array arithmetic, and a fake numpy convincing enough
to test them would itself be the thing most likely to be wrong. Those keep their
coverage from the recorded-transcript replay tests in `remote.rs`.

Run: `python3 -m unittest discover -s crates/core/tests/python`
"""

from __future__ import annotations

import io
import json
import re
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable

SCRIPTS = Path(__file__).resolve().parents[2] / "src" / "python"
# The real tags `remote.rs` uses (SENTINEL / PROGRESS_TAG there) — the scripts take
# them as parameters, so using the shipping values keeps the fixture honest.
SENTINEL = "CKPT_EXPLORER_META:"
PROGRESS = "CKPT_EXPLORER_PROG:"


# --------------------------------------------------------------------- fakes ----
class FakeTensor:
    """Just enough of a torch tensor for the metadata loop: dtype, shape, itemsize."""

    def __init__(self, dtype: str, shape: tuple[int, ...], itemsize: int) -> None:
        self.dtype = dtype
        self.shape = shape
        self._itemsize = itemsize

    def element_size(self) -> int:
        return self._itemsize


class FakeS3Client:
    """A boto3 S3 client over an in-memory bucket.

    `pages` forces `list_objects_v2` to paginate, so the continuation-token loop is
    actually exercised rather than assumed. `deny_tags` makes every tagging call fail,
    which is how the real bucket behaves without `s3:GetObjectTagging`.
    """

    def __init__(
        self,
        objects: dict[str, int],
        *,
        pages: int = 1,
        deny_tags: bool = False,
        head_fails: tuple[str, ...] = (),
        user_meta: dict[str, str] | None = None,
    ) -> None:
        self.objects = objects
        self.pages = pages
        self.deny_tags = deny_tags
        self.head_fails = head_fails
        self.user_meta = user_meta or {}
        self.heads = 0
        self.taggings = 0

    def list_objects_v2(self, **kw: Any) -> dict[str, Any]:
        keys = sorted(self.objects)
        per = max(1, len(keys) // self.pages)
        start = int(kw.get("ContinuationToken") or 0)
        chunk = keys[start : start + per]
        end = start + len(chunk)
        return {
            "Contents": [{"Key": k, "Size": self.objects[k]} for k in chunk],
            "IsTruncated": end < len(keys),
            "NextContinuationToken": str(end),
        }

    def head_object(self, **kw: Any) -> dict[str, Any]:
        self.heads += 1
        key = kw["Key"]
        if key in self.head_fails:
            raise RuntimeError("access denied")
        return {
            "ContentLength": self.objects[key],
            "ETag": '"etag-%s"' % key,
            "LastModified": FakeDate(),
            "Metadata": dict(self.user_meta),
            "ChecksumSHA256": "sum-%s" % key,
        }

    def get_object_tagging(self, **_kw: Any) -> dict[str, Any]:
        self.taggings += 1
        if self.deny_tags:
            raise RuntimeError("no s3:GetObjectTagging")
        return {"TagSet": [{"Key": "owner", "Value": "team"}]}


class FakeDate:
    def isoformat(self) -> str:
        return "2026-06-26T10:00:00+00:00"


class FakeModule(types.ModuleType):
    """A stand-in module whose attributes are supplied at construction.

    `types.ModuleType` declares no attributes of its own, so `mod.client = ...` is an
    error to every type checker — and each wants a differently-spelled suppression
    comment (ty does not honour mypy's `# type: ignore[attr-defined]`). Filling
    `__dict__` is what the import system reads anyway, so this needs no suppression at
    all, and it keeps each fake below to one expression.

    Setting `__getattr__` through it works too: PEP 562 looks that name up in the
    module's `__dict__`, which is how `test_an_import_failure_...` makes every attribute
    raise.
    """

    def __init__(self, name: str, **attrs: Any) -> None:
        super().__init__(name)
        self.__dict__.update(attrs)


def fake_boto3(client: FakeS3Client) -> types.ModuleType:
    def make_client(*_a: Any, **_k: Any) -> FakeS3Client:
        return client

    return FakeModule("boto3", client=make_client)


def fake_botocore() -> tuple[types.ModuleType, types.ModuleType]:
    def config_ctor(**_k: Any) -> None:
        return None

    config = FakeModule("botocore.config", Config=config_ctor)
    return FakeModule("botocore", config=config), config


def fake_cstorch(state: dict[str, Any] | None, *, fail: bool = False) -> types.ModuleType:
    def load(_src: str, **_kw: Any) -> dict[str, Any]:
        if fail:
            raise RuntimeError("dill barfed")
        assert state is not None
        return state

    return FakeModule("cerebras", pytorch=FakeModule("cerebras.pytorch", load=load))


# ------------------------------------------------------------------- harness ----
def run_script(
    name: str,
    params: dict[str, Any],
    modules: dict[str, types.ModuleType],
    *,
    tweak: Callable[[str], str] | None = None,
) -> list[str]:
    """Exec one script with `params` substituted and `modules` stubbed.

    Returns its stdout lines. Mirrors `remote.rs::with_params`, so the substitution
    under test is the same one that ships.
    """
    path = SCRIPTS / name
    text = path.read_text()
    if tweak:
        text = tweak(text)
    literal = json.dumps(json.dumps(params))
    text = text.replace("__PARAMS__", literal[1:-1])

    saved = {k: sys.modules.get(k) for k in modules}
    sys.modules.update(modules)
    out = io.StringIO()
    try:
        with redirect_stdout(out):
            # Compile under the script's real path: tracebacks point at the file being
            # tested, and coverage.py attributes the executed lines to it rather than to
            # a phantom module.
            exec(compile(text, str(path), "exec"), {"__name__": "__main__"})
    except SystemExit:
        pass  # the scripts exit(0) after reporting an error
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return out.getvalue().splitlines()


def meta(lines: list[str]) -> dict[str, Any]:
    """Return the last sentinel-tagged JSON line — the result the Rust side parses."""
    payload = [line for line in lines if line.startswith(SENTINEL)]
    assert payload, "no sentinel line in:\n%s" % "\n".join(lines)
    return json.loads(payload[-1][len(SENTINEL) :])


def progress(lines: list[str]) -> list[str]:
    return [line[len(PROGRESS) :] for line in lines if line.startswith(PROGRESS)]


DUMP_PARAMS = {
    "uri": "s3://bucket/ckpt",
    "want_s3": False,
    "sentinel": SENTINEL,
    "progress": PROGRESS,
}
STATE = {
    "a.weight": FakeTensor("torch.float16", (4, 8), 2),
    "b.weight": FakeTensor("torch.float32", (16,), 4),
    "not_a_tensor": "a string the state dict happens to hold",
}


class DumpScript(unittest.TestCase):
    def test_emits_every_tensor_with_its_dtype_shape_and_itemsize(self) -> None:
        lines = run_script(
            "dump.py",
            DUMP_PARAMS,
            {"cerebras": fake_cstorch(STATE), "cerebras.pytorch": fake_cstorch(STATE).pytorch},
        )
        result = meta(lines)
        by_name = {t["name"]: t for t in result["tensors"]}
        self.assertEqual(by_name["a.weight"]["dtype"], "torch.float16")
        self.assertEqual(by_name["a.weight"]["shape"], [4, 8])
        self.assertEqual(by_name["a.weight"]["itemsize"], 2)
        self.assertEqual(by_name["b.weight"]["shape"], [16])
        # A non-tensor entry is skipped rather than aborting the whole dump.
        self.assertNotIn("not_a_tensor", by_name)
        self.assertEqual(result["s3_objects"], [])

    def test_progress_starts_determinate_and_ends_at_the_total(self) -> None:
        lines = run_script(
            "dump.py",
            DUMP_PARAMS,
            {"cerebras": fake_cstorch(STATE), "cerebras.pytorch": fake_cstorch(STATE).pytorch},
        )
        steps = progress(lines)
        self.assertEqual(steps[0], "0/3", "the total is known up front, so the bar is determinate")
        self.assertEqual(steps[-1], "3/3")

    def test_a_load_failure_is_reported_as_json_with_an_s3_probe(self) -> None:
        client = FakeS3Client({"ckpt/__METADATA__": 0, "ckpt/a.weight": 12})
        botocore, config = fake_botocore()
        failing = fake_cstorch(None, fail=True)
        lines = run_script(
            "dump.py",
            DUMP_PARAMS,
            {
                "cerebras": failing,
                "cerebras.pytorch": failing.pytorch,
                "boto3": fake_boto3(client),
                "botocore": botocore,
                "botocore.config": config,
            },
        )
        result = meta(lines)
        self.assertIn("cstorch.load failed", result["error"])
        # The probe explains *why* rather than leaving only a dill traceback: it
        # notices the empty metadata object.
        probe = result["s3_probe"]
        self.assertEqual(probe["total"], 2)
        self.assertEqual(probe["empty"], 1)
        self.assertEqual(probe["metadata_key"], "__METADATA__")
        self.assertTrue(probe["metadata_empty"])

    def test_an_import_failure_names_the_missing_package(self) -> None:
        def explode(_name: str, *_a: Any, **_k: Any) -> Any:
            raise ImportError("No module named cerebras.pytorch")

        broken = FakeModule("cerebras", __getattr__=explode)
        lines = run_script("dump.py", DUMP_PARAMS, {"cerebras": broken})
        self.assertIn("import cerebras.pytorch failed", meta(lines)["error"])


class DumpScriptS3Phase(unittest.TestCase):
    """`want_s3`: the per-object metadata pass.

    That pass is where the parallelism and the tag early-stop live.
    """

    def objects(self) -> dict[str, int]:
        return {"ckpt/__METADATA__": 99, "ckpt/a.weight": 64, "ckpt/b.weight": 64}

    def run_with(self, client: FakeS3Client) -> tuple[dict[str, Any], list[str], FakeS3Client]:
        botocore, config = fake_botocore()
        cstorch = fake_cstorch(STATE)
        lines = run_script(
            "dump.py",
            {**DUMP_PARAMS, "want_s3": True},
            {
                "cerebras": cstorch,
                "cerebras.pytorch": cstorch.pytorch,
                "boto3": fake_boto3(client),
                "botocore": botocore,
                "botocore.config": config,
            },
        )
        return meta(lines), lines, client

    def test_reports_each_object_with_prefix_relative_keys(self) -> None:
        result, lines, client = self.run_with(FakeS3Client(self.objects()))
        keys = [o["key"] for o in result["s3_objects"]]
        self.assertEqual(keys, ["__METADATA__", "a.weight", "b.weight"], "keys are relative")
        first = result["s3_objects"][1]
        self.assertEqual(first["size"], 64)
        self.assertEqual(first["etag"], "etag-ckpt/a.weight", "quotes stripped")
        self.assertEqual(first["checksum"], ["sha256", "sum-ckpt/a.weight"])
        self.assertEqual(first["tags"], {"owner": "team"})
        self.assertEqual(client.heads, 3, "one HEAD per object")
        # The bar switches to the s3 phase and finishes there.
        self.assertTrue(any(s.endswith("/s3") for s in progress(lines)))
        self.assertEqual(progress(lines)[-1], "3/3/s3")

    def test_pagination_is_followed_to_the_last_page(self) -> None:
        result, _lines, _client = self.run_with(FakeS3Client(self.objects(), pages=3))
        self.assertEqual(len(result["s3_objects"]), 3, "all pages, not just the first")

    def test_a_denied_tagging_call_warns_once_and_then_stops_asking(self) -> None:
        result, _lines, client = self.run_with(FakeS3Client(self.objects(), deny_tags=True))
        warnings = [w for w in result["s3_warnings"] if "GetObjectTagging" in w]
        self.assertEqual(len(warnings), 1, "one warning, not one per object")
        self.assertLessEqual(
            client.taggings, 2, "stops asking after the first denial (%d calls)" % client.taggings
        )
        # Objects still come back — only their tags are missing, which the reader reads
        # as "not available" rather than "no tags".
        self.assertEqual(len(result["s3_objects"]), 3)
        self.assertNotIn("tags", result["s3_objects"][0])

    def test_a_failed_head_drops_that_object_and_warns(self) -> None:
        client = FakeS3Client(self.objects(), head_fails=("ckpt/b.weight",))
        result, _lines, _client = self.run_with(client)
        self.assertEqual([o["key"] for o in result["s3_objects"]], ["__METADATA__", "a.weight"])
        self.assertTrue(any("head_object failed" in w for w in result["s3_warnings"]))

    def test_user_metadata_is_carried_through_verbatim(self) -> None:
        # This is what the index-vs-object health cross-check reads.
        claim = '{"shapes": [[4, 8]], "dtypes": ["torch.float16"]}'
        client = FakeS3Client(self.objects(), user_meta={"metadata": claim})
        result, _lines, _client = self.run_with(client)
        self.assertEqual(result["s3_objects"][1]["metadata"], {"metadata": claim})


class ListObjectsScript(unittest.TestCase):
    def test_lists_every_page_with_prefix_relative_keys_and_sizes(self) -> None:
        client = FakeS3Client(
            {"ckpt/a.weight": 10, "ckpt/sub/b.weight": 20, "ckpt/__METADATA__": 30}, pages=2
        )
        lines = run_script(
            "list_objects.py",
            {"uri": "s3://bucket/ckpt", "sentinel": SENTINEL},
            {"boto3": fake_boto3(client)},
        )
        objects = meta(lines)["objects"]
        self.assertEqual(
            sorted(objects), [["__METADATA__", 30], ["a.weight", 10], ["sub/b.weight", 20]]
        )
        self.assertEqual(client.heads, 0, "listing must not HEAD anything")

    def test_a_boto3_failure_is_reported_as_json(self) -> None:
        def explode(*_a: Any, **_k: Any) -> Any:
            raise RuntimeError("no credentials")

        broken = FakeModule("boto3", client=explode)
        lines = run_script(
            "list_objects.py",
            {"uri": "s3://bucket/ckpt", "sentinel": SENTINEL},
            {"boto3": broken},
        )
        self.assertIn("no credentials", meta(lines)["error"])


class ParamSubstitution(unittest.TestCase):
    """`remote.rs` injects parameters through one JSON slot.

    A URI carrying a quote or a backslash must not be able to break out of it — the
    scripts are built by string substitution, so this is the one place an injection
    could happen.
    """

    def test_a_hostile_uri_survives_the_json_slot(self) -> None:
        nasty = 's3://bucket/it\'s "quoted" \\ and #hashed'
        client = FakeS3Client({})
        lines = run_script(
            "list_objects.py",
            {"uri": nasty, "sentinel": SENTINEL},
            {"boto3": fake_boto3(client)},
        )
        # It parsed and ran: an empty listing, not a SyntaxError.
        self.assertEqual(meta(lines)["objects"], [])

    def test_the_scripts_have_exactly_one_substitution_slot(self) -> None:
        # The slot must appear once, inside a JSON string literal — not in prose (the
        # scripts' comments mention it) and not twice, which would break the second
        # copy's quoting.
        for path in sorted(SCRIPTS.glob("*.py")):
            slots = len(re.findall(r'json\.loads\("__PARAMS__"\)', path.read_text()))
            # The prelude takes no parameters: it's spliced into scripts that do.
            expected = 0 if path.name == "cstorch_fast.py" else 1
            self.assertEqual(slots, expected, "%s has %d slots" % (path.name, slots))


class CstorchFastPrelude(unittest.TestCase):
    """The S3 fast-path prelude that stops cstorch re-HEADing one object per tensor.

    1155 HEADs of the same key, 4.97s of a 7.3s open — see the module's own docstring.

    It runs on every `cstorch.load` script, and it is written to fail *silently* if
    cstorch's internals move, so a mistake here is invisible: the read stays correct and
    quietly slow. That makes its branches worth testing more than most.
    """

    @staticmethod
    def _run(s3_module: types.ModuleType | None) -> None:
        """Exec the real prelude text with `cerebras…s3_storage` stubbed (or absent)."""
        names = [
            "cerebras",
            "cerebras.appliance",
            "cerebras.appliance.storage",
            "cerebras.appliance.storage.s3_storage",
        ]
        saved = {k: sys.modules.get(k) for k in names}
        try:
            if s3_module is None:
                # Present but empty, so `from … import S3Reader` raises ImportError and
                # the prelude takes its give-up path. Deterministic in a way that merely
                # deleting the keys is not: that would depend on cstorch being absent
                # from the machine, which is true here and not a property of the test.
                for k in names:
                    sys.modules[k] = FakeModule(k)
            else:
                for k in names[:-1]:
                    sys.modules.setdefault(k, FakeModule(k))
                sys.modules[names[-1]] = s3_module
            # Compile under the script's real path, like `run_script` does: coverage.py
            # attributes executed lines by filename, so a bare name lands on a phantom
            # module and the file still reports 0%.
            script = SCRIPTS / "cstorch_fast.py"
            exec(compile(script.read_text(), str(script), "exec"), {"__name__": "__main__"})
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    @staticmethod
    def _module_with(reader: type) -> types.ModuleType:
        return FakeModule("cerebras.appliance.storage.s3_storage", S3Reader=reader)

    def test_it_memoizes_stats_per_reader_path(self) -> None:
        calls = []

        class S3Reader:
            def __init__(self, path: str) -> None:
                self.path = path

            @property
            def stats(self) -> str:
                calls.append(self.path)
                return "stat:" + self.path

        self._run(self._module_with(S3Reader))

        a, b = S3Reader("s3://b/one"), S3Reader("s3://b/two")
        # Repeated access on the same path must hit the real getter exactly once — that
        # is the entire point, since each real call is an HTTPS round trip.
        self.assertEqual([a.stats, a.stats, a.stats], ["stat:s3://b/one"] * 3)
        self.assertEqual(calls, ["s3://b/one"])
        # A different path is a different entry, not a stale hit from the first.
        self.assertEqual(b.stats, "stat:s3://b/two")
        self.assertEqual(calls, ["s3://b/one", "s3://b/two"])
        # A second reader on the same path shares the memo (the key is the path).
        self.assertEqual(S3Reader("s3://b/one").stats, "stat:s3://b/one")
        self.assertEqual(len(calls), 2)

    def test_a_reader_without_a_path_is_cached_per_instance(self) -> None:
        # The key falls back to `id(self)`, so two pathless readers must not collide.
        calls = []

        class S3Reader:
            @property
            def stats(self) -> int:
                calls.append(1)
                return len(calls)

        self._run(self._module_with(S3Reader))
        x, y = S3Reader(), S3Reader()
        self.assertEqual((x.stats, x.stats), (1, 1), "memoized per instance")
        self.assertEqual(y.stats, 2, "a different instance is not a cache hit")

    def test_it_marks_the_class_so_a_second_splice_does_not_double_wrap(self) -> None:
        # The prelude is prepended to every script, and a session may exec more than one.
        # Wrapping the wrapper would memoize the memo and never see a real refresh.
        calls = []

        class S3Reader:
            def __init__(self) -> None:
                self.path = "p"

            @property
            def stats(self) -> int:
                calls.append(1)
                return 1

        mod = self._module_with(S3Reader)
        self._run(mod)
        first = S3Reader.stats
        self._run(mod)
        self.assertIs(S3Reader.stats, first, "the second run is a no-op")
        self.assertTrue(getattr(S3Reader, "_ckpt_studio_memoized", False))

    def test_it_leaves_an_unrecognised_shape_alone(self) -> None:
        # Best-effort by design: if `stats` is not a property whose getter we can call,
        # the correct-but-slow original has to survive untouched.
        class PlainAttr:
            stats = "not a property"

        def discard(*_: object) -> None:
            """Ignore every assignment — this property is a setter with no getter."""

        class NoGetter:
            # `fget is None`, so there is nothing for the prelude to memoize.
            stats = property(None, discard)

        class Missing:
            pass

        for cls, name in ((PlainAttr, "plain"), (NoGetter, "setter-only"), (Missing, "absent")):
            before = getattr(cls, "stats", "<absent>")
            self._run(self._module_with(cls))
            self.assertEqual(getattr(cls, "stats", "<absent>"), before, "%s was patched" % name)
            self.assertFalse(getattr(cls, "_ckpt_studio_memoized", False), name)

    def test_it_is_a_no_op_when_cstorch_is_not_installed(self) -> None:
        # Every local run takes this path, and it must not raise: the prelude is spliced
        # into scripts that would otherwise work.
        self._run(None)


if __name__ == "__main__":
    unittest.main()
