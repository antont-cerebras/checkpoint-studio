# Type stubs for the cluster-side packages

The scripts in `crates/core/src/python/` run on the ssh proxy inside the Cerebras
`cstorch` venv. Nothing here can install `cerebras.pytorch`, `torch`, `boto3` or
`zstandard` — torch alone is a multi-GB download, and cstorch isn't public — so every
value coming out of them was `Unknown`: pyright `strict` reported 226 errors, all of them
`reportUnknown*` noise about the absence of stubs rather than anything about our code, and
`ty` reported nothing at all because an unresolved import silences it entirely.

These stubs fix that, and they earn their keep twice over:

1. **They make type checking real.** With the imports resolving, `strict` has something to
   check, and a typo in a tensor method is a failed check rather than an `AttributeError`
   minutes into someone's cluster job. Measured, not assumed: mutating `torch.isfinite` →
   `torch.isfinit`, `list_objects_v2` → `list_objects_v3`, `Bucket=` → `Buckets=`,
   `zstandard.decompress` → `.decompres`, `Tensor.min` → `.minimum` and `.reshape` →
   `.reshapes` are all caught now, and not one of them was before. (A seventh probe,
   `np.uint16` → `np.uint17`, was already caught — numpy was the one import that resolved.)
2. **They are the contract.** They declare *exactly* the API surface these scripts depend
   on, and nothing more. That list is otherwise implicit and spread across five files, so
   there was no way to see what a cstorch or boto3 upgrade could break. If a future version
   changes one of these signatures, this directory is the checklist.

They are deliberately **partial**: only what the scripts call, read off the call sites
rather than copied from upstream. `Any` appears where a real type would be a guess — an
honest `Any` beats an invented signature that type-checks against something the cluster
does not actually do.

One entry is not public API: `cerebras/appliance/storage/s3_storage.pyi` describes the
internal `S3Reader` whose `stats` property `cstorch_fast.py` monkey-patches (4.97s of
redundant `head_object` calls → 0.09s on a 1155-tensor open). It is the most likely thing
here to break on a cstorch upgrade, which is why that patch fails soft.

## What is deliberately NOT here

**numpy.** It's pip-installable, small, and ships `py.typed` with precise inline types —
strictly better than anything hand-written, so it is installed instead of stubbed (CI does
`pip install "numpy<2"`; this box has 1.23.5). The `<2` bound matches the 1.x line the
proxy's cstorch venv is built on, so the types we check against are the types the scripts
actually meet. Bump it when the proxy does, deliberately. Its generic `ndarray` is why the
scripts annotate `np.ndarray[Any, Any]`.

**boto3, by contrast, IS stubbed even though it's pip-installable**, because its client
methods are generated at runtime from botocore's JSON service model — the real package
gives a type checker nothing to work with. The stub here is more precise than the genuine
article, and it doubles as the audit of what these scripts do to S3: three reads and a tag
read, no writes.

## Wiring

Two checkers, two mechanisms, same directory:

- pyright — `"stubPath": "typings"` in `pyrightconfig.json`.
- ty — `[tool.ty.environment] extra-paths = ["typings"]` in `pyproject.toml`; ty has no
  stub-path concept, so the stubs go on the module search path instead.

Ruff lints these files too (`typings` is in pyright's `include`, and ruff's `select = ["ALL"]`
turns on flake8-pyi, which exists for exactly this kind of hand-written stub). Two
consequences worth knowing before you edit one:

- **Notes go in `#` comments, not docstrings.** A docstring in a stub is a lint (PYI021):
  stubs carry types, and prose about *why a stub exists* is a note to the next editor.
- **Names are not ours to fix.** boto3's keyword arguments really are PascalCase
  (`Bucket`, `Key`, `MaxKeys`) and torch really does have parameters that shadow builtins,
  so `N801`/`N802`/`N803`/`A002`/`A003` are turned off for `typings/**/*.pyi` alone. A stub
  that renamed them would describe a library that doesn't exist.

## Checking a stub against reality

    ssh <proxy> 'source ~/venv/bin/activate && python3 -c "
    import cerebras.pytorch as cstorch, inspect; print(inspect.signature(cstorch.load))"'
