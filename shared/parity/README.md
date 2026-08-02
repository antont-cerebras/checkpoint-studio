# TUI ↔ web parity

The TUI renders in Rust; the web UI renders in the browser. A browser can't call into
`checkpoint-studio-core`, so a few display rules necessarily exist twice. Comments
saying "mirrors the TUI" do not stop two copies from drifting, and drift here is
silent — the same tensor simply reports a different size in the two UIs, and nobody
notices until someone compares screenshots.

`format.json` makes the agreement a test.

## How it works

- `tests/parity.rs` generates the fixture from the **Rust** implementations
  (`crates/core/src/utils.rs`) and asserts the committed file still matches them.
  Rust is the reference.
- `web/src/lib/parity.test.ts` reads the same file and asserts the **TypeScript**
  produces the same strings for the same inputs.

Change a rule on either side and one of the two tests fails, naming the case that
moved. Never hand-edit `format.json`; regenerate it:

```sh
UPDATE_PARITY=1 cargo test --test parity   # then run the web tests
cd web && npm test
```

## What is contracted

| rule | Rust | TypeScript |
| --- | --- | --- |
| byte sizes (`593.5 MiB`) | `utils::format_size` | `format.ts: humanSize` |
| parameter counts (`30.9B`) | `utils::format_parameters` | `format.ts: humanCount` |
| zero fraction (`0%` / `1.0e-7%` / `12.3%`) | `utils::format_percent` | `format.ts: percent` |
| a shard's file-browser row (`1062 tensors · 6.4% of params`) | `filetree::ShardTensors::note` | `format.ts: shardNote` |
| which tensors a search query matches | `SkimMatcherV2` (smart case) | `search.ts: searchTree` |
| the tree's **rows** — order, depth, kind, expandability | `TreeBuilder::flatten_tree` | `flatten.ts: flatten` + `expandedIds` |

The tree rows are the structural half of "the web UI looks like the TUI". The fixture
carries the tree as the server sends it (**rooted**, via `Session::build_rooted_tree` —
the same call both frontends make) plus the rows Rust flattens it into; the TypeScript
must flatten the same tree into the same rows. Two things this pins that used to be
wrong:

- **The initial fold state comes from the data.** Each group carries its own `expanded`
  flag, which the Rust flattener honors. The browser used to ignore it and seed only the
  root, so the same checkpoint opened with `model` expanded in the terminal and collapsed
  in the browser. `expandedIds` reads the flags now; the client owns folding only *after*
  load.
- **The fixture is rooted on purpose.** In the bare forest `model` sits at depth 0, so a
  client seeding only the top level would still produce the right rows — the test would
  have passed while the bug was live. Rooted, it fails (verified by mutation).

Two subtleties the fixture deliberately samples:

- **Rounding at exact ties.** Every size is a power-of-two division, so `1280 B` is
  *exactly* `1.25 KiB`. Rust's `{:.1}` rounds ties to even (`1.2`); JavaScript's
  `toFixed` rounds them away from zero (`1.3`). `humanSize` therefore has a `fixed1`
  helper that reproduces the Rust rule. `1280`, `1792`, `1310720`, `2952790016`,
  `1250` and `1750` are in the fixture for exactly this reason.
- **Smart case.** An all-lowercase query ignores case; a query with any uppercase is
  matched literally (so `norm` finds `LayerNorm`, `Norm` finds only the capitalised
  one). That is the `fuzzy-matcher` crate's default, which the TUI uses, and the web
  matcher now follows it. To make both case-insensitive instead, build the Rust
  matcher with `SkimMatcherV2::default().ignore_case()` and regenerate.

## What is deliberately NOT shared

These differ on purpose. If one of them starts looking like a bug, this is the list to
revisit — but each has a reason.

- **Numeric-grid cell precision.** The TUI prints `{:.4}` — fixed decimals, because a
  terminal grid has fixed-width columns and a ragged column is unreadable. The web
  uses `format.ts: num` (6 significant digits, exponential at the extremes) since an
  HTML table can size itself, and hovering a cell shows full precision anyway.
- **One-dimensional shape tuples.** The TUI shows `(2048)`; the web shows `(2048,)`,
  because its shape display doubles as the click-to-filter widget and matches its
  "copy shape (Python tuple)" button, where the trailing comma is what makes the text
  paste into Python. Unifying means changing `utils::format_shape` and regenerating
  the CLI snapshots.
- **Search result ranking.** Both matchers accept the same names (that *is*
  contracted), but they score differently: the TUI defers to `SkimMatcherV2`, while
  the web runs a small subsequence scorer that avoids a dependency and stays inside a
  few milliseconds per keystroke over 116k names. Only the order differs.
- **Histogram bin percentages.** A TUI-only readout; the web's histogram draws bars
  without per-bin percentages.

## `queryparams.json` — what the API accepts

Generated from the server's own allowlist (`src/web/mod.rs::accepted_params`, the table
`unknown_params` refuses against) and checked by `web/src/lib/queryparams.test.ts`, which drives the
real `api.*` calls through a stubbed `fetch` and asserts every key they put on a URL is in it.

    UPDATE_PARITY=1 cargo test the_accepted_parameters

This replaced a hand-copied list of the client's keys held in Rust. It was missing six parameters —
`align_fused`, `subtree`, `subtree_new`, `full`, `names_list`, `map_json` — and passed anyway, because
a stale copy of the client agrees with itself. A client parameter the server does not accept is a
`400` on a screen that used to work, so the check has to compare against what the client sends.

The scope half is checked in both directions: a parameter the server takes and no control produces is
called out too, unless it is on the test's short `deliberatelyUnsent` list (`names_list`, `map_json` —
accepted for a script to post, with no UI control today). That direction is how a server-only feature
gets noticed before it ships without a way to use it.
