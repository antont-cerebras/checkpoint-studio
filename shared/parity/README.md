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
| which tensors a search query matches | `SkimMatcherV2` (smart case) | `search.ts: searchTree` |

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
