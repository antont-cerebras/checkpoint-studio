# CLI ↔ web parity

**The goal: anything the CLI can do, the web UI can do — and the reverse.**

This file is the ledger for that. Every long flag clap defines appears below with what the web offers
instead, and `cli_web_parity` (in `src/parity_audit.rs`) fails the build when a flag is missing from
this table. So adding a CLI flag forces a decision about the web, recorded here, rather than a
divergence nobody wrote down.

That mechanism is deliberate, and copied from `tests/parity.rs`: comments saying "the web does this
too" do not stop the two from drifting, and drift here is silent — the CLI grows a flag, the web
quietly cannot do that thing, and nobody notices until someone tries.

Status key:

| | meaning |
|---|---|
| **yes** | the web can do this, by the means named |
| **part** | the web does some of it; the row says which half is missing |
| **gap** | the web cannot do this yet, and should be able to |
| **n/a** | meaningless in a browser (terminal rendering, exit codes, colour, one-shot stdout dumps) |

---

## Where the two surfaces stand

The web API has 23 routes: `tree` `files` `filter` `schema` `compact` `stats` `health` `check`
`layout` `model` `file` `tensor` `tensor/stats` `tensor/sample` `tensor/histogram` `diff` `difftree`
`reading` `version` plus `open` (POST), `compare` (POST/DELETE), `recents` (GET/DELETE) and the job starts
`jobs/values` / `jobs/verify-repack` (POST), each then polled and stopped by its id (GET/DELETE).

**A tab can tell it has gone stale.** `version` reports the hashed name of the entry script this
binary serves, and the browser compares it against the script it is running (`web/src/lib/build.ts`),
checked when it starts and whenever the tab comes back to the front. A tab outlives the server it was
loaded from — this project restarts the server under open tabs routinely — and the failure that
produces is silent: an old client reading a newer response shape once declared two checkpoints sharing
no tensor name "structurally identical", because every counter it looked for was missing and `NaN > 0`
is false.

**A mistyped parameter is refused, not ignored.** Each route declares what it accepts
(`web::accepted_params`) and anything else is a `400` naming it and listing the rest. `clap` refuses an
unknown `--flag`; the API is the same surface with the same typos available, and `?nmae=layers.1` used
to mean "compare all 117,664 tensors" under a heading saying the filter had been applied.

The headline gaps, in the order they matter:

1. ~~**`diff` scoping.**~~ Done, except `--tensor` (which needs values) and `--map` on the
   side-by-side. Both diff routes take `name` / `names` / `names_list` / `dtype_is` / `shape_is` /
   `only_tensors`, share the CLI's builders (`crate::compare::tensor_filter`, `name_map`) and its apply
   order, and report `matched M of N`. A differential test asserts the two surfaces produce the same
   report for the same scope, and the copyable command now carries it.
2. ~~**Value comparison.**~~ Done. `--verify-repack`, `--values`, `--histogram`, `--tensor`, `--dtype`
   and `--jobs` all run as polled jobs. `src/compare.rs` used to say these "stay on the CLI"; that is the
   decision this reversed.
3. **`convert`.** Repacking and in-place renaming have no web equivalent whatsoever. Both *write*, so
   they also raise a question the read-only API has never had to answer. **This is the only remaining
   headline gap** — `diff` itself is now at parity.

---

## `diff`

The subcommand this ledger was started for.

| flag | web | how / why not |
|---|---|---|
| `--recursive` | yes | `--recursive` at server start; `opening::Options.recursive` |
| `OLD#subtree` / `NEW#subtree` (operand syntax, not a flag) | yes | `subtree=` / `subtree_new=` on both diff routes, and a field per side in the scope bar's *Alignment* section. `CheckpointSummary::reroot` re-keys a side's tensors to their sub-path and `main::scope_tensors` does the same for the trees, so the two surfaces re-root identically; a prefix that selects nothing is a 400 naming its side, as the CLI's exit-2 message names it. The offered command puts it back on the operand, which is where `diff` takes it. Cross-checked against the CLI on a namespaced pair: `1 unchanged; 1 changed`, 24 B unchanged, both surfaces. Not available on the value / repack jobs, which read tensors by their real names — they refuse it rather than silently comparing the whole checkpoint |
| `--ssh-proxy` / `--ssh-venv` | yes | server start, or the config file; `/api/recents` reports the host |
| `--name` | yes | `name=` on `/api/diff` and `/api/difftree`, newline-separated (a repeated query key would collapse); `!` excludes. Reports `matched M of N` |
| `--names` / `--names-from` | yes | `names=` (comma-separated) and `names_list=` (the pasted content of what `--names-from` would read) |
| `--dtype-is` / `--shape-is` | yes | `dtype_is=` / `shape_is=`, the same globs |
| `--tensor` | yes | `…&tensor=NAME` on the values job; implies values and takes precedence over the scope, as it does on the command line. A name in neither checkpoint fails the job with that message |
| `--only-tensors` | yes | `only_tensors=1`. Note *any* filter also suppresses the metadata comparison — the CLI's rule (`DiffOpts { metadata: !only_tensors && !filtered }`), pinned by a test |
| `--map` / `--map-from` | yes | `map=` (newline-separated rules) and `map_json=`, on **both** views. The side-by-side rebuilds the baseline's tree from the renamed names (`difftree::tree_from_tensors`) — rewriting a leaf alone would leave the groups above it named from the name it used to have. Verified against the CLI: `-2 +2` becomes `2 unchanged` on both |
| `#SUBTREE` suffix | **gap** | accepted by the `diff` subcommand's `OLD`/`NEW`; `opening::resolve` does not take it |
| `--values` | yes | `POST /api/jobs/values?…&values=1`, polled — the **Data** view of a comparison, where the tensor count and the bytes to read are shown before the run starts. The summary used to say the numeric comparison had to be run in a terminal while the other screen was already running it in the browser; the terminal command is offered there as an alternative. Per-tensor findings from `compare::tensor_extras`, shared with the CLI, folded into the report by `diff::compare_with` — so a same-shape tensor whose bytes differ reads as *changed* on both. Numbers cross-checked against the CLI on the fixtures |
| `--histogram` / `--bins` | yes | `…&histogram=1&bins=N` on the same job |
| `--dtype` | yes | `…&dtype=V` on the values job, parsed by the same `view_of` every `/api/tensor/*` route uses |
| `--jobs` | yes | `…&jobs=N`; defaults to logical CPUs, as the CLI does |
| `--verify-repack` / `--repack-bits` | yes | `POST /api/jobs/verify-repack?left=…&right=…&<scope>[&repack_bits=N]`, polled. Candidate pairs, bit width and the "does anything else differ" question come from `compare::plan_repack`, shared with the CLI; decoding is `remote::verify_repack` on the proxy for an `s3://` pair, or the CLI's own `local_repack` for local files. Which pairs can run at all is `compare::data_where` — the **same** question a value comparison asks, so a safetensors directory on the proxy is verified against an `s3://` checkpoint rather than refused (it had its own rule wanting two `s3://` sides, while the value path was already reading both kinds on that host). Asked from the addresses **before** either side is read, in the browser; the CLI's operands are proxy-relative by then. Verified against a real pair: 2 pairs, 27 GB read on the remote, 318 s |
| `--repack-schema` / `--repack-schema-new` | yes | `repack_schema=` / `repack_schema_new=` on the verify-repack job and on `/api/command`, and a *Packing* row in the Data view with a field per side (`web/src/lib/packing.ts`, keyed off the generated parameter table). How each side packs its expert indices into a 16-bit word, as a list of bit widths: `[4]` is the sparse encoding (one index per word, four low bits used) and `[3,3,3,3,3]` is five consecutive experts merged into one word, each shifted three bits. **Said, because it is written nowhere**: inferring one width from the fold ratio describes a uniform merge and is silent about a sparse side, so a sparse-vs-merged pair verified as "not equivalent" with no way to say otherwise. One parser (`sample::PackingSchema::parse`) and one refusal (`compare::RepackSchemas`) for flag, query and field; the decoders read the pair from `compare::RepackPlan` — `repack::compare_packed` locally, the same widths as script parameters on the proxy. In the URL, so a link to a verification carries what it decoded with |
| `--align-fused` | yes | `align_fused=1` on both routes, and a checkbox on the scope bar. The rules are `diff::fused_layout_rules`, shared; the fold is `diff::OnCollision::Fold`, which merges several names onto one *counting the parts* instead of dropping all but one. The side-by-side folds its leaves too, labelled `×256` (`difftree::note_folds`, on the aligned rows — a fold belongs to the *side* that has the 256 tensors, and writing it into the tree's `label` cost a compacted row its name). Verified on the real pair: 80,107 against 933 and "nothing lines up" became 17 rows with `×256 → ×1` |
| `--full` (side-by-side) | yes | `full=1` on `/api/difftree`, `Collapse families` in the compare controls, `k` in both surfaces. Folded by default, like the report. `difftree::fold_families` folds sibling subtrees whose name starts with an index and whose alignment is identical on both sides — so 62 layers that changed the same way are one `{0-61}` row (`×62`) and the layer with an extra tensor stands beside them. Bottom-up, so a layer's experts fold before the layers are compared; the label is `diff::summarize_indices`, the report's own `{0-2,5}` wording. The tally is taken **before** folding, so a view control cannot change what the comparison says |
| `--full` (report) | yes | `full=1`, and the web defaults to *collapsed*, like the CLI: `Collapse families` in the report's controls. The grouping is `DiffReport::grouped` — the same `group_entries` / `group_changed` the terminal renders, so the two cannot collapse differently (a test pins the grouped rows to the rendered lines). Both lists are sent, so switching costs no round trip; the offered command gains `--full` when expanded. The real pair: 809 differing tensors read as 16 rows |
| `size:` / `params:` header | yes | delta and percentage included, from `diff::totals_line` — one implementation, pinned across languages by `shared/parity/format.json` (the `{:.1}` tie rule is where two languages disagree). Both views; a test pins the side-by-side's totals to the report's |
| `scope:` / `legend:` header | yes | both views. The side-by-side is nothing but `-`/`+`/`~` marks and had no key |
| `metadata: not compared (…)` | yes | `metadata_note` on `/api/diff`, from the server, which owns the rule. The browser used to print `Metadata (0)` — which says nothing differed, not that nothing was compared |
| `S3 objects:` section | yes | `S3Diff::summary_lines` is shared, so the ETag-confidence wording exists once; `compare::attach_s3` performs the comparison for both surfaces. Verified byte-identical against the CLI on a real s3-vs-s3 pair (1155 objects, 361 changed, grouped into 11 lines) |
| `modified:` line | yes | `DiffReport::modified_line`, humanised server-side rather than mirrored in TypeScript for one line |
| read progress (`1155/1155 S3 objects`) | yes | `GET /api/reading`, polled while a wait is on screen (`web/src/stores/reading.ts`). `hf::ReadProgress` wraps the `progress::LoadProgress` every reader already fills, so the counts reach whoever is waiting instead of a terminal bar on the server's log |
| swap the two sides | yes | `?swap=1`, in the URL like every other piece of view state. The CLI swaps by reordering its two operands; a test asserts the mirror, and the copyable command turns round with it |
| `--no-color` | n/a | CSS |
| exit codes 0/1/2 | n/a | HTTP status |

## The job subsystem

Built, reachable from the UI (`JobPanel`, beside the scope bar), and running every value mode:

```text
POST   /api/jobs/verify-repack?left=…&right=…&<scope>   -> { "id": 7 }
GET    /api/jobs/7                                      -> state, progress, findings so far
DELETE /api/jobs/7                                      -> ask it to stop
```

**Polling, not streaming.** `tiny_http` is synchronous and thread-per-request, so a held-open response
costs a worker for the whole run, and an SSE stream through an intervening proxy is commonly buffered or
timed out. Polling costs a request every half second, and the job outlives the tab — so a reload picks
the run back up rather than losing it.

**A registry of jobs**, each holding: `state` (`running` / `done` / `cancelled` / `failed` — four
states, so a cancelled run is not reported as a failure), atomics for `done` / `total` / `bytes`, the
item being worked on, and findings appended *as they land* (the first finding is often the answer, and
waiting minutes for it is what this design avoids). Finished jobs are kept so a late poll still gets
results, capped so a long-lived browser cannot accumulate them, and a *running* job is never evicted —
that would make it unpollable and unstoppable.

**Cancellation already exists.** A job carries the [`crate::hf::ReadProgress`] its work was handed, so
`DELETE` sets the flag the remote reader checks between chunks — the same mechanism as "stop the read
that is blocking you". `RepackResult` and friends are `Serialize` for this.

`src/web/repackjob.rs` and `src/web/valuesjob.rs` are the two workers; `web/src/stores/jobs.ts` polls and
`web/src/components/JobPanel.svelte` is the UI. Two things it establishes for the next ones:

- **Decide the mode before planning.** The CLI refuses a remote *safetensors* directory outright; planning
  first reported "no fold-pair tensors matched" instead — true, but about the wrong problem. Found by
  running one command through both surfaces and comparing what each said.
- **Accumulate progress.** The remote counts bytes per *tensor*, restarting at zero, so a job that stores
  the reading reports 27 GB and then 0.01 GB. `ByteTally` folds them into a run total that cannot regress.

## `check`

| flag | web | how / why not |
|---|---|---|
| `--recursive`, `--ssh-proxy`, `--ssh-venv` | yes | server start |
| `--values` | **gap** | value-scanning health checks; needs the job subsystem |
| `--name` | **gap** | scope the checks to matching tensors |
| `--jobs` | **gap** | parallelism for the value scan |
| `--strict` | **gap** | promotes warnings to failures — a query parameter on `/api/check` |
| `--format` | n/a | `/api/check` is JSON already |
| `--no-color` | n/a | CSS |

## `convert`

No web equivalent at all. Every flag is a gap, and this is the one subcommand that **writes**.

| flag | web | how / why not |
|---|---|---|
| `--codec`, `--level`, `--buffer` | **gap** | HDF5 repack settings |
| `--map`, `--map-from` | **gap** | in-place tensor rename |
| `--force` | **gap** | overwrite confirmation |

## `web`

| flag | web | how / why not |
|---|---|---|
| `--host`, `--port` | n/a | this *is* the web server's own configuration |
| `--recursive`, `--no-health-check`, `--ssh-proxy`, `--ssh-venv` | yes | they configure the server that serves the UI |

## Browsing (the root command)

Most of these open a TUI screen; the web equivalent is a URL. Listed for completeness because the
ledger's guard covers every flag clap defines.

| flag | web | how / why not |
|---|---|---|
| `--recursive`, `--no-health-check`, `--ssh-proxy`, `--ssh-venv` | yes | server start |
| `--tensor`, `--metadata` | yes | `#detail?tensor=…`, `#tree` with the entry selected |
| `--dtype`, `--values`, `--heatmap`, `--histogram`, `--bins`, `--slice`, `--window`, `--edge` | yes | the detail screen's tabs and their hash parameters (`/api/tensor/histogram` takes `bins`) |
| `--stats`, `--health`, `--files`, `--layout`, `--layout-select`, `--tree`, `--compact`, `--filter`, `--search`, `--sort`, `--overview`, `--abs-max`, `--zebra` | yes | the corresponding screen / hash parameter |
| `--base`, `--shape`, `--name`, `--rename`, `--rename-rule` | **gap** | `--rename` / `--rename-rule` write; `--base` and `--shape` reinterpret a tensor |
| `--diff-against` | yes | `#compare?lhs=…` — the summary view (`lhs`/`rhs` are the pair, in the order `diff OLD NEW` takes them) |
| `--compare-with`, `--compare-full` | yes | `#compare?lhs=…&rhs=…&view=browse[&full=1]`. **The browser has one comparison screen and three views of it** (`view=summary\|browse\|data`); the terminal keeps two screens, because there the choice is a keypress rather than a fork in the road — `d` opens the report, the palette opens the side-by-side, and neither hides the other. `--compare-full` carries the family fold state (`k`), the way `&full=1` does |
| `--compute-stats`, `--stats-shards`, `--health-findings`, `--print-arch` | **gap** | one-shot exports with no web equivalent |
| `--no-preload` | **gap** | the web never preloads, so the *default* differs rather than the flag being absent |
| `--tree-state`, `--emit-command`, `--print-view` | n/a | the URL hash is the web's state round-trip (`y` in the TUI) |
| `--print-tree`, `--print-tensors`, `--print-model` | n/a | `/api/tree`, `/api/model` |
| `--plain`, `--legend`, `--verbose`, `--exit`, `--format` | n/a | terminal-only |
| `--help`, `--version` | n/a | — |
