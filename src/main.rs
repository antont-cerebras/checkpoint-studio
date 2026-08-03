// An unwrap in a test IS the assertion — the panic is the failure report, and rewriting
// hundreds of them into `?` would make the tests harder to read for no gain. So
// `unwrap_used`/`expect_used` (denied for product code in Cargo.toml) are allowed in test
// builds only.
#![cfg_attr(
    test,
    allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)
)]

// This tool memory-maps multi-gigabyte checkpoints and converts 64-bit header offsets
// and element counts to `usize` throughout. That is only sound on a 64-bit target, so
// state it as a compile-time requirement instead of leaving it implied.
const _: () = assert!(
    usize::BITS >= 64,
    "checkpoint-studio requires a 64-bit target: file offsets and element counts are \
     converted to usize"
);

// The frontend-free core modules live in `checkpoint-studio-core`. Re-export
// them at the crate root so the (still bin-side) `explorer`/`ui` keep resolving
// their `crate::tree::…` / `crate::stats::…` paths unchanged during the refactor.
pub use checkpoint_studio_core::{
    arch, capability, check, codec, compact, config, diff, difftree, filetree, filter, gguf,
    health, hf, kernel, model, npy, progress, readers, remote, rename, repack, s3, safelayout,
    sample, sftp, stats, stheader, tensorfilter, tree, utils, viewstate,
};
#[cfg(feature = "hdf5")]
pub use checkpoint_studio_core::{convert, hdf5, hdf5_lz4, hdf5_zstd};

mod cli_config;
/// Loading the other side of a structural comparison, shared by the TUI's compare
/// screen and the web's `/api/diff` — see the module docs for which side is which.
mod compare;
mod explorer;
/// The one path from a typed spec to a read checkpoint, shared by the terminal and the
/// web server so both accept the same spellings when changing checkpoint at runtime.
mod opening;
/// The CLI↔web parity ledger's guard (tests only) — see `docs/cli-web-parity.md`.
#[cfg(test)]
mod parity_audit;
/// Data sources behind one trait — a new kind is an impl plus an arm in
/// [`source::resolve`], not another branch in the loader.
mod source;
mod tui;
mod ui;
mod web;

use anyhow::{Context, Result};
use clap::{Args as ClapArgs, Parser, Subcommand};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use crate::explorer::{DataLayout, Explorer, OpenRequest, OpenView};
use crate::tree::{MetadataInfo, TensorInfo};

/// `check --format` — the CLI's output-format choice. Lives here (not in the
/// frontend-free core) so `clap` doesn't leak into `core`; `run_check` dispatches
/// on it to the core report's `render` / `to_json` / `to_sarif`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    /// Human-readable report (default).
    #[default]
    Text,
    /// A structured JSON report: per-check status, findings, and the overall
    /// exit code — for scripts / agents / CI.
    Json,
    /// SARIF 2.1.0 — for GitHub code scanning / static-analysis tooling.
    Sarif,
}

/// Worked examples shown at the end of `--help` (not the terse `-h`), grouped by
/// the most useful things you can do. Written to read cleanly for both people
/// and coding agents: one commented, copy-pasteable command per line.
///
/// The `Examples:` title and each group's one-line description are styled to
/// match clap's own section headers (bold + underline) and sub-emphasis (bold),
/// but only when help is going to a colour-capable terminal — piped or
/// `NO_COLOR` output stays plain, like the rest of clap's help.
fn examples_help() -> String {
    use std::io::IsTerminal;
    let colour = std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal();
    // (title = bold+underline like `Options:`, group = bold sub-heading, r = reset).
    let (title, group, r) = if colour {
        ("\x1b[1m\x1b[4m", "\x1b[1m", "\x1b[0m")
    } else {
        ("", "", "")
    };
    format!(
        "\
{title}Examples:{r}
  {group}Browse a checkpoint{r} — a single file, a sharded directory, or a glob:
      checkpoint-studio model.safetensors
      checkpoint-studio /path/to/sharded-model/
      checkpoint-studio 'model-*.safetensors'

  {group}Look inside a tensor's data{r} — heatmap, numeric grid, histogram, statistics:
      checkpoint-studio model.safetensors --tensor model.layers.0.mlp.down_proj.weight --heatmap
      checkpoint-studio model.safetensors --tensor NAME --values --dtype u4   # decode packed 4-bit

  {group}Read a remote / S3 checkpoint over SSH{r} (only metadata leaves the host):
      checkpoint-studio --ssh-proxy user@host s3://bucket/model/checkpoint
      checkpoint-studio user@host:/opt/models/some-model          # scp-style; a safetensors dir
      checkpoint-studio :/opt/models/some-model                   # ':' prefix = the config's ssh_proxy

  {group}Export the structure for scripts / agents{r} (text, or --format json):
      checkpoint-studio model.safetensors --print-tree
      checkpoint-studio model.safetensors --print-tensors --format json
      checkpoint-studio model.safetensors --print-tree --name '*.mlp.*'   # !GLOB excludes

  {group}Compare two checkpoints{r} (exit 0 = identical, 1 = differ, 2 = error):
      checkpoint-studio diff old.safetensors new.safetensors
      checkpoint-studio diff old/ new/ --values --name '*.mlp.*'
      checkpoint-studio diff ':/opt/models/hf#language_model' s3://bkt/converted   # scope a side to a subtree

  {group}Health-check a checkpoint{r} (exit 0 = healthy, 1 = problems, 2 = error):
      checkpoint-studio check /path/to/model/
      checkpoint-studio check model.safetensors --values   # also scan for NaN/±Inf, all-zero

  {group}Repack an HDF5 checkpoint with an alternative codec{r} — smaller on disk (hdf5 build only):
      checkpoint-studio convert in.hdf5 out.hdf5 --codec zstd

  Per-subcommand help:  checkpoint-studio diff --help  ·  checkpoint-studio convert --help"
    )
}

#[derive(Parser)]
#[command(name = "checkpoint-studio")]
#[command(version)]
#[command(
    about = "Explore model checkpoints in the terminal — browse the tree, look inside tensor data, and diff (.safetensors / .gguf / .npy / .npz / .hdf5)"
)]
#[command(long_about = "\
Interactive terminal explorer for model checkpoints — .safetensors, .gguf, .npy/.npz, \
and (with the hdf5 build) .hdf5.

Beyond the tree of tensor names and shapes, it shows the actual data: ASCII heatmaps, \
numeric-value grids, value histograms, and exact whole-tensor statistics — streamed in \
bounded blocks, so multi-GB tensors work without loading them into RAM. Packed / \
quantized weights (4-bit, fused-codebook MoE) are decoded to their true values. \
Sharded / multi-file models, directories, and globs merge into one tree.

Remote checkpoints are read over an SSH proxy — a safetensors directory/file via SFTP, \
or an s3:// cstorch checkpoint via a remote venv — sending only metadata off the host, \
so data and credentials stay remote. Set the proxy you use most in a config file \
(~/.config/checkpoint-studio/config.toml) so you needn't pass --ssh-proxy every time:
    ssh_proxy = \"user@host\"
    ssh_venv  = \"~/venv\"      # optional; defaults to ~/venv
An explicit --ssh-proxy / --ssh-venv flag always overrides the config. With a proxy \
configured, prefix a path with `:` to read it there — `checkpoint-studio :/opt/models/foo` \
reads that path on the config's proxy host (the `:` keeps it explicit, so a same-named \
local directory is never routed off-host).

For scripts and agents there are one-shot --print-tree / --print-tensors exports (text \
or JSON) and a `diff` subcommand with diff-style exit codes.

Give one or more paths to browse; press `l` in any screen for its key legend. See the \
examples below and `<command> --help`.")]
#[command(after_long_help = examples_help())]
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Files/directories/globs to explore (the default action when no
    /// subcommand is given).
    #[command(flatten)]
    explore: ExploreArgs,
}

#[derive(ClapArgs)]
struct ExploreArgs {
    #[arg(help = "Checkpoint files, directories, or glob patterns to explore \
                (e.g. *.safetensors, model-*.gguf, *.npy, *.npz, *.hdf5).\n\
                \n\
                Remote paths work too (read over SSH — only metadata leaves the host):\n  \
                [USER@]HOST:/path   scp-style path, like --ssh-proxy\n  \
                :/path              read /path on the config's ssh_proxy host\n  \
                s3://…              an S3 checkpoint; pass with --ssh-proxy <HOST>")]
    paths: Vec<PathBuf>,

    #[arg(
        short,
        long,
        help = "Recursively search directories for checkpoint files"
    )]
    recursive: bool,

    #[arg(
        long = "no-health-check",
        help = "Skip the checkpoint health check (index vs. files on disk)"
    )]
    no_health_check: bool,

    #[arg(
        long = "no-preload",
        help = "Don't compute a tensor's statistics in the background when its detail screen opens (the scan reads the tensor, warming the OS/disk cache to speed up the heatmap/values views especially over NFS; with this flag, statistics are computed only when you press s)"
    )]
    no_preload: bool,

    #[arg(
        long,
        value_name = "NAME",
        help = "Open a specific tensor on startup (exact name); optional when the checkpoint has only one tensor (e.g. a .npy). Combine with --dtype/--shape/--values/--heatmap/--edge"
    )]
    tensor: Option<String>,

    #[arg(
        long,
        value_name = "NAME",
        conflicts_with = "tensor",
        help = "Reveal a metadata entry on startup (exact name, e.g. model.norm.weight.__metadata__) — opens the tree with it selected"
    )]
    metadata: Option<String>,

    #[arg(
        long,
        value_name = "DTYPE",
        value_parser = sample::parse_view_dtype,
        help = "Reinterpret the tensor's dtype: u4, i4, unpacked (fused codebook, needs a packing schema), f16, bf16, i16, u16, f32, i32, u32, f64, i64, u64, i8, u8, stored"
    )]
    dtype: Option<sample::ViewDtype>,

    #[arg(
        long,
        conflicts_with = "heatmap",
        help = "Open straight into the tensor's numeric-values grid"
    )]
    values: bool,

    #[arg(long, help = "Open straight into the tensor's heatmap")]
    heatmap: bool,

    #[arg(
        long,
        conflicts_with_all = ["values", "heatmap", "tree"],
        help = "Show the tensor's value histogram on its detail screen"
    )]
    histogram: bool,

    #[arg(
        long,
        value_name = "N",
        value_parser = parse_bins,
        conflicts_with_all = ["values", "heatmap", "tree"],
        help = "Histogram bucket count (1–512); implies --histogram"
    )]
    bins: Option<usize>,

    #[arg(
        long,
        conflicts_with_all = ["values", "heatmap"],
        help = "Reveal the tensor highlighted in the tree browser instead of opening a view"
    )]
    tree: bool,

    #[arg(
        long,
        visible_alias = "edges",
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "0.5,0.5",
        value_name = "RFRAC,CFRAC",
        conflicts_with_all = ["overview", "window"],
        help = "Show the first/last edges (padding) submode; optional ROW,COL head/tail split fractions 0..1 (0=first, 1=last, 0.5=balanced)"
    )]
    edge: Option<String>,

    #[arg(long, help = "Show the evenly-spaced overview submode")]
    overview: bool,

    #[arg(
        long = "abs-max",
        conflicts_with_all = ["edge", "overview", "window"],
        help = "Show the abs-max overview submode: each cell is the max |value| over its block (full scan; nothing sampled away)"
    )]
    abs_max: bool,

    #[arg(
        long,
        num_args = 0..=1,
        require_equals = true,
        default_missing_value = "0,0",
        value_name = "ROW,COL",
        conflicts_with_all = ["edge", "overview"],
        help = "Show the contiguous pannable window submode; optional ROW,COL top-left corner (default 0,0)"
    )]
    window: Option<String>,

    #[arg(
        long,
        value_name = "MODE",
        value_parser = ui::parse_stripe_mode,
        help = "Zebra-stripe the numeric grid by rows, cols, or off"
    )]
    zebra: Option<ui::StripeMode>,

    #[arg(
        long,
        value_name = "BASE",
        value_parser = ui::parse_num_base,
        help = "Numeral base for the numeric grid: dec, hex, oct, or bin (non-decimal shows raw stored bits)"
    )]
    base: Option<ui::NumBase>,

    #[arg(
        long,
        value_name = "INDEX",
        help = "Starting slice for a 3D tensor: an index (e.g. 12) or a percentage (e.g. 50%)"
    )]
    slice: Option<String>,

    #[arg(
        long,
        value_name = "DIMS",
        help = "Reinterpret the tensor's shape (same element count); dims like 10,100 or -1,768 (one dim may be -1/*/_ to infer)"
    )]
    shape: Option<String>,

    #[arg(
        long,
        help = "Start computing statistics immediately when opening the detail view (data views always compute them)"
    )]
    compute_stats: bool,

    #[arg(
        long,
        value_name = "STATE",
        value_parser = explorer::parse_tree_state,
        help = "Open the tree fully expanded or collapsed (the `E` / `C` keys): expanded or collapsed"
    )]
    tree_state: Option<explorer::TreeState>,

    #[arg(
        long,
        value_name = "QUERY",
        help = "Open the tree in search mode filtered to QUERY (the `/` key)"
    )]
    search: Option<String>,

    #[arg(
        long,
        help = "Overlay the requested screen's legend (the `l` key) — useful with --plain"
    )]
    legend: bool,

    #[arg(
        long,
        help = "Open straight into the health-check popup on the tree (the `h` key)"
    )]
    health: bool,

    #[arg(
        long,
        help = "Like --health, but with the per-finding detail expanded (the popup's `f` toggle)"
    )]
    health_findings: bool,

    #[arg(
        long,
        help = "Open straight into the full-screen checkpoint-stats view (the `s` key)"
    )]
    stats: bool,

    #[arg(
        long,
        help = "Like --stats, but with the on-disk per-shard breakdown expanded (the view's `f` toggle)"
    )]
    stats_shards: bool,

    #[arg(
        long = "print-arch",
        help = "Print the architecture inferred from the tensors alone — layers, experts, vocabulary, quantization, and the stored-vs-logical parameter counts — with the evidence for each, and what a model card lists that tensors cannot supply"
    )]
    print_arch: bool,

    #[arg(
        long = "compact",
        help = "Open the compact tree: uniform layers / experts folded into one templated subtree each, so only irregularities stand out (the `k` key). Structure at a glance — 31k tensors read as ~20 families"
    )]
    compact: bool,

    #[arg(
        long = "sort",
        value_name = "KEY[.DIR]",
        value_parser = viewstate::parse_sort,
        help = "Order the flat search / filter list by name, size, params, dtype or rank (the `o` key) — optionally `.asc` / `.desc`, e.g. --sort size.desc. `none` restores the natural order. The tree itself is never reordered"
    )]
    sort: Option<(viewstate::SortKey, viewstate::SortDir)>,

    #[arg(
        long = "diff-against",
        value_name = "PATH",
        help = "Open straight into the compare screen: a structural diff of this checkpoint against PATH (the tree's `d` command). Structure only — names, dtypes and shapes; `diff --values OLD NEW` compares the numbers"
    )]
    diff_against: Option<String>,

    #[arg(
        long = "compare-with",
        value_name = "PATH",
        help = "Open straight into the side-by-side compare screen: this checkpoint and PATH as one aligned tree, browsed in lockstep (the palette's *Compare side by side*). `--diff-against` opens the one-page report of the same pair instead"
    )]
    compare_with: Option<String>,

    #[arg(
        long = "compare-full",
        requires = "compare_with",
        help = "With --compare-with: show every layer as its own row. By default the compare screen folds uniform index families onto one row each (`{0-61}` ×62, the `k` key), so the layer that is *not* like its neighbours stands out"
    )]
    compare_full: bool,

    #[arg(
        long,
        help = "Open straight into the file browser — the checkpoint's directory tree (the `Tab` toggle)"
    )]
    files: bool,

    #[arg(
        long,
        value_name = "PATH",
        help = "Open straight into the safetensors byte-layout map for this file (Enter on a .safetensors in the file browser)"
    )]
    layout: Option<String>,

    #[arg(
        long = "layout-select",
        value_name = "NAME",
        requires = "layout",
        help = "Preselect this tensor in the --layout map (what the layout view's `y` records)"
    )]
    layout_select: Option<String>,

    #[arg(
        long,
        help = "Open straight into the in-place rename editor (local safetensors only; the `R` shortcut)"
    )]
    rename: bool,

    #[arg(
        long = "rename-rule",
        value_name = "SRC=>TGT",
        requires = "rename",
        help = "Seed a rename rule in the --rename editor: SOURCE=>NEW-NAME (schema form, {layer}/{expert} placeholders kept). Repeatable; what the editor's `y` records"
    )]
    rename_rule: Vec<String>,

    #[arg(
        long,
        help = "Render the requested view once and exit, without entering interactive navigation"
    )]
    exit: bool,

    #[arg(
        long,
        help = "Render the requested view once as plain text (no colour, no cursor control) and exit — for piping, grep, and end-to-end tests"
    )]
    plain: bool,

    #[arg(
        long,
        help = "Print the CLI command that reopens the requested view (what `y` copies) and exit, instead of rendering"
    )]
    emit_command: bool,

    #[arg(
        long = "print-tree",
        conflicts_with = "print_tensors",
        help = "Print the whole checkpoint tree (grouped, fully expanded) and exit — plain text, or a model.safetensors.index.json-style object with --format=json"
    )]
    print_tree: bool,

    #[arg(
        long = "print-tensors",
        help = "Print a flat list of every tensor and exit — plain text, or a JSON array with --format=json"
    )]
    print_tensors: bool,

    #[arg(
        long = "print-model",
        conflicts_with = "print_tree",
        conflicts_with = "print_tensors",
        help = "Print the whole central checkpoint model (files, headers, config, index) as JSON and exit — the serializable datatype the app reads everything into"
    )]
    print_model: bool,

    #[arg(
        long = "print-view",
        conflicts_with = "print_tree",
        conflicts_with = "print_tensors",
        conflicts_with = "print_model",
        help = "Print the tensor-tree screen's ViewModel (visible rows, selection, search) as JSON and exit — the kernel's frontend-agnostic output contract a web/MCP frontend would serve"
    )]
    print_view: bool,

    #[arg(
        long,
        value_enum,
        default_value_t = explorer::TreeFormat::default(),
        value_name = "FORMAT",
        help = "Output format for --print-tree / --print-tensors: text (default) or json"
    )]
    format: explorer::TreeFormat,

    #[arg(
        short = 'v',
        long = "verbose",
        action = clap::ArgAction::Count,
        help = "Add per-tensor detail to --print-tree / --print-tensors: the source file in text; a tensors block / detail objects in json"
    )]
    verbose: u8,

    #[arg(
        long = "name",
        value_name = "GLOB",
        help = "Filter --print-tree / --print-tensors to tensors whose name matches this glob (e.g. '*.mlp.*', 'model.layers.0.*'). Repeatable; prefix with ! to exclude ('!*.bias' = everything but biases)"
    )]
    name: Vec<String>,

    #[arg(
        long = "filter",
        value_name = "QUERY",
        help = "Filter --print-tree / --print-tensors by a rich query: facets AND, `!` negates, commas OR. e.g. 'dtype:F16,BF16 shape:(_,4096) size:>1MiB rank:>=3 name:re:^model\\.layers name:q_proj shard:00001'"
    )]
    filter: Option<String>,

    #[arg(
        long = "ssh-proxy",
        alias = "ssh-read",
        value_name = "[USER@]HOST",
        help = "Read a remote checkpoint's structure over an SSH proxy on [USER@]HOST (which has the access): an s3:// checkpoint, or a path to a safetensors directory/file on that host. Only the tensor metadata (names/dtypes/shapes) leaves the host — data/secrets stay remote. Metadata-only. Defaults to `ssh_proxy` in the config file (see --help)"
    )]
    ssh_proxy: Option<String>,

    #[arg(
        long = "ssh-venv",
        value_name = "PATH",
        help = "Path to the cstorch virtualenv on the --ssh-proxy host, activated with `source <PATH>/bin/activate` (default: ~/venv, or `ssh_venv` in the config file)"
    )]
    ssh_venv: Option<String>,
}

#[derive(Subcommand)]
// Parsed once at startup; the size gap between `Convert` and the many-flag `Diff`
// variant doesn't matter here.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Repack an HDF5 checkpoint, or rename tensors in place in a safetensors one.
    ///
    /// Without `--map`: repack the HDF5 INPUT into a new OUTPUT file, re-compressing
    /// every dataset with the chosen codec (gzip/zstd are ~2× smaller than the LZ4
    /// these checkpoints ship with).
    ///
    /// With `--map`: rename tensors **in place** in a local safetensors checkpoint
    /// (a directory or a single `.safetensors` file) — no OUTPUT is given. Only each
    /// shard's header region is rewritten (padded back to its original length, so
    /// tensor DATA IS NOT MOVED) and `model.safetensors.index.json` is updated to
    /// match. A rename whose new names don't fit in the existing header is refused
    /// (honouring it would mean rewriting the whole file). It confirms before
    /// touching anything unless `--force` is given.
    Convert {
        /// Source: an `.h5`/`.hdf5` checkpoint to repack, or (with `--map`) a local
        /// safetensors directory / `.safetensors` file to rename in place.
        input: PathBuf,
        /// Destination file to create (repack mode only; omit when using `--map`).
        output: Option<PathBuf>,
        /// Compression codec for the output (repack mode).
        #[arg(short, long, default_value_t = codec::Codec::default())]
        codec: codec::Codec,
        /// Compression level (gzip 0–9, zstd 1–22; ignored for lz4/none).
        /// Defaults to a sensible level for the codec (repack mode).
        #[arg(short, long)]
        level: Option<u8>,
        /// Streaming buffer per dataset block, e.g. `256M`, `1G` (repack mode).
        #[arg(short, long, default_value = "256M")]
        buffer: String,
        /// Repack mode: overwrite the output file if it exists. Rename mode: skip
        /// the confirmation prompt (apply without asking).
        #[arg(short, long)]
        force: bool,
        /// Rename tensors **in place** using this rule — regex
        /// `PATTERN=>REPLACEMENT` (with `$1`/`$name` captures); repeatable, applied
        /// in order. Switches `convert` to the in-place safetensors rename mode.
        /// E.g. drop a prefix: `--map 'model\.layers\.=>layers.'`
        #[arg(long = "map", value_name = "REGEX=>REPL")]
        map: Vec<String>,
        /// Load rename rules from a file (see `--map`; same format as `diff
        /// --map-from`): a `.json` array of `[pattern, replacement]` pairs, or one
        /// `REGEX=>REPLACEMENT` per line (`#` comments and blank lines ignored).
        #[arg(long = "map-from", value_name = "FILE")]
        map_from: Option<PathBuf>,
    },

    /// Compare two checkpoints and summarize their structural differences:
    /// tensors (by name, dtype, shape) and metadata (by name, value) that were
    /// added, removed, or changed. Tensor data/values are not compared.
    ///
    /// For an s3-vs-s3 diff (both sides `s3://` with `--ssh-proxy`), the underlying
    /// S3 objects' metadata is compared too — checksum/ETag, size, tags, and user
    /// metadata (best-effort, using the remote host's own AWS access); timestamps
    /// (last-modified and timestamp-valued tags/metadata) are shown as info only.
    /// `--values` / `--histogram` / `--tensor` are supported there as well: since the
    /// s3:// data isn't reachable locally, the element `|Δ|` and distribution stats
    /// are computed on the remote (only the small per-tensor results cross the wire).
    ///
    /// Exit status follows `diff`: 0 = structurally identical, 1 = differences
    /// found, 2 = trouble (a path couldn't be read).
    Diff {
        /// The baseline ("old") checkpoint — a file, directory, or glob. Append
        /// `#SUBTREE` to scope the comparison to a subtree, e.g.
        /// `hf-model#language_model` compares that model's `language_model.*` against
        /// the other side's root (descends into the subtree; siblings like
        /// `vision_tower` are left out of scope). Composes with --values locally;
        /// over --ssh-proxy it's structure-only.
        old: PathBuf,
        /// The checkpoint to compare against the baseline ("new"). Also takes a
        /// `#SUBTREE` scope suffix (see OLD).
        new: PathBuf,
        /// Recursively search directories for checkpoint files.
        #[arg(short, long)]
        recursive: bool,
        /// Compare only this one tensor (exact name) and, when it's present in
        /// both, also compare its element *values* (max/mean |Δ|), not just its
        /// dtype and shape. Without it, all tensors and metadata are compared
        /// structurally.
        #[arg(long, value_name = "NAME")]
        tensor: Option<String>,
        /// Compare only tensors — skip the checkpoints' metadata entirely.
        #[arg(long = "only-tensors")]
        only_tensors: bool,
        /// Also compare element values: read each tensor present in both (with a
        /// matching shape) and report max/mean |Δ| — turning a values-only change
        /// (same dtype & shape, different data) into a difference. Reads the whole
        /// checkpoint, so it's slower than the default structural diff.
        #[arg(long)]
        values: bool,
        /// Decode values under this view before comparing (with --values,
        /// --histogram, or --tensor): stored, u4, i4, unpacked (3-bit codebook, via
        /// the packing schema), f16, bf16, i16, u16, f32, i32, u32, f64, i64, u64,
        /// i8, u8.
        #[arg(long, value_name = "DTYPE", value_parser = sample::parse_view_dtype)]
        dtype: Option<sample::ViewDtype>,
        /// Compare value distributions: bin each common tensor's values (old & new
        /// over a shared layout) and report the total variation distance. With
        /// --tensor, prints the full bin-by-bin table. Reads the whole checkpoint.
        #[arg(long)]
        histogram: bool,
        /// Histogram bucket count (1–512) for --histogram; default picks a sensible
        /// count per dtype.
        #[arg(long, value_name = "N", value_parser = parse_bins)]
        bins: Option<usize>,
        /// List every changed entry instead of collapsing ones that share a name
        /// template and the same change (e.g. the same per-layer dtype change) into
        /// one line with a count and index range.
        #[arg(long)]
        full: bool,
        /// Never colorize the output (also off automatically when stdout isn't a
        /// terminal, or when `NO_COLOR` is set).
        #[arg(long = "no-color")]
        no_color: bool,
        /// Only diff tensors whose name matches this glob (e.g.
        /// '*.`mlp.down_proj.weight`', 'model.layers.*'). Repeatable — a tensor
        /// passes if it matches ANY; prefix with ! to exclude ('!*.bias' =
        /// everything but biases). Scopes the whole diff (structural + values)
        /// to the matching subset; metadata is not compared.
        #[arg(long = "name", value_name = "GLOB")]
        name: Vec<String>,
        /// Only diff these exact tensor names (comma-separated). Combine with
        /// --names-from; a tensor passes if it's in either list.
        #[arg(long = "names", value_name = "A,B,C")]
        names: Option<String>,
        /// Only diff the tensor names listed in this file (one per line; blank
        /// lines and '#' comments ignored).
        #[arg(long = "names-from", value_name = "FILE")]
        names_from: Option<PathBuf>,
        /// Only diff tensors whose stored dtype matches this glob, e.g. 'BF16',
        /// 'F*' (F16/F32/…). Case-insensitive.
        #[arg(long = "dtype-is", value_name = "GLOB")]
        dtype_is: Option<String>,
        /// Only diff tensors whose shape matches this glob. Dims are comma- or
        /// x-separated; '*' wildcards one dimension, '**' any number — e.g.
        /// '768,2048', '768,*', '*,2048', '768,**', '**,2048'.
        #[arg(long = "shape-is", value_name = "DIMS")]
        shape_is: Option<String>,
        /// Compare up to N tensors in parallel with --values / --histogram
        /// (default: number of logical CPUs; 1 = sequential). Reading tensor data
        /// is I/O-bound, so overlapping tensors speeds the whole run up.
        #[arg(short = 'j', long = "jobs", value_name = "N")]
        jobs: Option<usize>,
        /// Read each checkpoint's structure over SSH on [USER@]HOST (which holds the
        /// access): an s3:// checkpoint or a remote safetensors directory/file.
        /// Data/secrets stay remote. For an s3-vs-s3 pair, --values / --histogram /
        /// --tensor also work — the value/distribution comparison runs on the remote
        /// (only the per-tensor results cross the wire, never tensor data).
        #[arg(long = "ssh-proxy", alias = "ssh-read", value_name = "[USER@]HOST")]
        ssh_proxy: Option<String>,
        /// Path to the cstorch virtualenv on the --ssh-proxy host (default: ~/venv).
        #[arg(long = "ssh-venv", value_name = "PATH")]
        ssh_venv: Option<String>,
        /// Rename rule applied to the OLD checkpoint's tensor names before diffing,
        /// so tensors under a different naming scheme line up instead of showing as
        /// removed+added. Format 'REGEX=>REPLACEMENT' (regex, with $1 captures);
        /// repeatable, and rules apply in order. E.g. to diff a gpt-oss checkpoint
        /// against a block_sparse_moe-named one:
        ///   --map '\.mlp\.experts\.=>.`block_sparse_moe.experts`.'
        ///   --map 'experts\.(`down|gate_up`)_proj$=>experts.${1}_proj.weight'
        #[arg(long = "map", value_name = "REGEX=>REPL")]
        map: Vec<String>,
        /// Load rename rules from a file (see --map), merged after any --map. A
        /// '.json' file is a JSON array of [pattern, replacement] pairs; any other
        /// extension is one 'REGEX=>REPLACEMENT' rule per line ('#' comments and
        /// blank lines ignored).
        #[arg(long = "map-from", value_name = "FILE")]
        map_from: Option<PathBuf>,
        /// Verify two s3:// cstorch checkpoints are the SAME weights in different
        /// packings (s3-vs-s3 over --ssh-proxy). For each tensor present on both sides
        /// whose shape folds along dim 0 — old (E, …) sparse: one N-bit index per
        /// 16-bit word; new (ceil(E/fold), …) dense: `fold` indices per word, expert
        /// e at word e//fold, shift (e%fold)*bits — it decodes both on the remote and
        /// checks the indices match, and validates the packing (old words' bits above
        /// N, new words' bits above fold*N must be zero). Gate to a subset of layers
        /// with --name (this is a full data read, so scope it). Exit 0 = equivalent.
        #[arg(long = "verify-repack")]
        verify_repack: bool,
        /// Index bit-width for --verify-repack. Omit to auto-derive from the shape
        /// fold (fold 5 ⇒ 3-bit, fold 4 ⇒ 4-bit, …; `bits = 16 / fold`); pass N to
        /// override for a non-max-density packing.
        #[arg(long = "repack-bits", value_name = "N")]
        repack_bits: Option<usize>,
        /// Line an UNFUSED checkpoint up with its FUSED counterpart before comparing.
        ///
        /// Two layouts of one model share no tensor name, so a plain diff reports every
        /// tensor of both sides as one-sided — 80,107 against 933, "nothing lines up",
        /// which answers nothing. This drops the per-expert index (so the 256 tensors of
        /// `…experts.37.w2.weight` fold onto the one fused `…experts.down_proj.weight`
        /// that holds them, reported as `×256 → ×1`) and applies the standard layout
        /// synonyms — `w1`/`w3` ↔ `gate_up_proj`, `w2` ↔ `down_proj`, q/k/v ↔ `qkv_proj`,
        /// `.weight.qscale` ↔ `.qscale`, `e_score_correction_bias` ↔ `gate.bias`,
        /// a `language_model.` prefix or none. The rules are printed, and are exactly
        /// what --map takes, so a checkpoint they mis-align can be aligned by hand.
        /// Applied to both sides: each rule is a no-op on a checkpoint already fused.
        #[arg(long = "align-fused")]
        align_fused: bool,
    },

    /// Run health checks on a checkpoint and report any problems.
    ///
    /// Structural checks read only headers: byte-range integrity (safetensors
    /// spans are sized right, contiguous, and the file isn't truncated), HDF5
    /// chunk/dtype integrity (the equivalent for .hdf5), layer completeness
    /// (contiguous layer indices, a uniform tensor set per layer), shape/dtype
    /// sanity, and file/shard correspondence. --values additionally scans tensor
    /// data for NaN/±Inf and all-zero/constant tensors (reads the whole
    /// checkpoint locally).
    ///
    /// Exit status follows `diff`: 0 = healthy, 1 = problems found, 2 = trouble
    /// (a path couldn't be read). Warnings only fail the run under --strict.
    Check {
        /// The checkpoint to check — a file, directory, or glob (shards merge
        /// into one checkpoint).
        #[arg(value_name = "PATH", required = true)]
        paths: Vec<PathBuf>,
        /// Recursively search directories for checkpoint files.
        #[arg(short, long)]
        recursive: bool,
        /// Also scan tensor data (NaN/±Inf, all-zero, constant tensors). Reads the
        /// whole checkpoint, so it's slower and needs the files locally.
        #[arg(long)]
        values: bool,
        /// Fail (exit 1) on warnings too, not just errors.
        #[arg(long)]
        strict: bool,
        /// Limit the --values scan to tensors whose name matches this glob
        /// (repeatable; prefix ! to exclude). The structural checks always run on
        /// the whole checkpoint.
        #[arg(long = "name", value_name = "GLOB")]
        name: Vec<String>,
        /// Scan up to N tensors in parallel with --values (default: logical CPUs;
        /// 1 = sequential).
        #[arg(short = 'j', long = "jobs", value_name = "N")]
        jobs: Option<usize>,
        /// Output format: text (default), json (a structured report for scripts /
        /// agents / CI), or sarif (SARIF 2.1.0 for GitHub code scanning).
        #[arg(long, value_enum, default_value_t = Format::default(), value_name = "FORMAT")]
        format: Format,
        /// Never colorize the output (also off when stdout isn't a terminal, or
        /// when `NO_COLOR` is set).
        #[arg(long = "no-color")]
        no_color: bool,
        /// Check a remote checkpoint's structure over SSH on [USER@]HOST (which
        /// holds the access): an s3:// checkpoint or a remote safetensors
        /// directory/file. Only metadata leaves the host, so the structural
        /// checks run but --values (value scan) does not.
        #[arg(long = "ssh-proxy", alias = "ssh-read", value_name = "[USER@]HOST")]
        ssh_proxy: Option<String>,
        /// Path to the cstorch virtualenv on the --ssh-proxy host (default: ~/venv).
        #[arg(long = "ssh-venv", value_name = "PATH")]
        ssh_venv: Option<String>,
    },

    /// Serve a web UI (Svelte) showing the same information as the TUI, and block
    /// until Ctrl-C.
    ///
    /// The server supplies the checkpoint as JSON (the data); the browser owns the
    /// view state. It binds all interfaces by default and prints a URL using this
    /// machine's hostname (e.g. <http://your-vm.example.com:8080>/), so you can open
    /// it from another machine's browser with no tunnel. Local checkpoints only.
    Web {
        /// The checkpoint to serve — a file, directory, or glob (shards merge into
        /// one checkpoint). With --ssh-proxy, an `s3://…` URI or a remote path.
        #[arg(value_name = "PATH", required = true)]
        paths: Vec<PathBuf>,
        /// Recursively search directories for checkpoint files.
        #[arg(short, long)]
        recursive: bool,
        /// Port to serve on (0 = let the OS pick a free port, printed at startup).
        #[arg(long, value_name = "PORT", default_value_t = 8080)]
        port: u16,
        /// IP address to bind. Defaults to 0.0.0.0 (all interfaces) so it's
        /// reachable at your machine's hostname; use 127.0.0.1 for loopback only.
        #[arg(long, value_name = "ADDR", default_value = "0.0.0.0")]
        host: std::net::IpAddr,
        /// Skip parsing model.safetensors.index.json for the health check.
        #[arg(long = "no-health-check")]
        no_health_check: bool,
        /// Serve a remote checkpoint: read its structure over SSH on HOST (metadata
        /// only — data-value views need the file locally). PATH is then an `s3://…`
        /// URI or a remote path.
        #[arg(long = "ssh-proxy", alias = "ssh-read", value_name = "[USER@]HOST")]
        ssh_proxy: Option<String>,
        /// The Python venv on the SSH host that has `cerebras.pytorch` (for reading
        /// s3:// cstorch checkpoints). Defaults to ~/venv.
        #[arg(long = "ssh-venv", value_name = "PATH")]
        ssh_venv: Option<String>,
    },
}

fn main() -> Result<()> {
    // libhdf5 takes an flock() on every file it opens, which fails with errno 11
    // (EAGAIN) on filesystems that don't support it — NFS especially, exactly
    // where big checkpoints tend to live. We only ever read/repack (never share a
    // writer), so the lock buys nothing: disable it before the first HDF5 call,
    // unless the user set the variable themselves. No-op on the pure-Rust build.
    #[cfg(feature = "hdf5")]
    if std::env::var_os("HDF5_USE_FILE_LOCKING").is_none() {
        // SAFE: first statement in `main`, before any threads spawn or any HDF5
        // call runs (libhdf5 reads this only at its lazy init).
        unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };
    }

    let cli = Cli::parse();
    // User CLI defaults (e.g. the SSH proxy they usually use) — an explicit flag
    // always overrides (see `resolve_ssh_proxy`).
    let cfg = cli_config::CliConfig::load();

    match cli.command {
        Some(Command::Convert {
            input,
            output,
            codec,
            level,
            buffer,
            force,
            map,
            map_from,
        }) => {
            if map.is_empty() && map_from.is_none() {
                // Repack mode: an OUTPUT file is required.
                let output = output.ok_or_else(|| {
                    anyhow::anyhow!(
                        "convert needs an OUTPUT file to repack into (or --map to rename tensors in place)"
                    )
                })?;
                run_convert(&input, &output, codec, level, &buffer, force)
            } else {
                // In-place rename mode: no OUTPUT (the files are edited in place).
                if let Some(out) = output {
                    anyhow::bail!(
                        "convert --map renames tensors in place, so it takes no OUTPUT (got {})",
                        out.display()
                    );
                }
                run_rename(&input, &map, map_from.as_deref(), force)
            }
        }
        Some(Command::Diff {
            old,
            new,
            recursive,
            tensor,
            only_tensors,
            values,
            dtype,
            histogram,
            bins,
            full,
            no_color,
            name,
            names,
            names_from,
            dtype_is,
            shape_is,
            jobs,
            ssh_proxy,
            ssh_venv,
            map,
            map_from,
            verify_repack,
            repack_bits,
            align_fused,
        }) => {
            // `diff`-style exit codes (0 same / 1 differ / 2 trouble) don't map to
            // the `Result` convention `main` uses elsewhere, so exit explicitly.
            // Split off a `SOURCE#subtree` re-root (per operand) before touching the
            // address, so the reader gets a clean path/URI.
            let (old, old_root) = split_reroot(&old);
            let (new, new_root) = split_reroot(&new);
            // `--ssh-proxy` (or, for an s3:// pair / `:PATH`, the config default):
            // read each checkpoint's structure via the remote (secrets stay there).
            let (srcs, remote) =
                match resolve_remote_sources(&[old, new], ssh_proxy, ssh_venv, &cfg) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("checkpoint-studio diff: {e:#}");
                        std::process::exit(2);
                    }
                };
            // `diff` takes exactly two sources; the arg parser guarantees the pair.
            let [old, new] = srcs.as_slice() else {
                eprintln!("checkpoint-studio diff: expected exactly two checkpoints");
                std::process::exit(2);
            };
            let remote = remote.map(|(host, venv)| remote::RemoteRead::new(host, venv));
            let filter = match build_tensor_filter(
                &name,
                names.as_deref(),
                names_from.as_deref(),
                dtype_is.as_deref(),
                shape_is.as_deref(),
            ) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("checkpoint-studio diff: {e:#}");
                    std::process::exit(2);
                }
            };
            let filtered = filter.is_active();
            // `--align-fused` is a rename map with a name: the canonical unfused→fused rules, applied
            // to *both* sides and folding many names onto one (`diff::fused_layout_rules`). Printed, so
            // the transformation is checkable and adaptable — the pairs are what `--map` takes.
            let name_map = match build_name_map(&map, map_from.as_deref()) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("checkpoint-studio diff: {e:#}");
                    std::process::exit(2);
                }
            };
            let align = if align_fused {
                let rules = diff::fused_layout_rules();
                utils::eprint_note(
                    "checkpoint-studio diff: ",
                    "aligning the unfused layout onto the fused one — dropping the expert index and \
                     applying the standard synonyms. Each rule is `--map PATTERN=>REPLACEMENT`:",
                );
                for (pat, rep) in &rules {
                    eprintln!("    {pat}=>{rep}");
                }
                match diff::NameMap::from_pairs(rules) {
                    Ok(m) => Some(m),
                    Err(e) => {
                        eprintln!("checkpoint-studio diff: --align-fused: {e:#}");
                        std::process::exit(2);
                    }
                }
            } else {
                None
            };
            let opts = diff::DiffOpts {
                color: color_enabled(no_color),
                // A tensor filter scopes the diff to a subset, so metadata isn't
                // compared (like --only-tensors, but for a filtered run).
                metadata: !only_tensors && !filtered,
                group: !full,
                values,
                histogram,
                filtered,
            };
            let view = dtype.unwrap_or(sample::ViewDtype::Stored);
            // Default parallelism = logical CPUs; `--jobs 0` is treated as 1.
            let jobs = jobs.filter(|&j| j > 0).unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(4, std::num::NonZero::get)
            });
            let started = std::time::Instant::now();
            let code = run_diff(
                old,
                new,
                recursive,
                tensor.as_deref(),
                view,
                bins,
                opts,
                &filter,
                &name_map,
                jobs,
                remote.as_ref(),
                verify_repack,
                repack_bits,
                old_root.as_deref(),
                new_root.as_deref(),
                align.as_ref(),
            );
            // Report how long it took, by default (on stderr, so a piped diff on
            // stdout stays clean). Skip on trouble (exit 2) — the error already said.
            // Dimmed (when stderr is a colour terminal) as a secondary footer line.
            if code != 2 {
                use std::io::IsTerminal;
                let msg = format!(
                    "checkpoint-studio diff: done in {}",
                    format_elapsed(started.elapsed())
                );
                let dim = !no_color
                    && std::env::var_os("NO_COLOR").is_none()
                    && std::io::stderr().is_terminal();
                if dim {
                    eprintln!("\x1b[2m{msg}\x1b[0m");
                } else {
                    eprintln!("{msg}");
                }
            }
            std::process::exit(code)
        }
        Some(Command::Check {
            paths,
            recursive,
            values,
            strict,
            name,
            jobs,
            format,
            no_color,
            ssh_proxy,
            ssh_venv,
        }) => {
            let jobs = jobs.filter(|&j| j > 0).unwrap_or_else(|| {
                std::thread::available_parallelism().map_or(4, std::num::NonZero::get)
            });
            let (paths, remote) = resolve_remote_sources(&paths, ssh_proxy, ssh_venv, &cfg)?;
            let remote = remote.map(|(host, venv)| remote::RemoteRead::new(host, venv));
            std::process::exit(run_check(
                &paths,
                recursive,
                values,
                strict,
                &name,
                jobs,
                format,
                no_color,
                remote.as_ref(),
            ))
        }
        Some(Command::Web {
            paths,
            recursive,
            port,
            host,
            no_health_check,
            ssh_proxy,
            ssh_venv,
        }) => {
            let (paths, ssh) = resolve_remote_sources(&paths, ssh_proxy, ssh_venv, &cfg)?;
            run_web(&paths, recursive, no_health_check, host, port, ssh)
        }
        None => {
            // The default (no subcommand) is the interactive explorer. Fill the SSH
            // proxy/venv from the config file when the flags weren't given — but only
            // for a source that needs it (`s3://` or a `:PATH`), so a configured
            // default never routes a plain local path through SSH (an explicit
            // --ssh-proxy still does). `[user@]host:/path` is handled in run_explore.
            let mut args = cli.explore;
            let (proxy, venv) = (args.ssh_proxy.take(), args.ssh_venv.take());
            let (paths, remote) = resolve_remote_sources(&args.paths, proxy, venv, &cfg)?;
            args.paths = paths;
            if let Some((host, venv)) = remote {
                args.ssh_proxy = Some(host);
                args.ssh_venv = Some(venv);
            }
            run_explore(args)
        }
    }
}

/// Run health checks on a checkpoint and print the report. Returns the process
/// exit code: `0` healthy, `1` problems found (warnings only when `strict`), `2`
/// trouble (a path couldn't be read). With `remote`, the structural (header-only)
/// checks run over SSH; the `--values` scan needs the bytes locally.
// Four of these are bools, over clippy's threshold of three. They stay parameters because
// this is a CLI entry point called from exactly one place, where clap's own field names sit
// right next to the call — the readability problem the lint exists for (a bare
// `true, false, true` at a call site) doesn't arise.
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)] // one arg per flag
fn run_check(
    paths: &[PathBuf],
    recursive: bool,
    values: bool,
    strict: bool,
    name: &[String],
    jobs: usize,
    format: Format,
    no_color: bool,
    remote: Option<&remote::RemoteRead>,
) -> i32 {
    let filter = match filter::NameFilter::parse(name) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("checkpoint-studio check: {e:#}");
            return 2;
        }
    };

    // Build the report (or an exit code, on trouble).
    let report: Result<check::CheckReport, i32> = if let Some(r) = remote {
        if values {
            eprintln!(
                "checkpoint-studio check: --values needs the checkpoint locally \
                 (only metadata is read over --ssh-proxy)"
            );
            return 2;
        }
        let src = paths
            .first()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        (|| -> Result<check::CheckReport> {
            let mut password: Option<String> = None;
            let session = r.open_with(&mut password)?;
            eprintln!("checkpoint-studio check: reading tensor metadata over ssh …");
            let bars = progress::Bars::start(std::slice::from_ref(&src));
            let progress = bars.progress(0);
            // `check` fetches the S3 object metadata (an extra HEAD per object) on
            // purpose: cross-checking it against the checkpoint index is one of the
            // checks reported below, and verifying is what this command is for.
            let out = r
                .read(
                    &session,
                    &src,
                    &password,
                    progress.as_deref(),
                    remote::ObjectMeta::Fetch,
                    None,
                )
                .with_context(|| format!("reading {src}"));
            bars.finish(0, out.is_ok());
            bars.join();
            let rc = out?;
            // The checkpoint's config.json, fetched over the same session (no
            // second prompt) so the config check runs remotely too.
            let config = r.read_config(&session, &src);
            // Index/file consistency comes from the same read (computed from the
            // parsed shards), so `check` flags a botched/stale remote index just
            // like a local one — with no second index read.
            Ok(check::run(
                src.clone(),
                &rc.tensors,
                &rc.metadata,
                &[],
                &rc.health,
                config.as_ref(),
                &filter,
                // A remote read keeps the index it parsed but not the per-shard header
                // lengths, so the alignment half of the check reads n/a.
                check::HeaderInputs {
                    index: &rc.index,
                    unreadable: &rc.unreadable,
                    ..Default::default()
                },
                false,
                jobs,
            ))
        })()
        .map_err(|e: anyhow::Error| {
            eprintln!("checkpoint-studio check: {e:#}");
            2
        })
    } else {
        // Health enabled: its index-vs-disk report is folded into the file check.
        (|| -> Result<check::CheckReport> {
            let (files, index_specs) = collect_safetensors_files(paths, recursive, false)?;
            if files.is_empty() {
                anyhow::bail!("no checkpoint files found");
            }
            let (parts, cp) = Explorer::gather_checkpoint(&files, None)?;
            let opening::CheckpointParts {
                tensors,
                metadata,
                config,
                ..
            } = parts;
            // Index-vs-disk health from the tensors just loaded (no extra header
            // reads); folded into the file check below.
            let health: Vec<health::HealthReport> = index_specs
                .iter()
                .map(|spec| health::check_loaded(spec, &tensors))
                .filter(health::HealthReport::has_issues)
                .collect();
            let label = paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            Ok(check::run(
                label,
                &tensors,
                &metadata,
                &files,
                &health,
                config.as_ref(),
                &filter,
                // The read kept every shard's header length and the index it declares —
                // what the header-consistency check needs.
                cp.as_ref()
                    .map(check::HeaderInputs::from)
                    .unwrap_or_default(),
                values,
                jobs,
            ))
        })()
        .map_err(|e: anyhow::Error| {
            eprintln!("checkpoint-studio check: {e:#}");
            2
        })
    };

    let report = match report {
        Ok(r) => r,
        Err(code) => return code,
    };
    match format {
        Format::Text => print!("{}", report.render(color_enabled(no_color))),
        Format::Json => println!(
            "{}",
            serde_json::to_string_pretty(&report.to_json(strict)).unwrap_or_default()
        ),
        Format::Sarif => println!(
            "{}",
            serde_json::to_string_pretty(&report.to_sarif()).unwrap_or_default()
        ),
    }
    report.exit_code(strict)
}

/// Compare two checkpoints' structure and print the summary. Returns the process
/// exit code: `0` identical, `1` differences found, `2` trouble (unreadable path).
/// Whether to colorize the diff: off when `--no-color`, when `NO_COLOR` is set
/// (<https://no-color.org>), or when stdout isn't a terminal (so pipes stay clean).
fn color_enabled(no_color: bool) -> bool {
    use std::io::IsTerminal;
    !no_color && std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
}

/// The decode view, requested histogram bucket count, and each side's packing
/// schemas (for the `unpacked` view) used by the value / distribution comparison.
struct ValueCtx<'a> {
    view: sample::ViewDtype,
    bins: Option<usize>,
    old_schemas: &'a HashMap<String, sample::PackingSchema>,
    new_schemas: &'a HashMap<String, sample::PackingSchema>,
}

/// Build a [`diff::TensorFilter`] from the `diff` selection flags.
///
/// Reads `--names-from` and hands the *content* to [`crate::compare::tensor_filter`], which the web's
/// diff routes also call — so both surfaces scope a comparison by one implementation rather than two
/// that agree until one of them is edited.
fn build_tensor_filter(
    name: &[String],
    names: Option<&str>,
    names_from: Option<&Path>,
    dtype_is: Option<&str>,
    shape_is: Option<&str>,
) -> Result<diff::TensorFilter> {
    let lines = names_from
        .map(|path| {
            fs::read_to_string(path)
                .with_context(|| format!("reading --names-from {}", path.display()))
        })
        .transpose()?;
    compare::tensor_filter(&compare::ScopeText {
        name,
        names_csv: names,
        names_lines: lines.as_deref(),
        dtype_is,
        shape_is,
    })
    // Name the flag the glob came from; the shared builder cannot know which one it was.
    .map_err(|e| e.context("in a --name / --dtype-is / --shape-is pattern"))
}

/// Build a [`diff::NameMap`] from the `diff` rename flags: the `--map` rules first
/// (each a `PATTERN=>REPLACEMENT` line), then `--map-from`, whose extension picks
/// the format — a `.json` file is a JSON array of `[pattern, replacement]` pairs,
/// anything else is the same plain-text `PATTERN=>REPLACEMENT`-per-line form as
/// `--map` (and as `--names-from`). Errors (bad rule, unreadable/invalid file, bad
/// regex) bubble up to a `2` exit.
fn build_name_map(map: &[String], map_from: Option<&Path>) -> Result<diff::NameMap> {
    let text = map_from
        .map(|path| {
            fs::read_to_string(path)
                .with_context(|| format!("reading --map-from {}", path.display()))
        })
        .transpose()?;
    let json = map_from.is_some_and(|p| {
        p.extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("json"))
    });
    // `--map-from`'s extension picks the format; the web picks by which box was filled.
    let (lines, as_json) = if json {
        (None, text.as_deref())
    } else {
        (text.as_deref(), None)
    };
    compare::name_map(map, lines, as_json).with_context(|| {
        map_from.map_or_else(
            || "in a --map rule".to_string(),
            |p| format!("in --map-from {}", p.display()),
        )
    })
}

/// Shared state between the value-comparison workers and the spinner thread.
struct CompareState {
    /// Tensors currently being compared (one entry per in-flight worker).
    inflight: std::sync::Mutex<Vec<String>>,
    done: std::sync::atomic::AtomicUsize,
    total: usize,
    stop: std::sync::atomic::AtomicBool,
}

/// Live progress for `diff --values` / `--histogram`: reading tensor data is the
/// slow part, so — only in an interactive terminal — a background thread renders
/// a spinner plus **every tensor currently being compared** (one per line) on
/// **stderr** (stdout stays a clean diff), cleared when done. Workers call
/// [`Self::track`] for an RAII guard that keeps a tensor listed while it's being
/// compared. A no-op when stderr isn't a TTY (piped / headless) or nothing will
/// be compared.
struct CompareProgress {
    state: Option<std::sync::Arc<CompareState>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

/// RAII marker: a tensor is being compared while this is alive; on drop it leaves
/// the in-flight list and bumps the done count.
struct InFlight<'a> {
    state: Option<&'a CompareState>,
    name: String,
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering;
        if let Some(st) = self.state {
            if let Ok(mut v) = st.inflight.lock()
                && let Some(i) = v.iter().position(|n| n == &self.name)
            {
                v.swap_remove(i);
            }
            st.done.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl CompareProgress {
    fn start(total: usize) -> Self {
        use std::io::IsTerminal;
        if total == 0 || !std::io::stderr().is_terminal() {
            return Self {
                state: None,
                handle: None,
            };
        }
        let state = std::sync::Arc::new(CompareState {
            inflight: std::sync::Mutex::new(Vec::new()),
            done: std::sync::atomic::AtomicUsize::new(0),
            total,
            stop: std::sync::atomic::AtomicBool::new(false),
        });
        let worker = std::sync::Arc::clone(&state);
        let handle = std::thread::spawn(move || compare_spinner_loop(worker));
        Self {
            state: Some(state),
            handle: Some(handle),
        }
    }

    /// Mark `name` as being compared until the returned guard drops.
    fn track(&self, name: &str) -> InFlight<'_> {
        if let Some(st) = &self.state
            && let Ok(mut v) = st.inflight.lock()
        {
            v.push(name.to_string());
        }
        InFlight {
            state: self.state.as_deref(),
            name: name.to_string(),
        }
    }

    /// Stop the spinner thread (which erases its block on exit) and join it.
    fn finish(mut self) {
        use std::sync::atomic::Ordering;
        if let Some(st) = &self.state {
            st.stop.store(true, Ordering::Relaxed);
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// The spinner thread body: ~10×/s redraw a block — a header plus one line per
/// in-flight tensor — in place on stderr until told to stop, then erase it.
fn compare_spinner_loop(st: std::sync::Arc<CompareState>) {
    use std::io::Write;
    use std::sync::atomic::Ordering;
    let (width, height) = match crossterm::terminal::size() {
        Ok((c, r)) if c > 0 && r > 0 => (c as usize, r as usize),
        _ => (100, 24),
    };
    let mut prev_lines = 0usize;
    let mut frame = 0usize;
    while !st.stop.load(Ordering::Relaxed) {
        let mut names = st.inflight.lock().map(|v| v.clone()).unwrap_or_default();
        names.sort_unstable();
        let done = st.done.load(Ordering::Relaxed);
        let block = compare_progress_block(frame, done, st.total, &names, width, height);
        draw_block(&block, &mut prev_lines);
        frame += 1;
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    erase_block(prev_lines);
    let _ = std::io::stderr().flush();
}

/// The progress block: a spinner + `done/total` header, then one indented line
/// per in-flight tensor (capped to the terminal height, name tails kept).
fn compare_progress_block(
    frame: usize,
    done: usize,
    total: usize,
    names: &[String],
    width: usize,
    height: usize,
) -> Vec<String> {
    let spin = progress::spinner_frame(frame);
    let mut lines = vec![format!(
        "{spin} comparing tensors ({done}/{total}, {} in flight):",
        names.len()
    )];
    let indent = "    ";
    let max_rows = height.saturating_sub(2).max(1); // keep the block on screen
    let shown = names.len().min(max_rows);
    for name in names.get(..shown).unwrap_or(names) {
        let budget = width.saturating_sub(indent.chars().count());
        lines.push(format!("{indent}{}", truncate_tail(name, budget)));
    }
    if names.len() > shown {
        lines.push(format!("{indent}… and {} more", names.len() - shown));
    }
    lines
}

/// Redraw `lines` in place: move back to the previous block's top, clear
/// downward, and reprint. Leaves the cursor on the line just below the block.
fn draw_block(lines: &[String], prev_lines: &mut usize) {
    use std::io::Write;
    let mut out = String::new();
    if *prev_lines > 0 {
        let _ = write!(out, "\x1b[{prev_lines}A"); // up to the block's first line
    }
    out.push_str("\r\x1b[0J"); // column 0, clear to end of screen
    out.push_str(&lines.join("\n"));
    out.push('\n'); // rest on the line below, so the count is stable frame-to-frame
    eprint!("{out}");
    let _ = std::io::stderr().flush();
    *prev_lines = lines.len();
}

/// Erase a previously drawn block (on finish), leaving the cursor at its top.
fn erase_block(prev_lines: usize) {
    if prev_lines > 0 {
        eprint!("\x1b[{prev_lines}A\r\x1b[0J");
    }
}

/// Drive one standard [`progress::Bars`] bar per compared tensor from the remote
/// comparison's [`RepackEvent`](crate::remote::RepackEvent) stream — shared by
/// `--values` and `--verify-repack`. Each bar's total is the tensor's (old + new)
/// S3 byte size (rendered as human sizes) and it fills as the proxy streams the two
/// sides, settling to ✓ (compared / equivalent) or ✗ (error) as each lands. The
/// comparison itself stays on the proxy — only the byte counts cross ssh. Names
/// not among the pairs we asked for are ignored.
/// One aggregate progress bar for a whole remote value compare / repack verify.
///
/// A bar per tensor is unreadable at checkpoint scale: a 16B `MoE` has thousands, so
/// `diff --values` scrolled for screens and nothing on them stayed still long enough to
/// read. This is one bar filling over the *total* bytes both sides will stream, relabelled
/// with the tensor currently being read — so the line says how far along the run is and
/// where it is, in the space one bar takes.
///
/// The per-tensor byte events are cumulative *per tensor*, so summing the latest value
/// seen for each is what gives a total that only ever moves forward. Adding the deltas
/// would drift: a retried tensor re-reports from zero.
struct ValueBar {
    bars: progress::Bars,
    /// Latest cumulative bytes per tensor, keyed by name.
    seen: HashMap<String, u64>,
    /// Tensors finished, for the `k/n` note.
    done: usize,
    total_tensors: usize,
    /// Whether any tensor reported an error, so the bar can end as `✓` or `✗`.
    failed: bool,
}

impl ValueBar {
    /// Start the bar. `total_bytes` is what both sides add up to; `total_tensors` is the
    /// pair count, used for the `k/n tensors` note.
    fn start(label: &str, total_bytes: u64, total_tensors: usize) -> Self {
        let bars = progress::Bars::start(std::slice::from_ref(&label.to_string()));
        if let Some(p) = bars.progress(0) {
            p.set_unit(progress::Unit::Bytes);
            // Zero would render as a finished bar; leave the total unset so it sweeps
            // until the first size arrives.
            if total_bytes > 0 {
                p.set_total(total_bytes as usize);
            }
        }
        Self {
            bars,
            seen: HashMap::new(),
            done: 0,
            total_tensors,
            failed: false,
        }
    }

    /// Fold one event into the bar.
    fn on(&mut self, ev: remote::RepackEvent<'_>) {
        use crate::remote::RepackEvent as E;
        let Some(p) = self.bars.progress(0) else {
            return;
        };
        match ev {
            // Nothing to show yet: the checkpoints are still opening.
            E::Loading(_) | E::Size { .. } => {}
            E::Start { name, .. } => {
                p.set_item(&self.item_label(name));
            }
            E::Bytes {
                name,
                old_done,
                new_done,
            } => {
                self.seen.insert(name.to_string(), old_done + new_done);
                p.set_done(self.seen.values().sum::<u64>() as usize);
            }
            // The whole run is never "done reading" until the end, so a per-tensor
            // compare phase would flicker the note on and off; name the tensor instead.
            E::Comparing { name, spans } => {
                // How far *that* has got, when the tensor is compared in pieces: a gigabyte-scale
                // weight spends its time here, and the bar's counters do not move until it is done.
                let label = match spans {
                    Some((done, total)) => {
                        format!("{} · comparing {done}/{total}", self.item_label(name))
                    }
                    None => format!("{} · comparing", self.item_label(name)),
                };
                p.set_item(&label);
            }
            E::Done { name, status } => {
                self.done += 1;
                if status == remote::CompareStatus::Error {
                    self.failed = true;
                }
                // A tensor that never reported bytes still counted toward the total, so
                // credit it here or the bar can never fill.
                self.seen.entry(name.to_string()).or_insert(0);
                p.set_item(&self.item_label(name));
            }
        }
    }

    /// `[k/n] tensor.name` — the position in the run, then where it is.
    fn item_label(&self, name: &str) -> String {
        format!(
            "[{}/{}] {name}",
            (self.done + 1).min(self.total_tensors.max(1)),
            self.total_tensors
        )
    }

    /// Close the bar and wait for its final frame.
    fn finish(self) {
        self.bars.finish(0, !self.failed);
        self.bars.join();
    }
}

/// Human-readable elapsed time: `850ms`, `12.3s`, or `2m3s`.
fn format_elapsed(d: std::time::Duration) -> String {
    let secs = d.as_secs_f64();
    if secs < 1.0 {
        format!("{}ms", d.as_millis())
    } else if secs < 60.0 {
        format!("{secs:.1}s")
    } else {
        let mins = d.as_secs() / 60;
        format!("{mins}m{}s", d.as_secs() % 60)
    }
}

// Keeping a tensor name's informative end within a budget is `utils::truncate_keep_end`, which the
// TUI's screens already used. This was a second copy that answered `…` for a budget of zero — a
// character in a column that does not exist.
use crate::utils::truncate_keep_end as truncate_tail;

/// The tensors + metadata read from one checkpoint (local or remote).
type Loaded = (Vec<TensorInfo>, Vec<MetadataInfo>);
/// One diff side: its loaded structure plus, for an `s3://` source, the underlying
/// S3 objects' metadata (`None` for a local / SFTP source).
type SideLoad = (Loaded, Option<remote::S3Meta>);

/// Whether an error is the [`crate::sftp::RemoteSession::ABORTED`] marker — a read
/// cut short because the *other* side of a parallel `diff` failed first (not a
/// failure of this read itself).
fn is_aborted_err(e: &anyhow::Error) -> bool {
    e.chain()
        .any(|c| c.to_string().contains(sftp::RemoteSession::ABORTED))
}

/// An effective SSH proxy for a read: `(host, venv)`.
type SshProxy = (String, String);

/// Resolve the effective SSH proxy for a read. An explicit `--ssh-proxy` flag wins
/// and forces the proxy for *any* source. The config file's `ssh_proxy` default only
/// engages when the source genuinely needs a proxy — an `s3://` URI or a `:`-prefixed
/// remote path — so a configured default never hijacks a plain **local** path into an
/// SSH read. Returns `(host, venv)` (venv: flag → config → `~/venv`), or `None` for a
/// plain local read.
fn resolve_ssh_proxy(
    proxy: Option<String>,
    venv: Option<String>,
    cfg: &cli_config::CliConfig,
    source_needs_proxy: bool,
) -> Option<SshProxy> {
    let host = proxy.or_else(|| source_needs_proxy.then(|| cfg.ssh_proxy.clone()).flatten())?;
    let venv = venv
        .or_else(|| cfg.ssh_venv.clone())
        .unwrap_or_else(|| "~/venv".to_string());
    Some((host, venv))
}

/// Whether a path string names an `s3://` source — one case a configured default
/// proxy applies to (a plain local path is never routed through the config proxy).
fn is_s3_source(p: &Path) -> bool {
    p.to_string_lossy().starts_with("s3://")
}

/// A leading `:` on a positional path is the terse scp-style form for "read this over
/// my configured SSH proxy": `:/opt/models/foo` reads `/opt/models/foo` on the
/// `ssh_proxy` host from the config file. It complements the explicit
/// `[user@]host:/path` form (which carries its own host); unlike a bare local path
/// it's an unambiguous opt-in, so a same-named local directory is never silently
/// routed off-host.
fn config_proxy_prefixed(p: &Path) -> bool {
    p.to_string_lossy().starts_with(':')
}

/// Strip the leading `:` config-proxy marker (a no-op when absent).
fn strip_config_proxy_prefix(p: &Path) -> PathBuf {
    p.to_string_lossy()
        .strip_prefix(':')
        .map_or_else(|| p.to_path_buf(), PathBuf::from)
}

/// Resolve a command's positional sources + its effective SSH proxy in one place:
/// strip any `:` remote markers, then decide whether the read is remote — an explicit
/// `--ssh-proxy` (any source), or the config `ssh_proxy` default for an `s3://` URI or
/// a `:`-prefixed path. A `:`-prefixed source with no proxy resolvable is an error
/// (the `:` explicitly asked to go remote; don't silently read a local path instead).
/// Returns the de-prefixed paths and the proxy (`None` = local read).
fn resolve_remote_sources(
    paths: &[PathBuf],
    proxy: Option<String>,
    venv: Option<String>,
    cfg: &cli_config::CliConfig,
) -> Result<(Vec<PathBuf>, Option<SshProxy>)> {
    let prefixed = paths.iter().any(|p| config_proxy_prefixed(p));
    let needs_proxy = prefixed || paths.iter().any(|p| is_s3_source(p));
    let stripped: Vec<PathBuf> = paths.iter().map(|p| strip_config_proxy_prefix(p)).collect();
    // A path in scp form carries its own host: take it off, and let it stand in for `--ssh-proxy`.
    // Without this, `diff --ssh-proxy H H:/path` kept the host on the path and had it prefixed again.
    let (stripped, own_host) = split_off_scp_host(&stripped, proxy.as_deref())?;
    // A host on the path is as explicit as the flag, so it also decides that the read is remote.
    let has_own_host = own_host.is_some();
    let remote = resolve_ssh_proxy(proxy.or(own_host), venv, cfg, needs_proxy || has_own_host);
    if prefixed && remote.is_none() {
        let where_ = cli_config::CliConfig::path().map_or_else(
            || "the config file".to_string(),
            |p| p.display().to_string(),
        );
        anyhow::bail!(
            "`:PATH` reads over the SSH proxy, but none is configured — \
             set `ssh_proxy` in {where_}, or pass --ssh-proxy <HOST>"
        );
    }
    Ok((stripped, remote))
}

/// Split a `SOURCE#subtree` re-root suffix off a checkpoint source: `#language_model`
/// scopes the read to that subtree and re-roots it (see [`reroot_tensors`]). Split on
/// the LAST `#` — a subtree is always a dotted tensor-name prefix (no `#`), so a `#`
/// inside a path/key is preserved. `s3://` URIs and scp/`:` markers never contain `#`.
fn split_reroot(p: &Path) -> (PathBuf, Option<String>) {
    match p.to_string_lossy().rsplit_once('#') {
        Some((addr, sub)) if !sub.is_empty() => (PathBuf::from(addr), Some(sub.to_string())),
        _ => (p.to_path_buf(), None),
    }
}

/// The comparison *key* for a tensor under a `#subtree` re-root: `Some(sub-path)`
/// when the tensor is inside the subtree, `None` when it's a sibling to leave out of
/// scope. A `None` root keeps every name as-is. This is a scope change, not a rename:
/// only the *match key* moves; each tensor keeps its real name for data I/O.
fn scope_key(name: &str, root: Option<&str>) -> Option<String> {
    root.map_or_else(
        || Some(name.to_string()),
        |p| {
            name.strip_prefix(&format!("{}.", p.trim_end_matches('.')))
                .map(str::to_string)
        },
    )
}

/// A scoped copy of a tensor list for the structural summary: keep only the tensors
/// inside `prefix` and re-key them to their sub-path, so the summary's names and
/// totals describe the subtree. The originals are untouched (kept for value I/O).
pub(crate) fn scope_tensors(tensors: &[TensorInfo], prefix: &str) -> Vec<TensorInfo> {
    tensors
        .iter()
        .filter_map(|t| {
            scope_key(&t.name, Some(prefix)).map(|name| TensorInfo { name, ..t.clone() })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)] // a CLI entry point; each arg is a distinct flag
fn run_diff(
    old: &Path,
    new: &Path,
    recursive: bool,
    tensor: Option<&str>,
    view: sample::ViewDtype,
    bins: Option<usize>,
    mut opts: diff::DiffOpts,
    filter: &diff::TensorFilter,
    name_map: &diff::NameMap,
    jobs: usize,
    remote: Option<&remote::RemoteRead>,
    verify_repack: bool,
    repack_bits: Option<usize>,
    old_root: Option<&str>,
    new_root: Option<&str>,
    // `--align-fused`: the canonical unfused→fused rules, applied to **both** sides with folding.
    align: Option<&diff::NameMap>,
) -> i32 {
    const MAX_SCHEMA_LINES: usize = 40;
    let load_local = |path: &Path| -> Result<SideLoad> {
        let (files, _index_specs) =
            collect_safetensors_files(std::slice::from_ref(&path.to_path_buf()), recursive, true)?;
        if files.is_empty() {
            anyhow::bail!("no checkpoint files found at {}", path.display());
        }
        // `diff` compares structure only — the config sidecar isn't needed here.
        let (parts, _cp) = Explorer::gather_checkpoint(&files, None)?;
        let opening::CheckpointParts {
            tensors, metadata, ..
        } = parts;
        Ok(((tensors, metadata), None)) // local sources have no S3 object metadata
    };

    let (old_str, new_str) = (old.to_string_lossy(), new.to_string_lossy());
    // Both sides `s3://`: the pair the proxy can decode, and the pair whose S3 object
    // metadata is worth comparing.
    let s3_pair = old_str.starts_with("s3://") && new_str.starts_with("s3://");
    // Whether `--verify-repack` can run at all — asked **before** either side is read.
    // It depends on the two specs and the proxy, nothing else, and refusing after two
    // slow structure reads and a printed diff looked like a run that simply found
    // nothing to verify. Shared with the web's job, so both refuse in the same words.
    if verify_repack && let Err(e) = compare::repack_supported(remote.is_some(), s3_pair) {
        eprintln!("checkpoint-studio diff: {e:#}");
        return 2;
    }
    // Both checkpoints are read **in parallel**, remote or local: the two sides are independent, and
    // a comparison should cost the slower of them rather than their sum. Remote gets a session each
    // (ssh2 sessions aren't Sync) and a spinner line each; the password is entered once and reused,
    // so it is still one prompt, and agent/key auth needs none. Local reads print nothing, so two at
    // once need no coordination — a directory of shards is a rayon fan-out either way, and on a
    // network filesystem two of them overlap rather than queue.
    // The SSH password is entered once (on the first session) and reused for every
    // later session — the two parallel structure reads and, for an s3-vs-s3 value
    // diff, the comparison session opened afterwards — so the whole run is one prompt.
    let mut password: Option<String> = None;
    let loaded: Result<(SideLoad, SideLoad)> = remote.map_or_else(
        || {
            let (a, b) = std::thread::scope(|scope| {
                let ta = scope.spawn(|| {
                    load_local(old).with_context(|| format!("reading {}", old.display()))
                });
                let tb = scope.spawn(|| {
                    load_local(new).with_context(|| format!("reading {}", new.display()))
                });
                (ta.join(), tb.join())
            });
            let panicked = || anyhow::anyhow!("the thread reading a checkpoint panicked");
            Ok((a.map_err(|_| panicked())??, b.map_err(|_| panicked())??))
        },
        |r| {
            (|| -> Result<(SideLoad, SideLoad)> {
                // Open both sessions up front so the one password prompt happens here,
                // before the spinner. Opening is silent, so nothing is printed until
                // we're actually connected — then announce the read (not before, when
                // we're still authenticating and nothing is being read yet).
                let sa = r.open_with(&mut password)?;
                let sb = r.open_with(&mut password)?;
                utils::eprint_note(
                    "checkpoint-studio diff: ",
                    "reading each checkpoint's tensor list over ssh (names/dtypes/shapes \
                     only — no tensor data is transferred) …",
                );
                let bars = progress::Bars::start(&[old_str.to_string(), new_str.to_string()]);
                // If one side fails to load, there's no point finishing the *other*
                // side's (slow) S3-object-metadata scan — the diff can't proceed. A
                // failing read trips this flag; the sibling's read loop checks it between
                // streamed progress lines and bails promptly.
                let abort = std::sync::atomic::AtomicBool::new(false);
                let read =
                    |session: &sftp::RemoteSession, src: &str, i: usize| -> Result<SideLoad> {
                        let progress = bars.progress(i);
                        let out = r
                            // `diff` compares S3 object metadata for s3-vs-s3 → fetch it.
                            .read(
                                session,
                                src,
                                &password,
                                progress.as_deref(),
                                remote::ObjectMeta::Fetch,
                                Some(&abort),
                            )
                            .with_context(|| format!("reading {src}"));
                        // A failure trips the abort so the sibling stops promptly.
                        if out.is_err() {
                            abort.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        // Distinguish an abort (this side was cut short — dim `⊘`) from a
                        // real failure (`✗`), so a fine-but-cancelled read doesn't look
                        // broken.
                        match &out {
                            Ok(_) => bars.finish(i, true),
                            Err(e) if is_aborted_err(e) => bars.abort(i),
                            Err(_) => bars.finish(i, false),
                        }
                        // `diff` compares structure + (for s3://) S3 object metadata, not
                        // the on-disk footprint or health.
                        out.map(|rc| ((rc.tensors, rc.metadata), rc.s3))
                    };
                let (ra, rb) = std::thread::scope(|s| {
                    let (oref, nref): (&str, &str) = (&old_str, &new_str);
                    let ta = s.spawn(|| read(&sa, oref, 0));
                    let tb = s.spawn(|| read(&sb, nref, 1));
                    (ta.join(), tb.join())
                });
                bars.join();
                let ra = ra.map_err(|_| anyhow::anyhow!("remote read thread panicked"))?;
                let rb = rb.map_err(|_| anyhow::anyhow!("remote read thread panicked"))?;
                match (ra, rb) {
                    (Ok(a), Ok(b)) => Ok((a, b)),
                    (ra, rb) => {
                        // At least one side failed. Prefer the *real* failure over an
                        // abort-induced one (the sibling we cut short), so the reported
                        // error names the checkpoint that actually couldn't load.
                        Err(match (ra.err(), rb.err()) {
                            (Some(ea), Some(eb)) => {
                                if is_aborted_err(&ea) {
                                    eb
                                } else {
                                    ea
                                }
                            }
                            (Some(e), None) | (None, Some(e)) => e,
                            (None, None) => unreachable!("matched the failure arm"),
                        })
                    }
                }
            })()
        },
    );
    let (((old_t, old_m), old_s3), ((new_t, new_m), new_s3)) = match loaded {
        Ok(v) => v,
        Err(e) => {
            eprintln!("checkpoint-studio diff: {e:#}");
            return 2;
        }
    };

    // A `#subtree` re-root descends into a subtree and compares from there. Its value
    // comparison over --ssh-proxy runs on the remote *by tensor name*, and that path
    // doesn't yet re-root the names it fetches — so a scoped value comparison is
    // supported locally (data is read by byte offset, name-independent) but not over
    // the proxy. Refuse that one combination rather than mislead.
    if remote.is_some()
        && (old_root.is_some() || new_root.is_some())
        && (opts.values || opts.histogram || tensor.is_some())
    {
        eprintln!(
            "checkpoint-studio diff: over --ssh-proxy a '#subtree' re-root compares structure \
             only (the remote value comparison keys by tensor name) — drop --values / \
             --histogram / --tensor, or run the scoped value diff on local checkpoints"
        );
        return 2;
    }

    // Show the re-root in the diff header so a scoped comparison is self-explanatory.
    let label = |p: &Path, root: Option<&str>| {
        root.map_or_else(
            || p.display().to_string(),
            |r| format!("{}#{r}", p.display()),
        )
    };
    let (old_label, new_label) = (label(old, old_root), label(new, new_root));

    // Packing schemas (for the `unpacked` view) come from the full metadata —
    // independent of `--only-tensors`, which only hides the metadata *diff*. Only
    // needed when values / distributions are compared.
    let compares_data = opts.values || opts.histogram || tensor.is_some();
    let mut compares_data_unavailable = false;

    // A remote source's tensor *data* isn't reachable locally, so value/distribution
    // comparison for `--ssh-proxy` must run on the proxy. Only an s3-vs-s3 pair is
    // supported (both cstorch checkpoints the remote can load); for a non-s3 remote
    // (a safetensors dir) or a mixed pair, fall back to a structural diff and say so.
    // A side whose source cannot give us tensor bytes can't be value-compared at all. Asked
    // as a capability so every source answers the same way — this is what caught `hf://`,
    // which reads structure over HTTPS and would otherwise have fallen through to a local
    // read of a path that isn't one.
    if compares_data && remote.is_none() {
        for (label, spec) in [("OLD", &*old_str), ("NEW", &*new_str)] {
            let location = if hf::is_uri(spec) {
                capability::Location::Hf
            } else if s3::is_uri(spec) {
                capability::Location::S3
            } else {
                capability::Location::Local
            };
            if !capability::Capabilities::of(capability::Format::Safetensors, location).read_bytes {
                eprintln!(
                    "checkpoint-studio diff: {label} ({spec}) can't supply tensor bytes, so \
                     values can't be compared — comparing structure only.\n  {}",
                    capability::Capabilities::data_view_note(location).unwrap_or_default()
                );
                compares_data_unavailable = true;
                // Clear the request too, so the report's "scope:" header says element
                // values were NOT compared. Leaving it set printed a scope line claiming a
                // comparison that had just been refused two lines above.
                opts.values = false;
                opts.histogram = false;
            }
        }
    }
    let compares_data = compares_data && !compares_data_unavailable;

    // **How the proxy reads each side.** Under `--ssh-proxy` both operands are already proxy-relative —
    // the host was split off before either was read — so each is either an `s3://` URI (cstorch) or a
    // path on that host (safetensors), and the proxy can open both. That is what makes a value
    // comparison possible for a pair that is *entirely* over there, which used to be refused unless
    // both sides were `s3://`. The browser asks the same question of specs that still carry their host
    // (`compare::values_where`), so the two surfaces compare the same pairs.
    let remote_sides = remote.map(|_| {
        (
            compare::proxy_side_of(&old_str),
            compare::proxy_side_of(&new_str),
        )
    });
    let remote_values = match (remote, compares_data) {
        (Some(_), true) => {
            // **The proxy compares what is stored, and applies no decode.** `--dtype u4/unpacked/…`
            // are local decodes, and a remote comparison has no access to them; saying the remote's
            // tensors "are already the logical values" was a claim about *unquantized* checkpoints that
            // does not hold for a quantized one — where the stored array is indices or scaled integers
            // and the logical value needs a qscale (or a codebook) applied.
            if !matches!(view, sample::ViewDtype::Stored) {
                eprintln!(
                    "checkpoint-studio diff: --dtype is ignored on the proxy — a remote comparison \
                     reads what each side stores, with no decode applied"
                );
            }
            true
        }
        _ => false,
    };

    // Heads-up when both sides' S3 objects are byte-identical (same key + ETag) —
    // e.g. diffing a checkpoint against itself. The comparison still runs (so it can
    // be exercised / confirm identity); this just flags that it will read the data
    // and can only report "identical". The structure read already had the metadata,
    // so the check is free.
    if remote_values
        && let (Some(o), Some(n)) = (&old_s3, &new_s3)
        && s3_objects_identical(o, n)
    {
        utils::eprint_note(
            "checkpoint-studio diff: ",
            "note — both sides are byte-identical (same S3 objects); the value comparison \
             will read the data and confirm every tensor is identical. Pass two different \
             checkpoints to see real value differences.",
        );
    }

    let (old_schemas, new_schemas) = if compares_data && !remote_values {
        (
            sample::parse_packing_schemas(&old_t, &old_m),
            sample::parse_packing_schemas(&new_t, &new_m),
        )
    } else {
        (HashMap::new(), HashMap::new())
    };
    let ctx = ValueCtx {
        view,
        bins,
        old_schemas: &old_schemas,
        new_schemas: &new_schemas,
    };

    // `--tensor NAME`: focus on one tensor and also compare its element values.
    // (This single-tensor mode is its own selection; the subset filters apply to
    // the whole-checkpoint diff below, so note if both were given.)
    if let Some(name) = tensor {
        if filter.is_active() {
            eprintln!("checkpoint-studio diff: --tensor takes precedence; filters ignored");
        }
        if !name_map.is_empty() {
            eprintln!("checkpoint-studio diff: --map is ignored with --tensor");
        }
        if verify_repack {
            eprintln!("checkpoint-studio diff: --verify-repack is ignored with --tensor");
        }
        // Remote (s3-vs-s3): compare this one tensor's values/distribution on the
        // proxy. `full_hist` so the bin-by-bin table can be rendered locally.
        let remote_diff = if remote_values {
            let a = old_t.iter().find(|t| t.name == name);
            let b = new_t.iter().find(|t| t.name == name);
            match (a, b) {
                (Some(a), Some(b)) if a.shape == b.shape => {
                    let vopts = remote::RemoteValueOpts {
                        values: true,
                        histogram: opts.histogram,
                        bins,
                        full_hist: true,
                        jobs: jobs.clamp(1, 32),
                    };
                    #[allow(clippy::expect_used)]
                    let (old_side, new_side) = remote_sides
                        .as_ref()
                        .expect("remote_values implies two proxy-readable sides");
                    match fetch_remote_value_diff(
                        // `remote_values` is only set on the branch that has a remote (the
                        // `--ssh-proxy` path); the two travel together.
                        #[allow(clippy::expect_used)]
                        remote.expect("remote_values implies a remote"),
                        &mut password,
                        old_side,
                        new_side,
                        &[(name.to_string(), name.to_string())],
                        &vopts,
                        (a.size_bytes + b.size_bytes) as u64,
                    ) {
                        Ok(mut m) => m.remove(name),
                        Err(e) => {
                            eprintln!("checkpoint-studio diff: {e:#}");
                            return 2;
                        }
                    }
                }
                _ => None,
            }
        } else {
            None
        };
        return run_diff_tensor(
            &old_label,
            &new_label,
            name,
            &old_t,
            &new_t,
            &ctx,
            opts,
            remote_values,
            remote_diff.as_ref(),
        );
    }

    // `--only-tensors` / an active filter (opts.metadata == false): drop metadata
    // so it affects neither the report nor the exit code (its section becomes a
    // "not compared" note in the output).
    let empty: Vec<MetadataInfo> = Vec::new();
    let old_meta: &[MetadataInfo] = if opts.metadata { &old_m } else { &empty };
    let new_meta: &[MetadataInfo] = if opts.metadata { &new_m } else { &empty };
    // A `#subtree` re-root scopes the structural summary to that subtree (names +
    // totals describe it); the originals stay untouched for value I/O. A prefix that
    // matches nothing is a likely typo → stop with a clear message.
    let (old_scoped, new_scoped);
    let old_sum_src: &[TensorInfo] = match old_root {
        None => &old_t,
        Some(p) => {
            old_scoped = scope_tensors(&old_t, p);
            &old_scoped
        }
    };
    let new_sum_src: &[TensorInfo] = match new_root {
        None => &new_t,
        Some(p) => {
            new_scoped = scope_tensors(&new_t, p);
            &new_scoped
        }
    };
    for (src, root, label) in [
        (old_sum_src, old_root, &old_str),
        (new_sum_src, new_root, &new_str),
    ] {
        if let Some(prefix) = root
            && src.is_empty()
        {
            eprintln!(
                "checkpoint-studio diff: '#{prefix}' matched no tensors in {label} — \
                 no names start with '{prefix}.'"
            );
            return 2;
        }
    }
    let mut old_sum = diff::CheckpointSummary::from_loaded(old_sum_src, old_meta);
    let mut new_sum = diff::CheckpointSummary::from_loaded(new_sum_src, new_meta);
    // `--align-fused` first, and on **both** sides: it is a statement about layout, not about which
    // checkpoint is which, and every rule is a no-op on a side already fused. Folding, because the
    // whole point is that 256 per-expert tensors *are* the one fused tensor holding them.
    if let Some(align) = align {
        for sum in [&mut old_sum, &mut new_sum] {
            align.remap_summary_with(sum, diff::OnCollision::Fold);
        }
        let folds = old_sum.folds().len() + new_sum.folds().len();
        utils::eprint_note(
            "checkpoint-studio diff: ",
            &format!(
                "aligned: {folds} name(s) stand for several tensors after folding (shown as ×N on \
                 their row)"
            ),
        );
    }
    // Rename rules (`--map` / `--map-from`) rewrite the OLD side's tensor names
    // into the NEW side's naming scheme, so corresponding tensors line up in the
    // comparison below instead of showing as a removed/added pair. Applied before
    // the filter (which matches on the post-rename names, as they appear in the
    // report). A rule broad enough to collide two names onto one is warned about.
    if !name_map.is_empty() {
        let collisions = name_map.remap_summary(&mut old_sum);
        eprintln!(
            "checkpoint-studio diff: applied {} rename rule(s) to {old_label}",
            name_map.len()
        );
        for target in &collisions {
            utils::eprint_note(
                "checkpoint-studio diff: ",
                &format!(
                    "warning: a rename rule maps multiple tensors onto {target:?} \
                     (keeping the last)"
                ),
            );
        }
    }
    // Total distinct tensors across both sides *before* filtering, so the filter's
    // match line can show "matched M of N" (context for whether M looks right).
    let total_tensors = old_sum
        .tensors
        .keys()
        .chain(new_sum.tensors.keys())
        .collect::<HashSet<_>>()
        .len();
    // Scope the diff to the selected subset (no-op when no filter was given).
    filter.apply(&mut old_sum, &mut new_sum);

    // `(compared, differing)` tensor counts for the explicit value-comparison summary
    // (so the output states "values identical" rather than leaving it implicit in the
    // unchanged count). Set in the `--values`/`--histogram` branch below.
    let mut value_summary: Option<(usize, usize)> = None;
    // Auto-detected sparse-packed expert weights (`--values`): compared as N-bit
    // indices (repack-style), rendered in their own section after the report.
    let mut sparse_pairs: Vec<(String, String)> = Vec::new();
    let mut sparse_bits = 0usize;
    let mut sparse_results: HashMap<String, remote::RepackResult> = HashMap::new();

    let mut report = if opts.values || opts.histogram {
        use rayon::prelude::*;
        // Keyed by the *match* name (re-root sub-path, then any `--map` rename — the
        // same key space `common` and the summaries use), while the value keeps the
        // original `TensorInfo` (real name + byte offsets) for data I/O. An
        // out-of-scope tensor (re-root sibling) is dropped from the key space.
        let old_map: HashMap<String, &TensorInfo> = old_t
            .iter()
            .filter_map(|t| {
                scope_key(&t.name, old_root).map(|s| (name_map.map(&s).into_owned(), t))
            })
            .collect();
        let new_map: HashMap<String, &TensorInfo> = new_t
            .iter()
            .filter_map(|t| scope_key(&t.name, new_root).map(|s| (s, t)))
            .collect();
        // The tensors present on both sides — the ones we actually read/compare.
        let mut common: Vec<&str> = old_sum
            .tensors
            .keys()
            .filter(|k| new_sum.tensors.contains_key(*k))
            .map(String::as_str)
            .collect();

        // Auto-detect sparse-packed expert weights (a `*.weight` with a sibling
        // `*.codebook`): those are quantized indices, not floats — compare them as
        // N-bit indices below (repack-style), so drop them and their codebook/qscale
        // siblings from the plain value pass (their raw bits as F16 are meaningless).
        if opts.values {
            let (fp, fb, handled) = auto_sparse_families(&common, &old_map, &new_map);
            if let Some(b) = fb {
                sparse_pairs = fp;
                sparse_bits = b;
                common.retain(|k| !handled.contains(*k));
            }
        }

        // The per-tensor extras, keyed by (post-rename) name — computed on the proxy
        // (s3-vs-s3) or locally.
        let extras: HashMap<String, diff::TensorExtras> = if remote_values {
            // One proxy script reads both checkpoints and compares each common
            // same-shape pair; the pair carries (old original name, new/common name).
            let pairs: Vec<(String, String)> = common
                .iter()
                .filter_map(|&k| {
                    let a = old_map.get(k)?;
                    let b = new_map.get(k)?;
                    (a.shape == b.shape).then(|| (a.name.clone(), k.to_string()))
                })
                .collect();
            // Data to read (both sides, stored dtype) — for the view total + intro.
            let total_bytes: u64 = pairs
                .iter()
                .filter_map(|(_, k)| {
                    Some(
                        (old_map.get(k.as_str())?.size_bytes + new_map.get(k.as_str())?.size_bytes)
                            as u64,
                    )
                })
                .sum();
            let vopts = remote::RemoteValueOpts {
                values: opts.values,
                histogram: opts.histogram,
                bins,
                full_hist: false, // bulk path needs only the TVD summary
                jobs: jobs.clamp(1, 32),
            };
            if pairs.is_empty() {
                // Everything was auto-detected as sparse-packed (compared as indices
                // below) — don't load the checkpoints again for a no-op float pass.
                HashMap::new()
            } else {
                #[allow(clippy::expect_used)]
                let (old_side, new_side) = remote_sides
                    .as_ref()
                    .expect("remote_values implies two proxy-readable sides");
                match fetch_remote_value_diff(
                    // `remote_values` is only set on the branch that has a remote (the
                    // `--ssh-proxy` path); the two travel together.
                    #[allow(clippy::expect_used)]
                    remote.expect("remote_values implies a remote"),
                    &mut password,
                    old_side,
                    new_side,
                    &pairs,
                    &vopts,
                    total_bytes,
                ) {
                    Ok(m) => m
                        .into_iter()
                        .map(|(k, rd)| (k, extras_from_remote(&rd)))
                        .collect(),
                    Err(e) => {
                        eprintln!("checkpoint-studio diff: {e:#}");
                        return 2;
                    }
                }
            }
        } else {
            let progress = CompareProgress::start(common.len());
            let compute = |name: &str| -> diff::TensorExtras {
                let _tracked = progress.track(name);
                let (Some(a), Some(b)) = (old_map.get(name), new_map.get(name)) else {
                    return diff::TensorExtras::default();
                };
                // Shared with the web's values job, so `--values` means the same thing in both.
                compare::tensor_extras(
                    a,
                    b,
                    &compare::ValueOpts {
                        view: ctx.view,
                        bins: ctx.bins,
                        values: opts.values,
                        histogram: opts.histogram,
                        old_schemas: ctx.old_schemas,
                        new_schemas: ctx.new_schemas,
                    },
                )
            };
            // Reading tensor data is I/O-bound, so compare up to `jobs` tensors at
            // once (the results are order-independent). `jobs == 1` stays sequential.
            let pairs: Vec<(&str, diff::TensorExtras)> = if jobs <= 1 {
                common.iter().map(|&n| (n, compute(n))).collect()
            } else {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(jobs)
                    .build()
                    .map_or_else(
                        |_| common.iter().map(|&n| (n, compute(n))).collect(),
                        |pool| {
                            pool.install(|| common.par_iter().map(|&n| (n, compute(n))).collect())
                        },
                    )
            };
            progress.finish();
            pairs.into_iter().map(|(n, e)| (n.to_string(), e)).collect()
        };
        // Count compared tensors and how many actually differ in value/distribution,
        // for the explicit summary line below.
        let compared = extras.len();
        let differ = extras
            .values()
            .filter(|e| {
                e.values.is_some_and(|v| v.differing > 0)
                    || e.histogram.is_some_and(|h| h.tvd > 0.0)
            })
            .count();
        // Only the plain-float tensors count toward the "element values" verdict;
        // the sparse-packed families get their own section. Suppress the float
        // verdict when there were none (else it wrongly says "nothing to compare").
        value_summary = (compared > 0 || sparse_pairs.is_empty()).then_some((compared, differ));
        // Compare the auto-detected sparse-packed expert weights as N-bit indices
        // (repack-style, on the proxy for s3:// or locally) — reads their data once.
        if !sparse_pairs.is_empty() {
            sparse_results = run_auto_sparse(
                remote,
                &mut password,
                s3_pair,
                &old_str,
                &new_str,
                &old_t,
                &new_t,
                &sparse_pairs,
                sparse_bits,
            );
        }
        // Feed the precomputed extras into the (pure) comparison. Each common name
        // is requested exactly once, so `remove` moves the value out (no clone).
        let extras = std::cell::RefCell::new(extras);
        diff::compare_with(&old_sum, &new_sum, |name| {
            extras.borrow_mut().remove(name).unwrap_or_default()
        })
    } else {
        diff::compare(&old_sum, &new_sum)
    };
    // When a filter scoped the diff, say what it selected on stderr (so the diff
    // on stdout stays clean for piping): the match count — disambiguating an empty
    // diff caused by "0 matched" from "all identical" — plus the matched names
    // collapsed into their index-templated schema, so it's clear which layers /
    // experts the filter actually covered.
    if let Some(desc) = filter.describe() {
        // The matched set is the union of both sides (so it includes structurally
        // unchanged tensors, which the report only counts, not names).
        let mut names: Vec<&str> = old_sum
            .tensors
            .keys()
            .chain(new_sum.tensors.keys())
            .map(String::as_str)
            .collect();
        names.sort_unstable();
        names.dedup();
        if names.is_empty() {
            eprintln!(
                "checkpoint-studio diff: filter [{desc}] matched 0 of {total_tensors} tensor(s)"
            );
        } else {
            eprintln!(
                "checkpoint-studio diff: filter [{desc}] matched {} of {total_tensors} tensor(s):",
                names.len()
            );
            let schema = diff::name_schema(&names);
            for (tmpl, count) in schema.iter().take(MAX_SCHEMA_LINES) {
                if *count > 1 {
                    eprintln!("    {tmpl}  (×{count})");
                } else {
                    eprintln!("    {tmpl}");
                }
            }
            if schema.len() > MAX_SCHEMA_LINES {
                eprintln!(
                    "    … and {} more template(s)",
                    schema.len() - MAX_SCHEMA_LINES
                );
            }
        }
    }
    // S3 object metadata is compared only when BOTH sides are `s3://` (only then do
    // both carry it). last-modified / timestamp-like deltas are informational and
    // never affect the exit code (see `diff::compare_s3`). Shared with the web's
    // report, which showed no S3 section at all until it called this.
    if let Some(note) = compare::attach_s3(&mut report, old_s3.as_ref(), new_s3.as_ref()) {
        eprintln!("checkpoint-studio diff: {note}");
    }
    print!("{}", report.render(&old_label, &new_label, opts));

    // Explicit value-comparison verdict (so "values identical" is stated, not just
    // implied by the unchanged count).
    if let Some((compared, differ)) = value_summary {
        let what = if opts.values && opts.histogram {
            "values/distribution"
        } else if opts.histogram {
            "distribution"
        } else {
            "element values"
        };
        if compared == 0 {
            println!("{what}: no common same-shape tensor(s) to compare");
        } else if differ == 0 {
            println!("{what}: all {compared} compared tensor(s) IDENTICAL");
        } else {
            println!("{what}: {differ} of {compared} compared tensor(s) differ");
        }
    }

    // `--verify-repack`: after the structural diff, confirm the shape-folded expert
    // tensors encode the same indices in old (sparse) vs new (dense) packing. Runs on
    // the proxy (s3-vs-s3). Its verdict drives the exit code when requested.
    if verify_repack {
        return run_repack_verify(
            remote,
            &mut password,
            &old_str,
            &new_str,
            &old_t,
            &new_t,
            &old_sum,
            &new_sum,
            repack_bits,
        );
    }
    // Auto-detected sparse-packed expert weights (`--values`): their index / value
    // comparison prints its own section and contributes to the exit code.
    let sparse_differs = if sparse_pairs.is_empty() {
        false
    } else {
        render_auto_sparse(&sparse_pairs, &sparse_results) == 1
    };
    // Under a `--name` filter the exit code reflects the compared tensor subset
    // only; whole-prefix S3 object-metadata deltas (e.g. a re-uploaded `__METADATA__`
    // or last-modified bumps) are out of that scope, like the metadata section.
    i32::from(report.has_differences_with(!opts.filtered) || sparse_differs)
}

/// The `diff --verify-repack` path: find tensors present on both sides whose shapes
/// fold along dim 0 (already scoped by `--name`, in `old_sum`/`new_sum`), verify
/// their packed indices match — on the ssh proxy for `s3://` sources, or locally for
/// local checkpoint files — and print a verdict. Returns the exit code: 0 = every
/// matched tensor is equivalent (and nothing else differs), 1 otherwise, 2 on trouble.
///
/// Where the data has to be reachable from was settled before the reads
/// ([`compare::repack_supported`]), so a remote here is an s3-vs-s3 pair the proxy can decode.
#[allow(clippy::too_many_arguments)]
fn run_repack_verify(
    remote: Option<&remote::RemoteRead>,
    password: &mut Option<String>,
    old_uri: &str,
    new_uri: &str,
    old_t: &[TensorInfo],
    new_t: &[TensorInfo],
    old_sum: &diff::CheckpointSummary,
    new_sum: &diff::CheckpointSummary,
    repack_bits: Option<usize>,
) -> i32 {
    // Candidate pairs, bit width and "does anything else differ" — shared with the web's job, so the
    // browser verifies exactly the pairs a terminal would (`compare::plan_repack`).
    let plan = match compare::plan_repack(old_sum, new_sum, repack_bits) {
        Ok(plan) => plan,
        Err(e) => {
            eprintln!("checkpoint-studio diff: --verify-repack: {e:#}");
            return 2;
        }
    };
    let (pairs, bits, other_differs) = (plan.pairs, plan.bits, plan.other_differs);

    let results = if let Some(r) = remote {
        let labels = repack_bar_labels(&pairs, old_t, new_t);
        match fetch_remote_repack(r, password, old_uri, new_uri, &pairs, &labels, bits, false) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("checkpoint-studio diff: {e:#}");
                return 2;
            }
        }
    } else {
        local_repack(old_t, new_t, &pairs, bits, None)
    };
    render_repack_verdict(&pairs, &results, bits, other_differs)
}

/// Verify the fold-pairs locally (reading the checkpoint files here): decode both
/// sides' indices, compare, and diff the sibling codebook / scale tensors. Sequential
/// — each expert weight is read whole, so memory is one pair at a time.
fn local_repack(
    old_t: &[TensorInfo],
    new_t: &[TensorInfo],
    pairs: &[(String, String)],
    bits: usize,
    fold_override: Option<usize>,
) -> HashMap<String, remote::RepackResult> {
    let find = |ts: &[TensorInfo], n: &str| ts.iter().find(|t| t.name == n).cloned();
    let mut out = HashMap::new();
    for (oname, nname) in pairs {
        let (Some(ow), Some(nw)) = (find(old_t, oname), find(new_t, nname)) else {
            continue;
        };
        // Auto sparse↔sparse forces fold 1 (same shape both sides); --verify-repack
        // derives the fold from the shrinking dim-0.
        let Some(fold) = fold_override.or_else(|| detect_fold(&ow.shape, &nw.shape)) else {
            continue;
        };
        // Sibling codebook / scale tensor names (weight name minus ".weight").
        let sib = |name: &str, kind: &str| {
            name.strip_suffix(".weight")
                .map(|p| format!("{p}.{kind}"))
                .unwrap_or_default()
        };
        let (ocb, ncb) = (sib(oname, "codebook"), sib(nname, "codebook"));
        let (oqs, nqs) = (sib(oname, "qscale"), sib(nname, "qscale"));
        let rr = repack::verify_local(
            &ow,
            &nw,
            fold,
            bits,
            (
                &ocb,
                &ncb,
                find(old_t, &ocb).as_ref(),
                find(new_t, &ncb).as_ref(),
            ),
            (
                &oqs,
                &nqs,
                find(old_t, &oqs).as_ref(),
                find(new_t, &nqs).as_ref(),
            ),
        );
        out.insert(nname.clone(), rr);
    }
    out
}

/// Index bit-width from a codebook tensor's centroid count (its last dim): the bits
/// needed to address one centroid, `ceil(log2(centroids))` — `8`→3, `16`→4,
/// `2048`→11. `None` for a degenerate shape (detection then declines).
fn codebook_index_bits(cb_shape: &[usize]) -> Option<usize> {
    let c = *cb_shape.last()?;
    (c >= 2).then(|| ((c - 1).ilog2() + 1) as usize)
}

/// Detect sparse-packed expert-weight families for the auto `--values` path: a
/// `*.weight` present (equal shape) on both sides whose sibling `*.codebook` is also
/// present on both sides marks the weight as quantized indices. Returns the
/// `(old_name, new_name)` weight pairs, the index width `N` from the codebook's
/// centroid count (its last dim; assumed uniform, taken from the first family), and
/// the set of post-rename names handled here (each weight + its `.codebook`/`.qscale`
/// siblings) so the float value pass skips them — they're compared as indices (or,
/// if the top-bits check fails, as a fallback value diff) with their siblings.
fn auto_sparse_families(
    common: &[&str],
    old_map: &HashMap<String, &TensorInfo>,
    new_map: &HashMap<String, &TensorInfo>,
) -> (Vec<(String, String)>, Option<usize>, HashSet<String>) {
    let mut pairs = Vec::new();
    let mut bits = None;
    let mut handled = HashSet::new();
    for &k in common {
        let Some(stem) = k.strip_suffix(".weight") else {
            continue;
        };
        let cb = format!("{stem}.codebook");
        // The weight and its codebook must be present on both sides (the common key
        // space is post-rename, matching both maps).
        let (Some(ow), Some(_nw)) = (old_map.get(k), new_map.get(k)) else {
            continue;
        };
        let (Some(_ocb), Some(ncb)) = (old_map.get(cb.as_str()), new_map.get(cb.as_str())) else {
            continue;
        };
        let Some(n) = codebook_index_bits(&ncb.shape) else {
            continue;
        };
        bits.get_or_insert(n);
        pairs.push((ow.name.clone(), k.to_string())); // old original name, new/common name
        handled.insert(k.to_string());
        handled.insert(cb);
        handled.insert(format!("{stem}.qscale"));
    }
    (pairs, bits, handled)
}

/// The ordered set of tensors that get their own remote download bar: each weight,
/// then any sibling `.codebook` / `.qscale` present on both sides (the proxy streams
/// those too, so they shouldn't download invisibly). Labels are the *new* names —
/// the key the proxy tags every event with.
fn repack_bar_labels(
    pairs: &[(String, String)],
    old_t: &[TensorInfo],
    new_t: &[TensorInfo],
) -> Vec<String> {
    let has = |ts: &[TensorInfo], n: &str| ts.iter().any(|t| t.name == n);
    let mut labels = Vec::new();
    for (oname, nname) in pairs {
        labels.push(nname.clone());
        let (Some(ostem), Some(nstem)) =
            (oname.strip_suffix(".weight"), nname.strip_suffix(".weight"))
        else {
            continue;
        };
        for kind in ["codebook", "qscale"] {
            let (oaux, naux) = (format!("{ostem}.{kind}"), format!("{nstem}.{kind}"));
            if has(old_t, &oaux) && has(new_t, &naux) {
                labels.push(naux);
            }
        }
    }
    labels
}

/// Open a session (reusing the password) and run the proxy repack verification with
/// a per-tensor download bar (weights + sibling codebook/qscale) + a final I/O line.
#[allow(clippy::too_many_arguments)]
fn fetch_remote_repack(
    r: &remote::RemoteRead,
    password: &mut Option<String>,
    old_uri: &str,
    new_uri: &str,
    pairs: &[(String, String)],
    bar_labels: &[String],
    bits: usize,
    auto_sparse: bool,
) -> Result<HashMap<String, remote::RepackResult>> {
    let session = r.open_with(password)?;
    if auto_sparse {
        utils::eprint_note(
            "checkpoint-studio diff: ",
            &format!(
                "comparing {} sparse-packed expert weight(s) as {bits}-bit indices on {} \
                 (auto-detected sibling codebook), decoding on the remote …",
                pairs.len(),
                r.host,
            ),
        );
    } else {
        utils::eprint_note(
            "checkpoint-studio diff: ",
            &format!(
                "verifying repack of {} tensor(s) on {}, decoding {bits}-bit indices on the \
                 remote (reads the full tensors, {} at a time):",
                pairs.len(),
                r.host,
                pairs.len().clamp(1, 4),
            ),
        );
        // The two URIs on their own lines: they are long, unbreakable, and the pair is
        // what you check first when a verify looks wrong.
        eprintln!("  old (sparse) {old_uri}");
        eprintln!("  new (dense)  {new_uri}");
    }
    // One bar for the whole verify, relabelled with the tensor being read — a bar per
    // tensor is thousands of them at checkpoint scale (see `ValueBar`).
    let bar = std::cell::RefCell::new(ValueBar::start(
        &format!("verifying repack on {}", r.host),
        0,
        bar_labels.len(),
    ));
    // Ctrl-C sets the same flag every remote read watches, so a stop reaches the proxy rather than only
    // the waiting here.
    let out = r.verify_repack(
        &session,
        old_uri,
        new_uri,
        pairs,
        bits,
        auto_sparse,
        None,
        |ev| bar.borrow_mut().on(ev),
    );
    // Settle the bar (even on a fatal mid-run error) so the animation thread sees it
    // finished and `join` returns.
    bar.into_inner().finish();
    let (map, stats) = out?;
    if let Some(s) = stats {
        let elapsed = std::time::Duration::from_secs_f64(s.elapsed_s.max(0.0));
        let read = utils::format_size(s.bytes as usize);
        let rate = if s.elapsed_s > 0.0 {
            format!(
                " ({}/s from S3)",
                utils::format_size((s.bytes as f64 / s.elapsed_s) as usize)
            )
        } else {
            String::new()
        };
        eprintln!(
            "checkpoint-studio diff: verified {} tensor(s) on the remote in {} · read {read}{rate}",
            s.compared,
            format_elapsed(elapsed),
        );
    }
    Ok(map)
}

/// Print the repack verification section + verdict; return the exit code: 0 =
/// every matched tensor is equivalent AND nothing else differs (`other_differs`
/// false), 1 = any mismatch / format violation / other structural change.
fn render_repack_verdict(
    pairs: &[(String, String)],
    results: &HashMap<String, remote::RepackResult>,
    bits: usize,
    other_differs: bool,
) -> i32 {
    use std::io::IsTerminal;
    let color = std::io::stdout().is_terminal();
    let (green, yellow, red, dim, reset) = if color {
        ("\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "", "", "")
    };
    println!("\nrepack verification ({bits}-bit expert indices, folded along dim 0):");
    let mut all_ok = true;
    // Few lines (gated by --name), so list them plainly, sorted.
    let mut names: Vec<&String> = pairs.iter().map(|(_, n)| n).collect();
    names.sort();
    for name in names {
        let Some(rr) = results.get(name) else {
            all_ok = false;
            println!("  {red}✗{reset} {name}  {dim}(no result returned){reset}");
            continue;
        };
        if let Some(err) = &rr.error {
            all_ok = false;
            println!("  {red}✗{reset} {name}  {red}error:{reset} {err}");
            continue;
        }
        let counts = format!(
            "fold {}, {} indices",
            rr.fold,
            utils::format_parameters(rr.elements as usize)
        );
        if rr.sparse_bad > 0 || rr.dense_bad > 0 {
            all_ok = false;
            println!(
                "  {red}✗{reset} {name}  {red}format check FAILED{reset}: \
                 {} sparse word(s) with non-zero top bits, {} dense word(s) with non-zero MSB {dim}({counts}){reset}",
                rr.sparse_bad, rr.dense_bad,
            );
            print_repack_sample(rr, red, dim, reset);
        } else if rr.differing > 0 {
            all_ok = false;
            let where_ = rr
                .first_mismatch
                .map(|(e, off, o, n)| {
                    format!(" {dim}(first: expert {e}, offset {off}: {o} vs {n}){reset}")
                })
                .unwrap_or_default();
            // If every difference is to an adjacent index (max |Δ| == 1), it's the
            // signature of an independent re-quantization, not a lossless repack.
            let adj = if rr.max_delta <= 1 {
                format!("{dim}all by ±1 (same weights, independently re-quantized){reset}")
            } else {
                format!(
                    "{dim}max |Δ| {}, {} by >1{reset}",
                    rr.max_delta,
                    utils::format_parameters(rr.differing_gt1 as usize),
                )
            };
            println!(
                "  {yellow}≠{reset} {name}  {yellow}{} of {} indices differ{reset} — {adj}{where_}",
                utils::format_parameters(rr.differing as usize),
                utils::format_parameters(rr.elements as usize),
            );
            // Aggregate magnitude + whether the average value is preserved.
            println!(
                "      {dim}Σ|Δ| {} · mean |Δ|/param {:.4} · mean index {:.4} → {:.4} (Δ {:+.4}){reset}",
                utils::format_parameters(rr.sum_abs as usize),
                rr.mean_abs,
                rr.mean_old,
                rr.mean_new,
                rr.mean_new - rr.mean_old,
            );
            print_repack_sample(rr, red, dim, reset);
        } else {
            println!("  {green}✓{reset} {name}  {dim}equivalent ({counts}){reset}");
        }
        // The sibling codebook / scale value-diff (same shape on both sides, so the
        // structural diff can't see it). A differing codebook explains index diffs.
        print_repack_aux("codebook", rr.codebook.as_ref(), green, yellow, dim, reset);
        print_repack_aux("qscale", rr.qscale.as_ref(), green, yellow, dim, reset);
    }
    let verified = pairs.len();
    if !all_ok {
        println!("{red}verdict: NOT equivalent{reset} — see the mismatches above");
        return 1;
    }
    if other_differs {
        println!(
            "{yellow}verdict: expert weights equivalent modulo packing{reset} — {verified} tensor(s) verified; \
             but other tensors/metadata differ (see the diff above)"
        );
        1
    } else {
        println!(
            "{green}verdict: equivalent modulo packing{reset} — {verified} tensor(s) verified, all indices match; \
             no other differences"
        );
        0
    }
}

/// Whether a sibling codebook/qscale diff counts as a difference (missing on a
/// side, shape mismatch, or any differing value).
fn aux_differs(aux: Option<&remote::RepackAux>) -> bool {
    aux.is_some_and(|a| !a.present() || a.shape_mismatch.is_some() || a.differing > 0)
}

/// Run the auto-detected sparse-packed index comparison over `pairs` (codebooked
/// expert weights, sparse↔sparse, fold 1, index width `bits`) — on the ssh proxy for
/// `s3://`, else locally — reusing the repack machinery. Keyed by the new name;
/// empty on a remote error (already reported) or a non-s3 remote (data unreachable).
#[allow(clippy::too_many_arguments)]
fn run_auto_sparse(
    remote: Option<&remote::RemoteRead>,
    password: &mut Option<String>,
    s3_pair: bool,
    old_uri: &str,
    new_uri: &str,
    old_t: &[TensorInfo],
    new_t: &[TensorInfo],
    pairs: &[(String, String)],
    bits: usize,
) -> HashMap<String, remote::RepackResult> {
    remote.filter(|_| s3_pair).map_or_else(|| if remote.is_some() {
        eprintln!(
            "checkpoint-studio diff: sparse index compare needs s3:// data over --ssh-proxy \
             (a remote safetensors dir isn't reachable) — skipped"
        );
        HashMap::new()
    } else {
        local_repack(old_t, new_t, pairs, bits, Some(1))
    }, |r| {
        let labels = repack_bar_labels(pairs, old_t, new_t);
        match fetch_remote_repack(r, password, old_uri, new_uri, pairs, &labels, bits, true) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("checkpoint-studio diff: sparse index compare: {e:#}");
                HashMap::new()
            }
        }
    })
}

/// Render the auto-detected sparse-packed index comparison (the `--values` path on
/// codebooked expert weights): per tensor, the index verdict (±1 / format / 2-D
/// slice) + the zero fraction, or — when the top-bits check fails — a fallback plain
/// stored-dtype value diff. Also diffs the sibling codebook/qscale. Returns 1 if
/// anything differs, else 0.
fn render_auto_sparse(
    pairs: &[(String, String)],
    results: &HashMap<String, remote::RepackResult>,
) -> i32 {
    use std::io::IsTerminal;
    let color = std::io::stdout().is_terminal();
    let (green, yellow, red, dim, reset) = if color {
        ("\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "", "", "")
    };
    println!("\npacked expert indices (auto-detected via sibling codebook):");
    let mut differs = false;
    let mut names: Vec<&String> = pairs.iter().map(|(_, n)| n).collect();
    names.sort();
    for name in names {
        let Some(rr) = results.get(name) else {
            differs = true;
            println!("  {red}✗{reset} {name}  {dim}(no result returned){reset}");
            continue;
        };
        if let Some(err) = &rr.error {
            differs = true;
            println!("  {red}✗{reset} {name}  {red}error:{reset} {err}");
            continue;
        }
        if let Some(fb) = &rr.fallback {
            // The one-index-per-word (sparse) decode didn't apply. A codebooked
            // weight is packed indices regardless, so this just means it's packed
            // *densely* (several indices per 16-bit word) — normal, not a problem;
            // compare the raw words (equal words ⇒ identical packed data). The rare
            // exception is a codebooked weight stored as plain floats.
            let (mark, col) = if fb.differing > 0 {
                differs = true;
                (yellow, "≠")
            } else {
                (green, "✓")
            };
            let is_float = matches!(fb.dtype.as_str(), "F16" | "BF16" | "F32" | "F64");
            let unit = if is_float { "values" } else { "16-bit words" };
            let verdict = if fb.differing == 0 {
                format!("{unit} identical")
            } else if is_float {
                format!(
                    "{} of {} values differ (max |Δ| {:.5}, mean {:.5})",
                    utils::format_parameters(fb.differing as usize),
                    utils::format_parameters(fb.elements as usize),
                    fb.max_abs,
                    fb.mean_abs,
                )
            } else {
                format!(
                    "{} of {} 16-bit words differ",
                    utils::format_parameters(fb.differing as usize),
                    utils::format_parameters(fb.elements as usize),
                )
            };
            let how = if is_float {
                format!("stored as {} (not packed indices)", fb.dtype)
            } else {
                format!(
                    "dense-packed ({}-bit indices, several per 16-bit word)",
                    rr.bits
                )
            };
            println!(
                "  {mark}{col}{reset} {name}  {dim}{how} — compared as raw {unit}:{reset} {verdict}"
            );
        } else {
            let counts = format!(
                "{}-bit, {} indices, {:.0}% zero",
                rr.bits,
                utils::format_parameters(rr.elements as usize),
                rr.zero_frac * 100.0,
            );
            if rr.differing > 0 {
                differs = true;
                let where_ = rr
                    .first_mismatch
                    .map(|(e, off, o, n)| {
                        format!(" {dim}(first: expert {e}, offset {off}: {o} vs {n}){reset}")
                    })
                    .unwrap_or_default();
                let adj = if rr.max_delta <= 1 {
                    format!("{dim}all by ±1 (same weights, independently re-quantized){reset}")
                } else {
                    format!(
                        "{dim}max |Δ| {}, {} by >1{reset}",
                        rr.max_delta,
                        utils::format_parameters(rr.differing_gt1 as usize),
                    )
                };
                println!(
                    "  {yellow}≠{reset} {name}  {yellow}{} of {} indices differ{reset} — {adj}{where_} {dim}({counts}){reset}",
                    utils::format_parameters(rr.differing as usize),
                    utils::format_parameters(rr.elements as usize),
                );
                println!(
                    "      {dim}Σ|Δ| {} · mean |Δ|/param {:.4} · mean index {:.4} → {:.4} (Δ {:+.4}){reset}",
                    utils::format_parameters(rr.sum_abs as usize),
                    rr.mean_abs,
                    rr.mean_old,
                    rr.mean_new,
                    rr.mean_new - rr.mean_old,
                );
                print_repack_sample(rr, red, dim, reset);
            } else {
                println!("  {green}✓{reset} {name}  {dim}indices identical ({counts}){reset}");
            }
        }
        // The sibling codebook / scale value-diff (quantized expert weights).
        print_repack_aux("codebook", rr.codebook.as_ref(), green, yellow, dim, reset);
        print_repack_aux("qscale", rr.qscale.as_ref(), green, yellow, dim, reset);
        differs |= aux_differs(rr.codebook.as_ref()) || aux_differs(rr.qscale.as_ref());
    }
    i32::from(differs)
}

/// Print the decoded 2-D index window (experts × inner-offset) for a mismatched
/// tensor: the OLD grid, then the NEW grid with cells that differ from old in red —
/// so the pattern of the difference is visible (a consistent shift ⇒ a mapping bug;
/// scattered ±1 ⇒ independent quantizations). Indices are 0..7 (single digit).
fn print_repack_sample(rr: &remote::RepackResult, red: &str, dim: &str, reset: &str) {
    let Some(s) = &rr.sample else { return };
    let rows = s.old.len().min(s.new.len());
    if rows == 0 {
        return;
    }
    let cols = s.old.first().map_or(0, Vec::len);
    println!(
        "      {dim}decoded index slice — experts {}..{} (rows) × offset {}..{} (cols); \
         cells that differ are red in both grids:{reset}",
        s.e0,
        s.e0 + rows as u64,
        s.off0,
        s.off0 + cols as u64,
    );
    // Both grids highlight the differing cells (red), so the eye lands on the same
    // positions in old and new.
    let render = |a: &[Vec<u32>], b: &[Vec<u32>]| {
        // Old and new grids have the same shape (checked by the caller), so walk them
        // together instead of indexing the second by the first's position.
        for (i, (row, brow)) in a.iter().zip(b).take(rows).enumerate() {
            let cells: String = row
                .iter()
                .enumerate()
                .map(|(j, v)| {
                    if brow.get(j) == Some(v) {
                        format!(" {v}")
                    } else {
                        format!(" {red}{v}{reset}")
                    }
                })
                .collect();
            println!("        {dim}e{:<4}{reset}{cells}", s.e0 + i as u64);
        }
    };
    println!("      {dim}old:{reset}");
    render(&s.old, &s.new);
    println!("      {dim}new:{reset}");
    render(&s.new, &s.old);
}

/// Print the value diff of a sibling `codebook` / `qscale` tensor (which the
/// structural diff can't flag, being the same shape on both sides). `identical`,
/// `differs — max/mean |Δ|`, or `shape differs`.
fn print_repack_aux(
    label: &str,
    aux: Option<&remote::RepackAux>,
    green: &str,
    yellow: &str,
    dim: &str,
    reset: &str,
) {
    let Some(a) = aux else { return };
    // Show exactly which sibling tensor was compared (so a wrong-name inference is
    // visible), plus its shape.
    let named = if a.old_name == a.new_name {
        format!("{dim}{}{reset}", a.new_name)
    } else {
        format!("{dim}{} vs {}{reset}", a.old_name, a.new_name)
    };
    if !a.present() {
        let miss = match (a.old_present, a.new_present) {
            (false, false) => "not found on either side",
            (false, true) => "not found in old",
            (true, false) => "not found in new",
            _ => "",
        };
        println!("      {label} ({named}): {dim}{miss}{reset}");
    } else if let Some((o, n)) = &a.shape_mismatch {
        println!(
            "      {label} ({named}): {yellow}shape differs{reset} {dim}{o:?} vs {n:?}{reset}"
        );
    } else if a.differing == 0 {
        println!(
            "      {label} ({named} {:?}): {green}identical{reset}",
            a.shape
        );
    } else {
        println!(
            "      {label} ({named} {:?}): {yellow}differs{reset} {dim}— max |Δ| {:.5}, mean {:.5}, {}/{} entries{reset}",
            a.shape,
            a.max_abs,
            a.mean_abs,
            utils::format_parameters(a.differing as usize),
            utils::format_parameters(a.elements as usize),
        );
    }
}

/// Detect a fold along dim 0: old `(E, …)` ↔ new `(ceil(E/fold), …)` with the inner
/// dims equal and `2 ≤ fold ≤ 16` (so `bits = 16/fold ≥ 1`). Returns the fold, or
/// `None` if the shapes aren't a fold pair. The bit-width is derived separately.
fn detect_fold(old: &[usize], new: &[usize]) -> Option<usize> {
    // One implementation, shared with the web's repack job — see `compare::detect_fold`.
    compare::detect_fold(old, new)
}

/// Open a session (reusing the already-entered `password`, so no second prompt) and
/// run the proxy value comparison for `pairs`. The comparison happens **on the
/// remote** (it holds the S3 access); only the small per-tensor results come back,
/// never tensor data. One standard [`progress::Bars`] bar per tensor fills over its
/// S3 byte size as the proxy streams the two sides ([`drive_bars`]), and a final
/// line reports the I/O throughput. `total_bytes` is the data to be read (both
/// sides, stored dtype) — for the intro estimate. Used for both the bulk
/// `--values`/`--histogram` run and the single `--tensor` focus.
fn fetch_remote_value_diff(
    r: &remote::RemoteRead,
    password: &mut Option<String>,
    old_side: &remote::RemoteSide,
    new_side: &remote::RemoteSide,
    pairs: &[(String, String)],
    vopts: &remote::RemoteValueOpts,
    total_bytes: u64,
) -> Result<HashMap<String, remote::RemoteTensorDiff>> {
    let session = r.open_with(password)?;
    utils::eprint_note(
        "checkpoint-studio diff: ",
        &format!(
            "comparing {} tensor(s) on {} — reading ≈ {} from S3 (processed on the remote, \
             not streamed here; {}-way parallel — use --jobs to tune, --jobs 1 if the \
             remote misbehaves) …",
            pairs.len(),
            r.host,
            utils::format_size(total_bytes as usize),
            vopts.jobs.max(1),
        ),
    );
    // One bar for the whole compare, filling over the total both sides will stream and
    // relabelled with the tensor being read (the values are still compared on the proxy —
    // only the byte counts and result cross ssh). A bar per tensor is thousands of them on
    // a real checkpoint, which is screens of output nobody can read; see `ValueBar`.
    let bar = std::cell::RefCell::new(ValueBar::start(
        &format!("comparing values on {}", r.host),
        total_bytes,
        pairs.len(),
    ));
    let out = r.value_diff(&session, old_side, new_side, pairs, vopts, None, |ev| {
        bar.borrow_mut().on(ev);
    });
    bar.into_inner().finish();
    let (map, stats) = out?;
    // I/O + timing from the proxy: bytes read from S3 and the throughput, so a long
    // comparison's performance is visible (on stderr, keeping stdout's diff clean).
    // `elapsed_s` is the remote compare time (excludes the ssh handshake).
    if let Some(s) = stats {
        let elapsed = std::time::Duration::from_secs_f64(s.elapsed_s.max(0.0));
        let read = utils::format_size(s.bytes as usize);
        let rate = if s.elapsed_s > 0.0 {
            format!(
                " ({}/s from S3)",
                utils::format_size((s.bytes as f64 / s.elapsed_s) as usize)
            )
        } else {
            String::new()
        };
        let skipped = s.tensors.saturating_sub(s.compared);
        let skip_note = if skipped > 0 {
            format!(", {skipped} skipped")
        } else {
            String::new()
        };
        eprintln!(
            "checkpoint-studio diff: compared {} tensor(s){skip_note} on the remote in {} · read {read}{rate}",
            s.compared,
            format_elapsed(elapsed),
        );
    }
    Ok(map)
}

/// Whether two S3 object sets are byte-identical: the same `(key, ETag, size)` for
/// every object (order-independent, non-empty). An `ETag` is a content hash, so equal
/// key+ETag ⇒ identical bytes ⇒ identical tensor values — letting a value diff skip
/// the data read entirely (e.g. a checkpoint vs. itself or an unchanged copy).
fn s3_objects_identical(a: &remote::S3Meta, b: &remote::S3Meta) -> bool {
    if a.objects.is_empty() || a.objects.len() != b.objects.len() {
        return false;
    }
    let sorted = |m: &remote::S3Meta| {
        let mut v: Vec<(String, String, u64)> = m
            .objects
            .iter()
            .map(|o| (o.key.clone(), o.etag.clone(), o.size))
            .collect();
        v.sort();
        v
    };
    sorted(a) == sorted(b)
}

/// Map a proxy-computed [`RemoteTensorDiff`](crate::remote::RemoteTensorDiff) into the
/// [`diff::TensorExtras`] the report is built from — the value stats verbatim, the
/// histogram summarized to its `(tvd, bins)` shift. A per-tensor remote error leaves
/// both empty (the tensor then shows as a structural-only change).
fn extras_from_remote(rd: &remote::RemoteTensorDiff) -> diff::TensorExtras {
    diff::TensorExtras {
        values: rd.values,
        histogram: rd
            .hist_shift
            .map(|(tvd, bins)| diff::HistShift { tvd, bins }),
    }
}

/// The `diff --tensor NAME` path: compare one tensor's signature and, when it's
/// in both checkpoints, its element values. With `remote`, the values/histogram were
/// computed on the proxy (`remote_diff`, `None` when that tensor couldn't be
/// compared). Exits 2 if the name is in neither.
#[allow(clippy::too_many_arguments)] // one focused CLI path; each arg is distinct
fn run_diff_tensor(
    old_label: &str,
    new_label: &str,
    name: &str,
    old_t: &[TensorInfo],
    new_t: &[TensorInfo],
    ctx: &ValueCtx,
    opts: diff::DiffOpts,
    remote: bool,
    remote_diff: Option<&remote::RemoteTensorDiff>,
) -> i32 {
    let old_info = old_t.iter().find(|t| t.name == name);
    let new_info = new_t.iter().find(|t| t.name == name);
    if old_info.is_none() && new_info.is_none() {
        eprintln!("checkpoint-studio diff: tensor '{name}' not found in either checkpoint");
        return 2;
    }

    let old_sig = old_info.map(diff::TensorSig::of);
    let new_sig = new_info.map(diff::TensorSig::of);
    // Compare values only when the tensor is in both checkpoints. Remote (s3): use
    // the proxy-computed result; local: read the data here.
    let values = match (old_info, new_info) {
        (Some(a), Some(b)) if remote => Some(value_cmp_remote(a, b, remote_diff)),
        (Some(a), Some(b)) => Some(value_cmp(a, b, ctx)),
        _ => None,
    };

    print!(
        "{}",
        diff::render_tensor_focus(
            old_label,
            new_label,
            name,
            old_sig.as_ref(),
            new_sig.as_ref(),
            values.as_ref(),
            opts.color,
        )
    );

    // `--histogram`: append the full bin-by-bin distribution table when the tensor
    // is in both with a matching shape.
    let mut hist_differs = false;
    if opts.histogram
        && let (Some(a), Some(b)) = (old_info, new_info)
        && a.shape == b.shape
    {
        if remote {
            match remote_diff.and_then(|rd| rd.hist_full.as_ref()) {
                Some(hd) => {
                    hist_differs = hd.differs();
                    print!("{}", diff::render_histogram_table(name, hd, opts.color));
                }
                None => eprintln!(
                    "checkpoint-studio diff: histogram: not available for this tensor over ssh"
                ),
            }
        } else {
            match sample::histogram_diff(
                a,
                ctx.old_schemas.get(&a.name),
                b,
                ctx.new_schemas.get(&b.name),
                ctx.view,
                ctx.bins,
            ) {
                Ok(hd) => {
                    hist_differs = hd.differs();
                    print!("{}", diff::render_histogram_table(name, &hd, opts.color));
                }
                Err(e) => eprintln!("checkpoint-studio diff: histogram: {e}"),
            }
        }
    }

    let differs = diff::tensor_focus_differs(old_sig.as_ref(), new_sig.as_ref(), values.as_ref())
        || hist_differs;
    i32::from(differs)
}

/// Compare two tensors' values, mapping the result into a [`diff::ValueCmp`].
/// Shapes must match for an element-wise comparison; a mismatch (or a read /
/// dtype error) is reported as skipped rather than failing the whole diff.
fn value_cmp(a: &TensorInfo, b: &TensorInfo, ctx: &ValueCtx) -> diff::ValueCmp {
    if a.shape != b.shape {
        return diff::ValueCmp::Skipped("shapes differ".to_string());
    }
    match sample::compare_values(
        a,
        ctx.old_schemas.get(&a.name),
        b,
        ctx.new_schemas.get(&b.name),
        ctx.view,
    ) {
        Ok(vd) if vd.differing == 0 => diff::ValueCmp::Identical,
        Ok(vd) => diff::ValueCmp::Differ(vd),
        Err(e) => diff::ValueCmp::Skipped(e),
    }
}

/// Map a proxy-computed [`RemoteTensorDiff`](crate::remote::RemoteTensorDiff) into a
/// [`diff::ValueCmp`] for the `--tensor` focus. Shapes are checked here (the remote
/// only compares matching pairs); a per-tensor remote error, or values not having
/// been computed, is reported as skipped rather than failing the diff.
fn value_cmp_remote(
    a: &TensorInfo,
    b: &TensorInfo,
    rd: Option<&remote::RemoteTensorDiff>,
) -> diff::ValueCmp {
    if a.shape != b.shape {
        return diff::ValueCmp::Skipped("shapes differ".to_string());
    }
    match rd {
        Some(rd) => {
            if let Some(e) = &rd.error {
                return diff::ValueCmp::Skipped(e.clone());
            }
            match rd.values {
                Some(vd) if vd.differing == 0 => diff::ValueCmp::Identical,
                Some(vd) => diff::ValueCmp::Differ(vd),
                None => diff::ValueCmp::Skipped("values not compared".to_string()),
            }
        }
        None => diff::ValueCmp::Skipped("remote value comparison unavailable".to_string()),
    }
}

/// Parse a `ROW,COL` pair of non-negative integers (the `--window` top-left).
/// Parse and bound the `--bins` histogram bucket count to `1..=512`.
fn parse_bins(s: &str) -> std::result::Result<usize, String> {
    match s.trim().parse::<usize>() {
        Ok(n) if (1..=512).contains(&n) => Ok(n),
        Ok(_) => Err("must be between 1 and 512".to_string()),
        Err(_) => Err(format!("expected a whole number, got '{s}'")),
    }
}

fn parse_offset_pair(s: &str) -> Result<(usize, usize)> {
    let (r, c) = s
        .split_once(',')
        .with_context(|| format!("expected ROW,COL (two integers), got '{s}'"))?;
    let row = r
        .trim()
        .parse()
        .with_context(|| format!("invalid row '{r}'"))?;
    let col = c
        .trim()
        .parse()
        .with_context(|| format!("invalid col '{c}'"))?;
    Ok((row, col))
}

/// Parse a `RFRAC,CFRAC` pair of fractions in `0..=1` (the `--edge` head/tail
/// split: 0 keeps only the first indices, 1 only the last, 0.5 is balanced).
fn parse_fraction_pair(s: &str) -> Result<(f32, f32)> {
    let (r, c) = s
        .split_once(',')
        .with_context(|| format!("expected RFRAC,CFRAC (two fractions 0..1), got '{s}'"))?;
    let row: f32 = r
        .trim()
        .parse()
        .with_context(|| format!("invalid row '{r}'"))?;
    let col: f32 = c
        .trim()
        .parse()
        .with_context(|| format!("invalid col '{c}'"))?;
    if !(0.0..=1.0).contains(&row) || !(0.0..=1.0).contains(&col) {
        anyhow::bail!("edge split fractions must be between 0 and 1, got '{s}'");
    }
    Ok((row, col))
}

/// Split an scp-style `[user@]host:path` into (host, path). Returns `None` for a
/// local path or an `s3://…` URI (no host to derive — that needs `--ssh-proxy`).
/// Matches `scp`'s own rule: a `:` before any `/`, with a non-empty host to its
/// left.
fn split_scp(s: &str) -> Option<(String, String)> {
    // A URI is not an scp path, however much `scheme:` looks like `host:`. Without this,
    // `https://huggingface.co/owner/name` parsed as the host `https` and the tool tried to
    // ssh to it — and `hf://…` did the same.
    if s.contains("://") || hf::is_uri(s) {
        return None;
    }
    let colon = s.find(':')?;
    if colon == 0 || s[..colon].contains('/') {
        return None;
    }
    Some((s[..colon].to_string(), s[colon + 1..].to_string()))
}

/// Split an `[user@]host:` prefix off every path, when they carry one.
///
/// `Ok(None)` when none does — every path is local, or a URI, or the `:PATH` shorthand (whose colon
/// is at index 0, so [`split_scp`] declines it).
///
/// **Applied whether or not `--ssh-proxy` was also given**, which is the bug this exists for. It used
/// to run only when the flag was absent, so `--ssh-proxy H host:/path` kept the host on the path *and*
/// had the proxy host prefixed onto it again — producing `H:H:/path`, which reads as nothing ("no
/// safetensors files found at H:H:/path") and, worse, was written to the recents list, where it sat as
/// an entry that could never be opened.
///
/// Two hosts that disagree is a conflict worth naming rather than resolving by guess.
fn split_off_scp_host(
    paths: &[PathBuf],
    proxy: Option<&str>,
) -> Result<(Vec<PathBuf>, Option<String>)> {
    let Some((host, _)) = paths.iter().find_map(|p| split_scp(&p.to_string_lossy())) else {
        return Ok((paths.to_vec(), None));
    };
    if let Some(flag) = proxy
        && flag != host
    {
        anyhow::bail!(
            "the path names host `{host}` and --ssh-proxy names `{flag}` — one checkpoint has one \
             host; drop the host from the path, or drop the flag"
        );
    }
    let mut stripped = Vec::with_capacity(paths.len());
    for p in paths {
        match split_scp(&p.to_string_lossy()) {
            Some((h, path)) if h == host => stripped.push(PathBuf::from(path)),
            _ => anyhow::bail!(
                "can't mix local and scp-style ({host}:…) paths (or different hosts); \
                 list paths from one host, or use --ssh-proxy"
            ),
        }
    }
    Ok((stripped, Some(host)))
}

/// `web` subcommand: read a local checkpoint once and serve the web UI + JSON API,
/// blocking until Ctrl-C. The server supplies the data; the browser owns the view
/// state (see `crate::web`).
#[allow(clippy::too_many_arguments)] // a CLI entry point; each arg is a distinct flag
fn run_web(
    paths: &[PathBuf],
    recursive: bool,
    no_health_check: bool,
    host: std::net::IpAddr,
    port: u16,
    ssh: Option<(String, String)>,
) -> Result<()> {
    // Reserve the port *before* the read (5–10 s over SSH, seconds for a 31k-tensor local
    // checkpoint): a clash is reported immediately, and if the requested port is taken we
    // land on a free one and hold it while the read runs — so the wait is never wasted.
    let server = web::bind(host, port)?;
    let opts = opening::Options {
        recursive,
        no_health_check,
        // Already resolved by the caller, and re-resolving here would let a config file
        // override the flags the process was started with.
        proxy: None,
        venv: None,
    };
    let remote = ssh.map(|(rhost, venv)| remote::RemoteRead::new(rhost, venv));
    // One call for every source — local, Hub, ssh proxy. The three branches this replaced
    // each spelled the same read slightly differently; now `--web` and the browser's own
    // "open another checkpoint" go through one rule (see `crate::opening`).
    let opened = opening::Target::from_paths(paths, remote.clone(), &opts)?
        .read(opening::Want::Model, &hf::ReadProgress::default())?;
    // The user's list, kept across restarts — so an `hf://` repo typed once is still offered
    // after the server is restarted, which is the case a session-only list loses.
    let current = web::Current::new(opened, remote, opts, host, opening::Recents::persistent())?;
    web::serve_on(server, std::sync::Arc::new(current), host)
}

fn run_explore(mut args: ExploreArgs) -> Result<()> {
    if args.paths.is_empty() {
        eprintln!("checkpoint-studio: no checkpoint given.\n");
        eprintln!("Usage:");
        eprintln!(
            "  checkpoint-studio <PATH>...            browse a checkpoint (file, directory, or glob)"
        );
        eprintln!(
            "  checkpoint-studio <PATH> --print-tree  dump its structure (text, or --format json)"
        );
        eprintln!("  checkpoint-studio diff <OLD> <NEW>     compare two checkpoints");
        eprintln!(
            "  checkpoint-studio --ssh-proxy <HOST> <s3://…|/remote/path>   read a remote / S3 checkpoint"
        );
        eprintln!("\nRun `checkpoint-studio --help` for all options and examples.");
        std::process::exit(1);
    }

    // scp-style positional paths (`[user@]host:/path`) carry their own host: read the path part
    // remotely, so `checkpoint-studio host:/opt/model` just works — with or without an explicit
    // `--ssh-proxy` naming the same host (see `split_off_scp_host`).
    {
        let (paths, host) = split_off_scp_host(&args.paths, args.ssh_proxy.as_deref())?;
        args.paths = paths;
        args.ssh_proxy = args.ssh_proxy.or(host);
    }

    // `--ssh-proxy` delegates the read to cstorch on a remote host, so the s3://
    // URIs are kept verbatim (no local listing). Otherwise list local sources.
    let (files, index_specs) = if args.ssh_proxy.is_some() {
        (args.paths.clone(), Vec::new())
    } else {
        collect_safetensors_files(&args.paths, args.recursive, args.no_health_check)?
    };

    if files.is_empty() {
        eprintln!("Error: No checkpoint files found in the specified paths.");
        std::process::exit(1);
    }

    // Reading HDF5 needs the `hdf5` build feature; without it these files would
    // load as empty and the tree would misleadingly read "0 tensors, 0 params,
    // 0 B". Say so plainly instead. Directory scans already skip HDF5 when the
    // feature is off, so this only fires for files the user named explicitly.
    #[cfg(not(feature = "hdf5"))]
    {
        let hdf5: Vec<&PathBuf> = files
            .iter()
            .filter(|p| matches!(p.extension().and_then(|s| s.to_str()), Some("h5" | "hdf5")))
            .collect();
        if !hdf5.is_empty() {
            eprintln!(
                "Error: this build of checkpoint-studio was compiled without HDF5 support, so it cannot read:"
            );
            for path in &hdf5 {
                eprintln!("  {}", path.display());
            }
            eprintln!();
            eprintln!("Rebuild and reinstall with the `hdf5` feature, e.g.:");
            eprintln!("  cargo install --path . --features hdf5");
            std::process::exit(1);
        }
    }

    // Flags that target the tree browser rather than a tensor view: with no
    // `--tensor` (and no data view), they make the tree the opened screen, so
    // e.g. `--expand-all` or `--legend` alone don't demand a tensor.
    let tree_oriented = args.tree
        || args.tree_state.is_some()
        || args.search.is_some()
        || args.legend
        || args.health
        || args.health_findings
        || args.stats
        || args.stats_shards
        || args.files
        || args.layout.is_some()
        || args.rename
        || args.diff_against.is_some()
        || args.compare_with.is_some()
        || args.sort.is_some()
        || args.compact;
    let view = if args.values {
        OpenView::Values
    } else if args.heatmap {
        OpenView::Heatmap
    } else if args.tree || (tree_oriented && args.tensor.is_none()) {
        OpenView::Tree
    } else {
        OpenView::Detail
    };
    let layout = if args.window.is_some() {
        Some(DataLayout::Window)
    } else if args.edge.is_some() {
        Some(DataLayout::Edges)
    } else if args.abs_max {
        Some(DataLayout::OverviewMax)
    } else if args.overview {
        Some(DataLayout::Overview)
    } else {
        None
    };
    // Position within the layout: the window's top-left corner, or the edges
    // head/tail split — parsed from the optional `--window`/`--edge` value.
    let window_at = args.window.as_deref().map(parse_offset_pair).transpose()?;
    let edge_split = args.edge.as_deref().map(parse_fraction_pair).transpose()?;
    // Seed an open request when a tensor is named *or* any view/override flag is
    // given — the latter targets the sole tensor when the checkpoint has one.
    let wants_open = args.tensor.is_some()
        || args.metadata.is_some()
        || args.values
        || args.heatmap
        || args.tree
        || args.dtype.is_some()
        || args.edge.is_some()
        || args.overview
        || args.window.is_some()
        || args.zebra.is_some()
        || args.base.is_some()
        || args.slice.is_some()
        || args.shape.is_some()
        || args.compute_stats
        || args.histogram
        || args.bins.is_some()
        || args.tree_state.is_some()
        || args.search.is_some()
        || args.legend
        || args.health
        || args.health_findings
        || args.stats
        || args.stats_shards
        || args.files
        || args.layout.is_some()
        || args.rename
        || args.diff_against.is_some()
        || args.compare_with.is_some()
        || args.sort.is_some()
        || args.compact
        || args.exit;
    // `--tensor`/`--metadata` are mutually exclusive (clap enforces it); fold into
    // one target, and the detail-implies-parent flag pairs into 3-state requests.
    let target = if let Some(m) = args.metadata {
        explorer::OpenTarget::Metadata(m)
    } else if let Some(t) = args.tensor {
        explorer::OpenTarget::Tensor(t)
    } else {
        explorer::OpenTarget::SoleTensor
    };
    let histogram = match args.bins {
        Some(n) => explorer::HistogramReq::Bins(n),
        None if args.histogram => explorer::HistogramReq::Auto,
        None => explorer::HistogramReq::Off,
    };
    let health = if args.health_findings {
        explorer::HealthReq::Findings
    } else if args.health {
        explorer::HealthReq::Summary
    } else {
        explorer::HealthReq::Off
    };
    let stats = if args.stats_shards {
        explorer::StatsReq::Shards
    } else if args.stats {
        explorer::StatsReq::Summary
    } else {
        explorer::StatsReq::Off
    };
    let open = wants_open.then_some(OpenRequest {
        target,
        view,
        histogram,
        dtype: args.dtype,
        layout,
        window_at,
        edge_split,
        zebra: args.zebra,
        base: args.base,
        slice: args.slice,
        shape: args.shape,
        compute_stats: args.compute_stats,
        tree_state: args.tree_state,
        search: args.search,
        legend: args.legend,
        health,
        stats,
        exit_after: args.exit,
        files_view: args.files,
        layout_file: args.layout,
        layout_select: args.layout_select,
        rename: args.rename,
        diff_against: args.diff_against.clone(),
        compare_with: args.compare_with.clone(),
        compare_full: args.compare_full,
        sort: args.sort,
        compact: args.compact,
        rename_rules: args.rename_rule,
    });

    // `--print-model`: dump the whole central serializable model as JSON and exit
    // — a CLI frontend reading the kernel's model directly (the "serializable into
    // JSON" contract). Local only for now; the remote reader fills the model next.
    // `--print-arch`: the inferred architecture summary. Reads the structure only, so it
    // works for a Hugging Face repo as readily as a local checkpoint.
    if args.print_arch {
        let source = source::resolve(&args.paths, None)?;
        let (parts, _) = source.read(&hf::ReadProgress::default())?;
        let tensors = parts.tensors;
        let inferred = arch::infer(&tensors, None);
        match args.format {
            explorer::TreeFormat::Json => {
                println!("{}", serde_json::to_string_pretty(&inferred)?);
            }
            explorer::TreeFormat::Text => {
                let width = inferred
                    .facts
                    .iter()
                    .map(|(l, _)| l.len())
                    .max()
                    .unwrap_or(0);
                for (label, fact) in &inferred.facts {
                    println!("{label:width$}  {}", fact.value);
                    println!("{:width$}    ← {}", "", fact.from);
                }
                if !inferred.not_in_tensors.is_empty() {
                    println!("\nNot inferable from tensors:");
                    for (label, why) in &inferred.not_in_tensors {
                        println!("  {label} — {why}");
                    }
                }
            }
        }
        return Ok(());
    }

    if args.print_model {
        let model = if let Some(host) = args.ssh_proxy.as_ref() {
            let venv = args
                .ssh_venv
                .clone()
                .unwrap_or_else(|| "~/venv".to_string());
            // `--print-model` dumps the structure; the per-object S3 metadata would
            // add a HEAD per object for data it doesn't print.
            remote::RemoteRead::new(host.clone(), venv).read_checkpoint(
                &files
                    .first()
                    .map(|f| f.to_string_lossy().to_string())
                    .unwrap_or_default(),
                remote::ObjectMeta::Skip,
                // A one-shot dump: there is no second request that could ask it to stop.
                None,
                // On a terminal, so it draws its own bar.
                None,
            )?
        } else {
            readers::read_local(&files)?
        };
        println!("{}", serde_json::to_string_pretty(&model)?);
        return Ok(());
    }

    // Carry the read switches into the session: the palette's "Open another checkpoint…"
    // reads the next one with the same `--recursive` / `--no-health-check` as this one.
    let mut explorer = Explorer::new(files, index_specs, open, !args.no_preload).with_read_options(
        opening::Options {
            recursive: args.recursive,
            no_health_check: args.no_health_check,
            proxy: None,
            venv: None,
        },
    );
    if let Some(host) = args.ssh_proxy {
        let venv = args.ssh_venv.unwrap_or_else(|| "~/venv".to_string());
        explorer.set_remote_read(host, venv);
    }
    // AFTER the proxy is installed: recording the startup checkpoint has to know whether it is
    // remote, or it would absolutise a remote path against this machine's working directory.
    // The same persisted list the web server offers, so the two surfaces remember the same
    // checkpoints — and installed here rather than in `Explorer::new`, so the many explorers
    // built for tests and one-shot exports never touch the user's config directory.
    let mut explorer = explorer.with_persistent_recents();
    if let Some(query) = args.filter.as_deref() {
        explorer.set_tensor_filter(tensorfilter::TensorFilter::parse(query)?);
    }
    // One-shot exports: print the tree / tensor list and exit (honour --format,
    // -v, and the --name filter), before any interactive or --plain rendering.
    if args.print_tree || args.print_tensors {
        let detail = explorer::TreeDetail::from_verbosity(args.verbose);
        let filter = filter::NameFilter::parse(&args.name)?;
        return if args.print_tree {
            explorer.print_tree(args.format, detail, &filter)
        } else {
            explorer.print_tensors(args.format, detail, &filter)
        };
    }
    if args.print_view {
        let filter = filter::NameFilter::parse(&args.name)?;
        return explorer.print_view(&filter);
    }

    if args.emit_command {
        explorer.render_plain(true)
    } else if args.plain {
        explorer.render_plain(false)
    } else {
        explorer.run()
    }
}

/// `convert --map`: rename tensors **in place** in a local safetensors checkpoint.
/// Builds and validates the full remapping (all sources exist, no collisions, every
/// rewritten header fits without moving data), prints the plan, asks for
/// confirmation (unless `force`), then rewrites the shard headers and index.json.
/// Feature-independent — the rename never touches HDF5.
fn run_rename(input: &Path, map: &[String], map_from: Option<&Path>, force: bool) -> Result<()> {
    use anyhow::bail;
    use std::io::{IsTerminal, Write};

    // Local-only: reject an obvious remote URL up front (rename edits files in
    // place, which only makes sense on the local filesystem).
    if input.to_string_lossy().contains("://") {
        bail!(
            "rename is local-only, but {} looks like a remote URL",
            input.display()
        );
    }

    let name_map = build_name_map(map, map_from)?;
    let plan = rename::plan(input, &name_map)?;

    // Show the plan (renames, index note, warnings, the in-place caveat).
    for line in plan.summary_lines(40) {
        eprintln!("{line}");
    }

    if !force {
        if !std::io::stdin().is_terminal() {
            bail!(
                "refusing to rename without confirmation on a non-interactive terminal — \
                 pass --force to apply"
            );
        }
        eprint!("Proceed with the in-place rename? [y/N] ");
        std::io::stderr().flush().ok();
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer)?;
        if !matches!(answer.trim(), "y" | "Y" | "yes" | "Yes") {
            eprintln!("Aborted; nothing was changed.");
            return Ok(());
        }
    }

    rename::apply(&plan)?;
    eprintln!(
        "Renamed {} tensor(s) across {} shard file(s){}.",
        plan.rename_count(),
        plan.shard_count(),
        if plan.index.is_some() {
            " (+ model.safetensors.index.json)"
        } else {
            ""
        },
    );
    Ok(())
}

#[cfg(feature = "hdf5")]
fn run_convert(
    input: &Path,
    output: &Path,
    codec: codec::Codec,
    level: Option<u8>,
    buffer: &str,
    force: bool,
) -> Result<()> {
    use anyhow::bail;
    use std::io::Write;

    // Repacking is a capability of the (format, location) pair, so ask it rather than
    // re-testing the extension here: that keeps this refusal in step with the TUI's
    // repack command, which hides itself for exactly the inputs this rejects.
    let caps = capability::Capabilities::of(
        capability::Format::of_path(&input.to_string_lossy()).unwrap_or(capability::Format::Mixed),
        capability::Location::of_source_path(&input.to_string_lossy()),
    );
    if !caps.repack {
        bail!(
            "convert repacks HDF5 inputs (.h5/.hdf5) held locally, got: {}",
            input.display()
        );
    }
    // Refuse to read and write the same file (checked before --force removes the
    // output, so we never delete the input).
    if std::path::absolute(input).ok() == std::path::absolute(output).ok()
        && std::path::absolute(input).is_ok()
    {
        bail!("input and output are the same file: {}", input.display());
    }
    // Warn when the target codec is what the source already uses (a re-encode;
    // a plain file copy would be equivalent).
    if convert::source_codec(input) == Some(codec) {
        eprintln!(
            "warning: source is already {}; repacking just re-encodes it — a plain copy would be equivalent",
            codec.label()
        );
    }
    if force && output.exists() {
        fs::remove_file(output)
            .with_context(|| format!("removing existing {}", output.display()))?;
    }

    let level = codec.clamp_level(level.unwrap_or_else(|| codec.default_level()));
    let buffer_bytes = utils::parse_size(buffer).map_err(anyhow::Error::msg)?;
    let opts = convert::Options {
        codec,
        level,
        buffer_bytes,
    };
    let level_note = if codec.uses_level() {
        format!(" level {level}")
    } else {
        String::new()
    };
    eprintln!(
        "Repacking {} → {} ({}{level_note}, {} buffer)",
        input.display(),
        output.display(),
        codec.label(),
        utils::format_size(buffer_bytes),
    );

    let mut stderr = std::io::stderr();
    let report = convert::convert_hdf5(input, output, &opts, |done, total, name| {
        let bar = progress_bar(done, total, 28);
        let _ = write!(stderr, "\r{bar} [{done}/{total}] {name:.<48}\x1b[K");
        let _ = stderr.flush();
    })?;
    eprintln!("\rDone: {}\x1b[K", report.summary(codec));
    Ok(())
}

/// A `[####----]` progress bar of the given width.
#[cfg(feature = "hdf5")]
fn progress_bar(done: usize, total: usize, width: usize) -> String {
    let filled = (done * width).checked_div(total).unwrap_or(0);
    format!(
        "[{}{}]",
        "#".repeat(filled),
        "-".repeat(width.saturating_sub(filled))
    )
}

#[cfg(not(feature = "hdf5"))]
fn run_convert(
    _input: &Path,
    _output: &Path,
    _codec: codec::Codec,
    _level: Option<u8>,
    _buffer: &str,
    _force: bool,
) -> Result<()> {
    anyhow::bail!("`convert` requires building with `--features hdf5`")
}

fn collect_safetensors_files(
    paths: &[PathBuf],
    recursive: bool,
    no_health_check: bool,
) -> Result<(Vec<PathBuf>, Vec<health::IndexSpec>)> {
    let mut files = Vec::new();
    // Parsed indexes to health-check later against the loaded tensors (so shard
    // headers are read once, by the loader, not again here). Empty when health is
    // off. See `health::check_loaded`.
    let mut index_specs: Vec<health::IndexSpec> = Vec::new();

    for path in paths {
        // Remote checkpoints are read via `--ssh-proxy` (handled before this
        // function); a bare `s3://` here has no local credentials to read it with.
        let raw = path.to_string_lossy();
        if s3::is_uri(&raw) {
            // The shared reason, from the capability model — so this refusal and the
            // loader's (and any UI's) say the same thing.
            let why = capability::Location::S3
                .proxy_note()
                .unwrap_or("cannot be read from here");
            eprintln!("Warning: {raw}: {why}");
            continue;
        }
        // A Hugging Face reference is a URI, not a path: it must not be globbed, tilde-
        // expanded or checked for existence on this machine. `gather_checkpoint` reads it.
        if hf::is_uri(&raw) {
            files.push(path.clone());
            continue;
        }

        // A leading `~` first: a shell expands it before we ever see an argv path, but a
        // path typed *inside* the app (the web UI's compare box, the TUI's compare
        // prompt) has had no shell — and a quoted `'~/ckpt'` on a real command line has
        // had none either. Doing it here means every path the program is handed obeys the
        // same rule, whatever door it came in by. See `utils::expand_tilde`.
        let path = &utils::expand_tilde(&raw);

        // Try to expand as glob pattern
        let expanded_paths: Vec<PathBuf> = glob::glob(&path.to_string_lossy()).map_or_else(
            |_| vec![path.clone()],
            |paths| paths.filter_map(Result::ok).collect(),
        );

        // Process each expanded path
        for expanded_path in expanded_paths {
            if !expanded_path.exists() {
                eprintln!("Warning: Path does not exist: {}", expanded_path.display());
                continue;
            }

            if expanded_path.is_file() {
                let ext = expanded_path.extension().and_then(|s| s.to_str());
                if matches!(
                    ext,
                    Some("safetensors" | "gguf" | "h5" | "hdf5" | "npy" | "npz")
                ) || (ext != Some("safetensors") && readers::looks_like_hdf5(&expanded_path))
                {
                    files.push(expanded_path.clone());
                } else {
                    eprintln!(
                        "Warning: Skipping unsupported file: {}",
                        expanded_path.display()
                    );
                }
            } else if expanded_path.is_dir() {
                // Check for SafeTensors index file first
                let index_path = expanded_path.join("model.safetensors.index.json");
                let mut found_from_index = false;
                if index_path.exists() {
                    // Parse the index once. Its `weight_map` gives the shard list
                    // here, and (unless health is off) the spec is kept to compare
                    // against the loaded tensors later — so the shard headers are
                    // read a single time, by the loader, not again for the health
                    // check.
                    let spec = health::parse_index_spec(&expanded_path, &index_path)?;
                    let mut index_files: Vec<String> = spec.weight_map.values().cloned().collect();
                    index_files.sort();
                    index_files.dedup();
                    let mut missing = Vec::new();
                    for file in index_files {
                        let full_path = expanded_path.join(&file);
                        if full_path.exists() {
                            files.push(full_path);
                            found_from_index = true;
                        } else {
                            missing.push(file);
                        }
                    }
                    if !missing.is_empty() {
                        eprintln!(
                            "Warning: {} file(s) listed in {} were not found on disk (e.g. {}).",
                            missing.len(),
                            index_path.display(),
                            missing.first().map_or("", String::as_str),
                        );
                    }
                    if !found_from_index {
                        eprintln!(
                            "Warning: index file references no existing files (it may be stale); scanning {} directly instead.",
                            expanded_path.display()
                        );
                    }
                    if !no_health_check {
                        index_specs.push(spec);
                    }
                }

                // Always scan the directory as well, so files present on disk
                // but missing from the index — a partially-stale index, e.g.
                // extra `codebooks`/`qscales` shards — still show up, alongside
                // the no-index and fully-stale cases. Duplicates (a shard listed
                // in the index *and* found by the scan) are removed below.
                scan_directory(&expanded_path, recursive, &mut files)?;
            }
        }
    }

    // Sort for consistent ordering and drop duplicates — the same file can be
    // collected both from the index and the directory scan (identical paths).
    files.sort();
    files.dedup();
    Ok((files, index_specs))
}

fn scan_directory(dir: &Path, recursive: bool, files: &mut Vec<PathBuf>) -> Result<()> {
    // HDF5 is only scanned for when compiled in, to avoid surfacing files the
    // build cannot read.
    let exts: &[&str] = if cfg!(feature = "hdf5") {
        &["safetensors", "gguf", "h5", "hdf5", "npy", "npz"]
    } else {
        &["safetensors", "gguf", "npy", "npz"]
    };
    let glob_prefix = if recursive { "**/" } else { "" };
    let patterns: Vec<String> = exts
        .iter()
        .map(|ext| format!("{}/{glob_prefix}*.{ext}", dir.display()))
        .collect();

    for pattern in patterns {
        for entry in glob::glob(&pattern).context("Failed to read glob pattern")? {
            match entry {
                Ok(file_path) => files.push(file_path),
                Err(e) => eprintln!("Warning: Error reading file: {e}"),
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn splits_scp_style_paths_only() {
        assert_eq!(
            split_scp("net004:/opt/models/m"),
            Some(("net004".into(), "/opt/models/m".into()))
        );
        assert_eq!(
            split_scp("lab@host:rel/path"),
            Some(("lab@host".into(), "rel/path".into()))
        );
        // local paths and s3 URIs are not scp targets
        assert_eq!(split_scp("/opt/models/m"), None);
        assert_eq!(split_scp("./model.safetensors"), None);
        assert_eq!(split_scp("s3://bucket/key"), None);
        assert_eq!(split_scp("dir/a:b"), None); // colon after a slash → local
    }

    /// One bar over a whole run, so the numbers it shows have to be right: the label names
    /// where the run is, and the fill is every tensor's bytes added up.
    mod value_bar {
        use super::super::{ValueBar, remote::CompareStatus, remote::RepackEvent as E};

        /// A `ValueBar` with no terminal attached still folds every event, so these test
        /// the arithmetic rather than the drawing (which `progress.rs` covers).
        fn bar(total_bytes: u64, tensors: usize) -> ValueBar {
            ValueBar::start("comparing", total_bytes, tensors)
        }

        #[test]
        fn the_label_names_the_tensor_and_its_position_in_the_run() {
            let mut b = bar(100, 3);
            b.on(E::Start {
                done: 0,
                total: 3,
                name: "model.layers.0.mlp.down_proj.weight",
            });
            let p = b.bars.progress(0).expect("one bar");
            assert_eq!(
                p.item().as_deref(),
                Some("[1/3] model.layers.0.mlp.down_proj.weight")
            );
        }

        #[test]
        fn the_position_advances_as_tensors_finish() {
            let mut b = bar(100, 3);
            b.on(E::Done {
                name: "a",
                status: CompareStatus::Identical,
            });
            b.on(E::Start {
                done: 1,
                total: 3,
                name: "b",
            });
            let p = b.bars.progress(0).expect("one bar");
            assert_eq!(p.item().as_deref(), Some("[2/3] b"));
        }

        #[test]
        fn the_position_never_runs_past_the_total() {
            // The last tensor's Done makes `done` equal the total; `[4/3]` would be a
            // visible lie on the final frame.
            let mut b = bar(100, 3);
            for name in ["a", "b", "c"] {
                b.on(E::Done {
                    name,
                    status: CompareStatus::Identical,
                });
            }
            let p = b.bars.progress(0).expect("one bar");
            assert_eq!(p.item().as_deref(), Some("[3/3] c"));
        }

        /// The reason the bar keeps the latest value per tensor instead of summing
        /// deltas: the events are cumulative *per tensor*, and several tensors are in
        /// flight at once under `--jobs`.
        #[test]
        fn bytes_are_summed_across_tensors_in_flight() {
            let mut b = bar(400, 2);
            let p = b.bars.progress(0).expect("one bar");
            b.on(E::Bytes {
                name: "a",
                old_done: 50,
                new_done: 50,
            });
            assert_eq!(p.snapshot().0, 100);
            // A second tensor's progress adds to the first, not replaces it.
            b.on(E::Bytes {
                name: "b",
                old_done: 25,
                new_done: 25,
            });
            assert_eq!(p.snapshot().0, 150);
            // And `a` reporting again *replaces* a's contribution rather than adding.
            b.on(E::Bytes {
                name: "a",
                old_done: 100,
                new_done: 100,
            });
            assert_eq!(
                p.snapshot().0,
                250,
                "cumulative per tensor, not a delta sum"
            );
        }

        #[test]
        fn a_restarted_tensor_does_not_inflate_the_total() {
            // A retry re-reports from zero. Summing deltas would leave the bar past its
            // total and showing more bytes read than exist.
            let mut b = bar(200, 1);
            let p = b.bars.progress(0).expect("one bar");
            b.on(E::Bytes {
                name: "a",
                old_done: 90,
                new_done: 90,
            });
            b.on(E::Bytes {
                name: "a",
                old_done: 0,
                new_done: 0,
            });
            assert_eq!(p.snapshot().0, 0);
        }

        #[test]
        fn a_tensor_that_reported_no_bytes_still_counts_when_it_finishes() {
            // A skipped or cached tensor never emits Bytes; without crediting it on Done
            // the bar could never reach its total.
            let mut b = bar(0, 1);
            b.on(E::Done {
                name: "a",
                status: CompareStatus::Identical,
            });
            assert!(b.seen.contains_key("a"));
        }

        #[test]
        fn an_error_makes_the_bar_finish_as_a_failure() {
            let mut b = bar(10, 1);
            assert!(!b.failed);
            b.on(E::Done {
                name: "a",
                status: CompareStatus::Error,
            });
            assert!(b.failed, "one failed tensor must not report a clean run");
        }

        #[test]
        fn comparing_says_so_without_losing_the_tensor_name() {
            let mut b = bar(10, 1);
            b.on(E::Comparing {
                name: "w",
                spans: None,
            });
            let p = b.bars.progress(0).expect("one bar");
            let item = p.item().unwrap_or_default();
            assert!(item.contains('w'), "keeps the name: {item}");
            assert!(
                item.contains("comparing"),
                "and says what it is doing: {item}"
            );
        }

        #[test]
        fn events_before_any_tensor_leave_the_bar_alone() {
            let mut b = bar(10, 1);
            b.on(E::Loading("old"));
            b.on(E::Size {
                name: "w",
                old_bytes: 5,
                new_bytes: 5,
            });
            let p = b.bars.progress(0).expect("one bar");
            assert_eq!(p.item(), None, "nothing to name yet");
            assert_eq!(p.snapshot().0, 0);
        }
    }

    #[test]
    fn format_elapsed_scales() {
        use std::time::Duration;
        assert_eq!(format_elapsed(Duration::from_millis(850)), "850ms");
        assert_eq!(format_elapsed(Duration::from_millis(1500)), "1.5s");
        assert_eq!(format_elapsed(Duration::from_secs(125)), "2m5s");
    }

    #[test]
    fn config_default_proxy_only_for_s3_never_local() {
        let cfg = cli_config::CliConfig {
            ssh_proxy: Some("cfg@host".into()),
            ssh_venv: None,
        };
        // Config default engages for an s3:// source …
        assert_eq!(
            resolve_ssh_proxy(None, None, &cfg, true),
            Some(("cfg@host".to_string(), "~/venv".to_string()))
        );
        // … but NEVER hijacks a local path into an SSH read.
        assert_eq!(resolve_ssh_proxy(None, None, &cfg, false), None);
        // An explicit flag forces the proxy for any source (even a local path), and
        // overrides the config host + venv.
        assert_eq!(
            resolve_ssh_proxy(Some("flag@h".into()), Some("/v".into()), &cfg, false),
            Some(("flag@h".to_string(), "/v".to_string()))
        );
        // No flag, no config → a plain local read.
        assert_eq!(
            resolve_ssh_proxy(None, None, &cli_config::CliConfig::default(), true),
            None
        );
    }

    #[test]
    fn colon_prefix_reads_via_config_proxy_and_strips_the_marker() {
        let cfg = cli_config::CliConfig {
            ssh_proxy: Some("cfg@host".into()),
            ssh_venv: None,
        };
        // `:PATH` engages the config proxy (like s3://) and the `:` is stripped so the
        // remote reader gets the bare path.
        let (paths, remote) =
            resolve_remote_sources(&[PathBuf::from(":/opt/models/m")], None, None, &cfg).unwrap();
        assert_eq!(paths, vec![PathBuf::from("/opt/models/m")]);
        assert_eq!(remote, Some(("cfg@host".into(), "~/venv".into())));

        // A plain local path is untouched and stays local, even with a config proxy set.
        let (paths, remote) =
            resolve_remote_sources(&[PathBuf::from("/opt/models/m")], None, None, &cfg).unwrap();
        assert_eq!(paths, vec![PathBuf::from("/opt/models/m")]);
        assert_eq!(remote, None);

        // `:PATH` with no proxy resolvable is an error (don't silently read locally).
        let err = resolve_remote_sources(
            &[PathBuf::from(":/opt/models/m")],
            None,
            None,
            &cli_config::CliConfig::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("SSH proxy"), "{err}");
    }

    #[test]
    fn splits_reroot_suffix_off_the_source() {
        // `#subtree` splits off; the address keeps its `:`/scp/s3 form intact.
        assert_eq!(
            split_reroot(&PathBuf::from(":/opt/m/Kimi#language_model")),
            (PathBuf::from(":/opt/m/Kimi"), Some("language_model".into()))
        );
        assert_eq!(
            split_reroot(&PathBuf::from("s3://b/ckpt#language_model")),
            (PathBuf::from("s3://b/ckpt"), Some("language_model".into()))
        );
        // No suffix → unchanged. A trailing bare `#` is not a re-root.
        assert_eq!(
            split_reroot(&PathBuf::from("/opt/m/Kimi")),
            (PathBuf::from("/opt/m/Kimi"), None)
        );
        assert_eq!(
            split_reroot(&PathBuf::from("/opt/m/Kimi#")),
            (PathBuf::from("/opt/m/Kimi#"), None)
        );
        // Split on the LAST `#`, so a `#` inside the path is preserved.
        assert_eq!(
            split_reroot(&PathBuf::from("/od d/a#b/ckpt#model")),
            (PathBuf::from("/od d/a#b/ckpt"), Some("model".into()))
        );
    }

    #[test]
    fn scope_key_is_a_scope_not_a_rename() {
        // No root → identity. In scope → sub-path. Sibling → None (out of scope).
        assert_eq!(
            scope_key("model.norm.weight", None).as_deref(),
            Some("model.norm.weight")
        );
        assert_eq!(
            scope_key("language_model.model.norm.weight", Some("language_model")).as_deref(),
            Some("model.norm.weight")
        );
        assert_eq!(
            scope_key("vision_tower.enc.weight", Some("language_model")),
            None
        );
        // The prefix itself with no sub-path is not "inside" it.
        assert_eq!(scope_key("language_model", Some("language_model")), None);
    }

    #[test]
    fn scope_tensors_keeps_originals_and_drops_siblings() {
        use crate::tree::{Layout, Storage};
        let mk = |name: &str| TensorInfo {
            name: name.into(),
            dtype: "BF16".into(),
            shape: vec![7168],
            size_bytes: 14336,
            num_elements: 7168,
            storage: Storage::Unknown,
            source_path: "shard.safetensors".into(),
            layout: Layout::None,
        };
        let originals = vec![
            mk("language_model.model.norm.weight"),
            mk("vision_tower.encoder.0.weight"), // sibling — out of scope
        ];
        let scoped = scope_tensors(&originals, "language_model");
        // Only the in-scope tensor, re-keyed to its sub-path…
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].name, "model.norm.weight");
        // …while the source_path (for I/O) is preserved and the originals untouched.
        assert_eq!(scoped[0].source_path, "shard.safetensors");
        assert_eq!(originals[0].name, "language_model.model.norm.weight");
    }

    #[test]
    fn truncate_tail_keeps_the_end() {
        assert_eq!(truncate_tail("short", 10), "short"); // fits
        assert_eq!(truncate_tail("abcdefgh", 8), "abcdefgh"); // exact
        assert_eq!(truncate_tail("abcdefgh", 4), "…fgh"); // …+tail, total == max
        assert_eq!(truncate_tail("abcdefgh", 1), "…");
        // A long tensor name keeps its most-specific tail within the budget.
        let name = "model.layers.0.block_sparse_moe.experts.down_proj.weight";
        let t = truncate_tail(name, 20);
        assert_eq!(t.chars().count(), 20);
        assert!(t.starts_with('…') && t.ends_with("down_proj.weight"), "{t}");
    }

    /// A directory whose index lists only some of the `.safetensors` on disk (a
    /// partially-stale index) must still surface the extra files — the bug where
    /// `codebooks`/`qscales` shards were silently dropped.
    #[test]
    fn collects_extra_files_absent_from_a_stale_index() {
        let dir = std::env::temp_dir().join("ckpt_explorer_stale_index_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        // Three shards on disk; the index references only the first.
        fs::write(dir.join("model.safetensors"), b"x").unwrap();
        fs::write(dir.join("codebooks.safetensors"), b"x").unwrap();
        fs::write(dir.join("qscales.safetensors"), b"x").unwrap();
        fs::write(
            dir.join("model.safetensors.index.json"),
            br#"{"weight_map": {"w": "model.safetensors"}}"#,
        )
        .unwrap();

        let (files, _) =
            collect_safetensors_files(std::slice::from_ref(&dir), false, true).unwrap();
        let names: Vec<String> = files
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();

        for want in [
            "model.safetensors",
            "codebooks.safetensors",
            "qscales.safetensors",
        ] {
            assert!(
                names.iter().any(|n| n == want),
                "{want} should be collected; got {names:?}"
            );
        }
        // The shard listed in the index *and* found by the scan must appear once.
        let unique: HashSet<_> = files.iter().collect();
        assert_eq!(files.len(), unique.len(), "duplicate paths: {names:?}");

        let _ = fs::remove_dir_all(&dir);
    }
}
