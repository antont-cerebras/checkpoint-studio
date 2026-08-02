//! The CLI-facing "open request" surface: what `main.rs` asks the explorer to show on
//! startup, plus the small enums that describe it.
//!
//! Split out of `explorer/mod.rs` because these types are the boundary between the CLI
//! and the TUI — they're constructed by argument parsing, round-tripped through the `y`
//! command, and otherwise independent of the interactive machinery.

use super::{DataLayout, NumBase, Result, StripeMode, ViewDtype};

/// Which screen to jump straight to for a `--tensor` opened from the CLI.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenView {
    /// The tensor detail screen.
    Detail,
    /// The numeric values grid (`v`).
    Values,
    /// The ASCII heatmap (`m`).
    Heatmap,
    /// The tree browser, with the tensor revealed and highlighted (no view
    /// opened) — what `y` copies from the tree (`--tree`).
    Tree,
}

/// A bulk expansion state for the tree browser (`--tree-state`, the `E` / `C`
/// keys). Absent leaves the natural default (root expanded, layers collapsed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TreeState {
    Expanded,
    Collapsed,
}

impl TreeState {
    /// The `--tree-state` value that names this state.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Expanded => "expanded",
            Self::Collapsed => "collapsed",
        }
    }
}

/// Parse a `--tree-state` value.
pub(crate) fn parse_tree_state(s: &str) -> Result<TreeState, String> {
    match s.to_ascii_lowercase().as_str() {
        "expanded" => Ok(TreeState::Expanded),
        "collapsed" => Ok(TreeState::Collapsed),
        other => Err(format!(
            "invalid tree state '{other}' (expected: expanded, collapsed)"
        )),
    }
}

/// Output format for `--print-tree`. (The `t` copy shortcut always uses `Text`.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum TreeFormat {
    /// The grouped tree as text — one row per node, fully expanded, in the same
    /// layout the browser shows (no viewport limit, no header/footer chrome).
    #[default]
    Text,
    /// A `model.safetensors.index.json`-style object: a `metadata.total_size`
    /// and a `weight_map` of tensor name → its shard file. `-v` adds a `tensors`
    /// block with each tensor's dtype / shape / element count.
    Json,
}

/// How much per-tensor detail the tree export includes; raised by repeating `-v`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TreeDetail {
    /// Text: names + the browser's own fields. JSON: the bare index.json shape.
    Compact,
    /// Text: each tensor row also names its source file. JSON: adds a `tensors`
    /// block (dtype, shape, element count) alongside the `weight_map`.
    Full,
}

impl TreeDetail {
    /// Map a repeated-`-v` count to a detail level (0 → compact, ≥1 → full).
    pub(crate) fn from_verbosity(count: u8) -> Self {
        if count == 0 {
            Self::Compact
        } else {
            Self::Full
        }
    }
}

/// Which structure an export dumps: the grouped tree or a flat tensor list.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ExportShape {
    Tree,
    Tensors,
}

/// One entry in the `t` copy menu — a (shape, format, detail) combination, i.e.
/// one CLI `--print-tree`/`--print-tensors` [`--format json`] [`-v`] variant.
#[derive(Clone, Copy)]
pub(super) struct ExportChoice {
    pub(super) label: &'static str,
    pub(super) shape: ExportShape,
    pub(super) format: TreeFormat,
    pub(super) detail: TreeDetail,
}

/// The eight export variants offered by `t`, one per CLI combination. `+ files`
/// (text) appends each tensor's source file; `+ details` (JSON) adds a
/// per-tensor block/objects — both what `-v` does.
pub(super) const EXPORT_CHOICES: &[ExportChoice] = {
    use ExportShape::{Tensors, Tree};
    use TreeDetail::{Compact, Full};
    use TreeFormat::{Json, Text};
    &[
        ExportChoice {
            label: "tree · text",
            shape: Tree,
            format: Text,
            detail: Compact,
        },
        ExportChoice {
            label: "tree · text + files",
            shape: Tree,
            format: Text,
            detail: Full,
        },
        ExportChoice {
            label: "tree · JSON (index.json-style)",
            shape: Tree,
            format: Json,
            detail: Compact,
        },
        ExportChoice {
            label: "tree · JSON + tensor details",
            shape: Tree,
            format: Json,
            detail: Full,
        },
        ExportChoice {
            label: "tensors · text",
            shape: Tensors,
            format: Text,
            detail: Compact,
        },
        ExportChoice {
            label: "tensors · text + files",
            shape: Tensors,
            format: Text,
            detail: Full,
        },
        ExportChoice {
            label: "tensors · JSON (names)",
            shape: Tensors,
            format: Json,
            detail: Compact,
        },
        ExportChoice {
            label: "tensors · JSON + details",
            shape: Tensors,
            format: Json,
            detail: Full,
        },
    ]
};

/// How many lines of the highlighted export the `t` menu previews.
pub(super) const MENU_PREVIEW_LINES: usize = 14;

/// What a `--tensor` / `--metadata` open targets — mutually exclusive by
/// construction (both-set was representable with two `Option`s), with "neither"
/// meaning the sole tensor of a single-tensor checkpoint.
pub(crate) enum OpenTarget {
    /// No `--tensor`/`--metadata`: the sole tensor (a single-tensor file — always
    /// the case for `.npy` — needs no flag); ambiguous when there's more than one.
    SoleTensor,
    /// An exact tensor name (`--tensor`).
    Tensor(String),
    /// An exact metadata entry to reveal in the tree (`--metadata`); metadata lives
    /// only in the tree, so there's no separate view.
    Metadata(String),
}

impl OpenTarget {
    /// The explicit tensor name, if `--tensor` named one.
    pub(crate) fn tensor(&self) -> Option<&str> {
        match self {
            Self::Tensor(n) => Some(n),
            Self::SoleTensor | Self::Metadata(_) => None,
        }
    }
    /// The metadata entry name, if `--metadata` named one.
    pub(crate) fn metadata(&self) -> Option<&str> {
        match self {
            Self::Metadata(n) => Some(n),
            Self::SoleTensor | Self::Tensor(_) => None,
        }
    }
}

/// The histogram request (`--histogram` / `--bins N`): a bucket count implies
/// showing the histogram, so it's one enum, not a bool + an `Option` that could
/// disagree (bins set but histogram off).
pub(crate) enum HistogramReq {
    Off,
    /// Show it, buckets chosen automatically.
    Auto,
    /// Show it with a fixed bucket count.
    Bins(usize),
}

impl HistogramReq {
    pub(crate) fn on(&self) -> bool {
        !matches!(self, Self::Off)
    }
    pub(crate) fn bins(&self) -> Option<usize> {
        match self {
            Self::Bins(n) => Some(*n),
            Self::Off | Self::Auto => None,
        }
    }
}

/// The health-popup request (`--health` / `--health-findings`): the findings
/// detail implies opening the popup, so one 3-state enum, not two bools where
/// `findings` could be set without `health`.
pub(crate) enum HealthReq {
    Off,
    Summary,
    Findings,
}

impl HealthReq {
    pub(crate) fn wants(&self) -> bool {
        !matches!(self, Self::Off)
    }
    pub(crate) fn findings(&self) -> bool {
        matches!(self, Self::Findings)
    }
}

/// The stats-view request (`--stats` / `--stats-shards`): the shard breakdown
/// implies opening the view — one enum, not two bools.
pub(crate) enum StatsReq {
    Off,
    Summary,
    Shards,
}

impl StatsReq {
    pub(crate) fn wants(&self) -> bool {
        !matches!(self, Self::Off)
    }
    pub(crate) fn shards(&self) -> bool {
        matches!(self, Self::Shards)
    }
}

/// A tensor + view to open on startup, from the CLI flags.
pub(crate) struct OpenRequest {
    /// What to open (tensor / metadata / the sole tensor).
    pub target: OpenTarget,
    /// Which screen to show.
    pub view: OpenView,
    /// The value-histogram request for the detail screen (the `h`/`b` keys).
    pub histogram: HistogramReq,
    /// Optional dtype reinterpretation to apply first.
    pub dtype: Option<ViewDtype>,
    /// Which data-view layout to force (`--edge`/`--overview`/`--window`);
    /// `None` keeps the session default.
    pub layout: Option<DataLayout>,
    /// The window layout's top-left corner (row, col), from `--window=ROW,COL`.
    pub window_at: Option<(usize, usize)>,
    /// The edges layout's head/tail split (row, col fractions in `0..=1`), from
    /// `--edge=RFRAC,CFRAC`.
    pub edge_split: Option<(f32, f32)>,
    /// Optional zebra-striping mode to apply (numeric grid).
    pub zebra: Option<StripeMode>,
    /// Optional numeral base for the numeric grid (`--base dec/hex/oct/bin`).
    pub base: Option<NumBase>,
    /// Optional starting slice (3D tensors), as a raw `N` or `N%` string
    /// resolved against the tensor's slice count.
    pub slice: Option<String>,
    /// Optional shape override (a reshape with a matching element count), as a
    /// raw string like `10,100` or `-1,768`.
    pub shape: Option<String>,
    /// Start the statistics scan immediately on the detail view.
    pub compute_stats: bool,
    /// Bulk tree expansion (`--tree-state`, the `E` / `C` keys); `None` keeps the
    /// natural default.
    pub tree_state: Option<TreeState>,
    /// Open the tree in search mode with this query (`--search`, the `/` key).
    pub search: Option<String>,
    /// Overlay the requested screen's legend (`--legend`, the `l` key). A
    /// render-time aid (for `--plain` / inspection); not part of `y`'s round-trip
    /// since the legend is a transient overlay you dismiss.
    pub legend: bool,
    /// The health-check popup request (`--health` / `--health-findings`, the `h`
    /// key + its `f` toggle). Round-trips through `y`.
    pub health: HealthReq,
    /// The checkpoint-stats view request (`--stats` / `--stats-shards`, the `s`
    /// key + its `f` toggle). Round-trips through `y`.
    pub stats: StatsReq,
    /// Open the compare screen against this checkpoint (`--diff-against PATH`, the
    /// tree's `d` command). Round-trips through `y`.
    pub diff_against: Option<String>,
    /// Open the **side-by-side** compare screen against this checkpoint (`--compare-with PATH`, the
    /// palette's *Compare side by side*) — the aligned two-column tree, as against `diff_against`'s one-page report.
    /// Round-trips through `y`.
    pub compare_with: Option<String>,
    /// Show every layer there rather than folding uniform index families onto one row each
    /// (`--compare-full`, the `k` key). Round-trips through `y`.
    pub compare_full: bool,
    /// Order the flat search / filter list (`--sort KEY[.DIR]`, the `o` / `O` keys).
    /// Round-trips through `y`.
    pub sort: Option<(crate::viewstate::SortKey, crate::viewstate::SortDir)>,
    /// Open the compact (family-folded) tree (`--compact`, the `k` key). Round-trips
    /// through `y`.
    pub compact: bool,
    /// Render the view once and exit without interactive navigation.
    pub exit_after: bool,
    /// Land in the file browser (`--files`, the `Tab` toggle) once the tree is
    /// up. Round-trips through `y`: the file view's `y` copies `… --files`.
    pub files_view: bool,
    /// Open straight into the safetensors layout map for this file (`--layout
    /// PATH`). Round-trips through `y` from the layout view.
    pub layout_file: Option<String>,
    /// Preselect this tensor in the layout map (`--layout-select NAME`), so the
    /// layout view's `y` round-trips the selection.
    pub layout_select: Option<String>,
    /// Open straight into the in-place rename editor (`--rename`, the `R` key).
    /// Round-trips through `y`; only honoured for a local safetensors checkpoint.
    pub rename: bool,
    /// Seed the rename editor's rule pairs (`--rename-rule 'SRC=>TGT'`, repeatable),
    /// each a schema `source => new-name`. What the editor's `y` records.
    pub rename_rules: Vec<String>,
}
