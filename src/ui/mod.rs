//! The Ratatui rendering layer — everything the TUI puts on screen.
//!
//! [`UI`] is the facade: each screen is one `UI::render_*` entry point, and that
//! screen's `impl UI` block lives in the screen's own module, next to the row
//! builders and geometry helpers only it uses. Nothing here holds state — a
//! renderer takes a `&mut Frame` plus what to draw (a `DrawConfig`, or a handful
//! of borrowed slices) and hands back the click regions it laid out. The state
//! lives in [`Explorer`](crate::explorer), which drives these.
//!
//! - one module per screen: `tree`, `files`, `layout`, `detail`, `data`, `stats`,
//!   `rename`, `check`, `prompts`, `notice`
//! - one per piece of shared chrome: `badge`, `hints`, `legend`, `popup`, `scroll`
//! - and the shared vocabulary: `palette` (colours), `theme` (glyphs and the
//!   recurring span roles), `text` (wrapping and number formatting), `json`
//!   (metadata highlighting)
//!
//! What stays here is what both sides speak: what a screen needs in order to draw
//! ([`DrawConfig`], [`StatsView`], [`Overlay`]) and what a click can turn into
//! ([`Link`], [`ChipRegions`], [`LinkRegions`]).

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crossterm::event::KeyEvent;
use ratatui::layout::Rect;

use crate::sample::{PackingSchema, Stats};
use crate::tree::TreeNode;
// The numeric-grid presentation enums live in core (frontend-free, so the kernel
// can own them); re-exported here so the `crate::ui::{StripeMode,NumBase,…}` paths
// the renderers use keep resolving.
pub(crate) use crate::viewstate::{NumBase, StripeMode, parse_num_base, parse_stripe_mode};

mod badge;
mod check;
mod data;
mod detail;
mod diff;
mod files;
mod hints;
mod json;
mod layout;
mod legend;
mod notice;
mod palette;
mod popup;
mod prompts;
mod rename;
mod scroll;
mod stats;
#[cfg(test)]
mod tests_support;
mod text;
mod theme;
mod tree;

// What the rest of the app addresses as `ui::…`: the screens' inputs and the
// row/line builders the plain-text and clipboard exports reuse.
pub(crate) use badge::{AccessBadge, Badge, HealthAlert, status_badges};
pub(crate) use check::CheckPopup;
pub(crate) use hints::{HelpCtx, data_view_footer_lines, region_at, region_hit, shortcut_help};
pub(crate) use json::{highlight_json_lines, highlight_json_lines_inline, plain_json_lines_inline};
pub(crate) use legend::Legend;
pub(crate) use popup::render_shortcut_bubble;
pub(crate) use rename::{RenameCompletion, RenameRuleView, RenameView};
pub(crate) use scroll::VScrollbar;
pub(crate) use theme::{UNINDEXED_MARK, dim_span, success_span, unindexed_span};
pub(crate) use tree::{tensor_list_line, tree_row_line, tree_row_text};
// The footer builders are `ui` internals — the screens above call them directly.
// They're re-exported for tests only, where the explorer's mode tests assert that
// every mode's footer advertises that mode's command keys.
#[cfg(test)]
pub(crate) use detail::detail_footer_lines;
#[cfg(test)]
pub(crate) use hints::{
    ChipHit, data_view_footer_wrapped_lines, files_hint_lines, layout_hint_lines,
    rename_hint_lines, stats_hint_lines, tree_hint_lines,
};

/// A `Rect` of `height` rows starting at `y`, clamped to what `area` actually has.
///
/// Ratatui **panics** when a widget's rect runs past the buffer instead of clipping, so
/// any screen that computes a height — a header of N field lines, a footer wrapped to
/// the terminal width — can kill the TUI on a pane smaller than its own chrome. That
/// happened on four screens (the two data views, the detail screen, the tree and the
/// stats report) at 10×10 and smaller; every one of them now goes through here, and
/// `ui::small_terminal` sweeps all of them at seven sizes.
pub(crate) fn fit_rows(area: Rect, y: u16, height: u16) -> Rect {
    let y = y.min(area.height);
    Rect {
        x: area.x,
        y,
        width: area.width,
        height: height.min(area.height.saturating_sub(y)),
    }
}

/// A still-forming scan's progress indicator: a spinner glyph, the elapsed time,
/// and an optional completed fraction (`None` when the total isn't known).
pub(crate) type ScanProgress = (char, Duration, Option<f64>);

pub(crate) struct DrawConfig<'a> {
    pub tree: &'a [(TreeNode, usize)],
    pub current_file: &'a str,
    pub file_idx: usize,
    pub total_files: usize,
    pub selected_idx: usize,
    pub scroll_offset: usize,
    pub search_mode: bool,
    pub search_query: &'a str,
    /// Caret position within `search_query`, as a character index in `0..=len`.
    pub search_cursor: usize,
    /// The active persistent filter query (`--filter` / palette), or `""` when
    /// none — surfaced in the title so a narrowed tree reads as filtered, not tiny.
    pub filter_query: &'a str,
    /// Leading glyph for the status bar (e.g. `▪`, `▸`, `†`).
    pub status_icon: &'a str,
    /// Bottom status line: a tensor's full name, or a group's source
    /// file(s)/directory.
    pub status_bar: &'a str,
    /// Second status line, below `status_bar`: a tensor's source file (empty for
    /// groups).
    pub status_secondary: &'a str,
    /// Whether the loaded checkpoint can be repacked (a single HDF5 file), which
    /// gates the `r` hint.
    pub can_repack: bool,
    /// Whether the loaded checkpoint can be renamed in place (a writable local
    /// safetensors checkpoint), which gates the `R` hint.
    pub can_rename: bool,
    /// `source_path`s of tensors present on disk but not listed in the index
    /// (a stale `model.safetensors.index.json`), flagged in the tree.
    pub unindexed: &'a HashSet<String>,
    /// Per-tensor fused-codebook packing schemas, keyed by tensor name. A tensor
    /// with one shows its logical (unmerged) dtype and shape beside the physical.
    pub packing_schemas: &'a HashMap<String, PackingSchema>,
    /// A transient "✓ Copied …" confirmation to flash on the bottom line (over
    /// the secondary status), set by the tree's copy shortcuts.
    pub copied_flash: Option<&'a str>,
    /// Whether this frame is drawn to the live, interactive terminal. Gates the
    /// scroll bar: a headless `--plain` / screen-copy render is a static text
    /// dump with no viewport, so it shows no bar (see [`UI::tree_scrollbar`]).
    pub interactive: bool,
    /// The bottom-right status badges (access / health / metadata-only), from
    /// [`UI::status_badges`], right-to-left; and which one the mouse is over (for
    /// its hover bubble). One uniform bar — see [`UI::render_badge_bar`].
    pub badges: &'a [Badge],
    pub hovered_badge: Option<usize>,
}

/// A clickable link in the UI — the app-wide primitive for "click a name to jump".
/// A safetensors filename links to its byte-layout view; a *concrete* tensor name
/// links to its place in the tree (a schema with `{layer}`/`{expert}` placeholders
/// matches many tensors, so it is never a link). Recorded per screen and dispatched
/// by [`Explorer::open_link`](crate::explorer). See the `links` field on `Explorer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Link {
    /// Open the byte-layout view of this safetensors file (its full path).
    Layout(String),
    /// Reveal this concrete tensor in the tree.
    Tree(String),
}

/// A frame's footer-chip click regions — each replays a [`KeyEvent`].
pub(crate) type ChipRegions = Vec<(Rect, KeyEvent)>;
/// A frame's navigation-link regions — each opens another view via a [`Link`].
pub(crate) type LinkRegions = Vec<(Rect, Link)>;

/// How a screen should render the statistics area: not computed yet, a scan in
/// progress (with a spinner + running timer), or the finished `Stats`.
#[derive(Clone, Copy)]
pub(crate) enum StatsView<'a> {
    Pending,
    Computing {
        spinner: char,
        elapsed: Duration,
        /// Fraction scanned so far (`0.0..=1.0`) for the progress bar, or `None`
        /// when unknown (then only the spinner + timer show).
        progress: Option<f64>,
    },
    Ready(&'a Stats),
}

impl StatsView<'_> {
    /// The exact whole-tensor value range, available only once the scan has
    /// finished. Used to size numeric cells to the data actually present.
    pub(crate) fn value_range(&self) -> Option<(f64, f64)> {
        match self {
            StatsView::Ready(s) => Some((s.min, s.max)),
            StatsView::Pending | StatsView::Computing { .. } => None,
        }
    }
}

/// A floating pop-up the detail screen can show *over* its live frame — drawn as
/// the last layer of [`UI::render_detail`] so the screen behind it keeps
/// redrawing (a running scan's progress animates) while it's up. Dismissed by
/// any key. Composited via [`UI::render_legend_band`] / [`UI::render_command_band`].
pub(crate) enum Overlay {
    /// The context-sensitive glyph legend (`l`).
    Legend(Legend),
    /// The copied CLI command box (`y`); holds the command to display.
    Command(String),
    /// A metadata-only / unavailable notice (e.g. a remote `--ssh-proxy` source has
    /// no local bytes for data views); holds the message to display.
    Notice(String),
}

pub(crate) struct UI;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_base_parses_aliases() {
        assert_eq!(parse_num_base("dec"), Ok(NumBase::Decimal));
        assert_eq!(parse_num_base("DECIMAL"), Ok(NumBase::Decimal));
        assert_eq!(parse_num_base("hex"), Ok(NumBase::Hex));
        assert_eq!(parse_num_base("16"), Ok(NumBase::Hex));
        assert_eq!(parse_num_base(" Oct "), Ok(NumBase::Octal));
        assert_eq!(parse_num_base("bin"), Ok(NumBase::Binary));
        assert!(parse_num_base("base64").is_err());
    }

    #[test]
    fn num_base_cycles_and_round_trips_its_label() {
        // dec → hex → oct → bin → dec
        assert_eq!(NumBase::Decimal.next(), NumBase::Hex);
        assert_eq!(NumBase::Hex.next(), NumBase::Octal);
        assert_eq!(NumBase::Octal.next(), NumBase::Binary);
        assert_eq!(NumBase::Binary.next(), NumBase::Decimal);
        for b in [
            NumBase::Decimal,
            NumBase::Hex,
            NumBase::Octal,
            NumBase::Binary,
        ] {
            assert_eq!(parse_num_base(b.label()), Ok(b));
        }
    }

    #[test]
    fn num_base_digit_widths_match_bit_count() {
        // 32-bit element (e.g. F32/I32): 8 hex, 11 octal, 32 binary digits.
        assert_eq!(NumBase::Hex.digits(32), 8);
        assert_eq!(NumBase::Octal.digits(32), 11);
        assert_eq!(NumBase::Binary.digits(32), 32);
        // 8-bit and 4-bit elements.
        assert_eq!(NumBase::Hex.digits(8), 2);
        assert_eq!(NumBase::Hex.digits(4), 1);
        assert_eq!(NumBase::Octal.digits(8), 3);
    }
}

/// Every screen is drawn by bottom-pinning a **wrapped** footer under a header, so on a
/// small pane either can need more rows than the pane has — and a `Paragraph` rect that
/// runs past the buffer is a Ratatui panic, i.e. the TUI dying because someone shrank
/// their window. The data views and the detail screen both did this at 10×10; this
/// sweeps the rest so the next one is caught here rather than in a user's terminal.
#[cfg(test)]
mod small_terminal {
    use super::*;
    use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo};

    const SIZES: [(u16, u16); 7] = [
        (4, 2),
        (10, 10),
        (16, 4),
        (24, 12),
        (48, 20),
        (120, 3),
        (200, 60),
    ];

    fn tensor(name: &str) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dtype: "F32".into(),
            shape: vec![8, 8],
            size_bytes: 256,
            num_elements: 64,
            storage: Storage::Raw,
            source_path: "/ckpt/model.safetensors".into(),
            layout: Layout::ByteRange { start: 0, end: 256 },
        }
    }

    fn draw(what: &str, f: impl Fn(&mut ratatui::Frame) + Copy) {
        for (w, h) in SIZES {
            assert!(
                crate::tui::headless_render(w, h, f).is_ok(),
                "{what} panicked at {w}x{h}"
            );
        }
    }

    #[test]
    fn the_tree_screen_fits_any_terminal() {
        let nodes = vec![(
            TreeNode::Tensor {
                info: tensor("model.layers.0.weight"),
                label: None,
            },
            0usize,
        )];
        let unindexed = HashSet::new();
        let schemas = HashMap::new();
        let badges = status_badges(AccessBadge::ReadOnly, None, false);
        let config = DrawConfig {
            tree: &nodes,
            current_file: "/ckpt/model.safetensors",
            file_idx: 0,
            total_files: 1,
            selected_idx: 0,
            scroll_offset: 0,
            search_mode: false,
            search_query: "",
            search_cursor: 0,
            filter_query: "",
            status_icon: "▪",
            status_bar: "model.layers.0.weight",
            status_secondary: "/ckpt/model.safetensors",
            can_repack: false,
            can_rename: false,
            unindexed: &unindexed,
            packing_schemas: &schemas,
            copied_flash: None,
            interactive: true,
            badges: &badges,
            hovered_badge: None,
        };
        draw("the tree", |f| {
            UI::render_tree(f, &config);
        });
        // …and while searching, which adds the query row.
        let searching = DrawConfig {
            search_mode: true,
            search_query: "weight",
            search_cursor: 6,
            ..config
        };
        draw("the tree while searching", |f| {
            UI::render_tree(f, &searching);
        });
    }

    #[test]
    fn the_stats_screen_fits_any_terminal() {
        let stats = crate::stats::CheckpointStats::compute(&[tensor("a.weight")], None, None);
        draw("the stats screen", |f| {
            UI::render_stats_frame(f, &stats, 0, false);
        });
        draw("the stats screen with shards expanded", |f| {
            UI::render_stats_frame(f, &stats, 0, true);
        });
    }

    #[test]
    fn the_metadata_screen_fits_any_terminal() {
        let m = MetadataInfo {
            name: "format".into(),
            value: "pt".into(),
            value_type: "string".into(),
        };
        draw("the metadata screen", |f| UI::render_metadata_detail(f, &m));
    }

    #[test]
    fn the_files_browser_fits_any_terminal() {
        let rows = vec![crate::filetree::FileRow {
            depth: 0,
            name: "model.safetensors".into(),
            path: std::path::PathBuf::from("/ckpt/model.safetensors"),
            size: 256,
            kind: crate::filetree::FileRowKind::File {
                kind: crate::filetree::FileKind::Checkpoint,
                // Annotated, so the widest form of the row is the one squeezed here.
                shard: Some(crate::filetree::ShardTensors {
                    tensors: 1062,
                    params: 3_000_000_000,
                    params_share: 0.0641,
                }),
                size_share: 1.0,
                index: Some(crate::filetree::IndexMembership::Listed),
            },
        }];
        let badges = status_badges(AccessBadge::ReadOnly, None, false);
        draw("the file browser", |f| {
            UI::render_files(f, "/ckpt", &rows, 0, 0, None, true, &badges, None);
        });
    }

    #[test]
    fn the_layout_map_fits_any_terminal() {
        let map = crate::safelayout::from_tensors(
            "/ckpt/model.safetensors",
            512,
            64,
            &[tensor("a.weight")],
            &[],
        );
        draw("the layout map", |f| {
            UI::render_layout(f, &map, 0, 0, None, true);
        });
    }

    #[test]
    fn the_check_report_fits_any_terminal() {
        let report = crate::check::run(
            "/ckpt".into(),
            &[tensor("a.weight")],
            &[],
            &[],
            &[],
            None,
            &crate::filter::NameFilter::parse(&[]).expect("empty filter"),
            false,
            1,
        );
        for state in [
            CheckPopup::Idle {
                copied: None,
                can_scan: true,
            },
            CheckPopup::Scanning {
                done: 1,
                total: 4,
                frame: 0,
            },
        ] {
            draw("the check popup", |f| {
                UI::render_check_report(f, &report, state, 0, true);
            });
        }
    }

    #[test]
    fn the_overlays_fit_any_terminal() {
        draw("the legend", |f| UI::render_legend_band(f, Legend::Tree));
        draw("the command band", |f| {
            UI::render_command_band(f, "checkpoint-studio --tensor a.weight --values");
        });
        draw("the notice box", |f| {
            UI::render_notice_box(f, "metadata-only — data views need the file locally");
        });
        draw("the copied flash", |f| {
            UI::render_copied_flash(f, "the tensor name");
        });
        draw("a message", |f| {
            UI::render_message(f, "Title", "A message body");
        });
    }
}
