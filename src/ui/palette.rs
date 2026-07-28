//! The app's colour palette — the single source of truth for how each kind of
//! thing is styled, so the same role looks the same on every screen. Change a
//! colour here and it updates everywhere it's used: one place to see whether
//! two things that look alike really are alike.

use ratatui::style::Color;

/// Interactive keys in hint lines (rendered bold).
pub(super) const KEY: Color = Color::Indexed(14);
/// Secondary / de-emphasised hint text (ranges, "to cancel", …).
pub(super) const DIM: Color = Color::Indexed(8);
/// Selected tree row (foreground on background).
pub(super) const SELECT_FG: Color = Color::Indexed(0);
pub(super) const SELECT_BG: Color = Color::Indexed(15);
/// The slice-jump input box (foreground on background).
pub(super) const INPUT_FG: Color = Color::Indexed(15);
pub(super) const INPUT_BG: Color = Color::Indexed(4);
/// Something missing / wrong / out of range.
pub(super) const ERROR: Color = Color::Indexed(9);
/// Filled-red *background* for an alert badge (white text on it) — high
/// luminance contrast that reads clearly on the grey status bar, where any
/// red *foreground* stays muddy against the mid-grey. The health *error* badge.
pub(super) const ALERT: Color = Color::Indexed(160);
/// Filled-orange *background* for the health badge when there are only
/// warnings (e.g. extra files on disk) — a softer alert than the red [`ALERT`],
/// which is reserved for real errors (missing files/tensors).
pub(super) const WARN_BG: Color = Color::Indexed(166);
/// Something present but unexpected (a softer alert than [`ERROR`]).
pub(super) const WARN: Color = Color::Indexed(11);
/// The bottom status bar (foreground on background).
pub(super) const STATUS_FG: Color = Color::Indexed(15);
pub(super) const STATUS_BG: Color = Color::Indexed(8);
/// A success accent used as a *foreground* (e.g. the "✓ copied" confirmation).
pub(super) const SUCCESS: Color = Color::Indexed(10);
/// Marks a tensor present on disk but missing from the index — a vivid red
/// that stands out clearly against the tree's default and dimmed text.
pub(super) const UNINDEXED: Color = Color::Indexed(196);
/// Group names and expand arrows in the tree — the primary accent (a bright
/// sky-cyan), so the structure stands out from the leaf tensors.
pub(super) const ACCENT: Color = Color::Indexed(81);
/// A tensor's data type (warm amber, so the type pops).
pub(super) const DTYPE: Color = Color::Indexed(215);
/// Layout bands coloured by dtype *family* (`checkpoint_studio_core::stats::DtypeClass`).
///
/// One colour per family rather than per dtype name: a dozen-entry palette stops being
/// readable, and "is this shard half-precision weights or 8-bit quantised" is a question
/// about the family. Chosen to stay distinguishable from each other *and* from the header
/// (`META`, violet) and gap (`DIM`, grey) bands they sit between, and to keep `DTYPE`'s
/// amber for half-precision since that is what most published weights are — so the common
/// case still reads in the colour the rest of the UI already uses for dtypes.
pub(super) const DTYPE_FLOAT_WIDE: Color = Color::Indexed(75); // steel blue — F32/F64
pub(super) const DTYPE_FLOAT_HALF: Color = DTYPE; //              amber      — F16/BF16
pub(super) const DTYPE_FLOAT_NARROW: Color = Color::Indexed(213); // pink     — F8_*
pub(super) const DTYPE_INT_WIDE: Color = Color::Indexed(114); //   green      — I32/I64
pub(super) const DTYPE_INT_NARROW: Color = Color::Indexed(179); // ochre      — I8/U8/I16
pub(super) const DTYPE_BOOL: Color = Color::Indexed(146); //       pale slate — BOOL
pub(super) const DTYPE_OTHER: Color = Color::Indexed(245); //      neutral grey
/// Metadata entries (the `†` marker and the entry name) — a muted slate
/// violet, distinct from the cyan groups and amber dtypes but quiet enough
/// that metadata reads as a side note rather than competing with tensors.
pub(super) const META: Color = Color::Indexed(103);
/// Zebra striping for the numeric grid — two subtle dark backgrounds (one
/// "dark", one "less dark") that alternate to guide the eye along the rows
/// or columns, like a dim highlighter.
pub(super) const STRIPE_DARK: Color = Color::Indexed(234);
pub(super) const STRIPE_LITE: Color = Color::Indexed(237);
/// Background for floating pop-ups (legend, the `y` command panel, message
/// screens) — a neutral dark grey a few shades above black, in the same
/// family as the zebra greys above, so an overlay reads as a raised surface
/// over the main screen while staying within the dark theme. Light/accent
/// foregrounds keep their contrast; dim text stays legible.
pub(super) const PANEL_BG: Color = Color::Indexed(236);

/// Backdrop behind a full-frame message screen ([`Backdrop::Fill`]): one shade
/// darker than [`PANEL_BG`], so the box reads as a raised card over an even,
/// dark field. (Floating pop-ups like the legend keep the live screen behind
/// them and don't use this.)
pub(super) const SCRIM: Color = Color::Indexed(234);

// A structural diff's three outcomes. Aliases rather than new colours, so a change reads
// the same in the terminal as in `diff`'s own ANSI output (green 32 / yellow 33 / red 31)
// and in the browser's diff view.
pub(super) const ADDED: Color = SUCCESS;
pub(super) const REMOVED: Color = ERROR;
pub(super) const CHANGED: Color = WARN;
