//! Shared visual vocabulary: the glyphs that mark a tensor's storage, the
//! span constructors for the recurring text roles (dim / success / key), and the
//! colour ramps.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;

use super::palette;

/// Marks a tensor that's on disk but not listed in the index (an "extra"),
/// shown in [`palette::UNINDEXED`] in the tree, detail screen and legends.
pub(crate) const UNINDEXED_MARK: &str = "✚";

/// Marks a file whose bytes have more than one name (`st_nlink > 1`) — the same copy
/// reachable from elsewhere on the filesystem, so its size is shared, not its own.
pub(crate) const HARDLINK_MARK: &str = "⧉";

/// Marks a checkpoint file whose header wouldn't parse — the read carried on without it,
/// so its tensors are absent from everything else.
pub(crate) const UNREADABLE_MARK: &str = "✗";

/// Storage tag for a tensor stored uncompressed on disk. Shared by the tree row,
/// the detail screen and the legend so the wording stays consistent.
pub(super) const UNCOMPRESSED_TAG: &str = "(uncompressed)";

/// On-disk compression codec marker, e.g. `⇩ lz4`. Shared by the tree row, the
/// detail screen and the legend so the glyph stays consistent.
pub(super) const COMPRESSED_MARK: &str = "⇩";

/// Separator between a tensor's logical size and its (smaller) on-disk size,
/// e.g. `593 MiB → 588 MiB`. Shared by the tree rows and the legend.
pub(super) const SIZE_ARROW: &str = "→";

/// Translate a palette [`Color`] to the equivalent `yansi` color, so the JSON
/// highlighter can be styled from the same constants as the rest of the UI. The
/// 16 ANSI-named indices map to yansi's named colors (so e.g. `Indexed(8)` emits
/// the bright-black SGR, not `38;5;8`); other indices use the 256-color cube.
pub(super) fn to_yansi(color: Color) -> yansi::Color {
    use yansi::Color as Y;
    match color {
        Color::Black | Color::Indexed(0) => Y::Black,
        Color::Red | Color::Indexed(1) => Y::Red,
        Color::Green | Color::Indexed(2) => Y::Green,
        Color::Yellow | Color::Indexed(3) => Y::Yellow,
        Color::Blue | Color::Indexed(4) => Y::Blue,
        Color::Magenta | Color::Indexed(5) => Y::Magenta,
        Color::Cyan | Color::Indexed(6) => Y::Cyan,
        Color::Gray | Color::Indexed(7) => Y::White,
        Color::DarkGray | Color::Indexed(8) => Y::BrightBlack,
        Color::LightRed | Color::Indexed(9) => Y::Red,
        Color::LightGreen | Color::Indexed(10) => Y::Green,
        Color::LightYellow | Color::Indexed(11) => Y::Yellow,
        Color::LightBlue | Color::Indexed(12) => Y::Blue,
        Color::LightMagenta | Color::Indexed(13) => Y::Magenta,
        Color::LightCyan | Color::Indexed(14) => Y::Cyan,
        Color::White | Color::Indexed(15) => Y::White,
        Color::Indexed(n) => Y::Fixed(n),
        Color::Rgb(r, g, b) => Y::Rgb(r, g, b),
        // Reset and the dark neutrals have no yansi twin; anything NEW in ratatui's
        // `Color` should be a compile error here rather than a silent Primary.
        Color::Reset => Y::Primary,
    }
}

/// One styled span for a tree row: the kind's color normally, or the selection
/// highlight (black on white) when the row is selected (so the highlight reads
/// cleanly over the whole row, matching the old inverse-video selection).
pub(super) fn tree_span(selected: bool, color: Color, text: impl Into<String>) -> Span<'static> {
    let style = if selected {
        Style::default()
            .fg(palette::SELECT_FG)
            .bg(palette::SELECT_BG)
    } else {
        Style::default().fg(color)
    };
    Span::styled(text.into(), style)
}

/// A dimmed span (field labels, chrome) for the detail screen.
pub(crate) fn dim_span(text: impl Into<String>) -> Span<'static> {
    Span::styled(text.into(), Style::default().fg(palette::DIM))
}

/// A span in the vivid red that marks something on disk but absent from the index —
/// the tree's tensor mark, the detail screen's flag, the file browser's row, and the
/// legends explaining them. Exported so a legend doesn't need the palette itself; one
/// definition is what keeps the colour from drifting across four screens.
pub(crate) fn unindexed_span(text: impl Into<String>) -> Span<'static> {
    Span::styled(text.into(), Style::default().fg(palette::UNINDEXED))
}

/// The colour something missing or wrong is drawn in — an unreadable shard's mark on a
/// file row and in the legend that explains it. A function rather than a re-exported
/// constant so `palette` stays private to `ui`.
pub(crate) fn error_color() -> Color {
    palette::ERROR
}

/// A bold green span — a "✓ copied" style confirmation, matching the copy
/// flashes elsewhere. Used by the preview pop-up's copy hint.
pub(crate) fn success_span(text: impl Into<String>) -> Span<'static> {
    Span::styled(
        text.into(),
        Style::default()
            .fg(palette::SUCCESS)
            .add_modifier(Modifier::BOLD),
    )
}

/// A bold bright-cyan key span (e.g. `s`, `d`) — the Ratatui equivalent of the
/// raw [`key_hint`].
pub(super) fn key_span(key: impl Into<String>) -> Span<'static> {
    Span::styled(
        key.into(),
        Style::default()
            .fg(palette::KEY)
            .add_modifier(Modifier::BOLD),
    )
}

/// Map a normalized value in `[0, 1]` to a blue→green→red 256-color ramp
/// (the 6×6×6 ANSI color cube, indices 16..=231).
pub(super) fn heat_color(t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    let r = (t * 5.0).round() as u8;
    let b = ((1.0 - t) * 5.0).round() as u8;
    let g = ((t - 0.5).abs().mul_add(-2.0, 1.0) * 5.0).round() as u8;
    Color::Indexed(16 + 36 * r + 6 * g + b)
}
