//! The legend band — the `?`-key explainer for every glyph and colour the
//! screens use. Kept next to nothing else so it's obvious when a new marker was
//! added without documenting it.

use ratatui::Frame;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::UI;
use super::badge::{HEALTH_BADGE, METADATA_BADGE};
use super::palette;
use super::popup::{Backdrop, render_popup_box};
use super::theme::{
    COMPRESSED_MARK, SIZE_ARROW, UNCOMPRESSED_TAG, UNINDEXED_MARK, dim_span, heat_color,
};

/// Which screen a legend explains. The legend (`l`) is context-sensitive — it
/// lists only the glyphs and colour cues that appear on the screen it was opened
/// from.
#[derive(Clone, Copy)]
pub enum Legend {
    Tree,
    Detail,
    Heatmap,
    Values,
    Rename,
    Stats,
}

impl UI {
    /// Composite the context-sensitive glyph legend over the live frame as a
    /// centred, rounded [`Block`] pop-up (its context is the box title), drawn last
    /// so the screen behind keeps animating. Shared by every screen's `l` overlay
    /// and by `--plain --legend`.
    pub fn render_legend_band(frame: &mut Frame, legend: Legend) {
        render_popup_box(
            frame,
            legend_title(legend),
            legend_band_lines(legend),
            Backdrop::Float,
            None,
        );
    }
}

/// Worst-case display width of a legend symbol: every non-ASCII glyph is counted
/// as two cells. The symbols are box-drawing / geometric glyphs whose rendered
/// width is terminal-dependent (one cell in many terminals, two in others), so
/// assuming the wider case keeps the description column from ever overlapping
/// the symbol — see [`legend_desc_col`].
fn legend_symbol_width(symbol: &str) -> usize {
    symbol
        .chars()
        .map(|c| if c.is_ascii() { 1 } else { 2 })
        .sum()
}

/// The column (0-based) at which every legend description should start: past a
/// two-space indent, the widest symbol, and a two-space gap. `reserve` is an
/// extra minimum width for a non-symbol row drawn separately (e.g. the zebra
/// swatch) so its description lines up too.
fn legend_desc_col(rows: &[(Option<Color>, &str, &str)], reserve: usize) -> u16 {
    let widest = rows
        .iter()
        .map(|(_, sym, _)| legend_symbol_width(sym))
        .max()
        .unwrap_or(0)
        .max(reserve);
    (2 + widest + 2) as u16
}

/// One legend row as a styled [`Line`]: a two-space indent, the `symbol` (in
/// `color`, else default), then the description starting at absolute column
/// `desc_col`. The gap is filled with spaces sized to the symbol's *rendered*
/// display width, so the description lines up. An all-empty row is a blank
/// separator.
fn legend_row_line(color: Option<Color>, symbol: &str, desc: &str, desc_col: u16) -> Line<'static> {
    use unicode_width::UnicodeWidthStr;
    if symbol.is_empty() && desc.is_empty() {
        return Line::default();
    }
    let mut spans: Vec<Span> = vec![Span::raw("  ")];
    match color {
        Some(c) => spans.push(Span::styled(symbol.to_string(), Style::default().fg(c))),
        None => spans.push(Span::raw(symbol.to_string())),
    }
    // Pad from the current column (2 + rendered symbol width) to `desc_col`.
    let used = 2 + symbol.width();
    let pad = (desc_col as usize).saturating_sub(used).max(1);
    spans.push(Span::raw(" ".repeat(pad)));
    spans.push(Span::raw(desc.to_string()));
    Line::from(spans)
}

/// The legend pop-up's box title, one per screen.
fn legend_title(legend: Legend) -> &'static str {
    match legend {
        Legend::Tree => "Legend — checkpoint tree",
        Legend::Detail => "Legend — tensor details",
        Legend::Heatmap => "Legend — heatmap",
        Legend::Values => "Legend — numeric values",
        Legend::Rename => "Legend — rename tensors in place",
        Legend::Stats => "Legend — checkpoint stats",
    }
}

/// The legend pop-up's body rows (the framing title comes from [`legend_title`]).
fn legend_band_lines(legend: Legend) -> Vec<Line<'static>> {
    let mut lines: Vec<Line> = Vec::new();

    match legend {
        Legend::Tree => {
            let size_example = format!("A {SIZE_ARROW} B");
            let codec_example = format!("{COMPRESSED_MARK} lz4");
            let rows = [
                (
                    Some(palette::ACCENT),
                    "▾ ▸",
                    "a group, expanded / collapsed (Enter or Space toggles it)",
                ),
                (Some(palette::DIM), "·", "a tensor (a stored array)"),
                (
                    Some(palette::UNINDEXED),
                    UNINDEXED_MARK,
                    "an extra tensor on disk but not listed in the index (model.safetensors.index.json)",
                ),
                (
                    Some(palette::META),
                    "†",
                    "a metadata entry (shown beside its tensor, or in the Metadata group)",
                ),
                (
                    None,
                    "≡ N",
                    "number of layers (numbered sub-groups) in the group",
                ),
                (None, "▦ N", "number of tensors in the group / checkpoint"),
                (
                    None,
                    size_example.as_str(),
                    "logical size → on-disk size (shown only when they differ)",
                ),
                (
                    Some(palette::DIM),
                    codec_example.as_str(),
                    "compressed on disk; the codec is named after the glyph",
                ),
                (
                    Some(palette::DIM),
                    UNCOMPRESSED_TAG,
                    "stored uncompressed on disk",
                ),
                (None, "", ""),
                (
                    Some(palette::DTYPE),
                    "I16",
                    "the tensor's data type is tinted (warm amber)",
                ),
                (
                    None,
                    "▪ ▸",
                    "status bar: a single source file / a directory of shards",
                ),
            ];
            let col = legend_desc_col(&rows, 0);
            for (color, sym, desc) in rows {
                lines.push(legend_row_line(color, sym, desc, col));
            }
        }
        Legend::Detail => {
            let codec_example = format!("{COMPRESSED_MARK} lz4");
            let rows = [
                (
                    Some(palette::DIM),
                    codec_example.as_str(),
                    "on-disk compression codec; the N× beside it is the ratio (logical ÷ stored)",
                ),
                (
                    Some(palette::KEY),
                    "as",
                    "the active dtype reinterpretation (press d), e.g. 'BF16 as u4'",
                ),
                (
                    None,
                    "A – B",
                    "a byte range within the file (the tensor's data offsets)",
                ),
                (Some(palette::DIM), "·", "separates fields on a line"),
                (
                    Some(palette::UNINDEXED),
                    UNINDEXED_MARK,
                    "this tensor is an extra: on disk but not listed in the index (model.safetensors.index.json)",
                ),
                (
                    Some(palette::KEY),
                    "⠋",
                    "a statistics scan is running (press s to start; any key cancels)",
                ),
            ];
            let col = legend_desc_col(&rows, 0);
            for (color, sym, desc) in rows {
                lines.push(legend_row_line(color, sym, desc, col));
            }
            lines.push(legend_row_line(None, "", "", col));
            lines.push(Line::from(dim_span(
                "  Statistics:  zeros = fraction of exactly-zero values · non-finite = count of NaN/∞",
            )));
        }
        Legend::Heatmap => {
            let rows = [
                (
                    None,
                    "▀",
                    "one cell packs two data rows: its top half is the upper row, its lower half the next",
                ),
                (
                    None,
                    "A → B",
                    "the stored dtype/shape → the sampled grid size and value range",
                ),
            ];
            let col = legend_desc_col(&rows, 0);
            for (color, sym, desc) in rows {
                lines.push(legend_row_line(color, sym, desc, col));
            }
            // The actual colour ramp, so the scale is unambiguous.
            let mut ramp: Vec<Span> = vec![Span::raw("  "), dim_span("low ")];
            for i in 0..24 {
                ramp.push(Span::styled(
                    "█",
                    Style::default().fg(heat_color(i as f64 / 23.0)),
                ));
            }
            ramp.push(dim_span(" high"));
            ramp.push(Span::raw(
                "   colour scale: cool = low value, warm = high value",
            ));
            lines.push(Line::from(ramp));
        }
        Legend::Values => {
            let rows = [
                (
                    Some(palette::DIM),
                    "12  34",
                    "row / column indices into the full tensor (dimmed), not data values",
                ),
                (
                    Some(palette::DIM),
                    "⋯",
                    "columns were skipped here (the gap between the first and last columns)",
                ),
                (Some(palette::DIM), "⋮", "rows were skipped here"),
                (
                    Some(palette::DIM),
                    "⋱",
                    "both rows and columns were skipped (the corner)",
                ),
                (
                    None,
                    "1.2e-3",
                    "floats use scientific notation; integers print plain",
                ),
                (
                    None,
                    "3f800000",
                    "press b to cycle the base: dec / hex / oct / bin (raw stored bits)",
                ),
            ];
            // Reserve room for the wider zebra swatch row drawn below.
            let col = legend_desc_col(&rows, 8);
            for (color, sym, desc) in rows {
                lines.push(legend_row_line(color, sym, desc, col));
            }
            // A live zebra swatch, since it is a background cue, not a glyph.
            let mut swatch: Vec<Span> = vec![
                Span::raw("  "),
                Span::styled(" 12 ", Style::default().bg(palette::STRIPE_DARK)),
                Span::styled(" 34 ", Style::default().bg(palette::STRIPE_LITE)),
            ];
            // Pad to the description column (the swatch is 2 + 8 = 10 cells wide).
            let pad = (col as usize).saturating_sub(2 + 8).max(1);
            swatch.push(Span::raw(" ".repeat(pad)));
            swatch.push(Span::raw(
                "zebra striping traces a row or column (cycle rows/cols/off with z)",
            ));
            lines.push(Line::from(swatch));
        }
        Legend::Rename => {
            let rows = [
                (
                    Some(palette::ACCENT),
                    "{layer}",
                    "a numbered wildcard — matches any number and copies it into the new name (Tab inserts one)",
                ),
                (
                    None,
                    "12",
                    "a literal number matches only itself — so `…layers.0.…` renames just layer 0",
                ),
                (
                    Some(palette::SUCCESS),
                    "✓",
                    "the rule applies cleanly in place (the header fits the reserved space)",
                ),
                (
                    Some(palette::WARN),
                    "✗ won't fit",
                    "the rewritten header is larger than the shard's reserved space — shorten the new name",
                ),
                (
                    Some(palette::ERROR),
                    "⚠ collide",
                    "two tensors would end up with the same name — the rename is blocked",
                ),
                (
                    Some(palette::ACCENT),
                    "name",
                    "an underlined name is a link: a tensor opens the tree, a shard opens its byte-layout map",
                ),
            ];
            let col = legend_desc_col(&rows, 0);
            for (color, sym, desc) in rows {
                lines.push(legend_row_line(color, sym, desc, col));
            }
            lines.push(legend_row_line(None, "", "", col));
            lines.push(Line::from(dim_span(
                "  Space/: palette · Tab complete · ↑↓ fields · ↵ next field · ←→ caret · ^N add · ^D remove · ^R apply · ^S copy screen · ^Y copy cmd · ^A apply cmd",
            )));
        }
        Legend::Stats => {
            let rows = [
                (
                    Some(palette::ACCENT),
                    crate::stats::GLYPH_FILES,
                    "Files — one row per shard / file on disk",
                ),
                (
                    Some(palette::ACCENT),
                    crate::stats::GLYPH_TENSORS,
                    "Tensors — every stored array in the checkpoint",
                ),
                (
                    Some(palette::ACCENT),
                    crate::stats::GLYPH_LAYERS,
                    "Layers — the repeated transformer-block stack",
                ),
                (
                    Some(palette::ACCENT),
                    crate::stats::GLYPH_EXPERTS,
                    "Experts — MoE expert tensors, split by projection (down / gate / up)",
                ),
                (
                    None,
                    "each · total",
                    "a per-layer (or per-expert) average beside the whole-checkpoint total",
                ),
                (
                    Some(palette::ACCENT),
                    "▁▄█",
                    "per-layer sparkline: each cell a layer, low → high across the min–max range (a uniform metric shows a note instead)",
                ),
                (
                    Some(palette::ACCENT),
                    "█",
                    "composition bar: attention weights",
                ),
                (
                    Some(palette::DTYPE),
                    "▓",
                    "composition bar: MLP / expert (FFN) weights",
                ),
                (
                    Some(palette::META),
                    "░",
                    "composition bar: everything else (norms, router, rotary, …)",
                ),
            ];
            let col = legend_desc_col(&rows, 0);
            for (color, sym, desc) in rows {
                lines.push(legend_row_line(color, sym, desc, col));
            }
        }
    }

    // Common to every screen: the persistent bottom status-line badges. The rename
    // editor draws its own footer (not the status bar), so skip them there.
    if matches!(legend, Legend::Rename) {
        return lines;
    }
    lines.push(Line::default());
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            " read-only ",
            Style::default()
                .bg(palette::STATUS_BG)
                .fg(palette::SUCCESS)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(
            "      browsing never modifies the checkpoint — repack / convert write a new file",
        ),
    ]));
    lines.push(Line::from(vec![
        Span::raw("  "),
        Span::styled(
            " metadata-only ",
            Style::default()
                .bg(palette::STATUS_BG)
                .fg(palette::WARN)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(
            "  a remote source: only header metadata is loaded; data views need the file locally",
        ),
    ]));
    // The `⚠ health` alert badge only appears on the tree's status line, so it's
    // documented there: orange when the checkpoint has warnings only, red for a
    // real error. Aligned with the badges above.
    if matches!(legend, Legend::Tree) {
        use unicode_width::UnicodeWidthStr;
        let desc_col = 2 + METADATA_BADGE.width() + 2;
        let health = |bg: Color, desc: &str| -> Line<'static> {
            let pad = desc_col.saturating_sub(2 + HEALTH_BADGE.width()).max(1);
            Line::from(vec![
                Span::raw("  "),
                Span::styled(
                    HEALTH_BADGE.to_string(),
                    Style::default()
                        .bg(bg)
                        .fg(palette::STATUS_FG)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" ".repeat(pad)),
                Span::raw(desc.to_string()),
            ])
        };
        lines.push(health(
            palette::WARN_BG,
            "health: warnings only (e.g. files on disk the index doesn't list)",
        ));
        lines.push(health(
            palette::ALERT,
            "health: an error — a referenced file or tensor is missing on disk",
        ));
    }

    lines.push(Line::default());
    lines.push(Line::from(dim_span("Click or press any key to close.")));
    lines
}
