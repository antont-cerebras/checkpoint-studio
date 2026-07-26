//! String and [`Line`] plumbing shared by every screen: wrapping, truncation,
//! number and duration formatting, and the input box.

use std::time::Duration;

use ratatui::style::Style;
use ratatui::text::{Line, Span};

use crate::sample::ViewDtype;
use crate::utils::format_shape;

use super::palette;
use super::theme::{dim_span, key_span};

/// Greedy word-wrap of a short help string into lines no wider than `width`.
pub(super) fn wrap_help(text: &str, width: usize) -> Vec<Line<'static>> {
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        if cur.is_empty() {
            // First word on the line: nothing to separate it from.
        } else if cur.chars().count() + 1 + word.chars().count() <= width {
            cur.push(' ');
        } else {
            // Doesn't fit — start a new line with this word.
            lines.push(Line::from(std::mem::take(&mut cur)));
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        lines.push(Line::from(cur));
    }
    lines
}

/// The input box as styled spans — the Ratatui port of [`input_box`]: a padded,
/// input-coloured field with the caret drawn as an inverted character (or a block
/// at the end), padded to at least `min_chars`.
pub(super) fn input_box_spans(text: &str, cursor: usize, min_chars: usize) -> Vec<Span<'static>> {
    let field = Style::default().fg(palette::INPUT_FG).bg(palette::INPUT_BG);
    let caret = Style::default().fg(palette::INPUT_BG).bg(palette::INPUT_FG);
    let chars: Vec<char> = text.chars().collect();
    let cursor = cursor.min(chars.len());
    let mut spans: Vec<Span> = vec![Span::styled(" ", field)];
    for (i, ch) in chars.iter().enumerate() {
        let style = if i == cursor { caret } else { field };
        spans.push(Span::styled(ch.to_string(), style));
    }
    if cursor >= chars.len() {
        spans.push(Span::styled("█", field));
    }
    if chars.len() < min_chars {
        spans.push(Span::styled(" ".repeat(min_chars - chars.len()), field));
    }
    spans.push(Span::styled(" ", field));
    spans
}

/// Human-readable scan duration: milliseconds under a second, else seconds.
pub(super) fn fmt_duration(d: Duration) -> String {
    let ms = d.as_millis();
    if ms >= 1000 {
        format!("{:.1}s", d.as_secs_f64())
    } else {
        format!("{ms}ms")
    }
}

/// Format a heatmap legend / range value: integers without a fractional part,
/// floats with four decimals.
pub(super) fn fmt_value(v: f64, integer: bool) -> String {
    if integer {
        format!("{v:.0}")
    } else {
        format!("{v:.4}")
    }
}

/// Drop the leading `·`/unindexed bullet from a depth-0 tensor row: span 0 is the
/// (empty) indent and span 1 is the bullet, so remove it and trim the space that
/// prefixes the name in the following span, leaving the coloured fields intact.
pub(super) fn without_bullet(line: Line<'static>) -> Line<'static> {
    let mut spans = line.spans;
    if spans.len() >= 2 {
        spans.remove(1);
        if let Some(next) = spans.get_mut(1) {
            next.content = next.content.trim_start().to_string().into();
        }
    }
    Line::from(spans)
}

/// The plain text of a styled line (its span contents concatenated).
pub(super) fn line_to_text(line: &Line<'static>) -> String {
    line.spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
}

/// Truncate `s` to at most `width` characters, keeping the END (so a path's
/// file name stays visible) and prefixing `…` when truncated.
pub(super) fn truncate_keep_end(s: &str, width: usize) -> String {
    let count = s.chars().count();
    if count <= width {
        return s.to_string();
    }
    if width == 0 {
        return String::new();
    }
    let tail: String = s.chars().skip(count - (width - 1)).collect();
    format!("…{tail}")
}

/// Format an integer with thousands separators (e.g. 579133440 -> "579,133,440").
pub(super) fn with_thousands(n: usize) -> String {
    let digits = n.to_string();
    let len = digits.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// A horizontal bar `width` cells wide filled to `frac` of `[0, 1]`. Uses the
/// lower three-quarters block `▆` (rather than a full `█`) so its top sits below
/// the cell ceiling, leaving a thin gap between stacked bars; any non-zero bar
/// shows at least one cell so tiny bins stay visible.
pub(super) fn bar(frac: f64, width: usize) -> String {
    let frac = frac.clamp(0.0, 1.0);
    let mut cells = (frac * width as f64).round() as usize;
    if frac > 0.0 {
        cells = cells.max(1);
    }
    let cells = cells.min(width);
    if cells == 0 {
        // An empty (zero-count) bin still occupies the one-cell baseline so its
        // count lines up with the smallest non-zero bars rather than shifting
        // a column to the left.
        " ".to_string()
    } else {
        "▆".repeat(cells)
    }
}

/// Compact label for a range-histogram bin's lower edge.
pub(super) fn fmt_hist_edge(x: f64) -> String {
    if x == 0.0 {
        "0".to_string()
    } else if x.abs() >= 1e5 || x.abs() < 1e-3 {
        format!("{x:.2e}")
    } else {
        format!("{x:.4}")
    }
}

/// The `stored as VIEW` dtype span(s): the stored dtype plain, or — when a view
/// reinterpretation is active — a dimmed `stored as` followed by the bold view label.
///
/// One definition for the detail screen and both data views: they show the same fact in the
/// same place on screens the user steps between with one key, so any difference between them
/// would read as a bug.
pub(super) fn view_dtype_spans(
    stored: &str,
    view: ViewDtype,
    unpacked_label: Option<&str>,
) -> Vec<Span<'static>> {
    let label: Option<String> = match (view, unpacked_label) {
        (ViewDtype::Unpacked, Some(l)) => Some(format!("{l} (unpacked)")),
        _ => view.label().map(str::to_string),
    };
    label.map_or_else(
        || vec![Span::raw(stored.to_string())],
        |label| vec![dim_span(format!("{stored} as ")), key_span(label)],
    )
}

/// The shape span(s) beside the dtype: the shape plain, or a dimmed `stored as` followed by
/// the bold reinterpreted shape. Shared for the same reason as [`view_dtype_spans`].
pub(super) fn view_shape_spans(stored: &[usize], logical: &[usize]) -> Vec<Span<'static>> {
    if stored == logical {
        vec![Span::raw(format_shape(logical))]
    } else {
        vec![
            dim_span(format!("{} as ", format_shape(stored))),
            key_span(format_shape(logical)),
        ]
    }
}
