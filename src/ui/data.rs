//! The data views over a tensor's bytes: heatmap, numeric grid and histogram.

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use crate::sample::{HistBins, Histogram, Sample, SampleMode, ViewDtype};
use crate::tree::TensorInfo;
use crate::utils::format_shape;
use crate::viewstate::{NumBase, StripeMode};

use super::detail::{
    computing_gauge, detail_computing_spans, detail_stats_summary_spans, render_line_gauge,
};
use super::hints::{data_view_footer_wrapped_lines, data_view_regions, hint_spans};
use super::palette;
use super::text::{bar, fmt_duration, fmt_hist_edge, fmt_value, truncate_keep_end, with_thousands};
use super::theme::{dim_span, heat_color, key_span};
use super::{ScanProgress, StatsView, UI};

impl UI {
    /// Render a sampled tensor as a heatmap — the Ratatui port of
    /// [`UI::draw_heatmap`]. Each text row shows two data rows via the upper-half
    /// block `▀`: the cell's foreground is the upper data row's heat color, its
    /// background the lower row's. A trailing odd row keeps the default background
    /// for its empty lower half. The title / dtype-shape / slice / range chrome
    /// and the footer match the numeric grid; the layout is top-aligned so a small
    /// sample leaves the lower screen blank, exactly like the raw renderer (which
    /// wrote sequentially and cleared below).
    pub(crate) fn render_heatmap(
        frame: &mut Frame,
        tensor: &TensorInfo,
        sample: &Sample,
        stats: StatsView,
    ) -> Vec<(Rect, KeyEvent)> {
        let area = frame.area();
        let width = area.width as usize;
        let mut lines: Vec<Line> = data_view_title_lines("Heatmap", tensor, width);

        let integer = sample.view.is_integer(&tensor.dtype);
        // The exact whole-tensor range once stats are ready; else the sampled
        // range, flagged as such.
        let (rmin, rmax) = match stats {
            StatsView::Ready(s) => (s.min, s.max),
            _ => (sample.min, sample.max),
        };
        let lo = fmt_value(rmin, integer);
        let hi = fmt_value(rmax, integer);
        let range_note = if matches!(stats, StatsView::Ready(_)) {
            ""
        } else {
            " (sampled)"
        };
        let what = match sample.mode {
            SampleMode::Edges { .. } => "edges",
            SampleMode::Window { .. } => "window",
            SampleMode::Grid => "sampled",
            SampleMode::GridMax => "abs-max",
        };
        let mut dtype_line = view_dtype_spans(
            &tensor.dtype,
            sample.view,
            sample.unpacked.as_ref().map(|u| u.label.as_str()),
        );
        dtype_line.push(Span::raw(" "));
        dtype_line.extend(view_shape_spans(&tensor.shape, &sample.display_shape));
        dtype_line.push(Span::raw(format!(
            " → {what} {}×{}, value range [{lo}, {hi}]{range_note}",
            sample.rows.len(),
            sample.cols.len(),
        )));
        lines.push(Line::from(dtype_line));

        // A computing-with-fraction stats row is a native progress bar: reserve a
        // blank line and render a `LineGauge` over it after the paragraph.
        let stats_gauge_row = if computing_gauge(stats).is_some() {
            let row = lines.len();
            lines.push(Line::default());
            Some(row)
        } else {
            if let Some(stats_line) = data_stats_view_line(stats) {
                lines.push(stats_line);
            }
            None
        };
        if sample.slices > 1 {
            lines.push(slice_header_line(sample));
        }
        lines.push(Line::default());

        let range = rmax - rmin;
        let norm = |v: f64| {
            if range > 0.0 { (v - rmin) / range } else { 0.5 }
        };
        // Two data rows per text line: foreground = the upper row's value,
        // background = the lower row's; a trailing odd row keeps the default bg.
        let mut r = 0;
        while r < sample.values.len() {
            let top = &sample.values[r];
            let bottom = sample.values.get(r + 1);
            let mut spans: Vec<Span> = Vec::with_capacity(top.len());
            for (c, &tv) in top.iter().enumerate() {
                let mut style = Style::default().fg(heat_color(norm(tv)));
                if let Some(below) = bottom {
                    style = style.bg(heat_color(norm(below[c])));
                }
                spans.push(Span::styled("▀", style));
            }
            lines.push(Line::from(spans));
            r += 2;
        }

        lines.push(Line::default());
        let mut legend = vec![Span::raw(format!("{lo} low "))];
        for i in 0..24 {
            legend.push(Span::styled(
                "█",
                Style::default().fg(heat_color(i as f64 / 23.0)),
            ));
        }
        legend.push(Span::raw(format!(" high {hi}")));
        lines.push(Line::from(legend));

        let (footer, chips) = data_view_footer_wrapped_lines(
            sample.mode,
            sample.slices,
            true,
            true,
            StripeMode::Off,
            NumBase::Decimal,
            width,
        );
        // Bottom-pin the footer; the sampled content fills the region above it
        // (clipped if it would overflow), like every other view.
        // Reserve the bottom row for the access badge (drawn by render_data_frame),
        // so the footer's last chip never runs under it. The footer is *wrapped* to the
        // terminal width, so on a narrow pane it can be taller than the pane itself —
        // clamp its height to what's actually there, or the `Paragraph` rect runs past
        // the buffer and Ratatui panics (a 10×10 terminal did exactly that).
        let footer_top = area.height.saturating_sub(footer.len() as u16 + 1);
        let footer_len = (footer.len() as u16).min(area.height.saturating_sub(footer_top));
        Paragraph::new(lines).render(
            Rect {
                x: 0,
                y: 0,
                width: area.width,
                height: footer_top,
            },
            frame.buffer_mut(),
        );
        Paragraph::new(footer).render(
            Rect {
                x: 0,
                y: footer_top,
                width: area.width,
                height: footer_len,
            },
            frame.buffer_mut(),
        );
        if let (Some(row), Some((ratio, label))) = (stats_gauge_row, computing_gauge(stats)) {
            render_line_gauge(
                frame,
                Rect {
                    x: 0,
                    y: row as u16,
                    width: area.width,
                    height: 1,
                },
                label,
                ratio,
                Some(30),
            );
        }
        data_view_regions(frame, &chips, footer_top)
    }

    /// Render a sampled tensor as a grid of numeric values with row/column
    /// indices — the Ratatui port of [`UI::draw_values`]. Same title / dtype-shape
    /// / slice / footer chrome as the heatmap; each value cell is a styled span
    /// (right-aligned, optional zebra-stripe background, dimmed gap markers) built
    /// the same way [`write_grid_cell`] writes one. Top-aligned, like the raw
    /// renderer.
    pub(crate) fn render_values(
        frame: &mut Frame,
        tensor: &TensorInfo,
        sample: &Sample,
        stats: StatsView,
        stripe: StripeMode,
        base: NumBase,
    ) -> Vec<(Rect, KeyEvent)> {
        let area = frame.area();
        let width = area.width as usize;
        // Cell width adapts to the data (same call the sampler uses, so the column
        // count agrees).
        let cw = base.cell_width(sample.view, &tensor.dtype, stats.value_range());

        let mut lines: Vec<Line> = data_view_title_lines("Values", tensor, width);

        let mut dtype_line = view_dtype_spans(
            &tensor.dtype,
            sample.view,
            sample.unpacked.as_ref().map(|u| u.label.as_str()),
        );
        dtype_line.push(Span::raw(" "));
        dtype_line.extend(view_shape_spans(&tensor.shape, &sample.display_shape));
        let edges = matches!(sample.mode, SampleMode::Edges { .. });
        dtype_line.push(Span::raw(match sample.mode {
            SampleMode::Edges { .. } => format!(
                " → edges: {} of {} rows × {} of {} cols (indices shown)",
                edge_desc(&sample.rows, sample.total_rows),
                sample.total_rows,
                edge_desc(&sample.cols, sample.total_cols),
                sample.total_cols
            ),
            SampleMode::Window { .. } => format!(
                " → window: rows {} of {} × cols {} of {} (contiguous)",
                span_desc(&sample.rows),
                sample.total_rows,
                span_desc(&sample.cols),
                sample.total_cols
            ),
            SampleMode::Grid => format!(
                " → sampled {} of {} rows × {} of {} cols (indices shown)",
                sample.rows.len(),
                sample.total_rows,
                sample.cols.len(),
                sample.total_cols
            ),
            SampleMode::GridMax => format!(
                " → abs-max: {} × {} blocks over {} × {} (every element scanned)",
                sample.rows.len(),
                sample.cols.len(),
                sample.total_rows,
                sample.total_cols
            ),
        }));
        lines.push(Line::from(dtype_line));

        // A computing-with-fraction stats row is a native progress bar (see
        // `render_heatmap`).
        let stats_gauge_row = if computing_gauge(stats).is_some() {
            let row = lines.len();
            lines.push(Line::default());
            Some(row)
        } else {
            if let Some(stats_line) = data_stats_view_line(stats) {
                lines.push(stats_line);
            }
            None
        };
        if sample.slices > 1 {
            lines.push(slice_header_line(sample));
        }
        lines.push(Line::default());

        // The index after which rows/cols jump (the padding boundary in edges
        // mode), so the dotted separator can be drawn there.
        let gap = |idx: &[usize]| -> Option<usize> {
            edges
                .then(|| idx.windows(2).position(|w| w[1] != w[0] + 1))
                .flatten()
        };
        let row_gap = gap(&sample.rows);
        let col_gap = gap(&sample.cols);
        let lw = 6usize;
        let dim = Style::default().fg(palette::DIM);

        // Column-index header (with a "⋯" gap column). Wide cells fit the index
        // in a single row; narrow cells stagger labels across two rows.
        let idx_w = sample
            .cols
            .iter()
            .map(|&c| c.to_string().len())
            .max()
            .unwrap_or(1);
        if idx_w >= cw {
            let step = (idx_w + 1).div_ceil(2 * cw).max(1);
            let right_edge = |j: usize| -> usize {
                let gap_cells = matches!(col_gap, Some(g) if j > g) as usize;
                lw + (j + 1 + gap_cells) * cw
            };
            let hwidth = right_edge(sample.cols.len().saturating_sub(1)).max(lw);
            let mut top = vec![' '; hwidth];
            let mut bot = vec![' '; hwidth];
            let mut rank = 0usize;
            for (j, &c) in sample.cols.iter().enumerate() {
                if !j.is_multiple_of(step) {
                    continue;
                }
                let label = c.to_string();
                let end = right_edge(j);
                let start = end.saturating_sub(label.len());
                let buf = if rank.is_multiple_of(2) {
                    &mut top
                } else {
                    &mut bot
                };
                for (k, ch) in label.chars().enumerate() {
                    buf[start + k] = ch;
                }
                rank += 1;
            }
            if let Some(g) = col_gap {
                let pos = right_edge(g) + cw - 1;
                if pos < hwidth {
                    for buf in [&mut top, &mut bot] {
                        if buf[pos] == ' ' {
                            buf[pos] = '⋯';
                        }
                    }
                }
            }
            let top: String = top.into_iter().collect();
            let bot: String = bot.into_iter().collect();
            lines.push(Line::from(Span::styled(top.trim_end().to_string(), dim)));
            lines.push(Line::from(Span::styled(bot.trim_end().to_string(), dim)));
        } else {
            let mut header = String::new();
            header.push_str(&format!("{:>lw$}", ""));
            for (j, &c) in sample.cols.iter().enumerate() {
                header.push_str(&format!("{c:>cw$}"));
                if Some(j) == col_gap {
                    header.push_str(&format!("{:>cw$}", "⋯"));
                }
            }
            lines.push(Line::from(Span::styled(header, dim)));
        }

        let integer = sample.view.is_integer(&tensor.dtype);
        let signed = sample.view.is_signed_integer(&tensor.dtype);
        let band = |k: usize| {
            if k.is_multiple_of(2) {
                palette::STRIPE_DARK
            } else {
                palette::STRIPE_LITE
            }
        };
        for (i, row) in sample.values.iter().enumerate() {
            // Row striping bands the whole line; carried as a per-span background
            // so the index label is included like the raw path's band start.
            let row_bg = (stripe == StripeMode::Rows).then(|| band(i));
            let bg_style = |base: Style| match row_bg {
                Some(c) => base.bg(c),
                None => base,
            };
            let mut spans: Vec<Span> = Vec::new();
            // Dimmed row index.
            spans.push(Span::styled(
                format!("{:>lw$}", sample.rows[i]),
                bg_style(dim),
            ));
            let mut vcol = 0usize;
            for (j, &v) in row.iter().enumerate() {
                let s = match base {
                    // Print integers from their EXACT raw bits, not the decoded f64:
                    // past 2^53 the f64 rounds and `as i64` saturates at 2^63, so a
                    // wide I64/U64 element displayed a wrong number (the hex/oct/bin
                    // bases were always right because they read these same bits).
                    NumBase::Decimal if integer => {
                        let exact = sample
                            .raw
                            .get(i)
                            .and_then(|r| r.get(j))
                            .map(|&rb| crate::sample::format_int_bits(rb, signed));
                        match exact {
                            Some(s) => format!("{s:>cw$}"),
                            None => format!("{:>cw$}", v as i64),
                        }
                    }
                    NumBase::Decimal => format!("{v:>cw$.3e}"),
                    _ => {
                        let rb = sample.raw[i][j];
                        let d = base.digits(rb.width as u32);
                        let body = match base {
                            NumBase::Hex => format!("{:0d$x}", rb.bits),
                            NumBase::Octal => format!("{:0d$o}", rb.bits),
                            NumBase::Binary => format!("{:0d$b}", rb.bits),
                            NumBase::Decimal => unreachable!(),
                        };
                        format!("{body:>cw$}")
                    }
                };
                let col_bg = (stripe == StripeMode::Cols).then(|| band(vcol));
                spans.extend(grid_cell_spans(&s, col_bg, false, row_bg));
                vcol += 1;
                if Some(j) == col_gap {
                    let col_bg = (stripe == StripeMode::Cols).then(|| band(vcol));
                    spans.extend(grid_cell_spans(
                        &format!("{:>cw$}", "⋯"),
                        col_bg,
                        true,
                        row_bg,
                    ));
                    vcol += 1;
                }
            }
            lines.push(Line::from(spans));
            // Dotted row marking the rows skipped after the gap.
            if Some(i) == row_gap {
                let mut s = String::new();
                s.push_str(&format!("{:>lw$}", "⋮"));
                for j in 0..row.len() {
                    s.push_str(&format!("{:>cw$}", "⋮"));
                    if Some(j) == col_gap {
                        s.push_str(&format!("{:>cw$}", "⋱"));
                    }
                }
                lines.push(Line::from(Span::styled(s, dim)));
            }
        }

        let (footer, chips) = data_view_footer_wrapped_lines(
            sample.mode,
            sample.slices,
            sample.overridable,
            false,
            stripe,
            base,
            width,
        );
        // Bottom-pin the footer; the value grid fills the region above it (clipped
        // if it would overflow), like every other view.
        // Reserve the bottom row for the access badge (drawn by render_data_frame),
        // so the footer's last chip never runs under it. The footer is *wrapped* to the
        // terminal width, so on a narrow pane it can be taller than the pane itself —
        // clamp its height to what's actually there, or the `Paragraph` rect runs past
        // the buffer and Ratatui panics (a 10×10 terminal did exactly that).
        let footer_top = area.height.saturating_sub(footer.len() as u16 + 1);
        let footer_len = (footer.len() as u16).min(area.height.saturating_sub(footer_top));
        Paragraph::new(lines).render(
            Rect {
                x: 0,
                y: 0,
                width: area.width,
                height: footer_top,
            },
            frame.buffer_mut(),
        );
        Paragraph::new(footer).render(
            Rect {
                x: 0,
                y: footer_top,
                width: area.width,
                height: footer_len,
            },
            frame.buffer_mut(),
        );
        if let (Some(row), Some((ratio, label))) = (stats_gauge_row, computing_gauge(stats)) {
            render_line_gauge(
                frame,
                Rect {
                    x: 0,
                    y: row as u16,
                    width: area.width,
                    height: 1,
                },
                label,
                ratio,
                Some(30),
            );
        }
        data_view_regions(frame, &chips, footer_top)
    }
}

/// The data-view title block as styled [`Line`]s — the Ratatui port of
/// [`write_data_view_title`]: the view label and tensor name, then a dimmed
/// `File:` and source path, each clipped (tail-kept) to `width` so both stay on
/// screen above a grid of any size. `kind` is the view label (`Values` / `Heatmap`).
fn data_view_title_lines(kind: &str, tensor: &TensorInfo, width: usize) -> Vec<Line<'static>> {
    vec![
        Line::from(vec![
            Span::raw(format!("{kind}: ")),
            Span::raw(truncate_keep_end(
                &tensor.name,
                width.saturating_sub(kind.len() + 2),
            )),
        ]),
        Line::from(vec![
            dim_span("File: "),
            Span::raw(truncate_keep_end(
                &tensor.source_path,
                width.saturating_sub(6),
            )),
        ]),
    ]
}

/// The data-view dtype span(s) — Ratatui port of [`write_view_dtype`]: just the
/// stored dtype, or a dimmed `stored as` + the bold reinterpretation label.
fn view_dtype_spans(
    stored: &str,
    view: ViewDtype,
    unpacked_label: Option<&str>,
) -> Vec<Span<'static>> {
    let label: Option<String> = match (view, unpacked_label) {
        (ViewDtype::Unpacked, Some(l)) => Some(format!("{l} (unpacked)")),
        _ => view.label().map(str::to_string),
    };
    match label {
        Some(label) => vec![dim_span(format!("{stored} as ")), key_span(label)],
        None => vec![Span::raw(stored.to_string())],
    }
}

/// The data-view shape span(s) — Ratatui port of [`write_view_shape`].
fn view_shape_spans(stored: &[usize], logical: &[usize]) -> Vec<Span<'static>> {
    if stored == logical {
        vec![Span::raw(format_shape(logical))]
    } else {
        vec![
            dim_span(format!("{} as ", format_shape(stored))),
            key_span(format_shape(logical)),
        ]
    }
}

/// The one-line statistics view for a data screen as a styled [`Line`] — Ratatui
/// port of [`write_stats_view`]: the finished stats, a spinner while computing,
/// or `None` while pending (the raw path writes nothing then).
fn data_stats_view_line(stats: StatsView) -> Option<Line<'static>> {
    match stats {
        StatsView::Ready(s) => Some(Line::from(detail_stats_summary_spans(s))),
        StatsView::Computing {
            spinner,
            elapsed,
            progress,
        } => Some(Line::from(detail_computing_spans(
            spinner, elapsed, progress,
        ))),
        StatsView::Pending => None,
    }
}

/// The 3D slice-navigation header as a styled [`Line`] — Ratatui port of
/// [`write_slice_header`]. Only used when `sample.slices > 1`.
fn slice_header_line(sample: &Sample) -> Line<'static> {
    let mut spans: Vec<Span> = Vec::new();
    match sample.unpacked.as_ref().map(|u| u.field) {
        Some(f) => spans.push(Span::raw(format!(
            "expert {} of {} — stored word {}, field {}/{} ({}-bit) — ",
            sample.slice, sample.slices, f.stored_slice, f.field, f.len_p, f.field_bits,
        ))),
        None => spans.push(Span::raw(format!(
            "slice {} of {} (fixed leading index) — ",
            sample.slice, sample.slices
        ))),
    }
    let items: &[(&str, &str)] = if matches!(sample.mode, SampleMode::Grid) {
        &[
            ("← →", "step"),
            ("Shift+← →", "jump 5% (both wrap)"),
            ("/", "index or %"),
        ]
    } else {
        &[("[ ]", "step"), ("/", "index or %")]
    };
    spans.extend(hint_spans(items));
    Line::from(spans)
}

/// One numeric-grid cell as styled span(s) — the Ratatui port of
/// [`write_grid_cell`]. `col_bg` is the column-stripe background (which, like the
/// raw path, bands all but the cell's first column so every stripe is the same
/// width and a one-space gutter separates neighbours); `row_bg` is the ambient
/// row-stripe background carried across the whole cell (incl. the gutter), set
/// once by the caller for a row band. `dim` dims the glyphs (the "⋯" gap marker).
fn grid_cell_spans(
    s: &str,
    col_bg: Option<ratatui::style::Color>,
    dim: bool,
    row_bg: Option<ratatui::style::Color>,
) -> Vec<Span<'static>> {
    let base = if dim {
        Style::default().fg(palette::DIM)
    } else {
        Style::default()
    };
    let with_bg = |style: Style, bg: Option<ratatui::style::Color>| match bg {
        Some(c) => style.bg(c),
        None => style,
    };
    match col_bg {
        // Leave the first column an uncoloured gutter (just the row band, if any)
        // and band the rest, so the stripe is the same width for every column.
        Some(c) => {
            let split = s.char_indices().nth(1).map_or(s.len(), |(i, _)| i);
            let (gutter, band) = s.split_at(split);
            vec![
                Span::styled(gutter.to_string(), with_bg(base, row_bg)),
                Span::styled(band.to_string(), with_bg(base, Some(c))),
            ]
        }
        None => vec![Span::styled(s.to_string(), with_bg(base, row_bg))],
    }
}

/// Describe a contiguous window's extent along one axis — e.g. `120–179` for the
/// rows/cols currently shown (the header pairs it with the axis total).
fn span_desc(idx: &[usize]) -> String {
    match (idx.first(), idx.last()) {
        (Some(a), Some(b)) => format!("{a}–{b}"),
        _ => "—".to_string(),
    }
}

/// Describe an edges-view index slice for the header — e.g. `first 26 & last 25`,
/// `last 50`, `first 50`, or `all 50` when the whole axis fits — so the current
/// first/last split (and any bias the user dialed in) is visible at a glance.
fn edge_desc(idx: &[usize], total: usize) -> String {
    let n = idx.len();
    if n >= total {
        return format!("all {n}");
    }
    match idx.windows(2).position(|w| w[1] != w[0] + 1) {
        Some(g) => format!("first {} & last {}", g + 1, n - (g + 1)),
        None if idx.first() == Some(&0) => format!("first {n}"),
        None => format!("last {n}"),
    }
}

/// Render the value histogram into `rect` — the Ratatui port of
/// [`write_histogram_section`]: a heading (value count, any non-finite, the scan
/// indicator), then one bar per bin (label │ bar count (pct)), then a clipped-bin
/// note when they don't all fit. The whole section stays within `rect.height`.
/// Returns the number of rows actually drawn, so the caller can flow the footer
/// right below it (the raw renderer wrote these sequentially, so a small histogram
/// leaves the footer near the top rather than at the screen's bottom).
pub(super) fn render_histogram(
    frame: &mut Frame,
    rect: Rect,
    hist: &Histogram,
    scanning: Option<ScanProgress>,
) -> usize {
    let term_w = rect.width as usize;
    let max_rows = rect.height as usize;
    if max_rows == 0 {
        return 0;
    }
    let mut lines: Vec<Line> = Vec::new();

    // Heading.
    let mut head = vec![
        dim_span("Histogram: "),
        Span::raw(format!("{} values", with_thousands(hist.total as usize))),
    ];
    if hist.nonfinite > 0 {
        head.push(dim_span(format!(
            "  ·  {} non-finite",
            with_thousands(hist.nonfinite as usize)
        )));
    }
    if let Some((spinner, elapsed, progress)) = scanning {
        let mut s = format!("   {spinner} scanning");
        if let Some(p) = progress {
            s.push_str(&format!(" {:.0}%", p * 100.0));
        }
        s.push_str(&format!(" ({:.1}s)", elapsed.as_secs_f64()));
        head.push(Span::styled(s, Style::default().fg(palette::ACCENT)));
    } else if !hist.elapsed.is_zero() {
        head.push(dim_span(format!("  ({})", fmt_duration(hist.elapsed))));
    }
    lines.push(Line::from(head));
    let heading_rows = 1usize;

    let n = hist.counts.len();
    let labels: Vec<String> = (0..n)
        .map(|i| match hist.bins {
            HistBins::IntBins { start, step } => (start + i as i64 * step).to_string(),
            HistBins::Range { lo, hi } => fmt_hist_edge(lo + (hi - lo) * i as f64 / n as f64),
        })
        .collect();
    let label_w = labels.iter().map(|l| l.chars().count()).max().unwrap_or(1);
    let counts: Vec<String> = hist
        .counts
        .iter()
        .map(|c| with_thousands(*c as usize))
        .collect();
    let count_w = counts.iter().map(|s| s.chars().count()).max().unwrap_or(1);
    let max_count = hist.counts.iter().copied().max().unwrap_or(0).max(1);
    let total = hist.total.max(1);
    let pcts: Vec<String> = hist
        .counts
        .iter()
        .map(|&c| {
            let pct = c as f64 / total as f64 * 100.0;
            if c == 0 {
                "0.0%".to_string()
            } else if pct < 0.1 {
                format!("{pct:.1e}%")
            } else {
                format!("{pct:.1}%")
            }
        })
        .collect();
    let pct_w = pcts.iter().map(|s| s.chars().count()).max().unwrap_or(4);

    // The bar gets whatever width is left after `label │ … count (pct)`.
    let fixed = label_w + 3 + 1 + count_w + pct_w + 3;
    let bar_w = term_w.saturating_sub(fixed).clamp(1, 100);
    let bar_rows = max_rows.saturating_sub(heading_rows).max(1);
    let shown = if n <= bar_rows {
        n
    } else {
        bar_rows.saturating_sub(1).max(1)
    };

    let accent = Style::default().fg(palette::ACCENT);
    let bold = Style::default().add_modifier(Modifier::BOLD);
    for i in 0..shown {
        let frac = hist.counts[i] as f64 / max_count as f64;
        lines.push(Line::from(vec![
            Span::raw(format!("{:>label_w$} ", labels[i])),
            dim_span("│"),
            Span::styled(bar(frac, bar_w), accent),
            Span::styled(format!(" {:>count_w$} ", counts[i]), bold),
            dim_span("("),
            Span::raw(pcts[i].clone()),
            dim_span(")"),
        ]));
    }
    if n > shown {
        lines.push(Line::from(dim_span(format!(
            "… {} more bins (enlarge the terminal)",
            n - shown
        ))));
    }

    let used = lines.len().min(max_rows);
    Paragraph::new(lines).render(rect, frame.buffer_mut());
    used
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tests_support::strip_ansi_codes;

    fn tensor(name: &str, shape: &[usize]) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dtype: "F32".into(),
            shape: shape.to_vec(),
            size_bytes: shape.iter().product::<usize>() * 4,
            num_elements: shape.iter().product(),
            storage: crate::tree::Storage::Raw,
            source_path: "/ckpt/model.safetensors".into(),
            layout: crate::tree::Layout::None,
        }
    }

    /// A 3×3 sample with known values, spanning negative to positive so the ramp and
    /// the sign column both have something to show.
    fn sample(mode: SampleMode) -> Sample {
        let values = vec![
            vec![-1.0, 0.0, 1.0],
            vec![0.5, -0.5, 0.25],
            vec![2.0, -2.0, 0.0],
        ];
        let raw = values
            .iter()
            .map(|row| {
                row.iter()
                    .map(|v| crate::sample::RawBits {
                        bits: u64::from((*v as f32).to_bits()),
                        width: 32,
                    })
                    .collect()
            })
            .collect();
        Sample {
            rows: vec![0, 1, 2],
            cols: vec![0, 1, 2],
            values,
            raw,
            min: -2.0,
            max: 2.0,
            total_rows: 3,
            total_cols: 3,
            slices: 1,
            slice: 0,
            view: ViewDtype::Stored,
            overridable: true,
            mode,
            display_shape: vec![3, 3],
            unpacked: None,
        }
    }

    fn stats() -> crate::sample::Stats {
        crate::sample::Stats {
            count: 9,
            min: -2.0,
            max: 2.0,
            mean: 0.0277,
            std: 1.1,
            zeros: 2,
            nonfinite: 0,
            elapsed: std::time::Duration::from_millis(7),
        }
    }

    fn render(w: u16, h: u16, f: impl FnOnce(&mut Frame)) -> String {
        strip_ansi_codes(&crate::tui::headless_render(w, h, f).expect("headless render"))
    }

    #[test]
    fn the_heatmap_titles_itself_with_the_tensor_and_its_shape() {
        let t = tensor("model.layers.0.mlp.down_proj.weight", &[3, 3]);
        let s = sample(SampleMode::Grid);
        let out = render(120, 24, |f| {
            UI::render_heatmap(f, &t, &s, StatsView::Ready(&stats()));
        });
        assert!(out.contains("Heatmap"), "{out}");
        assert!(out.contains("down_proj.weight"), "{out}");
        assert!(out.contains("(3, 3)"), "the shape shows:\n{out}");
    }

    #[test]
    fn the_heatmap_reports_the_range_it_is_scaled_over() {
        let t = tensor("w", &[3, 3]);
        let s = sample(SampleMode::Grid);
        // With finished stats the scale is the exact whole-tensor range…
        let exact = render(120, 24, |f| {
            UI::render_heatmap(f, &t, &s, StatsView::Ready(&stats()));
        });
        assert!(exact.contains('2'), "the range endpoints show:\n{exact}");
        // …and without them, the sample's own range is used instead of nothing.
        let pending = render(120, 24, |f| {
            UI::render_heatmap(f, &t, &s, StatsView::Pending);
        });
        assert!(
            !pending.trim().is_empty(),
            "a pending scan still renders a map"
        );
    }

    #[test]
    fn the_abs_max_mode_is_labelled_so_a_full_scan_is_obvious() {
        let t = tensor("w", &[3, 3]);
        let s = sample(SampleMode::GridMax);
        let out = render(120, 24, |f| {
            UI::render_heatmap(f, &t, &s, StatsView::Ready(&stats()));
        });
        // GridMax reads every element, unlike the sampled grid — the header has to say
        // which one produced the picture.
        assert!(
            out.contains("max") || out.contains("abs"),
            "the abs-max mode must be named:\n{out}"
        );
    }

    #[test]
    fn the_values_grid_prints_cells_in_every_base() {
        let t = tensor("w", &[3, 3]);
        let s = sample(SampleMode::Grid);
        let dec = render(140, 24, |f| {
            UI::render_values(
                f,
                &t,
                &s,
                StatsView::Ready(&stats()),
                StripeMode::Off,
                NumBase::Decimal,
            );
        });
        assert!(dec.contains("Values"), "{dec}");
        assert!(
            dec.contains("-1.0000") || dec.contains("-1"),
            "a decimal cell:\n{dec}"
        );

        let hex = render(140, 24, |f| {
            UI::render_values(
                f,
                &t,
                &s,
                StatsView::Ready(&stats()),
                StripeMode::Off,
                NumBase::Hex,
            );
        });
        // 1.0f32 is 0x3f800000 — the hex view shows stored bits, not the decimal.
        assert!(
            hex.to_lowercase().contains("3f800000"),
            "a hex cell:\n{hex}"
        );

        let bin = render(200, 24, |f| {
            UI::render_values(
                f,
                &t,
                &s,
                StatsView::Ready(&stats()),
                StripeMode::Off,
                NumBase::Binary,
            );
        });
        assert!(bin.contains("00111111"), "a binary cell:\n{bin}");
    }

    #[test]
    fn zebra_striping_changes_the_grid_without_changing_the_numbers() {
        let t = tensor("w", &[3, 3]);
        let s = sample(SampleMode::Grid);
        let plain = render(140, 24, |f| {
            UI::render_values(
                f,
                &t,
                &s,
                StatsView::Ready(&stats()),
                StripeMode::Off,
                NumBase::Decimal,
            );
        });
        let striped = render(140, 24, |f| {
            UI::render_values(
                f,
                &t,
                &s,
                StatsView::Ready(&stats()),
                StripeMode::Rows,
                NumBase::Decimal,
            );
        });
        // Striping is a background colour, so the *cells* must be identical; only the
        // footer differs, since it echoes the current mode.
        let cells = |s: &str| {
            s.lines()
                .filter(|l| l.contains("e0") || l.contains("e-1"))
                .map(str::to_string)
                .collect::<Vec<_>>()
        };
        assert!(
            !cells(&plain).is_empty(),
            "no value rows found in:\n{plain}"
        );
        assert_eq!(
            cells(&plain),
            cells(&striped),
            "striping must not move or reformat a cell"
        );
        assert!(
            plain.contains("zebra: off"),
            "the footer names the mode:\n{plain}"
        );
        assert!(striped.contains("zebra: rows"), "{striped}");
    }

    #[test]
    fn the_grid_says_which_slice_it_is_showing() {
        let t = tensor("w", &[4, 3, 3]);
        let mut s = sample(SampleMode::Grid);
        s.slices = 4;
        s.slice = 2;
        let out = render(140, 24, |f| {
            UI::render_values(
                f,
                &t,
                &s,
                StatsView::Ready(&stats()),
                StripeMode::Off,
                NumBase::Decimal,
            );
        });
        assert!(
            out.contains('2') && out.contains('4'),
            "the slice position shows:\n{out}"
        );
    }

    #[test]
    fn a_scan_in_progress_is_visible_on_both_views() {
        let t = tensor("w", &[3, 3]);
        let s = sample(SampleMode::Grid);
        let scanning = StatsView::Computing {
            spinner: '⠋',
            elapsed: std::time::Duration::from_millis(1500),
            progress: Some(0.5),
        };
        for out in [
            render(120, 24, |f| {
                UI::render_heatmap(f, &t, &s, scanning);
            }),
            render(140, 24, |f| {
                UI::render_values(f, &t, &s, scanning, StripeMode::Off, NumBase::Decimal);
            }),
        ] {
            assert!(
                out.contains("1.5s") || out.contains('⠋'),
                "the scan shows:\n{out}"
            );
        }
    }

    #[test]
    fn both_data_views_survive_a_small_terminal() {
        let t = tensor("w", &[3, 3]);
        let s = sample(SampleMode::Grid);
        // A tiny pane is unusual but reachable, and Ratatui panics on an out-of-bounds
        // write — so "renders something" has to hold at every size.
        for (w, h) in [(10u16, 10u16), (24, 12), (48, 20), (80, 6)] {
            assert!(
                crate::tui::headless_render(w, h, |f| {
                    UI::render_heatmap(f, &t, &s, StatsView::Pending);
                })
                .is_ok(),
                "heatmap panicked at {w}x{h}"
            );
            assert!(
                crate::tui::headless_render(w, h, |f| {
                    UI::render_values(
                        f,
                        &t,
                        &s,
                        StatsView::Pending,
                        StripeMode::Cols,
                        NumBase::Decimal,
                    );
                })
                .is_ok(),
                "values panicked at {w}x{h}"
            );
        }
    }
}
