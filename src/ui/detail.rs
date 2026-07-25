//! The single-tensor detail screen: its field lines, the stats summary and the
//! progress gauge shown while a scan is running.

use std::time::Duration;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{LineGauge, Paragraph, Widget};

use crate::sample::{Histogram, PackingSchema, Stats, ViewDtype};
use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo};
use crate::utils::{format_parameters, format_percent, format_shape, format_size};

use super::data::render_histogram;
use super::hints::{ChipHit, Seg, chip_regions, close_button, hint_key, wrap_hint_items};
use super::json::highlight_json_lines;
use super::palette;
use super::text::{fmt_duration, fmt_value, with_thousands};
use super::theme::{COMPRESSED_MARK, UNCOMPRESSED_TAG, UNINDEXED_MARK, dim_span, key_span};
use super::{ChipRegions, Link, LinkRegions, Overlay, ScanProgress, StatsView, UI};

impl UI {
    /// Render the tensor detail screen. `view` is the active dtype reinterpretation
    /// (which changes the shown dtype, shape and parameter count); `overridable`
    /// gates the `d`/`r` hints. `histogram` adds the value-histogram section below
    /// the header. A pop-up `overlay` (legend / copied command) composites last.
    ///
    /// Header fields are one [`Line`] each (clipped, not wrapped); when a
    /// histogram is present the header pins to the top, the histogram fills the
    /// middle (sized to `h - header - footer - 1`), one blank row separates it from
    /// the footer pinned to the bottom — filling the screen exactly with no scroll.
    /// Without a histogram the header is immediately followed by the footer,
    /// top-aligned.
    #[allow(clippy::too_many_arguments)] // a screen renderer; the params are all distinct
    pub fn render_detail(
        frame: &mut Frame,
        tensor: &TensorInfo,
        shape: &[usize],
        view: ViewDtype,
        overridable: bool,
        unindexed: bool,
        stats: StatsView,
        histogram: Option<&Histogram>,
        hist_scanning: Option<ScanProgress>,
        schema: Option<&PackingSchema>,
        overlay: Option<&Overlay>,
    ) -> (ChipRegions, LinkRegions) {
        let area = frame.area();
        let (width, height) = (area.width, area.height);

        let (header, stats_gauge_row, links) =
            detail_field_lines(tensor, shape, view, unindexed, stats, schema, width);
        let remote = crate::remote::is_remote_source(&tensor.source_path);
        // The `Tab` → file-layout hint shows only for a local `.safetensors` shard
        // (the only source with a byte-layout map).
        let layout = !remote
            && std::path::Path::new(&tensor.source_path)
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
        let (footer, chips) = detail_footer_lines(overridable, remote, layout, width);
        let header_len = header.len();
        let footer_len = footer.len();

        // Header at the top; the footer is pinned to the **bottom** (above the remote
        // metadata-only banner), with any histogram filling the space between — the
        // same bottom-pinned footer every other view has. `footer_top` is the footer's
        // first screen row, so chip lines can be made absolute for hit-testing.
        let banner = usize::from(remote);
        let footer_top = (height as usize).saturating_sub(footer_len + banner) as u16;
        Paragraph::new(header).render(
            Rect {
                x: 0,
                y: 0,
                width,
                height: header_len as u16,
            },
            frame.buffer_mut(),
        );
        if let Some(hist) = histogram {
            // The histogram fills between the header and the footer (a blank spacer
            // row above the footer), so the screen fills exactly with no scroll.
            let section = (footer_top as usize).saturating_sub(header_len + 1).max(1);
            render_histogram(
                frame,
                Rect {
                    x: 0,
                    y: header_len as u16,
                    width,
                    height: section as u16,
                },
                hist,
                hist_scanning,
            );
        }
        Paragraph::new(footer).render(
            Rect {
                x: 0,
                y: footer_top,
                width,
                height: footer_len as u16,
            },
            frame.buffer_mut(),
        );

        // Metadata-only banner on the bottom row (remote `--ssh-proxy`) — the lower
        // part of the detail screen is otherwise blank, so it doesn't overlap.
        if crate::remote::is_remote_source(&tensor.source_path) {
            Paragraph::new(Line::from(Span::styled(
                " metadata-only — data views need the file locally ",
                Style::default()
                    .bg(palette::STATUS_BG)
                    .fg(palette::WARN)
                    .add_modifier(Modifier::BOLD),
            )))
            .render(
                Rect {
                    x: 0,
                    y: height.saturating_sub(1),
                    width,
                    height: 1,
                },
                frame.buffer_mut(),
            );
        }

        // The header rows sit at `y = index` in both layouts, so overlay the stats
        // progress bar (native LineGauge) on its reserved row.
        if let (Some(row), Some((ratio, label))) = (stats_gauge_row, computing_gauge(stats)) {
            render_line_gauge(
                frame,
                Rect {
                    x: 0,
                    y: row as u16,
                    width,
                    height: 1,
                },
                label,
                ratio,
                Some(30),
            );
        }

        // Clickable regions: each footer chip (made absolute via the footer's
        // start row) plus the top-right `[×]` (→ step back, like `⌫`).
        let mut regions = chip_regions(&chips, footer_top);
        regions.extend(close_button(
            frame,
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
        ));

        // A pop-up overlay composites last, over the live frame, so the detail
        // (including a running scan's progress) keeps animating behind it.
        match overlay {
            Some(Overlay::Legend(l)) => Self::render_legend_band(frame, *l),
            Some(Overlay::Command(c)) => Self::render_command_band(frame, c),
            Some(Overlay::Notice(m)) => Self::render_notice_box(frame, m),
            None => {}
        }
        (regions, links)
    }

    /// The Ratatui port of [`Self::draw_metadata_detail`]: the Key/Type/Value
    /// header, then the value — pretty, syntax-highlighted JSON converted from its
    /// ANSI form via `ansi-to-tui` (so the same `colored_json` palette shows
    /// through), or the raw text lines for a non-JSON value — with the same
    /// line-budget elision and footer.
    pub fn render_metadata_detail(frame: &mut Frame, metadata: &MetadataInfo) {
        let area = frame.area();
        let rows = area.height as usize;

        let mut lines: Vec<Line> = vec![
            Line::from(Span::styled(
                "Metadata Details",
                Style::default().fg(palette::ACCENT),
            )),
            Line::from(dim_span("================")),
            Line::from(vec![dim_span("Key: "), Span::raw(metadata.name.clone())]),
            Line::from(vec![
                dim_span("Type: "),
                Span::raw(metadata.value_type.clone()),
            ]),
            Line::from(dim_span("Value:")),
        ];

        // A JSON object/array is highlighted (via `colored_json`'s ANSI, parsed
        // back into styled spans); everything else falls back to plain text lines.
        let value_lines: Vec<Line> = highlight_json_lines(&metadata.value).unwrap_or_else(|| {
            metadata
                .value
                .lines()
                .map(|l| Line::from(l.to_string()))
                .collect()
        });

        // Show as many value lines as fit (header above + a short footer below),
        // noting how many were elided rather than cutting silently.
        let budget = rows.saturating_sub(8).max(1);
        let shown = value_lines.len().min(budget);
        for line in value_lines.iter().take(shown) {
            let mut indented = vec![Span::raw("  ")];
            indented.extend(line.spans.iter().cloned());
            lines.push(Line::from(indented));
        }
        if value_lines.len() > shown {
            lines.push(Line::from(dim_span(format!(
                "  … ({} more lines)",
                value_lines.len() - shown
            ))));
        }

        lines.push(Line::default());
        lines.push(Line::from(Span::raw("Click or press any key to return...")));
        Paragraph::new(lines).render(area, frame.buffer_mut());
    }
}

/// Build the detail screen's dtype span(s): the stored dtype plain, or — when a
/// view reinterpretation is active — a dimmed `stored as` followed by the bold
/// reinterpretation label. The Ratatui port of [`write_view_dtype`].
fn detail_dtype_spans(
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

/// Build the detail screen's shape span(s): the (unchanged) shape plain, or a
/// dimmed `stored as` followed by the bold reinterpreted shape. Port of
/// [`write_view_shape`].
fn detail_shape_spans(stored: &[usize], logical: &[usize]) -> Vec<Span<'static>> {
    if stored == logical {
        vec![Span::raw(format_shape(logical))]
    } else {
        vec![
            dim_span(format!("{} as ", format_shape(stored))),
            key_span(format_shape(logical)),
        ]
    }
}

/// The one-line statistics summary (mean, std, sparsity, non-finite count) as
/// styled spans — the Ratatui port of [`write_stats_line`]. Field labels dimmed;
/// the non-finite count highlighted (warn) when nonzero.
pub(super) fn detail_stats_summary_spans(s: &Stats) -> Vec<Span<'static>> {
    let mut spans = vec![
        dim_span("mean "),
        Span::raw(format!("{:.4}", s.mean)),
        dim_span(" · std "),
        Span::raw(format!("{:.4}", s.std)),
        dim_span(" · zeros "),
    ];
    // Distinguishing "no zeros" from "a tiny fraction" (which would round to a
    // misleading `0.0%`) is a shared rule — the web shows the same string.
    spans.push(Span::raw(format_percent(s.zero_fraction(), s.zeros == 0)));
    if s.nonfinite > 0 {
        spans.push(Span::styled(
            format!(" · {} non-finite", s.nonfinite),
            Style::default().fg(palette::WARN),
        ));
    }
    spans.push(dim_span(format!("  ({})", fmt_duration(s.elapsed))));
    spans
}

/// The "scan in progress" stats segment as styled spans — Ratatui port of
/// [`write_computing`]: an accent spinner, a dimmed label, a progress bar with a
/// percentage (when the fraction is known), and the running elapsed time.
/// Render a native ratatui [`LineGauge`] into `area`: `label` at the left, then a
/// thick line filled to `ratio` — accent for the done part, dim for the rest. The
/// one progress-bar primitive, shared by the full-screen repack bar and the inline
/// "computing…" statistics line.
/// `max_line` caps the gauge *line* to that many cells (the widget draws the label
/// then the line): the inline "computing…" bar passes `Some(30)` so it doesn't
/// stretch across the whole screen; the full-screen bar passes `None` (full width).
pub(super) fn render_line_gauge(
    frame: &mut Frame,
    area: Rect,
    label: Line<'static>,
    ratio: f64,
    max_line: Option<usize>,
) {
    let area = match max_line {
        // LineGauge lays out `label` then a space then the line, so bound the width
        // to the label plus the wanted line length (clamped to what's available).
        Some(cells) => Rect {
            width: ((label.width() + 1 + cells) as u16).min(area.width),
            ..area
        },
        None => area,
    };
    LineGauge::default()
        .line_set(ratatui::symbols::line::THICK)
        .filled_style(
            Style::default()
                .fg(palette::KEY)
                .add_modifier(Modifier::BOLD),
        )
        .unfilled_style(Style::default().fg(palette::DIM))
        .label(label)
        .ratio(ratio.clamp(0.0, 1.0))
        .render(area, frame.buffer_mut());
}

/// When statistics are computing *with a known fraction*, the `(ratio, label)` for
/// a [`render_line_gauge`] row; otherwise `None` (the caller shows the normal stats
/// text — the spinner-only "computing…", the finished stats, or the "press s" hint).
pub(super) fn computing_gauge(stats: StatsView) -> Option<(f64, Line<'static>)> {
    match stats {
        StatsView::Computing {
            spinner,
            elapsed,
            progress: Some(frac),
        } => {
            let frac = frac.clamp(0.0, 1.0);
            let label = Line::from(format!(
                "{spinner} computing statistics… {:>3.0}% · {} ",
                frac * 100.0,
                fmt_duration(elapsed)
            ));
            Some((frac, label))
        }
        _ => None,
    }
}

pub(super) fn detail_computing_spans(
    spinner: char,
    elapsed: Duration,
    progress: Option<f64>,
) -> Vec<Span<'static>> {
    let mut spans = vec![
        key_span(format!("{spinner} ")),
        dim_span("computing statistics… "),
    ];
    if let Some(frac) = progress {
        const WIDTH: usize = 16;
        let frac = frac.clamp(0.0, 1.0);
        let filled = (frac * WIDTH as f64).round() as usize;
        spans.push(Span::raw("["));
        spans.push(key_span("█".repeat(filled)));
        spans.push(dim_span("░".repeat(WIDTH - filled)));
        spans.push(Span::raw(format!("] {:>3.0}% · ", frac * 100.0)));
    }
    spans.push(Span::raw(fmt_duration(elapsed)));
    spans
}

/// Build the detail screen's header field lines (title + rule, Name, Data Type,
/// Shape, Parameters, optional Packing, Size [+ on-disk/codec], offsets/Chunks,
/// File, optional unindexed flag, blank, Statistics, blank) — one [`Line`] each,
/// clipped (not wrapped) by the caller's `Paragraph`.
fn detail_field_lines(
    tensor: &TensorInfo,
    shape: &[usize],
    view: ViewDtype,
    unindexed: bool,
    stats: StatsView,
    schema: Option<&PackingSchema>,
    width: u16,
) -> (Vec<Line<'static>>, Option<usize>, LinkRegions) {
    let mut lines: Vec<Line> = Vec::new();
    // Link regions in the header (currently just the `File:` path → layout map).
    // The header is rendered at `y = 0`, so a line's index is its screen row.
    let mut links: Vec<(Rect, Link)> = Vec::new();

    lines.push(Line::from(Span::styled(
        "Tensor Details",
        Style::default().fg(palette::ACCENT),
    )));
    lines.push(Line::from(dim_span("─".repeat(width as usize))));
    lines.push(Line::from(vec![
        dim_span("Name: "),
        Span::raw(tensor.name.clone()),
    ]));

    // Data type, with the active reinterpretation highlighted.
    let unpacked_label = schema.map(PackingSchema::label);
    let mut dtype_line = vec![dim_span("Data Type: ")];
    dtype_line.extend(detail_dtype_spans(
        &tensor.dtype,
        view,
        unpacked_label.as_deref(),
    ));
    lines.push(Line::from(dtype_line));

    // Shape and parameter count reflect the overrides.
    let logical = view.logical_shape_with(shape, &tensor.dtype, schema);
    let num_elements: usize = logical.iter().product();
    let mut shape_line = vec![dim_span("Shape: ")];
    shape_line.extend(detail_shape_spans(&tensor.shape, &logical));
    lines.push(Line::from(shape_line));
    lines.push(Line::from(vec![
        dim_span("Parameters: "),
        Span::raw(format!("{} ", format_parameters(num_elements))),
        dim_span(format!("({})", with_thousands(num_elements))),
    ]));

    // Codebook packing schema disclosure (only for tensors that carry one).
    if let Some(s) = schema {
        let widths = s
            .bit_widths()
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        let mode = s
            .quant_mode()
            .map(|m| format!(" · {m}"))
            .unwrap_or_default();
        let uniform = if s.uniform_width().is_some() {
            "uniform"
        } else {
            "non-uniform"
        };
        lines.push(Line::from(vec![
            dim_span("Packing: "),
            Span::raw(format!("{} ", s.label())),
            dim_span(format!(
                "(bit widths [{widths}] · {} experts/word · {uniform}{mode})",
                s.len_p()
            )),
        ]));
    }

    // Size, with on-disk size + codec for formats that track compression.
    let mut size_line = vec![
        dim_span("Size: "),
        Span::raw(format_size(tensor.size_bytes)),
    ];
    match &tensor.storage {
        Storage::Compressed {
            codec,
            stored_bytes,
        } => {
            let ratio = tensor.size_bytes as f64 / (*stored_bytes).max(1) as f64;
            size_line.push(Span::raw(format!(
                " · on disk: {} ",
                format_size(*stored_bytes)
            )));
            size_line.push(dim_span(format!(
                "({COMPRESSED_MARK} {codec}, {ratio:.1}×)"
            )));
        }
        Storage::Raw => {
            size_line.push(Span::raw(format!(
                " · on disk: {} {UNCOMPRESSED_TAG}",
                format_size(tensor.size_bytes)
            )));
        }
        Storage::Unknown => {}
    }
    lines.push(Line::from(size_line));

    // Where the data lives within the file.
    match &tensor.layout {
        Layout::ByteRange { start, end } => {
            lines.push(Line::from(vec![
                dim_span("Data offsets: "),
                Span::raw(format!(
                    "{} – {}  (within file data)",
                    with_thousands(*start as usize),
                    with_thousands(*end as usize)
                )),
            ]));
        }
        Layout::Offset(offset) => {
            lines.push(Line::from(vec![
                dim_span("Data offset: "),
                Span::raw(format!(
                    "{}  (within tensor data)",
                    with_thousands(*offset as usize)
                )),
            ]));
        }
        Layout::Chunked { chunk, num_chunks } => {
            lines.push(Line::from(vec![
                dim_span("Chunks: "),
                Span::raw(format!(
                    "{} × {}",
                    format_shape(chunk),
                    with_thousands(*num_chunks)
                )),
            ]));
        }
        Layout::None => {}
    }

    // Wrap the (possibly long, remote scp-style) path over several lines rather
    // than truncating it, so the whole path stays readable. Continuation lines are
    // indented to line up under the path after the "File: " label.
    let prefix = "File: ";
    let indent = " ".repeat(prefix.len());
    let avail = (width as usize).saturating_sub(prefix.len()).max(1);
    let path_chars: Vec<char> = tensor.source_path.chars().collect();
    // A local `.safetensors` shard's path is a link to its byte-layout map (accent
    // + underline, like the other in-app links); a remote / non-safetensors source
    // has no map, so it stays plain.
    let linkable = !crate::remote::is_remote_source(&tensor.source_path)
        && std::path::Path::new(&tensor.source_path)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
    let path_style = if linkable {
        Style::default()
            .fg(palette::ACCENT)
            .add_modifier(Modifier::UNDERLINED)
    } else {
        Style::default()
    };
    if path_chars.is_empty() {
        lines.push(Line::from(dim_span(prefix)));
    } else {
        // `prefix` and `indent` are the same width, so the path always starts at the
        // same column on every (wrapped) line.
        let x = prefix.len() as u16;
        for (i, chunk) in path_chars.chunks(avail).enumerate() {
            let seg: String = chunk.iter().collect();
            let seg_w = seg.chars().count() as u16;
            if linkable {
                links.push((
                    Rect {
                        x,
                        y: lines.len() as u16,
                        width: seg_w,
                        height: 1,
                    },
                    Link::Layout(tensor.source_path.clone()),
                ));
            }
            if i == 0 {
                lines.push(Line::from(vec![
                    dim_span(prefix),
                    Span::styled(seg, path_style),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::raw(indent.clone()),
                    Span::styled(seg, path_style),
                ]));
            }
        }
    }
    // Flag a tensor that's on disk but absent from the index.
    if unindexed {
        lines.push(Line::from(Span::styled(
            format!("{UNINDEXED_MARK} on disk but not listed in model.safetensors.index.json"),
            Style::default().fg(palette::UNINDEXED),
        )));
    }
    lines.push(Line::default());

    // Exact whole-tensor statistics: shown once computed, else a hint. While a
    // scan reports a fraction, the row is a native progress bar — reserve a blank
    // line here and hand the caller its index to render a `LineGauge` over.
    let stats_gauge_row = if computing_gauge(stats).is_some() {
        let row = lines.len();
        lines.push(Line::default());
        Some(row)
    } else {
        let stats_line: Vec<Span> = match stats {
            StatsView::Ready(s) => {
                let integer = view.is_integer(&tensor.dtype);
                let mut spans = vec![
                    dim_span("Statistics: "),
                    Span::raw(format!(
                        "min {} · max {} · ",
                        fmt_value(s.min, integer),
                        fmt_value(s.max, integer)
                    )),
                ];
                spans.extend(detail_stats_summary_spans(s));
                spans
            }
            // Only the fraction-less "computing…" reaches here (the gauge handles
            // the case with a fraction above).
            StatsView::Computing {
                spinner,
                elapsed,
                progress,
            } => {
                let mut spans = vec![dim_span("Statistics: ")];
                spans.extend(detail_computing_spans(spinner, elapsed, progress));
                spans
            }
            // A remote (`--ssh-proxy`) source has no local bytes to scan, so don't
            // offer the (non-working) `s` hint — say it's metadata-only instead.
            StatsView::Pending if crate::remote::is_remote_source(&tensor.source_path) => vec![
                dim_span("Statistics: "),
                Span::styled(
                    "unavailable — remote source, metadata-only",
                    Style::default().fg(palette::WARN),
                ),
            ],
            StatsView::Pending => vec![
                dim_span("Statistics: press "),
                key_span("s"),
                dim_span(" to scan the full tensor"),
            ],
        };
        lines.push(Line::from(stats_line));
        None
    };
    lines.push(Line::default());

    (lines, stats_gauge_row, links)
}

/// The detail screen's footer hint chips — the same borderless, ` · `-separated
/// `key label` format (and clickable [`ChipHit`]s) every other view uses, via the
/// shared [`wrap_hint_items`]. `overridable` gates `d`/`r`; `layout` gates `Tab`;
/// `remote` (metadata-only) hides `s` (there's nothing local to scan).
pub(crate) fn detail_footer_lines(
    overridable: bool,
    remote: bool,
    layout: bool,
    width: u16,
) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    use KeyCode::{Backspace, Tab};
    let plain = KeyModifiers::NONE;
    let mut items: Vec<(Vec<Seg>, &str)> = vec![
        (vec![Seg::Key("m", hint_key('m'))], "heatmap"),
        (vec![Seg::Key("v", hint_key('v'))], "values"),
        (vec![Seg::Key("h", hint_key('h'))], "histogram"),
        (vec![Seg::Key("b", hint_key('b'))], "bins"),
    ];
    if !remote {
        items.push((vec![Seg::Key("s", hint_key('s'))], "stats"));
    }
    if overridable {
        items.push((vec![Seg::Key("d", hint_key('d'))], "dtype"));
        items.push((vec![Seg::Key("r", hint_key('r'))], "reshape"));
    }
    if layout {
        items.push((
            vec![Seg::Key("Tab", KeyEvent::new(Tab, plain))],
            "file layout",
        ));
    }
    items.push((
        vec![
            Seg::Key("Space", hint_key(' ')),
            Seg::Sep("/"),
            Seg::Key(":", hint_key(':')),
        ],
        "commands",
    ));
    items.push((vec![Seg::Key("l", hint_key('l'))], "legend"));
    items.push((vec![Seg::Key("c", hint_key('c'))], "copy screen"));
    items.push((vec![Seg::Key("y", hint_key('y'))], "copy command"));
    items.push((
        vec![
            Seg::Key("⌫", KeyEvent::new(Backspace, plain)),
            Seg::Sep("/"),
            Seg::Key("\\", hint_key('\\')),
        ],
        "back/fwd",
    ));
    wrap_hint_items(items, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detail_file_path_links_to_the_layout_only_for_local_safetensors() {
        let ti = |source: &str| TensorInfo {
            name: "blk.0.weight".into(),
            dtype: "F32".into(),
            shape: vec![4],
            size_bytes: 16,
            num_elements: 4,
            storage: Storage::Unknown,
            source_path: source.into(),
            layout: Layout::None,
        };
        let links_for = |source: &str| -> LinkRegions {
            let t = ti(source);
            let (_, _, links) = detail_field_lines(
                &t,
                &t.shape,
                ViewDtype::Stored,
                false,
                StatsView::Pending,
                None,
                80,
            );
            links
        };

        // A local `.safetensors` shard's `File:` path links to its layout map.
        let local = links_for("/ckpt/model-00001.safetensors");
        assert_eq!(local.len(), 1, "one File link: {local:?}");
        assert!(
            matches!(&local[0].1, Link::Layout(p) if p == "/ckpt/model-00001.safetensors"),
            "links to the layout: {local:?}"
        );
        // A non-safetensors (or remote) source has no layout map — so no link.
        assert!(links_for("/ckpt/model.gguf").is_empty(), "gguf has no map");
    }
}
