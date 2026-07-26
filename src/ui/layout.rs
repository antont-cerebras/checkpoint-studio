//! The byte-layout screen: a shard's segments drawn as proportional bands.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use crate::utils::{format_shape, format_size};

use super::hints::{chip_regions, close_button, layout_hint_lines};
use super::palette;
use super::scroll::VScrollbar;
use super::text::truncate_keep_end;
use super::{ChipRegions, Link, LinkRegions, UI};

/// Header rows above the layout map's strip: the title, the size / tensor-count
/// summary, and the separator rule.
const LAYOUT_HEADER_ROWS: usize = 3;

impl UI {
    /// The first terminal row of the layout map's strip (its fixed header height),
    /// for the mouse click-to-select hit-test.
    pub(crate) fn layout_header_rows() -> usize {
        LAYOUT_HEADER_ROWS
    }

    /// Body rows the layout map's vertical strip occupies (total height minus the
    /// 3-row header and the footer hint line(s)).
    pub(crate) fn layout_visible_rows(width: u16, height: u16) -> usize {
        (height as usize)
            .saturating_sub(LAYOUT_HEADER_ROWS + layout_hint_lines(width).0.len())
            .max(1)
    }

    /// Render the safetensors **layout map** — a scrollable vertical strip of the
    /// file: a header (title + size / tensor-count / header-size summary), then one
    /// band per segment (header, each tensor by offset, any padding) whose height
    /// is proportional to its share of the file. Each band's first row carries its
    /// offset and a one-line label (name + dtype/shape + size); the header band's
    /// remaining rows list its `__metadata__` entries tree-like. The `selected`
    /// segment's label row is highlighted. Returns the max scroll offset (so the
    /// caller can clamp) and the clickable footer chips.
    pub(crate) fn render_layout(
        frame: &mut Frame,
        map: &crate::safelayout::LayoutMap,
        selected: usize,
        scroll: usize,
        copied: Option<&str>,
        interactive: bool,
    ) -> (usize, ChipRegions, LinkRegions, Option<VScrollbar>) {
        use crate::safelayout::SegmentKind;
        let area = frame.area();
        let (width, height) = (area.width, area.height);
        if height < (LAYOUT_HEADER_ROWS as u16 + 2) {
            return (0, Vec::new(), Vec::new(), None);
        }
        // A concrete tensor band's name links to that tensor in the tree; filled in
        // as the strip is drawn below.
        let mut links: Vec<(Rect, Link)> = Vec::new();

        // --- header (title, summary, rule) ---
        let dim = Style::default().fg(palette::DIM);
        let mut summary = vec![
            Span::styled(format_size(map.total_len as usize), Style::default()),
            Span::styled(" · ", dim),
            Span::raw(format!("{} tensors", map.tensor_count)),
            Span::styled(" · ", dim),
            Span::raw(format!("header {}", format_size(map.header_len as usize))),
        ];
        if map.metadata_entries() > 0 {
            summary.push(Span::styled(
                format!(" · {} metadata", map.metadata_entries()),
                dim,
            ));
        }
        let header_lines = vec![
            Line::from(Span::raw(format!("Layout - {}", map.name))),
            Line::from(summary),
            Line::from(Span::styled(
                "─".repeat(width as usize),
                Style::default().fg(palette::DIM),
            )),
        ];
        Paragraph::new(header_lines).render(
            crate::ui::fit_rows(area, 0, LAYOUT_HEADER_ROWS as u16),
            frame.buffer_mut(),
        );

        // --- footer hints (pinned to the bottom) ---
        // A copy confirmation temporarily takes over the footer's first line (its
        // own line, cleared full-width) so it never intermingles with the hints.
        let (mut hint_lines, chips) = layout_hint_lines(width);
        let footer_rows = hint_lines.len();
        let body_rows = (height as usize).saturating_sub(LAYOUT_HEADER_ROWS + footer_rows);
        if let Some(msg) = copied
            && let Some(first) = hint_lines.first_mut()
        {
            *first = Line::from(Span::styled(
                msg.to_string(),
                Style::default()
                    .fg(palette::SUCCESS)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        // A narrow terminal wraps the hint chips onto more lines than the terminal has,
        // so both the start row and the height need clamping — `fit_rows` does both.
        Paragraph::new(hint_lines).render(
            crate::ui::fit_rows(
                area,
                (height as usize).saturating_sub(footer_rows) as u16,
                footer_rows as u16,
            ),
            frame.buffer_mut(),
        );

        // --- the vertical strip (scrollable) ---
        let rows = band_rows(map, body_rows);
        let total_rows: usize = rows.iter().sum();
        // The bar is drawn by the engine; reserve its column here so the strip
        // doesn't underlap it (live TUI only), and hand the geometry back to draw.
        let vscroll = interactive
            .then(|| Self::layout_scrollbar(width, total_rows, body_rows, scroll))
            .flatten();
        let strip_width = width.saturating_sub(u16::from(vscroll.is_some()));

        let max_scroll = total_rows.saturating_sub(body_rows);
        let scroll = scroll.min(max_scroll);
        // Cumulative start row of each band.
        let mut starts = Vec::with_capacity(rows.len() + 1);
        let mut acc = 0usize;
        for &h in &rows {
            starts.push(acc);
            acc += h;
        }

        let sel = Style::default()
            .fg(palette::SELECT_FG)
            .bg(palette::SELECT_BG);
        let mut seg = 0usize; // segment whose band contains the current row
        let mut body: Vec<Line> = Vec::with_capacity(body_rows);
        for r in scroll..(scroll + body_rows).min(total_rows) {
            // Advance to the band containing global row `r`.
            while seg + 1 < starts.len() && r >= starts[seg] + rows[seg] {
                seg += 1;
            }
            let s = &map.segments[seg];
            let row_in = r - starts[seg];
            let first = row_in == 0;
            let selected_row = seg == selected && first;
            let rule = if r == 0 {
                '┬'
            } else if r == total_rows - 1 {
                '┴'
            } else {
                '│'
            };
            let (glyph, color) = band_style(s, map.total_len);
            let off = if first {
                format!("{:#014x}", s.start)
            } else {
                " ".repeat(14)
            };
            let mut spans = vec![
                Span::styled(off, dim),
                Span::raw(" "),
                Span::styled(rule.to_string(), dim),
                Span::raw(" "),
                Span::styled(glyph.to_string(), Style::default().fg(color)),
                Span::raw("  "),
            ];
            let label_w = strip_width.saturating_sub(20) as usize;
            if first {
                // One-line label: name (selection-highlighted), then a dim
                // dtype/shape + size, so nothing looks orphaned on a blank row.
                let name = truncate_keep_end(&s.name, label_w.saturating_sub(24));
                // A concrete tensor's name is a link to the tree (underlined, like
                // the other in-app links); the spans above are a fixed 20 columns
                // wide, so the name always starts at column 20.
                let is_tensor = matches!(s.kind, SegmentKind::Tensor { .. });
                let name_style = if selected_row {
                    sel
                } else if is_tensor {
                    Style::default()
                        .fg(color)
                        .add_modifier(Modifier::UNDERLINED)
                } else {
                    Style::default().fg(color)
                };
                if is_tensor {
                    links.push((
                        Rect {
                            x: 20,
                            y: LAYOUT_HEADER_ROWS as u16 + (r - scroll) as u16,
                            width: name.chars().count() as u16,
                            height: 1,
                        },
                        Link::Tree(s.name.clone()),
                    ));
                }
                spans.push(Span::styled(name, name_style));
                let mut detail = String::new();
                if let SegmentKind::Tensor { dtype, shape } = &s.kind {
                    detail.push_str(&format!("  {dtype}"));
                    if !shape.is_empty() {
                        detail.push_str(&format!(" {}", format_shape(shape)));
                    }
                }
                detail.push_str(&format!("  {}", format_size(s.len() as usize)));
                spans.push(Span::styled(detail, if selected_row { sel } else { dim }));
            } else if s.kind == SegmentKind::Header {
                // The header band's rows list its `__metadata__` entries tree-like.
                if let Some((k, v)) = map.metadata.get(row_in - 1) {
                    let val = truncate_keep_end(v, label_w.saturating_sub(k.len() + 6));
                    spans.push(Span::styled(
                        format!("† {k}  "),
                        Style::default().fg(palette::META),
                    ));
                    spans.push(Span::styled(val, dim));
                }
            }
            body.push(Line::from(spans));
        }
        Paragraph::new(body).render(
            Rect {
                width: strip_width,
                ..crate::ui::fit_rows(area, LAYOUT_HEADER_ROWS as u16, body_rows as u16)
            },
            frame.buffer_mut(),
        );

        // (The scroll bar itself is drawn by the engine — see `render_vscrollbar`.)

        // Clickable footer chips (hints start at the footer's top row) + `[×]`
        // (→ back to the tensor tree, like the file view's close).
        // saturating for the same reason as the layout footer above.
        let footer_top = (height as usize).saturating_sub(footer_rows) as u16;
        let mut regions = chip_regions(&chips, footer_top);
        regions.extend(close_button(
            frame,
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
        ));
        (max_scroll, regions, links, vscroll)
    }

    /// The layout map's vertical scrollbar: a `total`-row band strip showing
    /// `visible` rows from `offset`, just below the layout header.
    fn layout_scrollbar(
        width: u16,
        total: usize,
        visible: usize,
        offset: usize,
    ) -> Option<VScrollbar> {
        VScrollbar::for_body(
            Rect {
                x: 0,
                y: LAYOUT_HEADER_ROWS as u16,
                width,
                height: visible as u16,
            },
            total,
            offset,
        )
    }

    /// The cumulative start row of each layout-map band, plus the total row count
    /// as a trailing entry (so band `i` spans `[starts[i], starts[i+1])`). Lets the
    /// browsing loop map a click to a segment and snap the scroll to the selection,
    /// using the same band heights [`Self::render_layout`] draws.
    pub(crate) fn layout_band_starts(
        map: &crate::safelayout::LayoutMap,
        width: u16,
        height: u16,
    ) -> Vec<usize> {
        let body_rows = Self::layout_visible_rows(width, height);
        let mut starts = Vec::with_capacity(map.segments.len() + 1);
        let mut acc = 0usize;
        for h in band_rows(map, body_rows) {
            starts.push(acc);
            acc += h;
        }
        starts.push(acc);
        starts
    }
}

/// The band glyph (shaded by the segment's share of the file) and colour for the
/// layout-map strip: the header in the metadata violet, padding dim, tensors in
/// the dtype amber with a fuller block the larger they are — the shading is the
/// map's ASCII "graphic", so a big tensor reads as a solid column.
fn band_style(seg: &crate::safelayout::Segment, total_len: u64) -> (char, Color) {
    use crate::safelayout::SegmentKind;
    match &seg.kind {
        SegmentKind::Header => ('█', palette::META),
        SegmentKind::Gap => ('░', palette::DIM),
        SegmentKind::Tensor { .. } => {
            let share = seg.len() as f64 / total_len.max(1) as f64;
            let glyph = if share >= 0.10 {
                '█'
            } else if share >= 0.02 {
                '▓'
            } else if share >= 0.005 {
                '▒'
            } else {
                '░'
            };
            (glyph, palette::DTYPE)
        }
    }
}

/// Per-segment band heights for the layout strip: proportional to each segment's
/// share of the file, at least one row each (so every tensor is labelled), summing
/// to a scrollable total. `body_rows` seeds the resolution so a small file fills
/// the viewport while a large one scrolls.
fn band_rows(map: &crate::safelayout::LayoutMap, body_rows: usize) -> Vec<usize> {
    use crate::safelayout::SegmentKind;
    let total_len = map.total_len.max(1) as f64;
    let target = map.segments.len().max(body_rows.max(1));
    map.segments
        .iter()
        .map(|s| {
            let share = s.len() as f64 / total_len;
            let proportional = (share * target as f64).round() as usize;
            match &s.kind {
                // The header lists its `__metadata__` tree-like, so give it enough
                // rows to show them (a label row + one per entry) even when its
                // byte share is tiny — as it is for a multi-GB file.
                SegmentKind::Header => proportional.max(1 + map.metadata.len()),
                _ => proportional.max(1),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_map_renders_summary_and_bands() {
        use crate::safelayout::{LayoutMap, Segment, SegmentKind};
        let seg = |name: &str, start, end, kind| Segment {
            name: name.to_string(),
            start,
            end,
            kind,
        };
        let tensor = |dtype: &str, shape: Vec<usize>| SegmentKind::Tensor {
            dtype: dtype.to_string(),
            shape,
        };
        let map = LayoutMap {
            name: "model.safetensors".to_string(),
            total_len: 1_000_000,
            header_len: 200,
            tensor_count: 2,
            metadata: vec![("format".to_string(), "pt".to_string())],
            segments: vec![
                seg(
                    "header (8 B length + JSON metadata)",
                    0,
                    200,
                    SegmentKind::Header,
                ),
                seg(
                    "embed.weight",
                    200,
                    800_200,
                    tensor("BF16", vec![1000, 256]),
                ),
                seg("norm.weight", 800_200, 1_000_000, tensor("F32", vec![256])),
            ],
        };
        let out = crate::tui::headless_render(90, 24, |f| {
            UI::render_layout(f, &map, 1, 0, None, true);
        })
        .unwrap();
        assert!(out.contains("Layout - model.safetensors"), "title:\n{out}");
        assert!(out.contains("2 tensors"), "summary:\n{out}");
        assert!(out.contains("1 metadata"), "metadata count:\n{out}");
        assert!(out.contains("embed.weight"), "tensor band:\n{out}");
        assert!(out.contains("BF16"), "dtype shown:\n{out}");
        // The header band lists its __metadata__ entries tree-like.
        assert!(out.contains("† format"), "metadata entry shown:\n{out}");
        // Absolute offsets are shown in hex.
        assert!(out.contains("0x00000000"), "header offset:\n{out}");
    }

    #[test]
    fn layout_tensor_band_names_are_tree_links() {
        use crate::safelayout::{LayoutMap, Segment, SegmentKind};
        let seg = |name: &str, start, end, kind| Segment {
            name: name.to_string(),
            start,
            end,
            kind,
        };
        let tensor = |dtype: &str, shape: Vec<usize>| SegmentKind::Tensor {
            dtype: dtype.to_string(),
            shape,
        };
        let map = LayoutMap {
            name: "model.safetensors".to_string(),
            total_len: 1_000_000,
            header_len: 200,
            tensor_count: 2,
            metadata: vec![],
            segments: vec![
                seg("header", 0, 200, SegmentKind::Header),
                seg(
                    "embed.weight",
                    200,
                    800_200,
                    tensor("BF16", vec![1000, 256]),
                ),
                seg("norm.weight", 800_200, 1_000_000, tensor("F32", vec![256])),
            ],
        };
        let mut links = Vec::new();
        crate::tui::headless_render(90, 24, |f| {
            let (_, _, l, _) = UI::render_layout(f, &map, 1, 0, None, true);
            links = l;
        })
        .unwrap();
        // Each *tensor* band's name is a `Tree` link; the header band is not.
        let targets: Vec<&Link> = links.iter().map(|(_, l)| l).collect();
        assert!(
            targets
                .iter()
                .any(|l| matches!(l, Link::Tree(n) if n == "embed.weight")),
            "embed.weight should link to the tree: {targets:?}"
        );
        assert!(
            targets
                .iter()
                .any(|l| matches!(l, Link::Tree(n) if n == "norm.weight")),
            "norm.weight should link to the tree: {targets:?}"
        );
        assert!(
            !targets
                .iter()
                .any(|l| matches!(l, Link::Tree(n) if n == "header")),
            "the header band is not a tensor link: {targets:?}"
        );
        // The link starts at the name column (after the fixed 20-column prefix).
        assert!(
            links.iter().all(|(r, _)| r.x == 20),
            "name column: {links:?}"
        );
    }
}
