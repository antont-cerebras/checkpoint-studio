//! The compare screen: a structural diff of the open checkpoint against another.
//!
//! Shows what the `diff` subcommand shows — tensors and metadata added, removed and
//! changed — as a scrollable screen rather than a stream of text. The report itself
//! comes from `crate::compare`, shared with the CLI and the web endpoint, so the three
//! cannot disagree about what differs (there is a differential test that says so).
//!
//! Value comparison (`diff --values`) is deliberately absent here: it scans every byte
//! of both checkpoints, which needs a progress bar and somewhere to put per-tensor
//! findings. The footer names the CLI command that does it.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use checkpoint_studio_core::diff::{DiffReport, MetaChange, TensorChange, TensorSig};

use super::UI;
use super::hints::{chip_regions, close_button, diff_hint_lines};
use super::palette;
use super::scroll::VScrollbar;
use super::text::truncate_keep_end;
use super::theme::dim_span;
use crate::utils::{format_parameters, format_size};

/// The one-line status bar under the body.
const DIFF_FOOTER_HEIGHT: usize = 1;

/// A row of the compare screen's body, already reduced to what the renderer draws. Built
/// once per frame from the report so scrolling, hit-testing and the scroll bar all agree
/// on how many rows there are — the same reason the tree flattens before it renders.
enum DiffRow {
    /// A section heading (`Added (3)`), with the colour its entries use.
    Section {
        title: String,
        color: Color,
    },
    /// One tensor or metadata entry: a marker, a name, and its detail.
    Entry {
        marker: &'static str,
        name: String,
        detail: String,
        color: Color,
    },
    /// A note where a section would be (`none`), or an explanatory line.
    Note(String),
    Blank,
}

impl UI {
    /// Header rows above the compare body: title, the two sides, the verdict, a rule.
    pub(crate) fn diff_header_rows(_width: u16) -> usize {
        4
    }

    /// Rows the bottom-pinned key-hint footer occupies.
    pub(crate) fn diff_hint_rows(width: u16) -> usize {
        diff_hint_lines(width).0.len()
    }

    /// Body rows visible at this size, so the scroll offset the key handler clamps
    /// against matches what [`Self::render_diff`] draws.
    pub(crate) fn diff_visible_rows(width: u16, height: u16) -> usize {
        (height as usize)
            .saturating_sub(
                Self::diff_header_rows(width) + Self::diff_hint_rows(width) + DIFF_FOOTER_HEIGHT,
            )
            .max(1)
    }

    /// The total number of body rows for `report` — what the caller clamps scrolling to.
    pub(crate) fn diff_total_rows(report: &DiffReport, width: u16) -> usize {
        diff_rows(report, width).len()
    }

    /// The compare screen's scroll-bar geometry, or `None` when it all fits.
    pub(crate) fn diff_scrollbar(
        width: u16,
        height: u16,
        total: usize,
        offset: usize,
    ) -> Option<VScrollbar> {
        VScrollbar::for_body(
            Rect {
                x: 0,
                y: Self::diff_header_rows(width) as u16,
                width,
                height: Self::diff_visible_rows(width, height) as u16,
            },
            total,
            offset,
        )
    }

    /// Render the compare screen. `old_label` / `new_label` name the two sides (baseline
    /// first, matching `diff OLD NEW`), `verdict` is the shared one-line summary. Returns
    /// the clickable footer chips + `[×]`, like the other screens.
    pub(crate) fn render_diff(
        frame: &mut Frame,
        old_label: &str,
        new_label: &str,
        verdict: &str,
        report: &DiffReport,
        scroll: usize,
        interactive: bool,
    ) -> Vec<(Rect, KeyEvent)> {
        let area = frame.area();
        let (width, height) = (area.width, area.height);
        if height < (DIFF_FOOTER_HEIGHT as u16 + 1) {
            return Vec::new();
        }

        let text_w = (width as usize).saturating_sub(2);
        let mut header: Vec<Line> = vec![
            Line::from(Span::raw("Compare checkpoints")),
            Line::from(vec![
                dim_span("  old  "),
                Span::styled(
                    truncate_keep_end(old_label, text_w.saturating_sub(7)),
                    Style::default().fg(palette::REMOVED),
                ),
            ]),
            Line::from(vec![
                dim_span("  new  "),
                Span::styled(
                    truncate_keep_end(new_label, text_w.saturating_sub(7)),
                    Style::default().fg(palette::ADDED),
                ),
            ]),
        ];
        header.push(Line::from(Span::styled(
            "─".repeat(width as usize),
            Style::default().fg(palette::DIM),
        )));

        let (hint_lines, chips) = diff_hint_lines(width);
        let hint_rows = hint_lines.len();
        let header_rows = header.len();
        let body_rows =
            (height as usize).saturating_sub(header_rows + hint_rows + DIFF_FOOTER_HEIGHT);

        let rows = diff_rows(report, width);
        let bar = interactive && Self::diff_scrollbar(width, height, rows.len(), scroll).is_some();
        let body_width = width.saturating_sub(u16::from(bar));

        Paragraph::new(header).render(
            super::fit_rows(area, 0, header_rows as u16),
            frame.buffer_mut(),
        );

        let body: Vec<Line> = rows
            .iter()
            .skip(scroll)
            .take(body_rows)
            .map(diff_row_line)
            .collect();
        Paragraph::new(body).render(
            Rect {
                width: body_width,
                ..super::fit_rows(area, header_rows as u16, body_rows as u16)
            },
            frame.buffer_mut(),
        );

        let hint_y = height.saturating_sub(1 + hint_rows as u16);
        Paragraph::new(hint_lines).render(
            super::fit_rows(area, hint_y, hint_rows as u16),
            frame.buffer_mut(),
        );

        // Status bar: the verdict plus the size/params delta between the two sides.
        let delta = size_delta(report);
        let status = Line::from(vec![
            Span::styled(
                format!(" {verdict} "),
                Style::default()
                    .bg(palette::STATUS_BG)
                    .fg(palette::STATUS_FG),
            ),
            dim_span(format!("  {delta}")),
        ]);
        Paragraph::new(status).render(
            Rect {
                x: 0,
                y: height.saturating_sub(1),
                width,
                height: 1,
            },
            frame.buffer_mut(),
        );

        let mut regions = chip_regions(&chips, hint_y);
        regions.extend(close_button(
            frame,
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
        ));
        regions
    }
}

/// `1.2 GiB → 1.4 GiB (+195.3 MiB)`, or a note that both sides weigh the same.
fn size_delta(report: &DiffReport) -> String {
    let bytes = if report.old_bytes == report.new_bytes {
        format!("{} (unchanged)", format_size(report.new_bytes))
    } else {
        let (from, to) = (report.old_bytes, report.new_bytes);
        let sign = if to > from { '+' } else { '-' };
        let magnitude = format_size(to.abs_diff(from));
        format!(
            "{} → {} ({sign}{magnitude})",
            format_size(from),
            format_size(to)
        )
    };
    if report.old_params == report.new_params {
        format!("{bytes} · {} params", format_parameters(report.new_params))
    } else {
        format!(
            "{bytes} · {} → {} params",
            format_parameters(report.old_params),
            format_parameters(report.new_params)
        )
    }
}

/// The report as body rows. One function so the renderer, the scroll clamp and the
/// scroll bar cannot disagree about the row count.
fn diff_rows(report: &DiffReport, width: u16) -> Vec<DiffRow> {
    let name_w = (width as usize).saturating_sub(30).max(20);
    let mut rows = Vec::new();

    let section = |rows: &mut Vec<DiffRow>, title: &str, n: usize, color: Color| {
        if !rows.is_empty() {
            rows.push(DiffRow::Blank);
        }
        rows.push(DiffRow::Section {
            title: format!("{title} ({n})"),
            color,
        });
        if n == 0 {
            rows.push(DiffRow::Note("  none".to_string()));
        }
    };

    section(
        &mut rows,
        "Tensors added",
        report.tensors_added.len(),
        palette::ADDED,
    );
    for (name, sig) in &report.tensors_added {
        rows.push(DiffRow::Entry {
            marker: "+",
            name: truncate_keep_end(name, name_w),
            detail: sig_text(sig),
            color: palette::ADDED,
        });
    }

    section(
        &mut rows,
        "Tensors removed",
        report.tensors_removed.len(),
        palette::REMOVED,
    );
    for (name, sig) in &report.tensors_removed {
        rows.push(DiffRow::Entry {
            marker: "-",
            name: truncate_keep_end(name, name_w),
            detail: sig_text(sig),
            color: palette::REMOVED,
        });
    }

    section(
        &mut rows,
        "Tensors changed",
        report.tensors_changed.len(),
        palette::CHANGED,
    );
    for change in &report.tensors_changed {
        rows.push(DiffRow::Entry {
            marker: "~",
            name: truncate_keep_end(&change.name, name_w),
            detail: change_text(change),
            color: palette::CHANGED,
        });
    }

    rows.push(DiffRow::Blank);
    rows.push(DiffRow::Note(format!(
        "  {} tensors unchanged",
        report.tensors_unchanged
    )));

    let meta_total =
        report.meta_added.len() + report.meta_removed.len() + report.meta_changed.len();
    section(&mut rows, "Metadata", meta_total, palette::META);
    for (name, val) in &report.meta_added {
        rows.push(DiffRow::Entry {
            marker: "+",
            name: truncate_keep_end(name, name_w),
            detail: val.value.clone(),
            color: palette::ADDED,
        });
    }
    for (name, val) in &report.meta_removed {
        rows.push(DiffRow::Entry {
            marker: "-",
            name: truncate_keep_end(name, name_w),
            detail: val.value.clone(),
            color: palette::REMOVED,
        });
    }
    for change in &report.meta_changed {
        rows.push(DiffRow::Entry {
            marker: "~",
            name: truncate_keep_end(&change.name, name_w),
            detail: meta_change_text(change),
            color: palette::CHANGED,
        });
    }
    if report.meta_unchanged > 0 {
        rows.push(DiffRow::Note(format!(
            "  {} metadata entries unchanged",
            report.meta_unchanged
        )));
    }

    rows
}

fn sig_text(sig: &TensorSig) -> String {
    format!(
        "[{}, {}]",
        sig.dtype,
        crate::utils::format_shape(&sig.shape)
    )
}

fn change_text(change: &TensorChange) -> String {
    let (old, new) = (&change.old, &change.new);
    if old.dtype != new.dtype && old.shape != new.shape {
        return format!("{} → {}", sig_text(old), sig_text(new));
    }
    if old.dtype != new.dtype {
        return format!("dtype {} → {}", old.dtype, new.dtype);
    }
    if old.shape != new.shape {
        return format!(
            "shape {} → {}",
            crate::utils::format_shape(&old.shape),
            crate::utils::format_shape(&new.shape)
        );
    }
    // Same signature: the change came from a value / distribution comparison, which
    // only the CLI runs — so say so rather than showing an empty detail.
    "values differ".to_string()
}

fn meta_change_text(change: &MetaChange) -> String {
    format!("{} → {}", change.old.value, change.new.value)
}

fn diff_row_line(row: &DiffRow) -> Line<'static> {
    match row {
        DiffRow::Section { title, color } => Line::from(Span::styled(
            title.clone(),
            Style::default().fg(*color).add_modifier(Modifier::BOLD),
        )),
        DiffRow::Entry {
            marker,
            name,
            detail,
            color,
        } => Line::from(vec![
            Span::styled(format!("  {marker} "), Style::default().fg(*color)),
            Span::styled(name.clone(), Style::default().fg(Color::Reset)),
            dim_span(format!("  {detail}")),
        ]),
        DiffRow::Note(text) => Line::from(dim_span(text.clone())),
        DiffRow::Blank => Line::default(),
    }
}
