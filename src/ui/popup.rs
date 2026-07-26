//! The floating chrome: pop-up boxes, the scrollable pop-up, hover bubbles and
//! the bottom prompt band. Screen-agnostic — callers hand in the lines to draw.

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Padding, Paragraph, Widget};

use super::palette;
use super::text::wrap_help;
use super::theme::dim_span;

/// How a centred pop-up box treats the frame around it.
pub(super) enum Backdrop {
    /// Leave the live frame intact around the box, clearing only the box's own
    /// rect — for a true pop-up (the legend `l`) that floats over a still-visible
    /// tree / detail view.
    Float,
    /// Wipe the whole frame to the [`palette::SCRIM`] first — for standalone
    /// message screens that own the frame (nothing is drawn beneath), so no
    /// terminal default background shows around the box.
    Fill,
}

pub(super) fn render_popup_box(
    frame: &mut Frame,
    title: &str,
    content: Vec<Line<'static>>,
    backdrop: Backdrop,
    fixed_inner_w: Option<usize>,
) -> Rect {
    let area = frame.area();
    let inner_w = fixed_inner_w.unwrap_or_else(|| {
        content
            .iter()
            .map(Line::width)
            .max()
            .unwrap_or(0)
            .max(title.chars().count() + 2)
    });
    let box_w = ((inner_w + 4) as u16).min(area.width); // 2 borders + 2 padding
    let box_h = ((content.len() + 2) as u16).min(area.height); // 2 borders
    let rect = Rect {
        x: area.width.saturating_sub(box_w) / 2,
        y: area.height.saturating_sub(box_h) / 2,
        width: box_w,
        height: box_h,
    };
    match backdrop {
        // Float over the live frame: clear only the box's own rect so the block
        // paints on a clean surface, while the screen behind stays visible around it.
        Backdrop::Float => Clear.render(rect, frame.buffer_mut()),
        // Own the frame: wipe every glyph, then paint the scrim, so nothing shows
        // through around the box.
        Backdrop::Fill => {
            Clear.render(area, frame.buffer_mut());
            Block::default()
                .style(Style::default().bg(palette::SCRIM))
                .render(area, frame.buffer_mut());
        }
    }

    let panel = Style::default().bg(palette::PANEL_BG);
    let block = popup_block(title, panel);
    let inner = block.inner(rect);
    block.render(rect, frame.buffer_mut());
    Paragraph::new(content)
        .style(panel)
        .render(inner, frame.buffer_mut());
    inner
}

/// The frame every popup wears: a rounded accent border, its title in the key colour, a
/// column of horizontal padding, and the panel background.
///
/// One definition so the two popup renderers can't drift apart visually — they sit on top
/// of each other in the same session, and a border or title that differed by a shade
/// between them would look like a rendering bug.
fn popup_block(title: &str, panel: Style) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(palette::ACCENT))
        .title(Span::styled(
            format!(" {title} "),
            Style::default()
                .fg(palette::KEY)
                .add_modifier(Modifier::BOLD),
        ))
        .padding(Padding::horizontal(1))
        .style(panel)
}

/// A floating popup with a vertically-scrollable `body` and a pinned `footer`
/// row, sized to fit the frame (never taller than it). `scroll` is the first
/// visible body row (clamped internally); returns the maximum valid scroll so the
/// caller can clamp its own offset. When the body overflows, a dim indicator row
/// (range + scroll keys) sits just above the footer.
pub(super) fn render_scroll_popup(
    frame: &mut Frame,
    title: &str,
    body: &[Line<'static>],
    footer: Line<'static>,
    scroll: usize,
    clickable: &[(usize, KeyEvent)],
) -> (usize, Vec<(Rect, KeyEvent)>) {
    let area = frame.area();
    let panel = Style::default().bg(palette::PANEL_BG);
    let total = body.len();

    // Height first (independent of width): fit the content, but never taller than
    // the frame (1-row margin top+bottom). The footer takes the last inner row;
    // when the body doesn't fit in the rest, reserve one more for the scroll
    // indicator.
    // The `.max(3)` keeps a usable box on a short frame, but it must never exceed the
    // frame itself: on a 2-row terminal it produced a 3-row rect, and a rect past the
    // buffer is a Ratatui panic rather than a clip.
    let max_box_h = area.height.saturating_sub(2).max(3).min(area.height);
    let box_h = ((total + 3) as u16).min(max_box_h); // body + footer + 2 borders
    let inner_h = box_h.saturating_sub(2) as usize;
    let overflow = total > inner_h.saturating_sub(1);
    let visible = inner_h.saturating_sub(1 + usize::from(overflow));
    let max_scroll = total.saturating_sub(visible);
    let scroll = scroll.min(max_scroll);
    let indicator = overflow.then(|| {
        format!(
            "↑↓ PgUp/PgDn scroll · {}–{} of {total}",
            scroll + 1,
            scroll + visible
        )
    });

    // Width sizes to the widest of the body, footer, title, and the indicator (so
    // the indicator isn't clipped when the body lines are short).
    let inner_w = body
        .iter()
        .chain(std::iter::once(&footer))
        .map(Line::width)
        .max()
        .unwrap_or(0)
        .max(title.chars().count() + 2)
        .max(
            indicator
                .as_deref()
                .map(str::chars)
                .map_or(0, |c| c.count()),
        );
    let box_w = ((inner_w + 4) as u16).min(area.width); // 2 borders + 2 padding

    let rect = Rect {
        x: area.width.saturating_sub(box_w) / 2,
        y: area.height.saturating_sub(box_h) / 2,
        width: box_w,
        height: box_h,
    };
    Clear.render(rect, frame.buffer_mut());
    let block = popup_block(title, panel);
    let inner = block.inner(rect);
    block.render(rect, frame.buffer_mut());

    // Clone only the visible window (not the whole body) so scrolling a large
    // pop-up — e.g. a big safetensors header — stays O(screen), not O(content).
    let window: Vec<Line> = body.iter().skip(scroll).take(visible).cloned().collect();
    Paragraph::new(window).style(panel).render(
        Rect {
            height: visible as u16,
            ..inner
        },
        frame.buffer_mut(),
    );
    if let Some(hint) = indicator {
        Paragraph::new(Line::from(dim_span(hint)))
            .style(panel)
            .render(
                Rect {
                    y: inner.y + visible as u16,
                    height: 1,
                    ..inner
                },
                frame.buffer_mut(),
            );
    }
    Paragraph::new(footer).style(panel).render(
        Rect {
            y: inner.y + inner.height - 1,
            height: 1,
            ..inner
        },
        frame.buffer_mut(),
    );
    // Map each requested body-line index to its on-screen row (when currently
    // visible in the scrolled window), so the caller can hit-test clicks on it.
    let regions: Vec<(Rect, KeyEvent)> = clickable
        .iter()
        .filter(|(idx, _)| *idx >= scroll && *idx < scroll + visible)
        .map(|(idx, key)| {
            let row = Rect {
                y: inner.y + (idx - scroll) as u16,
                height: 1,
                ..inner
            };
            (row, *key)
        })
        .collect();
    (max_scroll, regions)
}

/// A hover help bubble floated adjacent to `anchor` — just above it, or below
/// when it hugs the top (as the tree's hints do) — with a `border` colour and an
/// optional `title` riding the border (both matching the element it describes).
/// Word-wrapped and clamped on-screen; the caller draws the screen first.
pub(super) fn render_hover_bubble(
    frame: &mut Frame,
    anchor: Rect,
    border: Color,
    title: Option<&str>,
    help: &str,
) {
    let area = frame.area();
    let wrap_w = 52.min((area.width as usize).saturating_sub(4)).max(8);
    let lines = wrap_help(help, wrap_w);
    if lines.is_empty() {
        return;
    }
    let title_w = title.map(|t| t.chars().count() + 2).unwrap_or(0);
    let inner_w = lines
        .iter()
        .map(Line::width)
        .max()
        .unwrap_or(0)
        .max(title_w);
    let box_w = ((inner_w + 4) as u16).min(area.width); // 2 borders + 2 padding
    let box_h = ((lines.len() + 2) as u16).min(area.height); // 2 borders
    // Prefer just above the anchor; drop below it when there isn't room above.
    let y = if anchor.y >= box_h {
        anchor.y - box_h
    } else {
        (anchor.y + anchor.height).min(area.height.saturating_sub(box_h))
    };
    // Left-align to the anchor, nudged left as needed to stay fully on-screen.
    let x = anchor.x.min(area.width.saturating_sub(box_w));
    let rect = Rect {
        x,
        y,
        width: box_w,
        height: box_h,
    };
    Clear.render(rect, frame.buffer_mut());
    let panel = Style::default().bg(palette::PANEL_BG);
    let mut block = Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border))
        .padding(Padding::horizontal(1))
        .style(panel);
    if let Some(t) = title {
        block = block.title(Span::styled(
            t.to_string(),
            Style::default().fg(border).add_modifier(Modifier::BOLD),
        ));
    }
    let inner = block.inner(rect);
    block.render(rect, frame.buffer_mut());
    Paragraph::new(lines)
        .style(panel)
        .render(inner, frame.buffer_mut());
}

/// A help bubble for a footer shortcut chip (no title, key-cyan border), floated
/// adjacent to the chip. See [`render_hover_bubble`].
pub(crate) fn render_shortcut_bubble(frame: &mut Frame, anchor: Rect, help: &str) {
    render_hover_bubble(frame, anchor, palette::KEY, None, help);
}

/// A full-width pop-up framed with only top+bottom borders (the `title` rides the
/// top rule) over the live frame, centred vertically. Its body rows stay flush at
/// column 0 — used by the copied-command pop-up so the command can still be
/// selected cleanly by hand when the OSC-52 copy doesn't reach the terminal.
pub(super) fn render_titled_bar(
    frame: &mut Frame,
    title: Line<'static>,
    content: Vec<Line<'static>>,
) {
    let area = frame.area();
    let box_h = ((content.len() + 2) as u16).min(area.height);
    let rect = Rect {
        x: 0,
        y: area.height.saturating_sub(box_h) / 2,
        width: area.width,
        height: box_h,
    };
    let panel = Style::default().bg(palette::PANEL_BG);
    let block = Block::default()
        .borders(Borders::TOP | Borders::BOTTOM)
        .border_style(Style::default().fg(palette::ACCENT))
        .title(title)
        .style(panel);
    let inner = block.inner(rect);
    Clear.render(rect, frame.buffer_mut());
    block.render(rect, frame.buffer_mut());
    Paragraph::new(content)
        .style(panel)
        .render(inner, frame.buffer_mut());
}

/// Composite a bottom-pinned two-row prompt (`prompt` on the second-to-last row,
/// `feedback` on the last) over the live frame — the Ratatui equivalent of the
/// raw prompts' `MoveTo(0, h-2)` / `MoveTo(0, h-1)` line writes. Each row is
/// cleared (its tail blanked) so a shorter new prompt leaves nothing stale behind.
pub(super) fn render_bottom_band(
    frame: &mut Frame,
    prompt: Line<'static>,
    feedback: Line<'static>,
) {
    let area = frame.area();
    if area.height < 2 {
        return;
    }
    // Clear the two rows first: the band overlays the live data view, whose own
    // footer sits on these same rows. A `Paragraph` only paints its glyphs, so
    // without this the footer bled through past the (often shorter) band text —
    // e.g. the dtype menu left "…irst/last rows" showing behind it.
    Clear.render(
        Rect {
            x: 0,
            y: area.height - 2,
            width: area.width,
            height: 2,
        },
        frame.buffer_mut(),
    );
    Paragraph::new(prompt).render(
        Rect {
            x: 0,
            y: area.height - 2,
            width: area.width,
            height: 1,
        },
        frame.buffer_mut(),
    );
    Paragraph::new(feedback).render(
        Rect {
            x: 0,
            y: area.height - 1,
            width: area.width,
            height: 1,
        },
        frame.buffer_mut(),
    );
}

/// The feedback line below a prompt: a red error message, or an empty line (which
/// still clears the row) when there's nothing to report.
pub(super) fn error_line(error: Option<&str>) -> Line<'static> {
    match error {
        Some(msg) => Line::from(Span::styled(
            msg.to_string(),
            Style::default().fg(palette::ERROR),
        )),
        None => Line::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tests_support::strip_ansi_codes;

    #[test]
    fn bottom_band_clears_the_rows_it_overlays() {
        // The band (dtype menu / slice / reshape prompts) overlays the live data
        // view, whose footer sits on the same bottom rows. It must clear them, or
        // the footer bleeds through past the shorter band text.
        let out = crate::tui::headless_render(80, 8, |f| {
            let a = f.area();
            let footer = "LEFT_edge_of_footer ······································· TAIL_MARKER";
            Paragraph::new(footer).render(
                Rect {
                    x: 0,
                    y: a.height - 2,
                    width: a.width,
                    height: 1,
                },
                f.buffer_mut(),
            );
            Paragraph::new("SECOND_ROW_MARKER").render(
                Rect {
                    x: 0,
                    y: a.height - 1,
                    width: a.width,
                    height: 1,
                },
                f.buffer_mut(),
            );
            render_bottom_band(f, Line::from("short prompt"), Line::from("short feedback"));
        })
        .unwrap();
        assert!(out.contains("short prompt"), "band prompt shown:\n{out}");
        assert!(
            out.contains("short feedback"),
            "band feedback shown:\n{out}"
        );
        assert!(
            !out.contains("TAIL_MARKER"),
            "footer tail bled through the band:\n{out}"
        );
        assert!(
            !out.contains("SECOND_ROW_MARKER"),
            "footer second row bled through the band:\n{out}"
        );
    }

    #[test]
    fn shortcut_bubble_shows_the_help_text() {
        let anchor = Rect {
            x: 4,
            y: 1,
            width: 1,
            height: 1,
        };
        let out = crate::tui::headless_render(80, 20, |f| {
            render_shortcut_bubble(f, anchor, "Expand every group in the tree.");
        })
        .unwrap();
        let plain = strip_ansi_codes(&out);
        assert!(
            plain.contains("Expand every group in the tree."),
            "bubble should show the help text:\n{plain}"
        );
    }

    #[test]
    fn scroll_popup_reports_overflow() {
        let body: Vec<Line> = (0..50).map(|i| Line::from(format!("row {i}"))).collect();
        let footer = Line::from("footer");

        // Tall frame: the whole body fits → nothing to scroll.
        let mut fits_max = usize::MAX;
        crate::tui::headless_render(40, 60, |f| {
            fits_max = render_scroll_popup(f, "T", &body, footer.clone(), 0, &[]).0;
        })
        .unwrap();
        assert_eq!(fits_max, 0, "a 50-row body in a 60-row frame fits");

        // Short frame: the body overflows → scrollable, and the indicator shows.
        let mut small_max = 0;
        let out = crate::tui::headless_render(40, 12, |f| {
            small_max = render_scroll_popup(f, "T", &body, footer.clone(), 0, &[]).0;
        })
        .unwrap();
        assert!(small_max > 0, "a 50-row body in a 12-row frame must scroll");
        assert!(
            strip_ansi_codes(&out).contains("of 50"),
            "the overflow indicator shows the total:\n{out}"
        );
    }
}
