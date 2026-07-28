//! The vertical scrollbar: geometry ([`VScrollbar`]) and the widget that draws
//! it. Per-screen constructors live with their screens.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::widgets::{Scrollbar, ScrollbarOrientation, ScrollbarState, StatefulWidget};

use super::UI;
use super::palette;

/// Where a mode's vertical scroll bar sits, how a pointer over it maps to a scroll
/// offset, and its current position. Built by [`VScrollbar::for_body`] (or the
/// tree/files wrappers); the `run_mode` engine draws it ([`UI::render_vscrollbar`])
/// and drag-scrubs it, so every mode gets a bar the same way.
#[derive(Clone, Copy)]
pub(crate) struct VScrollbar {
    /// Rightmost column of the body, reserved for the bar.
    pub col: u16,
    /// First body row (the terminal row just below the header).
    pub top: u16,
    /// Track height in rows — the number of visible body rows.
    pub rows: u16,
    /// The largest valid scroll offset (`total - visible`).
    pub max_offset: usize,
    /// The current scroll offset (first visible row), clamped to `max_offset`.
    pub offset: usize,
}

impl VScrollbar {
    /// The bar for a scrollable `body` region showing `total` rows starting at
    /// `offset`, or `None` when it all fits (or there's no room for a bar +
    /// content). The bar rides `body`'s rightmost column.
    pub(crate) fn for_body(body: Rect, total: usize, offset: usize) -> Option<Self> {
        let rows = body.height as usize;
        if body.width < 2 || rows == 0 || total <= rows {
            return None;
        }
        let max_offset = total - rows;
        Some(Self {
            col: body.x + body.width - 1,
            top: body.y,
            rows: body.height,
            max_offset,
            offset: offset.min(max_offset),
        })
    }

    /// The scroll offset a pointer at terminal `row` scrubs to: the top of the
    /// track maps to offset 0 and the bottom to `max_offset`, proportionally
    /// (rows above/below the track clamp to the ends).
    pub(crate) fn offset_at(&self, row: u16) -> usize {
        if self.rows <= 1 {
            return 0;
        }
        let rel = row.saturating_sub(self.top).min(self.rows - 1);
        let frac = f64::from(rel) / f64::from(self.rows - 1);
        (frac * self.max_offset as f64).round() as usize
    }

    /// Whether the terminal cell `(col, row)` lands on the scroll bar.
    pub(crate) fn hit(&self, col: u16, row: u16) -> bool {
        col == self.col && row >= self.top && row < self.top + self.rows
    }
}

impl UI {
    /// Draw a vertical scroll bar over its reserved column (the ratatui `Scrollbar`
    /// widget — dim `│` track, accent `█` thumb). The `run_mode` engine calls this
    /// for every mode that reports a [`VScrollbar`], so no mode draws its own.
    pub(crate) fn render_vscrollbar(frame: &mut Frame, sb: &VScrollbar) {
        // The geometry comes from the body rect its screen computed, which on a pane
        // shorter than that screen's chrome can start or end past the frame. Clamp here
        // — the one place every screen's bar is drawn — because a widget rect outside
        // the buffer is a Ratatui panic, not a clip. (Found by the pty resize test at
        // 12×2: the bar was asked to draw a row below the last one.)
        let area = frame.area();
        let rect = crate::ui::fit_rows(area, sb.top, sb.rows);
        if rect.height == 0 || area.width == 0 {
            return;
        }
        let mut state = ScrollbarState::new(sb.max_offset + 1)
            .position(sb.offset)
            .viewport_content_length(rect.height as usize);
        StatefulWidget::render(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .track_symbol(Some("│"))
                .thumb_symbol("█")
                .track_style(Style::default().fg(palette::DIM))
                .thumb_style(Style::default().fg(palette::ACCENT)),
            Rect {
                x: sb.col.min(area.width - 1),
                width: 1,
                ..rect
            },
            frame.buffer_mut(),
            &mut state,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vscrollbar_for_body_reports_overflow_only() {
        let body = Rect {
            x: 0,
            y: 3,
            width: 40,
            height: 10,
        };
        // Fits → no bar.
        assert!(VScrollbar::for_body(body, 10, 0).is_none());
        assert!(VScrollbar::for_body(body, 5, 0).is_none());
        // Overflows → a bar in the last column; the offset clamps to max_offset.
        let sb = VScrollbar::for_body(body, 30, 999).expect("overflow → bar");
        assert_eq!((sb.col, sb.top, sb.rows), (39, 3, 10));
        assert_eq!(sb.max_offset, 20); // 30 - 10
        assert_eq!(sb.offset, 20); // clamped from 999
        // Too narrow for a bar + content → none.
        assert!(VScrollbar::for_body(Rect { width: 1, ..body }, 30, 0).is_none());
    }

    #[test]
    fn render_vscrollbar_draws_thumb_and_track() {
        let sb = VScrollbar {
            col: 39,
            top: 0,
            rows: 10,
            max_offset: 20,
            offset: 5,
        };
        let out = crate::tui::headless_render(40, 10, |f| UI::render_vscrollbar(f, &sb)).unwrap();
        assert!(out.contains('█'), "thumb:\n{out}");
        assert!(out.contains('│'), "track:\n{out}");
    }
}
