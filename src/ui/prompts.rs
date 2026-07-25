//! The modal prompts and menus: dtype/choice menus, the slice and reshape
//! inputs, the confirm pop-up and the command palette.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use crate::sample::ViewDtype;
use crate::utils::format_shape;

use super::UI;
use super::hints::hint_spans;
use super::palette;
use super::popup::{Backdrop, error_line, render_bottom_band, render_popup_box};
use super::text::input_box_spans;
use super::theme::{dim_span, key_span};

impl UI {
    /// The Ratatui port of [`Self::draw_dtype_menu`]: overlay a dtype-selection
    /// menu on the bottom two rows of the live preview frame — a `view as:` label
    /// followed by the available views as buttons (`current` highlighted), with a
    /// hint line below. Composited *after* the preview is drawn into the frame.
    pub fn render_dtype_menu(frame: &mut Frame, options: &[ViewDtype], current: usize) {
        let mut menu: Vec<Span> = vec![dim_span("view as:")];
        for (i, opt) in options.iter().enumerate() {
            let label = format!(" {} ", opt.menu_label());
            if i == current {
                menu.push(Span::styled(
                    label,
                    Style::default()
                        .fg(palette::SELECT_FG)
                        .bg(palette::SELECT_BG)
                        .add_modifier(Modifier::BOLD),
                ));
            } else {
                menu.push(dim_span(label));
            }
        }
        let hints = Line::from(hint_spans(&[
            ("← → or d/D", "move"),
            ("Enter", "apply"),
            ("Esc", "cancel"),
        ]));
        render_bottom_band(frame, Line::from(menu), hints);
    }

    /// The Ratatui port of [`Self::draw_slice_prompt`]: a bottom-pinned prompt to
    /// jump to a slice by index (over the live data view), with a fixed-width
    /// input box and a feedback line below for an out-of-range / invalid entry.
    pub fn render_slice_prompt(frame: &mut Frame, slices: usize, input: &str, error: Option<&str>) {
        let mut prompt: Vec<Span> = vec![
            Span::styled("Go to slice ", Style::default().fg(palette::KEY)),
            dim_span(format!("(0-{} or 0-100%)", slices.saturating_sub(1))),
            Span::raw("  "),
        ];
        prompt.extend(input_box_spans(input, input.chars().count(), 5));
        prompt.push(Span::raw("  "));
        prompt.push(key_span("Enter"));
        prompt.push(dim_span(" to jump · "));
        prompt.push(key_span("Esc"));
        prompt.push(dim_span(" to cancel"));
        render_bottom_band(frame, Line::from(prompt), error_line(error));
    }

    /// The Ratatui port of [`Self::draw_reshape_prompt`]: shows the stored shape
    /// and the element count the entry must multiply to, the input box, and a
    /// feedback line for errors.
    pub fn render_reshape_prompt(
        frame: &mut Frame,
        elements: usize,
        stored: &[usize],
        input: &str,
        error: Option<&str>,
    ) {
        let mut prompt: Vec<Span> = vec![
            Span::styled(
                format!("Reshape {} ", format_shape(stored)),
                Style::default().fg(palette::KEY),
            ),
            dim_span(format!(
                "(dims multiplying to {elements}; `-1`/`*`/`_` infers one; empty clears)"
            )),
            Span::raw("  "),
        ];
        prompt.extend(input_box_spans(input, input.chars().count(), 16));
        prompt.push(Span::raw("  "));
        prompt.push(key_span("Enter"));
        prompt.push(dim_span(" to apply · "));
        prompt.push(key_span("Esc"));
        prompt.push(dim_span(" to cancel"));
        render_bottom_band(frame, Line::from(prompt), error_line(error));
    }

    /// The Ratatui port of [`Self::draw_text_prompt`]: a bottom-pinned free-text
    /// input (label + editable box + optional error line). Used for the repack
    /// output filename, buffer size, and histogram bin count.
    pub fn render_text_prompt(frame: &mut Frame, label: &str, input: &str, error: Option<&str>) {
        let mut prompt: Vec<Span> = vec![Span::styled(
            format!("{label} "),
            Style::default().fg(palette::KEY),
        )];
        prompt.extend(input_box_spans(input, input.chars().count(), 24));
        prompt.push(Span::raw("  "));
        prompt.push(key_span("Enter"));
        prompt.push(dim_span(" to confirm · "));
        prompt.push(key_span("Esc"));
        prompt.push(dim_span(" to cancel"));
        render_bottom_band(frame, Line::from(prompt), error_line(error));
    }

    /// The Ratatui port of [`Self::draw_choice_menu`]: a full-screen single-choice
    /// menu — a title, an underline rule, and a strip of `options` with `current`
    /// highlighted, plus a hint line. Used to pick the repack codec / confirm.
    pub fn render_choice_menu(frame: &mut Frame, title: &str, options: &[&str], current: usize) {
        let mut strip: Vec<Span> = Vec::new();
        for (i, opt) in options.iter().enumerate() {
            let label = format!(" {opt} ");
            if i == current {
                strip.push(Span::styled(
                    label,
                    Style::default()
                        .fg(palette::SELECT_FG)
                        .bg(palette::SELECT_BG)
                        .add_modifier(Modifier::BOLD),
                ));
            } else {
                strip.push(dim_span(label));
            }
            strip.push(Span::raw(" "));
        }
        let lines: Vec<Line> = vec![
            Line::from(Span::raw(title.to_string())),
            Line::from(Span::raw("=".repeat(title.len().max(10)))),
            Line::default(),
            Line::from(strip),
            Line::default(),
            Line::from(hint_spans(&[
                ("← →", "move"),
                ("Enter", "select"),
                ("Esc", "cancel"),
            ])),
        ];
        Paragraph::new(lines).render(frame.area(), frame.buffer_mut());
    }

    /// A yes/no confirmation **floated over the live frame** (the screen behind stays
    /// visible): a title, the `body` summary lines, then an `[Apply] [Cancel]`-style
    /// choice strip (the `selected` option inverted) and a key hint. Drives the
    /// in-place rename apply confirmation.
    pub fn render_confirm_popup(
        frame: &mut Frame,
        title: &str,
        body: &[String],
        options: &[&str],
        selected: usize,
    ) {
        let mut content: Vec<Line> = body
            .iter()
            .map(|l| Line::from(Span::raw(l.clone())))
            .collect();
        content.push(Line::default());
        let mut strip: Vec<Span> = Vec::new();
        for (i, opt) in options.iter().enumerate() {
            let label = format!(" {opt} ");
            if i == selected {
                strip.push(Span::styled(
                    label,
                    Style::default()
                        .fg(palette::SELECT_FG)
                        .bg(palette::SELECT_BG)
                        .add_modifier(Modifier::BOLD),
                ));
            } else {
                strip.push(dim_span(label));
            }
            strip.push(Span::raw("  "));
        }
        content.push(Line::from(strip));
        content.push(Line::default());
        content.push(Line::from(hint_spans(&[
            ("← →", "move"),
            ("Enter", "select"),
            ("Y", "apply"),
            ("Esc", "cancel"),
        ])));
        render_popup_box(frame, title, content, Backdrop::Float, None);
    }

    /// A floating selection menu: `items` numbered one per row with `selected`
    /// highlighted, a `preview` of the highlighted choice's output below, and a
    /// key hint. Used by the `t` copy-format picker; the caller drives selection
    /// and repaints. Returns each item's on-screen rect so clicks/hovers can be
    /// mapped back to a row.
    pub fn render_menu_box(
        frame: &mut Frame,
        title: &str,
        items: &[&str],
        selected: usize,
        preview: &[Line<'static>],
    ) -> Vec<Rect> {
        let mut content: Vec<Line> = items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let row = format!("{}. {item}", i + 1);
                if i == selected {
                    Line::from(Span::styled(
                        format!("▸ {row}"),
                        Style::default()
                            .fg(palette::ACCENT)
                            .add_modifier(Modifier::BOLD),
                    ))
                } else {
                    Line::from(Span::raw(format!("  {row}")))
                }
            })
            .collect();
        // Live, tree-coloured preview of the highlighted export (from the current
        // checkpoint), each line indented under a "preview:" header.
        content.push(Line::default());
        content.push(Line::from(dim_span("preview:")));
        for line in preview {
            let mut spans = vec![Span::raw("  ")];
            spans.extend(line.spans.iter().cloned());
            content.push(Line::from(spans));
        }
        content.push(Line::default());
        content.push(Line::from(dim_span(
            "↑/↓ or 1–8 choose  ·  Enter/click copy  ·  Esc cancel",
        )));
        // A fixed inner width keeps the box a constant size across options (the
        // preview rows are already a fixed count); over-wide lines are clipped.
        let width = (frame.area().width as usize)
            .saturating_sub(4)
            .clamp(24, 110);
        let inner = render_popup_box(frame, title, content, Backdrop::Float, Some(width));
        // The items occupy the first `items.len()` inner rows.
        (0..items.len())
            .map(|i| Rect {
                x: inner.x,
                y: inner.y + i as u16,
                width: inner.width,
                height: 1,
            })
            .collect()
    }

    /// The command palette: a query line above a fuzzy-filtered list of commands
    /// (each `key`, `title`, and `help`), the selected row inverted. Returns the
    /// on-screen rect of every listed row so a click can pick it. Fixed width so
    /// the box doesn't jump as the query filters the list.
    pub fn render_command_palette(
        frame: &mut Frame,
        query: &str,
        rows: &[(String, String, String, String)],
        selected: usize,
    ) -> Vec<Rect> {
        let key_w = rows
            .iter()
            .map(|(k, ..)| k.chars().count())
            .max()
            .unwrap_or(1);
        // `Group: Title` in one column, aligned so the help lines up.
        let label = |group: &str, title: &str| format!("{group}: {title}");
        let label_w = rows
            .iter()
            .map(|(_, g, t, _)| label(g, t).chars().count())
            .max()
            .unwrap_or(0);

        let mut content: Vec<Line> = Vec::new();
        let mut query_line = vec![Span::styled(
            "❯ ",
            Style::default()
                .fg(palette::KEY)
                .add_modifier(Modifier::BOLD),
        )];
        // Caret at the end (the palette input only appends / backspaces).
        query_line.extend(input_box_spans(query, query.chars().count(), 24));
        content.push(Line::from(query_line));
        content.push(Line::default());

        if rows.is_empty() {
            content.push(Line::from(dim_span("  (no matching commands)")));
        }
        for (i, (key, group, title, help)) in rows.iter().enumerate() {
            let pad_k = " ".repeat(key_w.saturating_sub(key.chars().count()));
            let pad_l = " ".repeat(label_w.saturating_sub(label(group, title).chars().count()));
            if i == selected {
                content.push(Line::from(Span::styled(
                    format!("  {pad_k}{key}  {group}: {title}{pad_l}  {help} "),
                    Style::default()
                        .fg(palette::SELECT_FG)
                        .bg(palette::SELECT_BG)
                        .add_modifier(Modifier::BOLD),
                )));
            } else {
                content.push(Line::from(vec![
                    Span::raw(format!("  {pad_k}")),
                    Span::styled(
                        key.clone(),
                        Style::default()
                            .fg(palette::KEY)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("  "),
                    // Category dimmed, command title normal — the VS Code look.
                    dim_span(format!("{group}: ")),
                    Span::raw(format!("{title}{pad_l}  ")),
                    dim_span(help.clone()),
                ]));
            }
        }
        content.push(Line::default());
        content.push(Line::from(dim_span(
            "↑/↓ move · Enter run · type to filter · Esc close",
        )));

        let width = (frame.area().width as usize)
            .saturating_sub(4)
            .clamp(30, 100);
        let inner = render_popup_box(frame, "Commands", content, Backdrop::Float, Some(width));
        // Rows start after the query line (0) and the blank separator (1).
        (0..rows.len())
            .map(|i| Rect {
                x: inner.x,
                y: inner.y + 2 + i as u16,
                width: inner.width,
                height: 1,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tests_support::strip_ansi_codes;

    #[test]
    fn command_palette_lists_commands_as_group_colon_title() {
        let rows = vec![
            (
                "c".to_string(),
                "Copy".to_string(),
                "Screen text".to_string(),
                "Copy the whole screen".to_string(),
            ),
            (
                "s".to_string(),
                "View".to_string(),
                "Checkpoint stats".to_string(),
                "Show stats".to_string(),
            ),
        ];
        let out = crate::tui::headless_render(90, 16, |f| {
            UI::render_command_palette(f, "cop", &rows, 0);
        })
        .unwrap();
        assert!(out.contains("Commands"), "titled box:\n{out}");
        // VS Code style: `Group: Title`, with the bound key beside it.
        assert!(out.contains("Copy: Screen text"), "{out}");
        assert!(out.contains("View: Checkpoint stats"), "{out}");
        assert!(out.contains("c  Copy: Screen text"), "key shown:\n{out}");
        // The query is echoed in the input line.
        assert!(out.contains("cop"), "query shown:\n{out}");
    }

    #[test]
    fn render_confirm_popup_shows_summary_and_choices() {
        let body = vec![
            "Rename 3 tensor(s) across 1 shard file(s):".to_string(),
            "Headers are rewritten in place — this cannot be undone.".to_string(),
        ];
        let out = crate::tui::headless_render(90, 20, |f| {
            UI::render_confirm_popup(f, "Apply rename in place?", &body, &["Apply", "Cancel"], 1);
        })
        .unwrap();
        let plain = strip_ansi_codes(&out);
        assert!(plain.contains("Apply rename in place?"), "{plain}");
        assert!(plain.contains("cannot be undone"), "{plain}");
        assert!(
            plain.contains("Apply") && plain.contains("Cancel"),
            "{plain}"
        );
    }
}
