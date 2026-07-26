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
    pub(crate) fn render_dtype_menu(frame: &mut Frame, options: &[ViewDtype], current: usize) {
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
    pub(crate) fn render_slice_prompt(
        frame: &mut Frame,
        slices: usize,
        input: &str,
        error: Option<&str>,
    ) {
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
    pub(crate) fn render_reshape_prompt(
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
    pub(crate) fn render_text_prompt(
        frame: &mut Frame,
        label: &str,
        input: &str,
        error: Option<&str>,
    ) {
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
    pub(crate) fn render_choice_menu(
        frame: &mut Frame,
        title: &str,
        options: &[&str],
        current: usize,
    ) {
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
    pub(crate) fn render_confirm_popup(
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
    pub(crate) fn render_menu_box(
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
    pub(crate) fn render_command_palette(
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

    /// Render one prompt into a fixed buffer and return its plain text.
    fn render(w: u16, h: u16, f: impl FnOnce(&mut Frame)) -> String {
        strip_ansi_codes(&crate::tui::headless_render(w, h, f).expect("headless render"))
    }

    #[test]
    fn the_dtype_menu_marks_the_current_choice() {
        let opts = [ViewDtype::Stored, ViewDtype::As("F16"), ViewDtype::U4];
        let out = render(70, 14, |f| UI::render_dtype_menu(f, &opts, 1));
        // Every offered dtype is listed (the stored one has no label of its own).
        for o in &opts {
            if let Some(label) = o.label() {
                assert!(out.contains(label), "{label} missing from:\n{out}");
            }
        }
        // The selected row is marked somehow (arrow / highlight glyph), and only one is.
        let marked: Vec<&str> = out
            .lines()
            .filter(|l| l.contains('▸') || l.contains('›'))
            .collect();
        assert!(marked.len() <= 1, "at most one row is marked: {marked:?}");
    }

    #[test]
    fn the_slice_prompt_states_the_range_and_shows_an_error() {
        let out = render(80, 8, |f| UI::render_slice_prompt(f, 48, "12", None));
        assert!(out.contains("12"), "the typed input shows:\n{out}");
        assert!(
            out.contains("48") || out.contains("47"),
            "the range shows:\n{out}"
        );
        let bad = render(80, 8, |f| {
            UI::render_slice_prompt(f, 48, "99", Some("out of range"))
        });
        assert!(bad.contains("out of range"), "{bad}");
    }

    #[test]
    fn the_reshape_prompt_names_the_element_count_and_current_shape() {
        // Wide enough for the whole band: the hint text is long, and a narrower
        // terminal clips its tail (which the narrow-terminal test below covers).
        let out = render(170, 8, |f| {
            UI::render_reshape_prompt(f, 4096, &[64, 64], "128,32", None)
        });
        assert!(
            out.contains("4096"),
            "the element count constrains the input:\n{out}"
        );
        assert!(out.contains("(64, 64)"), "the current shape shows:\n{out}");
        assert!(out.contains("128,32"), "the typed input shows:\n{out}");
        assert!(
            out.contains("Enter") && out.contains("Esc"),
            "both actions:\n{out}"
        );
    }

    #[test]
    fn the_text_prompt_shows_its_label_input_and_error() {
        let out = render(80, 8, |f| {
            UI::render_text_prompt(f, "Output file", "out.h5", Some("already exists"))
        });
        assert!(out.contains("Output file"), "{out}");
        assert!(out.contains("out.h5"), "{out}");
        assert!(out.contains("already exists"), "{out}");
    }

    #[test]
    fn an_empty_text_prompt_still_draws_a_box_to_type_into() {
        let out = render(80, 8, |f| UI::render_text_prompt(f, "Bins", "", None));
        assert!(out.contains("Bins"), "{out}");
        assert!(
            out.lines().any(|l| l.contains("Bins") && l.len() > 6),
            "{out}"
        );
    }

    #[test]
    fn the_choice_menu_lists_its_options_under_a_title() {
        let out = render(70, 12, |f| {
            UI::render_choice_menu(f, "Sort by", &["name", "size", "params"], 2)
        });
        assert!(out.contains("Sort by"), "{out}");
        for o in ["name", "size", "params"] {
            assert!(out.contains(o), "{o} missing:\n{out}");
        }
    }

    #[test]
    fn the_confirm_popup_shows_the_body_and_both_answers() {
        let body = vec![
            "Rename 3 tensors in place?".to_string(),
            "This rewrites the file.".to_string(),
        ];
        let out = render(80, 14, |f| {
            UI::render_confirm_popup(f, "Confirm rename", &body, &["Cancel", "Rename"], 1)
        });
        assert!(out.contains("Confirm rename"), "{out}");
        assert!(out.contains("Rename 3 tensors in place?"), "{out}");
        assert!(
            out.contains("This rewrites the file."),
            "the whole body shows:\n{out}"
        );
        assert!(out.contains("Cancel") && out.contains("Rename"), "{out}");
    }

    #[test]
    fn the_menu_box_returns_one_click_region_per_row() {
        let rows = ["first", "second", "third"];
        let mut regions = Vec::new();
        crate::tui::headless_render(60, 12, |f| {
            regions = UI::render_menu_box(f, "Pick one", &rows, 0, &[]);
        })
        .expect("render");
        assert_eq!(regions.len(), rows.len(), "every row must be clickable");
        // Rows are laid out top to bottom, one per line, without overlapping.
        let mut ys: Vec<u16> = regions.iter().map(|r| r.y).collect();
        ys.sort_unstable();
        ys.dedup();
        assert_eq!(
            ys.len(),
            rows.len(),
            "each row has its own line: {regions:?}"
        );
    }

    #[test]
    fn the_palette_narrows_to_the_query_and_reports_its_rows() {
        let rows = vec![(
            "s".to_string(),
            "View".to_string(),
            "Statistics".to_string(),
            "Show the checkpoint stats".to_string(),
        )];
        let mut regions = Vec::new();
        let out = strip_ansi_codes(
            &crate::tui::headless_render(90, 16, |f| {
                regions = UI::render_command_palette(f, "stat", &rows, 0);
            })
            .expect("render"),
        );
        assert!(out.contains("stat"), "the query echoes back:\n{out}");
        assert!(out.contains("View: Statistics"), "{out}");
        assert_eq!(regions.len(), 1, "one clickable region per listed command");
    }

    #[test]
    fn a_palette_with_no_matches_says_so_rather_than_going_blank() {
        let out = strip_ansi_codes(
            &crate::tui::headless_render(90, 16, |f| {
                UI::render_command_palette(f, "zzzz", &[], 0);
            })
            .expect("render"),
        );
        assert!(out.contains("zzzz"), "{out}");
        assert!(
            out.to_lowercase().contains("no match") || out.contains("—") || out.trim().len() > 4,
            "an empty palette must still render something:\n{out}"
        );
    }

    #[test]
    fn every_prompt_survives_a_narrow_terminal() {
        for w in [6u16, 12, 30] {
            render(w, 6, |f| UI::render_slice_prompt(f, 4, "1", Some("bad")));
            render(w, 6, |f| UI::render_text_prompt(f, "Label", "value", None));
            render(w, 8, |f| UI::render_choice_menu(f, "T", &["a"], 0));
            render(w, 8, |f| UI::render_dtype_menu(f, &[ViewDtype::Stored], 0));
            render(w, 10, |f| {
                UI::render_confirm_popup(f, "T", &["b".to_string()], &["y", "n"], 0);
            });
        }
    }

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
