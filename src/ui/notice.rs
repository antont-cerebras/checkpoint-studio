//! Transient overlays: the loading and progress screens, messages, the copy
//! confirmation flash and the command/export bands.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use super::UI;
use super::palette;
// Only the (hdf5-gated) conversion progress screen draws a gauge.
#[cfg(feature = "hdf5")]
use super::detail::render_line_gauge;
use super::popup::{Backdrop, render_popup_box, render_titled_bar};
use super::theme::{dim_span, key_span};

impl UI {
    /// Composite the copied-CLI-command pop-up over the live frame — a full-width
    /// [`render_titled_bar`] (label + copied confirmation ride the top border) with
    /// the wrapped command flush at column 0 so it stays cleanly selectable, then a
    /// dismiss hint.
    pub fn render_command_band(frame: &mut Frame, command: &str) {
        let term_w = frame.area().width as usize;
        let title = Line::from(vec![
            Span::styled(
                " CLI command ",
                Style::default()
                    .fg(palette::KEY)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "✓ copied to the clipboard ",
                Style::default().fg(palette::SUCCESS),
            ),
        ]);
        // The command, soft-wrapped at full width onto its own line(s), flush at
        // column 0 so it can still be selected cleanly by hand when the OSC-52
        // copy doesn't reach the terminal.
        let chars: Vec<char> = command.chars().collect();
        let cmd_rows = chars.len().div_ceil(term_w.max(1)).max(1);
        let mut content: Vec<Line> = (0..cmd_rows)
            .map(|r| {
                let seg: String = chars.iter().skip(r * term_w).take(term_w).collect();
                Line::from(Span::raw(seg))
            })
            .collect();
        content.push(Line::from(dim_span("click or press any key to dismiss")));
        render_titled_bar(frame, title, content);
    }

    /// The Ratatui port of [`Self::draw_loading`]: the tree browser's title + rule
    /// header, a spinner on the row where the tree's first node will land, and the
    /// cancel hint pinned to the bottom — so the chrome is up immediately and the
    /// tree fills into the same frame once the read finishes.
    pub fn render_loading(
        frame: &mut Frame,
        file: &str,
        total_files: usize,
        spinner: char,
        elapsed: std::time::Duration,
    ) {
        let area = frame.area();
        let width = area.width as usize;
        let height = area.height;

        // Title (row 0), with the same "+N more" note for a multi-file load.
        let mut title = vec![Span::raw(format!("Checkpoint Studio - {file}"))];
        if total_files > 1 {
            title.push(dim_span(format!("  (+{} more)", total_files - 1)));
        }
        // Full-width rule (row 1).
        let mut lines: Vec<Line> = vec![
            Line::from(title),
            Line::from(dim_span("─".repeat(width))),
            Line::default(),
        ];
        // The spinner lands on the row where the tree's first node will (row 3,
        // clamped). Rows above it are blank spacers added above.
        let spinner_row = 3u16.min(height.saturating_sub(2));
        for _ in lines.len() as u16..spinner_row {
            lines.push(Line::default());
        }
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(
                format!("{spinner} reading checkpoint structure"),
                Style::default().fg(palette::ACCENT),
            ),
            dim_span(format!("  ({:.1}s)", elapsed.as_secs_f64())),
        ]));
        Paragraph::new(lines).render(area, frame.buffer_mut());

        // Footer hint pinned to the bottom row.
        Paragraph::new(Line::from(vec![
            dim_span("Press "),
            key_span("q"),
            dim_span(" to cancel"),
        ]))
        .render(
            Rect {
                x: 0,
                y: height.saturating_sub(1),
                width: area.width,
                height: 1,
            },
            frame.buffer_mut(),
        );
    }

    /// The Ratatui port of [`Self::draw_progress`]: a full-screen progress view
    /// with a 40-cell bar, a `done/total` count and a detail line (e.g. the
    /// dataset currently being written).
    #[cfg(feature = "hdf5")]
    pub fn render_progress(
        frame: &mut Frame,
        title: &str,
        done: usize,
        total: usize,
        detail: &str,
    ) {
        let frac = if total > 0 {
            done as f64 / total as f64
        } else {
            0.0
        };
        let area = frame.area();
        // Title + rule on rows 0–1; a blank row 2; the gauge on row 3; the detail
        // line on row 4 — same layout as before, but the bar is a native LineGauge.
        Paragraph::new(vec![
            Line::from(Span::raw(title.to_string())),
            Line::from(Span::raw("=".repeat(title.len().max(10)))),
        ])
        .render(area, frame.buffer_mut());
        if area.height > 3 {
            render_line_gauge(
                frame,
                Rect {
                    x: 0,
                    y: 3,
                    width: area.width,
                    height: 1,
                },
                Line::from(format!("{done}/{total}")),
                frac,
                None,
            );
        }
        if area.height > 4 {
            Paragraph::new(Line::from(dim_span(detail.to_string()))).render(
                Rect {
                    x: 0,
                    y: 4,
                    width: area.width,
                    height: 1,
                },
                frame.buffer_mut(),
            );
        }
    }

    /// The Ratatui port of [`Self::draw_message`]: a simple full-screen message
    /// (title, underline rule, body, footer) over the pop-up panel surface.
    pub fn render_message(frame: &mut Frame, title: &str, message: &str) {
        render_popup_box(
            frame,
            title,
            vec![
                Line::from(Span::raw(message.to_string())),
                Line::default(),
                Line::from(dim_span("Click or press any key to return...")),
            ],
            Backdrop::Fill,
            None,
        );
    }

    /// A metadata-only / unavailable notice **floated over** the live frame (the
    /// screen behind stays visible — unlike [`Self::render_message`]), dismissed by
    /// any key. Used for [`Overlay::Notice`].
    pub fn render_notice_box(frame: &mut Frame, message: &str) {
        render_popup_box(
            frame,
            "Metadata-only",
            vec![
                Line::from(Span::raw(message.to_string())),
                Line::default(),
                Line::from(dim_span("Click or press any key to dismiss")),
            ],
            Backdrop::Float,
            None,
        );
    }

    /// Borderless band shown when a chosen export is too big for the terminal
    /// clipboard: it copies the concrete CLI command that reproduces it instead
    /// and shows it on its own full-width line(s) at column 0 (so a long path
    /// stays selectable even past the terminal width). Mirrors
    /// [`Self::render_command_band`].
    pub fn render_export_band(frame: &mut Frame, command: &str) {
        let term_w = (frame.area().width as usize).max(1);
        let title = Line::from(vec![
            Span::styled(
                " Too large to copy ",
                Style::default()
                    .fg(palette::KEY)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "— export command copied to the clipboard ",
                Style::default().fg(palette::SUCCESS),
            ),
        ]);
        // The command, soft-wrapped at full width onto its own line(s), flush at
        // column 0 so it stays selectable by hand when OSC-52 can't reach the
        // terminal.
        let chars: Vec<char> = command.chars().collect();
        let cmd_rows = chars.len().div_ceil(term_w).max(1);
        let mut content: Vec<Line> = (0..cmd_rows)
            .map(|r| {
                let seg: String = chars.iter().skip(r * term_w).take(term_w).collect();
                Line::from(Span::raw(seg))
            })
            .collect();
        content.push(Line::from(dim_span(
            "run it to export  ·  any key dismisses",
        )));
        render_titled_bar(frame, title, content);
    }

    /// Draw the copied CLI command as a borderless pop-up *over* the current
    /// screen (the surrounding view stays visible above and below the band; the
    /// caller redraws it on dismiss — the screen is not cleared). The command
    /// sits on its **own line at column 0**, bracketed by horizontal rules but
    /// with nothing before or after it on its row(s), so it can be selected
    /// cleanly with the mouse or a multiplexer's copy mode — important when the
    /// OSC-52 clipboard copy doesn't reach the terminal and it must be copied by
    /// hand. The terminal soft-wraps a long command, but it stays one logical
    /// line, so the selection still yields the whole command.
    /// Flash a "✓ Copied … to the clipboard" confirmation on the bottom line,
    /// over whatever the view drew there, until the next redraw clears it. Shared
    /// by every screen's copy shortcuts (tree, detail, data) so the confirmation
    /// never hides the content above it. `what` names what was copied.
    /// The Ratatui port of [`Self::draw_copied_flash`]: a bold green "✓ Copied …"
    /// confirmation composited over the frame's bottom row (clamped to the width
    /// so it never wraps and scrolls). Drawn last, over the live detail/data
    /// frame, so the content above it stays put.
    pub fn render_copied_flash(frame: &mut Frame, what: &str) {
        let area = frame.area();
        let width = area.width as usize;
        // The caller supplies the whole message (not just clipboard copies) — e.g.
        // "copied the screen to the clipboard" or "statistics already computed".
        let full = format!("✓ {what}");
        let msg: String = if full.chars().count() > width {
            full.chars()
                .take(width.saturating_sub(1))
                .chain(std::iter::once('…'))
                .collect()
        } else {
            full
        };
        Paragraph::new(Line::from(Span::styled(
            msg,
            Style::default()
                .fg(palette::SUCCESS)
                .add_modifier(Modifier::BOLD),
        )))
        .render(
            Rect {
                x: 0,
                y: area.height.saturating_sub(1),
                width: area.width,
                height: 1,
            },
            frame.buffer_mut(),
        );
    }

    /// Flash a transient warning `msg` on the bottom line (over whatever the view
    /// drew there), until the next redraw clears it — e.g. the wrong-keyboard-layout
    /// hint. Bold yellow, clamped to the width so it never wraps.
    pub fn render_notice(frame: &mut Frame, msg: &str) {
        let area = frame.area();
        let width = area.width as usize;
        let text: String = if msg.chars().count() > width {
            msg.chars()
                .take(width.saturating_sub(1))
                .chain(std::iter::once('…'))
                .collect()
        } else {
            msg.to_string()
        };
        Paragraph::new(Line::from(Span::styled(
            text,
            Style::default()
                .fg(palette::WARN)
                .add_modifier(Modifier::BOLD),
        )))
        .render(
            Rect {
                x: 0,
                y: area.height.saturating_sub(1),
                width: area.width,
                height: 1,
            },
            frame.buffer_mut(),
        );
    }
}
