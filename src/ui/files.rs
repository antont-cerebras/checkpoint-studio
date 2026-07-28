//! The file browser screen: the directory listing, its rows and its geometry.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use crate::utils::format_size;

use super::UI;
use super::badge::Badge;
use super::hints::{chip_regions, close_button, files_hint_lines};
use super::palette;
use super::popup::render_scroll_popup;
use super::scroll::VScrollbar;
use super::text::truncate_keep_end;
use super::theme::tree_span;

/// Footer rows below the file-browser list: a one-line status bar (the selected
/// entry's path / size, or a copy confirmation).
const FILES_FOOTER_HEIGHT: usize = 1;

impl UI {
    /// The first terminal row of the file browser's body — its header height
    /// (title + separator rule; the key hints are a bottom-pinned footer now).
    /// Shared with the mouse handler so a click at row `r >= this` maps to file row
    /// `scroll + (r - this)`.
    pub(crate) fn files_header_rows(_width: u16) -> usize {
        2 // title + rule
    }

    /// Rows the bottom-pinned key-hint footer occupies (above the one-line status
    /// bar). Kept in sync with [`Self::render_files`] so scroll / hit-testing align.
    pub(crate) fn files_hint_rows(width: u16) -> usize {
        files_hint_lines(width).0.len()
    }

    /// Body rows visible in the file browser at the given size (header + the
    /// bottom-pinned hint footer + the one-line status bar), so the scroll offset
    /// stays consistent with [`Self::render_files`]'s layout.
    pub(crate) fn files_visible_rows(width: u16, height: u16) -> usize {
        (height as usize)
            .saturating_sub(
                Self::files_header_rows(width) + Self::files_hint_rows(width) + FILES_FOOTER_HEIGHT,
            )
            .max(1)
    }

    /// How many file rows fit one screenful — used to size a PageUp/PageDown jump.
    pub(crate) fn visible_file_rows(width: u16, height: u16) -> usize {
        Self::files_visible_rows(width, height)
    }

    /// The file browser's scroll-bar geometry (reusing [`VScrollbar`]) for this
    /// size and a listing of `total` rows at `offset`, or `None` when it all fits.
    pub(crate) fn files_scrollbar(
        width: u16,
        height: u16,
        total: usize,
        offset: usize,
    ) -> Option<VScrollbar> {
        let rows = Self::files_visible_rows(width, height);
        VScrollbar::for_body(
            Rect {
                x: 0,
                y: Self::files_header_rows(width) as u16,
                width,
                height: rows as u16,
            },
            total,
            offset,
        )
    }

    /// Render the file browser: header (title, hint line(s), rule), the visible
    /// file rows from `scroll`, a scroll bar when the listing overflows, and a
    /// one-line status bar showing the selected entry's path (or a copy
    /// confirmation). Returns the clickable footer chips + `[×]` close, like
    /// [`Self::render_tree`].
    // A flat render signature (frame + view state) — a config struct would just
    // move the same fields behind one more indirection for no clarity.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render_files(
        frame: &mut Frame,
        root: &str,
        rows: &[crate::filetree::FileRow],
        selected: usize,
        scroll: usize,
        copied_flash: Option<&str>,
        interactive: bool,
        badges: &[Badge],
        hovered_badge: Option<usize>,
    ) -> Vec<(Rect, KeyEvent)> {
        let area = frame.area();
        let (width, height) = (area.width, area.height);
        if height < (FILES_FOOTER_HEIGHT as u16 + 1) {
            return Vec::new();
        }

        // --- header (title + rule); the key hints are a bottom-pinned footer ---
        let lines: Vec<Line> = vec![
            Line::from(Span::raw(format!("File browser - {root}"))),
            Line::from(Span::styled(
                "─".repeat(width as usize),
                Style::default().fg(palette::DIM),
            )),
        ];
        let (hint_lines, chips) = files_hint_lines(width);
        let hint_rows = hint_lines.len();

        let header_rows = lines.len();
        let body_rows =
            (height as usize).saturating_sub(header_rows + hint_rows + FILES_FOOTER_HEIGHT);

        // The bar is drawn by the engine (via `FilesMode::vscrollbar`); reserve its
        // column here so long rows don't underlap it (live TUI only).
        let scrollbar =
            interactive && Self::files_scrollbar(width, height, rows.len(), scroll).is_some();
        let body_width = width.saturating_sub(u16::from(scrollbar));

        Paragraph::new(lines).render(
            crate::ui::fit_rows(area, 0, header_rows as u16),
            frame.buffer_mut(),
        );

        let mut body: Vec<Line> = Vec::with_capacity(body_rows);
        for (idx, row) in rows.iter().enumerate().skip(scroll).take(body_rows) {
            body.push(file_row_line(row, idx == selected));
        }
        Paragraph::new(body).render(
            Rect {
                width: body_width,
                ..crate::ui::fit_rows(area, header_rows as u16, body_rows as u16)
            },
            frame.buffer_mut(),
        );

        // (The scroll bar itself is drawn by the engine — see `render_vscrollbar`.)

        // --- key-hint footer, pinned just above the one-line status bar ---
        let hint_y = height.saturating_sub(1 + hint_rows as u16);
        Paragraph::new(hint_lines).render(
            crate::ui::fit_rows(area, hint_y, hint_rows as u16),
            frame.buffer_mut(),
        );

        // --- one-line status bar (selected entry, or a copy confirmation) ---
        let reserve = Self::badge_bar_width(badges) as usize;
        let max_text = (width as usize).saturating_sub(6 + reserve);
        // A copy confirmation wins the bar while it lasts; otherwise the selected path,
        // or nothing when the listing is empty. Written as three cases in the order they
        // take precedence — a `map_or_else` pair would nest the second two inside a
        // closure and read backwards.
        // Three cases in precedence order (a flash, then the content, then a fallback).
        // `option_if_let_else` wants `map_or_else` for the outer test, which would nest the
        // other two inside a closure and put the fallback ahead of the common case — every
        // shape the lint accepts here reads worse than this one.
        #[allow(clippy::option_if_let_else)]
        let status = if let Some(flash) = copied_flash {
            Line::from(Span::styled(
                flash.to_string(),
                Style::default()
                    .fg(palette::SUCCESS)
                    .add_modifier(Modifier::BOLD),
            ))
        } else if let Some(row) = rows.get(selected) {
            let text = truncate_keep_end(&row.path.to_string_lossy(), max_text);
            Line::from(Span::styled(
                format!(" ▪ {text} "),
                Style::default()
                    .bg(palette::STATUS_BG)
                    .fg(palette::STATUS_FG),
            ))
        } else {
            Line::default()
        };
        Paragraph::new(status).render(
            Rect {
                x: 0,
                y: height.saturating_sub(1),
                width,
                height: 1,
            },
            frame.buffer_mut(),
        );
        Self::render_badge_bar(frame, badges, hovered_badge);

        // Clickable footer chips (the bottom-pinned hint block, at `hint_y`)
        // plus the top-right `[×]` (→ switch back to the tensor tree).
        let mut regions = chip_regions(&chips, hint_y);
        regions.extend(close_button(
            frame,
            KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
        ));
        regions
    }

    /// Float a sidecar file preview over the file browser: a scrollable pop-up of
    /// the file's contents (JSON syntax-highlighted, other text plain) or an info
    /// line for a binary. Reuses the scroll-pop-up chrome; returns its max scroll
    /// and clickable regions so the caller can clamp/handle them.
    pub(crate) fn render_file_preview(
        frame: &mut Frame,
        title: &str,
        body: &[Line<'static>],
        footer: Line<'static>,
        scroll: usize,
    ) -> (usize, Vec<(Rect, KeyEvent)>) {
        render_scroll_popup(frame, title, body, footer, scroll, &[])
    }
}

/// One file-browser row as a styled [`Line`]: a directory shows a fold arrow, its
/// name in the accent with a trailing `/`, and a dim size + file count; a file
/// shows a kind marker (a distinct glyph for openable checkpoints), its name
/// coloured by kind, and its dim size. `selected` draws the whole row in the
/// selection colours (via [`tree_span`], shared with the tensor tree).
fn file_row_line(row: &crate::filetree::FileRow, selected: bool) -> Line<'static> {
    use crate::filetree::{FileKind, FileRowKind};
    let indent = "  ".repeat(row.depth);
    let mut s: Vec<Span> = vec![tree_span(selected, Color::Reset, indent)];
    match row.kind {
        FileRowKind::Dir { expanded, files } => {
            let arrow = if expanded { "▾" } else { "▸" };
            s.push(tree_span(selected, palette::ACCENT, arrow));
            s.push(tree_span(selected, Color::Reset, " "));
            s.push(tree_span(
                selected,
                palette::ACCENT,
                format!("{}/", row.name),
            ));
            s.push(tree_span(
                selected,
                palette::DIM,
                format!(
                    "  {} · {files} {}",
                    format_size(row.size as usize),
                    if files == 1 { "file" } else { "files" }
                ),
            ));
        }
        FileRowKind::File { kind, shard } => {
            // A checkpoint gets the tensor glyph (it opens into the tree) and the
            // amber dtype accent; JSON/text/other stay quiet, so the openable ones
            // stand out.
            let (marker, name_color) = match kind {
                FileKind::Checkpoint => ("▦", palette::DTYPE),
                FileKind::Json => ("{}", palette::META),
                FileKind::Text => ("·", Color::Reset),
                FileKind::Other => ("·", palette::DIM),
            };
            s.push(tree_span(selected, palette::DIM, marker));
            s.push(tree_span(selected, Color::Reset, " "));
            s.push(tree_span(selected, name_color, row.name.clone()));
            s.push(tree_span(
                selected,
                palette::DIM,
                format!("  {}", format_size(row.size as usize)),
            ));
            // What the model reads out of this shard: sixteen equal-sized shards are
            // otherwise sixteen indistinguishable rows. The wording is core's, shared
            // with the browser's row through `shared/parity/format.json`.
            if let Some(sh) = shard {
                s.push(tree_span(
                    selected,
                    palette::DIM,
                    format!(" · {}", sh.note()),
                ));
            }
        }
    }
    Line::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::{AccessBadge, status_badges};

    #[test]
    fn file_browser_shows_dirs_files_and_footer() {
        use crate::filetree::{FileKind, FileRow, FileRowKind};
        let rows = vec![
            FileRow {
                depth: 0,
                name: "ckpt".into(),
                path: "/ckpt".into(),
                size: 100,
                kind: FileRowKind::Dir {
                    expanded: true,
                    files: 2,
                },
            },
            FileRow {
                depth: 1,
                name: "model.safetensors".into(),
                path: "/ckpt/model.safetensors".into(),
                size: 90,
                kind: FileRowKind::File {
                    kind: FileKind::Checkpoint,
                    shard: None,
                },
            },
            FileRow {
                depth: 1,
                name: "config.json".into(),
                path: "/ckpt/config.json".into(),
                size: 10,
                kind: FileRowKind::File {
                    kind: FileKind::Json,
                    shard: None,
                },
            },
        ];
        let badges = status_badges(AccessBadge::ReadOnly, None, false);
        let out = crate::tui::headless_render(90, 16, |f| {
            UI::render_files(f, "/ckpt", &rows, 1, 0, None, true, &badges, None);
        })
        .unwrap();
        assert!(out.contains("File browser - /ckpt"), "title:\n{out}");
        assert!(out.contains("ckpt/"), "dir row with trailing slash:\n{out}");
        assert!(out.contains("model.safetensors"), "checkpoint row:\n{out}");
        assert!(out.contains("config.json"), "json row:\n{out}");
        // The selected row's path lands on the status bar.
        assert!(
            out.contains("/ckpt/model.safetensors"),
            "status bar path:\n{out}"
        );
        // Footer advertises the Tab toggle back to the tensor tree.
        assert!(out.contains("tensor tree"), "footer hint:\n{out}");
    }

    #[test]
    fn a_shard_row_says_what_it_holds() {
        use crate::filetree::{FileKind, FileRow, FileRowKind, ShardTensors};
        let row = |name: &str, shard: Option<ShardTensors>| FileRow {
            depth: 0,
            name: name.into(),
            path: format!("/ckpt/{name}").into(),
            size: 3_900_000_000,
            kind: FileRowKind::File {
                kind: FileKind::Checkpoint,
                shard,
            },
        };
        let rows = vec![
            row(
                "model-00001-of-00016.safetensors",
                Some(ShardTensors {
                    tensors: 1062,
                    params: 641,
                    params_share: 0.0641,
                }),
            ),
            row(
                "codebooks.safetensors",
                Some(ShardTensors {
                    tensors: 1,
                    params: 1,
                    params_share: 0.0001,
                }),
            ),
            // Unattributed (a shard of some other checkpoint, or an unmatched tree):
            // the row keeps its old shape rather than claiming zero tensors.
            row("stranger.safetensors", None),
        ];
        let badges = status_badges(AccessBadge::ReadOnly, None, false);
        let out = crate::tui::headless_render(110, 12, |f| {
            UI::render_files(f, "/ckpt", &rows, 0, 0, None, true, &badges, None);
        })
        .unwrap();
        assert!(
            out.contains("1062 tensors · 6.4% of params"),
            "shard:\n{out}"
        );
        // Singular, and a share too small for one decimal — scientific, not "0.0%".
        assert!(
            out.contains("1 tensor · 1.0e-2% of params"),
            "codebooks:\n{out}"
        );
        assert!(
            !out.contains("0 tensors"),
            "an unattributed row claims nothing:\n{out}"
        );
    }
}
