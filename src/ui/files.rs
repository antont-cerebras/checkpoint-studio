//! The file browser screen: the directory listing, its rows and its geometry.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use crate::filetree::IndexMembership;
use crate::utils::format_size;

use super::UI;
use super::badge::Badge;
use super::detail::{render_line_gauge, rounded_to_cells};
use super::hints::{chip_regions, close_button, files_hint_lines};
use super::palette;
use super::popup::render_scroll_popup;
use super::scroll::VScrollbar;
use super::text::truncate_keep_end;
use super::theme::UNINDEXED_MARK;
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

        // One geometry for every row, so the sizes read as a column instead of
        // trailing each name at whatever column it happens to end.
        let layout = RowLayout::of(rows, body_width as usize, interactive);
        let mut body: Vec<Line> = Vec::with_capacity(body_rows);
        for (idx, row) in rows.iter().enumerate().skip(scroll).take(body_rows) {
            body.push(file_row_line(row, idx == selected, &layout));
        }
        Paragraph::new(body).render(
            Rect {
                width: body_width,
                ..crate::ui::fit_rows(area, header_rows as u16, body_rows as u16)
            },
            frame.buffer_mut(),
        );

        // The proportional size bars, overlaid on the blank column each row reserved
        // for them. Drawn as widgets rather than as glyphs in the line because the
        // gauge is the app's one bar primitive — `render_line_gauge`, the same
        // `LineGauge` the repack and statistics bars use.
        for (idx, row) in rows.iter().enumerate().skip(scroll).take(body_rows) {
            if let (Some(x), Some(share)) = (layout.gauge_x(), row.size_share()) {
                render_line_gauge(
                    frame,
                    Rect {
                        x: x as u16,
                        y: header_rows as u16 + (idx - scroll) as u16,
                        width: layout.bar as u16 + 1,
                        height: 1,
                    },
                    Line::default(),
                    rounded_to_cells(share, layout.bar),
                    None,
                );
            }
        }

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

/// Cells the right-aligned size column occupies — `1023.9 PiB`, the widest thing
/// [`format_size`] can produce.
const SIZE_W: usize = 10;
/// Cells the proportional size bar occupies.
const BAR_W: usize = 12;
/// Blank cells between the name column and the size.
const NAME_GAP: usize = 2;
/// A floor for the name column, so a listing of short names still has one.
const MIN_NAME_END: usize = 14;

/// Where each file row's columns sit. Computed once per frame from every row, so
/// the sizes read as a column rather than trailing each name at whatever width it
/// happens to have.
struct RowLayout {
    /// The column names are padded to (or truncated at).
    name_end: usize,
    /// Cells reserved for the proportional bar, `0` when it isn't drawn.
    bar: usize,
}

impl RowLayout {
    /// Fit the columns to `rows` within `width`. The bar is dropped when there's no
    /// colour to carry it (`--plain`): the gauge distinguishes full from empty by
    /// *style*, so in a plain dump every row would print the same twelve cells and
    /// quietly claim every file is the same size.
    fn of(rows: &[crate::filetree::FileRow], width: usize, interactive: bool) -> Self {
        let bar = if interactive { BAR_W } else { 0 };
        let widest = rows.iter().map(Self::label_width).max().unwrap_or(0);
        // Leave the size column, the bar and a little of the note room; a very narrow
        // terminal squeezes the name rather than pushing the size off-screen.
        let cap = width
            .saturating_sub(NAME_GAP + SIZE_W + 1 + bar + 2)
            .max(MIN_NAME_END);
        Self {
            name_end: widest.clamp(MIN_NAME_END, cap),
            bar,
        }
    }

    /// Columns a row's marker + name want, unpadded.
    fn label_width(row: &crate::filetree::FileRow) -> usize {
        row.depth * 2 + 2 + row.name.chars().count() + usize::from(row.is_dir())
    }

    /// Cells a row at `depth` has for its name (after the indent and marker).
    fn name_room(&self, depth: usize) -> usize {
        self.name_end.saturating_sub(depth * 2 + 2)
    }

    /// Where a row's gauge is drawn, or `None` when there is no bar.
    ///
    /// One column *before* the bar itself: a [`ratatui::widgets::LineGauge`] reserves
    /// its first cell as the gap after its label (empty here), so starting on the
    /// blank cell between the size and the bar puts the drawn line exactly on the
    /// cells the row reserved for it.
    fn gauge_x(&self) -> Option<usize> {
        (self.bar > 0).then_some(self.name_end + NAME_GAP + SIZE_W)
    }
}

/// One file-browser row as a styled [`Line`]: a directory shows a fold arrow, its
/// name in the accent with a trailing `/`, and its file count; a file shows a kind
/// marker (a distinct glyph for openable checkpoints), its name coloured by kind,
/// and what the model reads out of it. Between the two sits the shared size column —
/// right-aligned, with a blank run after it that [`UI::render_files`] overlays the
/// proportional bar onto. `selected` draws the whole row in the selection colours
/// (via [`tree_span`], shared with the tensor tree).
fn file_row_line(
    row: &crate::filetree::FileRow,
    selected: bool,
    layout: &RowLayout,
) -> Line<'static> {
    use crate::filetree::{FileKind, FileRowKind};
    let room = layout.name_room(row.depth);
    // The marker, the name (truncated from the left, so a shard's number survives)
    // and the trailing note, per row kind — the columns after them are shared.
    let (marker, marker_color, name_color, name, note) = match row.kind {
        FileRowKind::Dir { expanded, files } => (
            if expanded { "▾" } else { "▸" },
            palette::ACCENT,
            palette::ACCENT,
            // The `/` is part of the name's width, hence one cell less to fill.
            format!("{}/", truncate_keep_end(&row.name, room.saturating_sub(1))),
            format!("{files} {}", if files == 1 { "file" } else { "files" }),
        ),
        FileRowKind::File { kind, shard, .. } => {
            // A checkpoint gets the tensor glyph (it opens into the tree) and the
            // amber dtype accent; JSON/text/other stay quiet, so the openable ones
            // stand out.
            let (marker, name_color) = match kind {
                FileKind::Checkpoint => ("▦", palette::DTYPE),
                FileKind::Json => ("{}", palette::META),
                FileKind::Text => ("·", Color::Reset),
                FileKind::Other => ("·", palette::DIM),
            };
            (
                marker,
                palette::DIM,
                name_color,
                truncate_keep_end(&row.name, room),
                // What the model reads out of this shard: sixteen equal-sized shards
                // are otherwise sixteen indistinguishable rows. The wording is core's,
                // shared with the browser through `shared/parity/format.json`.
                shard.map(|sh| sh.note()).unwrap_or_default(),
            )
        }
    };

    let indent = "  ".repeat(row.depth);
    let used = indent.chars().count() + 2 + name.chars().count();
    let mut s: Vec<Span> = vec![
        tree_span(selected, Color::Reset, indent),
        tree_span(selected, marker_color, marker),
        tree_span(selected, Color::Reset, " "),
        tree_span(selected, name_color, name),
        // Pad to the shared column, then the right-aligned size.
        tree_span(
            selected,
            Color::Reset,
            " ".repeat(layout.name_end.saturating_sub(used) + NAME_GAP),
        ),
        tree_span(
            selected,
            palette::DIM,
            format!("{:>width$}", format_size(row.size as usize), width = SIZE_W),
        ),
    ];
    if layout.bar > 0 {
        // Blank cells for the overlaid gauge — one of separation, then its width.
        s.push(tree_span(
            selected,
            Color::Reset,
            " ".repeat(1 + layout.bar),
        ));
    }
    if !note.is_empty() {
        s.push(tree_span(selected, palette::DIM, format!("  {note}")));
    }
    // A checkpoint file the index doesn't declare, with the same mark and vivid red
    // the tensor tree and the detail screen use for an unindexed tensor — a loader
    // following only the index will not read this file.
    if row.index_membership() == Some(IndexMembership::Unlisted) {
        let lead = if note.is_empty() { "  " } else { " · " };
        s.push(tree_span(
            selected,
            palette::UNINDEXED,
            format!("{lead}{UNINDEXED_MARK} not in the index"),
        ));
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
                    size_share: 1.0,
                    index: None,
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
                    size_share: 0.1,
                    index: None,
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
        let row = |name: &str, shard: Option<ShardTensors>, index| FileRow {
            depth: 0,
            name: name.into(),
            path: format!("/ckpt/{name}").into(),
            size: 3_900_000_000,
            kind: FileRowKind::File {
                kind: FileKind::Checkpoint,
                shard,
                size_share: 1.0,
                index,
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
                Some(IndexMembership::Listed),
            ),
            row(
                "codebooks.safetensors",
                Some(ShardTensors {
                    tensors: 1,
                    params: 1,
                    params_share: 0.0001,
                }),
                Some(IndexMembership::Unlisted),
            ),
            // Unattributed (a shard of some other checkpoint, or an unmatched tree):
            // the row keeps its old shape rather than claiming zero tensors.
            row("stranger.safetensors", None, None),
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

        // Only the exception is marked. A listed shard says nothing (sixteen "in the
        // index" notes would bury the one row that isn't), and neither does a file the
        // question can't apply to.
        let marks = out.matches("not in the index").count();
        assert_eq!(marks, 1, "only the unlisted row is marked:\n{out}");
        let marked = out
            .lines()
            .find(|l| l.contains("not in the index"))
            .unwrap_or_default();
        assert!(
            marked.contains("codebooks.safetensors"),
            "…and it's the unlisted one:\n{out}"
        );
        assert!(
            marked.contains(UNINDEXED_MARK),
            "with the same mark the tensor tree uses:\n{out}"
        );
    }

    /// Rows of wildly different name lengths and sizes, for the column assertions.
    /// The shares are chosen so flooring and rounding to whole cells disagree — see
    /// [`the_size_bar_is_proportional_and_dropped_without_colour`].
    fn sized_rows() -> Vec<crate::filetree::FileRow> {
        use crate::filetree::{FileKind, FileRow, FileRowKind};
        [
            ("model-00001-of-00016.safetensors", 4_000_000_000u64, 1.0),
            ("model-00002-of-00016.safetensors", 3_998_000_000, 0.9995),
            ("qscales.safetensors", 700_000_000, 0.175),
            ("codebooks.safetensors", 200_000_000, 0.05),
            ("a.safetensors", 40_000_000, 0.01),
        ]
        .into_iter()
        .map(|(name, size, size_share)| FileRow {
            depth: 1,
            name: name.into(),
            path: format!("/ckpt/{name}").into(),
            size,
            kind: FileRowKind::File {
                kind: FileKind::Checkpoint,
                shard: None,
                size_share,
                index: Some(IndexMembership::Listed),
            },
        })
        .collect()
    }

    #[test]
    fn sizes_land_in_one_column_whatever_the_name_length() {
        let badges = status_badges(AccessBadge::ReadOnly, None, false);
        let out = crate::tui::headless_render(110, 12, |f| {
            UI::render_files(f, "/ckpt", &sized_rows(), 0, 0, None, true, &badges, None);
        })
        .unwrap();
        // Every size ends at the same column — the point of the column.
        let ends: Vec<usize> = out
            .lines()
            .filter(|l| l.trim_start().starts_with('▦'))
            .map(|l| l.find(" GiB").or_else(|| l.find(" MiB")).unwrap() + 4)
            .collect();
        assert_eq!(ends.len(), sized_rows().len(), "every file row:\n{out}");
        assert!(
            ends.windows(2).all(|w| w[0] == w[1]),
            "sizes right-aligned to one column, got {ends:?}:\n{out}"
        );
    }

    #[test]
    fn the_size_bar_is_proportional_and_dropped_without_colour() {
        let rows = sized_rows();
        let badges = status_badges(AccessBadge::ReadOnly, None, false);
        let buf = crate::tui::headless_buffer(110, 12, |f| {
            UI::render_files(f, "/ckpt", &rows, 0, 0, None, true, &badges, None);
        })
        .unwrap();

        // The gauge marks its filled part by colour alone, so count bar cells (the
        // gauge's own THICK glyph) by their colour rather than looking at the text.
        let bar_cells = |y: u16, fg: Color| -> usize {
            (0..110)
                .filter(|&x| buf[(x, y)].symbol() == "━" && buf[(x, y)].fg == fg)
                .count()
        };
        // Row 0 is the header, row 1 the rule, so the files start at y = 2. Cells are
        // the NEAREST whole number, not the truncation — the three assertions with a
        // "floor would" note are the ones that distinguish the two.
        assert_eq!(
            bar_cells(2, palette::KEY),
            BAR_W,
            "the largest file fills it"
        );
        assert_eq!(
            bar_cells(3, palette::KEY),
            BAR_W,
            "0.05% smaller is not a whole cell smaller (floor would say {})",
            BAR_W - 1
        );
        assert_eq!(bar_cells(4, palette::KEY), 2, "17.5% of {BAR_W} cells");
        assert_eq!(
            bar_cells(5, palette::KEY),
            1,
            "5% of {BAR_W} cells is over half a cell (floor would say 0)"
        );
        assert_eq!(
            bar_cells(6, palette::KEY),
            0,
            "1% rounds to nothing — which is the truth"
        );
        // …and the rest of each bar is drawn dim rather than left blank.
        assert_eq!(
            bar_cells(6, palette::DIM),
            BAR_W,
            "the empty bar is still a bar"
        );

        // Without colour (`--plain`) the bar would be BAR_W identical cells on every
        // row, claiming every file is the same size — so it isn't drawn at all.
        let plain = crate::tui::headless_buffer(110, 12, |f| {
            UI::render_files(f, "/ckpt", &rows, 0, 0, None, false, &badges, None);
        })
        .unwrap();
        assert!(
            (0..110).all(|x| plain[(x, 2)].symbol() != "━"),
            "no gauge in a plain render"
        );
    }
}
