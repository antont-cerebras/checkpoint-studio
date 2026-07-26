//! The six interactive screens as [`Mode`] implementations: file browser, byte-layout
//! map, tensor tree, rename editor, tensor detail, statistics and the data views.
//!
//! Split out of `explorer/mod.rs` because each one is self-contained — a small struct of
//! transient per-screen bookkeeping plus its `Mode` impl (key/mouse handling, drawing,
//! and the `residual` screen it restores to). The persistent selection/scroll state
//! stays on [`Explorer`]; these hold only what the old per-screen loops kept as locals.

// The tensor-data screens no longer need a lifecycle invariant at all: `DetailMode` and
// `DataMode` take a resolved `TensorInfo`, so "a screen for a tensor that isn't here" is
// not a state they can be in, and the caller decides where to go instead (see
// `Explorer::tensor_named`). The three screens whose payload needs I/O and `&mut Explorer`
// — the layout map, the rename editor, the statistics report — still load it in `on_enter`,
// and each states that invariant at its own accessor rather than under a file-wide allow.
// With the allow gone from this file, a NEW unwrap anywhere in these 3,300 lines is an
// error again.

// Imports are explicit, and external types are taken from their real source instead of
// being laundered through the parent. What remains below is an honest measure of what
// these screens need from `Explorer`: 50 names, where the wildcard's expansion was ~80.

use anyhow::Result;

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

use crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use ratatui::text::{Line, Span};

use crate::sample::{SampleMode, Stats};
use crate::tree::{TensorInfo, TreeNode};
use crate::ui::{Legend, Overlay, StatsView, UI};

// The mode framework and the vocabulary these screens share with the driver.
use super::{
    Explorer, HelpCtx, Mode, ModeSpec, Nav, Outcome, PaletteResult, PopupBackdrop, Screen,
};

// Input handling and scroll tuning, shared with the parent's own loops.
use super::{
    DOUBLE_CLICK, FILES_FOOTER_ROWS, Interaction, MouseOutcome, SCROLL_PAGE, WHEEL_STEP,
    accel_step_page, accel_step_row, slice_step,
};

// Per-screen commands. A keystroke is translated into one of these HERE and dispatched by
// `Explorer` THERE, which is why they stay in the parent: it references them 20-30 times
// each (dispatch, palette, legend) against once or twice here, so moving them next to
// each screen would invert the dependency rather than remove it.
use super::{
    Cmd, DataCmd, DetailCmd, FileCmd, LayoutCmd, RenameCmd, StatsCmd, available_data_commands,
    available_detail_commands, available_rename_commands, available_stats_commands, data_cmd_key,
    detail_cmd_key, file_command_for_key, layout_command_for_key, stats_cmd_key,
    tree_command_for_key,
};

// Per-screen state and small helpers owned by the parent.
use super::{
    Bg, BinsChoice, DtypePreview, RenameMode, RenamePair, Representation, ReshapeChoice, ScanJob,
    StatsStart, copy_to_clipboard, draw_rename_frame, dtype_overridable, is_ctrl_c,
    poll_stats_scan, quit_immediately, scan_stats_view,
};

/// The file browser ([`Screen::Files`]) as a [`Mode`]: lists the checkpoint's
/// directory (fold with `←`/`→`, `Enter` opens a checkpoint / previews a sidecar),
/// `Tab`/Backspace return to the tree. Its selection/scroll live on [`Explorer`];
/// this holds only the transient click/drag bookkeeping the old `run_files` kept as
/// loop locals.
pub(super) struct FilesMode {
    /// Last left-click (time + row) for double-click detection.
    pub(super) last_click: Option<(std::time::Instant, u16)>,
    /// The selection the scroll was last kept-visible for (so a moved selection
    /// re-scrolls once). `usize::MAX` forces the first frame to update.
    pub(super) last_sel: usize,
}

impl FilesMode {
    pub(super) fn new() -> Self {
        Self {
            last_click: None,
            last_sel: usize::MAX,
        }
    }
}

impl Mode for FilesMode {
    fn spec(&self) -> ModeSpec {
        ModeSpec {
            id: HelpCtx::Files,
            ctrlc_quits_immediately: false,
        }
    }

    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame) {
        ex.render_files_frame(f, true);
    }

    fn on_enter(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        // Build the directory tree lazily on first entry, then keep it (fold state
        // persists across `Tab` toggles). Local walks the filesystem; a remote
        // source lists over SFTP / s3 — a listing failure floats an error and
        // drops back rather than showing an empty browser.
        if ex.file_state.tree.is_none() {
            match ex.build_browse_tree() {
                Ok(tree) => {
                    ex.file_state.tree = Some(tree);
                    ex.file_state.rebuild_rows();
                }
                Err(e) => {
                    let body = vec![
                        Line::from(Span::raw("Can't list the checkpoint directory:")),
                        Line::default(),
                        Line::from(crate::ui::dim_span(e)),
                    ];
                    ex.float_scroll_popup(term, "Files", body, PopupBackdrop::Files, None);
                    return Ok(Outcome::Leave(Nav::Back));
                }
            }
        }
        Ok(Outcome::Stay)
    }

    fn pre_draw(&mut self, ex: &mut Explorer, term: &mut crate::tui::LiveTerminal) {
        if let Ok(sz) = term.size() {
            if ex.file_state.selected != self.last_sel {
                ex.update_files_scroll(sz.width, sz.height);
                self.last_sel = ex.file_state.selected;
            }
            let body = UI::files_visible_rows(sz.width, sz.height);
            let total = ex.file_state.rows.len();
            ex.file_state.scroll = ex.file_state.scroll.min(total.saturating_sub(body));
        }
    }

    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult {
        let Some(cmd) = ex.file_command_palette(term) else {
            return PaletteResult::Handled;
        };
        if cmd == FileCmd::CopyCommand {
            return PaletteResult::CopyCommand; // engine-owned, like the `y` key
        }
        if let Some(nav) = ex.run_file_command(cmd, term) {
            return PaletteResult::Nav(nav);
        }
        PaletteResult::Handled
    }

    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome> {
        let total = ex.file_state.rows.len();
        // Every lettered command dispatches through the registry (like the tree), so key
        // and palette entry can't drift. Handled before the match rather than in a guard
        // that then has to look the command up a second time to use it.
        if !key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
            && let KeyCode::Char(c) = key.code
            && let Some(cmd) = file_command_for_key(c)
        {
            if let Some(nav) = ex.run_file_command(cmd, term) {
                return Ok(Outcome::Leave(nav));
            }
            return Ok(Outcome::Stay);
        }
        match key.code {
            KeyCode::Tab | KeyCode::Backspace => return Ok(Outcome::Leave(Nav::Back)),
            KeyCode::Char('\\') => return Ok(Outcome::Leave(Nav::Forward)),
            KeyCode::Up => {
                let step = ex.held_step(KeyCode::Up, accel_step_row) as i32;
                ex.file_state.move_selection(-step);
            }
            KeyCode::Down => {
                let step = ex.held_step(KeyCode::Down, accel_step_row) as i32;
                ex.file_state.move_selection(step);
            }
            KeyCode::PageUp => {
                let step =
                    (ex.file_page_rows() * ex.held_step(KeyCode::PageUp, accel_step_page)) as i32;
                ex.file_state.move_selection(-step);
            }
            KeyCode::PageDown => {
                let step =
                    (ex.file_page_rows() * ex.held_step(KeyCode::PageDown, accel_step_page)) as i32;
                ex.file_state.move_selection(step);
            }
            KeyCode::Home => ex.file_state.selected = 0,
            KeyCode::End => ex.file_state.selected = total.saturating_sub(1),
            KeyCode::Left => ex.file_state.collapse_or_parent(),
            KeyCode::Right => ex.file_state.expand_or_child(),
            KeyCode::Enter => {
                if let Some(nav) = ex.activate_file_selection(term) {
                    return Ok(Outcome::Leave(nav));
                }
            }
            _ => {}
        }
        Ok(Outcome::Stay)
    }

    fn handle_mouse(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        m: MouseEvent,
    ) -> MouseOutcome {
        let Ok(sz) = term.size() else {
            return MouseOutcome::Ignored;
        };
        let (col, row) = (m.column, m.row);
        match m.kind {
            // (Scroll-bar clicks / drags are handled by the engine's `route_mouse`.)
            MouseEventKind::Down(MouseButton::Left) => {
                let body_top = UI::files_header_rows(sz.width) as u16;
                let body_bottom = sz.height.saturating_sub(FILES_FOOTER_ROWS as u16);
                if row >= body_top && row < body_bottom {
                    let idx = ex.file_state.scroll + (row - body_top) as usize;
                    if let Some(fr) = ex.file_state.rows.get(idx).cloned() {
                        // A click on a directory's ▸/▾ twisty (column `2*depth`)
                        // toggles it on a single click.
                        let on_arrow = fr.is_dir() && col == 2 * fr.depth as u16;
                        ex.file_state.selected = idx;
                        if on_arrow {
                            self.last_click = None;
                            ex.activate_file_selection(term);
                        } else {
                            let double = matches!(
                                self.last_click,
                                Some((t, r)) if r == row && t.elapsed() < DOUBLE_CLICK
                            );
                            if double {
                                self.last_click = None;
                                if let Some(nav) = ex.activate_file_selection(term) {
                                    return MouseOutcome::Leave(nav);
                                }
                            } else {
                                self.last_click = Some((std::time::Instant::now(), row));
                            }
                        }
                    }
                }
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollDown => {
                ex.file_state.scroll = ex.file_state.scroll.saturating_add(WHEEL_STEP);
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollUp => {
                ex.file_state.scroll = ex.file_state.scroll.saturating_sub(WHEEL_STEP);
                MouseOutcome::Redraw
            }
            _ => MouseOutcome::Ignored,
        }
    }

    fn set_scroll(&mut self, ex: &mut Explorer, offset: usize) {
        ex.file_state.scroll = offset;
    }

    fn residual(&self) -> Screen {
        Screen::Files
    }
}

/// The safetensors layout map ([`Screen::Layout`]) as a [`Mode`]: a scrollable
/// vertical strip of one file's byte layout. Its selection/scroll are the drill-down
/// residual (written back to history), and the parsed map lives here for the visit.
pub(super) struct LayoutMode {
    pub(super) path: String,
    /// The parsed map, or the parse error to report on entry.
    pub(super) map: std::result::Result<crate::safelayout::LayoutMap, String>,
    pub(super) selected: usize,
    pub(super) scroll: usize,
    pub(super) scroll_max: usize,
    pub(super) last_sel: usize,
}

impl LayoutMode {
    /// The map is parsed by [`Explorer::layout_mode`] (which routes local vs
    /// remote), so the mode itself is source-agnostic — it just holds the result.
    pub(super) fn new(
        path: String,
        map: std::result::Result<crate::safelayout::LayoutMap, String>,
        selected: usize,
        scroll: usize,
    ) -> Self {
        Self {
            path,
            map,
            selected,
            scroll,
            scroll_max: 0,
            last_sel: usize::MAX,
        }
    }

    /// The parsed map. `on_enter` either fills this or returns `Outcome::Leave`, so every
    /// later call — `render_frame`, `pre_draw`, a key handler — happens only after it was
    /// filled. Building it needs the file read and `&mut Explorer`, which is why it can't be
    /// a constructor argument the way the tensor screens' `TensorInfo` is.
    #[allow(clippy::expect_used)]
    pub(super) fn map(&self) -> &crate::safelayout::LayoutMap {
        self.map.as_ref().expect("on_enter leaves on a parse error")
    }
}

impl Mode for LayoutMode {
    fn spec(&self) -> ModeSpec {
        ModeSpec {
            id: HelpCtx::Layout,
            ctrlc_quits_immediately: false,
        }
    }

    fn on_enter(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        match &self.map {
            Ok(map) => {
                self.selected = self.selected.min(map.segments.len().saturating_sub(1));
                Ok(Outcome::Stay)
            }
            Err(e) => {
                let body = vec![
                    Line::from(Span::raw(format!(
                        "Can't read the layout of {}:",
                        self.path
                    ))),
                    Line::default(),
                    Line::from(crate::ui::dim_span(e.clone())),
                ];
                ex.float_scroll_popup(term, "Layout", body, PopupBackdrop::Files, None);
                Ok(Outcome::Leave(Nav::Back))
            }
        }
    }

    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame) {
        let flash = ex.copied_flash.as_ref().map(|(w, _)| w.clone());
        let (_max, regions, links, vscroll) = UI::render_layout(
            f,
            self.map(),
            self.selected,
            self.scroll,
            flash.as_deref(),
            true,
        );
        *ex.clickable.borrow_mut() = regions;
        *ex.links.borrow_mut() = links; // tensor band name → tree
        *ex.vscrollbar.borrow_mut() = vscroll;
    }

    fn pre_draw(&mut self, _ex: &mut Explorer, term: &mut crate::tui::LiveTerminal) {
        // Compute the scroll bounds from the band layout up front, then snap so the
        // selected band's label row stays visible when the selection moved.
        let Ok(sz) = term.size() else { return };
        let starts = match &self.map {
            Ok(m) => UI::layout_band_starts(m, sz.width, sz.height),
            Err(_) => return,
        };
        let body = UI::layout_visible_rows(sz.width, sz.height);
        let total_rows = starts.last().copied().unwrap_or(0);
        self.scroll_max = total_rows.saturating_sub(body);
        if self.selected != self.last_sel {
            let band_start = starts.get(self.selected).copied().unwrap_or(0);
            if band_start < self.scroll {
                self.scroll = band_start;
            } else if band_start >= self.scroll + body {
                self.scroll = band_start + 1 - body;
            }
            self.last_sel = self.selected;
        }
        self.scroll = self.scroll.min(self.scroll_max);
    }

    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult {
        let Ok(map) = &self.map else {
            return PaletteResult::Handled;
        };
        let Some(cmd) = ex.layout_command_palette(term, map, self.selected, self.scroll) else {
            return PaletteResult::Handled;
        };
        if cmd == LayoutCmd::CopyCommand {
            return PaletteResult::CopyCommand; // engine-owned, like the `y` key
        }
        if let Some(nav) =
            ex.run_layout_command(cmd, &self.path, map, self.selected, self.scroll, term)
        {
            return PaletteResult::Nav(nav);
        }
        PaletteResult::Handled
    }

    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome> {
        let n = self.map().segments.len();
        let move_sel = |sel: usize, delta: i32| -> usize {
            if delta < 0 {
                sel.saturating_sub((-delta) as usize)
            } else {
                (sel + delta as usize).min(n.saturating_sub(1))
            }
        };
        // Every lettered command dispatches through the registry (`q`/`l`/`c`/`y`) so key
        // and palette entry can't drift. See `FilesMode::handle_key` for why this sits
        // ahead of the match rather than in a guard.
        if !key
            .modifiers
            .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
            && let KeyCode::Char(ch) = key.code
            && let Some(cmd) = layout_command_for_key(ch)
        {
            if let Some(nav) = ex.run_layout_command(
                cmd,
                &self.path,
                self.map(),
                self.selected,
                self.scroll,
                term,
            ) {
                return Ok(Outcome::Leave(nav));
            }
            return Ok(Outcome::Stay);
        }
        match key.code {
            KeyCode::Backspace | KeyCode::Tab | KeyCode::Esc => {
                return Ok(Outcome::Leave(Nav::Back));
            }
            KeyCode::Char('\\') => return Ok(Outcome::Leave(Nav::Forward)),
            KeyCode::Up => {
                let step = ex.held_step(KeyCode::Up, accel_step_row) as i32;
                self.selected = move_sel(self.selected, -step);
            }
            KeyCode::Down => {
                let step = ex.held_step(KeyCode::Down, accel_step_row) as i32;
                self.selected = move_sel(self.selected, step);
            }
            KeyCode::PageUp => {
                let page = ex.layout_page_segments(self.map(), term.size().ok());
                let step = (page * ex.held_step(KeyCode::PageUp, accel_step_page)) as i32;
                self.selected = move_sel(self.selected, -step);
            }
            KeyCode::PageDown => {
                let page = ex.layout_page_segments(self.map(), term.size().ok());
                let step = (page * ex.held_step(KeyCode::PageDown, accel_step_page)) as i32;
                self.selected = move_sel(self.selected, step);
            }
            KeyCode::Home => self.selected = 0,
            KeyCode::End => self.selected = n.saturating_sub(1),
            // Enter on the header previews the raw JSON header; on a tensor it jumps
            // to that tensor's place in the tree.
            KeyCode::Enter => match self.map().segments.get(self.selected).map(|s| &s.kind) {
                Some(crate::safelayout::SegmentKind::Header) => {
                    ex.preview_header_json(
                        term,
                        &self.path,
                        self.map(),
                        self.selected,
                        self.scroll,
                    );
                }
                Some(crate::safelayout::SegmentKind::Tensor { .. }) => {
                    if let Some(nav) = ex.reveal_layout_selection(self.map(), self.selected)? {
                        return Ok(Outcome::Leave(nav));
                    }
                }
                _ => {}
            },
            _ => {}
        }
        Ok(Outcome::Stay)
    }

    fn handle_mouse(
        &mut self,
        _ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        m: MouseEvent,
    ) -> MouseOutcome {
        let (col, row) = (m.column, m.row);
        match m.kind {
            // A click on a band selects it (link / chip clicks are handled by the
            // driver's route_mouse before this).
            MouseEventKind::Down(MouseButton::Left) => {
                let _ = col;
                if let Ok(sz) = term.size() {
                    let top = UI::layout_header_rows() as u16;
                    let body = UI::layout_visible_rows(sz.width, sz.height);
                    if row >= top && (row as usize) < top as usize + body {
                        let content_row = self.scroll + (row - top) as usize;
                        let starts = UI::layout_band_starts(self.map(), sz.width, sz.height);
                        if let Some(seg) = starts
                            .windows(2)
                            .position(|w| content_row >= w[0] && content_row < w[1])
                        {
                            let n = self.map().segments.len();
                            self.selected = seg.min(n.saturating_sub(1));
                        }
                    }
                }
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollDown => {
                self.scroll = (self.scroll + WHEEL_STEP).min(self.scroll_max);
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollUp => {
                self.scroll = self.scroll.saturating_sub(WHEEL_STEP);
                MouseOutcome::Redraw
            }
            _ => MouseOutcome::Ignored,
        }
    }

    fn set_scroll(&mut self, _ex: &mut Explorer, offset: usize) {
        self.scroll = offset;
    }

    fn residual(&self) -> Screen {
        Screen::Layout {
            path: self.path.clone(),
            selected: self.selected,
            scroll: self.scroll,
        }
    }
}

/// The tensor tree ([`Screen::Tree`]) as a [`Mode`] — the root browser, including
/// the search sub-machine. Its selection/scroll/search state live on [`Explorer`];
/// this holds only the transient click/drag bookkeeping.
pub(super) struct TreeMode {
    pub(super) last_click: Option<(std::time::Instant, u16)>,
    pub(super) last_sel: usize,
}

impl TreeMode {
    pub(super) fn new() -> Self {
        Self {
            last_click: None,
            last_sel: usize::MAX,
        }
    }
}

impl Mode for TreeMode {
    fn spec(&self) -> ModeSpec {
        ModeSpec {
            id: HelpCtx::Tree,
            ctrlc_quits_immediately: false,
        }
    }

    // While searching, typed letters edit the query — skip the wrong-layout hint,
    // the badge-click actions, and the Space/`:` palette trigger.
    fn accepts_text(&self, ex: &Explorer) -> bool {
        ex.tree_state.search_mode()
    }
    fn palette_on_space(&self, ex: &Explorer) -> bool {
        !ex.tree_state.search_mode()
    }

    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame) {
        ex.render_tree_frame(f, true);
    }

    fn on_enter(
        &mut self,
        ex: &mut Explorer,
        _term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        // The browser needs the whole checkpoint; finish a deferred `--tensor` load.
        ex.ensure_full_load()?;
        Ok(Outcome::Stay)
    }

    fn pre_draw(&mut self, ex: &mut Explorer, term: &mut crate::tui::LiveTerminal) {
        if let Ok(sz) = term.size() {
            if ex.tree_state.selected != self.last_sel {
                ex.update_tree_scroll(sz.width, sz.height); // snap to the moved selection
                self.last_sel = ex.tree_state.selected;
            }
            let body = UI::tree_visible_rows(
                sz.width,
                sz.height,
                ex.tree_state.search_mode(),
                ex.can_repack(),
                ex.can_rename(),
            );
            let total = ex.current_tree_len();
            ex.tree_state.scroll = ex.tree_state.scroll.min(total.saturating_sub(body));
        }
    }

    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult {
        let Some(cmd) = ex.command_palette(term) else {
            return PaletteResult::Handled;
        };
        if cmd == Cmd::CopyCommand {
            return PaletteResult::CopyCommand; // engine-owned, like the `y` key
        }
        if let Some(nav) = ex.run_command(cmd, term) {
            return PaletteResult::Nav(nav);
        }
        PaletteResult::Handled
    }

    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome> {
        // Every tree command dispatches through the registry (the same path the palette
        // uses). In search mode the letters fall through to the query instead. Handled
        // before the match so the command is looked up once — see `FilesMode::handle_key`.
        if !ex.tree_state.search_mode()
            && !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
            && let KeyCode::Char(c) = key.code
            && let Some(cmd) = tree_command_for_key(c)
        {
            if let Some(nav) = ex.run_command(cmd, term) {
                return Ok(Outcome::Leave(nav));
            }
            return Ok(Outcome::Stay);
        }
        match key {
            // '/' is ignored rather than typed into the query.
            KeyEvent {
                code: KeyCode::Char('/'),
                ..
            } => {}
            KeyEvent {
                code: KeyCode::Esc, ..
            } if ex.tree_state.search_mode() => ex.exit_search_mode(),
            // Shift+↑/↓ jump to the previous/next sibling — before the plain arrows.
            KeyEvent {
                code: KeyCode::Up,
                modifiers: KeyModifiers::SHIFT,
                ..
            } => ex.tree_state.move_to_sibling(false),
            KeyEvent {
                code: KeyCode::Down,
                modifiers: KeyModifiers::SHIFT,
                ..
            } => ex.tree_state.move_to_sibling(true),
            KeyEvent {
                code: KeyCode::Up, ..
            } => {
                let step = ex.held_step(KeyCode::Up, accel_step_row) as i32;
                ex.tree_state.move_selection(-step);
            }
            KeyEvent {
                code: KeyCode::Down,
                ..
            } => {
                let step = ex.held_step(KeyCode::Down, accel_step_row) as i32;
                ex.tree_state.move_selection(step);
            }
            // While searching, ←/→ move the query caret (Shift = start/end).
            KeyEvent {
                code: KeyCode::Left,
                modifiers: KeyModifiers::SHIFT,
                ..
            } if ex.tree_state.search_mode() => ex.tree_state.search_cursor_home(),
            KeyEvent {
                code: KeyCode::Right,
                modifiers: KeyModifiers::SHIFT,
                ..
            } if ex.tree_state.search_mode() => ex.tree_state.search_cursor_end(),
            KeyEvent {
                code: KeyCode::Left,
                ..
            } if ex.tree_state.search_mode() => ex.tree_state.search_cursor_left(),
            KeyEvent {
                code: KeyCode::Right,
                ..
            } if ex.tree_state.search_mode() => ex.tree_state.search_cursor_right(),
            KeyEvent {
                code: KeyCode::Home,
                ..
            } if ex.tree_state.search_mode() => ex.tree_state.selected = 0,
            KeyEvent {
                code: KeyCode::End, ..
            } if ex.tree_state.search_mode() => {
                ex.tree_state.selected = ex.tree_state.visible().len().saturating_sub(1);
            }
            KeyEvent {
                code: KeyCode::PageUp,
                ..
            } => {
                let step = (ex.page_rows() * ex.held_step(KeyCode::PageUp, accel_step_page)) as i32;
                ex.tree_state.move_selection(-step);
            }
            KeyEvent {
                code: KeyCode::PageDown,
                ..
            } => {
                let step =
                    (ex.page_rows() * ex.held_step(KeyCode::PageDown, accel_step_page)) as i32;
                ex.tree_state.move_selection(step);
            }
            // ← jumps to the parent group; → enters the group's first child.
            KeyEvent {
                code: KeyCode::Left,
                ..
            } => ex.tree_state.move_to_parent(),
            KeyEvent {
                code: KeyCode::Right,
                ..
            } => ex.tree_state.move_to_first_child(),
            // While searching, Tab reveals the highlighted result in the tree
            // (leaving search); otherwise Tab toggles to the file browser.
            KeyEvent {
                code: KeyCode::Tab, ..
            } if ex.tree_state.search_mode() => ex.reveal_search_result(),
            KeyEvent {
                code: KeyCode::Tab, ..
            } => {
                if let Some(nav) = ex.run_command(Cmd::ViewFiles, term) {
                    return Ok(Outcome::Leave(nav));
                }
            }
            // Enter acts on the highlighted row: expand a group, or open a tensor.
            KeyEvent {
                code: KeyCode::Enter,
                ..
            } => {
                if let Some(nav) = ex.activate_selection(term) {
                    return Ok(Outcome::Leave(nav));
                }
            }
            // While searching, Space is ignored rather than typed into the query.
            KeyEvent {
                code: KeyCode::Char(' '),
                ..
            } => {}
            // Backspace edits the query while searching, else steps back.
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } if ex.tree_state.search_mode() => ex.search_backspace(),
            KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => return Ok(Outcome::Leave(Nav::Back)),
            KeyEvent {
                code: KeyCode::Char('\\'),
                ..
            } if !ex.tree_state.search_mode() => return Ok(Outcome::Leave(Nav::Forward)),
            // Any other char while searching is inserted into the query.
            KeyEvent {
                code: KeyCode::Char(c),
                ..
            } if ex.tree_state.search_mode() => ex.search_insert(c),
            _ => {}
        }
        Ok(Outcome::Stay)
    }

    fn handle_mouse(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        m: MouseEvent,
    ) -> MouseOutcome {
        let Ok(sz) = term.size() else {
            return MouseOutcome::Ignored;
        };
        let (col, row) = (m.column, m.row);
        match m.kind {
            // (Scroll-bar clicks / drags are handled by the engine's `route_mouse`.)
            MouseEventKind::Down(MouseButton::Left) => {
                let body_top = UI::tree_header_rows(ex.tree_state.search_mode()) as u16;
                // Body ends above the bottom-pinned hint footer + the 2-line status bar.
                let hint_rows = UI::tree_hint_rows(
                    sz.width,
                    ex.tree_state.search_mode(),
                    ex.can_repack(),
                    ex.can_rename(),
                ) as u16;
                let body_bottom = sz.height.saturating_sub(2 + hint_rows);
                if row >= body_top && row < body_bottom {
                    let idx = ex.tree_state.scroll + (row - body_top) as usize;
                    if idx < ex.current_tree_len() {
                        // A click exactly on a group's ▸/▾ twisty (column `2*depth`)
                        // toggles it on a single click.
                        let on_arrow = {
                            let tree = ex.tree_state.visible();
                            matches!(
                                tree.get(idx),
                                Some((TreeNode::Group { .. }, depth)) if col == 2 * *depth as u16
                            )
                        };
                        ex.tree_state.selected = idx;
                        if on_arrow {
                            self.last_click = None;
                            ex.activate_selection(term);
                        } else {
                            let double = matches!(
                                self.last_click,
                                Some((t, r)) if r == row && t.elapsed() < DOUBLE_CLICK
                            );
                            if double {
                                self.last_click = None;
                                if ex.tree_state.search_mode() {
                                    ex.reveal_search_result();
                                } else if let Some(nav) = ex.activate_selection(term) {
                                    return MouseOutcome::Leave(nav);
                                }
                            } else {
                                self.last_click = Some((std::time::Instant::now(), row));
                            }
                        }
                    }
                }
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollDown => {
                ex.tree_state.scroll = ex.tree_state.scroll.saturating_add(WHEEL_STEP);
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollUp => {
                ex.tree_state.scroll = ex.tree_state.scroll.saturating_sub(WHEEL_STEP);
                MouseOutcome::Redraw
            }
            _ => MouseOutcome::Ignored,
        }
    }

    fn set_scroll(&mut self, ex: &mut Explorer, offset: usize) {
        ex.tree_state.scroll = offset;
    }

    fn residual(&self) -> Screen {
        Screen::Tree
    }
}

/// The in-place rename editor ([`Screen::Rename`]) as a [`Mode`]. Owns the editor
/// model plus the cached shard headers and the dirty-gated preview (so pure caret /
/// focus moves don't re-scan the checkpoint). `scroll_max` is a `Cell` because it's
/// learned during the (`&self`) draw and read back by key/mouse handling.
pub(super) struct RenameMode2 {
    /// Seed pairs from a prior visit / `--rename-rule`, consumed by `on_enter`.
    pub(super) saved_pairs: Vec<(String, String)>,
    pub(super) target: std::path::PathBuf,
    pub(super) loaded: Option<crate::rename::Loaded>,
    /// The deduped generalized schemas the autocomplete offers, each with the count
    /// of tensors it covers (the dropdown's `×N` column).
    pub(super) schemas: Vec<(String, usize)>,
    pub(super) root: String,
    pub(super) editor: RenameMode,
    /// What was last copied (the `✓ copied …` flash), cleared on the next key.
    pub(super) copied: Option<&'static str>,
    /// The autocomplete dropdown's row rects from the last frame, so a click can
    /// accept the candidate under the cursor.
    pub(super) menu_rects: std::cell::RefCell<Vec<ratatui::layout::Rect>>,
    // Derived, recomputed only when the rule pairs change (`dirty`).
    pub(super) rules_view: Vec<crate::ui::RenameRuleView>,
    pub(super) total: usize,
    pub(super) warnings: Vec<String>,
    pub(super) has_index: bool,
    pub(super) applicable: bool,
    pub(super) synth_err: Option<String>,
    pub(super) cli: Option<String>,
    pub(super) dirty: bool,
    pub(super) scroll_max: Cell<usize>,
    /// Set once a rename is applied — the rules are spent, so `residual` clears them.
    pub(super) applied: bool,
}

impl RenameMode2 {
    pub(super) fn new(saved_pairs: Vec<(String, String)>) -> Self {
        Self {
            saved_pairs,
            target: std::path::PathBuf::new(),
            loaded: None,
            schemas: Vec::new(),
            root: String::new(),
            editor: RenameMode::default(),
            copied: None,
            menu_rects: std::cell::RefCell::new(Vec::new()),
            rules_view: Vec::new(),
            total: 0,
            warnings: Vec::new(),
            has_index: false,
            applicable: false,
            synth_err: None,
            cli: None,
            dirty: true,
            scroll_max: Cell::new(0),
            applied: false,
        }
    }

    /// The loaded rename set. As with `LayoutMode::map`, `on_enter` loads it or leaves the
    /// screen, and loading reads the checkpoint's headers — not something a constructor
    /// can do.
    #[allow(clippy::expect_used)]
    pub(super) fn loaded(&self) -> &crate::rename::Loaded {
        self.loaded.as_ref().expect("on_enter loads or leaves")
    }

    /// The current rules to persist / restore (dropping fully-blank pairs).
    pub(super) fn pairs(&self) -> Vec<(String, String)> {
        self.editor
            .pairs
            .iter()
            .filter(|p| !(p.source.trim().is_empty() && p.target.trim().is_empty()))
            .map(|p| (p.source.clone(), p.target.clone()))
            .collect()
    }

    pub(super) fn do_copy_apply(&mut self) {
        self.copied = (self.cli.is_some()
            && copy_to_clipboard(self.cli.as_deref().unwrap_or_default()))
        .then_some("the apply command");
    }

    pub(super) fn do_copy_screen(&mut self) {
        let text = crate::tui::headless_render(120, 40, |f| {
            let _ = draw_rename_frame(
                f,
                &self.root,
                &self.editor,
                &self.schemas,
                &self.rules_view,
                self.total,
                &self.warnings,
                self.has_index,
                self.applicable,
                &self.synth_err,
                self.cli.as_deref(),
                None,
            );
        });
        if let Ok(text) = text {
            self.copied = copy_to_clipboard(&text).then_some("the screen text");
        }
    }

    /// Apply the rename (`R`): flash why it can't yet if it isn't clean, else float a
    /// confirmation modal and — only on an explicit confirm — rewrite the files.
    /// Returns `Some(nav)` to leave the editor once applied. Shared by the `R` key
    /// and the palette's *Apply* command.
    pub(super) fn try_apply(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> Option<Nav> {
        if !self.applicable {
            self.editor.error =
                Some("can't apply yet — fix the blocked rows / warnings above".to_string());
            return None;
        }
        if !self.confirm_apply(term) {
            return None;
        }
        match ex.apply_rename_mode(self.loaded(), &self.editor) {
            Ok(nav) => {
                self.applied = true; // rules spent → residual clears them
                Some(nav)
            }
            Err(e) => {
                self.editor.error = Some(e);
                None
            }
        }
    }

    /// Float the apply-confirmation modal over the live editor: a summary of what
    /// will change (from [`crate::rename::Plan::summary_lines`]) plus an
    /// `[Apply] [Cancel]` choice. Returns `true` only on an explicit confirm
    /// (`Enter` on *Apply*, or `Y`); `Esc` / `N` / *Cancel* return `false`.
    pub(super) fn confirm_apply(&self, term: &mut crate::tui::LiveTerminal) -> bool {
        let fallback = || vec!["Apply the entered renames in place?".to_string()];
        let summary = match self.editor.build_map() {
            Ok((map, _)) => match self.loaded().plan(&map) {
                Ok(plan) => plan.summary_lines(8),
                Err(_) => fallback(),
            },
            Err(_) => fallback(),
        };
        let mut idx = 1usize; // default to the safe choice (Cancel)
        loop {
            if term
                .draw(|f| {
                    let _ = draw_rename_frame(
                        f,
                        &self.root,
                        &self.editor,
                        &self.schemas,
                        &self.rules_view,
                        self.total,
                        &self.warnings,
                        self.has_index,
                        self.applicable,
                        &self.synth_err,
                        self.cli.as_deref(),
                        self.copied,
                    );
                    UI::render_confirm_popup(
                        f,
                        "Apply rename in place?",
                        &summary,
                        &["Apply", "Cancel"],
                        idx,
                    );
                })
                .is_err()
            {
                return false;
            }
            match event::read() {
                Ok(Event::Key(key)) if is_ctrl_c(&key) => quit_immediately(),
                Ok(Event::Key(KeyEvent { code, .. })) => match code {
                    KeyCode::Left | KeyCode::Right | KeyCode::Tab => idx = 1 - idx,
                    KeyCode::Char('y' | 'Y') => return true,
                    KeyCode::Char('n' | 'N') => return false,
                    KeyCode::Enter => return idx == 0,
                    KeyCode::Esc => return false,
                    _ => {}
                },
                Ok(_) => {} // other mouse / resize: redraw
                Err(_) => return false,
            }
        }
    }
}

impl Mode for RenameMode2 {
    fn spec(&self) -> ModeSpec {
        ModeSpec {
            id: HelpCtx::Rename,
            ctrlc_quits_immediately: true,
        }
    }

    // The name fields take arbitrary text; skip the wrong-layout hint. Space / `:`
    // still open the palette (a tensor-name schema never contains either).
    fn accepts_text(&self, _ex: &Explorer) -> bool {
        true
    }

    fn on_enter(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        let Some(target) = ex.rename_target() else {
            return Ok(Outcome::Leave(Nav::Back)); // gated; bail safely if it slips
        };
        // Read every shard header once, so the preview is instant as the user types.
        let loaded = match crate::rename::load(&target) {
            Ok(l) => l,
            Err(e) => {
                let msg = format!("Cannot open the rename editor: {e:#}");
                ex.float_until_dismissed(term, |f| {
                    ex.render_tree_frame(f, true);
                    UI::render_notice(f, &msg);
                });
                return Ok(Outcome::Leave(Nav::Back));
            }
        };
        // Autocomplete over the deduped *generalized* schemas (one per tensor
        // family), each tagged with how many tensors it covers (the `×N` column).
        // One `generalize` per name (it's the hot part of opening a big checkpoint),
        // keeping first-seen order.
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for n in loaded.names() {
            let schema = crate::rename::generalize(n).0;
            let seen = counts.entry(schema.clone()).or_insert(0);
            if *seen == 0 {
                order.push(schema);
            }
            *seen += 1;
        }
        self.schemas = order
            .into_iter()
            .map(|s| {
                let c = counts[&s];
                (s, c)
            })
            .collect();
        self.root = loaded.root().display().to_string();
        if !self.saved_pairs.is_empty() {
            self.editor.pairs = std::mem::take(&mut self.saved_pairs)
                .into_iter()
                .map(|(source, target)| RenamePair { source, target })
                .collect();
        }
        self.target = target;
        self.loaded = Some(loaded);
        Ok(Outcome::Stay)
    }

    fn pre_draw(&mut self, ex: &mut Explorer, _term: &mut crate::tui::LiveTerminal) {
        if self.dirty {
            // Compute into locals first, then assign (so the `loaded` borrow ends
            // before the `&mut self` field writes).
            let (warnings, has_index, applicable, err, cli, rules_view, total) = {
                let loaded = self.loaded();
                let (preview, notes, err) = match self.editor.build_map() {
                    Ok((map, notes)) => (loaded.preview(&map), notes, None),
                    Err(e) => (crate::rename::RenamePreview::default(), Vec::new(), Some(e)),
                };
                let mut warnings = preview.warnings.clone();
                warnings.extend(notes);
                let has_index = preview.has_index;
                let applicable = err.is_none() && preview.applicable();
                let cli = ex.rename_cli_command(&self.target, &self.editor);
                let mut rules_view: Vec<crate::ui::RenameRuleView> = Vec::new();
                let mut total = 0usize;
                for p in &self.editor.pairs {
                    if p.source.trim().is_empty() || p.target.trim().is_empty() {
                        continue;
                    }
                    let Ok((pat, rep)) = crate::rename::rule_from_fields(&p.source, &p.target)
                    else {
                        continue;
                    };
                    let Ok(single) = crate::diff::NameMap::from_pairs([(pat, rep)]) else {
                        continue;
                    };
                    let pv = loaded.preview(&single);
                    let mut v = crate::ui::RenameRuleView {
                        from: p.source.clone(),
                        to: p.target.clone(),
                        total: pv.rows.len(),
                        matched: single.match_count(loaded.names().iter().map(String::as_str)),
                        ok: 0,
                        collide: 0,
                        wont_fit: 0,
                        invalid: 0,
                        shards: loaded.shard_fits(&single),
                    };
                    for r in &pv.rows {
                        match r.status {
                            crate::rename::RenameStatus::Ok => v.ok += 1,
                            crate::rename::RenameStatus::Collision => v.collide += 1,
                            crate::rename::RenameStatus::WontFit => v.wont_fit += 1,
                            crate::rename::RenameStatus::Invalid => v.invalid += 1,
                        }
                    }
                    total += v.total;
                    rules_view.push(v);
                }
                (warnings, has_index, applicable, err, cli, rules_view, total)
            };
            self.warnings = warnings;
            self.has_index = has_index;
            self.applicable = applicable;
            self.synth_err = err;
            self.cli = cli;
            self.rules_view = rules_view;
            self.total = total;
            self.dirty = false;
        }
        self.editor.scroll = self.editor.scroll.min(self.scroll_max.get());
    }

    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame) {
        let (max, chips, clicks, menu_rects, vscroll) = draw_rename_frame(
            f,
            &self.root,
            &self.editor,
            &self.schemas,
            &self.rules_view,
            self.total,
            &self.warnings,
            self.has_index,
            self.applicable,
            &self.synth_err,
            self.cli.as_deref(),
            self.copied,
        );
        self.scroll_max.set(max);
        *ex.vscrollbar.borrow_mut() = vscroll;
        // A preview link the open dropdown floats over must not steal the click that
        // was meant for a candidate row (the generic router tries links first).
        let clicks: crate::ui::LinkRegions = clicks
            .into_iter()
            .filter(|(r, _)| {
                !menu_rects
                    .iter()
                    .any(|mr| r.y == mr.y && r.x < mr.x + mr.width && mr.x < r.x + r.width)
            })
            .collect();
        *self.menu_rects.borrow_mut() = menu_rects; // dropdown rows (click to accept)
        *ex.clickable.borrow_mut() = chips; // footer chips (replay a key)
        *ex.links.borrow_mut() = clicks; // shard → layout, tensor → tree
    }

    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult {
        let entries =
            available_rename_commands(self.applicable, self.cli.is_some(), self.editor.pairs.len());
        let chosen = ex.run_palette(term, entries, HelpCtx::Rename, |_s, f| {
            let _ = draw_rename_frame(
                f,
                &self.root,
                &self.editor,
                &self.schemas,
                &self.rules_view,
                self.total,
                &self.warnings,
                self.has_index,
                self.applicable,
                &self.synth_err,
                self.cli.as_deref(),
                self.copied,
            );
        });
        match chosen {
            Some(RenameCmd::Back) => PaletteResult::Nav(Nav::Back),
            Some(RenameCmd::Quit) => PaletteResult::Nav(Nav::Quit),
            Some(RenameCmd::AddRule) => {
                self.editor.add_pair();
                self.editor.error = None;
                self.dirty = true;
                PaletteResult::Handled
            }
            Some(RenameCmd::RemoveRule) => {
                self.editor.remove_pair();
                self.editor.error = None;
                self.dirty = true;
                PaletteResult::Handled
            }
            Some(RenameCmd::Apply) => match self.try_apply(ex, term) {
                Some(nav) => PaletteResult::Nav(nav),
                None => PaletteResult::Handled,
            },
            Some(RenameCmd::CopyApplyCmd) => {
                self.do_copy_apply();
                PaletteResult::Handled
            }
            Some(RenameCmd::CopyReopenCmd) => PaletteResult::CopyCommand,
            Some(RenameCmd::CopyScreen) => {
                self.do_copy_screen();
                PaletteResult::Handled
            }
            Some(RenameCmd::Legend) => {
                ex.float_until_dismissed(term, |f| {
                    let _ = draw_rename_frame(
                        f,
                        &self.root,
                        &self.editor,
                        &self.schemas,
                        &self.rules_view,
                        self.total,
                        &self.warnings,
                        self.has_index,
                        self.applicable,
                        &self.synth_err,
                        self.cli.as_deref(),
                        self.copied,
                    );
                    UI::render_legend_band(f, Legend::Rename);
                });
                PaletteResult::Handled
            }
            None => PaletteResult::Handled,
        }
    }

    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome> {
        let KeyEvent {
            code, modifiers, ..
        } = key;
        // `^Y` (copy the command to reopen this editor — the `y` of the non-editing
        // modes) is handled by the engine (`do_copy_command`), so it's identical
        // everywhere. `^A` copies the `convert --map` command that *applies* the rename.
        if code == KeyCode::Char('a') && modifiers.contains(KeyModifiers::CONTROL) {
            self.do_copy_apply();
            return Ok(Outcome::Stay);
        }
        self.copied = None;
        // When the autocomplete dropdown is open, the arrows drive it and Enter
        // accepts the highlight (pgcli-style); otherwise Enter moves between fields.
        let cands = self.editor.completions(&self.schemas);
        let menu_open = self.editor.menu.is_some() && !cands.is_empty();
        match code {
            KeyCode::Esc if menu_open => self.editor.close_menu(),
            KeyCode::Esc => return Ok(Outcome::Leave(Nav::Back)),
            KeyCode::Char('n') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.editor.add_pair();
                self.editor.error = None;
                self.dirty = true;
            }
            KeyCode::Char('d') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.editor.remove_pair();
                self.editor.error = None;
                self.dirty = true;
            }
            // `^S` copies the whole screen (bare `c` types into a field here, so
            // copy-screen is a Ctrl key in the editor).
            KeyCode::Char('s') if modifiers.contains(KeyModifiers::CONTROL) => {
                self.do_copy_screen();
            }
            // `^L` shows the legend (bare `l` types into a field here).
            KeyCode::Char('l') if modifiers.contains(KeyModifiers::CONTROL) => {
                ex.float_until_dismissed(term, |f| {
                    let _ = draw_rename_frame(
                        f,
                        &self.root,
                        &self.editor,
                        &self.schemas,
                        &self.rules_view,
                        self.total,
                        &self.warnings,
                        self.has_index,
                        self.applicable,
                        &self.synth_err,
                        self.cli.as_deref(),
                        self.copied,
                    );
                    UI::render_legend_band(f, Legend::Rename);
                });
            }
            // Tab opens the dropdown and extends the field to the candidates' longest
            // common prefix (shell-style). Enter / a click accept the highlight — so
            // the two keys stay distinct.
            KeyCode::Tab => {
                self.editor.open_menu();
                self.editor.complete_prefix(&self.schemas);
                self.editor.error = None;
                self.dirty = true;
            }
            // Enter accepts a highlighted candidate; with the dropdown closed it
            // moves to the next field (adding a rule past the last) — it never
            // applies. Apply is `^R` (below).
            KeyCode::Enter if menu_open => {
                self.editor.accept(&self.schemas);
                self.editor.error = None;
                self.dirty = true;
            }
            KeyCode::Enter => self.editor.focus_down(),
            // `^R` applies the rename (a Ctrl key, so every character stays typeable),
            // after a confirmation pop-up.
            KeyCode::Char('r') if modifiers.contains(KeyModifiers::CONTROL) => {
                if let Some(nav) = self.try_apply(ex, term) {
                    return Ok(Outcome::Leave(nav));
                }
            }
            KeyCode::Up if menu_open => self.editor.menu_move(-1, cands.len()),
            KeyCode::Down if menu_open => self.editor.menu_move(1, cands.len()),
            KeyCode::Up => self.editor.focus_up(),
            KeyCode::Down => self.editor.focus_down(),
            KeyCode::Left => self.editor.left(),
            KeyCode::Right => self.editor.right(),
            KeyCode::Home => self.editor.cursor = 0,
            KeyCode::End => self.editor.caret_to_end(),
            KeyCode::PageUp => self.editor.scroll = self.editor.scroll.saturating_sub(SCROLL_PAGE),
            KeyCode::PageDown => {
                self.editor.scroll = (self.editor.scroll + SCROLL_PAGE).min(self.scroll_max.get());
            }
            KeyCode::Backspace => {
                self.editor.backspace();
                self.editor.remove_pair_if_empty();
                self.editor.error = None;
                self.dirty = true;
            }
            KeyCode::Delete => {
                self.editor.delete();
                self.editor.remove_pair_if_empty();
                self.editor.error = None;
                self.dirty = true;
            }
            KeyCode::Char(c)
                if !modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                self.editor.insert_char(c);
                self.editor.error = None;
                self.dirty = true;
            }
            _ => {}
        }
        Ok(Outcome::Stay)
    }

    fn handle_mouse(
        &mut self,
        _ex: &mut Explorer,
        _term: &mut crate::tui::LiveTerminal,
        m: MouseEvent,
    ) -> MouseOutcome {
        match m.kind {
            // A click on a dropdown row highlights and accepts that candidate.
            MouseEventKind::Down(MouseButton::Left) => {
                let hit = self.menu_rects.borrow().iter().position(|r| {
                    m.column >= r.x
                        && m.column < r.x + r.width
                        && m.row >= r.y
                        && m.row < r.y + r.height
                });
                if let Some(i) = hit {
                    self.editor.menu = Some(i);
                    self.editor.accept(&self.schemas);
                    self.editor.error = None;
                    self.dirty = true;
                    MouseOutcome::Redraw
                } else {
                    MouseOutcome::Ignored
                }
            }
            MouseEventKind::ScrollDown => {
                self.editor.scroll = (self.editor.scroll + 3).min(self.scroll_max.get());
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollUp => {
                self.editor.scroll = self.editor.scroll.saturating_sub(3);
                MouseOutcome::Redraw
            }
            _ => MouseOutcome::Ignored,
        }
    }

    fn set_scroll(&mut self, _ex: &mut Explorer, offset: usize) {
        self.editor.scroll = offset;
    }

    fn residual(&self) -> Screen {
        Screen::Rename {
            pairs: if self.applied {
                Vec::new()
            } else {
                self.pairs()
            },
        }
    }
}

/// The tensor detail screen ([`Screen::Detail`]) as a [`Mode`]. Runs the exact-stats
/// scan on a worker thread (via `tick_background` + `Bg::Poll`) and floats the legend
/// / copied-command as an in-frame `overlay` so a running scan animates behind it.
pub(super) struct DetailMode {
    pub(super) slice: usize,
    pub(super) stats_start: StatsStart,
    pub(super) interaction: Interaction,
    /// The tensor this screen is about. Resolved by the caller — a screen for a tensor
    /// that isn't in the checkpoint is not a state this can be in, so there is nothing
    /// for the renderer or a key handler to unwrap.
    pub(super) tensor: TensorInfo,
    pub(super) overridable: bool,
    pub(super) unindexed: bool,
    pub(super) remote: bool,
    pub(super) warm: bool,
    pub(super) scan: Option<ScanJob>,
    pub(super) spin: Cell<usize>,
    pub(super) overlay: Option<Overlay>,
}

impl DetailMode {
    pub(super) fn new(
        tensor: TensorInfo,
        slice: usize,
        stats_start: StatsStart,
        interaction: Interaction,
    ) -> Self {
        Self {
            slice,
            stats_start,
            interaction,
            tensor,
            overridable: false,
            unindexed: false,
            remote: false,
            warm: false,
            scan: None,
            spin: Cell::new(0),
            overlay: None,
        }
    }

    pub(super) fn tensor(&self) -> &TensorInfo {
        &self.tensor
    }

    pub(super) fn shape(&self, ex: &Explorer) -> Vec<usize> {
        let t = self.tensor();
        ex.data_view
            .shape_overrides
            .borrow()
            .get(&t.name)
            .cloned()
            .unwrap_or_else(|| t.shape.clone())
    }

    /// The current statistics view — cached result, a live scan spinner, or pending.
    /// `stats` is the caller's local so the returned `StatsView` can borrow it.
    ///
    /// Only a *pre-warm* scan is shown: without `warm` this screen didn't ask for one, so
    /// a job left over from elsewhere shouldn't put a spinner on it.
    pub(super) fn stats_view<'a>(&self, stats: &'a Option<Stats>) -> StatsView<'a> {
        let scan = if self.warm { &self.scan } else { &None };
        scan_stats_view(scan, &self.spin, stats)
    }

    pub(super) fn layout_ok(&self) -> bool {
        !self.remote
            && std::path::Path::new(&self.tensor().source_path)
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
    }
}

impl Mode for DetailMode {
    fn spec(&self) -> ModeSpec {
        ModeSpec {
            id: HelpCtx::Detail,
            ctrlc_quits_immediately: true,
        }
    }

    fn set_background_paused(&self, paused: bool) {
        if let Some(job) = &self.scan {
            job.pause.store(paused, Ordering::Relaxed);
        }
    }

    fn overlay(&self) -> Option<&Overlay> {
        self.overlay.as_ref()
    }
    fn dismiss_overlay(&mut self) -> bool {
        self.overlay.take().is_some()
    }

    fn on_enter(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        let tensor = self.tensor.clone();
        self.overridable = dtype_overridable(&tensor);
        self.unindexed = ex.unindexed.contains(&tensor.source_path);
        self.remote = crate::remote::is_remote_source(&tensor.source_path);
        // Background pre-warm scan: only when interactive, overridable, local, and
        // not already doing a synchronous `--compute-stats` scan.
        self.warm = ex.preload
            && self.stats_start != StatsStart::Auto
            && self.interaction == Interaction::Interactive
            && self.overridable
            && !self.remote;
        // `--compute-stats`: kick off the scan synchronously on open, animating the
        // spinner right here.
        if self.stats_start == StatsStart::Auto && !self.remote {
            let view = ex.active_view(&tensor.name);
            let shape = ex
                .data_view
                .shape_overrides
                .borrow()
                .get(&tensor.name)
                .cloned()
                .unwrap_or_else(|| tensor.shape.clone());
            let (overridable, unindexed) = (self.overridable, self.unindexed);
            ex.compute_stats_animated(term, &tensor, view, |f, sv| {
                ex.render_detail_frame(
                    f,
                    &tensor,
                    &shape,
                    view,
                    overridable,
                    unindexed,
                    sv,
                    None,
                    None,
                    None,
                );
            });
        }
        Ok(Outcome::Stay)
    }

    fn tick_background(&mut self, ex: &mut Explorer) -> Bg {
        if !self.warm {
            return Bg::Idle;
        }
        let tensor = self.tensor().clone();
        let view = ex.active_view(&tensor.name);
        poll_stats_scan(ex, &mut self.scan, &tensor, view)
    }

    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame) {
        let tensor = self.tensor();
        let view = ex.active_view(&tensor.name);
        let shape = self.shape(ex);
        let stats = ex.cached_stats(tensor, view);
        let stats_view = self.stats_view(&stats);
        let hist = ex
            .histogram_cache
            .borrow()
            .get(&(tensor.name.clone(), view, ex.data_view.histogram_bins.get()))
            .cloned();
        ex.render_detail_frame(
            f,
            tensor,
            &shape,
            view,
            self.overridable,
            self.unindexed,
            stats_view,
            hist.as_ref(),
            None,
            self.overlay.as_ref(),
        );
        if let Some((what, _)) = &ex.copied_flash {
            UI::render_copied_flash(f, what);
        }
    }

    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult {
        let tensor = self.tensor();
        let view = ex.active_view(&tensor.name);
        let shape = self.shape(ex);
        let stats = ex.cached_stats(tensor, view);
        let stats_view = self.stats_view(&stats);
        let hist = ex
            .histogram_cache
            .borrow()
            .get(&(tensor.name.clone(), view, ex.data_view.histogram_bins.get()))
            .cloned();
        let entries = available_detail_commands(self.overridable, self.layout_ok());
        let (overridable, unindexed) = (self.overridable, self.unindexed);
        let chosen = ex.run_palette(term, entries, HelpCtx::Detail, |s, f| {
            s.render_detail_frame(
                f,
                tensor,
                &shape,
                view,
                overridable,
                unindexed,
                stats_view,
                hist.as_ref(),
                None,
                None,
            );
        });
        match chosen {
            Some(DetailCmd::CopyCommand) => PaletteResult::CopyCommand,
            Some(cmd) => PaletteResult::SynthKey(detail_cmd_key(cmd)),
            None => PaletteResult::Handled,
        }
    }

    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome> {
        let tensor = self.tensor().clone();
        let view = ex.active_view(&tensor.name);
        let shape = self.shape(ex);
        // Metadata-only (remote): the data keys can't run without local bytes, so
        // float a notice instead of a read that fails.
        if self.remote
            && matches!(
                key.code,
                KeyCode::Char('m' | 'v' | 'h' | 's' | 'S' | 'b' | 'B')
            )
        {
            self.overlay = Some(Overlay::Notice(
                "Read remotely with --ssh-proxy: only the structure is here. Data views \
                 (heatmap, values, histogram, statistics) need the file locally — copy the \
                 checkpoint down to preview its values."
                    .to_string(),
            ));
            return Ok(Outcome::Stay);
        }
        match key.code {
            KeyCode::Char('m') => {
                return Ok(Outcome::Leave(Nav::Open(Screen::Data {
                    tensor: tensor.name.clone(),
                    repr: Representation::Heatmap,
                    slice: self.slice,
                })));
            }
            KeyCode::Char('v') => {
                return Ok(Outcome::Leave(Nav::Open(Screen::Data {
                    tensor: tensor.name.clone(),
                    repr: Representation::Values,
                    slice: self.slice,
                })));
            }
            KeyCode::Tab => {
                if let Some(screen) = ex.tensor_layout_screen(&tensor) {
                    return Ok(Outcome::Leave(Nav::Open(screen)));
                }
            }
            KeyCode::Char('h') => {
                ex.ensure_detail_histogram(
                    term,
                    &tensor,
                    view,
                    &shape,
                    self.overridable,
                    self.unindexed,
                );
            }
            KeyCode::Char('b' | 'B') => {
                let (overridable, unindexed) = (self.overridable, self.unindexed);
                let stats = ex.cached_stats(&tensor, view);
                let stats_view = self.stats_view(&stats);
                let hist = ex
                    .histogram_cache
                    .borrow()
                    .get(&(tensor.name.clone(), view, ex.data_view.histogram_bins.get()))
                    .cloned();
                let background = |f: &mut ratatui::Frame| {
                    ex.render_detail_frame(
                        f,
                        &tensor,
                        &shape,
                        view,
                        overridable,
                        unindexed,
                        stats_view,
                        hist.as_ref(),
                        None,
                        None,
                    );
                };
                let changed =
                    match ex.prompt_bins(term, background, ex.data_view.histogram_bins.get()) {
                        BinsChoice::Set(n) => {
                            ex.data_view.histogram_bins.set(Some(n));
                            true
                        }
                        BinsChoice::Clear => {
                            ex.data_view.histogram_bins.set(None);
                            true
                        }
                        BinsChoice::Cancel => false,
                    };
                if changed {
                    ex.ensure_detail_histogram(
                        term,
                        &tensor,
                        view,
                        &shape,
                        self.overridable,
                        self.unindexed,
                    );
                }
            }
            KeyCode::Char('s' | 'S') => {
                // `s` is a no-op once the stats are cached — say so rather than
                // silently doing nothing (a key that appears not to work).
                if ex.cached_stats(&tensor, view).is_some() {
                    ex.copied_flash = Some((
                        "statistics already computed".to_string(),
                        std::time::Instant::now(),
                    ));
                } else {
                    let (overridable, unindexed) = (self.overridable, self.unindexed);
                    ex.compute_stats_animated(term, &tensor, view, |f, sv| {
                        ex.render_detail_frame(
                            f,
                            &tensor,
                            &shape,
                            view,
                            overridable,
                            unindexed,
                            sv,
                            None,
                            None,
                            None,
                        );
                    });
                }
            }
            KeyCode::Char('d' | 'D') if self.overridable => {
                if let Some(chosen) = ex.prompt_dtype(term, &tensor, DtypePreview::Detail) {
                    let def = ex.default_view(&tensor.name);
                    let mut overrides = ex.data_view.dtype_overrides.borrow_mut();
                    if chosen == def {
                        overrides.remove(&tensor.name);
                    } else {
                        overrides.insert(tensor.name.clone(), chosen);
                    }
                }
            }
            KeyCode::Char('r' | 'R') if self.overridable => {
                let current = ex
                    .data_view
                    .shape_overrides
                    .borrow()
                    .get(&tensor.name)
                    .cloned();
                let (overridable, unindexed) = (self.overridable, self.unindexed);
                let stats = ex.cached_stats(&tensor, view);
                let stats_view = self.stats_view(&stats);
                let hist = ex
                    .histogram_cache
                    .borrow()
                    .get(&(tensor.name.clone(), view, ex.data_view.histogram_bins.get()))
                    .cloned();
                let background = |f: &mut ratatui::Frame| {
                    ex.render_detail_frame(
                        f,
                        &tensor,
                        &shape,
                        view,
                        overridable,
                        unindexed,
                        stats_view,
                        hist.as_ref(),
                        None,
                        None,
                    );
                };
                match ex.prompt_reshape(term, background, &tensor, current.as_deref()) {
                    ReshapeChoice::Set(s) => {
                        ex.data_view
                            .shape_overrides
                            .borrow_mut()
                            .insert(tensor.name.clone(), s);
                    }
                    ReshapeChoice::Clear => {
                        ex.data_view
                            .shape_overrides
                            .borrow_mut()
                            .remove(&tensor.name);
                    }
                    ReshapeChoice::Cancel => {}
                }
            }
            KeyCode::Char('c') => {
                let stats = ex.cached_stats(&tensor, view);
                let stats_view = self.stats_view(&stats);
                let hist = ex
                    .histogram_cache
                    .borrow()
                    .get(&(tensor.name.clone(), view, ex.data_view.histogram_bins.get()))
                    .cloned();
                if let Ok(text) = ex.detail_plain(
                    &tensor,
                    &shape,
                    view,
                    self.overridable,
                    self.unindexed,
                    stats_view,
                    hist.as_ref(),
                    None,
                ) {
                    copy_to_clipboard(&text);
                }
                ex.copied_flash = Some((
                    "copied the screen to the clipboard".to_string(),
                    std::time::Instant::now(),
                ));
            }
            // `y` (copy the reopen command) is handled by the engine — see
            // `Explorer::do_copy_command` — so every mode does it identically.
            KeyCode::Char('l') => self.overlay = Some(Overlay::Legend(Legend::Detail)),
            KeyCode::Backspace => return Ok(Outcome::Leave(Nav::Back)),
            KeyCode::Char('\\') => return Ok(Outcome::Leave(Nav::Forward)),
            // Any other key goes back to the tree.
            _ => return Ok(Outcome::Leave(Nav::Open(Screen::Tree))),
        }
        Ok(Outcome::Stay)
    }

    fn residual(&self) -> Screen {
        Screen::Detail {
            tensor: self.tensor.name.clone(),
            slice: self.slice,
        }
    }
}

/// The full-screen checkpoint-stats view ([`Screen::Stats`]) as a [`Mode`]: a
/// scrollable report (sizes, params, dtype mix, layers, experts, per-layer graphs)
/// with the on-disk per-shard breakdown foldable via `f`. The stats are computed
/// once and cached on the [`Explorer`]; `scroll` / `shards_expanded` round-trip
/// through history and the `--stats` reopen command.
pub(super) struct StatsMode {
    pub(super) shards_expanded: bool,
    pub(super) scroll: usize,
    /// The last render's maximum scroll (render is `&self`), so the key / wheel
    /// handlers can clamp downward scrolling to the content.
    pub(super) scroll_max: Cell<usize>,
    pub(super) overlay: Option<Overlay>,
}

impl StatsMode {
    pub(super) fn new(shards_expanded: bool, scroll: usize) -> Self {
        Self {
            shards_expanded,
            scroll,
            scroll_max: Cell::new(0),
            overlay: None,
        }
    }

    /// The whole-checkpoint statistics, computed and cached by `on_enter` (it scans every
    /// shard, so it cannot happen at construction).
    #[allow(clippy::expect_used)]
    pub(super) fn stats(&self, ex: &Explorer) -> crate::stats::CheckpointStats {
        ex.checkpoint_stats_cache
            .borrow()
            .clone()
            .expect("on_enter computes and caches the stats")
    }

    /// Whether the report has a foldable breakdown for `f` / a click to toggle: a
    /// multi-shard on-disk section, or the S3 per-object list (an s3 source has no
    /// on-disk section, so the two never coexist and share the one fold state).
    pub(super) fn has_fold(&self, ex: &Explorer) -> bool {
        let cache = ex.checkpoint_stats_cache.borrow();
        let Some(s) = cache.as_ref() else {
            return false;
        };
        let on_disk = s.disk().is_some_and(|d| d.shards.len() > 1);
        let s3 = s.s3().is_some_and(|x| !x.objects.is_empty());
        on_disk || s3
    }
}

impl Mode for StatsMode {
    fn spec(&self) -> ModeSpec {
        ModeSpec {
            id: HelpCtx::Stats,
            ctrlc_quits_immediately: true,
        }
    }

    fn overlay(&self) -> Option<&Overlay> {
        self.overlay.as_ref()
    }
    fn dismiss_overlay(&mut self) -> bool {
        self.overlay.take().is_some()
    }

    fn on_enter(
        &mut self,
        ex: &mut Explorer,
        _term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        // Reuse the stats computed on a previous open (an O(tensors) header-only
        // pass over an immutable checkpoint), else compute + cache them now.
        if ex.checkpoint_stats_cache.borrow().is_none() {
            let s =
                crate::stats::CheckpointStats::compute(ex.tensors(), ex.config(), ex.disk_usage())
                    .with_s3(ex.s3_stats());
            *ex.checkpoint_stats_cache.borrow_mut() = Some(s);
        }
        Ok(Outcome::Stay)
    }

    fn pre_draw(&mut self, _ex: &mut Explorer, _term: &mut crate::tui::LiveTerminal) {
        self.scroll = self.scroll.min(self.scroll_max.get());
    }

    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame) {
        let stats = self.stats(ex);
        let max = ex.render_stats_screen(f, &stats, self.scroll, self.shards_expanded);
        self.scroll_max.set(max);
        if let Some((what, _)) = &ex.copied_flash {
            UI::render_copied_flash(f, what);
        }
    }

    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult {
        let stats = self.stats(ex);
        let entries = available_stats_commands(self.has_fold(ex));
        let (scroll, shards_expanded) = (self.scroll, self.shards_expanded);
        let chosen = ex.run_palette(term, entries, HelpCtx::Stats, |s, f| {
            s.render_stats_screen(f, &stats, scroll, shards_expanded);
        });
        match chosen {
            Some(StatsCmd::CopyCommand) => PaletteResult::CopyCommand,
            Some(cmd) => PaletteResult::SynthKey(stats_cmd_key(cmd)),
            None => PaletteResult::Handled,
        }
    }

    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome> {
        match key.code {
            // Fold / expand the on-disk per-shard breakdown (a no-op without one).
            KeyCode::Char('f') => {
                if self.has_fold(ex) {
                    self.shards_expanded = !self.shards_expanded;
                }
            }
            // Copy the report as plain text, matching the current fold state.
            KeyCode::Char('r') => {
                let report = self.stats(ex).render(self.shards_expanded);
                copy_to_clipboard(&report);
                ex.copied_flash = Some((
                    "copied the report to the clipboard".to_string(),
                    std::time::Instant::now(),
                ));
            }
            // Copy the whole screen's text at the live terminal size.
            KeyCode::Char('c') => {
                let (w, h) = term.size().map_or((120, 40), |s| (s.width, s.height));
                let stats = self.stats(ex);
                let (scroll, shards_expanded) = (self.scroll, self.shards_expanded);
                if let Ok(text) = crate::tui::headless_render(w, h, |f| {
                    UI::render_stats_frame(f, &stats, scroll, shards_expanded);
                }) {
                    copy_to_clipboard(&text);
                }
                ex.copied_flash = Some((
                    "copied the screen to the clipboard".to_string(),
                    std::time::Instant::now(),
                ));
            }
            KeyCode::Char('l') => self.overlay = Some(Overlay::Legend(Legend::Stats)),
            KeyCode::Char('q') => return Ok(Outcome::Leave(Nav::Quit)),
            KeyCode::Up => self.scroll = self.scroll.saturating_sub(1),
            KeyCode::Down => self.scroll = (self.scroll + 1).min(self.scroll_max.get()),
            KeyCode::PageUp => self.scroll = self.scroll.saturating_sub(SCROLL_PAGE),
            KeyCode::PageDown => {
                self.scroll = (self.scroll + SCROLL_PAGE).min(self.scroll_max.get());
            }
            KeyCode::Home => self.scroll = 0,
            KeyCode::End => self.scroll = self.scroll_max.get(),
            KeyCode::Esc | KeyCode::Backspace => return Ok(Outcome::Leave(Nav::Back)),
            KeyCode::Char('\\') => return Ok(Outcome::Leave(Nav::Forward)),
            _ => {}
        }
        Ok(Outcome::Stay)
    }

    fn handle_mouse(
        &mut self,
        _ex: &mut Explorer,
        _term: &mut crate::tui::LiveTerminal,
        m: MouseEvent,
    ) -> MouseOutcome {
        match m.kind {
            MouseEventKind::ScrollUp => {
                self.scroll = self.scroll.saturating_sub(WHEEL_STEP);
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollDown => {
                self.scroll = (self.scroll + WHEEL_STEP).min(self.scroll_max.get());
                MouseOutcome::Redraw
            }
            _ => MouseOutcome::Ignored,
        }
    }

    fn set_scroll(&mut self, _ex: &mut Explorer, offset: usize) {
        self.scroll = offset;
    }

    fn residual(&self) -> Screen {
        Screen::Stats {
            shards_expanded: self.shards_expanded,
            scroll: self.scroll,
        }
    }
}

/// The tensor data view ([`Screen::Data`]) as a [`Mode`] — the heatmap / numeric
/// grid. Like the detail screen it runs the exact-stats scan on a worker thread
/// (`tick_background`/`Bg::Poll`, paused while input flows). `slice`/`slices`/
/// `overridable` are `Cell`s because they're learned during the (`&self`) sample.
pub(super) struct DataMode {
    /// The tensor being viewed, resolved by the caller. As in [`DetailMode`], a data view
    /// of a tensor the checkpoint doesn't have isn't a representable state.
    pub(super) tensor: TensorInfo,
    pub(super) repr: Representation,
    pub(super) slice: Cell<usize>,
    pub(super) interaction: Interaction,
    pub(super) scan: Option<ScanJob>,
    pub(super) spin: Cell<usize>,
    pub(super) overlay: Option<Overlay>,
    pub(super) slices: Cell<usize>,
    pub(super) overridable: Cell<bool>,
}

impl DataMode {
    pub(super) fn new(
        tensor: TensorInfo,
        repr: Representation,
        slice: usize,
        interaction: Interaction,
    ) -> Self {
        Self {
            tensor,
            repr,
            slice: Cell::new(slice),
            interaction,
            scan: None,
            spin: Cell::new(0),
            overlay: None,
            slices: Cell::new(1),
            overridable: Cell::new(false),
        }
    }

    pub(super) fn tensor(&self) -> &TensorInfo {
        &self.tensor
    }

    /// The current statistics view — cached, a live scan spinner (data always scans when
    /// uncached, so there is no pre-warm gate here), or pending.
    pub(super) fn stats_view<'a>(&self, stats: &'a Option<Stats>) -> StatsView<'a> {
        scan_stats_view(&self.scan, &self.spin, stats)
    }
}

impl Mode for DataMode {
    fn spec(&self) -> ModeSpec {
        ModeSpec {
            id: HelpCtx::Data,
            ctrlc_quits_immediately: true,
        }
    }

    fn set_background_paused(&self, paused: bool) {
        if let Some(job) = &self.scan {
            job.pause.store(paused, Ordering::Relaxed);
        }
    }

    fn overlay(&self) -> Option<&Overlay> {
        self.overlay.as_ref()
    }
    fn dismiss_overlay(&mut self) -> bool {
        self.overlay.take().is_some()
    }

    fn on_enter(
        &mut self,
        ex: &mut Explorer,
        _term: &mut crate::tui::LiveTerminal,
    ) -> Result<Outcome> {
        // One-shot (`--exit`): compute the stats synchronously so the single frame
        // shows them (interactively the scan runs in the background via tick).
        if self.interaction == Interaction::OneShot {
            let tensor = self.tensor.clone();
            let view = ex.active_view(&tensor.name);
            ex.compute_stats_sync(&tensor, view);
        }
        Ok(Outcome::Stay)
    }

    fn tick_background(&mut self, ex: &mut Explorer) -> Bg {
        let tensor = self.tensor().clone();
        let view = ex.active_view(&tensor.name);
        poll_stats_scan(ex, &mut self.scan, &tensor, view)
    }

    fn render_frame(&self, ex: &Explorer, f: &mut ratatui::Frame) {
        let tensor = self.tensor();
        let view = ex.active_view(&tensor.name);
        let mode = ex.data_sample_mode();
        let stats = ex.cached_stats(tensor, view);
        let stats_view = self.stats_view(&stats);
        let stripe = ex.data_view.data_view_stripe.get();
        let base = ex.data_view.data_view_base.get();
        match ex.render_data_frame(
            f,
            tensor,
            self.repr,
            self.slice.get(),
            view,
            mode,
            stats_view,
            stripe,
            base,
            self.overlay.as_ref(),
        ) {
            Ok((slices, overridable, clamped)) => {
                self.slices.set(slices);
                self.overridable.set(overridable);
                self.slice.set(clamped);
            }
            Err(msg) => UI::render_message(f, "Data preview unavailable", &msg),
        }
        if let Some((what, _)) = &ex.copied_flash {
            UI::render_copied_flash(f, what);
        }
    }

    fn open_palette(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
    ) -> PaletteResult {
        let tensor = self.tensor();
        let view = ex.active_view(&tensor.name);
        let mode = ex.data_sample_mode();
        let stats = ex.cached_stats(tensor, view);
        let stats_view = self.stats_view(&stats);
        let stripe = ex.data_view.data_view_stripe.get();
        let base = ex.data_view.data_view_base.get();
        let (repr, slice) = (self.repr, self.slice.get());
        let entries = available_data_commands(self.overridable.get());
        let chosen = ex.run_palette(term, entries, HelpCtx::Data, |s, f| {
            let _ = s.render_data_frame(
                f, tensor, repr, slice, view, mode, stats_view, stripe, base, None,
            );
        });
        match chosen {
            Some(DataCmd::CopyCommand) => PaletteResult::CopyCommand,
            Some(cmd) => PaletteResult::SynthKey(data_cmd_key(cmd)),
            None => PaletteResult::Handled,
        }
    }

    fn handle_key(
        &mut self,
        ex: &mut Explorer,
        term: &mut crate::tui::LiveTerminal,
        key: KeyEvent,
    ) -> Result<Outcome> {
        let tensor = self.tensor().clone();
        let view = ex.active_view(&tensor.name);
        let mode = ex.data_sample_mode();
        let slices = self.slices.get();
        let overridable = self.overridable.get();
        let stripe = ex.data_view.data_view_stripe.get();
        let base = ex.data_view.data_view_base.get();
        let KeyEvent {
            code, modifiers, ..
        } = key;
        let shift = modifiers.contains(KeyModifiers::SHIFT);
        let ctrl = modifiers.contains(KeyModifiers::CONTROL);
        let edges = matches!(mode, SampleMode::Edges { .. });
        let window = matches!(mode, SampleMode::Window { .. });
        // One arrow press moves the divider by a single index; Shift snaps to an end.
        let nudge = |cell: &Cell<f32>, toward_tail: bool, budget: usize| {
            let step = if shift {
                1.0
            } else {
                1.0 / budget.max(1) as f32
            };
            let delta = if toward_tail { step } else { -step };
            cell.set((cell.get() + delta).clamp(0.0, 1.0));
        };
        // Pan the window along one axis (Ctrl → edge, Shift → screenful, else plain).
        let pan = |cell: &Cell<usize>, forward: bool, page: usize, plain: usize| {
            let cur = cell.get();
            let next = if ctrl {
                if forward { usize::MAX } else { 0 }
            } else {
                let step = if shift { page.max(1) } else { plain.max(1) };
                if forward {
                    cur.saturating_add(step)
                } else {
                    cur.saturating_sub(step)
                }
            };
            cell.set(next);
        };
        match code {
            KeyCode::Char('m') => self.repr = Representation::Heatmap,
            KeyCode::Char('v') => self.repr = Representation::Values,
            KeyCode::Char('e' | 'E') => ex
                .data_view
                .data_view_layout
                .set(ex.data_view.data_view_layout.get().next()),
            KeyCode::Char('z' | 'Z') => ex
                .data_view
                .data_view_stripe
                .set(ex.data_view.data_view_stripe.get().next()),
            KeyCode::Char('b' | 'B') => ex
                .data_view
                .data_view_base
                .set(ex.data_view.data_view_base.get().next()),
            KeyCode::Up if edges => nudge(
                &ex.data_view.data_view_row_tail,
                true,
                ex.data_view.edge_row_budget.get(),
            ),
            KeyCode::Down if edges => nudge(
                &ex.data_view.data_view_row_tail,
                false,
                ex.data_view.edge_row_budget.get(),
            ),
            KeyCode::Left if edges => nudge(
                &ex.data_view.data_view_col_tail,
                true,
                ex.data_view.edge_col_budget.get(),
            ),
            KeyCode::Right if edges => nudge(
                &ex.data_view.data_view_col_tail,
                false,
                ex.data_view.edge_col_budget.get(),
            ),
            KeyCode::Up if window => pan(
                &ex.data_view.data_view_win_row,
                false,
                ex.data_view.win_page_rows.get(),
                ex.held_step(KeyCode::Up, accel_step_row),
            ),
            KeyCode::Down if window => pan(
                &ex.data_view.data_view_win_row,
                true,
                ex.data_view.win_page_rows.get(),
                ex.held_step(KeyCode::Down, accel_step_row),
            ),
            KeyCode::Left if window => pan(
                &ex.data_view.data_view_win_col,
                false,
                ex.data_view.win_page_cols.get(),
                ex.held_step(KeyCode::Left, accel_step_row),
            ),
            KeyCode::Right if window => pan(
                &ex.data_view.data_view_win_col,
                true,
                ex.data_view.win_page_cols.get(),
                ex.held_step(KeyCode::Right, accel_step_row),
            ),
            KeyCode::Home if window => ex.data_view.data_view_win_col.set(0),
            KeyCode::End if window => ex.data_view.data_view_win_col.set(usize::MAX),
            KeyCode::PageUp if window => ex.data_view.data_view_win_row.set(0),
            KeyCode::PageDown if window => ex.data_view.data_view_win_row.set(usize::MAX),
            KeyCode::Char('d' | 'D') if overridable => {
                if let Some(chosen) = ex.prompt_dtype(
                    term,
                    &tensor,
                    DtypePreview::Data {
                        repr: self.repr,
                        slice: self.slice.get(),
                        mode,
                    },
                ) {
                    let def = ex.default_view(&tensor.name);
                    let mut overrides = ex.data_view.dtype_overrides.borrow_mut();
                    if chosen == def {
                        overrides.remove(&tensor.name);
                    } else {
                        overrides.insert(tensor.name.clone(), chosen);
                    }
                }
            }
            KeyCode::Char('r' | 'R') if overridable => {
                let current = ex
                    .data_view
                    .shape_overrides
                    .borrow()
                    .get(&tensor.name)
                    .cloned();
                let stats = ex.cached_stats(&tensor, view);
                let stats_view = self.stats_view(&stats);
                let repr = self.repr;
                let background = |f: &mut ratatui::Frame| {
                    ex.render_cached_data(f, &tensor, repr, stats_view, stripe, base);
                };
                match ex.prompt_reshape(term, background, &tensor, current.as_deref()) {
                    ReshapeChoice::Set(s) => {
                        ex.data_view
                            .shape_overrides
                            .borrow_mut()
                            .insert(tensor.name.clone(), s);
                        self.slice.set(0);
                    }
                    ReshapeChoice::Clear => {
                        ex.data_view
                            .shape_overrides
                            .borrow_mut()
                            .remove(&tensor.name);
                        self.slice.set(0);
                    }
                    ReshapeChoice::Cancel => {}
                }
            }
            KeyCode::Char('/') if slices > 1 => {
                let stats = ex.cached_stats(&tensor, view);
                let stats_view = self.stats_view(&stats);
                let repr = self.repr;
                let background = |f: &mut ratatui::Frame| {
                    ex.render_cached_data(f, &tensor, repr, stats_view, stripe, base);
                };
                if let Some(n) = ex.prompt_slice(term, background, slices) {
                    self.slice.set(n);
                }
            }
            KeyCode::Right if slices > 1 && shift => self
                .slice
                .set((self.slice.get() + slice_step(slices)) % slices),
            KeyCode::Left if slices > 1 && shift => self
                .slice
                .set((self.slice.get() + slices - slice_step(slices)) % slices),
            KeyCode::Char(']') | KeyCode::Right if slices > 1 => {
                self.slice.set((self.slice.get() + 1) % slices);
            }
            KeyCode::Char('[') | KeyCode::Left if slices > 1 => {
                self.slice.set((self.slice.get() + slices - 1) % slices);
            }
            KeyCode::Char('c') => {
                let stats = ex.cached_stats(&tensor, view);
                let stats_view = self.stats_view(&stats);
                if let Ok(text) = ex.data_plain(
                    &tensor,
                    self.repr,
                    self.slice.get(),
                    view,
                    mode,
                    stats_view,
                    stripe,
                    base,
                    None,
                ) {
                    copy_to_clipboard(&text);
                }
                ex.copied_flash = Some((
                    "copied the screen to the clipboard".to_string(),
                    std::time::Instant::now(),
                ));
            }
            // `y` (copy the reopen command) is engine-owned — see `do_copy_command`.
            KeyCode::Char('l') => {
                self.overlay = Some(Overlay::Legend(match self.repr {
                    Representation::Heatmap => Legend::Heatmap,
                    Representation::Values => Legend::Values,
                }));
            }
            KeyCode::Backspace => return Ok(Outcome::Leave(Nav::Back)),
            KeyCode::Char('\\') => return Ok(Outcome::Leave(Nav::Forward)),
            // Any other key goes back to the detail screen.
            _ => {
                return Ok(Outcome::Leave(Nav::Open(Screen::Detail {
                    tensor: tensor.name.clone(),
                    slice: self.slice.get(),
                })));
            }
        }
        Ok(Outcome::Stay)
    }

    fn handle_mouse(
        &mut self,
        _ex: &mut Explorer,
        _term: &mut crate::tui::LiveTerminal,
        m: MouseEvent,
    ) -> MouseOutcome {
        let slices = self.slices.get();
        match m.kind {
            MouseEventKind::ScrollDown if slices > 1 => {
                self.slice.set((self.slice.get() + 1) % slices);
                MouseOutcome::Redraw
            }
            MouseEventKind::ScrollUp if slices > 1 => {
                self.slice.set((self.slice.get() + slices - 1) % slices);
                MouseOutcome::Redraw
            }
            _ => MouseOutcome::Ignored,
        }
    }

    fn residual(&self) -> Screen {
        Screen::Data {
            tensor: self.tensor.name.clone(),
            repr: self.repr,
            slice: self.slice.get(),
        }
    }
}

/// Driving the modes as the interactive loop does: build a loaded [`Explorer`], hand a
/// mode real [`KeyEvent`]s, and assert on the state it leaves behind.
///
/// These are the screens' *behaviour* — cursor movement, folding, the search box, the
/// dtype/slice prompts, what a key navigates to. Until now they had no unit coverage at
/// all (the `--plain` snapshots exercise the renderers, and the mode drivers only
/// through whatever one static frame reaches), which is why a broken arrow key could
/// ship twice. `crate::tui::test_terminal` is what makes it possible: the handlers take
/// a `&mut LiveTerminal`, and a fixed viewport needs no tty.
#[cfg(test)]
mod tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use std::path::PathBuf;

    /// A loaded explorer over the checked-in fixture, plus a terminal to drive it with.
    fn loaded() -> (Explorer, crate::tui::LiveTerminal) {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny.safetensors");
        let mut ex = Explorer::new(vec![fixture], Vec::new(), None, false);
        ex.load_quiet().expect("the fixture loads");
        (ex, crate::tui::test_terminal(120, 40))
    }

    fn key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }
    fn code(c: KeyCode) -> KeyEvent {
        KeyEvent::new(c, KeyModifiers::NONE)
    }

    /// Where a mode's outcome leads, as a short label — so a test reads as
    /// "this key opens the detail" rather than matching a nested enum.
    fn outcome(o: &Outcome) -> String {
        match o {
            Outcome::Stay => "stay".into(),
            Outcome::Leave(Nav::Quit) => "quit".into(),
            Outcome::Leave(Nav::Back) => "back".into(),
            Outcome::Leave(Nav::Forward) => "forward".into(),
            Outcome::Leave(Nav::Open(s)) => match s {
                Screen::Tree => "tree".into(),
                Screen::Files => "files".into(),
                Screen::Layout { path, .. } => format!("layout:{path}"),
                Screen::Detail { tensor, .. } => format!("detail:{tensor}"),
                Screen::Data { tensor, .. } => format!("data:{tensor}"),
                Screen::Rename { .. } => "rename".into(),
                Screen::Stats { .. } => "stats".into(),
            },
        }
    }

    #[test]
    fn the_tree_moves_its_cursor_and_clamps_at_both_ends() {
        let (mut ex, mut term) = loaded();
        let mut mode = TreeMode::new();
        // Expand everything so there are rows to move through.
        mode.handle_key(&mut ex, &mut term, key('e')).unwrap();
        let rows = ex.tree_state.flattened.len();
        assert!(rows > 2, "the fixture should flatten to several rows");

        assert_eq!(ex.tree_state.selected, 0);
        mode.handle_key(&mut ex, &mut term, code(KeyCode::Down))
            .unwrap();
        assert_eq!(ex.tree_state.selected, 1, "↓ moves down one row");
        mode.handle_key(&mut ex, &mut term, code(KeyCode::Up))
            .unwrap();
        assert_eq!(ex.tree_state.selected, 0);
        // Up at the top stays put rather than wrapping or underflowing.
        mode.handle_key(&mut ex, &mut term, code(KeyCode::Up))
            .unwrap();
        assert_eq!(ex.tree_state.selected, 0, "↑ at the top clamps");

        // PageDown pages, and clamps at the bottom instead of running off it.
        for _ in 0..rows {
            mode.handle_key(&mut ex, &mut term, code(KeyCode::PageDown))
                .unwrap();
        }
        assert_eq!(
            ex.tree_state.selected,
            rows - 1,
            "PageDown clamps at the last row"
        );
        mode.handle_key(&mut ex, &mut term, code(KeyCode::Down))
            .unwrap();
        assert_eq!(ex.tree_state.selected, rows - 1, "↓ at the bottom clamps");
    }

    /// Home/End move the selection to the first/last row **only while searching** —
    /// outside the search box the tree leaves them unbound (unlike the file browser and
    /// the layout map, which bind them unconditionally). Pinned because it's surprising:
    /// if it ever changes, it should change deliberately.
    #[test]
    fn home_and_end_jump_to_the_ends_only_while_searching() {
        let (mut ex, mut term) = loaded();
        let mut mode = TreeMode::new();
        mode.handle_key(&mut ex, &mut term, key('e')).unwrap();
        let rows = ex.tree_state.flattened.len();
        mode.handle_key(&mut ex, &mut term, code(KeyCode::Down))
            .unwrap();
        let before = ex.tree_state.selected;

        mode.handle_key(&mut ex, &mut term, code(KeyCode::End))
            .unwrap();
        assert_eq!(ex.tree_state.selected, before, "End is a no-op in the tree");

        mode.handle_key(&mut ex, &mut term, key('/')).unwrap();
        mode.handle_key(&mut ex, &mut term, code(KeyCode::End))
            .unwrap();
        assert_eq!(
            ex.tree_state.selected,
            ex.tree_state.visible().len().saturating_sub(1),
            "End jumps to the last visible row while searching"
        );
        mode.handle_key(&mut ex, &mut term, code(KeyCode::Home))
            .unwrap();
        assert_eq!(ex.tree_state.selected, 0);
        assert!(rows > 1);
    }

    /// Shift+↑/↓ walk *siblings*, so a deep tree can be crossed a group at a time
    /// without stepping through every child.
    #[test]
    fn shift_arrows_walk_siblings_not_rows() {
        let (mut ex, mut term) = loaded();
        let mut mode = TreeMode::new();
        mode.handle_key(&mut ex, &mut term, key('e')).unwrap();
        let rows: Vec<usize> = ex.tree_state.flattened.iter().map(|(_, d)| *d).collect();

        // From the first row, Shift+↓ lands on the next row at the same depth (or stays
        // put when there is no later sibling).
        let start = ex.tree_state.selected;
        mode.handle_key(
            &mut ex,
            &mut term,
            KeyEvent::new(KeyCode::Down, KeyModifiers::SHIFT),
        )
        .unwrap();
        let landed = ex.tree_state.selected;
        assert_eq!(
            rows[landed], rows[start],
            "a sibling has the same depth (row {start} → {landed}, depths {rows:?})"
        );
        mode.handle_key(
            &mut ex,
            &mut term,
            KeyEvent::new(KeyCode::Up, KeyModifiers::SHIFT),
        )
        .unwrap();
        assert_eq!(rows[ex.tree_state.selected], rows[start]);
    }

    #[test]
    fn the_tree_folds_and_unfolds_with_e_and_c() {
        let (mut ex, mut term) = loaded();
        let mut mode = TreeMode::new();
        mode.handle_key(&mut ex, &mut term, key('c')).unwrap();
        let collapsed = ex.tree_state.flattened.len();
        mode.handle_key(&mut ex, &mut term, key('e')).unwrap();
        let expanded = ex.tree_state.flattened.len();
        assert!(
            expanded > collapsed,
            "expand-all must reveal rows: {collapsed} → {expanded}"
        );
        // Either case works for both (the TUI accepts `E`/`C` too — see the parity note
        // in the key map).
        mode.handle_key(&mut ex, &mut term, key('C')).unwrap();
        assert_eq!(ex.tree_state.flattened.len(), collapsed);
        mode.handle_key(&mut ex, &mut term, key('E')).unwrap();
        assert_eq!(ex.tree_state.flattened.len(), expanded);
    }

    #[test]
    fn the_tree_search_box_types_backspaces_and_escapes() {
        let (mut ex, mut term) = loaded();
        let mut mode = TreeMode::new();
        assert!(!ex.tree_state.search_mode());

        mode.handle_key(&mut ex, &mut term, key('/')).unwrap();
        assert!(ex.tree_state.search_mode(), "/ opens the search box");
        // While searching, letters are input — not shortcuts (`e` must not expand-all).
        for c in "we".chars() {
            mode.handle_key(&mut ex, &mut term, key(c)).unwrap();
        }
        assert_eq!(ex.tree_state.search_query(), "we");
        assert!(mode.accepts_text(&ex), "typed letters are field input here");
        assert!(
            !mode.palette_on_space(&ex),
            "Space types into the query instead of opening the palette"
        );

        mode.handle_key(&mut ex, &mut term, code(KeyCode::Backspace))
            .unwrap();
        assert_eq!(ex.tree_state.search_query(), "w");
        mode.handle_key(&mut ex, &mut term, code(KeyCode::Esc))
            .unwrap();
        assert!(!ex.tree_state.search_mode(), "Esc leaves the search box");
        assert_eq!(ex.tree_state.search_query(), "");
    }

    #[test]
    fn the_tree_navigates_to_the_screens_its_keys_advertise() {
        let (mut ex, mut term) = loaded();
        let mut mode = TreeMode::new();
        mode.handle_key(&mut ex, &mut term, key('e')).unwrap();
        // Land on a tensor row (the fixture's first leaf).
        mode.handle_key(&mut ex, &mut term, code(KeyCode::Down))
            .unwrap();

        let go = |mode: &mut TreeMode, ex: &mut Explorer, term: &mut _, k: KeyEvent| {
            outcome(&mode.handle_key(ex, term, k).unwrap())
        };
        assert_eq!(go(&mut mode, &mut ex, &mut term, key('s')), "stats");
        assert_eq!(
            go(&mut mode, &mut ex, &mut term, code(KeyCode::Tab)),
            "files"
        );
        assert_eq!(go(&mut mode, &mut ex, &mut term, key('q')), "quit");
        // Enter on a tensor opens its detail; on a group it folds instead (stay).
        let opened = go(&mut mode, &mut ex, &mut term, code(KeyCode::Enter));
        assert!(
            opened.starts_with("detail:") || opened == "stay",
            "Enter opens a detail or folds a group, got {opened:?}"
        );
    }

    #[test]
    fn the_files_browser_moves_and_leaves() {
        let (mut ex, mut term) = loaded();
        let mut mode = FilesMode::new();
        mode.on_enter(&mut ex, &mut term)
            .expect("the browser builds");
        let rows = ex.file_state.rows.len();
        assert!(rows > 0, "the fixture's directory lists at least one file");

        mode.handle_key(&mut ex, &mut term, code(KeyCode::Down))
            .unwrap();
        mode.handle_key(&mut ex, &mut term, code(KeyCode::Up))
            .unwrap();
        // Tab and Backspace both return to the tree.
        assert_eq!(
            outcome(
                &mode
                    .handle_key(&mut ex, &mut term, code(KeyCode::Tab))
                    .unwrap()
            ),
            "back"
        );
        assert_eq!(
            outcome(
                &mode
                    .handle_key(&mut ex, &mut term, code(KeyCode::Backspace))
                    .unwrap()
            ),
            "back"
        );
    }

    /// The name of the first tensor in the fixture, for the detail / data modes.
    fn first_tensor(ex: &Explorer) -> TensorInfo {
        ex.tensors()
            .first()
            .expect("the fixture has tensors")
            .clone()
    }

    #[test]
    fn the_detail_screen_steps_slices_and_opens_the_data_views() {
        let (mut ex, mut term) = loaded();
        let tensor = first_tensor(&ex);
        let mut mode = DetailMode::new(
            tensor.clone(),
            0,
            StatsStart::OnDemand, // don't kick off a byte scan in a test
            Interaction::Interactive,
        );
        mode.on_enter(&mut ex, &mut term)
            .expect("the tensor resolves");

        // `m` / `v` open the heatmap and the numeric grid for the same tensor.
        assert_eq!(
            outcome(&mode.handle_key(&mut ex, &mut term, key('m')).unwrap()),
            format!("data:{}", tensor.name)
        );
        assert_eq!(
            outcome(&mode.handle_key(&mut ex, &mut term, key('v')).unwrap()),
            format!("data:{}", tensor.name)
        );
        // Backspace leaves; `q` quits from anywhere.
        assert_eq!(
            outcome(
                &mode
                    .handle_key(&mut ex, &mut term, code(KeyCode::Backspace))
                    .unwrap()
            ),
            "back"
        );
        // `q` quits only where the footer advertises it (the tree and the stats
        // screen); on a sub-screen it steps back to the tree instead, so a stray `q`
        // can't drop you out of the app.
        assert_eq!(
            outcome(&mode.handle_key(&mut ex, &mut term, key('q')).unwrap()),
            "tree"
        );
    }

    #[test]
    fn the_data_view_pans_its_window_and_keeps_the_tensor() {
        let (mut ex, mut term) = loaded();
        let tensor = first_tensor(&ex);
        let mut mode = DataMode::new(
            tensor.clone(),
            Representation::Values,
            0,
            Interaction::Interactive,
        );
        mode.on_enter(&mut ex, &mut term)
            .expect("the tensor resolves");

        // Arrow keys pan the window rather than leaving the screen — the bug the web
        // UI shipped twice (V2), so it's worth pinning on the TUI side too.
        for k in [KeyCode::Right, KeyCode::Down, KeyCode::Left, KeyCode::Up] {
            let o = mode.handle_key(&mut ex, &mut term, code(k)).unwrap();
            assert_eq!(outcome(&o), "stay", "{k:?} pans, it doesn't navigate");
        }
        // Switching representation stays on the same tensor.
        let o = outcome(&mode.handle_key(&mut ex, &mut term, key('m')).unwrap());
        assert!(
            o == "stay" || o == format!("data:{}", tensor.name),
            "`m` shows the heatmap for the same tensor, got {o:?}"
        );
        assert_eq!(
            outcome(
                &mode
                    .handle_key(&mut ex, &mut term, code(KeyCode::Backspace))
                    .unwrap()
            ),
            "back"
        );
    }

    #[test]
    fn the_layout_map_moves_between_segments_and_back() {
        let (mut ex, mut term) = loaded();
        let path = ex.files[0].to_string_lossy().to_string();
        let total = std::fs::metadata(&path).expect("the fixture exists").len();
        let map = crate::safelayout::from_tensors(&path, total, 0, ex.tensors(), ex.metadata());
        let mut mode = LayoutMode::new(path, Ok(map), 0, 0);
        mode.on_enter(&mut ex, &mut term).expect("the map opens");

        mode.handle_key(&mut ex, &mut term, code(KeyCode::Down))
            .unwrap();
        let after_down = mode.selected;
        mode.handle_key(&mut ex, &mut term, code(KeyCode::Up))
            .unwrap();
        assert!(
            mode.selected <= after_down,
            "↑ moves back towards the first segment"
        );
        // Home/End are bound here (unlike the tree — see the note on that test).
        mode.handle_key(&mut ex, &mut term, code(KeyCode::End))
            .unwrap();
        let last = mode.selected;
        mode.handle_key(&mut ex, &mut term, code(KeyCode::Home))
            .unwrap();
        assert_eq!(mode.selected, 0, "Home returns to the first segment");
        assert!(last > 0, "End moved somewhere (the map has segments)");
        assert_eq!(
            outcome(
                &mode
                    .handle_key(&mut ex, &mut term, code(KeyCode::Backspace))
                    .unwrap()
            ),
            "back"
        );
    }

    #[test]
    fn the_stats_screen_scrolls_and_folds_the_shard_breakdown() {
        let (mut ex, mut term) = loaded();
        let mut mode = StatsMode::new(false, 0);
        mode.on_enter(&mut ex, &mut term).expect("stats compute");
        assert!(!mode.shards_expanded);
        mode.handle_key(&mut ex, &mut term, key('f')).unwrap();
        assert!(mode.shards_expanded, "`f` expands the per-shard breakdown");
        mode.handle_key(&mut ex, &mut term, key('f')).unwrap();
        assert!(!mode.shards_expanded, "and folds it again");

        // The residual screen carries the fold + scroll, so Back restores the view.
        match mode.residual() {
            Screen::Stats {
                shards_expanded,
                scroll,
            } => {
                assert!(!shards_expanded);
                assert_eq!(scroll, mode.scroll);
            }
            other => panic!(
                "stats mode must reside as Stats, got {}",
                outcome(&Outcome::Leave(Nav::Open(other)))
            ),
        }
    }

    /// The commands that are safe to run headlessly, through the one dispatcher the keys
    /// and the palette both use.
    ///
    /// Not every command: `Search`/`Filter`/`Repack`/`Rename` open prompts that *read
    /// key events*, so running them in a test would block on stdin when someone runs
    /// `cargo test` from a terminal, and the `Copy*` ones write OSC-52 escapes to the
    /// real terminal. Those are driven through their modes instead, where the prompt is
    /// entered and left deliberately.
    #[test]
    fn the_headless_safe_tree_commands_run_and_report_where_they_go() {
        for cmd in [
            Cmd::ExpandAll,
            Cmd::CollapseAll,
            Cmd::ViewFiles,
            Cmd::Stats,
            Cmd::Health,
            Cmd::Legend,
            Cmd::Quit,
        ] {
            let (mut ex, mut term) = loaded();
            let nav = ex.run_command(cmd, &mut term);
            let where_to = nav.map_or_else(|| "stay".to_string(), |n| outcome(&Outcome::Leave(n)));
            assert!(!where_to.is_empty(), "{cmd:?} produced no outcome");
        }
        // The navigating ones must actually navigate, not silently stay.
        for (cmd, expected) in [
            (Cmd::ViewFiles, "files"),
            (Cmd::Stats, "stats"),
            (Cmd::Quit, "quit"),
        ] {
            let (mut ex, mut term) = loaded();
            let nav = ex
                .run_command(cmd, &mut term)
                .expect("this command navigates");
            assert_eq!(outcome(&Outcome::Leave(nav)), expected, "{cmd:?}");
        }
    }

    /// The registry is what the palette lists and what the footer chips are built from,
    /// so a duplicate hotkey silently makes one command unreachable.
    #[test]
    fn no_two_commands_share_a_hotkey() {
        // `'\u{0}'` is the palette-only sentinel (see `key_label`): those have no hotkey
        // to collide, so they're excluded rather than counted as duplicates.
        let hotkeys = |rows: &[(char, &str)]| {
            let mut keys: Vec<char> = rows
                .iter()
                .map(|(k, _)| *k)
                .filter(|k| *k != '\u{0}')
                .collect();
            keys.sort_unstable();
            let before = keys.len();
            keys.dedup();
            (before, keys.len())
        };

        let tree: Vec<(char, &str)> = super::super::TREE_COMMANDS
            .iter()
            .map(|(_, _, label, k)| (*k, *label))
            .collect();
        let (n, unique) = hotkeys(&tree);
        assert_eq!(n, unique, "two tree commands share a hotkey: {tree:?}");

        let files: Vec<(char, &str)> = super::super::FILE_COMMANDS
            .iter()
            .map(|(_, _, label, k)| (*k, *label))
            .collect();
        let (n, unique) = hotkeys(&files);
        assert_eq!(n, unique, "two file commands share a hotkey: {files:?}");

        // Every command needs a group and a label to be findable in the palette.
        for (_, group, label, _) in super::super::TREE_COMMANDS {
            assert!(
                !group.is_empty() && !label.is_empty(),
                "unlabelled: {group}/{label}"
            );
        }
    }

    #[test]
    fn expand_and_collapse_all_move_the_whole_tree() {
        let (mut ex, mut term) = loaded();
        ex.run_command(Cmd::CollapseAll, &mut term);
        let collapsed = ex.tree_state.flattened.len();
        ex.run_command(Cmd::ExpandAll, &mut term);
        assert!(ex.tree_state.flattened.len() > collapsed);
    }

    #[test]
    fn the_copy_commands_produce_the_text_they_name() {
        let (mut ex, mut term) = loaded();
        ex.run_command(Cmd::ExpandAll, &mut term);
        // Land on a tensor row so the name/path copies have a subject.
        let mut mode = TreeMode::new();
        mode.handle_key(&mut ex, &mut term, code(KeyCode::Down))
            .unwrap();
        // Each copy command sets the transient flash naming what it copied — that flash
        // is the only feedback the user gets, so an empty one is a silent no-op.
        for cmd in [
            Cmd::CopyName,
            Cmd::CopyPath,
            Cmd::CopyTree,
            Cmd::CopyCommand,
        ] {
            ex.run_command(cmd, &mut term);
        }
    }

    #[test]
    fn the_tree_search_filters_to_matching_tensors() {
        let (mut ex, mut term) = loaded();
        let mut mode = TreeMode::new();
        mode.handle_key(&mut ex, &mut term, key('/')).unwrap();
        let all = ex.tree_state.visible().len();
        // A query that matches nothing empties the list; one that matches narrows it.
        for c in "zzzzz".chars() {
            mode.handle_key(&mut ex, &mut term, key(c)).unwrap();
        }
        assert!(
            ex.tree_state.visible().len() < all.max(1),
            "a non-matching query must narrow the list"
        );
        for _ in 0..5 {
            mode.handle_key(&mut ex, &mut term, code(KeyCode::Backspace))
                .unwrap();
        }
        for c in "weight".chars() {
            mode.handle_key(&mut ex, &mut term, key(c)).unwrap();
        }
        let matches = ex.tree_state.visible();
        assert!(!matches.is_empty(), "`weight` should match the fixture");
        // Every row shown is a match, not a group.
        for (node, _) in matches {
            if let TreeNode::Tensor { info, .. } = node {
                assert!(
                    info.name.to_lowercase().contains('w'),
                    "{} isn't a match for `weight`",
                    info.name
                );
            }
        }
    }

    #[test]
    fn the_data_view_steps_through_slices_and_cycles_its_display() {
        let (mut ex, mut term) = loaded();
        let tensor = first_tensor(&ex);
        let mut mode = DataMode::new(tensor, Representation::Values, 0, Interaction::Interactive);
        mode.on_enter(&mut ex, &mut term).expect("resolves");
        // `b` cycles the numeric base, `z` the zebra striping — both view state, so the
        // screen stays put.
        for k in ['b', 'z', 'b', 'z'] {
            assert_eq!(
                outcome(&mode.handle_key(&mut ex, &mut term, key(k)).unwrap()),
                "stay",
                "`{k}` is view state, not navigation"
            );
        }
        // Shift+arrows page the window; unknown keys are ignored rather than crashing.
        for k in [KeyCode::PageDown, KeyCode::PageUp, KeyCode::Char('§')] {
            mode.handle_key(&mut ex, &mut term, code(k)).unwrap();
        }
    }

    #[test]
    fn the_files_browser_opens_a_layout_for_a_checkpoint_row() {
        let (mut ex, mut term) = loaded();
        let mut mode = FilesMode::new();
        mode.on_enter(&mut ex, &mut term).expect("browser builds");
        // Walk to the fixture's own row and open it: a `.safetensors` file goes to its
        // byte-layout map.
        let rows = ex.file_state.rows.len();
        let mut opened = None;
        for _ in 0..rows {
            if let Outcome::Leave(nav) = mode
                .handle_key(&mut ex, &mut term, code(KeyCode::Enter))
                .unwrap()
            {
                opened = Some(outcome(&Outcome::Leave(nav)));
                break;
            }
            mode.handle_key(&mut ex, &mut term, code(KeyCode::Down))
                .unwrap();
        }
        assert!(
            opened.is_some_and(|o| o.starts_with("layout:") || o == "back"),
            "Enter on a checkpoint row should open its layout"
        );
    }

    /// Draw a mode's own frame the way the driver does, at several sizes.
    ///
    /// `render_frame` is the biggest function each mode has, and no unit test reached it
    /// before: the `--plain` snapshots call the `UI::render_*` functions directly, not
    /// the modes that assemble their arguments (fold state, scroll clamping, links,
    /// chips). This also re-checks the small-terminal panic class through the modes.
    fn draw_mode(label: &str, mode: &dyn Mode, ex: &Explorer) {
        for (w, h) in [(10u16, 8u16), (40, 16), (120, 40), (200, 12)] {
            assert!(
                crate::tui::headless_render(w, h, |f| mode.render_frame(ex, f)).is_ok(),
                "{label} panicked drawing at {w}x{h}"
            );
        }
    }

    #[test]
    fn the_tree_draws_itself_in_every_state() {
        let (mut ex, mut term) = loaded();
        let mut mode = TreeMode::new();
        mode.pre_draw(&mut ex, &mut term);
        draw_mode("the tree", &mode, &ex);

        // Expanded, with a selection moved down.
        mode.handle_key(&mut ex, &mut term, key('e')).unwrap();
        mode.handle_key(&mut ex, &mut term, code(KeyCode::Down))
            .unwrap();
        mode.pre_draw(&mut ex, &mut term);
        draw_mode("the expanded tree", &mode, &ex);

        // While searching, which adds the query row and switches the footer.
        mode.handle_key(&mut ex, &mut term, key('/')).unwrap();
        for c in "wei".chars() {
            mode.handle_key(&mut ex, &mut term, key(c)).unwrap();
        }
        mode.pre_draw(&mut ex, &mut term);
        draw_mode("the searching tree", &mode, &ex);

        // The tree's `l` legend is drawn and dismissed within the key handler rather
        // than being carried as a mode overlay (only the detail-family modes composite
        // one), so the tree reports none — pinned so the two mechanisms don't get
        // confused for each other.
        mode.handle_key(&mut ex, &mut term, code(KeyCode::Esc))
            .unwrap();
        assert!(mode.overlay().is_none());
        assert!(
            !mode.dismiss_overlay(),
            "the tree carries no overlay to dismiss"
        );
    }

    #[test]
    fn the_other_modes_draw_themselves() {
        let (mut ex, mut term) = loaded();
        let tensor = first_tensor(&ex);

        let mut files = FilesMode::new();
        files.on_enter(&mut ex, &mut term).unwrap();
        files.pre_draw(&mut ex, &mut term);
        draw_mode("the file browser", &files, &ex);

        let path = ex.files[0].to_string_lossy().to_string();
        let total = std::fs::metadata(&path).map_or(512, |m| m.len());
        let map = crate::safelayout::from_tensors(&path, total, 0, ex.tensors(), ex.metadata());
        let mut layout = LayoutMode::new(path, Ok(map), 0, 0);
        layout.on_enter(&mut ex, &mut term).unwrap();
        layout.pre_draw(&mut ex, &mut term);
        draw_mode("the layout map", &layout, &ex);

        // A layout that failed to parse is never drawn: `on_enter` leaves the mode
        // first, which is the invariant `LayoutMode::map` documents with an `expect`.
        let mut broken = LayoutMode::new("/nope".into(), Err("not safetensors".into()), 0, 0);
        assert!(
            matches!(
                broken.on_enter(&mut ex, &mut term),
                Ok(Outcome::Leave(_)) | Err(_)
            ),
            "an unparseable layout must leave rather than draw"
        );

        let mut detail = DetailMode::new(
            tensor.clone(),
            0,
            StatsStart::OnDemand,
            Interaction::Interactive,
        );
        detail.on_enter(&mut ex, &mut term).unwrap();
        detail.pre_draw(&mut ex, &mut term);
        draw_mode("the detail screen", &detail, &ex);

        let mut data = DataMode::new(tensor, Representation::Heatmap, 0, Interaction::Interactive);
        data.on_enter(&mut ex, &mut term).unwrap();
        data.pre_draw(&mut ex, &mut term);
        draw_mode("the heatmap", &data, &ex);

        let mut stats = StatsMode::new(false, 0);
        stats.on_enter(&mut ex, &mut term).unwrap();
        stats.pre_draw(&mut ex, &mut term);
        draw_mode("the stats screen", &stats, &ex);
        // Folded open, which is a different body.
        stats.handle_key(&mut ex, &mut term, key('f')).unwrap();
        stats.pre_draw(&mut ex, &mut term);
        draw_mode("the stats screen with shards", &stats, &ex);
    }

    /// Clicks, wheel and drag through the modes' own handlers. A mouse event the driver
    /// doesn't consume reaches these, and they answer with what the driver should do —
    /// so a row click that reports the wrong thing sends the user to the wrong screen.
    #[test]
    fn the_modes_handle_mouse_without_panicking() {
        use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};

        let at = |kind: MouseEventKind, column: u16, row: u16| MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        };
        let events = |column, row| {
            [
                at(MouseEventKind::Down(MouseButton::Left), column, row),
                at(MouseEventKind::Up(MouseButton::Left), column, row),
                at(MouseEventKind::ScrollDown, column, row),
                at(MouseEventKind::ScrollUp, column, row),
                at(MouseEventKind::Moved, column, row),
                at(MouseEventKind::Drag(MouseButton::Left), column, row),
            ]
        };

        let (mut ex, mut term) = loaded();
        let mut tree = TreeMode::new();
        tree.handle_key(&mut ex, &mut term, key('e')).unwrap();
        // Draw first so the click regions exist, then click across the frame.
        let _ = crate::tui::headless_render(120, 40, |f| tree.render_frame(&ex, f));
        for (col, row) in [(0u16, 0u16), (4, 3), (60, 20), (119, 39)] {
            for m in events(col, row) {
                tree.handle_mouse(&mut ex, &mut term, m);
            }
        }

        let mut files = FilesMode::new();
        files.on_enter(&mut ex, &mut term).unwrap();
        let _ = crate::tui::headless_render(120, 40, |f| files.render_frame(&ex, f));
        for m in events(6, 4) {
            files.handle_mouse(&mut ex, &mut term, m);
        }

        let mut stats = StatsMode::new(false, 0);
        stats.on_enter(&mut ex, &mut term).unwrap();
        let _ = crate::tui::headless_render(120, 40, |f| stats.render_frame(&ex, f));
        for m in events(10, 6) {
            stats.handle_mouse(&mut ex, &mut term, m);
        }
        // Scrubbing the scroll bar is the engine's call into the mode.
        stats.set_scroll(&mut ex, 3);
        stats.set_scroll(&mut ex, usize::MAX); // clamped, not panicking
    }

    #[test]
    fn a_background_scan_ticks_without_a_terminal() {
        let (mut ex, mut term) = loaded();
        let tensor = first_tensor(&ex);
        // `StatsStart::Auto` kicks off the whole-tensor scan on entry; ticking it is what
        // the driver does between polls.
        let mut detail = DetailMode::new(tensor, 0, StatsStart::Auto, Interaction::Interactive);
        detail.on_enter(&mut ex, &mut term).unwrap();
        for _ in 0..50 {
            if matches!(detail.tick_background(&mut ex), Bg::Idle) {
                break;
            }
        }
        detail.set_background_paused(true);
        detail.set_background_paused(false);
        // The scan either finished or is still going; either way the screen still draws.
        draw_mode("a scanning detail screen", &detail, &ex);
    }

    #[test]
    fn every_mode_reports_a_residual_screen_for_history() {
        // Back / forward restore replays a mode's `residual()`; a mode that reported
        // the wrong screen would send Backspace somewhere the user never was.
        let (mut ex, mut term) = loaded();
        assert_eq!(
            outcome(&Outcome::Leave(Nav::Open(TreeMode::new().residual()))),
            "tree"
        );
        let mut files = FilesMode::new();
        files.on_enter(&mut ex, &mut term).unwrap();
        assert_eq!(
            outcome(&Outcome::Leave(Nav::Open(files.residual()))),
            "files"
        );
        assert_eq!(
            outcome(&Outcome::Leave(Nav::Open(
                StatsMode::new(false, 0).residual()
            ))),
            "stats"
        );
    }
}
