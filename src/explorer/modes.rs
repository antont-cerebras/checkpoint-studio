//! The six interactive screens as [`Mode`] implementations: file browser, byte-layout
//! map, tensor tree, rename editor, tensor detail, statistics and the data views.
//!
//! Split out of `explorer/mod.rs` because each one is self-contained — a small struct of
//! transient per-screen bookkeeping plus its `Mode` impl (key/mouse handling, drawing,
//! and the `residual` screen it restores to). The persistent selection/scroll state
//! stays on [`Explorer`]; these hold only what the old per-screen loops kept as locals.

// A child module of the one it was split out of, so it needs its parent's private
// vocabulary: the `Explorer` internals, the `Mode`/`Outcome`/`Screen` machinery, the
// per-screen command enums and the shared scroll constants. Enumerating them is ~80
// names that would churn on every edit, and the lint is aimed at wildcards across
// module boundaries you don't own — not at a submodule importing its own parent.
//
// The size of that list is itself the finding: these modes reach deep into `Explorer`.
// Narrowing that surface (so each mode takes only what it needs) is the next step, and
// this import is where the coupling will show up as it shrinks.
#[allow(clippy::wildcard_imports)]
use super::*;

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
        match key.code {
            // Every lettered command dispatches through the registry (like the tree),
            // so key and palette entry can't drift.
            KeyCode::Char(c)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                    && file_command_for_key(c).is_some() =>
            {
                let cmd = file_command_for_key(c).expect("guarded by is_some");
                if let Some(nav) = ex.run_file_command(cmd, term) {
                    return Ok(Outcome::Leave(nav));
                }
            }
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

    /// The parsed map — only reached after `on_enter` has bailed on a parse error.
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
        let map = match &self.map {
            Ok(m) => m,
            Err(_) => return PaletteResult::Handled,
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
        match key.code {
            // Every lettered command dispatches through the registry (`q`/`l`/`c`/`y`)
            // so key and palette entry can't drift.
            KeyCode::Char(ch)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                    && layout_command_for_key(ch).is_some() =>
            {
                let cmd = layout_command_for_key(ch).expect("guarded by is_some");
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
            }
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
        match key {
            // Every tree command dispatches through the registry (the same path the
            // palette uses). In search mode the letters fall through to the query.
            KeyEvent {
                code: KeyCode::Char(c),
                modifiers,
                ..
            } if !ex.tree_state.search_mode()
                && !modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT)
                && tree_command_for_key(c).is_some() =>
            {
                let cmd = tree_command_for_key(c).expect("guarded by is_some");
                if let Some(nav) = ex.run_command(cmd, term) {
                    return Ok(Outcome::Leave(nav));
                }
            }
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
    pub(super) scroll_max: std::cell::Cell<usize>,
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
            scroll_max: std::cell::Cell::new(0),
            applied: false,
        }
    }

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
                self.do_copy_screen()
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
    pub(super) tensor_name: String,
    pub(super) slice: usize,
    pub(super) stats_start: StatsStart,
    pub(super) interaction: Interaction,
    pub(super) tensor: Option<TensorInfo>,
    pub(super) overridable: bool,
    pub(super) unindexed: bool,
    pub(super) remote: bool,
    pub(super) warm: bool,
    pub(super) scan: Option<ScanJob>,
    pub(super) spin: std::cell::Cell<usize>,
    pub(super) overlay: Option<Overlay>,
}

impl DetailMode {
    pub(super) fn new(
        tensor_name: String,
        slice: usize,
        stats_start: StatsStart,
        interaction: Interaction,
    ) -> Self {
        Self {
            tensor_name,
            slice,
            stats_start,
            interaction,
            tensor: None,
            overridable: false,
            unindexed: false,
            remote: false,
            warm: false,
            scan: None,
            spin: std::cell::Cell::new(0),
            overlay: None,
        }
    }

    pub(super) fn tensor(&self) -> &TensorInfo {
        self.tensor.as_ref().expect("on_enter resolves or leaves")
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
    pub(super) fn stats_view<'a>(&self, stats: &'a Option<Stats>) -> StatsView<'a> {
        match stats {
            Some(s) => StatsView::Ready(s),
            None if self.warm && self.scan.is_some() => {
                let job = self.scan.as_ref().unwrap();
                if job.started.elapsed() >= std::time::Duration::from_millis(120) {
                    self.spin.set(self.spin.get().wrapping_add(1));
                    StatsView::Computing {
                        spinner: STATS_SPINNER[self.spin.get() % STATS_SPINNER.len()],
                        elapsed: job.started.elapsed(),
                        progress: job.progress(),
                    }
                } else {
                    StatsView::Pending
                }
            }
            None => StatsView::Pending,
        }
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
        let Some(tensor) = ex
            .tensors()
            .iter()
            .find(|t| t.name == self.tensor_name)
            .cloned()
        else {
            return Ok(Outcome::Leave(Nav::Open(Screen::Tree)));
        };
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
        self.tensor = Some(tensor);
        Ok(Outcome::Stay)
    }

    fn tick_background(&mut self, ex: &mut Explorer) -> Bg {
        if !self.warm {
            return Bg::Idle;
        }
        let tensor = self.tensor().clone();
        let view = ex.active_view(&tensor.name);
        if ex.cached_stats(&tensor, view).is_some() {
            self.scan = None;
            return Bg::Idle;
        }
        // (Re)start the scan for the current view; harvest it when finished.
        if self.scan.as_ref().is_none_or(|j| j.view != view) {
            self.scan = Some(ex.spawn_stats_scan(&tensor, view));
        }
        let finished = self
            .scan
            .as_ref()
            .and_then(|j| j.handle.as_ref())
            .is_some_and(|h| h.is_finished());
        if finished {
            let mut job = self.scan.take().unwrap();
            if let Some(h) = job.handle.take()
                && let Ok(Ok(s)) = h.join()
            {
                ex.stats_cache
                    .borrow_mut()
                    .insert((tensor.name.clone(), view), s);
            }
        }
        if self.scan.is_some() {
            Bg::Poll
        } else {
            Bg::Idle
        }
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
            KeyCode::Char('b') | KeyCode::Char('B') => {
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
            KeyCode::Char('s') | KeyCode::Char('S') => {
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
            KeyCode::Char('d') | KeyCode::Char('D') if self.overridable => {
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
            KeyCode::Char('r') | KeyCode::Char('R') if self.overridable => {
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
            tensor: self.tensor_name.clone(),
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
    pub(super) scroll_max: std::cell::Cell<usize>,
    pub(super) overlay: Option<Overlay>,
}

impl StatsMode {
    pub(super) fn new(shards_expanded: bool, scroll: usize) -> Self {
        Self {
            shards_expanded,
            scroll,
            scroll_max: std::cell::Cell::new(0),
            overlay: None,
        }
    }

    /// The cached stats (computed in `on_enter`).
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
                let (w, h) = term
                    .size()
                    .map(|s| (s.width, s.height))
                    .unwrap_or((120, 40));
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
                self.scroll = (self.scroll + SCROLL_PAGE).min(self.scroll_max.get())
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
    pub(super) tensor_name: String,
    pub(super) repr: Representation,
    pub(super) slice: std::cell::Cell<usize>,
    pub(super) interaction: Interaction,
    pub(super) tensor: Option<TensorInfo>,
    pub(super) scan: Option<ScanJob>,
    pub(super) spin: std::cell::Cell<usize>,
    pub(super) overlay: Option<Overlay>,
    pub(super) slices: std::cell::Cell<usize>,
    pub(super) overridable: std::cell::Cell<bool>,
}

impl DataMode {
    pub(super) fn new(
        tensor_name: String,
        repr: Representation,
        slice: usize,
        interaction: Interaction,
    ) -> Self {
        Self {
            tensor_name,
            repr,
            slice: std::cell::Cell::new(slice),
            interaction,
            tensor: None,
            scan: None,
            spin: std::cell::Cell::new(0),
            overlay: None,
            slices: std::cell::Cell::new(1),
            overridable: std::cell::Cell::new(false),
        }
    }

    pub(super) fn tensor(&self) -> &TensorInfo {
        self.tensor.as_ref().expect("on_enter resolves or leaves")
    }

    /// The current statistics view — cached, a live scan spinner (data always
    /// scans when uncached), or pending. `stats` is the caller's local.
    pub(super) fn stats_view<'a>(&self, stats: &'a Option<Stats>) -> StatsView<'a> {
        match stats {
            Some(s) => StatsView::Ready(s),
            None if self.scan.is_some() => {
                let job = self.scan.as_ref().unwrap();
                if job.started.elapsed() >= std::time::Duration::from_millis(120) {
                    self.spin.set(self.spin.get().wrapping_add(1));
                    StatsView::Computing {
                        spinner: STATS_SPINNER[self.spin.get() % STATS_SPINNER.len()],
                        elapsed: job.started.elapsed(),
                        progress: job.progress(),
                    }
                } else {
                    StatsView::Pending
                }
            }
            None => StatsView::Pending,
        }
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
        let Some(tensor) = ex
            .tensors()
            .iter()
            .find(|t| t.name == self.tensor_name)
            .cloned()
        else {
            return Ok(Outcome::Leave(Nav::Back));
        };
        // One-shot (`--exit`): compute the stats synchronously so the single frame
        // shows them (interactively the scan runs in the background via tick).
        if self.interaction == Interaction::OneShot {
            let view = ex.active_view(&tensor.name);
            ex.compute_stats_sync(&tensor, view);
        }
        self.tensor = Some(tensor);
        Ok(Outcome::Stay)
    }

    fn tick_background(&mut self, ex: &mut Explorer) -> Bg {
        let tensor = self.tensor().clone();
        let view = ex.active_view(&tensor.name);
        if ex.cached_stats(&tensor, view).is_some() {
            self.scan = None;
            return Bg::Idle;
        }
        if self.scan.as_ref().is_none_or(|j| j.view != view) {
            self.scan = Some(ex.spawn_stats_scan(&tensor, view));
        }
        let finished = self
            .scan
            .as_ref()
            .and_then(|j| j.handle.as_ref())
            .is_some_and(|h| h.is_finished());
        if finished {
            let mut job = self.scan.take().unwrap();
            if let Some(h) = job.handle.take()
                && let Ok(Ok(s)) = h.join()
            {
                ex.stats_cache
                    .borrow_mut()
                    .insert((tensor.name.clone(), view), s);
            }
        }
        if self.scan.is_some() {
            Bg::Poll
        } else {
            Bg::Idle
        }
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
            KeyCode::Char('e') | KeyCode::Char('E') => ex
                .data_view
                .data_view_layout
                .set(ex.data_view.data_view_layout.get().next()),
            KeyCode::Char('z') | KeyCode::Char('Z') => ex
                .data_view
                .data_view_stripe
                .set(ex.data_view.data_view_stripe.get().next()),
            KeyCode::Char('b') | KeyCode::Char('B') => ex
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
            KeyCode::Char('d') | KeyCode::Char('D') if overridable => {
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
            KeyCode::Char('r') | KeyCode::Char('R') if overridable => {
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
                self.slice.set((self.slice.get() + 1) % slices)
            }
            KeyCode::Char('[') | KeyCode::Left if slices > 1 => {
                self.slice.set((self.slice.get() + slices - 1) % slices)
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
            tensor: self.tensor_name.clone(),
            repr: self.repr,
            slice: self.slice.get(),
        }
    }
}
