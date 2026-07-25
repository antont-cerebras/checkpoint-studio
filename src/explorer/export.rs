//! Turning the tree into text: the `--print-tree` / `--print-tensors` renderings (plain
//! and JSON), the interactive copy/export menu, and the CLI command the `y` key emits.
//!
//! Split out of `explorer/mod.rs` as a third `impl Explorer` block. This is the surface
//! scripts and agents consume, so it is worth having in one place: the same text and
//! JSON shapes back the one-shot CLI exports, the clipboard copy, and the export menu's
//! preview pane.

#[allow(clippy::wildcard_imports)] // a submodule of the module it was split from
use super::*;

impl Explorer {
    /// The whole tree as text — every group and tensor in the browser's row
    /// layout, fully expanded regardless of the live collapse state, with no
    /// viewport limit or header/footer chrome. Backs the `t` copy and
    /// `--print-tree`. `Full` appends each tensor's source file.
    pub(super) fn tree_text(&self, detail: TreeDetail) -> String {
        fn walk(
            node: &TreeNode,
            depth: usize,
            detail: TreeDetail,
            unindexed: &HashSet<String>,
            schemas: &HashMap<String, PackingSchema>,
            out: &mut Vec<String>,
        ) {
            let mut line = crate::ui::tree_row_text(node, depth, unindexed, schemas);
            if detail == TreeDetail::Full
                && let TreeNode::Tensor { info, .. } = node
            {
                line.push_str(&format!("  ← {}", file_basename(&info.source_path)));
            }
            out.push(line);
            if let TreeNode::Group { children, .. } = node {
                for child in children {
                    walk(child, depth + 1, detail, unindexed, schemas, out);
                }
            }
        }
        // Render a fully-expanded copy so every group's arrow (▾) matches the
        // listing below it, regardless of the live collapse state.
        let mut tree = self.tree_state.tree.clone();
        TreeBuilder::set_all_expanded(&mut tree, true);
        let mut out = Vec::new();
        for node in &tree {
            walk(
                node,
                0,
                detail,
                &self.unindexed,
                &self.packing_schemas,
                &mut out,
            );
        }
        out.join("\n")
    }

    /// A flat, one-line-per-tensor listing (in natural-sorted order), reusing the
    /// browser's tensor-row field layout but without its leading `·` bullet — a
    /// flat list needs no tree marker. `Full` appends each tensor's source file.
    pub(super) fn tensors_text(&self, detail: TreeDetail) -> String {
        self.tensors()
            .iter()
            .map(|t| {
                let line = crate::ui::tensor_list_line(t, &self.unindexed, &self.packing_schemas);
                let mut text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
                if detail == TreeDetail::Full {
                    text.push_str(&format!("  ← {}", file_basename(&t.source_path)));
                }
                text
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The tree as `model.safetensors.index.json`-style JSON: `metadata.total_size`
    /// (summed logical bytes) and a `weight_map` of tensor name → shard file.
    /// `Full` adds a `tensors` block with each tensor's dtype / shape / counts.
    pub(super) fn tree_json(&self, detail: TreeDetail) -> String {
        let total_size: usize = self.tensors().iter().map(|t| t.size_bytes).sum();
        let weight_map: serde_json::Map<String, serde_json::Value> = self
            .tensors()
            .iter()
            .map(|t| (t.name.clone(), file_basename(&t.source_path).into()))
            .collect();
        let mut root = serde_json::Map::new();
        root.insert(
            "metadata".into(),
            serde_json::json!({ "total_size": total_size }),
        );
        root.insert("weight_map".into(), serde_json::Value::Object(weight_map));
        if detail == TreeDetail::Full {
            let tensors: serde_json::Map<String, serde_json::Value> = self
                .tensors()
                .iter()
                .map(|t| (t.name.clone(), tensor_facts(t)))
                .collect();
            root.insert("tensors".into(), serde_json::Value::Object(tensors));
        }
        serde_json::to_string_pretty(&serde_json::Value::Object(root)).unwrap_or_default()
    }

    /// A JSON list of tensors: bare names (`Compact`) or objects with name,
    /// dtype, shape, element count and source file (`Full`). Natural-sorted.
    pub(super) fn tensors_json(&self, detail: TreeDetail) -> String {
        let items: Vec<serde_json::Value> = match detail {
            TreeDetail::Compact => self
                .tensors()
                .iter()
                .map(|t| t.name.clone().into())
                .collect(),
            TreeDetail::Full => self
                .tensors()
                .iter()
                .map(|t| {
                    let mut o = tensor_facts(t);
                    if let serde_json::Value::Object(m) = &mut o {
                        m.insert("name".into(), t.name.clone().into());
                        m.insert("file".into(), file_basename(&t.source_path).into());
                    }
                    o
                })
                .collect(),
        };
        serde_json::to_string_pretty(&serde_json::Value::Array(items)).unwrap_or_default()
    }

    /// The `t` shortcut: open a modal menu to pick which export variant to copy
    /// (tree / tensor list × text / JSON × plain / verbose — every CLI
    /// `--print-*` combination), then copy that. `↑`/`↓` (or `1`–`8`) move,
    /// Enter copies, Esc / click cancels. `term` is the borrowed live terminal.
    pub(super) fn copy_menu(&mut self, term: &mut crate::tui::LiveTerminal) {
        let labels: Vec<&str> = EXPORT_CHOICES.iter().map(|c| c.label).collect();
        let last = EXPORT_CHOICES.len() - 1;
        let mut sel = 0usize;
        // The preview is regenerated only when the highlight moves (it renders
        // the real export, which is cheap but not free on a huge checkpoint).
        let mut previewed = usize::MAX;
        let mut preview: Vec<Line<'static>> = Vec::new();
        let mut item_rects: Vec<ratatui::layout::Rect> = Vec::new();
        // A wrong-keyboard-layout key flashes the shared hint rather than being
        // silently ignored; cleared on the next input.
        let mut layout_hint: Option<char> = None;
        loop {
            if sel != previewed {
                preview = self.export_preview(EXPORT_CHOICES[sel]);
                previewed = sel;
            }
            let hint = layout_hint;
            if term
                .draw(|f| {
                    self.render_tree_frame(f, true);
                    item_rects = UI::render_menu_box(f, "Copy as…", &labels, sel, &preview);
                    if let Some(c) = hint {
                        UI::render_notice(f, &layout_hint_msg(c));
                    }
                })
                .is_err()
            {
                return;
            }
            // Which menu row (if any) is under a mouse position.
            let hit = |col: u16, row: u16| -> Option<usize> {
                item_rects.iter().position(|r| {
                    row >= r.y && row < r.y + r.height && col >= r.x && col < r.x + r.width
                })
            };
            match event::read() {
                Ok(Event::Key(key)) => {
                    if is_ctrl_c(&key) {
                        quit_immediately();
                    }
                    if let Some(c) = wrong_layout_char(&key) {
                        layout_hint = Some(c);
                        continue;
                    }
                    layout_hint = None;
                    match key.code {
                        KeyCode::Up => sel = if sel == 0 { last } else { sel - 1 },
                        KeyCode::Down => sel = if sel == last { 0 } else { sel + 1 },
                        KeyCode::Home => sel = 0,
                        KeyCode::End => sel = last,
                        // 1–8 pick a row directly.
                        KeyCode::Char(d @ '1'..='9') => {
                            let i = d as usize - '1' as usize;
                            if i <= last {
                                self.copy_export(term, EXPORT_CHOICES[i]);
                                return;
                            }
                        }
                        KeyCode::Enter => {
                            self.copy_export(term, EXPORT_CHOICES[sel]);
                            return;
                        }
                        KeyCode::Esc | KeyCode::Char('q') => return,
                        _ => {}
                    }
                }
                Ok(Event::Mouse(m)) => match m.kind {
                    MouseEventKind::ScrollUp => sel = if sel == 0 { last } else { sel - 1 },
                    MouseEventKind::ScrollDown => sel = if sel == last { 0 } else { sel + 1 },
                    // Hover highlights the row under the cursor.
                    MouseEventKind::Moved | MouseEventKind::Drag(_) => {
                        if let Some(i) = hit(m.column, m.row) {
                            sel = i;
                        }
                    }
                    // Click a row to copy it; a click off the list cancels.
                    MouseEventKind::Down(_) => match hit(m.column, m.row) {
                        Some(i) => {
                            self.copy_export(term, EXPORT_CHOICES[i]);
                            return;
                        }
                        None => return,
                    },
                    _ => {}
                },
                Ok(_) => {}       // resize etc.: redraw
                Err(_) => return, // input closed
            }
        }
    }

    /// The head of a menu `choice`'s export, styled like the tree, for the
    /// picker's live preview — real output from this checkpoint. Always returns a
    /// fixed number of rows (blank-padded, then a "+N more" / blank summary) so
    /// the menu box is the same size for every option.
    pub(super) fn export_preview(&self, choice: ExportChoice) -> Vec<Line<'static>> {
        let (mut lines, total) = match (choice.shape, choice.format) {
            (ExportShape::Tree, TreeFormat::Text) => self.tree_preview_lines(choice.detail),
            (ExportShape::Tensors, TreeFormat::Text) => self.tensors_preview_lines(choice.detail),
            // JSON: syntax-highlight it with the same palette as the metadata
            // view (falling back to plain lines if it somehow doesn't parse).
            (_, TreeFormat::Json) => {
                let full = self.export_text(choice);
                let styled = crate::ui::highlight_json_lines(&full).unwrap_or_else(|| {
                    full.lines()
                        .map(|l| Line::from(Span::raw(l.to_string())))
                        .collect()
                });
                let total = styled.len();
                (styled.into_iter().take(MENU_PREVIEW_LINES).collect(), total)
            }
        };
        lines.resize_with(MENU_PREVIEW_LINES, Line::default);
        lines.push(if total > MENU_PREVIEW_LINES {
            Line::from(crate::ui::dim_span(format!(
                "… (+{} more lines)",
                total - MENU_PREVIEW_LINES
            )))
        } else {
            Line::default()
        });
        lines
    }

    /// Styled preview rows for the tree export (first [`MENU_PREVIEW_LINES`]), plus
    /// the total row count. Walks fully expanded (forcing the open ▾ on collapsed
    /// groups) without cloning the tree.
    pub(super) fn tree_preview_lines(&self, detail: TreeDetail) -> (Vec<Line<'static>>, usize) {
        fn walk(
            node: &TreeNode,
            depth: usize,
            detail: TreeDetail,
            unindexed: &HashSet<String>,
            schemas: &HashMap<String, PackingSchema>,
            out: &mut Vec<Line<'static>>,
            total: &mut usize,
        ) {
            *total += 1;
            if out.len() < MENU_PREVIEW_LINES {
                let mut line = crate::ui::tree_row_line(node, depth, unindexed, schemas);
                if let TreeNode::Group {
                    expanded: false, ..
                } = node
                {
                    for span in &mut line.spans {
                        if span.content == "▸" {
                            span.content = "▾".into();
                            break;
                        }
                    }
                }
                if detail == TreeDetail::Full
                    && let TreeNode::Tensor { info, .. } = node
                {
                    line.spans.push(crate::ui::dim_span(format!(
                        "  ← {}",
                        file_basename(&info.source_path)
                    )));
                }
                out.push(line);
            }
            if let TreeNode::Group { children, .. } = node {
                for child in children {
                    walk(child, depth + 1, detail, unindexed, schemas, out, total);
                }
            }
        }
        let mut out = Vec::new();
        let mut total = 0;
        for node in &self.tree_state.tree {
            walk(
                node,
                0,
                detail,
                &self.unindexed,
                &self.packing_schemas,
                &mut out,
                &mut total,
            );
        }
        (out, total)
    }

    /// Styled preview rows for the flat tensor list (first [`MENU_PREVIEW_LINES`]),
    /// plus the total tensor count.
    pub(super) fn tensors_preview_lines(&self, detail: TreeDetail) -> (Vec<Line<'static>>, usize) {
        let lines = self
            .tensors()
            .iter()
            .take(MENU_PREVIEW_LINES)
            .map(|t| {
                let mut line =
                    crate::ui::tensor_list_line(t, &self.unindexed, &self.packing_schemas);
                if detail == TreeDetail::Full {
                    line.spans.push(crate::ui::dim_span(format!(
                        "  ← {}",
                        file_basename(&t.source_path)
                    )));
                }
                line
            })
            .collect();
        (lines, self.tensors().len())
    }

    /// Copy the export text for `choice`. If it fits the terminal clipboard, copy
    /// it directly (with a confirmation flash); otherwise copy the exact CLI
    /// command that reproduces it and show that in a dismissible band.
    pub(super) fn copy_export(
        &mut self,
        term: &mut crate::tui::LiveTerminal,
        choice: ExportChoice,
    ) {
        let text = self.export_text(choice);
        if copy_to_clipboard(&text) {
            self.flash_copied(choice.label);
        } else {
            let command = self.export_command(choice);
            copy_to_clipboard(&command); // the command itself is small — always fits
            self.float_until_dismissed(term, |f| {
                self.render_tree_frame(f, true);
                UI::render_export_band(f, &command);
            });
        }
    }

    /// The exported text for a menu `choice`.
    pub(super) fn export_text(&self, choice: ExportChoice) -> String {
        match (choice.shape, choice.format) {
            (ExportShape::Tree, TreeFormat::Text) => self.tree_text(choice.detail),
            (ExportShape::Tree, TreeFormat::Json) => self.tree_json(choice.detail),
            (ExportShape::Tensors, TreeFormat::Text) => self.tensors_text(choice.detail),
            (ExportShape::Tensors, TreeFormat::Json) => self.tensors_json(choice.detail),
        }
    }

    /// The concrete CLI command reproducing a menu `choice`, built the way `y`
    /// builds its reopen command (real paths, scp-style / `--ssh-proxy` for a
    /// remote source), so it runs as-is.
    pub(super) fn export_command(&self, choice: ExportChoice) -> String {
        let mut parts = self.command_prefix();
        parts.extend(self.checkpoint_path_parts());
        parts.push(
            match choice.shape {
                ExportShape::Tree => "--print-tree",
                ExportShape::Tensors => "--print-tensors",
            }
            .to_string(),
        );
        if choice.format == TreeFormat::Json {
            parts.push("--format".to_string());
            parts.push("json".to_string());
        }
        if choice.detail == TreeDetail::Full {
            parts.push("-v".to_string());
        }
        parts.join(" ")
    }
}
