//! The rename screen: the rule fields, the live diff of the resulting names and
//! the completion dropdown.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph, Widget};

use super::hints::{ChipHit, chip_regions, rename_hint_lines};
use super::palette;
use super::scroll::VScrollbar;
use super::text::input_box_spans;
use super::theme::dim_span;
use super::{ChipRegions, Link, LinkRegions, UI};

/// One rule's line in the rename preview: the before→after *schema* plus how many
/// tensors it touches and how they break down by [`RenameStatus`]. Summarising per
/// rule keeps the preview a few lines even when a rule matches every layer.
pub(crate) struct RenameRuleView {
    pub from: String,
    pub to: String,
    /// Tensors whose name the rule *changes* (the rows the preview lists).
    pub total: usize,
    /// Tensors the pattern *matches*, changed or not — so a no-op rule (a
    /// just-autocompleted source whose new name is still identical) reads as
    /// "matches N · unchanged", not the misleading "matches no tensors".
    pub matched: usize,
    pub ok: usize,
    pub collide: usize,
    pub wont_fit: usize,
    pub invalid: usize,
    /// Per changed shard: how the rewritten header sizes up (the detail behind a
    /// `won't fit` verdict — which file, and by how many bytes).
    pub shards: Vec<crate::rename::ShardFit>,
}

/// One entry in the rename editor's autocomplete dropdown: a tensor-family schema,
/// how many tensors it covers, and (optionally) the char range of the typed query
/// within it to embolden.
pub(crate) struct RenameCompletion {
    pub text: String,
    /// Tensors this family schema covers — shown as a dim `×N` metadata column.
    pub count: usize,
    /// `(start, end)` char range of the literal query match, to embolden; `None`
    /// for a number-agnostic match (where the query has no literal counterpart).
    pub hl: Option<(usize, usize)>,
}

/// Everything [`UI::render_rename`] draws: the dynamic list of source→new-name rule
/// pairs (with the focused field + its autocomplete) and a compact, per-rule
/// before→after preview marking each rule's in-place feasibility.
pub(crate) struct RenameView<'a> {
    pub root: &'a str,
    /// `(source, new-name)` for each rule pair, in order.
    pub pairs: &'a [(String, String)],
    pub focus_pair: usize,
    /// Which field of `focus_pair` has focus: `false` = source, `true` = new-name.
    pub on_target: bool,
    /// Caret position (char index) within the focused field.
    pub cursor: usize,
    /// Whether the autocomplete dropdown is open at the focused field.
    pub menu_open: bool,
    /// The highlighted candidate in the dropdown (an index into `completions`).
    pub menu_sel: usize,
    /// Autocomplete candidates for the focused field; empty ⇒ no dropdown drawn.
    pub completions: &'a [RenameCompletion],
    /// One summary per complete rule (the before→after preview).
    pub rules: &'a [RenameRuleView],
    /// Total tensors renamed across all rules (for the header).
    pub total: usize,
    pub warnings: &'a [String],
    /// Whether `model.safetensors.index.json` will be updated too.
    pub has_index: bool,
    /// Whether the rename can be applied (every affected tensor clean).
    pub applicable: bool,
    /// Preview-pane scroll offset.
    pub scroll: usize,
    pub error: Option<&'a str>,
    /// The `convert --map …` CLI command equivalent to the entered renames (shown
    /// above the footer, copyable with `^Y`), or `None` until a rule is complete.
    pub cli: Option<&'a str>,
    /// What was just copied to the clipboard (e.g. `"the apply command"`), shown as
    /// a `✓ copied …` flash on the command row; `None` when nothing was just copied.
    pub copied: Option<&'a str>,
}

impl UI {
    /// Draw the in-place rename editor ([`Screen::Rename`](crate::explorer)): a
    /// title + rule header (the same borderless chrome as the tree / detail /
    /// layout views — it's a first-class view, not a pop-up dialog), the dynamic
    /// list of source→new-name rule pairs (with the focused field's autocomplete),
    /// a live before→after diff preview marking each tensor OK / collides /
    /// won't-fit, and the common footer / confirm bar. Returns the preview pane's
    /// max scroll offset, the footer chip regions (clickable, like the other
    /// views), and the preview's nav-link regions.
    pub(crate) fn render_rename(
        frame: &mut Frame,
        view: &RenameView,
    ) -> (
        usize,
        ChipRegions,
        LinkRegions,
        Vec<Rect>,
        Option<VScrollbar>,
    ) {
        let area = frame.area();
        let (width, height) = (area.width, area.height);
        if height < 7 || width < 12 {
            return (0, Vec::new(), Vec::new(), Vec::new(), None);
        }

        // Header: a title line then a full-width rule, matching the other views'
        // chrome (no surrounding border, no panel fill).
        let header = vec![
            Line::from(Span::styled(
                format!("Rename tensors in place — {}", view.root),
                Style::default()
                    .fg(palette::ACCENT)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                "─".repeat(width as usize),
                Style::default().fg(palette::DIM),
            )),
        ];
        let header_h = header.len() as u16;
        Paragraph::new(header).render(
            Rect {
                x: 0,
                y: 0,
                width,
                height: header_h,
            },
            frame.buffer_mut(),
        );

        // One field row: `label  value` — the focused field shows the caret via an
        // input box; others show their value plainly (not dimmed).
        let field_row = |label: &str, value: &str, focused: bool| -> Line<'static> {
            let mut spans = vec![Span::styled(
                format!("  {label:<4} "),
                Style::default().fg(palette::KEY),
            )];
            if focused {
                spans.extend(input_box_spans(value, view.cursor, 0));
            } else {
                spans.push(Span::raw(value.to_string()));
            }
            Line::from(spans)
        };

        // --- rule-pair editor lines ---
        // The focused field's row index within `editor` — the autocomplete dropdown
        // floats just beneath it (resolved to an absolute row once `editor` is laid
        // out). The dropdown itself is drawn last, over the content below.
        let mut editor: Vec<Line<'static>> = Vec::new();
        let mut focus_line = 0usize;
        for (i, (src, tgt)) in view.pairs.iter().enumerate() {
            if view.pairs.len() > 1 {
                editor.push(Line::from(Span::styled(
                    format!("rule {}", i + 1),
                    Style::default()
                        .fg(palette::DTYPE)
                        .add_modifier(Modifier::BOLD),
                )));
            }
            let focused_src = i == view.focus_pair && !view.on_target;
            let focused_tgt = i == view.focus_pair && view.on_target;
            if focused_src {
                focus_line = editor.len();
            }
            editor.push(field_row("from", src, focused_src));
            if focused_tgt {
                focus_line = editor.len();
            }
            editor.push(field_row("to", tgt, focused_tgt));
            editor.push(Line::from(Span::raw("")));
        }

        // --- compact, per-rule before → after preview ---
        let mut preview: Vec<Line<'static>> = Vec::new();
        let mut summary = format!(
            "Preview — {} tensor(s) across {} rule(s)",
            view.total,
            view.rules.len()
        );
        // Only claim the index will change when a rule actually matches something.
        if view.has_index && view.total > 0 {
            summary.push_str(" · updates index.json");
        }
        preview.push(Line::from(Span::styled(
            summary,
            Style::default()
                .fg(palette::KEY)
                .add_modifier(Modifier::BOLD),
        )));
        if view.rules.is_empty() {
            preview.push(Line::from(dim_span(
                "  autocomplete a source and edit its new name to preview the changes",
            )));
        }
        // Clickable links in the preview: (preview line index, x within inner,
        // width, target) — resolved to screen Rects once scroll is known. Shard
        // names open the layout view; a *concrete* source tensor opens the tree.
        let mut hits: Vec<(usize, u16, u16, Link)> = Vec::new();
        for (i, rule) in view.rules.iter().enumerate() {
            // A one-line status per rule (coloured by the worst outcome), then the
            // before → after schema on their own lines so nothing is truncated. The
            // count reflects tensors *changed*, except a matched-but-unchanged rule
            // (a just-autocompleted source whose new name is still identical), which
            // reports how many it *matches* so it doesn't read as "matches nothing".
            let (count, label, color) = if rule.collide > 0 || rule.invalid > 0 {
                let mut parts = Vec::new();
                if rule.collide > 0 {
                    parts.push(format!("{} collide", rule.collide));
                }
                if rule.invalid > 0 {
                    parts.push(format!("{} invalid target", rule.invalid));
                }
                (
                    rule.total,
                    format!("⚠ {}", parts.join(", ")),
                    palette::ERROR,
                )
            } else if rule.wont_fit > 0 {
                (
                    rule.total,
                    format!("⚠ {} won't fit in place", rule.wont_fit),
                    palette::WARN,
                )
            } else if rule.total == 0 {
                if rule.matched > 0 {
                    (
                        rule.matched,
                        "new name unchanged — edit the “to” field".to_string(),
                        palette::WARN,
                    )
                } else {
                    (0, "matches no tensors".to_string(), palette::DIM)
                }
            } else {
                (
                    rule.total,
                    "✓ applies cleanly".to_string(),
                    palette::SUCCESS,
                )
            };
            preview.push(Line::default());
            preview.push(Line::from(vec![
                Span::styled(
                    format!("rule {} · {} tensor(s) · ", i + 1, count),
                    Style::default().fg(palette::DTYPE),
                ),
                Span::styled(label, Style::default().fg(color)),
            ]));
            // A concrete source (no `{…}` placeholder) is one real tensor, so it's a
            // link to the tree; a schema source matches many, so it stays plain.
            if rule.from.contains('{') {
                preview.push(Line::from(Span::raw(format!("    {}", rule.from))));
            } else {
                hits.push((
                    preview.len(),
                    4, // "    " indent
                    rule.from.chars().count() as u16,
                    Link::Tree(rule.from.clone()),
                ));
                preview.push(Line::from(vec![
                    Span::raw("    "),
                    Span::styled(
                        rule.from.clone(),
                        Style::default()
                            .fg(palette::ACCENT)
                            .add_modifier(Modifier::UNDERLINED),
                    ),
                ]));
            }
            preview.push(Line::from(vec![
                Span::styled("  → ", Style::default().fg(palette::DIM)),
                Span::styled(rule.to.clone(), Style::default().fg(color)),
            ]));
            // Per-shard header sizing — which file fits in place and by how much.
            for sf in &rule.shards {
                let (mark, note, c) = if sf.fits() {
                    ("✓", format!("{} B to spare", sf.spare()), palette::SUCCESS)
                } else {
                    ("✗", format!("{} B over", sf.over()), palette::WARN)
                };
                // The filename is a link to that shard's layout view; record its
                // region. `"      {mark} "` = 6 spaces + mark + space = 8 columns.
                hits.push((
                    preview.len(),
                    8,
                    sf.file.chars().count() as u16,
                    Link::Layout(sf.path.clone()),
                ));
                preview.push(Line::from(vec![
                    Span::styled(format!("      {mark} "), Style::default().fg(palette::DIM)),
                    Span::styled(
                        sf.file.clone(),
                        Style::default()
                            .fg(palette::ACCENT)
                            .add_modifier(Modifier::UNDERLINED),
                    ),
                    Span::styled(
                        format!(
                            "  header {} B → {} B  ({note}, {} tensor(s))",
                            sf.current, sf.needed, sf.tensors
                        ),
                        Style::default().fg(c),
                    ),
                ]));
            }
        }
        if !view.warnings.is_empty() {
            preview.push(Line::default());
        }
        for w in view.warnings {
            preview.push(Line::from(Span::styled(
                format!("note: {w}"),
                Style::default().fg(palette::WARN),
            )));
        }

        // --- footer: the common clickable chip hint, or the error bar ---
        // Build it first so its (possibly wrapped) height reserves the bottom rows,
        // like the layout/file views size their footers. (Apply confirmation is a
        // floating modal now, not an inline footer bar.)
        let (footer_lines, chip_hits): (Vec<Line<'static>>, Vec<ChipHit>) =
            if let Some(err) = view.error {
                (
                    vec![Line::from(Span::styled(
                        format!("⚠ {err}"),
                        Style::default().fg(palette::ERROR),
                    ))],
                    Vec::new(),
                )
            } else {
                rename_hint_lines(width, view.applicable)
            };
        let footer_h = (footer_lines.len() as u16).max(1);
        let footer_top = height.saturating_sub(footer_h);

        // --- lay out: header, editor, preview (scroll), command row, footer ---
        // The apply-command row sits just above the footer when there's room.
        let cmd_y = footer_top.checked_sub(1).filter(|y| *y > header_h + 1);
        let editor_h = (editor.len() as u16).min(height.saturating_sub(header_h + footer_h + 2));
        Paragraph::new(editor).render(
            Rect {
                x: 0,
                y: header_h,
                width,
                height: editor_h,
            },
            frame.buffer_mut(),
        );

        let sep_y = header_h + editor_h;
        let preview_bottom = cmd_y.unwrap_or(footer_top);
        if sep_y < preview_bottom {
            Paragraph::new(Line::from(dim_span("─".repeat(width as usize)))).render(
                Rect {
                    x: 0,
                    y: sep_y,
                    width,
                    height: 1,
                },
                frame.buffer_mut(),
            );
        }
        let preview_y = sep_y + 1;
        let preview_h = preview_bottom.saturating_sub(preview_y);
        let visible = preview_h as usize;
        let max_scroll = preview.len().saturating_sub(visible);
        let scroll = view.scroll.min(max_scroll);
        // The preview's scroll bar (drawn by the engine), over its last column.
        let vscroll = VScrollbar::for_body(
            Rect {
                x: 0,
                y: preview_y,
                width,
                height: preview_h,
            },
            preview.len(),
            scroll,
        );
        let window: Vec<Line> = preview.iter().skip(scroll).take(visible).cloned().collect();
        Paragraph::new(window).render(
            Rect {
                x: 0,
                y: preview_y,
                width,
                height: preview_h,
            },
            frame.buffer_mut(),
        );
        // Map the visible link hits to on-screen Rects (target per region).
        let clicks: Vec<(Rect, Link)> = hits
            .into_iter()
            .filter(|(idx, ..)| *idx >= scroll && *idx < scroll + visible)
            .map(|(idx, x, w, target)| {
                (
                    Rect {
                        x,
                        y: preview_y + (idx - scroll) as u16,
                        width: w,
                        height: 1,
                    },
                    target,
                )
            })
            .collect();

        // The equivalent apply command (copyable with ^Y), just above the footer.
        if let Some(y) = cmd_y {
            let cmd_line = if let Some(what) = view.copied {
                Line::from(Span::styled(
                    format!("✓ copied {what} to the clipboard"),
                    Style::default()
                        .fg(palette::SUCCESS)
                        .add_modifier(Modifier::BOLD),
                ))
            } else if let Some(cmd) = view.cli {
                Line::from(vec![
                    Span::styled("apply: ", Style::default().fg(palette::DIM)),
                    Span::styled(cmd.to_string(), Style::default().fg(palette::META)),
                    Span::styled("   (^A copy)", Style::default().fg(palette::DIM)),
                ])
            } else {
                Line::from(dim_span(
                    "enter a rename above to get the `convert --map` command that applies it",
                ))
            };
            Paragraph::new(cmd_line).render(
                Rect {
                    x: 0,
                    y,
                    width,
                    height: 1,
                },
                frame.buffer_mut(),
            );
        }

        Paragraph::new(footer_lines).render(
            Rect {
                x: 0,
                y: footer_top,
                width,
                height: footer_h,
            },
            frame.buffer_mut(),
        );
        // Footer chips → absolute clickable regions (each replays its key).
        let chips = chip_regions(&chip_hits, footer_top);

        // The autocomplete dropdown floats over everything, anchored just below the
        // focused field (when it's on-screen) — drawn last so nothing overpaints it.
        let menu_rects =
            if view.menu_open && !view.completions.is_empty() && focus_line < editor_h as usize {
                render_completion_menu(
                    frame,
                    RENAME_MENU_X,
                    header_h + focus_line as u16,
                    view.completions,
                    view.menu_sel,
                )
            } else {
                Vec::new()
            };

        (max_scroll, chips, clicks, menu_rects, vscroll)
    }
}

/// A centred, content-sized pop-up over the frame: a rounded [`Block`] (accent
/// border, `title` on the top edge, panel background) wrapping `content`. With
/// [`Backdrop::Float`] the surrounding frame is left untouched (only the box rect
/// is cleared) so the screen behind stays visible — a real pop-up; with
/// [`Backdrop::Fill`] the whole frame is wiped to the scrim first, for standalone
/// message screens. Shared by the legend pop-up and message screens.
/// Draw a centered popup box and return its inner (content) rect, so callers
/// that need to hit-test the content — e.g. a clickable menu — can map screen
/// coordinates to rows. `fixed_inner_w`, when set, pins the content width (lines
/// wider than it are clipped) so the box is a constant size regardless of its
/// content — otherwise the box sizes to the widest line.
/// The left column the rename editor's autocomplete dropdown anchors at — under
/// the field's value box (2-space indent + 4-wide label + space).
const RENAME_MENU_X: u16 = 7;

/// Draw the rename editor's autocomplete dropdown: a background-filled block (no
/// box-drawing border — it's outlined by its fill colour), floating just below the
/// focused field, one candidate per row with the highlighted one inverted, the
/// matched substring emboldened, and a dim right-aligned `×N` tensor count. A final
/// dim caption row spells out the keys. `field_row` is the focused field's absolute
/// row; the box drops beneath it, or flips above when it would overflow the frame.
/// Returns each candidate row's on-screen rect (the caption excluded) so a click
/// can accept it.
fn render_completion_menu(
    frame: &mut Frame,
    anchor_x: u16,
    field_row: u16,
    cands: &[RenameCompletion],
    selected: usize,
) -> Vec<Rect> {
    let area = frame.area();
    if cands.is_empty() || area.width <= anchor_x {
        return Vec::new();
    }
    const CAPTION: &str = "↑/↓ pick · ↵ accept · Tab complete · Esc close";
    let count_label = |n: usize| format!("×{n}");
    let name_w = cands
        .iter()
        .map(|c| c.text.chars().count())
        .max()
        .unwrap_or(0);
    let count_w = cands
        .iter()
        .map(|c| count_label(c.count).chars().count())
        .max()
        .unwrap_or(0);
    // 1 lead + name + 2 gap + count + 1 trail, but at least wide enough for the
    // caption, and never past the frame's right edge.
    let inner_w = (1 + name_w + 2 + count_w + 1).max(CAPTION.chars().count() + 2);
    let box_w = (inner_w as u16).min(area.width - anchor_x).max(1);
    let box_h = (cands.len() as u16 + 1).min(area.height); // +1 caption row
    // Prefer dropping below the field; flip above when there's no room beneath.
    let below = field_row + 1;
    let box_y = if below + box_h <= area.height {
        below
    } else {
        field_row.saturating_sub(box_h)
    };
    let box_x = anchor_x.min(area.width.saturating_sub(box_w));
    let rect = Rect {
        x: box_x,
        y: box_y,
        width: box_w,
        height: box_h,
    };
    let base = Style::default().fg(palette::INPUT_FG).bg(palette::PANEL_BG);
    let sel_style = Style::default()
        .fg(palette::SELECT_FG)
        .bg(palette::SELECT_BG);
    Clear.render(rect, frame.buffer_mut());
    Block::default()
        .style(base)
        .render(rect, frame.buffer_mut());

    let mut rects = Vec::new();
    for (i, c) in cands.iter().enumerate() {
        let row_y = box_y + i as u16;
        let picked = i == selected;
        let row = if picked { sel_style } else { base };
        let count_style = if picked {
            sel_style
        } else {
            Style::default().fg(palette::META).bg(palette::PANEL_BG)
        };
        let mut spans = vec![Span::styled(" ", row)];
        for (ci, ch) in c.text.chars().enumerate() {
            let mut st = row;
            if let Some((s, e)) = c.hl
                && ci >= s
                && ci < e
            {
                st = st.add_modifier(Modifier::BOLD);
                if !picked {
                    st = st.fg(palette::ACCENT);
                }
            }
            spans.push(Span::styled(ch.to_string(), st));
        }
        let name_len = c.text.chars().count();
        if name_len < name_w {
            spans.push(Span::styled(" ".repeat(name_w - name_len), row));
        }
        spans.push(Span::styled("  ", row));
        let cl = count_label(c.count);
        let cl_w = cl.chars().count();
        if cl_w < count_w {
            spans.push(Span::styled(" ".repeat(count_w - cl_w), count_style));
        }
        spans.push(Span::styled(cl, count_style));
        Paragraph::new(Line::from(spans)).render(
            Rect {
                x: box_x,
                y: row_y,
                width: box_w,
                height: 1,
            },
            frame.buffer_mut(),
        );
        rects.push(Rect {
            x: box_x,
            y: row_y,
            width: box_w,
            height: 1,
        });
    }
    // The key caption, dimmed, on the last row.
    Paragraph::new(Line::from(vec![
        Span::styled(" ", base),
        Span::styled(
            CAPTION,
            Style::default().fg(palette::DIM).bg(palette::PANEL_BG),
        ),
    ]))
    .render(
        Rect {
            x: box_x,
            y: box_y + cands.len() as u16,
            width: box_w,
            height: 1,
        },
        frame.buffer_mut(),
    );
    rects
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tests_support::strip_ansi_codes;

    #[test]
    fn render_rename_shows_fields_and_marks_the_diff() {
        let pairs = vec![(
            "model.layers.{layer}.self_attn.q_proj.weight".to_string(),
            "model.layers.{layer}.attn.q_proj.weight".to_string(),
        )];
        let rules = vec![RenameRuleView {
            from: "model.layers.{layer}.self_attn.q_proj.weight".to_string(),
            to: "model.layers.{layer}.attn.q_proj.weight".to_string(),
            total: 48,
            matched: 48,
            ok: 0,
            collide: 0,
            wont_fit: 48,
            invalid: 0,
            shards: vec![crate::rename::ShardFit {
                file: "model.safetensors".to_string(),
                path: "/ckpt/model.safetensors".to_string(),
                current: 512,
                needed: 560,
                tensors: 48,
            }],
        }];
        let view = RenameView {
            root: "/ckpt",
            pairs: &pairs,
            focus_pair: 0,
            on_target: true,
            cursor: 0,
            menu_open: false,
            menu_sel: 0,
            completions: &[],
            rules: &rules,
            total: 48,
            warnings: &[],
            has_index: true,
            applicable: false,
            scroll: 0,
            error: None,
            cli: Some("checkpoint-studio convert /ckpt --map 'a=>b'"),
            copied: None,
        };
        let mut clicks = Vec::new();
        let out = crate::tui::headless_render(120, 30, |f| {
            let (_, _chips, c, _menu, _) = UI::render_rename(f, &view);
            clicks = c;
        })
        .unwrap();
        let plain = strip_ansi_codes(&out);
        assert!(plain.contains("Rename tensors in place"), "{plain}");
        assert!(plain.contains("from") && plain.contains("to"), "{plain}");
        // The per-rule schema before→after and its "won't fit" marker.
        assert!(
            plain.contains("model.layers.{layer}.attn.q_proj.weight"),
            "{plain}"
        );
        assert!(plain.contains("won't fit in place"), "{plain}");
        assert!(plain.contains("updates index.json"), "{plain}");
        // The shard name is a clickable region linking to that file's layout.
        assert!(
            clicks.iter().any(|(_, t)| matches!(
                t,
                Link::Layout(p) if p == "/ckpt/model.safetensors"
            )),
            "expected a clickable shard region"
        );
    }

    #[test]
    fn render_rename_dropdown_lists_candidates_with_counts_and_click_targets() {
        let pairs = vec![("q_proj".to_string(), String::new())];
        let cands = vec![
            RenameCompletion {
                text: "model.layers.{layer}.self_attn.q_proj.weight".into(),
                count: 32,
                hl: Some((0, 5)),
            },
            RenameCompletion {
                text: "model.layers.{layer}.self_attn.k_proj.weight".into(),
                count: 32,
                hl: None,
            },
        ];
        let view = RenameView {
            root: "/ckpt",
            pairs: &pairs,
            focus_pair: 0,
            on_target: false,
            cursor: 6,
            menu_open: true,
            menu_sel: 0,
            completions: &cands,
            rules: &[],
            total: 0,
            warnings: &[],
            has_index: false,
            applicable: false,
            scroll: 0,
            error: None,
            cli: None,
            copied: None,
        };
        let mut menu = Vec::new();
        let out = crate::tui::headless_render(120, 30, |f| {
            let (_, _chips, _links, m, _) = UI::render_rename(f, &view);
            menu = m;
        })
        .unwrap();
        let plain = strip_ansi_codes(&out);
        // Both candidates, the `×N` metadata column, and the key caption are drawn.
        assert!(plain.contains("self_attn.q_proj.weight"), "{plain}");
        assert!(plain.contains("self_attn.k_proj.weight"), "{plain}");
        assert!(plain.contains("×32"), "count column: {plain}");
        assert!(plain.contains("Tab complete"), "key caption: {plain}");
        // One click target per candidate row (the caption row is not clickable).
        assert_eq!(menu.len(), 2, "a click rect per candidate");
    }
}
