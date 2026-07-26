//! The tensor tree: the main screen's renderer, its row builders (shared with
//! the plain-text exports) and its scroll geometry.

use std::collections::{HashMap, HashSet};

use crossterm::event::KeyEvent;
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use crate::sample::{PackingSchema, ViewDtype};
use crate::tree::{Storage, TensorInfo, TreeNode, metadata_short};
use crate::utils::{format_parameters, format_shape, format_size};

use super::hints::{chip_regions, close_button, hint_key, tree_hint_lines};
use super::palette;
use super::scroll::VScrollbar;
use super::text::{line_to_text, truncate_keep_end, without_bullet};
use super::theme::{COMPRESSED_MARK, SIZE_ARROW, UNCOMPRESSED_TAG, UNINDEXED_MARK, tree_span};
use super::{DrawConfig, UI};

/// Rows of chrome above the tree list: the title, the search/hint line, and the
/// separator rule.
const TREE_HEADER_HEIGHT: usize = 3;

/// Rows of chrome below the tree list: the two-line status bar.
/// Footer rows below the tree list: the two-line status bar. (The metadata-only
/// state is now a badge on that bar, not a separate banner row.)
const TREE_FOOTER_HEIGHT: usize = 2;

impl UI {
    /// How many tree rows are visible at once (one screenful), used to size a
    /// PageUp/PageDown jump. `terminal_height` is the full terminal height.
    pub(crate) fn visible_tree_rows(terminal_height: u16) -> usize {
        (terminal_height as usize)
            .saturating_sub(TREE_HEADER_HEIGHT + TREE_FOOTER_HEIGHT)
            .max(1)
    }

    /// Rows the tree's bottom-pinned key-hint footer occupies (0 while searching —
    /// the search bar rides the header instead). Kept in sync with
    /// [`Self::render_tree`] so scroll / hit-testing align.
    pub(crate) fn tree_hint_rows(
        width: u16,
        search_mode: bool,
        can_repack: bool,
        can_rename: bool,
    ) -> usize {
        if search_mode {
            0
        } else {
            tree_hint_lines(can_repack, can_rename, width).0.len()
        }
    }

    /// Body rows visible in the tree at the given size — used to compute the
    /// scroll offset so it stays consistent with [`Self::render_tree`]'s layout
    /// (header = title + optional search line + rule; a bottom-pinned hint footer;
    /// then the two status lines).
    pub(crate) fn tree_visible_rows(
        width: u16,
        height: u16,
        search_mode: bool,
        can_repack: bool,
        can_rename: bool,
    ) -> usize {
        let header = Self::tree_header_rows(search_mode);
        let hints = Self::tree_hint_rows(width, search_mode, can_repack, can_rename);
        (height as usize)
            .saturating_sub(header + hints + TREE_FOOTER_HEIGHT)
            .max(1)
    }

    /// The first terminal row of the tree body — the header height (title + the
    /// search line while searching + rule; the key hints are a bottom footer now).
    /// Used for mouse hit-testing: a click at row `r >= tree_header_rows()` and above
    /// the hint footer lands on tree row `scroll_offset + (r - tree_header_rows())`.
    pub(crate) fn tree_header_rows(search_mode: bool) -> usize {
        if search_mode { 3 } else { 2 } // title + [search] + rule
    }

    /// Geometry of the tree's vertical scroll bar for this terminal size and a
    /// tree of `total` rows, or `None` when the whole tree fits the viewport (so
    /// no bar is drawn and no column reserved). Shared by [`Self::render_tree`]
    /// and the mouse handler, so click / drag hit-testing lines up with what's
    /// drawn. The bar rides the rightmost column of the body region.
    pub(crate) fn tree_scrollbar(
        width: u16,
        height: u16,
        search_mode: bool,
        can_repack: bool,
        can_rename: bool,
        total: usize,
        offset: usize,
    ) -> Option<VScrollbar> {
        let rows = Self::tree_visible_rows(width, height, search_mode, can_repack, can_rename);
        VScrollbar::for_body(
            Rect {
                x: 0,
                y: Self::tree_header_rows(search_mode) as u16,
                width,
                height: rows as u16,
            },
            total,
            offset,
        )
    }

    /// Ratatui render of the tree browser: header (title, hint or search line,
    /// rule), the visible tree rows from `config.scroll_offset`, and the bottom
    /// two-line status bar, driven by the shared `DrawConfig`.
    pub(crate) fn render_tree(frame: &mut Frame, config: &DrawConfig) -> Vec<(Rect, KeyEvent)> {
        let area = frame.area();
        let (width, height) = (area.width, area.height);
        if height < (TREE_FOOTER_HEIGHT as u16 + 1) {
            return Vec::new();
        }

        // --- header + tree rows (the region above the 2-line status bar) ---
        let mut lines: Vec<Line> = Vec::new();

        // Title. (A health-check warning is surfaced on the status bar instead —
        // see the `⚠ health` alert beside the read-only badge below.)
        let mut title = vec![Span::raw(format!(
            "Checkpoint Studio - {} ({}/{})",
            config.current_file,
            config.file_idx + 1,
            config.total_files
        ))];
        if !config.filter_query.is_empty() {
            let n = config.tree.len();
            title.push(Span::styled(
                format!(
                    "  ·  filter: {} ({} match{})",
                    config.filter_query,
                    n,
                    if n == 1 { "" } else { "es" }
                ),
                Style::default().fg(palette::ACCENT),
            ));
        }
        lines.push(Line::from(title));

        // The search bar rides the header while searching; the key hints are a
        // bottom-pinned footer (built below), so they don't push the tree down.
        if config.search_mode {
            lines.push(tree_search_line(config));
        }

        // Separator rule.
        lines.push(Line::from(Span::styled(
            "─".repeat(width as usize),
            Style::default().fg(palette::DIM),
        )));

        // The bottom hint footer (absent while searching — the search bar is the
        // input, and the status bar spells out Esc/Enter).
        let (hint_lines, chips) = if config.search_mode {
            (Vec::new(), Vec::new())
        } else {
            tree_hint_lines(config.can_repack, config.can_rename, width)
        };
        let hint_rows = hint_lines.len();

        let header_rows = lines.len();
        let footer_rows = TREE_FOOTER_HEIGHT;
        let body_rows = (height as usize).saturating_sub(header_rows + hint_rows + footer_rows);

        // A vertical scroll bar rides the rightmost column when the tree
        // overflows the viewport — but only in the live TUI; a headless
        // `--plain` / screen-copy render is a static dump with no viewport.
        // The bar itself is drawn by the `run_mode` engine (via `TreeMode::vscrollbar`)
        // so every mode gets it uniformly; here we only reserve its column so long
        // tree rows don't underlap it (live TUI only — a headless dump has no bar).
        let scrollbar = config.interactive
            && Self::tree_scrollbar(
                width,
                height,
                config.search_mode,
                config.can_repack,
                config.can_rename,
                config.tree.len(),
                config.scroll_offset,
            )
            .is_some();
        let body_width = width.saturating_sub(if scrollbar { 1 } else { 0 });

        // Header (title, hint(s), rule) spans the full width.
        Paragraph::new(lines).render(
            crate::ui::fit_rows(area, 0, header_rows as u16),
            frame.buffer_mut(),
        );

        // Visible tree rows from the (pre-computed) scroll offset, clipped to
        // `body_width` so the reserved scroll-bar column stays clear.
        if !(config.search_mode && config.tree.is_empty()) {
            let mut body: Vec<Line> = Vec::with_capacity(body_rows);
            for (idx, (node, depth)) in config
                .tree
                .iter()
                .enumerate()
                .skip(config.scroll_offset)
                .take(body_rows)
            {
                let selected = idx == config.selected_idx;
                body.push(tree_node_line(
                    node,
                    *depth,
                    selected,
                    config.unindexed,
                    config.packing_schemas,
                    MetaDisplay::Capped, // live tree keeps rows short
                ));
            }
            Paragraph::new(body).render(
                Rect {
                    width: body_width,
                    ..crate::ui::fit_rows(area, header_rows as u16, body_rows as u16)
                },
                frame.buffer_mut(),
            );
        }

        // (The scroll bar itself is drawn by the engine — see `render_vscrollbar`.)

        // --- key-hint footer, pinned just above the two-line status bar ---
        let hint_y = height.saturating_sub(TREE_FOOTER_HEIGHT as u16 + hint_rows as u16);
        if hint_rows > 0 {
            Paragraph::new(hint_lines).render(
                crate::ui::fit_rows(area, hint_y, hint_rows as u16),
                frame.buffer_mut(),
            );
        }

        // --- bottom two-line status bar ---
        // Reserve room on the right of the bottom status line for the persistent
        // badges drawn there (access, and any health / metadata-only badge), so the
        // status text never runs under them.
        let reserve = Self::badge_bar_width(config.badges) as usize;
        let max_text = (width as usize).saturating_sub(6 + reserve);
        let row0 = if config.search_mode && config.tree.is_empty() {
            Line::from(vec![
                Span::raw(format!(
                    "No results found for \"{}\" | Press ",
                    config.search_query
                )),
                Span::styled(
                    "Esc",
                    Style::default()
                        .fg(palette::KEY)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" to exit search"),
            ])
        } else if !config.status_bar.is_empty() {
            let text = truncate_keep_end(config.status_bar, max_text);
            Line::from(Span::styled(
                format!(" {} {text} ", config.status_icon),
                Style::default()
                    .bg(palette::STATUS_BG)
                    .fg(palette::STATUS_FG),
            ))
        } else {
            Line::default()
        };
        Paragraph::new(row0).render(
            Rect {
                x: 0,
                y: height.saturating_sub(2),
                width,
                height: 1,
            },
            frame.buffer_mut(),
        );

        // Second line: a transient copy confirmation (green, shown verbatim)
        // overrides the dimmed source file.
        let row1 = if let Some(flash) = config.copied_flash {
            Line::from(Span::styled(
                flash.to_string(),
                Style::default()
                    .fg(palette::SUCCESS)
                    .add_modifier(Modifier::BOLD),
            ))
        } else if !config.status_secondary.is_empty() {
            let text = truncate_keep_end(config.status_secondary, max_text);
            Line::from(Span::styled(
                format!("   {text}"),
                Style::default().fg(palette::DIM),
            ))
        } else {
            Line::default()
        };
        Paragraph::new(row1).render(
            Rect {
                x: 0,
                y: height.saturating_sub(1),
                width,
                height: 1,
            },
            frame.buffer_mut(),
        );

        // Right-aligned status badges (access, and any health / metadata-only), with
        // the hovered one's bubble — all through the one uniform bar.
        Self::render_badge_bar(frame, config.badges, config.hovered_badge);

        // Clickable regions: each footer chip (the bottom-pinned hint block, at
        // `hint_y`) plus the top-right `[×]` (→ quit the tree).
        let mut regions = chip_regions(&chips, hint_y);
        regions.extend(close_button(frame, hint_key('q')));
        regions
    }
}

/// Returns the number of layers when `children` form a stack of numbered
/// subgroups (as in a transformer's `layers` group): there is at least one
/// subgroup and every subgroup has a purely numeric name. A single layer
/// counts too, so incomplete checkpoints still report their depth. Returns
/// `None` when the children are not such a stack.
fn layer_count(children: &[TreeNode]) -> Option<usize> {
    let mut numbered = 0;
    let mut groups = 0;
    for child in children {
        if let TreeNode::Group { name, .. } = child {
            groups += 1;
            if !name.is_empty() && name.chars().all(|c| c.is_ascii_digit()) {
                numbered += 1;
            }
        }
    }
    (groups > 0 && numbered == groups).then_some(numbered)
}

/// The search bar header line: `Search [query▒]  N matches  Enter view · …`.
fn tree_search_line(config: &DrawConfig) -> Line<'static> {
    let dim = Style::default().fg(palette::DIM);
    let key_style = Style::default()
        .fg(palette::KEY)
        .add_modifier(Modifier::BOLD);
    let mut spans: Vec<Span> = vec![Span::styled("Search ", dim)];

    // Input box: leading space, the query, a caret block when the cursor is at
    // the end, padded to a minimum width, then a trailing space.
    let q = config.search_query;
    let qlen = q.chars().count();
    let mut boxed = String::from(" ");
    boxed.push_str(q);
    if config.search_cursor >= qlen {
        boxed.push('█');
    }
    for _ in qlen..16 {
        boxed.push(' ');
    }
    boxed.push(' ');
    spans.push(Span::styled(
        boxed,
        Style::default().bg(palette::INPUT_BG).fg(palette::INPUT_FG),
    ));

    if q.is_empty() {
        spans.push(Span::raw("  "));
    } else {
        let n = config.tree.len();
        spans.push(Span::styled(
            format!("  {n} {}  ", if n == 1 { "match" } else { "matches" }),
            dim,
        ));
    }
    for (i, (key, label)) in [("Enter", "view"), ("Tab", "in tree"), ("Esc", "exit")]
        .iter()
        .enumerate()
    {
        if i > 0 {
            spans.push(Span::styled(" · ", dim));
        }
        spans.push(Span::styled(key.to_string(), key_style));
        spans.push(Span::raw(format!(" {label}")));
    }
    Line::from(spans)
}

/// One tree row as a styled [`Line`]: group names in the accent and dtypes amber,
/// with the name, shape and size at full strength and only the leaf marker /
/// storage tag dimmed; a `selected` row is drawn plain so the caller's highlight
/// reads cleanly.
/// The plain text of one tree row (no colour), exactly as [`tree_node_line`]
/// draws it — the shared building block for exporting the tree / a tensor list
/// (`t`, `--print-tree`, `--print-tensors`).
pub(crate) fn tree_row_text(
    node: &TreeNode,
    depth: usize,
    unindexed: &HashSet<String>,
    packing_schemas: &HashMap<String, PackingSchema>,
) -> String {
    line_to_text(&tree_row_line(node, depth, unindexed, packing_schemas))
}

/// The styled tree row (the colour the browser draws) — the building block for
/// the export text and the copy-menu preview.
pub(crate) fn tree_row_line(
    node: &TreeNode,
    depth: usize,
    unindexed: &HashSet<String>,
    packing_schemas: &HashMap<String, PackingSchema>,
) -> Line<'static> {
    tree_node_line(
        node,
        depth,
        false,
        unindexed,
        packing_schemas,
        MetaDisplay::Full,
    )
}

/// A tensor's row for the flat list: the same coloured fields as the tree, at
/// its full name, but without the leading `·` bullet a flat list doesn't need.
pub(crate) fn tensor_list_line(
    info: &TensorInfo,
    unindexed: &HashSet<String>,
    packing_schemas: &HashMap<String, PackingSchema>,
) -> Line<'static> {
    let node = TreeNode::Tensor {
        info: info.clone(),
        label: None,
    };
    without_bullet(tree_node_line(
        &node,
        0,
        false,
        unindexed,
        packing_schemas,
        MetaDisplay::Capped,
    ))
}

/// How a metadata value is rendered in a tree row: capped to keep the live
/// tree's rows short, or in full for exports (`--print-tree`, the `t` preview)
/// where the whole value is wanted.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MetaDisplay {
    Capped,
    Full,
}

fn tree_node_line(
    node: &TreeNode,
    depth: usize,
    selected: bool,
    unindexed: &HashSet<String>,
    packing_schemas: &HashMap<String, PackingSchema>,
    meta: MetaDisplay,
) -> Line<'static> {
    let indent = "  ".repeat(depth);
    let plain = |t: String| tree_span(selected, Color::Reset, t);
    let mut s: Vec<Span> = vec![tree_span(selected, Color::Reset, indent)];

    match node {
        TreeNode::Group {
            name,
            children,
            expanded,
            tensor_count,
            params,
            total_size,
            stored_size,
        } => {
            let arrow = if *expanded { "▾" } else { "▸" };
            let layer_prefix = match layer_count(children) {
                Some(n) => format!("≡ {n}, "),
                None => String::new(),
            };
            let size_field = if stored_size != total_size {
                format!(
                    "{} {SIZE_ARROW} {}",
                    format_size(*total_size),
                    format_size(*stored_size)
                )
            } else {
                format_size(*total_size)
            };
            s.push(tree_span(selected, palette::ACCENT, arrow));
            s.push(tree_span(selected, Color::Reset, " "));
            s.push(tree_span(selected, palette::ACCENT, name.clone()));
            let meta = if depth == 0 {
                format!(
                    " (▦ {tensor_count}, {} params, {size_field})",
                    format_parameters(*params)
                )
            } else {
                format!(" ({layer_prefix}▦ {tensor_count}, {size_field})")
            };
            s.push(plain(meta));
        }
        TreeNode::Tensor { info, label } => {
            let display_name = if depth == 0 {
                info.name.as_str()
            } else if let Some(label) = label {
                label.as_str()
            } else {
                crate::tree::last_segment(&info.name)
            };
            if unindexed.contains(&info.source_path) {
                s.push(tree_span(selected, palette::UNINDEXED, UNINDEXED_MARK));
            } else {
                s.push(tree_span(selected, palette::DIM, "·"));
            }
            s.push(plain(format!(" {display_name} [")));
            s.push(tree_span(selected, palette::DTYPE, info.dtype.clone()));
            let schema = packing_schemas.get(&info.name);
            if let Some(sc) = schema {
                s.push(tree_span(selected, palette::DIM, " as "));
                s.push(tree_span(selected, palette::DTYPE, sc.label()));
            }
            s.push(plain(format!(", {}", format_shape(&info.shape))));
            if let Some(sc) = schema {
                let logical =
                    ViewDtype::Unpacked.logical_shape_with(&info.shape, &info.dtype, Some(sc));
                s.push(tree_span(selected, palette::DIM, " as "));
                s.push(plain(format_shape(&logical)));
            }
            s.push(plain(", ".to_string()));
            match &info.storage {
                Storage::Compressed {
                    codec,
                    stored_bytes,
                } => {
                    s.push(plain(format!(
                        "{} {SIZE_ARROW} {} ",
                        format_size(info.size_bytes),
                        format_size(*stored_bytes)
                    )));
                    s.push(tree_span(
                        selected,
                        palette::DIM,
                        format!("({COMPRESSED_MARK} {codec})"),
                    ));
                }
                Storage::Raw => {
                    s.push(plain(format!("{} ", format_size(info.size_bytes))));
                    s.push(tree_span(selected, palette::DIM, UNCOMPRESSED_TAG));
                }
                Storage::Unknown => s.push(plain(format_size(info.size_bytes))),
            }
            s.push(plain("]".to_string()));
        }
        TreeNode::Metadata { info } => {
            let flat = info.value.split_whitespace().collect::<Vec<_>>().join(" ");
            // Exports keep the whole value; the live tree caps it so rows stay short.
            let truncated_value = if meta == MetaDisplay::Full || flat.chars().count() <= 50 {
                flat
            } else {
                let head: String = flat.chars().take(47).collect();
                format!("{head}...")
            };
            s.push(tree_span(selected, palette::META, "†"));
            s.push(tree_span(selected, Color::Reset, " "));
            s.push(tree_span(
                selected,
                palette::META,
                metadata_short(&info.name),
            ));
            s.push(tree_span(
                selected,
                palette::DIM,
                format!(" [{}]: {truncated_value}", info.value_type),
            ));
        }
    }
    Line::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::MetadataInfo;
    use crate::ui::{AccessBadge, Badge, status_badges};

    #[test]
    fn tree_scrollbar_geometry_and_mapping() {
        // Everything fits the viewport → no bar.
        assert!(UI::tree_scrollbar(80, 40, false, false, false, 5, 0).is_none());
        // Too narrow for a bar plus content → no bar.
        assert!(UI::tree_scrollbar(1, 40, false, false, false, 999, 0).is_none());

        // Overflow → a bar in the rightmost column, tracking the visible rows.
        let visible = UI::tree_visible_rows(80, 20, false, false, false);
        let sb = UI::tree_scrollbar(80, 20, false, false, false, visible + 50, 0)
            .expect("overflow shows bar");
        assert_eq!(sb.col, 79);
        assert_eq!(sb.rows as usize, visible);
        assert_eq!(sb.max_offset, 50);
        assert_eq!(sb.top as usize, UI::tree_header_rows(false));

        // Track top → offset 0, track bottom → max_offset; outside the track clamps.
        assert_eq!(sb.offset_at(sb.top), 0);
        assert_eq!(sb.offset_at(sb.top + sb.rows - 1), sb.max_offset);
        assert_eq!(sb.offset_at(0), 0);
        assert_eq!(sb.offset_at(sb.top + sb.rows + 99), sb.max_offset);
        // The middle track row maps near the middle offset (25 = max_offset/2).
        // A discrete bar only lands on multiples of one track-step, so the closest
        // position to the true centre is within a step — the tolerance tracks the
        // viewport height rather than assuming a fixed parity.
        let mid = sb.offset_at(sb.top + (sb.rows - 1) / 2);
        let step = (sb.max_offset as f64 / f64::from(sb.rows - 1)).ceil() as i64;
        assert!(
            (mid as i64 - 25).abs() <= step,
            "midpoint offset {mid} ≈ 25 (±{step})"
        );

        // Hit-testing: only the bar's own column, within the track rows.
        assert!(sb.hit(79, sb.top));
        assert!(sb.hit(79, sb.top + sb.rows - 1));
        assert!(!sb.hit(78, sb.top)); // wrong column
        assert!(!sb.hit(79, sb.top + sb.rows)); // just past the track
        assert!(!sb.hit(79, sb.top - 1)); // header row above the track
    }

    #[test]
    fn tree_scrollbar_drawn_only_when_interactive() {
        // A helper config over `nodes`, differing only in the `interactive` gate.
        fn cfg<'a>(
            nodes: &'a [(TreeNode, usize)],
            unindexed: &'a HashSet<String>,
            schemas: &'a HashMap<String, PackingSchema>,
            badges: &'a [Badge],
            interactive: bool,
        ) -> DrawConfig<'a> {
            DrawConfig {
                tree: nodes,
                current_file: "f",
                file_idx: 0,
                total_files: 1,
                selected_idx: 0,
                scroll_offset: 0,
                search_mode: false,
                search_query: "",
                search_cursor: 0,
                filter_query: "",
                status_icon: "▪",
                status_bar: "",
                status_secondary: "",
                can_repack: false,
                can_rename: false,
                unindexed,
                packing_schemas: schemas,
                copied_flash: None,
                interactive,
                badges,
                hovered_badge: None,
            }
        }

        // 40 rows into a 20-row terminal → the tree overflows the viewport.
        let nodes: Vec<(TreeNode, usize)> = (0..40)
            .map(|i| {
                (
                    TreeNode::Metadata {
                        info: MetadataInfo {
                            name: format!("entry_{i}"),
                            value: "v".to_string(),
                            value_type: "str".to_string(),
                        },
                    },
                    0usize,
                )
            })
            .collect();
        let unindexed = HashSet::new();
        let schemas = HashMap::new();
        let badges = status_badges(AccessBadge::ReadOnly, None, false);

        // Interactive: render_tree reserves the column and the engine draws the bar
        // (thumb █ over a │ track) via render_vscrollbar — the live composition.
        let live = crate::tui::headless_render(80, 20, |f| {
            UI::render_tree(f, &cfg(&nodes, &unindexed, &schemas, &badges, true));
            if let Some(sb) = UI::tree_scrollbar(80, 20, false, false, false, nodes.len(), 0) {
                UI::render_vscrollbar(f, &sb);
            }
        })
        .unwrap();
        assert!(live.contains('█'), "expected a thumb:\n{live}");
        assert!(live.contains('│'), "expected a track:\n{live}");

        // Non-interactive (headless dump): render_tree draws no bar on its own.
        let plain = crate::tui::headless_render(80, 20, |f| {
            UI::render_tree(f, &cfg(&nodes, &unindexed, &schemas, &badges, false));
        })
        .unwrap();
        assert!(
            !plain.contains('█') && !plain.contains('│'),
            "headless dump must show no scroll bar:\n{plain}"
        );
    }
}
