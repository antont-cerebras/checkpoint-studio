//! The **side-by-side compare screen**: two checkpoints in two panes, browsed in lockstep.
//!
//! Renders [`checkpoint_studio_core::difftree`]'s aligned tree — the same model the browser draws,
//! so the two surfaces cannot disagree about what differs. One aligned row becomes one terminal
//! line: the baseline's content in the left pane, the current checkpoint's in the right, and a blank
//! pane where a row exists on only one side. Because there is one tree, there is one fold state and
//! one scroll position; nothing has to be kept in step.
//!
//! The gutter carries `+`/`-`/`~` in the same colours the diff report uses, so a row's status reads
//! the same whether you are looking at the report or at the panes.

use ratatui::Frame;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use super::palette;
use checkpoint_studio_core::difftree::{AlignedNode, Side, Status};

/// One visible line: which node, how deep, and whether its children are showing.
pub(crate) struct CompareRow<'a> {
    pub node: &'a AlignedNode,
    pub depth: usize,
    pub expanded: bool,
}

/// Rows above the list: the two paths, the difference count, and the rule under them.
const HEADER_ROWS: usize = 3;

/// How many list rows fit — what the caller clamps scrolling to. The footer wraps at narrow
/// widths, so its height is asked for rather than assumed.
pub(crate) fn body_rows(height: u16) -> usize {
    body_rows_at(height, 80)
}

fn body_rows_at(height: u16, width: u16) -> usize {
    let footer = super::hints::compare_hint_lines(width).0.len();
    (height as usize).saturating_sub(HEADER_ROWS + footer)
}

/// Draw the screen. Returns the maximum scroll offset, so the key handler can clamp to content
/// without duplicating the layout arithmetic.
/// Everything the screen draws, so the call reads as named fields rather than eight positions.
pub(crate) struct CompareFrame<'a> {
    pub base_label: &'a str,
    pub new_label: &'a str,
    pub rows: &'a [CompareRow<'a>],
    /// Index into `rows` of the highlighted line.
    pub cursor: usize,
    pub scroll: usize,
    /// How many differing **tensors** the comparison found — the tally's number, so it is the same one
    /// the browser and the one-page report print, folded or not.
    pub differences: usize,
    /// How many *rows* `n`/`N` stop on, and which of them the cursor is on. Not the same as
    /// `differences` once families are folded: one `{0-61}` row can stand for 62 differing tensors, and
    /// a position counted against the tensor total would claim stops that do not exist.
    pub stops: usize,
    pub cursor_of: Option<usize>,
}

pub(crate) fn render(f: &mut Frame, v: &CompareFrame<'_>) -> usize {
    let CompareFrame {
        base_label,
        new_label,
        rows,
        cursor,
        scroll,
        differences,
        stops,
        cursor_of,
    } = *v;
    let area = f.area();
    let visible = body_rows_at(area.height, area.width);

    // The two panes split what is left after the gutter. Equal halves: neither side is the one
    // being read, so giving either more room would be a claim about which matters.
    let gutter = 2u16;
    let pane = area.width.saturating_sub(gutter) / 2;

    let mut lines: Vec<Line<'static>> = Vec::with_capacity(visible + HEADER_ROWS);
    lines.push(Line::from(vec![
        Span::styled(
            format!(
                "{:<w$}",
                truncate(base_label, pane as usize),
                w = pane as usize
            ),
            Style::default().fg(palette::REMOVED),
        ),
        Span::raw("  "),
        Span::styled(
            truncate(new_label, pane as usize),
            Style::default().fg(palette::ADDED),
        ),
    ]));
    // The matching case says what it checked, on the same line.
    //
    // "structurally identical" is easy to over-read as "the weights are the same". This comparison
    // never touches a tensor's bytes — that is `diff --values`, which has a progress bar for it — so
    // the qualifier travels with the verdict. The browser says the same thing in a banner; here the
    // header is a fixed three rows, so it is one line with the verdict emphasised and the caveat dim.
    lines.push(if differences == 0 {
        Line::from(vec![
            Span::styled(
                "structurally identical",
                Style::default()
                    .fg(palette::ADDED)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " — names, dtypes, shapes and metadata all match; values not compared",
                Style::default().fg(palette::DIM),
            ),
        ])
    } else {
        let at = cursor_of.map_or_else(String::new, |i| {
            format!(
                " · row {} of {}",
                crate::utils::format_count(i + 1),
                crate::utils::format_count(stops)
            )
        });
        Line::from(Span::styled(
            format!(
                "{} difference{}{at}",
                crate::utils::format_count(differences),
                plural(differences)
            ),
            Style::default().fg(palette::DIM),
        ))
    });
    lines.push(Line::from(Span::styled(
        "─".repeat(area.width as usize),
        Style::default().fg(palette::DIM),
    )));

    for (i, row) in rows.iter().skip(scroll).take(visible).enumerate() {
        let selected = scroll + i == cursor;
        lines.push(row_line(row, pane, selected));
    }

    Paragraph::new(lines).render(area, f.buffer_mut());

    // The footer, pinned to the bottom like every other screen's: `n`/`N` are why this screen
    // exists, so they have to be on it.
    let (hint_lines, _chips) = super::hints::compare_hint_lines(area.width);
    let rows_tall = u16::try_from(hint_lines.len()).unwrap_or(1);
    let footer = ratatui::layout::Rect {
        x: 0,
        y: area.height.saturating_sub(rows_tall),
        width: area.width,
        height: rows_tall.min(area.height),
    };
    Paragraph::new(hint_lines).render(footer, f.buffer_mut());

    rows.len().saturating_sub(visible)
}

/// One aligned row as one line: gutter mark, then each side in its own pane.
fn row_line(row: &CompareRow<'_>, pane: u16, selected: bool) -> Line<'static> {
    let (mark, colour) = match row.node.status {
        Status::Same => (' ', palette::DIM),
        Status::Changed => ('~', palette::CHANGED),
        Status::OnlyNew => ('+', palette::ADDED),
        Status::OnlyOld => ('-', palette::REMOVED),
    };
    let base = if selected {
        Style::default().add_modifier(Modifier::REVERSED)
    } else {
        Style::default()
    };
    Line::from(vec![
        Span::styled(format!("{mark} "), base.fg(colour)),
        Span::styled(
            format!(
                "{:<w$}",
                truncate(&cell(row, row.node.old.as_ref()), pane as usize),
                w = pane as usize
            ),
            base.fg(
                if matches!(row.node.status, Status::Changed | Status::OnlyOld) {
                    palette::REMOVED
                } else {
                    palette::DIM
                },
            ),
        ),
        Span::styled(
            truncate(&cell(row, row.node.new.as_ref()), pane as usize),
            base.fg(
                if matches!(row.node.status, Status::Changed | Status::OnlyNew) {
                    palette::ADDED
                } else {
                    palette::DIM
                },
            ),
        ),
    ])
}

/// What one pane shows for a row: indent, twisty, name, and the side's signature. Empty when the
/// row does not exist on that side — the blank is what makes a one-sided row obvious.
fn cell(row: &CompareRow<'_>, side: Option<&Side>) -> String {
    let Some(side) = side else {
        return String::new();
    };
    let indent = "  ".repeat(row.depth);
    let twisty = if row.node.is_group() {
        if row.expanded { "▾ " } else { "▸ " }
    } else {
        "· "
    };
    let detail = match side {
        // `×256` when an alignment folded several tensors onto this row — the terminal's compare screen
        // says the same thing the report's rows do.
        Side::Tensor { info, fold } => format!(
            "  [{} {}]{}",
            info.dtype,
            shape(&info.shape),
            fold.map(|parts| format!("  ×{parts}")).unwrap_or_default()
        ),
        Side::Metadata { value, .. } => format!("  {value}"),
        Side::Group { tensor_count, .. } => {
            // A folded group that hides differences says so; an open one would only repeat what the
            // rows beneath it already show.
            if row.expanded {
                String::new()
            } else if row.node.differing > 0 {
                format!("  ({} differ)", row.node.differing)
            } else {
                format!("  (▦ {tensor_count})")
            }
        }
    };
    // `×62` when family folding put 62 identical layers on this row — the browser says the same.
    let family = if row.node.members > 1 {
        format!("  ×{}", row.node.members)
    } else {
        String::new()
    };
    format!("{indent}{twisty}{}{family}{detail}", row.node.name)
}

/// `(4,)` for a vector, `(6, 4)` otherwise — the spelling the rest of the UI uses, so a shape reads
/// the same here as on the detail screen.
fn shape(dims: &[usize]) -> String {
    match dims {
        [one] => format!("({one},)"),
        many => format!(
            "({})",
            many.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn plural(n: usize) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// Cut to `width` with a trailing `…`, so a long tensor name cannot push a pane into its neighbour.
fn truncate(s: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let count = s.chars().count();
    if count <= width {
        return s.to_string();
    }
    let keep = width.saturating_sub(1);
    let mut out: String = s.chars().take(keep).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use checkpoint_studio_core::difftree::align_rooted;
    use checkpoint_studio_core::tree::{Layout, Storage, TensorInfo, TreeNode};

    fn tensor(name: &str, dtype: &str, shape: &[usize]) -> TreeNode {
        TreeNode::Tensor {
            info: TensorInfo {
                name: name.to_string(),
                dtype: dtype.to_string(),
                shape: shape.to_vec(),
                size_bytes: 4,
                num_elements: 1,
                storage: Storage::Raw,
                source_path: "/ckpt/model.safetensors".to_string(),
                layout: Layout::None,
            },
            label: None,
        }
    }

    /// Align `old` against `new`, draw the whole thing collapsed at `w`×`h`, and return the text.
    ///
    /// Every test here needs the same five steps; spelling them out per test made the setup longer
    /// than the assertion and duplicated enough to trip the clone gate.
    fn drawn(
        old: &[TreeNode],
        new: &[TreeNode],
        w: u16,
        h: u16,
        differences: usize,
        cursor_of: Option<usize>,
    ) -> String {
        let aligned = align_rooted(old, new);
        let rows: Vec<CompareRow<'_>> = aligned
            .iter()
            .map(|node| CompareRow {
                node,
                depth: 0,
                expanded: false,
            })
            .collect();
        crate::tui::headless_render(w, h, |f| {
            render(
                f,
                &CompareFrame {
                    base_label: "old.safetensors",
                    new_label: "new.safetensors",
                    rows: &rows,
                    cursor: 0,
                    scroll: 0,
                    differences,
                    // In these tests every difference is its own row, so the stops are the differences.
                    stops: differences,
                    cursor_of,
                },
            );
        })
        .expect("renders")
    }

    fn root(name: &str, children: Vec<TreeNode>) -> Vec<TreeNode> {
        vec![TreeNode::Group {
            name: name.to_string(),
            children,
            expanded: true,
            tensor_count: 0,
            params: 0,
            total_size: 0,
            stored_size: 0,
        }]
    }

    /// The whole point of the screen: each side in its own pane, and a blank where a row is absent.
    #[test]
    fn the_two_panes_show_their_own_side() {
        let old = root(
            "old",
            vec![tensor("w", "F16", &[4]), tensor("gone", "F16", &[1])],
        );
        let new = root(
            "new",
            vec![tensor("w", "U16", &[4]), tensor("added", "F32", &[1])],
        );
        let text = drawn(&old, &new, 80, 12, 3, Some(0));

        assert!(text.contains("old.safetensors"), "{text}");
        assert!(text.contains("new.safetensors"), "{text}");
        // The total counts differing *tensors*; the position counts the rows `n`/`N` stop on, which is
        // the same number here because nothing is folded.
        assert!(text.contains("3 differences · row 1 of 3"), "{text}");
        // `w` differs, so both panes carry it with their own dtype.
        let w_line = text
            .lines()
            .find(|l| l.contains("w  [F16"))
            .expect("the changed row");
        assert!(w_line.contains("U16"), "both dtypes on one line: {w_line}");
        assert!(w_line.trim_start().starts_with('~'), "marked: {w_line}");
        // A removed row is on the left only; an added one on the right only.
        let gone = text
            .lines()
            .find(|l| l.contains("gone"))
            .expect("removed row");
        assert!(gone.trim_start().starts_with('-'), "{gone}");
        let added = text
            .lines()
            .find(|l| l.contains("added"))
            .expect("added row");
        assert!(added.trim_start().starts_with('+'), "{added}");
        // The added row's left pane is blank — that is what makes a one-sided row obvious.
        let left_half: String = added.chars().skip(2).take(38).collect();
        assert!(
            left_half.trim().is_empty(),
            "left pane should be blank: {added:?}"
        );
    }

    /// The keys the screen exists for have to be on the screen. Without a footer, `n`/`N` were
    /// undiscoverable — the tree's footer was the last one drawn, so it advertised the wrong keys.
    #[test]
    fn the_footer_advertises_the_difference_keys() {
        let t = root("ckpt", vec![tensor("w", "F16", &[4])]);
        let text = drawn(&t, &t, 100, 10, 0, None);
        assert!(text.contains("next/prev difference"), "{text}");
        assert!(text.contains("fold/unfold"), "{text}");
    }

    #[test]
    fn identical_checkpoints_say_so_instead_of_counting_zero() {
        let t = root("ckpt", vec![tensor("w", "F16", &[4])]);
        let text = drawn(&t, &t, 120, 8, 0, None);
        assert!(text.contains("structurally identical"), "{text}");
        // And say what "identical" covers. Without this the phrase reads as "the weights match", which
        // a structure-only comparison has not checked — two differently-trained checkpoints of one
        // architecture are structurally identical.
        assert!(text.contains("values not compared"), "{text}");
    }

    #[test]
    fn a_long_name_is_cut_rather_than_spilling_into_the_other_pane() {
        let long = "model.layers.0.block_sparse_moe.experts.down_proj.weight_scale_inv";
        let old = root("old", vec![tensor(long, "F16", &[4096, 4096])]);
        let text = drawn(&old, &old, 60, 8, 0, None);
        // 60 columns, 2 for the gutter → 29 per pane. Every line must fit the terminal.
        for l in text.lines() {
            assert!(l.chars().count() <= 60, "line overruns the width: {l:?}");
        }
        assert!(text.contains('…'), "the long name should be elided: {text}");
    }

    #[test]
    fn a_folded_group_reports_what_it_hides() {
        let old = root(
            "ckpt",
            vec![TreeNode::Group {
                name: "layers".to_string(),
                children: vec![
                    tensor("layers.a", "F16", &[1]),
                    tensor("layers.b", "F16", &[1]),
                ],
                expanded: true,
                tensor_count: 2,
                params: 0,
                total_size: 0,
                stored_size: 0,
            }],
        );
        let new = root(
            "ckpt",
            vec![TreeNode::Group {
                name: "layers".to_string(),
                children: vec![
                    tensor("layers.a", "U16", &[1]),
                    tensor("layers.b", "F16", &[1]),
                ],
                expanded: true,
                tensor_count: 2,
                params: 0,
                total_size: 0,
                stored_size: 0,
            }],
        );
        let text = drawn(&old, &new, 70, 8, 1, Some(0));
        assert!(
            text.contains("(1 differ)"),
            "a folded group must say what it hides: {text}"
        );
    }
}
