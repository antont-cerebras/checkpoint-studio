//! The status badges on the tree's top line (read-only / metadata-only /
//! health) and their right-to-left layout, hit-testing and hover bubbles.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use super::UI;
use super::palette;
use super::popup::render_hover_bubble;

/// Severity of the checkpoint-health badge on the tree's status line: a real
/// error (missing files/tensors — the checkpoint may be incomplete) shows a red
/// badge; warnings only (e.g. extra files on disk not in the index) show a softer
/// orange one, so the screaming red is reserved for genuine problems.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum HealthAlert {
    Warning,
    Error,
}

/// What the bottom-right access badge advertises about the currently open
/// checkpoint: whether the tool can rewrite it in place. Only a **local
/// safetensors** checkpoint is [`Editable`](AccessBadge::Editable) — the in-place
/// rename (`convert --map` / the `R` action) is the one path that modifies it;
/// everything else (a remote `--ssh-proxy` read, an HDF5 file, plain exports) is
/// [`ReadOnly`](AccessBadge::ReadOnly), and browsing never modifies it either way.
/// It is the rightmost [`Badge`] in the [`status bar`](UI::render_badge_bar).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AccessBadge {
    ReadOnly,
    Editable,
}

impl AccessBadge {
    /// The chip text, symmetrically padded with one space on each side.
    const fn label(self) -> &'static str {
        match self {
            Self::ReadOnly => " read-only ",
            Self::Editable => " editable ",
        }
    }

    /// The chip foreground: reassuring green when read-only, attention-drawing
    /// amber when the checkpoint can be rewritten in place.
    fn color(self) -> Color {
        match self {
            Self::ReadOnly => palette::SUCCESS,
            Self::Editable => palette::WARN,
        }
    }

    /// The hover-bubble text explaining what the badge means.
    fn hover(self) -> &'static str {
        match self {
            Self::ReadOnly => {
                "The checkpoint you open is never modified — browsing and exports only \
                 ever read it. (Repack / convert write a new file, leaving the original \
                 untouched.)"
            }
            Self::Editable => {
                "Browsing and exports never modify this checkpoint. The one exception is \
                 the in-place rename (R / convert --map), which rewrites the headers \
                 after you confirm."
            }
        }
    }
}

pub(super) const HEALTH_BADGE: &str = " ⚠ health ";

pub(super) const METADATA_BADGE: &str = " metadata-only ";

/// Default-background columns left between adjacent badges (and between the
/// leftmost badge and the status text), so the `STATUS_BG` chips read as separate
/// badges rather than one bar.
const BADGE_GAP: u16 = 2;

/// One chip in the bottom-right **status bar** — the uniform model behind the
/// access / health / metadata-only badges. Built by [`UI::status_badges`] and laid
/// out / drawn / hit-tested by the `badge_bar_*` functions, so every badge shares
/// one path for width, gap, colour, hover bubble and click action (they used to be
/// three hand-threaded implementations that kept drifting).
#[derive(Clone, Copy)]
pub(crate) struct Badge {
    /// Chip text, already padded (`" read-only "`, `" ⚠ health "`, …); also the
    /// hover bubble's title.
    label: &'static str,
    fg: Color,
    bg: Color,
    /// The hover bubble's border / title colour.
    accent: Color,
    /// The hover bubble body (word-wrapped by [`render_hover_bubble`]).
    help: &'static str,
    /// The key a click on this badge synthesizes (e.g. `h` opens the health
    /// report), or `None` if the badge isn't actionable.
    action: Option<char>,
}

impl Badge {
    /// The chip's display width (the `⚠` glyph is wide).
    fn width(self) -> u16 {
        use unicode_width::UnicodeWidthStr;
        self.label.width() as u16
    }

    /// The key a click on this badge acts as, if any.
    pub(crate) fn action(self) -> Option<char> {
        self.action
    }
}

/// Build the bottom-right status badges in **right-to-left** order (index 0 is the
/// rightmost, hugging the edge): the access badge always, then the health badge
/// when the index/file check flagged something, then the metadata-only badge for a
/// remote source. This is the single source of truth both the renderer and the
/// hover / click hit-test build from, so they can't disagree.
pub(crate) fn status_badges(
    access: AccessBadge,
    health: Option<HealthAlert>,
    metadata_only: bool,
) -> Vec<Badge> {
    let mut badges = vec![Badge {
        label: access.label(),
        fg: access.color(),
        bg: palette::STATUS_BG,
        accent: access.color(),
        help: access.hover(),
        action: None,
    }];
    if let Some(alert) = health {
        let (bg, help) = match alert {
            HealthAlert::Error => (
                palette::ALERT,
                "Index / file mismatch — files or tensors the index references are \
                 missing on disk, so the checkpoint may be incomplete. Click (or press \
                 h) for the health report.",
            ),
            HealthAlert::Warning => (
                palette::WARN_BG,
                "Index / file mismatch (warnings only) — e.g. files on disk the index \
                 doesn't reference. Click (or press h) for the health report.",
            ),
        };
        badges.push(Badge {
            label: HEALTH_BADGE,
            fg: palette::STATUS_FG,
            bg,
            accent: bg,
            help,
            action: Some('h'),
        });
    }
    if metadata_only {
        badges.push(Badge {
            label: METADATA_BADGE,
            fg: palette::WARN,
            bg: palette::STATUS_BG,
            accent: palette::WARN,
            help: "A remote source: only header metadata is loaded, so the data views \
                   (heatmap / grid / histogram / statistics) need the file locally.",
            action: None,
        });
    }
    badges
}

/// The on-screen rect of each badge on the bottom row — right-aligned, index 0 at
/// the edge, each `BADGE_GAP` apart. `None` for a badge that doesn't fit the frame.
/// The one geometry the renderer, hit-test and reserve all share.
fn badge_rects(width: u16, height: u16, badges: &[Badge]) -> Vec<Option<Rect>> {
    let mut rects = Vec::with_capacity(badges.len());
    let mut right = 0u16; // columns already spoken for to the right (incl. gaps)
    for b in badges {
        let w = b.width();
        let rect = (height > 0 && width > right + w).then(|| Rect {
            x: width - right - w,
            y: height - 1,
            width: w,
            height: 1,
        });
        rects.push(rect);
        right += w + BADGE_GAP;
    }
    rects
}

impl UI {
    /// Draw the bottom-right **status bar** — every badge in `badges` (from
    /// [`status_badges`]) right-aligned on the last row, and, when `hovered` is
    /// `Some(i)`, that badge's hover bubble floated above it. Rendered last on a
    /// view so the chips sit over whatever occupies that row.
    pub(crate) fn render_badge_bar(frame: &mut Frame, badges: &[Badge], hovered: Option<usize>) {
        let area = frame.area();
        let rects = badge_rects(area.width, area.height, badges);
        for (b, rect) in badges.iter().zip(&rects) {
            if let Some(r) = rect {
                Paragraph::new(Line::from(Span::styled(
                    b.label,
                    Style::default()
                        .bg(b.bg)
                        .fg(b.fg)
                        .add_modifier(Modifier::BOLD),
                )))
                .render(*r, frame.buffer_mut());
            }
        }
        // The hover bubble goes last so it floats over the neighbouring chips.
        if let Some(i) = hovered
            && let (Some(b), Some(Some(r))) = (badges.get(i), rects.get(i))
        {
            render_hover_bubble(frame, *r, b.accent, Some(b.label), b.help);
        }
    }

    /// The index of the badge under `(col, row)`, if any — for the hover bubble and
    /// click actions. Shares [`badge_rects`] with the renderer, so they can't drift.
    pub(crate) fn badge_bar_hit(
        width: u16,
        height: u16,
        col: u16,
        row: u16,
        badges: &[Badge],
    ) -> Option<usize> {
        badge_rects(width, height, badges)
            .into_iter()
            .position(|r| r.is_some_and(|r| row == r.y && col >= r.x && col < r.x + r.width))
    }

    /// Columns the badge bar reserves on the right of the status line, so the
    /// status text never runs under it (a [`BADGE_GAP`] before each badge).
    pub(crate) fn badge_bar_width(badges: &[Badge]) -> u16 {
        badges.iter().map(|b| b.width() + BADGE_GAP).sum()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use super::*;
    use crate::tree::TreeNode;
    use crate::ui::DrawConfig;
    use crate::ui::tests_support::strip_ansi_codes;

    #[test]
    fn health_badge_sits_by_the_read_only_badge_with_a_hover_bubble() {
        let nodes: Vec<(TreeNode, usize)> = Vec::new();
        let unindexed = HashSet::new();
        let schemas = HashMap::new();
        // access (idx 0) + health (idx 1); hovering the health badge = Some(1).
        let badges = status_badges(AccessBadge::ReadOnly, Some(HealthAlert::Error), false);
        let mk = |hovered_badge: Option<usize>| DrawConfig {
            tree: &nodes,
            current_file: "model",
            file_idx: 0,
            total_files: 1,
            selected_idx: 0,
            scroll_offset: 0,
            search_mode: false,
            search_query: "",
            search_cursor: 0,
            filter_query: "",
            status_icon: "▪",
            status_bar: "model.safetensors",
            status_secondary: "",
            can_repack: false,
            can_rename: false,
            unindexed: &unindexed,
            packing_schemas: &schemas,
            copied_flash: None,
            interactive: true,
            badges: &badges,
            hovered_badge,
        };

        // Not hovering: the short `⚠ health` badge shows on the bottom line, on the
        // same row as `read-only` and to its left — never in the title.
        let out = crate::tui::headless_render(120, 40, |f| {
            UI::render_tree(f, &mk(None));
        })
        .unwrap();
        let plain = strip_ansi_codes(&out);
        let lines: Vec<&str> = plain.lines().collect();
        assert!(
            !lines[0].contains('⚠'),
            "no alert in the title: {:?}",
            lines[0]
        );
        let badge_row = lines
            .iter()
            .find(|l| l.contains("read-only"))
            .expect("the read-only badge renders");
        assert!(
            badge_row.contains("⚠ health"),
            "the health badge should share the read-only line: {badge_row:?}"
        );
        assert!(
            badge_row.find("⚠ health") < badge_row.find("read-only"),
            "the health badge should sit left of read-only: {badge_row:?}"
        );
        // No hover → no help bubble.
        assert!(
            !plain.contains("Index / file mismatch"),
            "bubble only on hover:\n{plain}"
        );

        // Hovering the badge (index 1) floats its help bubble.
        let hovered = crate::tui::headless_render(120, 40, |f| {
            UI::render_tree(f, &mk(Some(1)));
        })
        .unwrap();
        assert!(
            strip_ansi_codes(&hovered).contains("Index / file mismatch"),
            "hovering the health badge should float its help bubble:\n{hovered}"
        );
    }

    #[test]
    fn access_badge_reflects_editability_with_symmetric_padding() {
        for mode in [AccessBadge::ReadOnly, AccessBadge::Editable] {
            // One space of padding on each side (the user flagged an asymmetric chip).
            let label = mode.label();
            assert!(
                label.starts_with(' ')
                    && label.ends_with(' ')
                    && !label.starts_with("  ")
                    && !label.ends_with("  "),
                "{mode:?} label should be symmetrically padded: {label:?}"
            );
        }
        for (mode, word) in [
            (AccessBadge::ReadOnly, "read-only"),
            (AccessBadge::Editable, "editable"),
        ] {
            let badges = status_badges(mode, None, false);
            let out =
                crate::tui::headless_render(120, 6, |f| UI::render_badge_bar(f, &badges, None))
                    .unwrap();
            let plain = strip_ansi_codes(&out);
            let last = plain.lines().last().unwrap_or_default();
            assert!(
                last.trim_end().ends_with(word),
                "{mode:?} should show {word:?}: {last:?}"
            );
        }
        // Hovering the editable badge (index 0) floats its in-place hint.
        let badges = status_badges(AccessBadge::Editable, None, false);
        let hint = strip_ansi_codes(
            &crate::tui::headless_render(120, 12, |f| UI::render_badge_bar(f, &badges, Some(0)))
                .unwrap(),
        );
        assert!(
            hint.contains("in-place") || hint.contains("convert --map"),
            "editable hint should mention the in-place exception:\n{hint}"
        );
    }

    #[test]
    fn badge_bar_hit_finds_the_badge_under_the_cursor() {
        // access (idx 0, rightmost) + health (idx 1) on a 120×40 frame.
        let badges = status_badges(AccessBadge::ReadOnly, Some(HealthAlert::Error), false);
        let rects = badge_rects(120, 40, &badges);
        let r0 = rects[0].expect("access fits");
        let r1 = rects[1].expect("health fits");
        assert_eq!(r0.y, 39, "on the bottom row");
        assert_eq!(r0.x + r0.width, 120, "access badge is flush right");
        assert!(
            r1.x + r1.width < r0.x,
            "health sits left of access: {r1:?} {r0:?}"
        );
        // A click maps to whichever badge is under it (and misses the row above).
        assert_eq!(UI::badge_bar_hit(120, 40, r0.x, 39, &badges), Some(0));
        assert_eq!(UI::badge_bar_hit(120, 40, r1.x, 39, &badges), Some(1));
        assert_eq!(UI::badge_bar_hit(120, 40, r1.x, 38, &badges), None);
        // Too narrow → nothing fits, nothing hits.
        assert!(badge_rects(8, 40, &badges).iter().all(Option::is_none));
        assert_eq!(UI::badge_bar_hit(8, 40, 4, 39, &badges), None);
    }

    #[test]
    fn badge_bar_lays_out_right_to_left_with_gaps() {
        let (w, h) = (120u16, 40u16);
        // access(0) + health(1) + metadata(2), each a BADGE_GAP left of the previous.
        let badges = status_badges(AccessBadge::ReadOnly, Some(HealthAlert::Error), true);
        let rects: Vec<Rect> = badge_rects(w, h, &badges)
            .into_iter()
            .map(|r| r.expect("fits"))
            .collect();
        assert_eq!(
            rects[0].x + rects[0].width,
            w,
            "rightmost badge is flush right"
        );
        for i in 1..rects.len() {
            assert_eq!(
                rects[i].x + rects[i].width + BADGE_GAP,
                rects[i - 1].x,
                "badge {i} sits a gap left of badge {}",
                i - 1
            );
            assert_eq!(rects[i].y, h - 1);
        }
        // Dropping the metadata badge leaves just the two.
        assert_eq!(
            badge_rects(
                w,
                h,
                &status_badges(AccessBadge::ReadOnly, Some(HealthAlert::Error), false)
            )
            .len(),
            2
        );
    }
}
