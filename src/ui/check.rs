//! The integrity-check report pop-up, including the fold that keeps a report
//! with thousands of findings navigable.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::UI;
use super::palette;
use super::popup::render_scroll_popup;

impl UI {
    /// Float the health-check report (`h` in the tree) over the live tree. Built
    /// as styled lines directly from the [`CheckReport`](crate::check::CheckReport)
    /// (so every span sits on the popup's panel background, matching the box) —
    /// coloured marks per check, indented findings, a verdict, and a `state`-driven
    /// footer. While scanning, the "Value scan" row becomes an animated spinner.
    /// Render the health-check popup, its body scrolled by `scroll` rows (the
    /// footer stays pinned). Returns the max valid scroll so the caller can clamp.
    pub(crate) fn render_check_report(
        frame: &mut Frame,
        report: &crate::check::CheckReport,
        state: CheckPopup,
        scroll: usize,
        expanded: bool,
    ) -> (usize, Vec<(Rect, KeyEvent)>) {
        use crate::check::{Severity, Status, count_phrase, fmt_elapsed};
        let bg = palette::PANEL_BG;
        // Every span carries the panel background, so text and box match.
        let sty = |s: String, style: Style| Span::styled(s, style.bg(bg));
        // Body-line indices of the per-check findings toggles (all clickable → `f`).
        let mut fold_lines: Vec<usize> = Vec::new();

        // Title column width, including the synthetic "Value scan" row.
        let checks = report.checks();
        let width = checks
            .iter()
            .map(|r| r.title.len())
            .chain(std::iter::once("Value scan".len()))
            .max()
            .unwrap_or(0);

        let mut lines: Vec<Line> = vec![Line::from(sty(
            format!(
                "{} file(s) · {} tensors · {} params",
                report.n_files,
                report.n_tensors,
                crate::utils::format_parameters(report.params)
            ),
            Style::default().fg(palette::DIM),
        ))];

        for r in &checks {
            let (mark, mc) = match r.status() {
                Status::Pass => ("✓", palette::SUCCESS),
                Status::Warn => ("⚠", palette::WARN),
                Status::Fail => ("✗", palette::ERROR),
                Status::Na => ("⊘", palette::DIM),
            };
            // Every applicable row shows its `note` — what passing verifies — as an
            // inline explanation (the TUI has no hover, so this is its equivalent of
            // the web's per-check tooltip); a warn/fail row adds the finding count.
            let mut trailer_text = match r.status() {
                Status::Pass => format!("— {}", r.summary().unwrap_or(r.note)),
                Status::Na => "— n/a for this checkpoint".to_string(),
                _ => format!("— {}  ({})", r.note, count_phrase(r.errors(), r.warnings())),
            };
            // The value scan carries its wall-clock time (like the CLI bar).
            if let Some(d) = r.elapsed() {
                trailer_text.push_str(&format!("  ({})", fmt_elapsed(d)));
            }
            let trailer = sty(trailer_text, Style::default().fg(palette::DIM));
            lines.push(check_row(
                sty(mark.into(), Style::default().fg(mc)),
                r.title,
                width,
                trailer,
                bg,
            ));
            // The per-finding detail is folded away by default (like the stats
            // popup's per-shard list). Under each check with findings sits a
            // toggle aligned with the check title; `f` (or a click on it, either
            // state) reveals the full list. The `f` hint lives in the footer, with
            // the other keys, so it stays put and consistently styled.
            if !r.findings().is_empty() {
                let arrow = if expanded { "▾" } else { "▸" };
                fold_lines.push(lines.len());
                lines.push(Line::from(vec![
                    sty(
                        format!("    {arrow} "),
                        Style::default().fg(palette::ACCENT),
                    ),
                    sty(
                        format!(
                            "{} finding{}",
                            r.findings().len(),
                            if r.findings().len() == 1 { "" } else { "s" }
                        ),
                        Style::default().fg(palette::DIM),
                    ),
                ]));
                if expanded {
                    for f in r.findings() {
                        let (fm, fc) = match f.severity {
                            Severity::Error => ("✗", palette::ERROR),
                            Severity::Warning => ("⚠", palette::WARN),
                        };
                        let mut spans = vec![
                            sty("      ".into(), Style::default()),
                            sty(fm.into(), Style::default().fg(fc)),
                            sty(" ".into(), Style::default()),
                        ];
                        if let Some(subj) = &f.subject {
                            spans.push(sty(
                                format!("{subj}  "),
                                Style::default().add_modifier(Modifier::BOLD),
                            ));
                        }
                        spans.push(sty(f.message.clone(), Style::default()));
                        lines.push(Line::from(spans));
                    }
                }
            }
        }

        // The value tier isn't in `results` until it runs: show a spinner while
        // scanning, else a "not run" hint.
        if !report.values {
            let (mark, mc, trailer) = match state {
                // The count lives in the footer bar — don't repeat it here.
                CheckPopup::Scanning { frame, .. } => (
                    CHECK_SPINNER[frame % CHECK_SPINNER.len()].to_string(),
                    palette::ACCENT,
                    sty("— scanning…".into(), Style::default().fg(palette::DIM)),
                ),
                // Only suggest `v` when the scan is actually available — it isn't
                // for a remote checkpoint (data stays on the host).
                CheckPopup::Idle { can_scan, .. } => (
                    "·".into(),
                    palette::DIM,
                    sty(
                        if can_scan {
                            "— not run (press v)"
                        } else {
                            "— not run"
                        }
                        .into(),
                        Style::default().fg(palette::DIM),
                    ),
                ),
            };
            lines.push(check_row(
                sty(mark, Style::default().fg(mc)),
                "Value scan",
                width,
                trailer,
                bg,
            ));
        }

        let (e, w) = (report.errors(), report.warnings());
        let verdict = if e > 0 {
            sty(
                format!("FAIL — {}", count_phrase(e, w)),
                Style::default().fg(palette::ERROR),
            )
        } else if w > 0 {
            sty(
                format!("OK with warnings — {}", count_phrase(0, w)),
                Style::default().fg(palette::WARN),
            )
        } else if report.values {
            sty(
                "OK — no issues found".into(),
                Style::default().fg(palette::SUCCESS),
            )
        } else {
            sty(
                "OK — no metadata issues found".into(),
                Style::default().fg(palette::SUCCESS),
            )
        };
        lines.push(Line::from(vec![
            sty("  ".into(), Style::default()),
            verdict,
        ]));

        // Every per-check findings toggle is clickable (→ `f`), so a click folds
        // or unfolds in either state.
        let clickable: Vec<(usize, KeyEvent)> = fold_lines
            .iter()
            .map(|&i| (i, KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE)))
            .collect();
        // The `f` fold hint goes in the footer (only when there are findings).
        let fold = (!fold_lines.is_empty()).then_some(expanded);
        // The key-hint footer stays pinned while the body (checks + findings)
        // scrolls, so a report with many findings never overflows the popup.
        render_scroll_popup(
            frame,
            "Health check",
            &lines,
            check_footer_line(&state, fold, bg),
            scroll,
            &clickable,
        )
    }
}

/// The state of the health-check popup ([`UI::render_check_report`]).
#[derive(Clone, Copy)]
pub(crate) enum CheckPopup {
    /// Showing the report. `copied` briefly flashes what was just copied
    /// (`"command"` / `"report"` / `"screen"`); `can_scan` offers the `v` value
    /// scan (off for a remote source or once it has run).
    Idle {
        copied: Option<&'static str>,
        can_scan: bool,
    },
    /// A value scan is running: `done`/`total` tensors, `frame` animates the row
    /// spinner and drives the footer bar.
    Scanning {
        done: usize,
        total: usize,
        frame: usize,
    },
}

/// Braille spinner frames for the in-progress "Value scan" row.
const CHECK_SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// One check row: `  <mark> <title padded>  <trailer>`, all on the panel `bg`.
fn check_row(
    mark: Span<'static>,
    title: &str,
    width: usize,
    trailer: Span<'static>,
    bg: Color,
) -> Line<'static> {
    Line::from(vec![
        Span::styled("  ", Style::default().bg(bg)),
        mark,
        Span::styled(format!(" {title:<width$}  "), Style::default().bg(bg)),
        trailer,
    ])
}

/// The popup footer: the value-scan bar while scanning, a copy confirmation right
/// after `y`, or the key hints — with the key glyphs bold/accented (not dimmed)
/// so it's clear they're actionable.
fn check_footer_line(state: &CheckPopup, fold: Option<bool>, bg: Color) -> Line<'static> {
    let key = |k: &str| {
        Span::styled(
            k.to_string(),
            Style::default()
                .fg(palette::KEY)
                .add_modifier(Modifier::BOLD)
                .bg(bg),
        )
    };
    // Descriptions in the default foreground, only " · "/"cancel"-style connective
    // text dimmed — matching the tree view's footer style.
    let dim = |s: &str| Span::styled(s.to_string(), Style::default().fg(palette::DIM).bg(bg));
    let label = |s: &str| Span::styled(s.to_string(), Style::default().bg(bg));
    match *state {
        CheckPopup::Scanning { done, total, .. } => {
            const W: usize = 18;
            let filled = if total == 0 {
                0
            } else {
                (((done as f64 / total as f64) * W as f64).round() as usize).min(W)
            };
            Line::from(vec![
                Span::styled(
                    "━".repeat(filled),
                    Style::default().fg(palette::ACCENT).bg(bg),
                ),
                Span::styled(
                    "━".repeat(W - filled),
                    Style::default().fg(palette::DIM).bg(bg),
                ),
                Span::styled(format!("  {done}/{total}   "), Style::default().bg(bg)),
                key("Esc"),
                label(" cancel"),
            ])
        }
        CheckPopup::Idle {
            copied: Some(what), ..
        } => Line::from(Span::styled(
            format!("✓ copied {what} to the clipboard"),
            Style::default().fg(palette::SUCCESS).bg(bg),
        )),
        CheckPopup::Idle { can_scan, .. } => {
            let mut items: Vec<(&str, &str)> = Vec::new();
            // The findings-fold key, when there are findings to fold — a footer
            // hint (not inline text) so it matches the other keys and stays visible
            // whether folded or expanded.
            match fold {
                Some(true) => items.push(("f", " fold findings")),
                Some(false) => items.push(("f", " expand findings")),
                None => {}
            }
            if can_scan {
                items.push(("v", " value scan"));
            }
            items.push(("c", " copy screen"));
            items.push(("r", " copy report"));
            items.push(("y", " copy command"));
            items.push(("Esc", " dismiss"));
            let mut spans = Vec::new();
            for (i, (k, lbl)) in items.iter().enumerate() {
                if i > 0 {
                    spans.push(dim(" · "));
                }
                spans.push(key(k));
                spans.push(label(lbl));
            }
            Line::from(spans)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The styled check-popup fold render (`render_check_report`) is exercised
    // here in the bin, since the frontend-free core can't depend on ratatui. The
    // core `check` module keeps the data-level checks.
    #[test]
    fn check_popup_folds_findings_like_the_stats_popup() {
        use crate::check::{CheckReport, CheckResult, CheckpointFormat, Finding, StorageCheck};
        let findings: Vec<Finding> = (0..250)
            .map(|i| Finding::error(Some(format!("tensor-{i:04}")), "bad byte range".into()))
            .collect();
        let report = CheckReport {
            label: "x".into(),
            n_files: 1,
            n_tensors: 250,
            params: 1,
            values: false,
            format: CheckpointFormat::Safetensors,
            storage: StorageCheck::ByteRanges(CheckResult::done(
                "byte_ranges",
                "Byte-range integrity",
                "n",
                findings,
            )),
            results: vec![],
        };
        let render = |expanded| {
            crate::tui::headless_render(200, 400, |f| {
                UI::render_check_report(
                    f,
                    &report,
                    CheckPopup::Idle {
                        copied: None,
                        can_scan: false,
                    },
                    0,
                    expanded,
                );
            })
            .unwrap()
        };

        let folded = render(false);
        assert!(
            folded.contains("250 findings"),
            "fold toggle shows the count"
        );
        assert!(
            folded.contains("expand findings"),
            "footer hints `f` to expand:\n{folded}"
        );
        assert!(
            folded.contains("    ▸ 250 findings"),
            "toggle indented under the check title:\n{folded}"
        );
        assert!(
            !folded.contains("tensor-0000"),
            "findings hidden when folded"
        );

        let expanded = render(true);
        assert!(expanded.contains("tensor-0000"), "first finding shown");
        assert!(
            expanded.contains("tensor-0249"),
            "last finding shown too (no cap)"
        );
        assert!(
            expanded.contains("fold findings"),
            "footer offers folding back:\n{expanded}"
        );
    }
}
