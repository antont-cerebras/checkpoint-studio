//! The checkpoint statistics screen and the pop-up that shares its body lines.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

use crate::utils::{format_parameters, format_size};

use super::UI;
use super::hints::{chip_regions, close_button, stats_hint_lines};
use super::palette;
use super::scroll::VScrollbar;
use super::theme::dim_span;

impl UI {
    /// The overall-checkpoint stats popup (the `s` key on the tree). Returns the
    /// max scroll offset, like [`Self::render_check_report`], so the caller can
    /// clamp its scroll state to what actually fit.
    /// Build the checkpoint-stats report as styled body lines, plus the body-line
    /// index of the on-disk per-shard fold toggle (for click hit-testing) when a
    /// breakdown is present. Shared by the full-screen stats mode and headless
    /// `--stats`, so both render identically.
    fn stats_body_lines(
        s: &crate::stats::CheckpointStats,
        shards_expanded: bool,
    ) -> (Vec<Line<'static>>, Option<usize>) {
        let sty = |t: String, style: Style| Span::styled(t, style);
        let plain = |t: String| sty(t, Style::default());
        let dim = |t: String| sty(t, Style::default().fg(palette::DIM));

        // A section header, then indented "label   value" rows. Labels align to a
        // fixed column so the values line up down the popup.
        const LW: usize = 12;
        let header = |t: &str| {
            Line::from(sty(
                t.to_string(),
                Style::default()
                    .fg(palette::ACCENT)
                    .add_modifier(Modifier::BOLD),
            ))
        };
        // A glyphed section header like the tree — "▦ Tensors  ×116175" — with the
        // glyph + title in accent, the `count` plain (not dim, so it stands out),
        // and a dim `qualifier` (e.g. " per layer", " safetensors").
        let section = |glyph: &str, title: &str, count: String, qualifier: &str| {
            let accent = Style::default().fg(palette::ACCENT);
            let mut spans = vec![
                sty(format!("{glyph} "), accent),
                sty(title.to_string(), accent.add_modifier(Modifier::BOLD)),
            ];
            if !count.is_empty() {
                spans.push(plain(format!("  {count}")));
            }
            if !qualifier.is_empty() {
                spans.push(dim(qualifier.to_string()));
            }
            Line::from(spans)
        };
        // Pad the label to `LW`, then a guaranteed separator space — so a label
        // that exactly fills `LW` (e.g. "Architecture") still has a gap before it.
        let row = |label: &str, mut value: Vec<Span<'static>>| {
            let mut spans = vec![plain(format!("  {label:<LW$} "))];
            spans.append(&mut value);
            Line::from(spans)
        };
        // "<size> each · <size> total", the shared shape of the layer/expert rows.
        let each_total = |each: usize, total: usize, fmt: fn(usize) -> String| {
            vec![
                plain(fmt(each)),
                dim(" each · ".into()),
                plain(fmt(total)),
                dim(" total".into()),
            ]
        };

        let mut lines: Vec<Line> = Vec::new();
        // Body-line index of the per-shard fold toggle, once emitted (for click
        // hit-testing).
        let mut fold_line: Option<usize> = None;

        // ── Overview ──────────────────────────────────────────────────────────
        lines.push(header("Overview"));
        if let Some(mt) = &s.model_type {
            lines.push(row("Architecture", vec![plain(mt.clone())]));
        }
        lines.push(row("Parameters", vec![plain(format_parameters(s.params))]));
        // On-disk vs logical, with a compression ratio when they differ.
        let size_value = if s.compressed && s.disk_bytes > 0 {
            vec![
                plain(format_size(s.disk_bytes)),
                dim(" on disk · ".into()),
                plain(format_size(s.logical_bytes)),
                dim(format!(
                    " logical ({:.2}× smaller)",
                    s.logical_bytes as f64 / s.disk_bytes as f64
                )),
            ]
        } else {
            vec![plain(format_size(s.logical_bytes))]
        };
        lines.push(row("Size", size_value));

        // ── Files (per-shard logical size) ────────────────────────────────────
        lines.push(Line::from(sty(String::new(), Style::default())));
        let kind = if s.files.noun.starts_with("safetensors") {
            " safetensors"
        } else {
            ""
        };
        lines.push(section(
            crate::stats::GLYPH_FILES,
            "Files",
            format!("×{}", s.files.count),
            kind,
        ));
        // A `size  name` value, size padded and the name dimmed — shared by the
        // per-file and per-tensor largest/smallest rows so they read alike.
        let named = |n: &crate::stats::NamedSize| {
            vec![
                plain(format!("{:<9} ", format_size(n.bytes))),
                dim(n.name.clone()),
            ]
        };
        if let Some(l) = &s.files.largest {
            lines.push(row("Largest", named(l)));
        }
        if let Some(sm) = &s.files.smallest {
            lines.push(row("Smallest", named(sm)));
        }
        lines.push(row("Average", vec![plain(format_size(s.files.mean))]));
        lines.push(row("Median", vec![plain(format_size(s.files.median))]));

        // ── Tensors (count + size) ────────────────────────────────────────────
        lines.push(Line::from(sty(String::new(), Style::default())));
        lines.push(section(
            crate::stats::GLYPH_TENSORS,
            "Tensors",
            format!("×{}", s.n_tensors),
            "",
        ));
        if let Some(l) = &s.largest {
            lines.push(row("Largest", named(l)));
        }
        if let Some(sm) = &s.smallest {
            lines.push(row("Smallest", named(sm)));
        }
        lines.push(row("Average", vec![plain(format_size(s.mean_bytes))]));
        lines.push(row("Median", vec![plain(format_size(s.median_bytes))]));

        // ── Layers ───────────────────────────────────────────────────────────
        if let Some(l) = &s.layers {
            lines.push(Line::from(sty(String::new(), Style::default())));
            lines.push(section(
                crate::stats::GLYPH_LAYERS,
                "Layers",
                format!("×{}", l.count),
                "",
            ));
            lines.push(row(
                "Params",
                each_total(l.params_each(), l.params, format_parameters),
            ));
            lines.push(row(
                "Size",
                each_total(l.bytes_each(), l.bytes, format_size),
            ));
        }

        // ── Experts (MoE) ─────────────────────────────────────────────────────
        if let Some(x) = &s.experts {
            lines.push(Line::from(sty(String::new(), Style::default())));
            let (count, qualifier) = match x.layout.per_layer() {
                Some(pl) => (format!("×{pl}"), " per layer"),
                None => (String::new(), ""),
            };
            lines.push(section(
                crate::stats::GLYPH_EXPERTS,
                "Experts",
                count,
                qualifier,
            ));
            let mut storage = x.layout.label().to_string();
            if x.gate_up_fused {
                storage.push_str(" · gate+up fused");
            }
            lines.push(row("Storage", vec![plain(storage)]));
            // Per-expert averages are only meaningful once we know the count.
            if x.layout.per_layer().is_some() {
                lines.push(row(
                    "Params",
                    each_total(x.params_each(), x.params, format_parameters),
                ));
                lines.push(row(
                    "Size",
                    each_total(x.bytes_each(), x.bytes, format_size),
                ));
            }
            // Per-projection split (down/gate/up), each with its per-layer footprint.
            for c in &x.by_category {
                lines.push(row(
                    &c.name,
                    each_total(c.bytes / x.layers.max(1), c.bytes, format_size),
                ));
            }
        }

        // ── Per-layer profile ───────────────────────────────────────────────────
        // Per metric, a min-anchored sparkline when it varies across the stack, else
        // a plain "uniform" note (a flat sparkline says nothing). Plus one
        // 100%-stacked composition bar (attention / ffn-experts / other) — its
        // segments are distinct glyphs *and* colours (colour is stripped headless).
        const LBL: usize = 13;
        if let Some(pl) = &s.per_layer {
            lines.push(Line::from(sty(String::new(), Style::default())));
            lines.push(header("Per-layer profile"));
            let metric = |label: &str, vals: &[usize], fmt: fn(usize) -> String| -> Line<'static> {
                let (lo, hi) = (
                    vals.iter().copied().min().unwrap_or(0),
                    vals.iter().copied().max().unwrap_or(0),
                );
                if lo == hi {
                    Line::from(vec![
                        plain(format!("  {label:<LBL$}  ")),
                        dim(format!("uniform · {} each", fmt(lo))),
                    ])
                } else {
                    let glyphs = crate::stats::spark_string(vals, crate::stats::GRAPH_W);
                    Line::from(vec![
                        plain(format!("  {label:<LBL$}  ")),
                        sty(glyphs, Style::default().fg(palette::ACCENT)),
                        dim(format!("  {}–{}", fmt(lo), fmt(hi))),
                    ])
                }
            };
            // A blank line between each graph so they read as separate charts.
            let blank = || Line::from(sty(String::new(), Style::default()));
            lines.push(metric("Size/layer", &pl.bytes_series(), format_size));
            lines.push(blank());
            lines.push(metric(
                "Params/layer",
                &pl.params_series(),
                format_parameters,
            ));
            lines.push(blank());
            lines.push(metric("Tensors/layer", &pl.tensor_series(), |n| {
                n.to_string()
            }));
            lines.push(blank());

            // Composition: a swatch + % key on the "Composition" line, and the
            // 100%-stacked bar just below it (indented under, so the pure-glyph bar
            // isn't mistaken for part of the key). attn → accent, ffn/experts →
            // dtype amber, other → slate.
            let comp = pl.composition_totals();
            let total: usize = comp.iter().sum();
            if total > 0 {
                let colors = [palette::ACCENT, palette::DTYPE, palette::META];
                let names = ["attention", "ffn/experts", "other"];
                let pct = |x: usize| -> String {
                    let p = (x * 100 + total / 2) / total;
                    if x > 0 && p == 0 {
                        "<1%".into()
                    } else {
                        format!("{p}%")
                    }
                };
                // Bar and key on ONE line (bar first, then the swatch/percent key) —
                // a separate key row directly above the bar reads as one stacked
                // block. Any non-zero share shows at least a one-cell sliver.
                let cells = crate::stats::composition_cells(comp, crate::stats::BAR_W);
                let mut line = vec![plain(format!("  {:<LBL$}  ", "Composition"))];
                for (i, &n) in cells.iter().enumerate() {
                    if n > 0 {
                        line.push(sty(
                            crate::stats::SHADES[i].to_string().repeat(n),
                            Style::default().fg(colors[i]),
                        ));
                    }
                }
                line.push(plain("   ".into()));
                for i in 0..3 {
                    if i > 0 {
                        line.push(dim(" · ".into()));
                    }
                    line.push(sty(
                        format!("{} ", crate::stats::SHADES[i]),
                        Style::default().fg(colors[i]),
                    ));
                    line.push(plain(format!("{} {}", names[i], pct(comp[i]))));
                }
                lines.push(Line::from(line));
            }
        }

        // ── dtype mix ─────────────────────────────────────────────────────────
        if !s.dtypes.is_empty() {
            lines.push(Line::from(sty(String::new(), Style::default())));
            lines.push(header("By dtype"));
            let dw = s.dtypes.iter().map(|d| d.dtype.len()).max().unwrap_or(0);
            for d in &s.dtypes {
                lines.push(Line::from(vec![
                    sty(
                        format!("  {:<dw$}  ", d.dtype),
                        Style::default().fg(palette::DTYPE),
                    ),
                    plain(format!("{:>7}", format_size(d.bytes))),
                    plain(format!("  {}", d.count)),
                    dim(format!(" tensor{}", if d.count == 1 { "" } else { "s" })),
                ]));
            }
        }

        // ── S3 objects (an `s3://` cstorch source) ─────────────────────────────
        // Summary + a per-object breakdown folded away by default. Shares the one
        // `fold_line` with the on-disk section — an s3 source has no local
        // filesystem, so the two never both appear.
        if let Some(s3) = s.s3().filter(|x| !x.objects.is_empty()) {
            lines.push(Line::from(sty(String::new(), Style::default())));
            lines.push(section(
                crate::stats::GLYPH_S3,
                "S3 objects",
                format!("×{}", s3.count()),
                "",
            ));
            lines.push(row(
                "Total",
                vec![plain(format_size(s3.total_bytes() as usize))],
            ));
            lines.push(row(
                "Checksums",
                vec![plain(crate::stats::s3_checksums_phrase(s3))],
            ));
            lines.push(row(
                "ETags",
                vec![plain(format!("{} of {} present", s3.etags(), s3.count()))],
            ));
            lines.push(row("Tags", vec![plain(crate::stats::s3_tags_phrase(s3))]));
            if let Some(m) = crate::stats::s3_modified_phrase(s3) {
                lines.push(row("Modified", vec![plain(m)]));
            }
            let umeta = s3.with_user_meta();
            if umeta > 0 {
                lines.push(row(
                    "User meta",
                    vec![plain(format!(
                        "{umeta} object{}",
                        if umeta == 1 { "" } else { "s" }
                    ))],
                ));
            }
            // A click on this line or `f` toggles the per-object list.
            let arrow = if shards_expanded { "▾" } else { "▸" };
            fold_line = Some(lines.len());
            lines.push(Line::from(vec![
                sty(format!("  {arrow} "), Style::default().fg(palette::ACCENT)),
                plain("per-object breakdown".into()),
                dim(format!("  ({})", s3.count())),
            ]));
            if shards_expanded {
                let kw = s3.objects.iter().map(|o| o.key.len()).max().unwrap_or(0);
                for o in &s3.objects {
                    lines.push(Line::from(vec![
                        sty(
                            format!("    {:<kw$}  ", o.key),
                            Style::default().fg(palette::META),
                        ),
                        plain(crate::stats::s3_object_detail(o)),
                    ]));
                }
            }
            for w in &s3.warnings {
                lines.push(Line::from(dim(format!("  ⚠ {w}"))));
            }
        }

        // ── On disk (filesystem allocation) ────────────────────────────────────
        if let Some(d) = s.disk() {
            lines.push(Line::from(sty(String::new(), Style::default())));
            lines.push(header("On disk (filesystem)"));
            lines.push(row(
                "Allocated",
                vec![
                    plain(format_size(d.total_allocated as usize)),
                    dim(format!(
                        "  ({} apparent, {})",
                        format_size(d.total_apparent as usize),
                        crate::stats::ratio_phrase(d.total_apparent, d.total_allocated),
                    )),
                ],
            ));
            // The per-shard breakdown is folded away by default (a many-shard
            // model is otherwise a wall of rows); a click on this line or `f`
            // toggles it. Only shards the filesystem actually shrank are listed.
            if d.shards.len() > 1 {
                let savers: Vec<&crate::stats::ShardDisk> = d
                    .shards
                    .iter()
                    .filter(|sh| crate::stats::has_saving(sh.apparent, sh.allocated))
                    .collect();
                let arrow = if shards_expanded { "▾" } else { "▸" };
                // The `f` hint lives in the footer with the other keys; the toggle
                // itself just labels the breakdown (and, folded, the saver count).
                let tail = if shards_expanded {
                    String::new()
                } else {
                    format!("  ({} of {} smaller)", savers.len(), d.shards.len())
                };
                fold_line = Some(lines.len());
                lines.push(Line::from(vec![
                    sty(format!("  {arrow} "), Style::default().fg(palette::ACCENT)),
                    plain("per-shard breakdown".into()),
                    dim(tail),
                ]));
                if shards_expanded {
                    // Unfolding shows *every* shard (savers and not) — the folded
                    // summary already gave the "N of M smaller" headline, so the
                    // expanded view is the full breakdown, not a filtered one.
                    let nw = d.shards.iter().map(|sh| sh.name.len()).max().unwrap_or(0);
                    for sh in &d.shards {
                        lines.push(Line::from(vec![
                            sty(
                                format!("    {:<nw$}  ", sh.name),
                                Style::default().fg(palette::META),
                            ),
                            plain(format!("{:>9}", format_size(sh.apparent as usize))),
                            dim(" → ".into()),
                            plain(format!("{:>9}", format_size(sh.allocated as usize))),
                            dim(format!(
                                "  ({})",
                                crate::stats::ratio_phrase(sh.apparent, sh.allocated)
                            )),
                        ]));
                    }
                }
            }
        }

        (lines, fold_line)
    }

    /// Render the full-screen checkpoint-stats view: a title header, the scrollable
    /// report body (with an overflow indicator), and the bottom-pinned command
    /// footer. Returns the maximum valid scroll offset (so the mode can clamp its
    /// own) and the clickable regions — the on-disk fold toggle, the footer chips,
    /// and the top-right `[×]`. Used by the interactive [`StatsMode`] and headless
    /// `--stats`, so the two stay byte-identical.
    pub fn render_stats_frame(
        frame: &mut Frame,
        s: &crate::stats::CheckpointStats,
        scroll: usize,
        shards_expanded: bool,
    ) -> (usize, Vec<(Rect, KeyEvent)>, Option<VScrollbar>) {
        let area = frame.area();
        let (width, height) = (area.width, area.height);

        let (body, fold_line) = Self::stats_body_lines(s, shards_expanded);
        let (footer, chips) = stats_hint_lines(fold_line.is_some(), shards_expanded, width);
        let footer_len = footer.len() as u16;

        // Header: a title and a full-width rule — the shared view-header look.
        let header = vec![
            Line::from(Span::styled(
                " Checkpoint stats",
                Style::default()
                    .fg(palette::ACCENT)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                "─".repeat(width as usize),
                Style::default().fg(palette::DIM),
            )),
        ];
        let header_len = header.len() as u16;
        Paragraph::new(header).render(crate::ui::fit_rows(area, 0, header_len), frame.buffer_mut());

        // Footer pinned above the bottom row, which the caller reserves for the
        // access badge (drawn after this, like the detail / data views).
        let footer_top = height.saturating_sub(footer_len + 1);
        Paragraph::new(footer).render(
            crate::ui::fit_rows(area, footer_top, footer_len),
            frame.buffer_mut(),
        );

        // Body window between header and footer; when it overflows, reserve one
        // row just above the footer for the scroll indicator.
        let avail = footer_top.saturating_sub(header_len) as usize;
        let total = body.len();
        let overflow = total > avail;
        let visible = if overflow {
            avail.saturating_sub(1)
        } else {
            avail
        };
        let max_scroll = total.saturating_sub(visible);
        let scroll = scroll.min(max_scroll);
        // Clone only the visible window (not the whole body) so scrolling a large
        // report stays O(screen), not O(content).
        let window: Vec<Line> = body.iter().skip(scroll).take(visible).cloned().collect();
        Paragraph::new(window).render(
            crate::ui::fit_rows(area, header_len, visible as u16),
            frame.buffer_mut(),
        );
        if overflow {
            let indicator = format!(
                "↑↓ PgUp/PgDn scroll · {}–{} of {total}",
                scroll + 1,
                scroll + visible
            );
            Paragraph::new(Line::from(dim_span(indicator))).render(
                crate::ui::fit_rows(area, header_len + visible as u16, 1),
                frame.buffer_mut(),
            );
        }

        // Clickable regions: the fold toggle (when visible in the scrolled window),
        // the footer chips, and the top-right `[×]` (→ step back, like `⌫`).
        let mut regions: Vec<(Rect, KeyEvent)> = Vec::new();
        if let Some(i) = fold_line
            && i >= scroll
            && i < scroll + visible
        {
            regions.push((
                Rect {
                    x: 0,
                    y: header_len + (i - scroll) as u16,
                    width,
                    height: 1,
                },
                KeyEvent::new(KeyCode::Char('f'), KeyModifiers::NONE),
            ));
        }
        regions.extend(chip_regions(&chips, footer_top));
        regions.extend(close_button(
            frame,
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
        ));
        // The scroll bar (drawn by the engine): over the body region's last column.
        let vscroll = VScrollbar::for_body(
            Rect {
                x: 0,
                y: header_len,
                width,
                height: visible as u16,
            },
            total,
            scroll,
        );
        (max_scroll, regions, vscroll)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{Layout, Storage, TensorInfo};

    #[test]
    fn stats_popup_renders_s3_section() {
        use crate::stats::{CheckpointStats, S3ObjectStat, S3Stats};
        use crate::tree::{Layout, Storage, TensorInfo};
        let tensors = vec![TensorInfo {
            name: "w".into(),
            dtype: "F32".into(),
            shape: vec![4],
            size_bytes: 16,
            num_elements: 4,
            storage: Storage::Unknown,
            source_path: "s3://bucket/ckpt".into(),
            layout: Layout::None,
        }];
        let s3 = S3Stats {
            objects: vec![
                S3ObjectStat {
                    key: "model-00000.safetensors".into(),
                    size: 100,
                    etag: "abcdef0123456789".into(),
                    checksum: Some(crate::remote::S3Checksum {
                        algorithm: "SHA256".into(),
                        value: "9f8e7d6c5b4a3210".into(),
                    }),
                    last_modified: "2026-07-19T00:00:00+00:00".into(),
                    tags: Some(1),
                    user_meta: 1,
                },
                S3ObjectStat {
                    key: "model-00001.safetensors".into(),
                    size: 200,
                    etag: "112233445566".into(),
                    checksum: Some(crate::remote::S3Checksum {
                        algorithm: "SHA256".into(),
                        value: "aabbccddeeff".into(),
                    }),
                    last_modified: "2026-07-18T00:00:00+00:00".into(),
                    tags: Some(0),
                    user_meta: 0,
                },
            ],
            warnings: Vec::new(),
        };
        let stats = CheckpointStats::compute(&tensors, None, None).with_s3(Some(s3));

        // Expanded: the S3 summary + every object row (abbreviated etag/checksum).
        let expanded = crate::tui::headless_render(100, 50, |f| {
            UI::render_stats_frame(f, &stats, 0, true);
        })
        .unwrap();
        assert!(expanded.contains("S3 objects"), "{expanded}");
        assert!(expanded.contains("2 with SHA256"), "{expanded}");
        assert!(expanded.contains("2026-07-18 – 2026-07-19"), "{expanded}");
        assert!(expanded.contains("model-00000.safetensors"), "{expanded}");
        assert!(expanded.contains("sha256 9f8e7d6c5b4a3210"), "{expanded}");

        // Folded (default): the per-object list collapses to a single toggle line.
        let folded = crate::tui::headless_render(100, 50, |f| {
            UI::render_stats_frame(f, &stats, 0, false);
        })
        .unwrap();
        assert!(folded.contains("per-object breakdown"), "{folded}");
        assert!(!folded.contains("model-00000.safetensors"), "{folded}");
    }

    #[test]
    fn stats_popup_renders_on_disk_section() {
        use crate::stats::{CheckpointStats, DiskUsage, ShardDisk};
        let tensors = vec![TensorInfo {
            name: "w".into(),
            dtype: "F32".into(),
            shape: vec![4],
            size_bytes: 16,
            num_elements: 4,
            storage: Storage::Unknown,
            source_path: "m.safetensors".into(),
            layout: Layout::None,
        }];
        // One shard squeezed 4× among two the filesystem left alone.
        let disk = DiskUsage::from_shards(vec![
            ShardDisk {
                name: "shard-saver.safetensors".into(),
                apparent: 4 << 20,
                allocated: 1 << 20,
            },
            ShardDisk {
                name: "shard-plain.safetensors".into(),
                apparent: 4 << 20,
                allocated: 4 << 20,
            },
        ]);
        let stats = CheckpointStats::compute(&tensors, None, disk);

        // Expanded: *every* shard is listed — the saver and the untouched one.
        let expanded = crate::tui::headless_render(100, 50, |f| {
            UI::render_stats_frame(f, &stats, 0, true);
        })
        .unwrap();
        assert!(expanded.contains("On disk (filesystem)"), "{expanded}");
        assert!(expanded.contains("Allocated"), "{expanded}");
        assert!(expanded.contains("shard-saver.safetensors"), "{expanded}");
        assert!(expanded.contains("4.00×"), "{expanded}");
        assert!(expanded.contains("shard-plain.safetensors"), "{expanded}");
        assert!(
            !expanded.contains("shard with no filesystem saving"),
            "{expanded}"
        );

        // Folded (default): the shard list collapses to a single toggle line.
        let folded = crate::tui::headless_render(100, 50, |f| {
            UI::render_stats_frame(f, &stats, 0, false);
        })
        .unwrap();
        assert!(folded.contains("per-shard breakdown"), "{folded}");
        assert!(!folded.contains("shard-saver.safetensors"), "{folded}");
    }

    #[test]
    fn render_stats_frame_draws_per_layer_graphs() {
        use crate::stats::CheckpointStats;
        use crate::tree::{Layout, Storage, TensorInfo};
        let ti = |name: &str, elems: usize| TensorInfo {
            name: name.into(),
            dtype: "F16".into(),
            shape: vec![elems],
            size_bytes: elems * 2,
            num_elements: elems,
            storage: Storage::Unknown,
            source_path: "m.safetensors".into(),
            layout: Layout::None,
        };
        // 6 layers, attention growing with depth (so the size sparkline ramps),
        // an expert (ffn) and a norm (other) each layer for the composition chart.
        let mut tensors = Vec::new();
        for l in 0..6 {
            tensors.push(ti(
                &format!("model.layers.{l}.self_attn.q_proj.weight"),
                4 + l * 2,
            ));
            tensors.push(ti(
                &format!("model.layers.{l}.mlp.experts.0.down_proj.weight"),
                20,
            ));
            tensors.push(ti(&format!("model.layers.{l}.input_layernorm.weight"), 2));
        }
        let stats = CheckpointStats::compute(&tensors, None, None);
        // Render tall enough that the whole report fits (no scroll fold).
        let out = crate::tui::headless_render(120, 60, |f| {
            UI::render_stats_frame(f, &stats, 0, false);
        })
        .unwrap();
        assert!(out.contains("Per-layer profile"), "{out}");
        assert!(out.contains("Size/layer"), "{out}");
        // The composition legend + all three stacked bands (glyphs, colour stripped).
        assert!(
            out.contains("attention") && out.contains("ffn/experts") && out.contains("other"),
            "{out}"
        );
        assert!(
            out.contains('▓') && out.contains('░'),
            "composition bands:\n{out}"
        );
        // The size sparkline ramps with depth → both the lowest and highest glyphs.
        assert!(out.contains('▁') && out.contains('█'), "sparkline:\n{out}");
    }

    // The stats mode (which previously scrolled with no bar) reports a scroll bar to
    // the engine when its report overflows, and none when it fits — the engine draws
    // it uniformly, so this is what makes "a scrolling mode with no bar" impossible.
    #[test]
    fn render_stats_frame_reports_a_scrollbar_when_overflowing() {
        use crate::stats::CheckpointStats;
        use crate::tree::{Layout, Storage, TensorInfo};
        let ti = |name: &str| TensorInfo {
            name: name.into(),
            dtype: "F16".into(),
            shape: vec![4],
            size_bytes: 8,
            num_elements: 4,
            storage: Storage::Unknown,
            source_path: "m.safetensors".into(),
            layout: Layout::None,
        };
        let tensors: Vec<TensorInfo> = (0..40)
            .map(|i| ti(&format!("model.layers.{i}.mlp.down_proj.weight")))
            .collect();
        let stats = CheckpointStats::compute(&tensors, None, None);
        // A short frame → the report overflows → a bar is reported.
        let mut overflow = false;
        crate::tui::headless_render(120, 16, |f| {
            overflow = UI::render_stats_frame(f, &stats, 0, false).2.is_some();
        })
        .unwrap();
        assert!(overflow, "a short stats frame must report a scroll bar");
        // A tall frame → it all fits → no bar.
        let mut fits = true;
        crate::tui::headless_render(120, 80, |f| {
            fits = UI::render_stats_frame(f, &stats, 0, false).2.is_some();
        })
        .unwrap();
        assert!(!fits, "a tall stats frame fits → no scroll bar");
    }
}
