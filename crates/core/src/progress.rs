//! Live progress bars for remote reads: one colored spinner + elapsed timer per
//! read, settling to `✓` (green) or `✗` (red). Animated on a background thread —
//! off the main thread doing the blocking SSH reads, touching only shared atomics,
//! so it never races the sessions — and suppressed when stderr isn't a terminal
//! (escape codes never pollute a pipe/log). Callers must do any password prompt
//! *before* starting the bars.

use std::io::{IsTerminal, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Shared, thread-safe progress for the checkpoint-structure load: a count of
/// completed units (shards / files) the reader bumps as it goes, and a total
/// that starts at 0 and is set once known (e.g. after a remote directory is
/// listed). The loading screen polls [`LoadProgress::snapshot`] to draw a bar
/// instead of a bare spinner, so a slow SSH read visibly makes progress.
#[derive(Default)]
pub struct LoadProgress {
    done: AtomicUsize,
    total: AtomicUsize,
    /// What the count counts, for the bar's label (0 = unlabelled).
    unit: AtomicU8,
    /// A trailing activity note (0 = none) for a bar whose count has maxed out but
    /// whose work continues — see [`Phase`].
    phase: AtomicU8,
    /// Which step of the read is running (0 = unknown) — see [`Stage`].
    stage: AtomicU8,
}

/// What a [`LoadProgress`] count measures — shown after the `done/total` on the
/// bar so it's clear (e.g. `64/64 shards`, not a bare `64/64`).
#[derive(Clone, Copy)]
pub enum Unit {
    Shards,
    Tensors,
    /// S3 objects being HEADed for their metadata (the `diff` s3-vs-s3 phase).
    S3Objects,
    /// Tensors compared value-by-value (the remote `diff --values`/`--histogram`
    /// phase, computed on the ssh proxy).
    Compared,
    /// Bytes read (a per-tensor download bar on the ssh proxy) — rendered as human
    /// sizes (`3.2 GiB/12.6 GiB`) instead of a raw count.
    Bytes,
}

/// A short trailing note on a bar whose count has reached its total but whose work
/// isn't finished — so a full bar with a still-running timer reads as *active*, not
/// stuck. Used by the remote value comparison: once a tensor's bytes are all read,
/// the proxy still has to decode + compare it (which the byte bar can't show).
#[derive(Clone, Copy)]
pub enum Phase {
    /// Bytes all read; the proxy is now decoding + comparing the tensor.
    Comparing,
}

/// Which step of a read is currently running, shown dimmed after the timer so a bar
/// that sits still for a second or two says *why*. A count alone can't: opening an
/// `s3://` checkpoint spends its first ~1.5s starting the remote reader before the
/// first tensor is counted, and then counts twice — once for tensors, once for the
/// S3 objects behind the stats screen's S3 section.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Stage {
    /// Remote python is starting and `cstorch.load` is opening the checkpoint —
    /// before any tensor can be counted.
    Index,
    /// Listing the checkpoint's shards over SFTP (the count isn't known yet).
    Listing,
    /// Reading each shard's safetensors header.
    Shards,
    /// Reading the tensor metadata out of the opened checkpoint.
    Tensors,
    /// HEADing each S3 object for the stats screen's S3 section.
    S3Objects,
}

impl Stage {
    /// The dimmed text shown after the timer.
    const fn label(self) -> &'static str {
        match self {
            Stage::Index => "loading the checkpoint index",
            Stage::Listing => "listing the checkpoint files",
            Stage::Shards => "reading shard headers",
            Stage::Tensors => "reading tensor metadata",
            Stage::S3Objects => "reading S3 object metadata",
        }
    }

    /// A terse form for a narrow terminal, where the full phrase wouldn't fit.
    ///
    /// Empty where the `done/total` count already names the thing being read — on a
    /// tight line, `1155/1155 tensors … tensors` is worse than nothing. The two steps
    /// that keep a short form are the ones with no count at all, which are exactly the
    /// ones where a bar otherwise looks stuck.
    const fn short(self) -> &'static str {
        match self {
            Stage::Index => "index",
            Stage::Listing => "listing files",
            Stage::Shards | Stage::Tensors | Stage::S3Objects => "",
        }
    }

    /// Every stage, so the tests can check the labels are distinct and complete.
    #[cfg(test)]
    const ALL: [Stage; 5] = [
        Stage::Index,
        Stage::Listing,
        Stage::Shards,
        Stage::Tensors,
        Stage::S3Objects,
    ];
}

/// Pick the widest stage text that fits in `room` columns (including the two leading
/// spaces): the full phrase, else the terse one, else nothing. The path label keeps
/// its own budget, so on a narrow terminal the stage is what gives way — never the
/// line's width, since a wrapped line breaks the in-place redraw.
fn fit_stage(room: usize, long: &'static str, short: &'static str) -> &'static str {
    if room >= long.chars().count() + 2 {
        long
    } else if !short.is_empty() && room >= short.chars().count() + 2 {
        short
    } else {
        ""
    }
}

impl LoadProgress {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record how many units the load comprises, once that's known.
    pub fn set_total(&self, total: usize) {
        self.total.store(total, Ordering::Relaxed);
    }

    /// Label the count's unit (for the bar); call once the reader knows it.
    pub fn set_unit(&self, unit: Unit) {
        let code = match unit {
            Unit::Shards => 1,
            Unit::Tensors => 2,
            Unit::S3Objects => 3,
            Unit::Compared => 4,
            Unit::Bytes => 5,
        };
        self.unit.store(code, Ordering::Relaxed);
    }

    /// The unit label for the bar (`""` when unset).
    pub fn unit_label(&self) -> &'static str {
        match self.unit.load(Ordering::Relaxed) {
            1 => " shards",
            2 => " tensors",
            3 => " S3 objects",
            4 => " compared",
            _ => "",
        }
    }

    /// Whether the count is a byte count (rendered as human sizes, not a raw count).
    pub fn is_bytes(&self) -> bool {
        self.unit.load(Ordering::Relaxed) == 5
    }

    /// Set the trailing activity note (shown only while the bar is still running).
    pub fn set_phase(&self, phase: Phase) {
        let code = match phase {
            Phase::Comparing => 1,
        };
        self.phase.store(code, Ordering::Relaxed);
    }

    /// The trailing activity note for the bar (`""` when none).
    pub fn phase_note(&self) -> &'static str {
        match self.phase.load(Ordering::Relaxed) {
            1 => " · comparing…",
            _ => "",
        }
    }

    /// Say which step is running now (shown dimmed after the timer).
    pub fn set_stage(&self, stage: Stage) {
        let code = match stage {
            Stage::Index => 1,
            Stage::Listing => 2,
            Stage::Shards => 3,
            Stage::Tensors => 4,
            Stage::S3Objects => 5,
        };
        self.stage.store(code, Ordering::Relaxed);
    }

    /// The step running now, or `None` until one is set.
    pub fn stage(&self) -> Option<Stage> {
        match self.stage.load(Ordering::Relaxed) {
            1 => Some(Stage::Index),
            2 => Some(Stage::Listing),
            3 => Some(Stage::Shards),
            4 => Some(Stage::Tensors),
            5 => Some(Stage::S3Objects),
            _ => None,
        }
    }

    /// Mark one more unit complete.
    pub fn advance(&self) {
        self.done.fetch_add(1, Ordering::Relaxed);
    }

    /// Set the absolute completed count (for a reader that reports totals rather
    /// than ticking — e.g. the remote cstorch dump's progress lines).
    pub fn set_done(&self, done: usize) {
        self.done.store(done, Ordering::Relaxed);
    }

    /// `(done, total)` for rendering; `total` is 0 until [`Self::set_total`].
    pub fn snapshot(&self) -> (usize, usize) {
        (
            self.done.load(Ordering::Relaxed),
            self.total.load(Ordering::Relaxed),
        )
    }
}

const RUNNING: u8 = 0;
const OK: u8 = 1;
const ERR: u8 = 2;
/// Cut short (not a failure of this read itself) — e.g. a parallel `diff` cancelled
/// this side because the *other* checkpoint failed to load.
const ABORTED: u8 = 3;

/// A set of progress bars, one per labelled read. Create with [`Bars::start`],
/// call [`Bars::finish`] as each read lands, and [`Bars::join`] once all are done.
pub struct Bars {
    states: Vec<Arc<AtomicU8>>,
    durations: Vec<Arc<AtomicU64>>,
    progress: Vec<Arc<LoadProgress>>,
    start: Instant,
    handle: Option<JoinHandle<()>>,
}

impl Bars {
    /// Reserve one bar per label and (on a terminal) start animating them.
    pub fn start(labels: Vec<String>) -> Bars {
        let n = labels.len();
        let states: Vec<_> = (0..n).map(|_| Arc::new(AtomicU8::new(RUNNING))).collect();
        let durations: Vec<_> = (0..n).map(|_| Arc::new(AtomicU64::new(0))).collect();
        let progress: Vec<_> = (0..n).map(|_| Arc::new(LoadProgress::new())).collect();
        let start = Instant::now();
        let handle = std::io::stderr().is_terminal().then(|| {
            spawn(
                labels,
                states.clone(),
                durations.clone(),
                progress.clone(),
                start,
            )
        });
        Bars {
            states,
            durations,
            progress,
            start,
            handle,
        }
    }

    /// The shared progress handle for read `i` — hand it to the reader so it can
    /// report shard/file completion, and the bar fills in as they land.
    pub fn progress(&self, i: usize) -> Option<Arc<LoadProgress>> {
        self.progress.get(i).cloned()
    }

    /// Mark read `i` finished — freezing its timer and showing `✓` (ok) or `✗`.
    pub fn finish(&self, i: usize, ok: bool) {
        if let Some(d) = self.durations.get(i) {
            d.store(self.start.elapsed().as_millis() as u64, Ordering::Relaxed);
        }
        if let Some(s) = self.states.get(i) {
            s.store(if ok { OK } else { ERR }, Ordering::Release);
        }
    }

    /// Mark read `i` **aborted** — cut short through no fault of its own (the other
    /// side of a `diff` failed first). Rendered as a dim `⊘` + an "aborted" note, so
    /// it doesn't read as a failure of this checkpoint.
    pub fn abort(&self, i: usize) {
        if let Some(s) = self.states.get(i) {
            s.store(ABORTED, Ordering::Release);
        }
    }

    /// Wait for the animation thread to draw the final state and exit.
    pub fn join(mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// Shorten `s` to at most `max` columns by replacing the middle with `…`, keeping
/// both ends — so a URI/path keeps its scheme/host prefix and its tail.
fn truncate_middle(s: &str, max: usize) -> String {
    let n = s.chars().count();
    if n <= max {
        return s.to_string();
    }
    if max <= 1 {
        return "…".to_string();
    }
    let keep = max - 1; // room for the ellipsis
    let head = keep.div_ceil(2);
    let tail = keep - head;
    let h: String = s.chars().take(head).collect();
    let t: String = s.chars().skip(n - tail).collect();
    format!("{h}…{t}")
}

/// Width of the drawn `━━━━━━` progress bar, in columns.
const BAR_COLS: usize = 16;

/// How many of `width` bar columns are filled for `done`/`total` (rounded,
/// clamped to `width`; empty when `total` is 0).
fn filled_cols(done: usize, total: usize, width: usize) -> usize {
    if total == 0 {
        return 0;
    }
    (((done as f64 / total as f64) * width as f64).round() as usize).min(width)
}

/// Start column of the indeterminate bar's bright `win`-wide window at animation
/// `frame`, ping-ponging across a `width`-wide bar (so it shows an alive bar
/// while the total is still unknown — connecting / listing the directory).
fn sweep_pos(frame: usize, width: usize, win: usize) -> usize {
    let span = width.saturating_sub(win);
    if span == 0 {
        return 0;
    }
    let t = frame % (span * 2);
    if t <= span { t } else { span * 2 - t }
}

fn spawn(
    labels: Vec<String>,
    states: Vec<Arc<AtomicU8>>,
    durations: Vec<Arc<AtomicU64>>,
    progress: Vec<Arc<LoadProgress>>,
    start: Instant,
) -> JoinHandle<()> {
    // Fit labels to the terminal width so a line (mark + path + bar + timer) can't
    // wrap and break the fixed-height redraw. Truncate in the *middle* so both
    // ends — the `s3://`/`host:` prefix and the checkpoint tail — stay visible.
    let cols = crate::utils::term_width(80);
    // Reserve room for the widest possible tail so the line can't wrap: the bar
    // itself + a byte count (`999.9 MiB/999.9 MiB`) + the `· comparing…` note +
    // the timer + separators — "  ⠋ <label>  [bar] 999.9 MiB/999.9 MiB · comparing…  12.3s".
    // The stage text appended after the timer takes whatever is left over (see
    // `fit_stage`), so it never costs the path any columns.
    let budget = cols.saturating_sub(BAR_COLS + 48).max(20);
    let labels: Vec<String> = labels.iter().map(|l| truncate_middle(l, budget)).collect();
    std::thread::spawn(move || {
        const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        // Bold cyan spinner, bold green ✓, bold red ✗; dimmed labels so the
        // coloured mark and the timer stand out.
        const RUN: &str = "\x1b[1;36m";
        const DONE: &str = "\x1b[1;32m";
        const FAIL: &str = "\x1b[1;31m";
        const DIM: &str = "\x1b[2m";
        const RESET: &str = "\x1b[0m";
        let n = labels.len();
        let mut err = std::io::stderr();
        for _ in 0..n {
            let _ = writeln!(err); // reserve n lines; cursor ends just below them
        }
        let mut i = 0usize;
        loop {
            let now: Vec<u8> = states.iter().map(|s| s.load(Ordering::Acquire)).collect();
            // Assemble the whole frame into one buffer and emit it in a single
            // write, so the terminal never renders a half-drawn frame (writing each
            // line separately to the unbuffered stderr is what makes it flicker).
            let mut frame = String::with_capacity(n * 96);
            frame.push_str(&format!("\x1b[{n}A")); // back up to the first reserved line
            for (k, &st) in now.iter().enumerate() {
                let (color, mark) = match st {
                    OK => (DONE, '✓'),
                    ERR => (FAIL, '✗'),
                    ABORTED => (DIM, '⊘'), // cut short, not a failure — dim, not red
                    _ => (RUN, FRAMES[i % FRAMES.len()]),
                };
                let ms = if st == RUNNING {
                    start.elapsed().as_millis() as u64
                } else {
                    durations[k].load(Ordering::Relaxed)
                };
                let secs = ms as f64 / 1000.0;
                // A `[███░░░] done/total` bar once the total is known (e.g. after a
                // remote dir is listed); until then just the spinner + timer.
                let (done, total) = progress[k].snapshot();
                let unit = progress[k].unit_label();
                // Each branch returns the drawn segment plus how many columns it
                // actually occupies (escape codes are zero-width), so the stage text
                // below knows how much room is left on the line.
                let (bar, bar_cols) = if st == ABORTED {
                    // Aborted: the partial count/timer would read as "failed partway",
                    // so replace them with a clear note (see the trailing timer too).
                    const NOTE: &str = "aborted — the other checkpoint failed to load";
                    (format!("  {DIM}{NOTE}{RESET}"), 2 + NOTE.chars().count())
                } else if total > 0 {
                    // Determinate: a thin bar in the TUI `LineGauge` style
                    // (`symbols::line::THICK`) — done part in the mark's colour, the
                    // rest dim — plus the `done/total` count (human sizes for a byte
                    // count) and its unit.
                    let filled = filled_cols(done, total, BAR_COLS);
                    let count = if progress[k].is_bytes() {
                        format!(
                            "{}/{}",
                            crate::utils::format_size(done),
                            crate::utils::format_size(total)
                        )
                    } else {
                        format!("{done}/{total}{unit}")
                    };
                    // A trailing note (e.g. `· comparing…`) when work continues past
                    // a full bar — but only while running, so a finished `✓` bar
                    // doesn't keep claiming to compare.
                    let note = if st == RUNNING {
                        progress[k].phase_note()
                    } else {
                        ""
                    };
                    (
                        format!(
                            "  {color}{}{RESET}{DIM}{}{RESET} {count}{DIM}{note}{RESET}",
                            "━".repeat(filled),
                            "━".repeat(BAR_COLS - filled),
                        ),
                        2 + BAR_COLS + 1 + count.chars().count() + note.chars().count(),
                    )
                } else if st == RUNNING {
                    // Total not known yet (still connecting / listing the dir) or an
                    // `s3://` read with no per-shard count: an indeterminate bar with
                    // a bright window sweeping across, so a live bar shows from the
                    // start instead of a bare spinner.
                    let win = 3.min(BAR_COLS);
                    let pos = sweep_pos(i, BAR_COLS, win);
                    (
                        format!(
                            "  {DIM}{}{RESET}{color}{}{RESET}{DIM}{}{RESET}",
                            "━".repeat(pos),
                            "━".repeat(win),
                            "━".repeat(BAR_COLS - pos - win),
                        ),
                        2 + BAR_COLS,
                    )
                } else {
                    (String::new(), 0) // finished with no known total: mark + timer only
                };
                // No timer on an aborted line — a partial time reads as a failure.
                let (timer, timer_cols) = if st == ABORTED {
                    (String::new(), 0)
                } else {
                    let text = format!("{secs:.1}s");
                    let cols = 1 + text.chars().count();
                    (format!(" {color}{text}{RESET}"), cols)
                };
                // Which step is running, dimmed, after the timer — so a bar that sits
                // at a steady count still says what it's doing. Only while running: a
                // finished `✓` line shouldn't claim to still be reading. Sized to the
                // columns left over, so it never pushes the line into wrapping.
                let stage = match progress[k].stage().filter(|_| st == RUNNING) {
                    None => String::new(),
                    Some(s) => {
                        let used = 4 + labels[k].chars().count() + bar_cols + timer_cols;
                        match fit_stage(cols.saturating_sub(used), s.label(), s.short()) {
                            "" => String::new(),
                            text => format!("  {DIM}{text}{RESET}"),
                        }
                    }
                };
                // `\r` + text + clear-to-EOL (`\x1b[K` *after* the text, so there's
                // no blank-then-fill flash) — overwrites the line in place.
                frame.push_str(&format!(
                    "\r  {color}{mark}{RESET} {DIM}{}{RESET}{bar}{timer}{stage}\x1b[K\n",
                    labels[k]
                ));
            }
            let _ = err.write_all(frame.as_bytes());
            let _ = err.flush();
            if now.iter().all(|&st| st != RUNNING) {
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
            i += 1;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{
        LoadProgress, Phase, Stage, Unit, filled_cols, fit_stage, sweep_pos, truncate_middle,
    };

    #[test]
    fn load_progress_tracks_done_and_total() {
        let p = LoadProgress::new();
        assert_eq!(p.snapshot(), (0, 0)); // total unknown until set
        assert_eq!(p.unit_label(), ""); // no unit until labelled
        p.set_total(48);
        p.set_unit(Unit::Shards);
        p.advance();
        p.advance();
        assert_eq!(p.snapshot(), (2, 48));
        assert_eq!(p.unit_label(), " shards");
    }

    #[test]
    fn stage_is_unset_until_told_and_round_trips() {
        let p = LoadProgress::new();
        assert!(p.stage().is_none(), "no stage until the reader names one");
        for s in Stage::ALL {
            p.set_stage(s);
            assert_eq!(p.stage(), Some(s));
        }
    }

    #[test]
    fn every_stage_has_its_own_label() {
        let mut seen: Vec<&str> = Vec::new();
        for s in Stage::ALL {
            let l = s.label();
            assert!(!l.is_empty(), "{s:?} has no label");
            assert!(!seen.contains(&l), "{s:?} duplicates the label {l:?}");
            seen.push(l);
        }
    }

    /// A terse form exists only for the steps with no `done/total` count. Where the
    /// count already names the unit, a short stage would just repeat it.
    #[test]
    fn only_the_countless_stages_have_a_short_form() {
        assert_eq!(Stage::Index.short(), "index");
        assert_eq!(Stage::Listing.short(), "listing files");
        for s in [Stage::Shards, Stage::Tensors, Stage::S3Objects] {
            assert_eq!(s.short(), "", "{s:?} should fall back to nothing");
        }
    }

    #[test]
    fn fit_stage_gives_way_before_the_line_wraps() {
        let (long, short) = (Stage::Index.label(), Stage::Index.short());
        // Exactly enough room (text + the two leading spaces) keeps the long form.
        assert_eq!(fit_stage(long.len() + 2, long, short), long);
        assert_eq!(fit_stage(long.len() + 1, long, short), short);
        assert_eq!(fit_stage(short.len() + 2, long, short), short);
        assert_eq!(fit_stage(short.len() + 1, long, short), "");
        assert_eq!(fit_stage(0, long, short), "");
        // With no short form, the long one is all-or-nothing.
        let t = Stage::Tensors;
        assert_eq!(
            fit_stage(t.label().len() + 2, t.label(), t.short()),
            t.label()
        );
        assert_eq!(fit_stage(t.label().len() + 1, t.label(), t.short()), "");
    }

    #[test]
    fn phase_note_is_off_until_set() {
        let p = LoadProgress::new();
        assert_eq!(p.phase_note(), ""); // no note by default
        p.set_phase(Phase::Comparing);
        assert_eq!(p.phase_note(), " · comparing…");
    }

    #[test]
    fn sweep_window_ping_pongs_within_bounds() {
        let (w, win) = (16usize, 3usize);
        let span = w - win; // 13
        assert_eq!(sweep_pos(0, w, win), 0); // starts at the left
        assert_eq!(sweep_pos(span, w, win), span); // reaches the right edge
        assert_eq!(sweep_pos(span + 1, w, win), span - 1); // then reverses
        // Never runs the window past the bar.
        for f in 0..100 {
            assert!(sweep_pos(f, w, win) + win <= w, "frame {f} overflows");
        }
    }

    #[test]
    fn bar_fill_is_proportional_and_clamped() {
        assert_eq!(filled_cols(0, 48, 16), 0);
        assert_eq!(filled_cols(24, 48, 16), 8); // half
        assert_eq!(filled_cols(48, 48, 16), 16); // full
        assert_eq!(filled_cols(47, 48, 16), 16); // rounds up, still clamped
        assert_eq!(filled_cols(5, 0, 16), 0); // no total → empty, no divide-by-zero
    }

    #[test]
    fn middle_truncation_keeps_both_ends() {
        // Short enough → untouched.
        assert_eq!(truncate_middle("s3://bucket/key", 100), "s3://bucket/key");
        // Ellipsis goes in the middle, both ends kept.
        assert_eq!(truncate_middle("abcdefghij", 5), "ab…ij");
        assert_eq!(truncate_middle("abcdefghij", 1), "…");
        // A long string is elided in the middle: the kept head and tail are a real
        // prefix and suffix of the input, and the result fits the budget.
        let s = "s3://inference-opensource/minimax-m2.5/4bit/260402";
        let t = truncate_middle(s, 24);
        assert!(t.chars().count() <= 24 && t.contains('…'), "{t}");
        let (head, tail) = t.split_once('…').unwrap();
        assert!(s.starts_with(head) && s.ends_with(tail), "{t}");
    }
}
