//! Live progress bars for remote reads: one colored spinner + elapsed timer per
//! read, settling to `✓` (green) or `✗` (red). Animated on a background thread —
//! off the main thread doing the blocking SSH reads, touching only shared atomics,
//! so it never races the sessions — and suppressed when stderr isn't a terminal
//! (escape codes never pollute a pipe/log). Callers must do any password prompt
//! *before* starting the bars.

use std::borrow::Cow;
use std::fmt::Write as _;
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
    /// S3 objects being `HEADed` for their metadata (the `diff` s3-vs-s3 phase).
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
    /// `HEADing` each S3 object for its *storage* facts — size on S3, `ETag`, checksum,
    /// date, tags — which the checkpoint index doesn't carry. Not a second read of the
    /// tensor metadata: that came from the one `__METADATA__` object during
    /// [`Stage::Index`].
    S3Objects,
}

impl Stage {
    /// The dimmed text shown after the timer.
    const fn label(self) -> &'static str {
        match self {
            Self::Index => "loading the checkpoint index",
            Self::Listing => "listing the checkpoint files",
            Self::Shards => "reading shard headers",
            Self::Tensors => "reading tensor metadata",
            Self::S3Objects => "reading S3 storage metadata",
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
            Self::Index => "index",
            Self::Listing => "listing files",
            Self::Shards | Self::Tensors | Self::S3Objects => "",
        }
    }

    /// Every stage, so the tests can check the labels are distinct and complete.
    #[cfg(test)]
    const ALL: [Self; 5] = [
        Self::Index,
        Self::Listing,
        Self::Shards,
        Self::Tensors,
        Self::S3Objects,
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
    #[must_use]
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
    #[must_use]
    pub fn start(labels: &[String]) -> Self {
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
        Self {
            states,
            durations,
            progress,
            start,
            handle,
        }
    }

    /// The shared progress handle for read `i` — hand it to the reader so it can
    /// report shard/file completion, and the bar fills in as they land.
    #[must_use]
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

/// Spinner frames for a running bar.
const FRAMES: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
// Bold cyan spinner, bold green ✓, bold red ✗; dimmed labels so the coloured mark and
// the timer stand out.
const RUN: &str = "\x1b[1;36m";
const DONE: &str = "\x1b[1;32m";
const FAIL: &str = "\x1b[1;31m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";

/// One bar at one instant: what [`render_line`] needs, read out of the atomics by the
/// animation thread — or built directly by a test.
struct BarView<'a> {
    label: &'a str,
    state: u8,
    ms: u64,
    done: usize,
    total: usize,
    unit: &'static str,
    is_bytes: bool,
    note: &'static str,
    stage: Option<Stage>,
}

/// Draw one bar line — escape codes, bar, count, timer and stage — to fit `cols`
/// columns, ready to `\r`-overwrite the line it's on.
///
/// Split out of the animation thread so it's a pure function of a snapshot: the column
/// arithmetic here is what keeps a line from wrapping (a wrapped line breaks the
/// fixed-height in-place redraw), and inside the thread there was no way to check it
/// without a terminal and an eyeball.
/// The dimmed step text that trails the timer, fitted to the `room` columns left over —
/// the full phrase, the terse one, or nothing at all.
///
/// A function rather than a closure chain at the call site so the three outcomes read
/// top-to-bottom: no step to report, a step that doesn't fit, a step that does.
fn stage_text(stage: Option<Stage>, room: usize) -> String {
    let Some(s) = stage else {
        return String::new();
    };
    match fit_stage(room, s.label(), s.short()) {
        "" => String::new(),
        text => format!("  {DIM}{text}{RESET}"),
    }
}

fn render_line(v: &BarView, frame: usize, cols: usize) -> String {
    let (color, mark) = match v.state {
        OK => (DONE, '✓'),
        ERR => (FAIL, '✗'),
        ABORTED => (DIM, '⊘'), // cut short, not a failure — dim, not red
        _ => (RUN, FRAMES[frame % FRAMES.len()]),
    };
    let secs = v.ms as f64 / 1000.0;
    // Each piece below carries the columns it actually occupies alongside its text, since
    // the escape codes are zero-width and the fitting further down counts columns.
    //
    // The ━ gauge and the `done/total` count are separate pieces on purpose: the gauge is
    // the first thing given up on a narrow terminal, and the count is far more use than
    // the picture of it.
    let (gauge, gauge_cols) = if v.state == ABORTED || (v.total == 0 && v.state != RUNNING) {
        (String::new(), 0) // aborted, or finished with no total: mark + text only
    } else if v.total > 0 {
        // Determinate: a thin bar in the TUI `LineGauge` style (`symbols::line::THICK`) —
        // done part in the mark's colour, the rest dim.
        let filled = filled_cols(v.done, v.total, BAR_COLS);
        (
            format!(
                "  {color}{}{RESET}{DIM}{}{RESET}",
                "━".repeat(filled),
                "━".repeat(BAR_COLS - filled),
            ),
            2 + BAR_COLS,
        )
    } else {
        // Total not known yet (still connecting / listing the dir) or an `s3://` read
        // with no per-shard count: an indeterminate bar with a bright window sweeping
        // across, so a live bar shows from the start instead of a bare spinner.
        let win = 3.min(BAR_COLS);
        let pos = sweep_pos(frame, BAR_COLS, win);
        (
            format!(
                "  {DIM}{}{RESET}{color}{}{RESET}{DIM}{}{RESET}",
                "━".repeat(pos),
                "━".repeat(win),
                "━".repeat(BAR_COLS - pos - win),
            ),
            2 + BAR_COLS,
        )
    };
    let (count, count_cols) = if v.state == ABORTED {
        // Aborted: a partial count and a partial time both read as "died partway", so
        // they're replaced by the reason (and the timer is dropped below). The reason
        // always shows in some form — losing it would leave a bare `⊘` — so it has a
        // terse fallback for a narrow pane rather than a fit that can come back empty.
        const LONG: &str = "aborted — the other checkpoint failed to load";
        const SHORT: &str = "aborted";
        let room = cols.saturating_sub(4 + v.label.chars().count());
        let text = if room >= LONG.chars().count() + 2 {
            LONG
        } else {
            SHORT
        };
        (format!("  {DIM}{text}{RESET}"), 2 + text.chars().count())
    } else if v.total > 0 {
        // `done/total` and its unit — human sizes for a byte count — plus a trailing note
        // (e.g. `· comparing…`) when work continues past a full bar, but only while
        // running, so a finished `✓` bar doesn't keep claiming to compare.
        let text = if v.is_bytes {
            format!(
                "{}/{}",
                crate::utils::format_size(v.done),
                crate::utils::format_size(v.total)
            )
        } else {
            format!("{}/{}{}", v.done, v.total, v.unit)
        };
        let note = if v.state == RUNNING { v.note } else { "" };
        (
            format!(" {text}{DIM}{note}{RESET}"),
            1 + text.chars().count() + note.chars().count(),
        )
    } else {
        (String::new(), 0)
    };
    // No timer on an aborted line — see the count above.
    let (timer, timer_cols) = if v.state == ABORTED {
        (String::new(), 0)
    } else {
        let text = format!("{secs:.1}s");
        let width = 1 + text.chars().count();
        (format!(" {color}{text}{RESET}"), width)
    };
    // In a terminal too narrow for all of it, give up the gauge first and then trim the
    // path — never the line's width. `spawn` already fits the label to a budget, but that
    // budget has a floor (a three-column path tells you nothing), so in a ~40-column pane
    // the line would otherwise run past the edge, wrap, and send the bars marching down
    // the screen as each frame redrew below the last.
    let mut label = Cow::Borrowed(v.label);
    let (mut gauge, mut gauge_cols) = (gauge, gauge_cols);
    let (mut count, mut count_cols) = (count, count_cols);
    let width = |label: &str, gauge_cols: usize, count_cols: usize| {
        4 + label.chars().count() + gauge_cols + count_cols + timer_cols
    };
    if width(&label, gauge_cols, count_cols) > cols {
        (gauge, gauge_cols) = (String::new(), 0);
    }
    if width(&label, gauge_cols, count_cols) > cols {
        let room = cols.saturating_sub(4 + count_cols + timer_cols);
        label = Cow::Owned(truncate_middle(&label, room));
    }
    if width(&label, gauge_cols, count_cols) > cols {
        // Last resort: a pane narrower than `999.9 MiB/999.9 MiB · comparing…` isn't one
        // this display can serve — but it still must not wrap, so the count goes and the
        // path takes back the room.
        (count, count_cols) = (String::new(), 0);
        label = Cow::Owned(truncate_middle(
            v.label,
            cols.saturating_sub(4 + timer_cols),
        ));
    }
    // Which step is running, dimmed, after the timer — so a bar that sits at a steady
    // count still says what it's doing. Only while running: a finished `✓` line
    // shouldn't claim to still be reading. Sized to the columns left over, so it never
    // pushes the line into wrapping.
    let stage = stage_text(
        v.stage.filter(|_| v.state == RUNNING),
        cols.saturating_sub(width(&label, gauge_cols, count_cols)),
    );
    // `\r` + text + clear-to-EOL (`\x1b[K` *after* the text, so there's no
    // blank-then-fill flash) — overwrites the line in place.
    format!("\r  {color}{mark}{RESET} {DIM}{label}{RESET}{gauge}{count}{timer}{stage}\x1b[K\n")
}

fn spawn(
    labels: &[String],
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
            let _ = write!(frame, "\x1b[{n}A"); // back up to the first reserved line
            for (k, &st) in now.iter().enumerate() {
                // A `[███░░░] done/total` bar once the total is known (e.g. after a
                // remote dir is listed); until then just the spinner + timer.
                let (done, total) = progress[k].snapshot();
                let view = BarView {
                    label: &labels[k],
                    state: st,
                    ms: if st == RUNNING {
                        start.elapsed().as_millis() as u64
                    } else {
                        durations[k].load(Ordering::Relaxed)
                    },
                    done,
                    total,
                    unit: progress[k].unit_label(),
                    is_bytes: progress[k].is_bytes(),
                    note: progress[k].phase_note(),
                    stage: progress[k].stage(),
                };
                frame.push_str(&render_line(&view, i, cols));
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
        ABORTED, BAR_COLS, BarView, DIM, ERR, LoadProgress, OK, Phase, RESET, RUN, RUNNING, Stage,
        Unit, filled_cols, fit_stage, render_line, sweep_pos, truncate_middle,
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

    // --- the drawn line -----------------------------------------------------------
    //
    // `render_line` is what the animation thread emits, so these are the only tests that
    // see what a user sees. They assert on the *visible* text (escape codes stripped)
    // plus the one invariant the column arithmetic in there exists for: the line must
    // never reach the terminal's width, because a wrapped line breaks the fixed-height
    // in-place redraw and the bars start marching down the screen.

    /// Drop the ANSI escape sequences, leaving what the terminal actually shows.
    fn visible(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c != '\x1b' {
                if c != '\r' && c != '\n' {
                    out.push(c);
                }
                continue;
            }
            // CSI: `\x1b[` … final byte in @-~
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        }
        out
    }

    fn view(label: &str, state: u8) -> BarView<'_> {
        BarView {
            label,
            state,
            ms: 1234,
            done: 0,
            total: 0,
            unit: "",
            is_bytes: false,
            note: "",
            stage: None,
        }
    }

    #[test]
    fn a_determinate_bar_shows_the_count_its_unit_and_the_timer() {
        let mut v = view("host:/ckpt", RUNNING);
        v.done = 24;
        v.total = 48;
        v.unit = " shards";
        let line = visible(&render_line(&v, 0, 120));
        assert!(line.contains("host:/ckpt"), "{line}");
        assert!(line.contains("24/48 shards"), "{line}");
        assert!(line.contains("1.2s"), "{line}");
        // Half done → half the bar is drawn in the bright colour, the rest dim. The
        // escape codes are what carry that, so check the raw form for the split.
        let raw = render_line(&v, 0, 120);
        assert!(raw.contains(&format!(
            "{RUN}{}{RESET}{DIM}{}",
            "━".repeat(8),
            "━".repeat(8)
        )));
    }

    #[test]
    fn a_byte_count_is_drawn_as_human_sizes() {
        let mut v = view("host:/ckpt", RUNNING);
        v.done = 3 << 30;
        v.total = 12 << 30;
        v.is_bytes = true;
        v.unit = ""; // bytes carry their unit in the formatted size
        let line = visible(&render_line(&v, 0, 120));
        assert!(line.contains("3.0 GiB/12.0 GiB"), "{line}");
    }

    /// The `· comparing…` note says a full bar is still working — but only while it is.
    #[test]
    fn the_phase_note_disappears_once_the_bar_finishes() {
        let mut v = view("host:/ckpt", RUNNING);
        (v.done, v.total, v.note) = (48, 48, " · comparing…");
        assert!(visible(&render_line(&v, 0, 120)).contains("· comparing…"));
        v.state = OK;
        let done = visible(&render_line(&v, 0, 120));
        assert!(
            !done.contains("comparing"),
            "a ✓ line must not claim to work: {done}"
        );
        assert!(done.contains('✓'), "{done}");
    }

    #[test]
    fn an_indeterminate_bar_sweeps_a_full_width_bar() {
        let v = view("ckpt", RUNNING); // total 0 → no count to draw (and no `/` of its own)
        for frame in 0..40 {
            let line = visible(&render_line(&v, frame, 120));
            let bar: String = line.chars().filter(|&c| c == '━').collect();
            assert_eq!(bar.chars().count(), BAR_COLS, "frame {frame}: {line}");
            assert!(
                !line.contains('/'),
                "no count while the total is unknown: {line}"
            );
        }
    }

    /// An aborted read is not a failure of *this* checkpoint, so it gets neither the red
    /// ✗ nor a timer — a partial time reads as "died partway".
    #[test]
    fn an_aborted_line_explains_itself_and_drops_the_timer() {
        let mut v = view("host:/ckpt", ABORTED);
        (v.done, v.total) = (12, 48); // a partial count that must NOT be shown
        let line = visible(&render_line(&v, 0, 120));
        assert!(line.contains('⊘') && !line.contains('✗'), "{line}");
        assert!(
            line.contains("aborted — the other checkpoint failed to load"),
            "{line}"
        );
        assert!(!line.contains("1.2s") && !line.contains("12/48"), "{line}");
    }

    #[test]
    fn a_finished_read_with_no_total_is_just_the_mark_and_the_timer() {
        let ok = visible(&render_line(&view("host:/ckpt", OK), 0, 120));
        assert_eq!(ok, "  ✓ host:/ckpt 1.2s");
        let err = visible(&render_line(&view("host:/ckpt", ERR), 0, 120));
        assert_eq!(err, "  ✗ host:/ckpt 1.2s");
    }

    #[test]
    fn the_stage_shows_only_while_running() {
        let mut v = view("host:/ckpt", RUNNING);
        v.stage = Some(Stage::Tensors);
        assert!(visible(&render_line(&v, 0, 120)).contains("reading tensor metadata"));
        v.state = OK;
        let done = visible(&render_line(&v, 0, 120));
        assert!(
            !done.contains("reading"),
            "a ✓ line still claims to read: {done}"
        );
    }

    /// What a narrow pane gives up, in order: the ━ gauge first (a picture of the
    /// progress), then the path, and never the `done/total` count — which is the part
    /// actually worth reading.
    #[test]
    fn a_narrow_pane_drops_the_gauge_but_keeps_the_count() {
        let mut v = view("s3://bucket/ckpt/260626", RUNNING);
        (v.done, v.total, v.unit) = (823, 1155, " tensors");
        let wide = visible(&render_line(&v, 0, 120));
        assert!(
            wide.contains('━') && wide.contains("823/1155 tensors"),
            "{wide}"
        );
        let narrow = visible(&render_line(&v, 0, 60));
        assert!(!narrow.contains('━'), "the gauge should go first: {narrow}");
        assert!(
            narrow.contains("823/1155 tensors"),
            "the count must stay: {narrow}"
        );
        assert!(
            narrow.contains("s3://") && narrow.contains("260626"),
            "{narrow}"
        );
        // Narrower still: the path gives way next, keeping both its ends.
        let tiny = visible(&render_line(&v, 0, 40));
        assert!(
            tiny.contains("823/1155 tensors") && tiny.contains('…'),
            "{tiny}"
        );
    }

    /// An aborted line must always say *why*, so its note has a terse form rather than a
    /// fit that can come back empty — a bare `⊘` would be a mystery.
    #[test]
    fn the_aborted_reason_shortens_rather_than_vanishing() {
        let v = view("s3://bucket/ckpt/260626", ABORTED);
        assert!(visible(&render_line(&v, 0, 120)).contains("the other checkpoint failed"));
        let narrow = visible(&render_line(&v, 0, 40));
        assert!(narrow.contains("aborted"), "{narrow}");
        assert!(!narrow.contains("other checkpoint"), "{narrow}");
    }

    /// The stage is what gives way on a narrow terminal — long form, then terse, then
    /// nothing — and the path keeps its own budget either way.
    #[test]
    fn a_narrowing_terminal_degrades_the_stage_not_the_line() {
        let mut v = view("host:/ckpt", RUNNING);
        v.stage = Some(Stage::Index);
        let at = |cols| visible(&render_line(&v, 0, cols));
        assert!(at(120).contains("loading the checkpoint index"));
        assert!(
            at(46).contains("index") && !at(46).contains("loading"),
            "{}",
            at(46)
        );
        assert!(!at(20).contains("index"), "{}", at(20));
    }

    /// The invariant the whole `bar_cols`/`timer_cols` accounting is for. Swept across
    /// every state, both bar kinds, the widest count and note, and terminal widths from
    /// a narrow split pane to generous — the drawn line must always fit the terminal.
    ///
    /// Exactly filling the width is allowed: what advances the line is the trailing `\n`,
    /// and nothing printable is written after it before the next frame's `\r`, so the
    /// pending-wrap state is never resolved into a real wrap. One column *past* the width
    /// is what breaks the fixed-height redraw and sends the bars marching down the screen.
    #[test]
    fn a_drawn_line_never_overruns_the_terminal_width() {
        for cols in [20usize, 40, 60, 80, 100, 120, 200] {
            // What `spawn` would hand us: the label is pre-truncated to its budget.
            let budget = cols.saturating_sub(BAR_COLS + 48).max(20);
            let label = truncate_middle("s3://inference-testing/kimi-k2.6/3bit-22s/260626", budget);
            for state in [RUNNING, OK, ERR, ABORTED] {
                for (done, total, is_bytes) in [
                    (0, 0, false),
                    (24, 48, false),
                    (999 << 20, 1023 << 20, true),
                ] {
                    for stage in Stage::ALL.map(Some).into_iter().chain([None]) {
                        let v = BarView {
                            label: &label,
                            state,
                            ms: 999_999, // "1000.0s" — the widest timer we'd ever draw
                            done,
                            total,
                            unit: " S3 objects", // the widest unit label
                            is_bytes,
                            note: " · comparing…",
                            stage,
                        };
                        let drawn = visible(&render_line(&v, 7, cols)).chars().count();
                        assert!(
                            drawn <= cols,
                            "{drawn} cols drawn in a {cols}-col terminal \
                             (state {state}, {done}/{total}, stage {stage:?})",
                        );
                    }
                }
            }
        }
    }
}
