//! Per-view caches and the background statistics scan.
//!
//! Split out of `explorer/mod.rs` as one concern: everything here exists so that
//! redrawing a data view doesn't redo expensive work. Re-opening a reader dominates the
//! cost of panning (and for HDF5 discards libhdf5's chunk cache), and re-sampling on
//! every spinner frame would contend with the scan worker's HDF5 lock and make
//! keystrokes pile up — so the reader, the sampled grid and the scan handle are all
//! held across frames and keyed by exactly what they depend on.

#[allow(clippy::wildcard_imports)] // a submodule of the module it was split from
use super::*;

/// Which representation a tensor data view renders.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Representation {
    /// ASCII heatmap (`m`).
    Heatmap,
    /// Numeric values grid (`v`).
    Values,
}

/// An open reader for the tensor currently being viewed, kept across redraws so
/// panning / slice-stepping a data view doesn't re-open the file every frame
/// (re-opening dominates the cost and, for HDF5, also discards libhdf5's chunk
/// cache — see the `window_pan_open_cost` benchmark).
pub(super) struct CachedReader {
    pub(super) source_path: String,
    pub(super) name: String,
    pub(super) reader: Box<dyn crate::sample::TensorReader>,
}

/// A statistics scan running on a worker thread for a data view's current
/// `(tensor, view)`. The view stays fully interactive while it runs; the main
/// loop polls [`Self::handle`], caches the result when it lands, and animates the
/// spinner meanwhile. Dropping the job — because the view closed or the dtype
/// changed — cancels the worker at its next block boundary so no work is wasted.
pub(super) struct ScanJob {
    pub(super) view: ViewDtype,
    pub(super) cancel: Arc<AtomicBool>,
    /// Set to make the worker wait between blocks (releasing the file lock) so a
    /// foreground read can run uncontended; cleared to resume where it left off.
    pub(super) pause: Arc<AtomicBool>,
    pub(super) handle: Option<std::thread::JoinHandle<Result<Stats, String>>>,
    pub(super) started: std::time::Instant,
    /// Stored bytes the worker has scanned so far (it bumps this per block), and
    /// the total it will scan (`size_bytes`). Together they drive the progress bar.
    pub(super) done: Arc<AtomicUsize>,
    pub(super) total: usize,
}

impl ScanJob {
    /// Fraction of the tensor scanned so far (`0.0..=1.0`), or `None` when the
    /// total is unknown (empty tensor) so the caller shows just the spinner.
    pub(super) fn progress(&self) -> Option<f64> {
        (self.total > 0)
            .then(|| (self.done.load(Ordering::Relaxed) as f64 / self.total as f64).min(1.0))
    }
}

impl Drop for ScanJob {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

/// Keep at most one scan alive in `slot` for `tensor`'s current `view`, harvesting a
/// finished one into the stats cache. Returns whether the caller must keep polling.
///
/// Shared by the detail and data screens: they show a tensor's statistics with different
/// chrome, but their relationship to the scan is the same — one job per `(tensor, view)`,
/// restarted when the dtype view changes, joined into the cache when it lands. Each screen
/// keeps its own decision about *whether* to scan (detail only pre-warms when asked to);
/// this is only the mechanics, which had drifted into two copies.
pub(super) fn poll_stats_scan(
    ex: &mut Explorer,
    slot: &mut Option<ScanJob>,
    tensor: &TensorInfo,
    view: ViewDtype,
) -> Bg {
    if ex.cached_stats(tensor, view).is_some() {
        *slot = None;
        return Bg::Idle;
    }
    if slot.as_ref().is_none_or(|j| j.view != view) {
        *slot = Some(ex.spawn_stats_scan(tensor, view));
    }
    let finished = slot
        .as_ref()
        .and_then(|j| j.handle.as_ref())
        .is_some_and(std::thread::JoinHandle::is_finished);
    if finished
        && let Some(mut job) = slot.take()
        && let Some(h) = job.handle.take()
        && let Ok(Ok(s)) = h.join()
    {
        ex.stats_cache
            .borrow_mut()
            .insert((tensor.name.clone(), view), s);
    }
    if slot.is_some() { Bg::Poll } else { Bg::Idle }
}

/// How a scan in `slot` should be described to the renderer: the cached result, a live
/// spinner with its elapsed time and progress, or nothing yet.
///
/// The spinner only appears after 120 ms, so a scan that finishes quickly never flashes
/// one. `spin` is a `Cell` because this is called from `render_frame`, which has only `&self`.
pub(super) fn scan_stats_view<'a>(
    slot: &Option<ScanJob>,
    spin: &Cell<usize>,
    stats: &'a Option<Stats>,
) -> StatsView<'a> {
    if let Some(s) = stats {
        return StatsView::Ready(s);
    }
    let Some(job) = slot else {
        return StatsView::Pending;
    };
    if job.started.elapsed() < std::time::Duration::from_millis(120) {
        return StatsView::Pending;
    }
    spin.set(spin.get().wrapping_add(1));
    StatsView::Computing {
        spinner: crate::progress::spinner_frame(spin.get()),
        elapsed: job.started.elapsed(),
        progress: job.progress(),
    }
}

/// The last sample a data view rendered, reused when nothing that affects it
/// changed. This keeps the spinner-frame redraws during a stats scan from
/// re-reading (and, for HDF5, re-decompressing) the tensor every frame — those
/// reads block on the scan worker's HDF5 lock, which otherwise lags the UI and
/// lets keystrokes pile up. Keyed by everything the sampled grid depends on.
pub(super) struct CachedSample {
    pub(super) key: SampleKey,
    pub(super) sample: crate::sample::Sample,
}

/// Everything that determines a data view's sampled grid. `max_rows`/`max_cols`
/// fold in the terminal size and (for the numeric grid) the stats-derived cell
/// width, so the key changes — and the grid re-samples once — when the exact
/// stats land.
pub(super) type SampleKey = (
    String,         // tensor name
    Representation, // heatmap vs numeric grid
    usize,          // slice
    ViewDtype,      // dtype reinterpretation
    SampleMode,     // layout + offsets / tails
    usize,          // max_rows
    usize,          // max_cols
    Vec<usize>,     // effective shape (stored, or a shape override)
);

/// Cache key for a computed histogram: tensor name, view (dtype reinterpretation)
/// and the requested bucket count (`None` = automatic) — a different count caches
/// separately rather than reusing a stale layout.
pub(super) type HistKey = (String, ViewDtype, Option<usize>);
