//! Long-running work, polled — the browser's route to `diff --values` / `--verify-repack`.
//!
//! Every other handler answers in one request because everything else the API serves is either
//! precomputed or a bounded read. These are not: verifying a repack reads *both* tensors of every
//! selected pair, minutes of it, and reports per-tensor findings as they land. A synchronous request
//! would hold a `tiny_http` worker for the whole run, give the browser nothing until the end, and lose
//! everything on a reload.
//!
//! So a job is a thing you start, poll, and can stop:
//!
//! ```text
//! POST   /api/jobs/verify-repack?left=…&right=…&<scope>   -> { "id": 7 }
//! GET    /api/jobs/7                                      -> state, progress, findings so far
//! DELETE /api/jobs/7                                      -> ask it to stop
//! ```
//!
//! **Polling rather than streaming.** `tiny_http` is synchronous and thread-per-request, so a
//! held-open response costs a worker for the run's whole duration; and an SSE stream through an
//! intervening proxy is commonly buffered or timed out. Polling costs a request every half second and
//! survives a reload, because the job outlives the tab that started it.
//!
//! **Cancellation is the same mechanism as a read's.** A job carries the
//! [`crate::hf::ReadProgress`] its work was handed, so `DELETE` sets the flag the remote reader already
//! checks between chunks — see `Current::begin_read_taking_over` for the other user of it.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use serde_json::{Value, json};

/// Where a job has got to. An enum rather than a pair of booleans: "running", "finished" and "failed"
/// are three states, and a client renders each differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum State {
    Running,
    /// Finished on its own.
    Done,
    /// Stopped because someone asked.
    Cancelled,
    /// Gave up — `Job::error` says why.
    Failed,
}

impl State {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Done => "done",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        }
    }
}

/// One job's shared state. The worker writes; pollers read.
///
/// Atomics for the counters so a poll never blocks the worker, and a `Mutex` only around the findings
/// list and the two strings — which are touched per *tensor*, not per byte.
pub(crate) struct Job {
    pub(crate) id: u64,
    /// What kind of work this is, for the poll response.
    kind: &'static str,
    state: RwLock<State>,
    done: AtomicUsize,
    total: AtomicUsize,
    /// Bytes read so far, where the work reports them — a repack verify reads whole tensors, and the
    /// byte count is the only honest measure of how far through a big one it is.
    bytes: AtomicU64,
    /// What it is working on right now.
    current: Mutex<String>,
    /// Per-item results as they land, so a poll shows findings before the run ends.
    findings: Mutex<Vec<Value>>,
    error: Mutex<Option<String>>,
    started: std::time::Instant,
    /// The handle the work was given, so a stop request reaches it.
    progress: Arc<crate::hf::ReadProgress>,
}

impl Job {
    /// How many items, once known. Reported so a bar has a denominator.
    pub(crate) fn set_total(&self, total: usize) {
        self.total.store(total, Ordering::Relaxed);
    }

    /// Move on to `name`, having finished `done` items.
    pub(crate) fn progress_to(&self, done: usize, name: &str) {
        self.done.store(done, Ordering::Relaxed);
        *self
            .current
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = name.to_string();
    }

    /// Cumulative bytes read.
    pub(crate) fn set_bytes(&self, bytes: u64) {
        self.bytes.store(bytes, Ordering::Relaxed);
    }

    /// Record one item's result. Appended as it lands rather than collected at the end, because the
    /// first finding is often the answer and waiting minutes for it is the thing this design avoids.
    pub(crate) fn add_finding(&self, finding: Value) {
        self.findings
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(finding);
    }

    /// Whether a stop has been asked for — the work checks this, and so does the remote reader it calls.
    pub(crate) fn cancelled(&self) -> bool {
        self.progress.cancelled()
    }

    /// The read handle to hand to the work, so cancelling reaches inside it.
    pub(crate) fn read_progress(&self) -> &crate::hf::ReadProgress {
        &self.progress
    }

    fn finish(&self, outcome: State, error: Option<String>) {
        *self
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = outcome;
        if let Some(msg) = error {
            *self
                .error
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(msg);
        }
    }

    /// The poll response.
    pub(crate) fn snapshot(&self) -> Value {
        let state = *self
            .state
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        json!({
            "id": self.id,
            "kind": self.kind,
            "state": state.as_str(),
            "done": self.done.load(Ordering::Relaxed),
            // 0 until the work knows how many items there are — a client shows a spinner rather than a
            // bar at zero, the same rule `ReadProgress` uses.
            "total": self.total.load(Ordering::Relaxed),
            "bytes": self.bytes.load(Ordering::Relaxed),
            "current": *self.current.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
            "elapsed_s": self.started.elapsed().as_secs_f64(),
            "findings": *self.findings.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
            "error": *self.error.lock().unwrap_or_else(std::sync::PoisonError::into_inner),
        })
    }
}

/// Every job this process has started.
///
/// Kept after finishing, so a poll that arrives after the last item still gets the results — and so a
/// reload can pick a run back up. Capped, because a browser left open could otherwise start one every
/// few minutes forever.
pub(crate) struct Jobs {
    jobs: Mutex<Vec<Arc<Job>>>,
    next: AtomicU64,
}

/// How many finished jobs to keep. Enough to compare a few runs; small enough that their findings
/// cannot grow without bound.
const KEEP: usize = 16;

impl Default for Jobs {
    fn default() -> Self {
        Self {
            jobs: Mutex::new(Vec::new()),
            next: AtomicU64::new(1),
        }
    }
}

impl Jobs {
    /// Register a job and return it, for a worker to fill in.
    pub(crate) fn start(&self, kind: &'static str) -> Arc<Job> {
        let job = Arc::new(Job {
            id: self.next.fetch_add(1, Ordering::Relaxed),
            kind,
            state: RwLock::new(State::Running),
            done: AtomicUsize::new(0),
            total: AtomicUsize::new(0),
            bytes: AtomicU64::new(0),
            current: Mutex::new(String::new()),
            findings: Mutex::new(Vec::new()),
            error: Mutex::new(None),
            started: std::time::Instant::now(),
            progress: Arc::new(crate::hf::ReadProgress::default()),
        });
        self.register(Arc::clone(&job));
        job
    }

    /// Add a job to the registry and evict old finished ones.
    ///
    /// Its own function so the registry lock's lifetime is the function's, rather than reaching to the
    /// end of `start` — the caller then runs minutes of work, and holding this across that would block
    /// every poll.
    fn register(&self, job: Arc<Job>) {
        let mut jobs = self
            .jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        jobs.push(job);
        // Drop the oldest *finished* jobs. A running one is never evicted — its worker holds an `Arc`
        // either way, but forgetting it would make the run unpollable and unstoppable.
        while jobs.len() > KEEP {
            let Some(at) = jobs.iter().position(|j| {
                *j.state
                    .read()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    != State::Running
            }) else {
                break;
            };
            jobs.remove(at);
        }
        // Explicit, because the caller goes on to run minutes of work and holding the registry lock
        // across that would block every poll. `clippy::significant_drop_tightening` asks for it too.
        drop(jobs);
    }

    pub(crate) fn get(&self, id: u64) -> Option<Arc<Job>> {
        self.jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .find(|j| j.id == id)
            .map(Arc::clone)
    }

    /// Mark a job finished. `Err` becomes `Failed`; a cancelled job says so rather than reporting the
    /// abort as a failure, because it is not one.
    pub(crate) fn finish(job: &Job, outcome: anyhow::Result<()>) {
        match outcome {
            Ok(()) if job.cancelled() => job.finish(State::Cancelled, None),
            Ok(()) => job.finish(State::Done, None),
            // An abort surfaces as an error from deep inside the reader; it is still a cancellation.
            Err(_) if job.cancelled() => job.finish(State::Cancelled, None),
            Err(e) => job.finish(State::Failed, Some(format!("{e:#}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_job_reports_its_progress_and_findings_as_they_land() {
        let jobs = Jobs::default();
        let job = jobs.start("verify-repack");
        assert_eq!(job.id, 1);

        let snap = job.snapshot();
        assert_eq!(snap["state"], "running");
        assert_eq!(snap["total"], 0, "unknown until the work says");
        assert_eq!(snap["findings"].as_array().map(Vec::len), Some(0));

        job.set_total(3);
        job.progress_to(1, "model.layers.1.w");
        job.set_bytes(4096);
        job.add_finding(json!({ "name": "model.layers.1.w", "differing": 0 }));
        let snap = job.snapshot();
        assert_eq!(snap["done"], 1);
        assert_eq!(snap["total"], 3);
        assert_eq!(snap["current"], "model.layers.1.w");
        assert_eq!(snap["bytes"], 4096);
        // The first finding is often the answer; waiting for the run to end to see it is what polling
        // exists to avoid.
        assert_eq!(snap["findings"].as_array().map(Vec::len), Some(1));
    }

    #[test]
    fn a_finished_job_stays_pollable() {
        let jobs = Jobs::default();
        let job = jobs.start("verify-repack");
        Jobs::finish(&job, Ok(()));
        let held = jobs.get(job.id).expect("a finished job is still there");
        assert_eq!(held.snapshot()["state"], "done");
    }

    /// A cancelled job is not a failed one — it says what happened, so a UI does not show an error for
    /// something the reader asked for.
    #[test]
    fn cancelling_is_reported_as_cancelled_not_failed() {
        let jobs = Jobs::default();
        let job = jobs.start("verify-repack");
        job.read_progress().cancel();
        assert!(job.cancelled());
        // Even when the work reports the abort as an error, which is how it surfaces from the reader.
        Jobs::finish(
            &job,
            Err(anyhow::anyhow!("read stopped before it finished")),
        );
        let snap = job.snapshot();
        assert_eq!(snap["state"], "cancelled");
        assert_eq!(
            snap["error"],
            Value::Null,
            "a stop is not an error to report"
        );
    }

    #[test]
    fn a_failure_keeps_its_message() {
        let jobs = Jobs::default();
        let job = jobs.start("verify-repack");
        Jobs::finish(&job, Err(anyhow::anyhow!("no objects under that prefix")));
        let snap = job.snapshot();
        assert_eq!(snap["state"], "failed");
        assert!(
            snap["error"].as_str().is_some_and(|e| e.contains("prefix")),
            "the message is what the UI shows: {snap}"
        );
    }

    #[test]
    fn ids_are_distinct_and_lookup_misses_are_none() {
        let jobs = Jobs::default();
        let a = jobs.start("verify-repack");
        let b = jobs.start("verify-repack");
        assert_ne!(a.id, b.id);
        assert!(jobs.get(9999).is_none());
    }

    /// Old *finished* jobs are dropped; a running one never is, or its run would become unpollable and
    /// unstoppable.
    #[test]
    fn the_registry_evicts_finished_jobs_but_never_a_running_one() {
        let jobs = Jobs::default();
        let running = jobs.start("verify-repack");
        for _ in 0..KEEP + 4 {
            let j = jobs.start("verify-repack");
            Jobs::finish(&j, Ok(()));
        }
        assert!(
            jobs.get(running.id).is_some(),
            "the running job must survive eviction"
        );
        let held = jobs
            .jobs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len();
        assert!(held <= KEEP + 1, "the registry should stay bounded: {held}");
    }
}
