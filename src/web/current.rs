//! The checkpoint **currently** being served, and how to change it without restarting.
//!
//! [`WebState`] is still read-once and immutable — that is what lets every handler take a
//! plain `&WebState` and lets a 14 MB `/api/tree` body be encoded once and handed out as
//! bytes. What changes is that the server no longer holds *one* of them for its lifetime: it
//! holds a cell containing the current one.
//!
//! ## Why a swap rather than mutation
//!
//! Each request takes a snapshot ([`Current::snapshot`]) — one `Arc` clone under a read
//! lock — and then works entirely from it. So:
//!
//! - A tensor scan already running when the checkpoint changes finishes against the
//!   checkpoint it started on, and answers about *that* one. It cannot half-see a new
//!   checkpoint, because nothing it holds was mutated.
//! - The read of the new checkpoint (seconds locally, longer over ssh) happens off to the
//!   side; the old state keeps serving until the moment the pointer moves. There is no window
//!   where the API is unavailable, and a failed open leaves the previous checkpoint intact.
//! - Every per-checkpoint cache — the encoded static bodies, the whole-tensor stats memo —
//!   lives *inside* `WebState`, so replacing it invalidates them by construction. That is
//!   the reason this is a swap and not a set of fields to reset: a cache that outlived the
//!   swap would serve the old checkpoint's tree under the new one's name.
//!
//! One current checkpoint, not a registry: browser tabs share it, and switching back re-reads
//! (~2 s for a 31k-tensor local checkpoint). The alternative — keeping several resident —
//! costs ~110 MB each and makes every endpoint grow an optional "which checkpoint" parameter.

use std::net::IpAddr;
use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Result, bail};

use super::WebState;
use crate::opening::{self, Opened};

/// The state cell plus everything needed to read a *different* checkpoint later.
pub(crate) struct Current {
    /// The checkpoint being served. `RwLock<Arc<_>>` rather than `Arc<RwLock<_>>`: readers
    /// clone the `Arc` and drop the guard immediately, so a request never holds a lock
    /// across its own work, and a multi-second scan cannot block a swap.
    state: RwLock<Arc<WebState>>,
    /// Held for the duration of an open, so two simultaneous requests cannot both read and
    /// then race to install. The loser is told to wait rather than queued: an open takes
    /// seconds, and a queued second one would swap the checkpoint out from under whoever
    /// asked first.
    opening: Mutex<()>,
    /// The read the open slot is busy with — published so a second request can describe it and, if
    /// asked, stop it.
    ///
    /// A second field rather than data inside `opening`, because the whole point is to be readable
    /// *while* the lock is held, which a `Mutex`'s contents are not. [`Reading`] keeps the two in
    /// step: it is set when the lock is taken and cleared when the guard drops, so there is no path
    /// that leaves this describing a read that has finished.
    in_flight: RwLock<Option<InFlight>>,
    recents: Mutex<opening::Recents>,
    /// The host a `:PATH` spec would resolve to — this server's own proxy, else the `--ssh-proxy`
    /// flag, else the config file's `ssh_proxy`.
    ///
    /// Not the same question as "was this server started on a remote checkpoint". A server serving a
    /// *local* checkpoint still resolves `:PATH` through the configured proxy, because
    /// [`opening::resolve`] reads the config on every open — so answering "is this proxied" from
    /// `remote` alone told the UI `false` while the shorthand worked, and left it unable to name the
    /// host in a label it was already showing. Resolved once here rather than per request, since it is
    /// fixed for the process.
    proxy: Option<String>,
    /// The flags this process started with, so a checkpoint opened at runtime is read the
    /// same way the one on the command line was.
    opts: opening::Options,
    /// The bind address, to re-derive the access warning for each newly built state.
    host: IpAddr,
    /// The comparison that is set up, if any — **one** field, so there is no state in which half of
    /// it is installed.
    ///
    /// Each resident side costs what the served checkpoint does (~110 MB for a 31k-tensor
    /// checkpoint), so a comparison is opt-in and droppable.
    comparison: RwLock<Option<Comparison>>,
    /// Hands out [`Comparison::id`]s. Monotonic for the life of the process, so an id is never reused
    /// and a stale one can always be told apart from a current one.
    next_comparison: std::sync::atomic::AtomicU64,
    /// Long-running work — the value-reading diff modes, which take minutes and report findings as
    /// they land. See [`super::jobs`].
    jobs: super::jobs::Jobs,
}

/// The pair a comparison is between, with an identity.
///
/// **Why an id.** There is one comparison slot for the whole server, and `/api/difftree` used to take
/// no parameters and answer straight from it. Two overlapping clients therefore received each other's
/// results: A set up its pair, B replaced it, and A's `GET` returned B's comparison with a `200` — a
/// confident wrong answer, where the behaviour it replaced was at least a hard error. Every set-up now
/// takes a fresh id, `/api/difftree` requires the id its caller was given, and a mismatch is a `409`.
/// Contention still costs a retry; it can no longer cost a plausible answer about the wrong pair.
struct Comparison {
    id: u64,
    /// The baseline — the `OLD` of `diff OLD NEW`.
    left: Arc<WebState>,
    /// The newer side, or `None` when it *is* the served checkpoint. That is the common case and costs
    /// no second read; naming a different one fills this instead, so a comparison is two specs
    /// independent of what the rest of the app is browsing.
    right: Option<Arc<WebState>>,
}

/// The answer to "give me the comparison with this id".
///
/// Three arms rather than an `Option`, because "there is no comparison" and "there is one, but it is
/// not yours" call for different things from the caller — set one up, versus retry because someone
/// else got in first. Flattening those together is what let a client be handed the wrong pair.
pub(crate) enum ComparisonLookup {
    /// Nothing is set up.
    None,
    /// Something is, but a later request replaced what this caller was told about.
    Replaced { current: u64 },
    /// The caller's pair: the baseline, and the newer side resolved to the served checkpoint when it
    /// was not named.
    Found {
        base: Arc<WebState>,
        right: Arc<WebState>,
    },
}

/// What [`Current::set_comparison`] hands back: the identity to quote on the follow-up request, and
/// the two specs **as they resolved**, so a client can check that the comparison it receives is the
/// one it asked for without re-deriving a resolution only the server performed.
pub(crate) struct ComparisonSet {
    pub id: u64,
    pub left_spec: String,
    pub right_spec: String,
}

impl Current {
    /// Install the checkpoint read at startup.
    ///
    /// `host` is the bind address, kept rather than applied once: it decides the
    /// no-access-control caution the UI shows, and every state built by a later switch needs
    /// it too. Taking it here (instead of a `with_exposure` afterwards) means there is never
    /// a moment where the served state has the wrong answer about its own exposure.
    /// `recents` is passed in rather than built here so a test can hold an in-memory list while
    /// the real server holds the user's persistent one.
    pub(crate) fn new(
        opened: Opened,
        remote: Option<crate::remote::RemoteRead>,
        opts: opening::Options,
        host: IpAddr,
        mut recents: opening::Recents,
    ) -> Result<Self> {
        // The durable spelling of what was opened: recorded in the list *and* served as the
        // checkpoint's address, which is not the same string as its display root.
        let remembered = opened.target.recorded_spec(&opened.target.spec());
        recents.record(&remembered);
        // Whichever proxy a later `:PATH` would go through: this server's own remote, then the
        // `--ssh-proxy` flag, then the config file — the same precedence `opening::resolve` applies.
        let proxy = remote
            .as_ref()
            .map(|r| r.host.clone())
            .or_else(|| opts.proxy.clone())
            .or_else(|| crate::cli_config::CliConfig::load().ssh_proxy);
        Ok(Self {
            state: RwLock::new(Arc::new(state_from(opened, host, remembered)?)),
            opening: Mutex::new(()),
            in_flight: RwLock::new(None),
            recents: Mutex::new(recents),
            proxy,
            opts,
            host,
            comparison: RwLock::new(None),
            next_comparison: std::sync::atomic::AtomicU64::new(1),
            jobs: super::jobs::Jobs::default(),
        })
    }

    /// The checkpoint to answer this request from. Cheap: one `Arc` clone.
    pub(crate) fn snapshot(&self) -> Arc<WebState> {
        Arc::clone(
            &self
                .state
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        )
    }

    /// Most-recently-opened first, for the open prompt's list.
    pub(crate) fn recents(&self) -> Vec<String> {
        self.recents
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .list()
    }

    /// Take the single read slot, publishing what it is being taken for.
    ///
    /// `None` when another read holds it; the caller turns that into the 409 with
    /// [`Self::busy_message`]. `try_lock`, not `lock`: a caller told "something else is reading" can
    /// decide to retry, where one silently queued behind a 30 s remote read cannot tell why it hangs.
    fn begin_read(&self, spec: &str) -> Option<Reading<'_>> {
        let guard = self.opening.try_lock().ok()?;
        let progress = Arc::new(crate::hf::ReadProgress::default());
        *self
            .in_flight
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(InFlight {
            spec: spec.to_string(),
            since: std::time::Instant::now(),
            progress: Arc::clone(&progress),
        });
        Some(Reading {
            current: self,
            progress,
            _guard: guard,
        })
    }

    /// Take the read slot, stopping whatever holds it first.
    ///
    /// What "Stop it and read this instead" does. The refusal it replaces was a dead end: the server
    /// reads one checkpoint at a time, so being told to *wait* left the only way forward — abandoning a
    /// read nobody is waiting for any more — unavailable. Cancelling is cooperative, so this then waits
    /// for the slot rather than assuming it is free; if the loser does not let go it is reported rather
    /// than waited on forever.
    fn begin_read_taking_over(&self, spec: &str) -> Result<Reading<'_>> {
        let stopped = {
            let held = self
                .in_flight
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // Nothing to stop leaves the name empty and falls through to taking the slot normally.
            held.as_ref().map_or_else(String::new, |f| {
                f.progress.cancel();
                f.spec.clone()
            })
        };
        // Poll rather than `lock()`: a read that ignores the flag would otherwise hang this request
        // (and its worker) indefinitely, which is the failure mode `try_lock` exists to avoid.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            if let Some(reading) = self.begin_read(spec) {
                return Ok(reading);
            }
            if std::time::Instant::now() >= deadline {
                bail!(
                    "asked {} to stop, and it has not let go after 20s — it may be blocked on the \
                     network; try again shortly",
                    if stopped.is_empty() {
                        "the running read"
                    } else {
                        &stopped
                    }
                );
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    /// Take the read slot the way `busy` asks for.
    fn take_slot(&self, spec: &str, busy: WhenBusy) -> Result<Reading<'_>> {
        match busy {
            WhenBusy::StopTheOther => self.begin_read_taking_over(spec),
            WhenBusy::Refuse => self.begin_read(spec).ok_or_else(|| {
                let (held, secs) = self
                    .busy_with()
                    .unwrap_or_else(|| ("another checkpoint".to_string(), 0.0));
                // Says what is running, for how long, and that stopping it is an option — the caller
                // turns this into a 409 the UI can offer a button for.
                anyhow::anyhow!("{held} is being read ({secs:.0}s so far)")
            }),
        }
    }

    /// What is holding the read slot, for a caller that has to explain the refusal.
    ///
    /// `None` when the lock was contended but freed between the two operations — rare, and not worth a
    /// second lock to make impossible.
    pub(crate) fn busy_with(&self) -> Option<(String, f64)> {
        self.in_flight
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .map(|f| (f.spec.clone(), f.since.elapsed().as_secs_f64()))
    }

    /// The read in flight, in as much detail as it reports: what it is reading, for how long, how far
    /// it has got, and what it is counting.
    ///
    /// The browser's wait had an elapsed timer and nothing else, while a terminal reading the same
    /// checkpoint showed `1155/1155 S3 objects · reading S3 storage metadata`. The numbers existed the
    /// whole time — they were being animated onto the log of a process nobody was watching. This is the
    /// channel that gets them to whoever is actually waiting (`GET /api/reading`).
    pub(crate) fn reading(&self) -> Option<ReadingProgress> {
        // The counters are `Arc`-shared with the read itself, so taking a handle is all this needs the
        // lock for — and it releases it before reading them. Polled several times a second while a read
        // is running, and the read takes the same lock when it finishes.
        let held = self
            .in_flight
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let f = held.as_ref()?;
        let (spec, since, progress) = (f.spec.clone(), f.since, Arc::clone(&f.progress));
        drop(held);

        let load = progress.load();
        let (done, total) = load.snapshot();
        Some(ReadingProgress {
            spec,
            seconds: since.elapsed().as_secs_f64(),
            done,
            total,
            unit: load.unit_label().trim().to_string(),
            stage: load.stage().map(|s| s.note().to_string()),
        })
    }

    /// Read the checkpoint at `spec` and make it the one being served.
    ///
    /// Returns the new root on success. On failure **nothing changes** — the read happens in
    /// full before the pointer moves, so a typo leaves the browser on the checkpoint it was
    /// already showing rather than on an error page with no data behind it.
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the read slot is held for the whole read on purpose — that is what serialises reads; \
                  releasing it as soon as `progress` has been taken would let a second read start"
    )]
    pub(crate) fn open(&self, spec: &str, busy: WhenBusy) -> Result<Arc<WebState>> {
        let reading = self.take_slot(spec, busy)?;
        let opened = opening::resolve(spec, &self.opts)?
            // The slot's own handle, so a later "stop it" reaches this read rather than a throwaway.
            .read(opening::Want::Model, &reading.progress)?;
        // Record the durable spelling of what opened, not the string typed: a relative path
        // becomes absolute and a proxied path becomes `host:/path`, so the entry still names
        // this checkpoint from another directory or another config (see `recorded_spec`).
        let remembered = opened.target.recorded_spec(spec);
        let state = Arc::new(state_from(opened, self.host, remembered.clone())?);
        // Record only what actually opened, so the list can't fill with typos.
        self.recents
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .record(&remembered);
        *self
            .state
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Arc::clone(&state);
        Ok(state)
    }

    /// Drop a checkpoint from the recents list. Returns whether it was there.
    ///
    /// Only the list is touched: the checkpoint itself is not, and neither is what is being
    /// served — forgetting the one you are looking at is allowed, and leaves you looking at it.
    pub(crate) fn forget_recent(&self, spec: &str) -> bool {
        self.recents
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .forget(spec)
    }

    /// The comparison the caller quoted, as `(baseline, newer side)`.
    ///
    /// Three answers, deliberately distinguished: no comparison at all, one whose id is not the
    /// caller's, and the pair. Collapsing the middle case into either of the others is the bug this
    /// replaced — answering from "whatever is in the slot" handed one client another's comparison.
    pub(crate) fn comparison_for(&self, id: u64) -> ComparisonLookup {
        let held = self
            .comparison
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match held.as_ref() {
            None => ComparisonLookup::None,
            Some(c) if c.id != id => ComparisonLookup::Replaced { current: c.id },
            Some(c) => ComparisonLookup::Found {
                base: Arc::clone(&c.left),
                // The newer side is the named one, else whatever is being served *now*.
                right: c.right.clone().unwrap_or_else(|| self.snapshot()),
            },
        }
    }

    /// Set up a comparison between two specs.
    ///
    /// `right` may be empty, or name the served checkpoint, in which case no second read happens and
    /// the served one is used — the common case, and the one that costs nothing.
    ///
    /// Shares the open lock with [`Self::open`]: these are multi-second reads that install a
    /// checkpoint, and letting them run at once would have reads competing for the same disk while
    /// the caller cannot tell which one it is waiting for.
    #[expect(
        clippy::significant_drop_tightening,
        reason = "the read slot is held for the whole read on purpose — that is what serialises reads; \
                  releasing it as soon as `progress` has been taken would let a second read start"
    )]
    pub(crate) fn set_comparison(
        &self,
        left: &str,
        right: &str,
        busy: WhenBusy,
    ) -> Result<ComparisonSet> {
        let reading = self.take_slot(left, busy)?;
        let served = self.snapshot();
        let served_spec = served.spec.clone();
        // Drop whatever was set up before reading anything. A failed set-up used to leave the
        // *previous* comparison in the slot, so `/api/difftree` kept answering 200 with a pair
        // nobody asked for — a stale comparison served as if it were the requested one, which is
        // worse than the 409 that says there is none.
        self.clear_comparison();
        let (left_state, left_spec) = self.read_side(left, &reading.progress)?;
        // Only read the right side when it is genuinely a third checkpoint. Comparing against what
        // is already loaded should not cost a second copy of it.
        let right_read = match right.trim() {
            "" => None,
            r if r == served_spec => None,
            r => Some(self.read_side(r, &reading.progress)?),
        };
        let (right_state, right_spec) = match right_read {
            Some((state, spec)) => (Some(state), spec),
            // The newer side is the served checkpoint, so *its* address is what resolved — the client
            // needs the effective spec, not the empty string it sent.
            None => (None, served_spec),
        };
        let id = self
            .next_comparison
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        *self
            .comparison
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Comparison {
            id,
            left: left_state,
            right: right_state,
        });
        // Record only once the comparison is actually set up.
        //
        // Both sides are checkpoints you opened, so they belong in the list — but the baseline used to
        // be recorded the moment *it* read, which put it there even when the newer side then failed
        // and no comparison existed. The list is of addresses worth reopening; one that was part of a
        // comparison that never happened has not earned a place in it.
        {
            let mut recents = self
                .recents
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            recents.record(&left_spec);
            recents.record(&right_spec);
        }
        Ok(ComparisonSet {
            id,
            left_spec,
            right_spec,
        })
    }

    /// The job registry — started, polled and stopped through `/api/jobs/…`.
    pub(crate) fn jobs(&self) -> &super::jobs::Jobs {
        &self.jobs
    }

    /// The read switches this server started with, for a handler that has to resolve a *second*
    /// checkpoint — the diff report's baseline reads the same way the served one did.
    ///
    /// Handed out rather than wrapped in a `Current` method so `/api/diff` keeps the shape every
    /// other handler has (state, query, no socket, no lock), and stays as easy to unit-test.
    pub(crate) fn read_options(&self) -> &opening::Options {
        &self.opts
    }

    /// Read one side of a comparison, with the durable spelling of what opened.
    ///
    /// Does not touch the recents list — see [`Self::set_comparison`], which records both sides
    /// together once there is a comparison for them to be part of.
    fn read_side(
        &self,
        spec: &str,
        progress: &crate::hf::ReadProgress,
    ) -> Result<(Arc<WebState>, String)> {
        let opened = opening::resolve(spec, &self.opts)?.read(opening::Want::Model, progress)?;
        let remembered = opened.target.recorded_spec(spec);
        let state = Arc::new(state_from(opened, self.host, remembered.clone())?);
        Ok((state, remembered))
    }

    /// Drop the comparison, freeing whatever it held. Idempotent.
    pub(crate) fn clear_comparison(&self) {
        *self
            .comparison
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    /// Whether a `:PATH` spec would resolve here — which is what a client needs to know to say what
    /// kind of address its prompts accept.
    ///
    /// Answered from [`Self::proxy`], not from `remote`: a server serving a local checkpoint still
    /// resolves the shorthand through the configured proxy, so `remote.is_some()` was the wrong
    /// question and reported `false` for a server where `:PATH` worked.
    pub(crate) fn is_proxied(&self) -> bool {
        self.proxy.is_some()
    }

    /// The proxy host, when there is one.
    ///
    /// Served because the client cannot work it out: `:/path` means "on whatever `ssh_proxy`
    /// names", and only this side has read the config. Without it the loading screen could only
    /// echo the `:` back, which names the checkpoint to nobody who does not already know the
    /// config's contents.
    pub(crate) fn proxy_host(&self) -> Option<&str> {
        self.proxy.as_deref()
    }
}

/// What to do when another read already holds the slot.
///
/// An enum rather than a `bool`, because the two arms are different *behaviours* with different failure
/// modes — refuse immediately, or stop the incumbent and wait for it — and a bare `true` at a call site
/// says neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WhenBusy {
    /// Leave the running read alone and refuse, so the caller can offer the choice.
    Refuse,
    /// Ask the running read to stop, then take the slot when it lets go.
    StopTheOther,
}

/// How far the read in flight has got, for whoever is waiting on it.
///
/// `total == 0` means the reader has not learned a denominator yet — a spinner, not a bar at zero, the
/// same rule the terminal's bars use. `unit` and `stage` are empty until the reader says what it is
/// counting and which step it is on; both come from the reader itself rather than being guessed from
/// the spec, because only the reader knows whether it is listing files or `HEAD`ing objects.
#[derive(serde::Serialize)]
pub(crate) struct ReadingProgress {
    pub spec: String,
    pub seconds: f64,
    pub done: usize,
    pub total: usize,
    /// `shards`, `S3 objects`, `tensors` — what the count counts.
    pub unit: String,
    /// `reading S3 storage metadata`, `listing the checkpoint files` — which step is running.
    pub stage: Option<String>,
}

/// A read in progress: what it is reading, since when, and how to ask it to stop.
struct InFlight {
    spec: String,
    since: std::time::Instant,
    /// The read's own progress-and-control handle. Cancelling through it tears down the remote
    /// command, so a `Stop it` really stops rather than merely stopping the *waiting*.
    progress: Arc<crate::hf::ReadProgress>,
}

/// The single read slot, held.
///
/// Exists so the lock and the "what is being read" announcement cannot disagree: the announcement is
/// written when this is created and cleared when it drops, including on the `?` of a failed read. Two
/// bare fields updated by hand would have had a path — an early return — that left the server claiming
/// to be busy with a read that had already failed.
struct Reading<'a> {
    current: &'a Current,
    /// This read's progress-and-control handle, shared with `in_flight` so a later request can cancel
    /// it. The read itself is handed this, not a throwaway.
    progress: Arc<crate::hf::ReadProgress>,
    _guard: std::sync::MutexGuard<'a, ()>,
}

impl Drop for Reading<'_> {
    fn drop(&mut self) {
        *self
            .current
            .in_flight
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }
}

/// Build the served state from a read, insisting on the model the API serves.
///
/// [`opening::Want::Model`] promises one, so `None` here is a broken contract rather than a
/// user error — but it is still reported instead of unwrapped, because the alternative is a
/// panicking worker thread.
fn state_from(opened: Opened, host: IpAddr, spec: String) -> Result<WebState> {
    let Some(model) = opened.checkpoint else {
        bail!(
            "{}: read produced no model to serve",
            crate::model::root_label(&opened.target.requested)
        );
    };
    Ok(
        WebState::build(model, &opened.target.resolved, &opened.target.index_specs)
            .with_exposure(host)
            .with_spec(spec),
    )
}
