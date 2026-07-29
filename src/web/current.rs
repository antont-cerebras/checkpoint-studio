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
    recents: Mutex<opening::Recents>,
    /// The proxy to read further checkpoints over, when this server was started with one.
    /// A `--ssh-proxy` server opens remote checkpoints; a local one opens local paths.
    remote: Option<crate::remote::RemoteRead>,
    /// The flags this process started with, so a checkpoint opened at runtime is read the
    /// same way the one on the command line was.
    opts: opening::Options,
    /// The bind address, to re-derive the access warning for each newly built state.
    host: IpAddr,
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
        Ok(Self {
            state: RwLock::new(Arc::new(state_from(opened, host, remembered)?)),
            opening: Mutex::new(()),
            recents: Mutex::new(recents),
            remote,
            opts,
            host,
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

    /// Read the checkpoint at `spec` and make it the one being served.
    ///
    /// Returns the new root on success. On failure **nothing changes** — the read happens in
    /// full before the pointer moves, so a typo leaves the browser on the checkpoint it was
    /// already showing rather than on an error page with no data behind it.
    pub(crate) fn open(&self, spec: &str) -> Result<Arc<WebState>> {
        // `try_lock`, not `lock`: see the field docs. A caller told "an open is already in
        // progress" can retry; one silently queued behind a 10 s ssh read cannot tell why it
        // is hanging.
        let Ok(_guard) = self.opening.try_lock() else {
            bail!("another checkpoint is already being opened — retry when it finishes");
        };
        let opened = opening::resolve(spec, &self.opts)?
            .read(opening::Want::Model, &crate::hf::ReadProgress::default())?;
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

    /// Whether this server reads over an ssh proxy — which is what a client needs to know to
    /// say what kind of path its open prompt accepts.
    pub(crate) fn is_proxied(&self) -> bool {
        self.remote.is_some()
    }

    /// The proxy host, when there is one.
    ///
    /// Served because the client cannot work it out: `:/path` means "on whatever `ssh_proxy`
    /// names", and only this side has read the config. Without it the loading screen could only
    /// echo the `:` back, which names the checkpoint to nobody who does not already know the
    /// config's contents.
    pub(crate) fn proxy_host(&self) -> Option<&str> {
        self.remote.as_ref().map(|r| r.host.as_str())
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
