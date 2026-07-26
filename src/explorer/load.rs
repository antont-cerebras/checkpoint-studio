//! Reading a checkpoint into the explorer: opening readers, loading shard headers
//! (locally or over an ssh proxy), the background statistics scan, and the derived
//! per-tensor view state that depends on them.
//!
//! Split out of `explorer/mod.rs` as a second `impl Explorer` block — Rust allows
//! inherent impls across modules of one crate, and a child module can still reach its
//! parent's private fields, so this needs no widening of `Explorer`'s internals.

#[allow(clippy::wildcard_imports)] // a submodule of the module it was split from
use super::*;

impl Explorer {
    /// Run `f` with an open reader for `t`, reusing the cached one when it is
    /// still for the same tensor and opening (and caching) a fresh one otherwise.
    /// Lets the data view re-sample on every pan / slice step without paying the
    /// file-open cost each frame.
    pub(super) fn with_reader<R>(
        &self,
        t: &TensorInfo,
        f: impl FnOnce(&dyn crate::sample::TensorReader) -> Result<R, String>,
    ) -> Result<R, String> {
        {
            let mut cache = self.reader_cache.borrow_mut();
            let stale = cache
                .as_ref()
                .is_none_or(|c| c.source_path != t.source_path || c.name != t.name);
            if stale {
                let reader = crate::sample::open_reader(t)?;
                *cache = Some(CachedReader {
                    source_path: t.source_path.clone(),
                    name: t.name.clone(),
                    reader,
                });
            }
        }
        let cache = self.reader_cache.borrow();
        let Some(cached) = cache.as_ref() else {
            // The block above either found a live reader or replaced it, so this can only
            // be `None` if something cleared the cache in between — nothing does, and an
            // error beats a panic if that ever changes.
            return Err("the tensor reader vanished from the cache".to_string());
        };
        f(cached.reader.as_ref())
    }

    /// Cached exact statistics for `(tensor, view)`, or `None` if not yet
    /// computed (cheap lookup — never scans).
    pub(super) fn cached_stats(&self, tensor: &TensorInfo, view: ViewDtype) -> Option<Stats> {
        self.stats_cache
            .borrow()
            .get(&(tensor.name.clone(), view))
            .copied()
    }

    /// Start a statistics scan for `(tensor, view)` on a worker thread. Used by
    /// the data view, which polls the returned [`ScanJob`] and stays interactive
    /// while it runs (see [`Self::run_data`]).
    pub(super) fn spawn_stats_scan(&self, tensor: &TensorInfo, view: ViewDtype) -> ScanJob {
        let cancel = Arc::new(AtomicBool::new(false));
        let pause = Arc::new(AtomicBool::new(false));
        let done = Arc::new(AtomicUsize::new(0));
        let owned = tensor.clone();
        let schema = self.schema_for(&tensor.name).cloned();
        let worker_cancel = Arc::clone(&cancel);
        let worker_pause = Arc::clone(&pause);
        let worker_done = Arc::clone(&done);
        let handle = std::thread::spawn(move || {
            crate::sample::tensor_stats(
                &owned,
                view,
                schema.as_ref(),
                &worker_cancel,
                &worker_pause,
                Some(&*worker_done),
            )
        });
        ScanJob {
            view,
            cancel,
            pause,
            handle: Some(handle),
            started: std::time::Instant::now(),
            done,
            total: tensor.size_bytes,
        }
    }

    /// Compute and cache exact statistics for `(tensor, view)` on a miss. The
    /// scan runs on a worker thread; while it runs, `redraw` is called each frame
    /// with a [`StatsView::Computing`] state so the caller can animate a spinner
    /// *in place* on its own screen. Ctrl-C quits the app; **any other key
    /// cancels** the scan (the worker stops at the next block) and returns
    /// [`ScanOutcome::Cancelled`] right away, so a slow scan never traps the UI.
    /// Small tensors finish before the spinner ever appears.
    pub(super) fn compute_stats_animated(
        &self,
        term: &mut crate::tui::LiveTerminal,
        tensor: &TensorInfo,
        view: ViewDtype,
        render: impl Fn(&mut ratatui::Frame, StatsView<'_>),
    ) -> ScanOutcome {
        const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
        if self.cached_stats(tensor, view).is_some() {
            return ScanOutcome::Completed;
        }

        // The same worker the data view spawns; the difference here is that this screen
        // animates it in place and waits for it, rather than polling it between frames.
        // `job.cancel` lets a key press abort cooperatively: we set it and return without
        // joining, so the UI stays responsive and the worker winds down at its next block
        // boundary. (`ScanJob`'s own `Drop` sets it too, which is what stops the worker if
        // this returns early.)
        let mut job = self.spawn_stats_scan(tensor, view);
        let (cancel, done, total, started) = (
            Arc::clone(&job.cancel),
            Arc::clone(&job.done),
            job.total,
            job.started,
        );
        let Some(handle) = job.handle.take() else {
            return ScanOutcome::Completed;
        };

        let mut frame = 0usize;
        while !handle.is_finished() {
            // Only animate once it's clearly not instant, to avoid a flash for
            // small tensors (which return before the first frame).
            if started.elapsed() >= std::time::Duration::from_millis(120) {
                let sv = StatsView::Computing {
                    spinner: SPINNER[frame % SPINNER.len()],
                    elapsed: started.elapsed(),
                    progress: (total > 0)
                        .then(|| (done.load(Ordering::Relaxed) as f64 / total as f64).min(1.0)),
                };
                let _ = term.draw(|f| render(f, sv));
                frame += 1;
            }
            // Frame delay that also stays responsive to keys while we wait.
            if event::poll(std::time::Duration::from_millis(80)).unwrap_or(false)
                && let Ok(Event::Key(key)) = event::read()
            {
                if is_ctrl_c(&key) {
                    quit_immediately();
                }
                cancel.store(true, Ordering::Relaxed);
                return ScanOutcome::Cancelled;
            }
        }

        match handle.join() {
            Ok(Ok(s)) => {
                self.stats_cache
                    .borrow_mut()
                    .insert((tensor.name.clone(), view), s);
                ScanOutcome::Completed
            }
            // Surface a failure instead of silently doing nothing.
            Ok(Err(msg)) => {
                let _ = term.draw(|f| UI::render_message(f, "Statistics unavailable", &msg));
                let _ = event::read();
                ScanOutcome::Completed
            }
            Err(_) => {
                let _ = term.draw(|f| {
                    UI::render_message(f, "Statistics unavailable", "the scan thread panicked");
                });
                let _ = event::read();
                ScanOutcome::Completed
            }
        }
    }

    /// Read `s3://…` sources' metadata over SSH via cstorch on `host` (activating
    /// the venv at `venv`), instead of directly — so credentials stay on the
    /// remote (`--ssh-proxy` / `--ssh-venv`).
    pub(crate) fn set_remote_read(&mut self, host: String, venv: String) {
        // The file browser adapts to the remote source kind, derived from the raw
        // `--ssh-proxy` argument: an `s3://…` URI browses s3-natively; any other
        // path is the SFTP directory to browse (or, for a single shard, its
        // parent). `browse_root` (a local `.parent()`) doesn't apply remotely.
        let browse = self.files.first().map(|first| {
            let src = first.to_string_lossy().into_owned();
            if src.starts_with("s3://") {
                RemoteBrowse::S3(src)
            } else if src.ends_with(".safetensors") {
                let dir = src
                    .rsplit_once('/')
                    .map_or_else(|| ".".to_string(), |(d, _)| d.to_string());
                RemoteBrowse::Sftp(dir)
            } else {
                RemoteBrowse::Sftp(src.trim_end_matches('/').to_string())
            }
        });
        self.remote = Some(RemoteContext {
            read: crate::remote::RemoteRead::new(host, venv),
            browse,
            session: RefCell::new(None),
            password: RefCell::new(None),
            disk: None,
            s3_meta: None,
        });
    }

    /// The remote reader (`--ssh-proxy`), when this is a remote run.
    pub(super) fn remote_read(&self) -> Option<&crate::remote::RemoteRead> {
        self.remote.as_ref().map(|r| &r.read)
    }

    /// The file browser's remote source kind, when browsing a remote checkpoint.
    pub(super) fn remote_browse(&self) -> Option<&RemoteBrowse> {
        self.remote.as_ref().and_then(|r| r.browse.as_ref())
    }

    /// The captured remote on-disk usage, if any.
    pub(super) fn remote_disk(&self) -> Option<crate::stats::DiskUsage> {
        self.remote.as_ref().and_then(|r| r.disk.clone())
    }

    /// The remote S3 object metadata, if any.
    pub(super) fn remote_s3_meta(&self) -> Option<&crate::remote::S3Meta> {
        self.remote.as_ref().and_then(|r| r.s3_meta.as_ref())
    }

    /// Run `f` with the live remote session, reopening once (with the cached
    /// password) if the stored session errors — a `--ssh-proxy` connection can idle
    /// out between the initial read and a later browse. All remote file-browser /
    /// layout / sidecar reads go through this so they never re-prompt. Errors only
    /// when there's no `--ssh-proxy` configured or the (re)open itself fails.
    pub(super) fn with_remote_session<T>(
        &self,
        f: impl Fn(&crate::sftp::RemoteSession) -> Result<T>,
    ) -> Result<T> {
        let rc = self
            .remote
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no --ssh-proxy session configured"))?;
        // Try the stored session first; a success returns immediately.
        if let Some(session) = rc.session.borrow().as_ref()
            && let Ok(v) = f(session)
        {
            return Ok(v);
        }
        // No session, or it errored (likely an idle timeout): reopen once with the
        // cached password (entered during the pre-TUI read, so no new prompt).
        let session = {
            let mut pw = rc.password.borrow_mut();
            rc.read.open_with(&mut pw)?
        };
        let out = f(&session);
        *rc.session.borrow_mut() = Some(session);
        out
    }

    pub(super) fn load_all_files(&mut self) -> Result<()> {
        // Already loaded (e.g. a remote `--ssh-proxy` structure read synchronously
        // before the TUI started) — don't re-read.
        if self.full_loaded {
            return Ok(());
        }

        // Read the checkpoint structure on a worker thread so the UI stays
        // responsive: a cold file (e.g. a large HDF5 over the network) can take
        // seconds to enumerate, and we'd otherwise show an empty screen. Animate
        // a loading frame — the same header/footer chrome as the tree, with a
        // spinner in place of the rows — until the worker finishes.
        let files = self.files.clone();
        let remote = self.remote_read().cloned();
        let handle = std::thread::spawn(move || Self::gather_checkpoint(&files, remote.as_ref()));

        let label = self
            .files
            .first()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let total = self.files.len();
        let started = std::time::Instant::now();
        let mut frame = 0usize;
        loop {
            // Wait one ~12 fps tick — also catches `q` / Ctrl-C to abort. Polling
            // *before* drawing means a fast (cached) load finishes within the
            // first tick and never flashes the spinner; only a slow load reaches
            // the draw below.
            if event::poll(std::time::Duration::from_millis(80)).unwrap_or(false)
                && let Ok(Event::Key(key)) = event::read()
                && (is_ctrl_c(&key) || matches!(key.code, KeyCode::Char('q')))
            {
                quit_immediately();
            }
            if handle.is_finished() {
                break;
            }
            // Animate through the live terminal when one is up (the interactive
            // session); headless `--plain` uses `load_quiet` and never gets here.
            if let Some(term) = self.terminal.as_mut() {
                let spinner = STATS_SPINNER[frame % STATS_SPINNER.len()];
                let elapsed = started.elapsed();
                let _ = term.draw(|f| UI::render_loading(f, &label, total, spinner, elapsed));
            }
            frame += 1;
        }
        let ((tensors, metadata, config, disk, health), checkpoint) = handle
            .join()
            .map_err(|_| anyhow::anyhow!("checkpoint loader thread panicked"))??;
        if let Some(rc) = self.remote.as_mut() {
            rc.disk = disk;
        }
        // Remote index/file health (empty for a local read, whose reports were
        // gathered up front); fold it in so the popup and `⚠ health` badge show it.
        self.health_reports.extend(health);
        self.finalize_load(tensors, metadata, config, checkpoint);
        Ok(())
    }

    /// Read the checkpoint structure synchronously, with no loading animation —
    /// for `--plain`, which renders once to a buffer and must not write spinner
    /// frames to stdout.
    pub(super) fn load_quiet(&mut self) -> Result<()> {
        // A remote read opens one SSH session and keeps it alive on `self`, so the
        // file browser (and remote layout / sidecar reads) reuse it without a
        // second auth prompt. A local read uses the plain static gatherer.
        let ((tensors, metadata, config, disk, health), checkpoint) =
            if self.remote_read().is_some() {
                self.gather_remote_keeping_session()?
            } else {
                Self::gather_checkpoint(&self.files, None)?
            };
        if let Some(rc) = self.remote.as_mut() {
            rc.disk = disk;
        }
        self.health_reports.extend(health);
        self.finalize_load(tensors, metadata, config, checkpoint);
        Ok(())
    }

    /// Remote (`--ssh-proxy`) structure read that **keeps** the SSH session alive on
    /// `self` — stashed in [`Self::remote_session`], its password in
    /// [`Self::remote_password`] — so the file browser and remote layout / sidecar
    /// reads reuse it without re-prompting. Mirrors
    /// [`crate::remote::RemoteRead::fetch_with_config`] but over one owned session.
    pub(super) fn gather_remote_keeping_session(
        &mut self,
    ) -> Result<(CheckpointParts, Option<crate::model::Checkpoint>)> {
        // Bind the remote context once. Re-reading `self.remote` at each use meant three
        // separate assertions that it is `Some` in a function whose whole purpose is the
        // remote read; the caller's contract is now stated once, as an error.
        let Some(remote) = self.remote.as_ref() else {
            return Err(anyhow::anyhow!(
                "reading over ssh needs --ssh-proxy (no remote is configured)"
            ));
        };
        let r = remote.read.clone();
        // One authenticated session for the whole run; the password entered here is
        // cached (any later reopen after an idle timeout reuses it silently).
        let session = {
            let mut pw = remote.password.borrow_mut();
            r.open_with(&mut pw)?
        };
        eprintln!("checkpoint-studio: reading tensor metadata over ssh …");

        let mut tensors: Vec<TensorInfo> = Vec::new();
        let mut metadata: Vec<MetadataInfo> = Vec::new();
        let mut config: Option<crate::config::ModelConfig> = None;
        let mut disk_shards: Vec<crate::stats::ShardDisk> = Vec::new();
        let mut health: Vec<crate::health::HealthReport> = Vec::new();
        let mut s3_meta: Option<crate::remote::S3Meta> = None;
        for file_path in &self.files {
            let as_str = file_path.to_string_lossy().into_owned();
            let bars = crate::progress::Bars::start(std::slice::from_ref(&as_str));
            let progress = bars.progress(0);
            // Fetch the S3 object metadata up front for an `s3://` source (checksums
            // /ETags/tags — a HEAD per object), so the stats report's S3 section is
            // ready; `want_s3` is a no-op for an SFTP source.
            let pw = remote.password.borrow().clone();
            let out = r.read(
                &session,
                &as_str,
                &pw,
                progress.as_deref(),
                crate::remote::ObjectMeta::Fetch,
                None,
            );
            bars.finish(0, out.is_ok());
            bars.join();
            let rc = out?;
            tensors.extend(rc.tensors);
            metadata.extend(rc.metadata);
            if let Some(d) = rc.disk {
                disk_shards.extend(d.shards);
            }
            // `rc.health` carries the s3 index-vs-object cross-check for an `s3://`
            // source (built by the reader, which has both halves) and the
            // index-vs-files report for a remote safetensors directory.
            health.extend(rc.health);
            if let Some(s3) = rc.s3 {
                s3_meta = Some(s3); // one `s3://` source per run
            }
            if config.is_none() {
                config = r.read_config(&session, &as_str);
            }
        }
        *remote.session.borrow_mut() = Some(session);

        // Build the central model from what was just read — no extra network I/O:
        // group tensors by their source file into shard headers, roll the on-disk
        // `stat` results into file entries, and carry config + S3 metadata. The
        // remote views still read over SSH lazily; this makes the model (and
        // `--print-model`, serialization) cover remote checkpoints too.
        let (root, source) = match self.remote_browse() {
            Some(RemoteBrowse::Sftp(dir)) => (
                format!("{}:{dir}", self.remote_host_label()),
                crate::model::Source::Sftp {
                    host: self.remote_host_label(),
                    root: dir.clone(),
                },
            ),
            Some(RemoteBrowse::S3(uri)) => {
                (uri.clone(), crate::model::Source::S3 { uri: uri.clone() })
            }
            None => (String::new(), crate::model::Source::Local),
        };
        let mut order: Vec<String> = Vec::new();
        let mut by_src: HashMap<String, Vec<TensorInfo>> = HashMap::new();
        for t in &tensors {
            if !by_src.contains_key(&t.source_path) {
                order.push(t.source_path.clone());
            }
            by_src
                .entry(t.source_path.clone())
                .or_default()
                .push(t.clone());
        }
        let mut shards: Vec<crate::model::ShardHeader> = order
            .iter()
            .enumerate()
            .map(|(i, src)| crate::model::ShardHeader {
                path: src.clone(),
                total_len: 0,
                header_len: 0,
                tensors: by_src.remove(src).unwrap_or_default(),
                // All `__metadata__` on the first shard (it isn't keyed per file).
                metadata: if i == 0 { metadata.clone() } else { Vec::new() },
            })
            .collect();
        if shards.is_empty() && !metadata.is_empty() {
            shards.push(crate::model::ShardHeader {
                path: root.clone(),
                total_len: 0,
                header_len: 0,
                tensors: Vec::new(),
                metadata: metadata.clone(),
            });
        }
        // The same listing the web server shows, from the same numbers (see
        // `remote::remote_file_entries`) — an `s3://` source is described by its object
        // metadata, an SFTP one by the per-shard disk usage.
        let files = crate::remote::remote_file_entries(s3_meta.as_ref(), &disk_shards);
        let cp = crate::model::Checkpoint {
            source,
            root,
            files,
            shards,
            config: config.clone(),
            index: Vec::new(),
            s3: s3_meta.clone(),
        };

        if let Some(rc) = self.remote.as_mut() {
            rc.s3_meta = s3_meta;
        }
        let parts = (
            tensors,
            metadata,
            config,
            crate::stats::DiskUsage::from_shards(disk_shards),
            health,
        );
        Ok((parts, Some(cp)))
    }

    /// The health badge's alert level: red for a real error (a referenced file or
    /// tensor is missing on disk), a softer orange when there are only warnings
    /// (e.g. extra files on disk), `None` when there's nothing to flag.
    pub(super) fn health_alert(&self) -> Option<crate::ui::HealthAlert> {
        if self.health_reports.is_empty() {
            None
        } else if self
            .health_reports
            .iter()
            .any(checkpoint_studio_core::health::HealthReport::has_errors)
        {
            Some(crate::ui::HealthAlert::Error)
        } else {
            Some(crate::ui::HealthAlert::Warning)
        }
    }

    /// Files on disk but absent from the index (per the health reports' extra
    /// files), resolved to absolute paths so they match each tensor's
    /// `source_path` — the tree dims their rows.
    pub(super) fn unindexed_files(reports: &[crate::health::HealthReport]) -> HashSet<String> {
        let mut unindexed = HashSet::new();
        for report in reports {
            if let Some(dir) = Path::new(&report.index_path).parent() {
                for file in &report.extra_files {
                    unindexed.insert(absolute_path(&dir.join(file)));
                }
            }
        }
        unindexed
    }

    /// Shared post-read setup: dedup, sort, parameter/schema/tree build.
    pub(super) fn finalize_load(
        &mut self,
        tensors: Vec<TensorInfo>,
        metadata: Vec<MetadataInfo>,
        config: Option<crate::config::ModelConfig>,
        model: Option<crate::model::Checkpoint>,
    ) {
        // Local index/file health, computed from the freshly-parsed tensors (before
        // dedup, so a name in two shards is seen in both) — the loader already read
        // every header, so this re-reads nothing. Remote index health was folded in
        // by the caller; append the local reports, then derive the unindexed-file
        // set (files on disk but absent from the index) for the tree's dimming.
        let local: Vec<crate::health::HealthReport> = self
            .index_specs
            .iter()
            .map(|spec| crate::health::check_loaded(spec, &tensors))
            .filter(checkpoint_studio_core::health::HealthReport::has_issues)
            .collect();
        self.health_reports.extend(local);
        self.unindexed = Self::unindexed_files(&self.health_reports);

        // The derived reports (health / stats) are keyed to the tensors — drop any
        // cached from a prior load so they're recomputed against the new set.
        *self.cached_check.borrow_mut() = None;
        *self.checkpoint_stats_cache.borrow_mut() = None;
        *self.cached_group_files.borrow_mut() = None;

        // Install the session — the single owner of the canonical (deduped +
        // natural-sorted) tensors/metadata/config. A local read hands over the
        // serializable model; a remote read without an assembled model supplies
        // the parts directly. Dedup + natural-sort now live in the kernel.
        self.session = Some(model.map_or_else(
            || crate::kernel::Session::from_parts(tensors, metadata, config),
            crate::kernel::Session::from_model,
        ));

        let schemas = crate::sample::parse_packing_schemas(self.tensors(), self.metadata());
        self.packing_schemas = schemas;
        self.build_tree();
        // Apply a `--filter` to the live tree (non-destructive view filter); a no-op
        // when none is set. Print paths override this with a destructive prune.
        self.refresh_filter();
        self.full_loaded = true;
    }

    /// Run the full structure load if it hasn't happened yet. The fast `--tensor`
    /// path reads a single tensor and leaves the rest unread; this brings in the
    /// whole tree the first time it's needed (e.g. on returning to the browser),
    /// showing the loading spinner only then.
    pub(super) fn ensure_full_load(&mut self) -> Result<()> {
        if !self.full_loaded {
            self.load_all_files()?;
        }
        Ok(())
    }

    /// Try to read just `name` (plus its packing schema) without enumerating the
    /// whole checkpoint, so a direct `--tensor X` view appears without the cold
    /// full-load spinner. Only the single-HDF5-file case is worth special-casing
    /// — other formats read their whole structure in one cheap header pass, and a
    /// multi-file checkpoint may hold the tensor in any shard. Returns whether the
    /// fast read succeeded; on `false` the caller does a normal full load.
    // `self` is used throughout the `hdf5` branch below; without that feature the body is
    // a stub that ignores it. The receiver stays so the one call site doesn't need two
    // spellings — the same cfg-pair reasoning as `readers::read_hdf5_shard`.
    #[allow(clippy::unused_self)]
    pub(super) fn try_load_single_tensor(&mut self, name: &str) -> bool {
        #[cfg(feature = "hdf5")]
        {
            let [path] = self.files.as_slice() else {
                return false;
            };
            if !matches!(
                path.extension().and_then(|s| s.to_str()),
                Some("h5" | "hdf5")
            ) {
                return false;
            }
            match crate::hdf5::read_one(path, name) {
                Ok(Some((tensor, metadata))) => {
                    // A single-tensor fast open with no full model yet: install a
                    // parts-only session so the data view reads it like any other
                    // (the full load, which replaces this, still runs on `Tab`).
                    self.session = Some(crate::kernel::Session::from_parts(
                        vec![tensor],
                        metadata,
                        None,
                    ));
                    let schemas =
                        crate::sample::parse_packing_schemas(self.tensors(), self.metadata());
                    self.packing_schemas = schemas;
                    true
                }
                // Not found or a read error — let the full load handle it (and
                // surface the "tensor not found" message).
                _ => false,
            }
        }
        #[cfg(not(feature = "hdf5"))]
        {
            let _ = name;
            false
        }
    }

    /// The fused-codebook packing schema for `name`, if the checkpoint declared one.
    pub(super) fn schema_for(&self, name: &str) -> Option<&PackingSchema> {
        self.packing_schemas.get(name)
    }

    /// The view a tensor opens in with no explicit override: the codebook
    /// [`ViewDtype::Unpacked`] when it carries a packing schema, else `Stored`.
    pub(super) fn default_view(&self, name: &str) -> ViewDtype {
        if self.packing_schemas.contains_key(name) {
            ViewDtype::Unpacked
        } else {
            ViewDtype::Stored
        }
    }

    /// The active view for a tensor: an explicit `d`/`--dtype` override if set,
    /// otherwise its [`default_view`].
    pub(super) fn active_view(&self, name: &str) -> ViewDtype {
        self.data_view
            .dtype_overrides
            .borrow()
            .get(name)
            .copied()
            .unwrap_or_else(|| self.default_view(name))
    }

    /// The value range to bin the histogram over: the intrinsic codebook span
    /// `0..=2^max_width-1` for the unmerged view (so every index gets a bar, even
    /// absent ones — like the 4-bit views show all 16), otherwise the tensor's
    /// exact min/max once a stats scan has cached it.
    pub(super) fn histogram_range(
        &self,
        tensor: &TensorInfo,
        view: ViewDtype,
    ) -> Option<(f64, f64)> {
        if view == ViewDtype::Unpacked
            && let Some(s) = self.schema_for(&tensor.name)
        {
            return Some((0.0, ((1u64 << s.max_width()) - 1) as f64));
        }
        self.cached_stats(tensor, view).map(|s| (s.min, s.max))
    }

    /// Read every file's top-level structure (tensors + metadata) into owned
    /// vectors. A free function (no `&self`) so it can run on a worker thread
    /// while the UI animates a loading spinner, and so the `diff` subcommand can
    /// load a checkpoint's structure headlessly.
    pub(crate) fn gather_checkpoint(
        files: &[PathBuf],
        remote: Option<&crate::remote::RemoteRead>,
    ) -> Result<(CheckpointParts, Option<crate::model::Checkpoint>)> {
        // `--ssh-proxy`: every source is read on the remote (an s3:// cstorch
        // checkpoint, or a remote safetensors directory/file), keeping the
        // credentials and data there. (The central model is filled by the remote
        // reader in a later step; the remote path returns the parts tuple only.)
        if let Some(r) = remote {
            let mut tensors: Vec<TensorInfo> = Vec::new();
            let mut metadata: Vec<MetadataInfo> = Vec::new();
            let mut config: Option<crate::config::ModelConfig> = None;
            let mut disk_shards: Vec<crate::stats::ShardDisk> = Vec::new();
            let mut remote_health: Vec<crate::health::HealthReport> = Vec::new();
            for file_path in files {
                let as_str = file_path.to_string_lossy();
                let (t, m, cfg, disk, health) = r.fetch_with_config(&as_str)?;
                tensors.extend(t);
                metadata.extend(m);
                config = config.or(cfg);
                if let Some(d) = disk {
                    disk_shards.extend(d.shards);
                }
                remote_health.extend(health);
            }
            let parts = (
                tensors,
                metadata,
                config,
                crate::stats::DiskUsage::from_shards(disk_shards),
                remote_health,
            );
            return Ok((parts, None));
        }
        // Local: a bare s3:// URI has no local credentials to read.
        for file_path in files {
            let as_str = file_path.to_string_lossy();
            if crate::s3::is_uri(&as_str) {
                anyhow::bail!(
                    "{as_str}: reading an s3:// checkpoint needs --ssh-proxy <[user@]host> \
                     (its credentials stay on the remote)"
                );
            }
        }
        // Read the whole local checkpoint into the central model in one pass (fs
        // walk + every header + config + index); the tuple is derived from it.
        let cp = crate::readers::read_local(files)?;
        let parts = (
            cp.tensors_vec(),
            cp.metadata_vec(),
            cp.config.clone(),
            None,
            Vec::new(),
        );
        Ok((parts, Some(cp)))
    }
}
