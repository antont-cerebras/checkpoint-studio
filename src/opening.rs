//! **Opening a checkpoint** — the one path from "a spec someone typed" to "a checkpoint
//! read and ready to install", shared by the terminal and the web server.
//!
//! Both frontends can now change checkpoint without restarting, and both must accept the
//! same spellings when they do: a file, a directory, a glob, `hf://owner/repo`,
//! `s3://bucket/prefix`, `:/path/on/the/proxy`, `~/expanded`. That list is a *rule*, not a
//! feature of either frontend — so it lives here, and each frontend calls [`resolve`].
//!
//! Before this, reading a checkpoint at startup was spelled twice: the terminal went
//! through [`crate::source`], while `run_web` had its own chain of three branches (an
//! ssh-proxy arm, a Hub arm, a local fall-through). Two spellings of the same rule is how
//! the arms drifted last time — the Hub arm computed its parts one way and the remote arm
//! another — which is what [`crate::source`] was introduced to stop. A runtime "open
//! another checkpoint" would have made it three.
//!
//! ## What the two frontends genuinely need differently
//!
//! Exactly one thing, and it is named rather than duplicated: [`Want`]. The terminal drives
//! a [`crate::kernel::Session`] from the flat parts and lets the remote reader fill the
//! serializable model in a later step; the JSON API has to *serve* that model, so it needs
//! one up front. For a proxied read those are two different calls into
//! [`crate::remote::RemoteRead`] (and, deliberately, two different `ObjectMeta` costs), so
//! the caller says which it wants and this module owns the difference.

use std::path::PathBuf;

use anyhow::{Result, bail};

use crate::explorer::CheckpointParts;
use crate::hf::ReadProgress;
use crate::model::Checkpoint;
use crate::{cli_config, health, remote};

/// What a caller needs out of a read.
///
/// Not a `bool`: the two arms differ in what work they do remotely (`ObjectMeta::Fetch`
/// costs a HEAD per object), so the choice deserves a name at every call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Want {
    /// The flat parts only — what the terminal installs into its session, leaving the
    /// remote reader to assemble the model later.
    Parts,
    /// The full serializable [`Checkpoint`], which the JSON API serves directly.
    Model,
}

/// The switches a read needs beyond the spec itself, so a checkpoint opened at runtime is
/// read the same way the one on the command line was.
#[derive(Debug, Clone, Default)]
pub(crate) struct Options {
    /// `--recursive`: descend into subdirectories when resolving a directory.
    pub recursive: bool,
    /// `--no-health-check`: skip parsing `model.safetensors.index.json` for the
    /// index/shard cross-check.
    pub no_health_check: bool,
    /// `--ssh-proxy` / `--ssh-venv` as given on the command line. `None` falls back to the
    /// config file, exactly as at startup.
    pub proxy: Option<String>,
    pub venv: Option<String>,
}

/// **Where** to read from, resolved but not yet read.
///
/// Resolution and reading are separate steps because the two frontends need the split in
/// different places. The terminal re-points itself at a target and then runs its *existing*
/// animated load (worker thread, spinner, shards gauge) — so it must be able to resolve
/// without reading. The web server reads on the request thread and swaps the result in. Both
/// then go through [`Target::read`], which is the single place a checkpoint is actually read.
///
/// Resolution does touch the filesystem — a directory walk and the index parse — but reads no
/// shard headers, which is the expensive part.
#[derive(Debug, Clone)]
pub(crate) struct Target {
    /// What the user asked to open, after `:PATH` and `~` resolution — the terminal keeps
    /// this as "what to re-read", and it is what a recents list records.
    pub requested: Vec<PathBuf>,
    /// The local shard files the request resolved to. **Empty** for a source with no local
    /// files (an ssh-proxy read), which is what tells the web state there is no local
    /// directory to walk or to name in a reproducible `diff` command.
    pub resolved: Vec<PathBuf>,
    /// Parsed indexes to cross-check against the loaded tensors; empty when health
    /// checking is off or the source has no local index.
    pub index_specs: Vec<health::IndexSpec>,
    /// The proxy to read over, if this is a remote target.
    pub remote: Option<remote::RemoteRead>,
}

/// A checkpoint that has been read and is ready for a frontend to install.
pub(crate) struct Opened {
    /// Where it came from, so a caller can install the target alongside the data.
    pub target: Target,
    pub parts: CheckpointParts,
    /// The serializable model, when the source built one — always `Some` for
    /// [`Want::Model`], which is the contract the web server relies on.
    pub checkpoint: Option<Checkpoint>,
}

/// Resolve the checkpoint named by `spec` — the interactive "open this" path.
///
/// One spec, not many, because that is what a person types into a prompt. Startup keeps its
/// whole path list and calls [`Target::from_paths`]: a glob on the command line can name
/// shards across directories, and dropping that to one path would be a regression.
pub(crate) fn resolve(spec: &str, opts: &Options) -> Result<Target> {
    let spec = spec.trim();
    if spec.is_empty() {
        bail!("no checkpoint given");
    }
    let cfg = cli_config::CliConfig::load();
    // The same resolution the command line performs, so `:PATH` reaches the configured
    // proxy and a bare `s3://` URI is routed to it rather than failing locally.
    let (requested, proxy) = crate::resolve_remote_sources(
        &[PathBuf::from(spec)],
        opts.proxy.clone(),
        opts.venv.clone(),
        &cfg,
    )?;
    let remote = proxy.map(|(host, venv)| remote::RemoteRead::new(host, venv));
    Target::from_paths(&requested, remote, opts)
        // Name what was typed. Deeper layers report the resolved path, which for `:PATH` and
        // `~/…` is not the string the prompt still has on screen.
        .map_err(|e| e.context(format!("opening {spec}")))
}

impl Target {
    /// Resolve inputs that are already paths — startup's entry point, and where [`resolve`]
    /// lands once it has turned a spec into paths.
    pub(crate) fn from_paths(
        requested: &[PathBuf],
        remote: Option<remote::RemoteRead>,
        opts: &Options,
    ) -> Result<Self> {
        // A remote target has no local directory to walk: the paths go to the reader as they
        // are, and there is no local index to parse.
        if remote.is_some() {
            return Ok(Self {
                requested: requested.to_vec(),
                resolved: Vec::new(),
                index_specs: Vec::new(),
                remote,
            });
        }
        // Local or Hub. `collect_safetensors_files` expands `~`, globs, walks a directory and
        // parses the index — and passes an `hf://` URI through untouched, which is what keeps
        // the Hub out of this branch's business.
        let (resolved, index_specs) =
            crate::collect_safetensors_files(requested, opts.recursive, opts.no_health_check)?;
        if resolved.is_empty() {
            // The paths, not `model::root_label`: that renders a *display* label (the last
            // component), so a missing `/nope/not/here` reported itself as "not found at
            // here" — which names nothing the reader can act on.
            bail!("no checkpoint files found at {}", spec_of_paths(requested));
        }
        Ok(Self {
            requested: requested.to_vec(),
            resolved,
            index_specs,
            remote: None,
        })
    }

    /// A target whose paths are already resolved — no directory walk, no index parse.
    ///
    /// For a caller that did that work earlier and is only asking for the *read*: the
    /// terminal, whose startup walked the paths and whose interactive open resolves before it
    /// re-points. Walking again here would double the cost of every load and, on a directory
    /// that changed underneath, would silently read a different set of shards than the caller
    /// thinks it holds.
    pub(crate) fn already_resolved(paths: &[PathBuf], remote: Option<remote::RemoteRead>) -> Self {
        let remote_read = remote.is_some();
        Self {
            requested: paths.to_vec(),
            // A remote target reads from `requested`; a local one from `resolved`.
            resolved: if remote_read {
                Vec::new()
            } else {
                paths.to_vec()
            },
            // The caller owns its own index specs in this case (the terminal keeps them on
            // the Explorer), so this target carries none.
            index_specs: Vec::new(),
            remote,
        }
    }

    /// How this target is written back to a person — the paths as asked for.
    ///
    /// This is what a recents entry must be, because picking one retypes it verbatim: a
    /// display label ("Qwen3-Coder-30B-A3B-lut-3bit") is not a path, and recording one at
    /// startup and a path on every later open put the same checkpoint in the list twice under
    /// two spellings.
    pub(crate) fn spec(&self) -> String {
        spec_of_paths(&self.requested)
    }

    /// Read the checkpoint. The single place a read happens for either frontend.
    pub(crate) fn read(self, want: Want, progress: &ReadProgress) -> Result<Opened> {
        if let Some(r) = self.remote.clone() {
            return self.read_proxied(&r, want, progress);
        }
        let (parts, checkpoint) = crate::source::resolve(&self.resolved, None)?.read(progress)?;
        Ok(Opened {
            target: self,
            parts,
            checkpoint,
        })
    }
}

impl Target {
    /// The ssh-proxy arm — the one place the two [`Want`]s genuinely diverge.
    fn read_proxied(
        self,
        r: &remote::RemoteRead,
        want: Want,
        progress: &ReadProgress,
    ) -> Result<Opened> {
        match want {
            // One read that assembles the model, with the per-object metadata the stats and
            // health screens show for an `s3://` source.
            Want::Model => {
                let [one] = self.requested.as_slice() else {
                    bail!(
                        "serving over an ssh proxy takes a single remote checkpoint; got {} paths",
                        self.requested.len()
                    );
                };
                let cp = r.read_checkpoint(&one.to_string_lossy(), remote::ObjectMeta::Fetch)?;
                let parts = (
                    cp.tensors_vec(),
                    cp.metadata_vec(),
                    cp.config.clone(),
                    None,
                    Vec::new(),
                );
                Ok(Opened {
                    target: self,
                    parts,
                    checkpoint: Some(cp),
                })
            }
            // The structure-only read, which skips those HEADs — the terminal assembles what
            // it needs from the parts and fills the model in a later step.
            Want::Parts => {
                let (parts, checkpoint) =
                    crate::source::resolve(&self.requested, Some(r))?.read(progress)?;
                Ok(Opened {
                    target: self,
                    parts,
                    checkpoint,
                })
            }
        }
    }
}

/// Paths as a person would retype them: space-joined, because that is how a multi-path glob
/// was given on the command line in the first place.
pub(crate) fn spec_of_paths(paths: &[PathBuf]) -> String {
    paths
        .iter()
        .map(|p| p.to_string_lossy())
        .collect::<Vec<_>>()
        .join(" ")
}

/// The checkpoints opened this run, most recent first.
///
/// Offered by both frontends' open prompt, so switching back to where you were is a pick
/// rather than a retype. Deliberately *not* persisted to disk: it is a within-session
/// convenience, and a file of recently-read paths is a surprising thing for a read-only
/// browser to start writing to a user's home directory.
#[derive(Debug, Clone)]
pub(crate) struct Recents {
    specs: Vec<String>,
    cap: usize,
}

impl Default for Recents {
    fn default() -> Self {
        Self::with_cap(10)
    }
}

impl Recents {
    pub(crate) fn with_cap(cap: usize) -> Self {
        Self {
            specs: Vec::new(),
            cap: cap.max(1),
        }
    }

    /// Record a spec as the most recent, moving it up rather than duplicating it. Reopening
    /// the same checkpoint is the common case, and a list that grew a second identical row
    /// each time would push the others out for no reason.
    pub(crate) fn record(&mut self, spec: &str) {
        let spec = spec.trim();
        if spec.is_empty() {
            return;
        }
        self.specs.retain(|s| s != spec);
        self.specs.insert(0, spec.to_string());
        self.specs.truncate(self.cap);
    }

    /// Most recent first.
    pub(crate) fn list(&self) -> &[String] {
        &self.specs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_recent_open_moves_to_the_front_rather_than_repeating() {
        let mut r = Recents::default();
        r.record("/a");
        r.record("/b");
        r.record("/a");
        assert_eq!(
            r.list(),
            ["/a", "/b"],
            "reopening /a should move it, not duplicate it"
        );
    }

    #[test]
    fn recents_are_bounded_and_drop_the_oldest() {
        let mut r = Recents::with_cap(2);
        r.record("/a");
        r.record("/b");
        r.record("/c");
        assert_eq!(r.list(), ["/c", "/b"], "the cap should evict the oldest");
    }

    #[test]
    fn blank_and_padded_specs_are_handled_like_the_prompt_gives_them() {
        let mut r = Recents::default();
        r.record("   ");
        assert!(
            r.list().is_empty(),
            "a blank prompt submission is not an open"
        );
        r.record("  /a  ");
        r.record("/a");
        assert_eq!(
            r.list(),
            ["/a"],
            "padding must not make two entries of one path"
        );
    }

    /// The empty spec is the prompt's own failure mode (submit with nothing typed), and it
    /// should be a message rather than a resolution attempt.
    #[test]
    fn opening_nothing_says_so() {
        let e = resolve("  ", &Options::default()).expect_err("an empty spec resolves to nothing");
        assert!(
            e.to_string().contains("no checkpoint given"),
            "unexpected message: {e}"
        );
    }

    /// A path that resolves to no checkpoint files names the **whole path**, because that is
    /// what the person typed and what they have to correct. Reporting a display label instead
    /// ("no checkpoint files found at here") named nothing actionable.
    #[test]
    fn resolving_a_path_with_no_checkpoint_names_the_whole_spec() {
        let dir = std::env::temp_dir().join(format!("cs_open_none_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let spec = dir.join("not-a-checkpoint");
        std::fs::create_dir_all(&spec).expect("mkdir");
        let e = resolve(&spec.to_string_lossy(), &Options::default())
            .expect_err("an empty directory is not a checkpoint");
        let msg = format!("{e:#}");
        assert!(
            msg.contains(&spec.to_string_lossy().into_owned()),
            "the message should carry the path typed, got: {msg}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Resolution must not read shard headers — that split is what lets the terminal re-point
    /// itself and then run its animated load, instead of freezing while it reads.
    #[test]
    fn resolving_finds_the_files_without_reading_them() {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny.safetensors");
        let t = resolve(&fixture.to_string_lossy(), &Options::default()).expect("resolves");
        assert_eq!(t.resolved, vec![fixture.clone()], "the shard file is found");
        assert!(t.remote.is_none(), "a local path needs no proxy");
        assert_eq!(
            t.spec(),
            fixture.to_string_lossy(),
            "and is spelled back as given"
        );
    }
}
