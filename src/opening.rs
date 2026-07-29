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

use std::path::{Path, PathBuf};

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

/// The checkpoints opened recently, most recent first.
///
/// Offered by both frontends' open prompt, so switching back to where you were is a pick rather
/// than a retype — and **kept across restarts**, because a within-session list loses exactly
/// what it is for: an `hf://owner/repo` you typed once and would have to retype from memory
/// after the server restarts. A local path can be tab-completed in a shell; a Hub URI or an
/// `s3://` prefix cannot.
///
/// ## Meant to be edited by hand
///
/// So it is TOML, in the directory this app already owns — `recents.toml` beside `config.toml`.
/// One array, most recent first, one entry per line:
///
/// ```toml
/// recent = [
///   "hf://moonshotai/Kimi-K3",
///   "/models/checkpoint_1000",
/// ]
/// ```
///
/// Its own file rather than a section of `config.toml`, because this list is *rewritten* every
/// time a checkpoint is opened. `config.toml` is hand-maintained — rewriting it would put the
/// user's comments and layout at the mercy of a serializer, for no gain.
///
/// The file — not this struct — is the source of truth for a persistent list. Every read goes
/// back to disk and every write merges into what is on disk, so an edit made while the app is
/// running is picked up rather than overwritten by whatever the process happened to remember.
/// The file is tiny, so re-reading it per use costs nothing worth measuring.
///
/// Persistence is opt-in ([`Self::persistent`]) rather than automatic, so a list built in a test
/// or a one-shot export never writes to the user's config directory.
#[derive(Debug, Clone)]
pub(crate) struct Recents {
    /// The in-memory list. For a persistent one this is a fallback for when the file cannot be
    /// read, not the authority — see the type docs.
    specs: Vec<String>,
    cap: usize,
    /// The file this list lives in; `None` keeps it in memory only.
    store: Option<PathBuf>,
}

/// Written at the top of the file on every save, so the file explains itself to whoever opens
/// it — including that it gets rewritten, which is the one thing a hand-editor needs to know.
const RECENTS_HEADER: &str = "\
# checkpoint-studio — recently opened checkpoints, most recent first.
#
# Edit this freely: a path, a glob, `hf://owner/repo`, an `s3://` prefix, or `:/path` for the
# configured ssh proxy. Reorder to change what the open prompt offers first.
#
# Opening a checkpoint rewrites this file (newest first, capped), which drops any comments you
# add below. Your entries survive: a write merges with whatever the file holds at the time, so
# an edit made while the app is running is not lost.
";

/// The file's shape. One array, so a hand-editor has one thing to get right — and
/// `#[serde(default)]` so an empty or partial file reads as "no entries" rather than an error.
#[derive(serde::Deserialize, Default)]
struct RecentsFile {
    #[serde(default)]
    recent: Vec<String>,
}

impl Default for Recents {
    fn default() -> Self {
        Self::with_cap(10)
    }
}

impl Recents {
    /// An in-memory list — nothing is read from or written to disk.
    pub(crate) fn with_cap(cap: usize) -> Self {
        Self {
            specs: Vec::new(),
            cap: cap.max(1),
            store: None,
        }
    }

    /// The user's list, in a file that is re-read on use and merged into on change.
    ///
    /// Best-effort throughout: a missing, unreadable or malformed file reads as an empty list,
    /// and a failed write is dropped. A convenience list is never worth failing a checkpoint
    /// read over.
    pub(crate) fn persistent() -> Self {
        let store = Self::store_path();
        let specs = store.as_deref().map(Self::read_file).unwrap_or_default();
        Self {
            specs,
            cap: 10,
            store,
        }
    }

    /// A persistent list kept in `path` — for tests, which must not touch the real config dir.
    #[cfg(test)]
    pub(crate) fn persistent_at(path: PathBuf, cap: usize) -> Self {
        let specs = Self::read_file(&path);
        Self {
            specs,
            cap: cap.max(1),
            store: Some(path),
        }
    }

    /// `recents.toml` beside the config file — the directory this app already owns
    /// (`$XDG_CONFIG_HOME/checkpoint-studio/`, else `$HOME/.config/checkpoint-studio/`).
    fn store_path() -> Option<PathBuf> {
        cli_config::CliConfig::path().map(|p| p.with_file_name("recents.toml"))
    }

    /// Read the file's `recent` array: blanks dropped, duplicates collapsed keeping the first
    /// (topmost = most recent). Tolerant by design — this is a file people edit, so a syntax
    /// error reads as "no entries" rather than breaking an open.
    fn read_file(path: &Path) -> Vec<String> {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Vec::new();
        };
        let parsed: RecentsFile = toml::from_str(&text).unwrap_or_default();
        let mut out: Vec<String> = Vec::new();
        for spec in parsed.recent {
            let spec = spec.trim();
            if !spec.is_empty() && !out.iter().any(|s| s == spec) {
                out.push(spec.to_string());
            }
        }
        out
    }

    /// Record a spec as the most recent, moving it up rather than duplicating it. Reopening the
    /// same checkpoint is the common case, and a list that grew a second identical row each time
    /// would push the others out for no reason.
    ///
    /// For a persistent list this merges with the file as it stands, so a hand edit made since
    /// the process started survives.
    pub(crate) fn record(&mut self, spec: &str) {
        let spec = spec.trim();
        if spec.is_empty() {
            return;
        }
        let mut specs = self.list();
        specs.retain(|s| s != spec);
        specs.insert(0, spec.to_string());
        specs.truncate(self.cap);
        self.specs = specs;
        self.save();
    }

    /// Most recent first. Reads the file for a persistent list, so a hand edit shows up without
    /// a restart; falls back to what is in memory if the file cannot be read.
    pub(crate) fn list(&self) -> Vec<String> {
        self.store.as_deref().map_or_else(
            || self.specs.clone(),
            |path| {
                let from_file = Self::read_file(path);
                // Fall back to memory only when the file gives nothing: a file that has been
                // emptied on purpose should read as empty, but one that has gone missing or
                // unreadable should not silently discard what this process knows.
                if from_file.is_empty() {
                    self.specs.clone()
                } else {
                    from_file
                }
            },
        )
    }

    /// Write the list, if this one is persistent. Silent on failure — see [`Self::persistent`].
    fn save(&self) {
        let Some(path) = &self.store else {
            return;
        };
        let mut text = String::from(RECENTS_HEADER);
        text.push_str("\nrecent = [\n");
        for s in &self.specs {
            // TOML's own quoting, so a path containing a quote, a backslash or a `#` survives
            // the round trip instead of corrupting the entry (or the rest of the file).
            text.push_str("  ");
            text.push_str(&toml::Value::String(s.clone()).to_string());
            text.push_str(",\n");
        }
        text.push_str("]\n");
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(path, text);
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

    /// A scratch file unique to this process and test — never the user's real config dir.
    fn recents_file(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cs_recents_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir.join("recents.toml")
    }

    #[test]
    fn a_persisted_list_survives_a_restart() {
        // The whole point: an `hf://` repo typed once is still offered after the process is
        // gone, which is what a session-only list loses.
        let path = recents_file("survives");
        let mut first = Recents::persistent_at(path.clone(), 10);
        first.record("hf://moonshotai/Kimi-K3");
        first.record("/models/checkpoint_1000");

        let reopened = Recents::persistent_at(path, 10);
        assert_eq!(
            reopened.list(),
            ["/models/checkpoint_1000", "hf://moonshotai/Kimi-K3"],
            "a new process should see the list the last one wrote, most recent first"
        );
    }

    #[test]
    fn the_file_is_toml_a_person_can_edit() {
        let path = recents_file("editable");
        let mut r = Recents::persistent_at(path.clone(), 10);
        r.record("/models/a");
        let text = std::fs::read_to_string(&path).expect("the file was written");
        // A header that explains itself, and one entry per line so a diff of a hand edit is
        // readable.
        assert!(
            text.starts_with("# checkpoint-studio"),
            "header missing: {text}"
        );
        assert!(
            text.contains("recent = [\n  \"/models/a\",\n]"),
            "layout: {text}"
        );
        // And it parses as TOML, which is the point of choosing it.
        let back: toml::Value = toml::from_str(&text).expect("valid TOML");
        assert!(back.get("recent").is_some());
    }

    #[test]
    fn a_hand_edit_is_read_back_and_not_overwritten() {
        // The file is the source of truth, so an edit made while the app is running has to
        // survive the next open rather than being clobbered by what the process remembered.
        let path = recents_file("hand_edit");
        let mut r = Recents::persistent_at(path.clone(), 10);
        r.record("/models/a");
        std::fs::write(
            &path,
            "# my own note\nrecent = [\n  \"/models/hand-added\",\n  \"/models/a\",\n]\n",
        )
        .expect("write by hand");

        assert_eq!(
            r.list(),
            ["/models/hand-added", "/models/a"],
            "the list should be read back from the file, not from memory"
        );
        r.record("/models/b");
        assert_eq!(
            r.list(),
            ["/models/b", "/models/hand-added", "/models/a"],
            "a later open should merge into the hand-edited list, not replace it"
        );
    }

    #[test]
    fn a_broken_or_missing_file_is_not_an_error() {
        // A convenience list is never worth failing a checkpoint read over.
        let path = recents_file("broken");
        assert!(
            Recents::persistent_at(path.clone(), 10).list().is_empty(),
            "a missing file reads as an empty list"
        );
        std::fs::write(&path, "recent = [ this is not toml").expect("write junk");
        assert!(
            Recents::persistent_at(path.clone(), 10).list().is_empty(),
            "a malformed file reads as an empty list"
        );
        // And it recovers: a record over the junk writes a valid file.
        let mut r = Recents::persistent_at(path.clone(), 10);
        r.record("/models/a");
        assert_eq!(Recents::persistent_at(path, 10).list(), ["/models/a"]);
    }

    #[test]
    fn a_path_with_awkward_characters_round_trips() {
        // TOML's quoting is why this is not hand-rolled: a `#` would start a comment and a
        // backslash or quote would end the string early.
        let path = recents_file("awkward");
        let awkward = ["/models/a b/ckpt", "/models/has#hash", "/models/has\"quote"];
        let mut r = Recents::persistent_at(path.clone(), 10);
        for a in awkward {
            r.record(a);
        }
        let mut expected: Vec<&str> = awkward.to_vec();
        expected.reverse(); // most recent first
        assert_eq!(Recents::persistent_at(path, 10).list(), expected);
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
