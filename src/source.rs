//! **Data sources**, behind one trait — so adding a new kind is writing an impl, not
//! editing a dispatch.
//!
//! Reading a checkpoint's structure used to be a chain of `if`s: an ssh-proxy branch, a
//! bare-`s3://` refusal, a Hugging Face branch, and the local fall-through, all inside one
//! function. Every new source meant finding that function and threading another arm through
//! it — and the arms had already drifted (the Hub branch computed its parts one way, the
//! remote branch another).
//!
//! Now each source is a [`Source`] implementation that answers two questions: what can it
//! do ([`crate::capability`]), and how does it read ([`Source::read`]). [`resolve`] picks
//! one from what the user asked for. Adding a source is a new impl plus one line there;
//! nothing else in the app changes, because every frontend and subcommand goes through this
//! trait and asks capabilities rather than matching on a source kind.
//!
//! The capability model is the other half of this: a feature asks
//! `capabilities().read_bytes` (or `modify_in_place`, or `reach`) instead of "is this
//! local", so a source that gains an ability turns it on in one row rather than in every
//! caller.

use std::path::PathBuf;

use anyhow::{Result, bail};

use crate::capability::Location;
use crate::hf::ReadProgress;
use crate::model::Checkpoint;
use crate::opening::CheckpointParts;

/// What a read produced: the flat parts every screen needs, and the central serializable
/// model when the source can build one.
///
/// The remote (ssh-proxy) reader fills the model in a later step, so it returns `None`
/// here — which is why this is a pair rather than just a [`Checkpoint`]. Modelling that
/// honestly beat pretending every source can produce a model up front.
///
/// A pair and not a struct: two elements whose types could not be confused for each other, read at
/// every call site as "the parts, and the model if there is one". The *parts* were a five-element
/// tuple and are [`CheckpointParts`] now, which is where the positional risk actually was.
pub(crate) type Read = (CheckpointParts, Option<Checkpoint>);

/// One kind of checkpoint source: what it can do, and how to read it.
pub(crate) trait Source {
    /// A short description for errors and progress lines.
    fn describe(&self) -> String;

    /// Where this source lives — the location half of the capability pair.
    fn location(&self) -> Location;

    /// Read the checkpoint's structure. `progress` is updated by sources that can count
    /// their work (the Hub knows its shard count up front); others leave it alone and the
    /// caller shows a spinner.
    fn read(&self, progress: &ReadProgress) -> Result<Read>;
}

/// A local path, directory or glob — the only source that can currently read tensor bytes.
pub(crate) struct LocalSource {
    pub files: Vec<PathBuf>,
}

impl Source for LocalSource {
    fn describe(&self) -> String {
        crate::model::root_label(&self.files)
    }

    fn location(&self) -> Location {
        Location::Local
    }

    fn read(&self, _progress: &ReadProgress) -> Result<Read> {
        // A bare `s3://` has no local credentials; say so with the shared reason rather
        // than failing later as a missing file.
        for path in &self.files {
            let as_str = path.to_string_lossy();
            if crate::s3::is_uri(&as_str) {
                bail!(
                    "{as_str}: {}",
                    Location::S3
                        .proxy_note()
                        .unwrap_or("cannot be read from here")
                );
            }
        }
        // One pass: the fs walk, every header, the config and the index.
        let cp = crate::readers::read_local(&self.files)?;
        Ok((parts_of(&cp), Some(cp)))
    }
}

/// A Hugging Face Hub repository, read over HTTPS — headers only, no weights.
pub(crate) struct HfSource {
    pub repo: crate::hf::RepoRef,
}

impl Source for HfSource {
    fn describe(&self) -> String {
        self.repo.spec()
    }

    fn location(&self) -> Location {
        Location::Hf
    }

    fn read(&self, progress: &ReadProgress) -> Result<Read> {
        let cp = crate::hf::read_checkpoint(&self.repo, progress)?;
        Ok((parts_of(&cp), Some(cp)))
    }
}

/// A checkpoint read on an SSH proxy — a remote safetensors directory, or an `s3://`
/// cstorch prefix whose credentials stay there. Only metadata comes back.
pub(crate) struct ProxiedSource {
    pub paths: Vec<PathBuf>,
    pub remote: crate::remote::RemoteRead,
    /// Whether the paths are `s3://` URIs. This is what makes a proxied source report
    /// `Location::S3` rather than `Location::Sftp` — the same transport, different
    /// capabilities (per-object metadata is S3's alone).
    pub s3: bool,
}

impl Source for ProxiedSource {
    fn describe(&self) -> String {
        self.paths
            .first()
            .map_or_else(String::new, |p| p.to_string_lossy().into_owned())
    }

    fn location(&self) -> Location {
        if self.s3 {
            Location::S3
        } else {
            Location::Sftp
        }
    }

    fn read(&self, progress: &ReadProgress) -> Result<Read> {
        let mut tensors = Vec::new();
        let mut metadata = Vec::new();
        let mut config = None;
        let mut disk_shards = Vec::new();
        let mut health = Vec::new();
        for path in &self.paths {
            let (t, m, cfg, disk, h) = self.remote.fetch_with_config(
                &path.to_string_lossy(),
                Some(progress.abort_flag()),
                Some(progress.load()),
            )?;
            tensors.extend(t);
            metadata.extend(m);
            config = config.or(cfg);
            if let Some(d) = disk {
                disk_shards.extend(d.shards);
            }
            health.extend(h);
        }
        // The central model is assembled by the remote reader in a later step.
        let parts = CheckpointParts {
            tensors,
            metadata,
            config,
            disk_usage: crate::stats::DiskUsage::from_shards(disk_shards),
            health,
        };
        Ok((parts, None))
    }
}

/// The flat parts derived from a model — the same projection for every source that can
/// build one, which the Hub and local branches used to each spell out.
fn parts_of(cp: &Checkpoint) -> CheckpointParts {
    CheckpointParts {
        tensors: cp.tensors_vec(),
        metadata: cp.metadata_vec(),
        config: cp.config.clone(),
        disk_usage: None,
        health: Vec::new(),
    }
}

/// Pick the source for what the user asked for.
///
/// The one place that maps a request to an implementation — so a new kind of source is an
/// impl above plus an arm here, and nothing else.
pub(crate) fn resolve(
    paths: &[PathBuf],
    remote: Option<&crate::remote::RemoteRead>,
) -> Result<Box<dyn Source>> {
    // A proxy was configured, so every path is read there — an `s3://` prefix or a remote
    // directory. Checked first because it changes how the *same* path is read.
    if let Some(r) = remote {
        let s3 = paths
            .first()
            .is_some_and(|p| crate::s3::is_uri(&p.to_string_lossy()));
        return Ok(Box::new(ProxiedSource {
            paths: paths.to_vec(),
            remote: r.clone(),
            s3,
        }));
    }
    if let Some(first) = paths.first()
        && crate::hf::is_uri(&first.to_string_lossy())
    {
        if paths.len() > 1 {
            bail!(
                "one Hugging Face repo at a time (got {} paths)",
                paths.len()
            );
        }
        return Ok(Box::new(HfSource {
            repo: crate::hf::parse(&first.to_string_lossy())?,
        }));
    }
    Ok(Box::new(LocalSource {
        files: paths.to_vec(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(spec: &[&str]) -> Vec<PathBuf> {
        spec.iter().map(PathBuf::from).collect()
    }

    /// Resolution is the whole dispatch: one arm per kind, and this pins which arm each
    /// request lands on. It is the test a new source extends by one line.
    #[test]
    fn resolution_picks_the_source_for_the_request() {
        let at = |spec: &str| resolve(&paths(&[spec]), None).map(|s| (s.location(), s.describe()));

        assert_eq!(
            at("/m/model.safetensors").ok().map(|s| s.0),
            Some(Location::Local)
        );
        assert_eq!(
            at("hf://owner/name").ok(),
            Some((Location::Hf, "hf://owner/name".to_string()))
        );
        // A browser URL is the same source, normalised.
        assert_eq!(
            at("https://huggingface.co/owner/name").ok(),
            Some((Location::Hf, "hf://owner/name".to_string())),
            "a pasted URL and the hf:// form are one source"
        );
    }

    /// One repo at a time: a Hub read is a repo, not a set of files, so two is a mistake
    /// worth naming rather than silently reading the first.
    #[test]
    fn two_hub_repos_is_an_error() {
        let Err(err) = resolve(&paths(&["hf://a/b", "hf://c/d"]), None) else {
            panic!("two repos should not resolve to one source");
        };
        assert!(format!("{err}").contains("one Hugging Face repo"), "{err}");
    }

    /// A bare `s3://` without a proxy is refused *with the shared reason*, not left to fail
    /// later as a missing file.
    #[test]
    fn a_bare_s3_uri_is_refused_with_the_shared_reason() {
        let Ok(source) = resolve(&paths(&["s3://bucket/model"]), None) else {
            panic!("without a proxy it resolves as a local path, then refuses on read");
        };
        assert_eq!(
            source.location(),
            Location::Local,
            "without a proxy it is just a path we cannot read"
        );
        let Err(err) = source.read(&ReadProgress::default()) else {
            panic!("a bare s3:// URI must not read as a local path");
        };
        let msg = format!("{err}");
        assert!(msg.contains("--ssh-proxy"), "names the flag: {msg}");
        assert!(
            Location::S3
                .proxy_note()
                .is_some_and(|note| msg.contains(note)),
            "and it is the shared sentence, not a second wording: {msg}"
        );
    }

    /// Capabilities come from the source, so a subcommand asks the source rather than
    /// matching on its kind — the property the whole seam exists for.
    #[test]
    fn a_source_answers_capability_questions() {
        use crate::capability::{Capabilities, Format};

        let Ok(local) = resolve(&paths(&["/m/model.safetensors"]), None) else {
            panic!("a local path resolves");
        };
        let caps = Capabilities::of(Format::Safetensors, local.location());
        assert!(caps.read_bytes && caps.modify_in_place);

        let Ok(hub) = resolve(&paths(&["hf://owner/name"]), None) else {
            panic!("a repo reference resolves");
        };
        let caps = Capabilities::of(Format::Safetensors, hub.location());
        assert!(!caps.read_bytes, "the Hub serves headers, not weights");
        assert!(
            caps.layout_map,
            "but safetensors byte ranges work over any transport"
        );
        assert!(
            !caps.modify_in_place,
            "and nothing remote is editable in place"
        );
    }
}
