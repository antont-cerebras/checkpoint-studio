//! What a checkpoint source **can do** — the two axes it depends on, and the capabilities
//! derived from them.
//!
//! A source is a pair: the **format** (safetensors, HDF5, GGUF, `NumPy`) and the **location**
//! (a local filesystem, an SFTP host, an S3 prefix behind a proxy, a Hugging Face repo).
//! Almost every feature's availability is a function of that pair, not of either half:
//! renaming needs safetensors *and* local; the byte-layout map needs safetensors over any
//! transport that can read a header; per-object metadata is S3 and only S3.
//!
//! Before this, each feature asked its own ad-hoc question — `require_local`,
//! `file_view_available`, `remote_read().is_some()`, `can_rename`, `repack_input`,
//! `open_reader` erroring on a remote path — and adding the Hugging Face location meant
//! finding all of them. Now a feature asks the capability, and a new location answers every
//! question at once by filling in one row.
//!
//! **Capabilities are facts about the source, not policy.** `read_bytes` says whether the
//! bytes are *reachable*, not whether we currently bother: HTTP `Range` on the Hub and SFTP
//! reads could both serve tensor data, and modelling that as a capability rather than as
//! "is it local" is what will let those turn on without another sweep of conditionals.
//! Where a capability is reachable-but-unimplemented, [`Capabilities::data_view_note`] says
//! so, so a
//! UI can explain the difference instead of implying the data doesn't exist.

use crate::model::{Checkpoint, Source};

/// The on-disk (or on-the-wire) format of a checkpoint's shards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    /// `.safetensors` — a JSON header with per-tensor byte ranges.
    Safetensors,
    /// HDF5 (`.h5` / `.hdf5`) — chunked, optionally compressed datasets.
    Hdf5,
    /// GGUF — llama.cpp's single-file format.
    Gguf,
    /// `NumPy` `.npy` / `.npz`.
    Numpy,
    /// More than one format in the same checkpoint, or none recognised. Capabilities take
    /// the pessimistic reading, since a feature that works for one shard and not another
    /// is worse than one that is simply unavailable.
    Mixed,
}

impl Format {
    /// Classify by file extension, the way the readers dispatch.
    #[must_use]
    pub fn of_path(path: &str) -> Option<Self> {
        let ext = path.rsplit_once('.')?.1.to_ascii_lowercase();
        match ext.as_str() {
            "safetensors" => Some(Self::Safetensors),
            "h5" | "hdf5" => Some(Self::Hdf5),
            "gguf" => Some(Self::Gguf),
            "npy" | "npz" => Some(Self::Numpy),
            _ => None,
        }
    }

    /// The one format every path in `paths` has, or [`Self::Mixed`].
    #[must_use]
    pub fn of_paths<'a>(paths: impl IntoIterator<Item = &'a str>) -> Self {
        let mut seen: Option<Self> = None;
        for p in paths {
            match (Self::of_path(p), seen) {
                (Some(f), None) => seen = Some(f),
                (Some(f), Some(prev)) if f == prev => {}
                // Two formats, or an unrecognised one alongside a known one.
                (_, _) => return Self::Mixed,
            }
        }
        seen.unwrap_or(Self::Mixed)
    }

    /// Whether this format carries per-tensor byte ranges, which the layout map needs.
    #[must_use]
    pub const fn has_byte_ranges(self) -> bool {
        matches!(self, Self::Safetensors)
    }
}

/// Where a checkpoint's bytes live — the transport half of the pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Location {
    Local,
    /// A directory on an SSH host, read over SFTP.
    Sftp,
    /// An `s3://` prefix, read by a script on the proxy host.
    S3,
    /// A Hugging Face Hub repository, read over HTTPS.
    Hf,
}

impl Location {
    #[must_use]
    pub const fn of(source: &Source) -> Self {
        match source {
            Source::Local => Self::Local,
            Source::Sftp { .. } => Self::Sftp,
            Source::S3 { .. } => Self::S3,
            Source::Hf { .. } => Self::Hf,
        }
    }

    /// The location a tensor's `source_path` implies — `hf://…`, `s3://…`, an scp-style
    /// `host:/path`, or a local path.
    ///
    /// For code that holds a tensor rather than a source (the sampler, deep in core), so it
    /// can give the *right* reason instead of assuming every remote is an ssh proxy.
    #[must_use]
    pub fn of_source_path(path: &str) -> Self {
        if path.starts_with("hf://") {
            return Self::Hf;
        }
        if path.starts_with("s3://") {
            return Self::S3;
        }
        if crate::remote::is_remote_source(path) {
            return Self::Sftp;
        }
        Self::Local
    }

    /// Whether the bytes are on this machine. Not the same question as "can we read tensor
    /// data" — see [`Capabilities::read_bytes`].
    #[must_use]
    pub const fn is_local(self) -> bool {
        matches!(self, Self::Local)
    }
}

/// How a location is reached — the axis that says whether this machine can open it at all.
///
/// Distinct from every capability below, which describe what you can *do* once the
/// checkpoint is open. This is about getting there: an `s3://` prefix is unreadable without
/// a host that holds the credentials, so the answer to "can I read this at all" is neither
/// a format nor a feature question. It was enforced by two hand-written strings — one in
/// `collect_safetensors_files`, one in the loader — which is exactly the kind of thing a
/// fourth location makes unmaintainable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reach {
    /// Openable from this machine: a local path, or a public HTTPS endpoint.
    Direct,
    /// Only through an SSH proxy that holds the access (`--ssh-proxy`). The data and the
    /// credentials stay there; only metadata comes back.
    ViaSshProxy,
}

impl Location {
    /// How this location is reached.
    #[must_use]
    pub const fn reach(self) -> Reach {
        match self {
            // A local path, and the Hub over public HTTPS.
            Self::Local | Self::Hf => Reach::Direct,
            // SFTP *is* the proxy; S3 needs one because the credentials live there.
            Self::Sftp | Self::S3 => Reach::ViaSshProxy,
        }
    }

    /// Why this location can't be opened without a proxy, phrased for a user — or `None`
    /// when it can. One sentence, so the CLI's refusal and a UI's explanation match.
    #[must_use]
    pub const fn proxy_note(self) -> Option<&'static str> {
        match self.reach() {
            Reach::Direct => None,
            Reach::ViaSshProxy => Some(
                "needs --ssh-proxy <[user@]host>: the credentials and data stay on that \
                 host, and only the checkpoint's metadata is sent back",
            ),
        }
    }
}

/// What a given (format, location) pair supports. Serializable, so a server can hand the
/// set to a browser rather than have the client re-derive it from the source's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Capabilities {
    /// Read tensor **bytes** — what the heatmap, value grid, histogram and whole-tensor
    /// statistics need. Currently local-only; see the module note on why that is a
    /// capability rather than a synonym for `is_local`.
    pub read_bytes: bool,
    /// Rewrite the checkpoint **in place** (the tensor-rename editor).
    pub modify_in_place: bool,
    /// Repack into a **new** file with a different codec (HDF5). Distinct from
    /// `modify_in_place`: it writes a copy and never touches the input.
    pub repack: bool,
    /// Show the byte-layout map — needs per-tensor byte ranges in the format, and a
    /// readable header, which every location provides.
    pub layout_map: bool,
    /// Browse the checkpoint's file tree — needs a listing, which every location has: a
    /// local checkpoint walks its containing directory (so even a single file has one), an
    /// SFTP source lists the remote dir, S3 its keys, the Hub its tree.
    ///
    /// Whether a *browser* for a given remote is implemented is a separate question about
    /// this build, not about the source, and stays with the caller.
    pub browse_files: bool,
    /// Per-object storage metadata (`ETag`, checksums, storage class) — S3 only.
    pub object_metadata: bool,
    /// Per-dataset compression codec and stored-vs-logical size — HDF5 only.
    pub codec_info: bool,
    /// How the source is reached — whether this machine can open it directly at all.
    /// Carried alongside the capabilities so a frontend gets one answer, not two lookups.
    pub reach: Reach,
}

impl Capabilities {
    /// Derive the set from the pair.
    #[must_use]
    pub const fn of(format: Format, location: Location) -> Self {
        let local = location.is_local();
        Self {
            // Only the local readers open tensor data today. The Hub could serve it by
            // `Range` and SFTP by a remote read; when they do, this row changes and every
            // caller follows, because they ask this and not "is it local".
            read_bytes: local,
            // Rewriting a header in place is a safetensors operation, and only where we
            // hold the file.
            modify_in_place: local && matches!(format, Format::Safetensors),
            repack: local && matches!(format, Format::Hdf5),
            layout_map: format.has_byte_ranges(),
            // Every location has something to list — see the field's note. This row was
            // briefly `!local || multi_file`, which broke the file browser for a
            // single-file local checkpoint; the existing badge/navigation tests caught it.
            browse_files: true,
            object_metadata: matches!(location, Location::S3),
            codec_info: matches!(format, Format::Hdf5),
            reach: location.reach(),
        }
    }

    /// Why a data view is unavailable, phrased for a user, or `None` when it is available.
    ///
    /// One sentence in one place: the terminal floats it as a notice and the browser shows
    /// it in the data pane, and they used to word it separately.
    #[must_use]
    pub fn data_view_note(location: Location) -> Option<&'static str> {
        match location {
            Location::Local => None,
            Location::Sftp | Location::S3 => Some(
                "Read remotely with --ssh-proxy: only the structure is here. Data views \
                 (heatmap, values, histogram, statistics) need the file locally — copy the \
                 checkpoint down to preview its values.",
            ),
            Location::Hf => Some(
                "Read from the Hugging Face Hub: only the structure is here — the shard \
                 headers, not the weights. Data views (heatmap, values, histogram, \
                 statistics) need the tensor bytes; download the repo to preview its values.",
            ),
        }
    }
}

impl Checkpoint {
    /// The format of this checkpoint's shards.
    #[must_use]
    pub fn format(&self) -> Format {
        Format::of_paths(self.shards.iter().map(|s| s.path.as_str()))
    }

    /// Where this checkpoint lives.
    #[must_use]
    pub const fn location(&self) -> Location {
        Location::of(&self.source)
    }

    /// What this checkpoint supports — the one question a feature should ask.
    #[must_use]
    pub fn capabilities(&self) -> Capabilities {
        Capabilities::of(self.format(), self.location())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_format_comes_from_the_extension_and_disagreement_is_mixed() {
        assert_eq!(
            Format::of_path("a/model.safetensors"),
            Some(Format::Safetensors)
        );
        assert_eq!(
            Format::of_path("m.HDF5"),
            Some(Format::Hdf5),
            "case-insensitive"
        );
        assert_eq!(Format::of_path("m.gguf"), Some(Format::Gguf));
        assert_eq!(Format::of_path("m.npz"), Some(Format::Numpy));
        assert_eq!(Format::of_path("README"), None, "no extension");

        assert_eq!(
            Format::of_paths(["a.safetensors", "b.safetensors"]),
            Format::Safetensors
        );
        // A checkpoint whose shards disagree takes the pessimistic reading: a feature that
        // works for one shard and not another is worse than one that is plainly absent.
        assert_eq!(Format::of_paths(["a.safetensors", "b.gguf"]), Format::Mixed);
        let none: [&str; 0] = [];
        assert_eq!(Format::of_paths(none), Format::Mixed, "nothing known");
    }

    /// The table this type exists to make explicit. Each row is a claim about the *pair*,
    /// which is the point — none of these follow from the format or the location alone.
    #[test]
    fn capabilities_depend_on_both_axes() {
        let st_local = Capabilities::of(Format::Safetensors, Location::Local);
        let st_hf = Capabilities::of(Format::Safetensors, Location::Hf);
        let h5_local = Capabilities::of(Format::Hdf5, Location::Local);
        let st_s3 = Capabilities::of(Format::Safetensors, Location::S3);

        // Renaming in place needs safetensors AND local — neither half suffices.
        assert!(st_local.modify_in_place);
        assert!(!st_hf.modify_in_place, "local safetensors only");
        assert!(!h5_local.modify_in_place, "hdf5 is repacked, not rewritten");
        assert!(h5_local.repack && !st_local.repack);

        // The layout map follows the FORMAT over any transport — this is the row that a
        // "remote means no" rule got wrong, and it is why the Hub can show a layout map.
        assert!(st_local.layout_map && st_hf.layout_map && st_s3.layout_map);
        assert!(!h5_local.layout_map, "hdf5 has chunks, not byte ranges");

        // Bytes are local-only today, everywhere else structure-only.
        assert!(st_local.read_bytes);
        assert!(!st_hf.read_bytes && !st_s3.read_bytes);

        // Location-specific extras.
        assert!(st_s3.object_metadata && !st_hf.object_metadata);
        assert!(h5_local.codec_info && !st_local.codec_info);
    }

    /// A tensor's path tells core which reason to give — the sampler holds a tensor, not a
    /// source, and used to assume every remote was an ssh proxy.
    #[test]
    fn a_location_can_be_read_off_a_source_path() {
        assert_eq!(
            Location::of_source_path("hf://owner/name/shard.safetensors"),
            Location::Hf
        );
        assert_eq!(Location::of_source_path("s3://bucket/key"), Location::S3);
        assert_eq!(
            Location::of_source_path("host:/opt/ckpt/shard.safetensors"),
            Location::Sftp
        );
        assert_eq!(
            Location::of_source_path("/local/shard.safetensors"),
            Location::Local
        );
        assert_eq!(
            Location::of_source_path("shard.safetensors"),
            Location::Local,
            "a bare relative name is local"
        );
    }

    /// Reachability is its own axis: `s3://` is not openable without a proxy however
    /// capable the format is, and the Hub is openable without one however remote it is.
    /// Conflating "remote" with "needs a proxy" is what the two hand-written strings did.
    #[test]
    fn reach_separates_needing_a_proxy_from_being_remote() {
        assert_eq!(Location::Local.reach(), Reach::Direct);
        assert_eq!(
            Location::Hf.reach(),
            Reach::Direct,
            "the Hub is remote but reachable over public HTTPS"
        );
        assert_eq!(Location::S3.reach(), Reach::ViaSshProxy);
        assert_eq!(Location::Sftp.reach(), Reach::ViaSshProxy);

        assert!(Location::Local.proxy_note().is_none());
        assert!(Location::Hf.proxy_note().is_none(), "no proxy needed");
        let s3 = Location::S3.proxy_note().expect("a reason");
        assert!(s3.contains("--ssh-proxy"), "it names the flag: {s3}");

        // And it travels with the capability set rather than needing a second lookup.
        assert_eq!(
            Capabilities::of(Format::Safetensors, Location::S3).reach,
            Reach::ViaSshProxy
        );
    }

    /// Every location has a listing, so browsing is always offered — a local checkpoint
    /// browses its containing directory, which is why even a single file has one. Pinned
    /// because getting this wrong removed the file browser from the commonest case there is.
    #[test]
    fn browsing_is_available_for_every_location() {
        for location in [Location::Local, Location::Sftp, Location::S3, Location::Hf] {
            assert!(
                Capabilities::of(Format::Safetensors, location).browse_files,
                "{location:?} has a listing to browse"
            );
        }
    }

    /// The unavailability message names the actual reason, and each remote kind explains
    /// itself — "copy it down" is wrong advice for a Hub repo.
    #[test]
    fn the_data_view_note_explains_the_specific_location() {
        assert_eq!(Capabilities::data_view_note(Location::Local), None);
        let hf = Capabilities::data_view_note(Location::Hf).expect("a note");
        assert!(hf.contains("Hugging Face"), "{hf}");
        assert!(hf.contains("download"), "and what to do about it: {hf}");
        let sftp = Capabilities::data_view_note(Location::Sftp).expect("a note");
        assert!(sftp.contains("--ssh-proxy"), "{sftp}");
        assert_ne!(hf, sftp, "the two reasons are not the same reason");
    }
}
