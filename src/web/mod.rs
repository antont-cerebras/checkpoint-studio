//! `--web`: a headless HTTP server (sync/blocking, no async runtime) that serves
//! the checkpoint as JSON — the **data** — plus the embedded Svelte UI, which owns
//! its own **view state**.
//!
//! Remote sources work the same way the TUI's do: `web --ssh-proxy` reads the structure
//! over SSH (an `s3://` URI or a remote path) and serves it. What a remote read cannot
//! offer is the same in both frontends — the data views need the bytes, so the heatmap,
//! value grid, histogram and whole-tensor scan return a 400 explaining that (see
//! `handlers::require_local`), exactly as the terminal shows its `metadata-only` badge.
//!
//! `WebState` is read once at startup and shared read-only across worker threads
//! (`Arc`); every derived view/report is precomputed so request handling needs no
//! `&mut` (only the on-demand tensor-data scans touch disk, behind a small cache).

mod assets;
/// The checkpoint currently being served, and how to switch it without a restart.
pub(crate) mod current;
#[cfg(test)]
mod current_tests;
pub(crate) mod diffscope;
pub(crate) mod dto;
pub(crate) mod handlers;
pub(crate) mod jobs;
pub(crate) mod params;
pub(crate) mod repackjob;
pub(crate) mod valuesjob;

pub(crate) use current::Current;

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::{
    check, filetree, filter, health, kernel, model, remote, safelayout, sample, stats, tree,
};
use handlers::{Query, Reply};

/// Everything the API serves, computed once from a local read. Shared read-only.
pub(crate) struct WebState {
    /// The checkpoint's **display** root: the containing directory for a single-file
    /// checkpoint. Good for a tree label; wrong as an address.
    pub root: String,
    /// What was **opened** to get this checkpoint, in the durable spelling
    /// ([`crate::opening::recorded_spec`]) — an absolute path, `host:/path`, or a URI.
    ///
    /// Distinct from `root`, and it has to be: a directory of three HDF5 checkpoints has one
    /// `root` and three specs. The browser's address bar and the `?ckpt=` a link carries are both
    /// *addresses*, so they need this — showing `root` there offered a path that, on Enter, would
    /// have opened something else.
    pub spec: String,
    /// The no-access-control caution to show in the UI, or `None` when the server is
    /// bound to loopback and so only reachable from this machine. Set by
    /// [`Self::with_exposure`] from the bind address; the *same string* the startup
    /// banner prints, so the terminal and the page cannot say different things.
    pub access_warning: Option<String>,
    /// The checkpoint's files as resolved at startup. Kept so `/api/diff` can emit the
    /// `checkpoint-studio diff OLD NEW` command that reproduces its report: `root` is the
    /// containing *directory* for a single-file checkpoint, which would name a different
    /// (larger) comparison than the one being served.
    pub files: Vec<PathBuf>,
    /// The full serializable model (backs `/api/model`).
    pub checkpoint: model::Checkpoint,
    /// The tensor-tree hierarchy (client folds/selects/searches it).
    pub tree: Vec<tree::TreeNode>,
    pub file_tree: dto::WebFileNode,
    pub stats: stats::CheckpointStats,
    pub health: Vec<health::HealthReport>,
    /// `source_path`s of tensors on disk but not listed in the index, **sorted** — the
    /// browser marks those tree rows with it, exactly as the terminal does. Sorted
    /// because a `HashSet`'s order is randomised per process, and this body is encoded
    /// once and cached: an arbitrary order would make the same checkpoint serve a
    /// different `/api/tree` on every restart.
    pub unindexed: Vec<String>,
    pub check: Option<check::CheckReport>,
    pub layouts: Vec<safelayout::LayoutMap>,
    /// Canonical (deduped, natural-sorted) tensors, for detail + data-view lookup.
    pub tensors: Vec<tree::TensorInfo>,
    tensor_index: HashMap<String, usize>,
    schemas: HashMap<String, sample::PackingSchema>,
    /// Per-`(tensor, view)` whole-tensor stats, memoized (also feeds histogram range).
    stats_cache: Mutex<HashMap<(String, String), dto::StatsDto>>,
    /// Fully-encoded bodies for the endpoints whose content is fixed for the process
    /// lifetime, keyed by `(endpoint, gzipped)`.
    ///
    /// The checkpoint is read once, so these never change — yet `/api/tree` (14 MB of
    /// JSON for a 31k-tensor checkpoint) was rebuilt AND re-gzipped on every request:
    /// ~310 ms of CPU and ~250 MB of transient allocation each time. Freed arenas
    /// aren't returned to the OS, so resident memory climbed ~250 MB per page load
    /// (measured: 110 MB after startup, 2.08 GB after eight loads). Encoding once
    /// makes repeat loads essentially free and keeps memory flat.
    static_bodies: StaticBodies,
}

/// Endpoints derived purely from the read-once model, so their encoded body can be
/// cached. Everything else takes query parameters or reads tensor bytes on demand.
const STATIC_ENDPOINTS: &[&str] = &["tree", "files", "stats", "health", "check", "model"];

/// A cached response: the bytes as they go on the wire, plus how long the body is
/// *before* any encoding.
///
/// The identity length is the one number a browser can't work out for itself. Under
/// `Content-Encoding: gzip` the `Content-Length` it sees is the compressed size, while the
/// stream it reads is decoded — so a client counting bytes to draw a progress bar has a
/// numerator in one unit and a denominator in another. Announcing this makes the fraction
/// real (see `X-Uncompressed-Length`).
pub(crate) struct CachedBody {
    pub bytes: Vec<u8>,
    pub identity_len: u64,
}

/// `(endpoint, gzipped)` -> the fully-encoded response body.
type StaticBodies = Mutex<HashMap<(&'static str, bool), Arc<CachedBody>>>;

impl WebState {
    /// Build the shared state from a local checkpoint read. `files`/`index_specs`
    /// are what `run_explore` already resolved (for the structural check + health).
    pub(crate) fn build(
        checkpoint: model::Checkpoint,
        files: &[PathBuf],
        index_specs: &[health::IndexSpec],
    ) -> Self {
        let root = checkpoint.root.clone();
        let disk = checkpoint.disk_usage();

        // Canonicalize through a Session (dedup + natural sort) so the tree and
        // tensor list match the TUI exactly. Clone the model so we keep it for
        // `/api/model`; `stats_with_disk` needs `&mut`, so call it last.
        let mut session = kernel::Session::from_model(checkpoint.clone());
        let tensors: Vec<tree::TensorInfo> = session.tensors().to_vec();
        let metadata: Vec<tree::MetadataInfo> = session.metadata().to_vec();
        let config = session.config().cloned();
        // The whole tree, summarising root included — the same call the TUI makes, so
        // the two surfaces cannot disagree about the tree or its label.
        let tree = session.build_rooted_tree(files);
        // The S3 section too, for an `s3://` model — the same projection the TUI uses,
        // so the browser's stats screen isn't missing a section the terminal shows.
        let checkpoint_stats = session
            .stats_with_disk(disk)
            .clone()
            .with_s3(checkpoint.s3.as_ref().map(remote::S3Meta::to_stats));

        let tensor_index = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        let schemas = sample::parse_packing_schemas(&tensors, &metadata);

        // A local read walks the directory; a remote one has no local directory to walk, so
        // the tree comes from the listing the read already returned (`Checkpoint::files` —
        // S3 object keys, or the SFTP shard listing). Without this the browser's Files
        // screen was empty for every `--ssh-proxy` source while the terminal listed it.
        // Each shard is annotated with the tensors read from it, so a browsed listing of
        // sixteen same-sized shards says which one holds what (see `ShardTensors`).
        let file_tree = if matches!(checkpoint.source, model::Source::Local) {
            let mut node = filetree::build(Path::new(&root), 8);
            node.attribute_tensors(&tensors);
            node.attribute_index(&checkpoint.index);
            node.attribute_read_errors(&checkpoint.unreadable);
            dto::WebFileNode::from_node(&node, Path::new(&root))
        } else {
            // The model's file listing, which carries each entry's link count for an
            // `--ssh-proxy` read (an S3 object reports one name — no inode to share).
            let objects: Vec<filetree::ObjectEntry> = checkpoint
                .files
                .iter()
                .map(|f| filetree::ObjectEntry {
                    key: f.rel_path.clone(),
                    size: f.apparent(),
                    links: f.node.links(),
                })
                .collect();
            let label = root
                .trim_end_matches('/')
                .rsplit('/')
                .find(|s| !s.is_empty())
                .unwrap_or(&root)
                .to_string();
            let mut node = filetree::build_from_keys(&label, &objects);
            node.attribute_tensors(&tensors);
            node.attribute_index(&checkpoint.index);
            node.attribute_read_errors(&checkpoint.unreadable);
            dto::WebFileNode::from_node(&node, Path::new(""))
        };

        let mut health: Vec<health::HealthReport> = index_specs
            .iter()
            .map(|spec| health::check_loaded(spec, &tensors))
            .collect();
        // An `s3://` source has no index.json to reconcile, but it does describe every
        // tensor twice — in the checkpoint index and in each object's own metadata — so
        // cross-check those, exactly as the TUI does. Derived from the model we were
        // handed, so any caller that has the object metadata gets the check.
        if let Some(s3) = &checkpoint.s3 {
            health.push(health::check_s3_correspondence(&root, &tensors, s3));
        }

        // Structural check only (values = false → no byte scan at startup).
        // A literal empty filter: `parse` only fails on a malformed pattern, and there
        // are none.
        #[allow(clippy::expect_used)]
        let all = filter::NameFilter::parse(&[]).expect("empty NameFilter is valid");
        let check = Some(check::run(
            root.clone(),
            &tensors,
            &metadata,
            files,
            &health,
            config.as_ref(),
            &all,
            check::HeaderInputs::from(&checkpoint),
            false,
            1,
        ));

        let mut unindexed: Vec<String> = health::unindexed_files(&health).into_iter().collect();
        unindexed.sort();

        let layouts = checkpoint
            .shards
            .iter()
            .map(|sh| {
                safelayout::from_tensors(
                    &sh.path,
                    sh.total_len,
                    sh.header_len,
                    &sh.tensors,
                    &sh.metadata,
                )
            })
            .collect();

        Self {
            root,
            // Set by `with_spec`; the display root until then, which is right for the callers
            // that have no separate spec (a test building a state directly).
            spec: String::new(),
            // Loopback until told otherwise, so a state built in a test or a headless
            // context never claims to be exposed.
            access_warning: None,
            files: files.to_vec(),
            checkpoint,
            tree,
            file_tree,
            stats: checkpoint_stats,
            health,
            unindexed,
            check,
            layouts,
            tensors,
            tensor_index,
            schemas,
            stats_cache: Mutex::new(HashMap::new()),
            static_bodies: Mutex::new(HashMap::new()),
        }
    }

    /// Record what was opened to produce this state — see [`Self::spec`].
    pub(crate) fn with_spec(mut self, spec: String) -> Self {
        self.spec = spec;
        self
    }

    /// Record how the server is bound. A non-loopback bind means anyone who can reach
    /// the port gets everything this UI serves — there is no authentication — so the UI
    /// says so. Loopback leaves it `None` and no banner appears.
    pub(crate) fn with_exposure(mut self, host: IpAddr) -> Self {
        self.access_warning = (!host.is_loopback()).then(|| access_warning(host));
        self
    }

    /// The encoded body for a fixed-content endpoint, building (and caching) it on
    /// first use. `None` for endpoints that aren't cacheable.
    fn cached_body(&self, api: &str, gzipped: bool) -> Option<Arc<CachedBody>> {
        let name = STATIC_ENDPOINTS.iter().copied().find(|&e| e == api)?;
        // Bind the lookup to a local so the guard is dropped before the build below,
        // rather than living to the end of an `if let`.
        // Poison-tolerant: this is a pure memo of immutable data, and `.ok()?` would
        // silently disable the cache for the rest of the process — reintroducing the
        // 250 MB-per-request rebuild this cache exists to prevent.
        let hit = self
            .static_bodies
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&(name, gzipped))
            .map(Arc::clone);
        if let Some(body) = hit {
            return Some(body);
        }
        // Build with the lock RELEASED: encoding `/api/tree` takes ~150 ms and holding
        // the mutex across it would stall every other request behind it. Two racing
        // first-requests may both build; the loser's copy is just dropped.
        let (status, json) = match name {
            "tree" => handlers::tree(self),
            "files" => handlers::files(self),
            "stats" => handlers::stats(self),
            "health" => handlers::health(self),
            "check" => handlers::check(self),
            "model" => handlers::model(self),
            _ => return None,
        };
        if status != 200 {
            return None; // don't cache an error; let the normal path report it
        }
        let identity_len = json.len() as u64;
        let body = if gzipped {
            gzip_bytes(&json).ok()?
        } else {
            json
        };
        let body = Arc::new(CachedBody {
            bytes: body,
            identity_len,
        });
        self.static_bodies
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert((name, gzipped), Arc::clone(&body));
        Some(body)
    }
}
/// Reserve the server socket **up front**, before any slow work (e.g. a 5–10 s
/// remote read), so a port clash is reported in milliseconds rather than after the
/// wait. `host` is the bind address (default `0.0.0.0`). If a *specific* requested
/// port is already in use, fall back to an OS-assigned free port so the command
/// still comes up — and warn where it landed (so a stale server, or a chosen port,
/// doesn't just fail).
pub(crate) fn bind(host: IpAddr, port: u16) -> Result<tiny_http::Server> {
    match tiny_http::Server::http(SocketAddr::new(host, port)) {
        Ok(server) => Ok(server),
        Err(e) if port != 0 => {
            let server = tiny_http::Server::http(SocketAddr::new(host, 0))
                .map_err(|e2| anyhow::anyhow!("failed to start web server on {host}: {e2}"))?;
            let freed = server.server_addr().to_ip().map_or(0, |a| a.port());
            eprintln!(
                "checkpoint-studio: port {port} is already in use ({e}) — serving on free \
                 port {freed} instead (use --port to pick another, or free {port} first)."
            );
            Ok(server)
        }
        Err(e) => Err(anyhow::anyhow!(
            "failed to start web server on {host}:{port}: {e}"
        )),
    }
}

/// Serve on an already-[`bind`]-ed socket and block until stopped. `host` is only
/// used to render a reachable URL (a wildcard bind isn't clickable).
pub(crate) fn serve_on(
    server: tiny_http::Server,
    current: Arc<Current>,
    host: IpAddr,
) -> Result<()> {
    let bound = server.server_addr().to_ip().map_or(0, |a| a.port());
    // Print a URL the browser can actually reach: a wildcard bind (0.0.0.0 / ::)
    // isn't clickable, so show this host's FQDN instead of the bind address.
    let display = if host.is_unspecified() {
        fqdn().unwrap_or_else(|| "localhost".to_string())
    } else {
        host.to_string()
    };
    let url = format!("http://{display}:{bound}/");
    print_serve_banner(&url, host);

    // A small worker pool so a static-asset / metadata request stays responsive
    // while another worker is inside a multi-second tensor scan.
    let server = Arc::new(server);
    let workers = std::thread::available_parallelism()
        .map_or(4, std::num::NonZero::get)
        .clamp(2, 8);
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let server = Arc::clone(&server);
        let current = Arc::clone(&current);
        handles.push(std::thread::spawn(move || {
            while let Ok(req) = server.recv() {
                handle(&current, req);
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

/// Announce the running server with the **URL as the focal point** — bold,
/// underlined, bright cyan on its own padded line — so it stands out even when a
/// coloured load-progress bar was printed just above it. Plain (unstyled, one line)
/// when stdout isn't a terminal or `NO_COLOR` is set, so a captured log stays clean.
fn print_serve_banner(url: &str, host: IpAddr) {
    const DIM: &str = "\x1b[2m";
    const URL: &str = "\x1b[1;4;96m"; // bold + underline + bright cyan (link-like)
    const WARN: &str = "\x1b[38;5;210m"; // light red — a caution, not an error
    const RESET: &str = "\x1b[0m";
    use std::io::IsTerminal;
    let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let exposed = !host.is_loopback();
    if !color {
        println!("checkpoint-studio web UI: {url}  (Ctrl-C to stop)");
        if exposed {
            println!("{}", access_warning(host));
        }
        return;
    }
    // Raw ANSI via named consts + a terminal/NO_COLOR guard — the same convention
    // the other one-shot CLI status lines use (e.g. `sftp`'s retry notice, the
    // `diff: done` footer, `progress`'s bars). `yansi` is reserved for the TUI layer.
    // Blank line to break from the finished ✓ load bar above; the label dim, the URL
    // the one bright thing on the line so it wins the eye.
    println!(
        "\n  {DIM}checkpoint-studio web UI ▸{RESET}  {URL}{url}{RESET}\n  {DIM}Ctrl-C to stop{RESET}"
    );
    // The server has no authentication of any kind. On a wildcard or specific
    // non-loopback bind that means anyone who can reach the port gets everything this
    // UI can show — including `POST /api/compare?left=PATH`, which will read any checkpoint
    // path the serving user can read. Light red: worth reading, not a failure.
    if exposed {
        println!("  {WARN}{}{RESET}\n", access_warning(host));
    } else {
        println!();
    }
}

/// The no-access-control caution, shared by the terminal banner and (via
/// `/api/tree`'s envelope) the banner the web page shows.
fn access_warning(host: IpAddr) -> String {
    let where_ = if host.is_unspecified() {
        "all interfaces"
    } else {
        "a network interface"
    };
    format!(
        "⚠ No access control: bound to {where_}, so anyone who can reach this port can \
         read this checkpoint — and any checkpoint path this user can read (see /api/compare). \
         Use --host 127.0.0.1 to restrict it to this machine."
    )
}

const JSON_CT: &str = "application/json; charset=utf-8";

/// A response body, either freshly built or shared from the fixed-content cache. Shared
/// so a repeat `/api/tree` hands out an `Arc` instead of copying 14 MB.
enum Body {
    Owned(Vec<u8>),
    Shared(Arc<CachedBody>),
}

impl Body {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v,
            Self::Shared(a) => &a.bytes,
        }
    }
}

/// A fully-resolved response, computed before anything is written to the socket — which
/// is what lets the computation run inside a panic boundary (see `handle`).
struct Prepared {
    status: u16,
    body: Body,
    content_type: &'static str,
    gzipped: bool,
    cache_control: Option<&'static str>,
    /// The body's length before encoding, when it differs from what `Content-Length` will
    /// say — i.e. when the response is gzipped. Sent as `X-Uncompressed-Length` so a client
    /// reading the decoded stream can draw a progress bar against the right total.
    identity_len: Option<u64>,
}

/// `&Arc<Current>` rather than `&Current`: starting a job hands a clone to a worker thread that outlives
/// the request, so the reference count has to be reachable from here.
fn handle(current: &Arc<Current>, req: tiny_http::Request) {
    let url = req.url().to_string();
    let gzip = accepts_gzip(&req);
    // Every endpoint but one is a read, and a read is a GET. `/api/open` changes what the
    // whole server serves, so it is a POST — see `route_api`.
    let method = req.method().clone();
    // Contain a panic. The worker pool is small (2-8 threads) and each worker loops on
    // `server.recv()`, so an unwinding handler would kill that worker permanently —
    // after a handful of bad requests the process would still accept connections and
    // answer none: alive, but silently hung. (We shipped exactly such a handler
    // earlier: a histogram bin-count overflow that indexed an empty vector.) Resolving
    // the response before touching the socket means a panic costs one 500, not a
    // worker.
    let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (path, query_str) = url.split_once('?').unwrap_or((url.as_str(), ""));
        prepare(current, &method, path, query_str, gzip)
    }))
    .unwrap_or_else(|_| {
        // The default panic hook has already printed the message and location; add the
        // request that triggered it, which the hook doesn't know.
        eprintln!(
            "checkpoint-studio web: panic while serving {url} — replied 500; the server is still running"
        );
        Prepared {
            status: 500,
            body: Body::Owned(
                br#"{"error":"internal error handling this request (see the server log)"}"#
                    .to_vec(),
            ),
            content_type: JSON_CT,
            gzipped: false,
            cache_control: Some("no-store"),
            identity_len: None,
        }
    });
    send_encoded(
        req,
        prepared.status,
        prepared.body.as_slice(),
        prepared.content_type,
        prepared.gzipped,
        prepared.cache_control,
        prepared.identity_len,
    );
}

/// Resolve a request to bytes. Pure: touches no socket, so it is safe to run inside the
/// panic boundary above.
fn prepare(
    current: &Arc<Current>,
    method: &tiny_http::Method,
    path: &str,
    query_str: &str,
    gzip: bool,
) -> Prepared {
    let Some(api) = path.strip_prefix("/api/") else {
        return prepare_asset(path, gzip);
    };
    let q = parse_query(query_str);
    // One snapshot for this request, taken before any work: the checkpoint can be swapped
    // (see `crate::web::current`) while a scan is running, and an answer assembled from two
    // different checkpoints would be worse than a slightly stale one.
    let state = current.snapshot();
    // The API reflects one read-once checkpoint; a browser must never reuse a response
    // from a prior server run — or from before an `/api/open` swapped the checkpoint on
    // this very port — hence `no-store`. Server-side we still reuse it: a fixed-content
    // endpoint is encoded once per checkpoint and handed out as bytes (`cached_body`),
    // and the cache lives inside the state so a swap discards it.
    if q.is_empty()
        && let Some(body) = state.cached_body(api, gzip)
    {
        let identity_len = gzip.then_some(body.identity_len);
        return Prepared {
            status: 200,
            body: Body::Shared(body),
            content_type: JSON_CT,
            gzipped: gzip,
            cache_control: Some("no-store"),
            identity_len,
        };
    }
    let (status, data) = route_api(current, &state, method, api, &q);
    let identity_len = data.len() as u64;
    let (body, gzipped) = maybe_gzip(data, gzip);
    Prepared {
        status,
        body: Body::Owned(body),
        content_type: JSON_CT,
        gzipped,
        cache_control: Some("no-store"),
        identity_len: gzipped.then_some(identity_len),
    }
}

/// The selection parameters both diff routes take — exactly what
/// [`diffscope::DiffScope::from_query`] reads.
/// The selection parameters, from the one table that also renders them as `diff` flags — see
/// [`params`]. A hand-written second list is how a control comes to be accepted and never rendered.
fn scope_params() -> Vec<&'static str> {
    params::keys(params::SCOPE)
}

/// Which shared parameter tables an endpoint takes on top of its own — see [`params`].
///
/// Named tables rather than a `bool`, so the accepted set and the rendered command come from the same
/// rows: a route that renders a check's flags must accept that check's parameters, and one list cannot
/// drift from the other.
enum Scoped {
    /// The selection only — every route that answers *about* a comparison.
    Yes,
    /// The selection and the check: the route that renders a terminal invocation.
    AndCheck,
    No,
}

/// What each endpoint accepts, so anything else is a refusal rather than a silent no-op.
///
/// A mistyped parameter is not a harmless extra: `?nmae=model.layers.1.*` is not a filter, it is a
/// request for the *whole* comparison — answered with a confident `200` and 117,664 rows, which reads
/// as "your filter matched everything". `clap` refuses an unknown `--flag` on the command line for
/// exactly this reason, and the API is the same surface with the same typos available.
///
/// `None` means "no route this table knows about": leave it to the router, which answers for the path
/// itself rather than complaining about the parameters of an endpoint that does not exist.
fn accepted_params(path: &str) -> Option<(&'static [&'static str], Scoped)> {
    Some(match path {
        "open" => (&["path", "stop_other"], Scoped::No),
        // GET reads the list; DELETE names the entry to drop.
        "recents" => (&["path"], Scoped::No),
        "compare" => (&["left", "right", "stop_other"], Scoped::No),
        // `full` says the reader turned family folding off — every layer as its own row.
        "difftree" => (&["id", "full"], Scoped::Yes),
        // Both diff views work from the comparison slot, so both take its id: `swap` reads the pair
        // the other way round, and `full` says the reader expanded the families, which the offered
        // command has to carry.
        "diff" => (&["id", "swap", "full"], Scoped::Yes),
        // Only the *alignment* half of the scope changes this answer, but the client sends the scope
        // it holds and a stricter list here would refuse a request that means the same thing.
        "diffnames" => (&["id", "q", "limit"], Scoped::Yes),
        // No scope: the answer is what a re-root would be applied *to*, so applying one here would
        // offer prefixes of prefixes.
        "subtrees" => (&["id", "side", "q", "limit"], Scoped::No),
        // The two operands; the checks come from the table this route renders from.
        "command" => (&["left", "right"], Scoped::AndCheck),
        // The packing schemas arrive with the scope: they are how a side is *decoded* before anything is
        // compared, which is the scope's job (`params::SCOPE`).
        "jobs/verify-repack" => (&["left", "right", "repack_bits"], Scoped::Yes),
        "jobs/values" => (
            &[
                "left",
                "right",
                "values",
                "histogram",
                "bins",
                "dtype",
                "jobs",
                "tensor",
            ],
            Scoped::Yes,
        ),
        "tree" | "files" | "stats" | "health" | "check" | "model" | "reading" | "version" => {
            (&[], Scoped::No)
        }
        "filter" | "compact" | "schema" => (&["q"], Scoped::No),
        "tensor" => (&["name"], Scoped::No),
        "layout" => (&["file"], Scoped::No),
        "file" => (&["path"], Scoped::No),
        "tensor/stats" => (&["name", "dtype"], Scoped::No),
        "tensor/sample" => (
            &[
                "name", "dtype", "mode", "rows", "cols", "slice", "row_off", "col_off", "row_tail",
                "col_tail", "raw",
            ],
            Scoped::No,
        ),
        "tensor/histogram" => (&["name", "dtype", "bins"], Scoped::No),
        // `jobs/<id>` — polling and stopping take no parameters. An id that is not a number is the
        // router's 404, not a parameter complaint.
        other => {
            other.strip_prefix("jobs/")?.parse::<u64>().ok()?;
            (&[], Scoped::No)
        }
    })
}

/// A `400` naming every parameter this endpoint does not take, or `None` when they all check out.
fn unknown_params(path: &str, q: &Query) -> Option<Reply> {
    let (own, scoped) = accepted_params(path)?;
    let scope: Vec<&str> = match scoped {
        Scoped::Yes => scope_params(),
        Scoped::AndCheck => {
            let mut all = scope_params();
            all.extend(params::keys(params::CHECK));
            all
        }
        Scoped::No => Vec::new(),
    };
    let mut unknown: Vec<&str> = q
        .keys()
        .map(String::as_str)
        .filter(|k| !own.contains(k) && !scope.contains(k))
        .collect();
    if unknown.is_empty() {
        return None;
    }
    unknown.sort_unstable();
    let mut accepted: Vec<&str> = own.iter().copied().chain(scope).collect();
    accepted.sort_unstable();
    let named = unknown.join(", ");
    Some(handlers::err(
        400,
        format!(
            "unknown query parameter{}: {named} — /api/{path} accepts {}",
            if unknown.len() == 1 { "" } else { "s" },
            if accepted.is_empty() {
                "no parameters".to_string()
            } else {
                accepted.join(", ")
            }
        ),
    ))
}

fn route_api(
    current: &Arc<Current>,
    s: &WebState,
    method: &tiny_http::Method,
    path: &str,
    q: &Query,
) -> Reply {
    // A parameter this endpoint does not take is a mistake worth naming, not something to ignore —
    // see `accepted_params`. Checked before dispatch so every route gets it, including the ones
    // routed by verb below.
    if let Some(refusal) = unknown_params(path, q) {
        return refusal;
    }
    // The one endpoint that changes server state gets the one non-GET method. A GET that
    // swapped the served checkpoint would be a link a browser could follow on its own — a
    // prefetch, a history restore, a crawler — and the checkpoint would change with nobody
    // having asked.
    if path == "open" {
        return if *method == tiny_http::Method::Post {
            handlers::open(current, q)
        } else {
            handlers::err(
                405,
                "opening a checkpoint changes what this server serves — use POST /api/open?path=…",
            )
        };
    }
    // Long-running work: started with POST, polled with GET, stopped with DELETE. `jobs/verify-repack`
    // is the start route; `jobs/<id>` is one job.
    if let Some(rest) = path.strip_prefix("jobs/") {
        #[allow(clippy::wildcard_enum_match_arm)] // foreign enum; see FOREIGN_ENUM_WILDCARDS
        return match (rest, method) {
            ("verify-repack", tiny_http::Method::Post) => handlers::start_verify_repack(current, q),
            ("values", tiny_http::Method::Post) => handlers::start_values(current, q),
            ("verify-repack" | "values", _) => handlers::err(
                405,
                "starting a job changes what this server is doing — use \
                 POST /api/jobs/verify-repack or POST /api/jobs/values",
            ),
            (id, m) => id.parse::<u64>().map_or_else(
                |_| handlers::err(404, format!("no such job route: jobs/{id}")),
                |id| match *m {
                    tiny_http::Method::Get => handlers::job_status(current, id),
                    tiny_http::Method::Delete => handlers::cancel_job(current, id),
                    _ => handlers::err(405, "GET /api/jobs/<id> to poll, DELETE to stop it"),
                },
            ),
        };
    }
    // The baseline is set with POST and dropped with DELETE — the same verb rule as the rest: a
    // GET never changes what the server holds.
    if path == "compare" {
        #[allow(clippy::wildcard_enum_match_arm)] // foreign enum; see FOREIGN_ENUM_WILDCARDS
        return match *method {
            tiny_http::Method::Post => handlers::set_comparison(current, q),
            tiny_http::Method::Delete => handlers::clear_comparison(current),
            _ => handlers::err(
                405,
                "POST /api/compare?left=…&right=… to set a comparison up, DELETE to drop it",
            ),
        };
    }
    // The recents list is read with GET and pruned with DELETE; anything else on it is a 405, so
    // a mistyped method cannot fall through to the read and look like it worked.
    if path == "recents" {
        // A wildcard over `tiny_http::Method` — a foreign enum with a dozen variants that gains
        // more between releases, and every one of them means the same thing here: not a method
        // this endpoint answers. Same reasoning as FOREIGN_ENUM_WILDCARDS in `explorer::mod`.
        #[allow(clippy::wildcard_enum_match_arm)]
        return match *method {
            tiny_http::Method::Get => handlers::recents(current),
            tiny_http::Method::Delete => handlers::forget_recent(current, q),
            _ => handlers::err(
                405,
                "GET /api/recents to read the list, DELETE /api/recents?path=… to drop an entry",
            ),
        };
    }
    match path {
        "tree" => handlers::tree(s),
        // How the read someone is waiting on is going. Polled, so it must stay cheap: it reads two
        // atomics and a clone of the spec.
        "reading" => handlers::reading(current),
        // Which build this is, for a tab checking whether it has gone stale.
        "version" => handlers::version(s),
        // The two checkpoints aligned into one tree — the whole side-by-side in one response.
        "difftree" => handlers::difftree(current, s, q),
        // The names both sides share, for the exact-name picker in the comparison settings.
        "diffnames" => handlers::diff_names(current, q),
        // One side's namespaces, for the subtree pickers.
        "subtrees" => handlers::subtrees(current, q),
        // The terminal invocation for whatever parameters the caller holds — one renderer for every
        // surface that offers a command.
        "command" => handlers::command(q),
        "files" => handlers::files(s),
        "filter" => handlers::filter(s, q),
        "compact" => handlers::compact(s, q),
        "schema" => handlers::schema(s, q),
        "stats" => handlers::stats(s),
        "health" => handlers::health(s),
        "check" => handlers::check(s),
        "model" => handlers::model(s),
        "tensor" => handlers::tensor(s, q),
        "layout" => handlers::layout(s, q),
        "file" => handlers::file(s, q),
        "diff" => handlers::diff(current, q),
        "tensor/stats" => handlers::tensor_stats(s, q),
        "tensor/sample" => handlers::tensor_sample(s, q),
        "tensor/histogram" => handlers::tensor_histogram(s, q),
        other => handlers::err(404, format!("no such endpoint: /api/{other}")),
    }
}

/// Parse `k=v&k=v` into a map, percent-decoding each **value** (tensor names carry
/// `/` and `.`, which the client sends `encodeURIComponent`-ed).
fn parse_query(qs: &str) -> Query {
    let mut map = HashMap::new();
    for pair in qs.split('&').filter(|p| !p.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        let val = percent_encoding::percent_decode_str(v)
            .decode_utf8_lossy()
            .into_owned();
        map.insert(k.to_string(), val);
    }
    map
}

fn prepare_asset(path: &str, gzip: bool) -> Prepared {
    let rel = path.trim_start_matches('/');
    let rel = if rel.is_empty() { "index.html" } else { rel };
    // Serve the asset, else fall back to index.html (client-side routing). Vite
    // fingerprints everything under `assets/` (index-<hash>.js), so those are
    // immutable-forever; index.html (and any SPA fallback) must revalidate every
    // load, else a redeploy is invisible until a hard refresh — the cause of more
    // than one "tested a stale bundle" review.
    let (data, ctype, cache) = match assets::WebAssets::get(rel) {
        Some(f) if rel.starts_with("assets/") => (
            f.data.into_owned(),
            assets::content_type(rel),
            "public, max-age=31536000, immutable",
        ),
        Some(f) => (f.data.into_owned(), assets::content_type(rel), "no-cache"),
        // A miss under `assets/` is a missing *fingerprinted* file — a stale
        // index.html asking for a bundle a redeploy removed. Returning the SPA shell
        // there hands the browser HTML to parse as JavaScript; a 404 lets it fail
        // cleanly (and index.html is `no-cache`, so the next load self-heals).
        None if rel.starts_with("assets/") => {
            return Prepared {
                status: 404,
                body: Body::Owned(format!("no such asset: {rel}").into_bytes()),
                content_type: "text/plain; charset=utf-8",
                gzipped: false,
                cache_control: Some("no-cache"),
                identity_len: None,
            };
        }
        None => match assets::WebAssets::get("index.html") {
            Some(f) => (
                f.data.into_owned(),
                assets::content_type("index.html"),
                "no-cache",
            ),
            None => {
                return Prepared {
                    status: 404,
                    body: Body::Owned(
                        b"web UI not built \xe2\x80\x94 run `cd web && npm ci && npm run build`"
                            .to_vec(),
                    ),
                    content_type: "text/plain; charset=utf-8",
                    gzipped: false,
                    cache_control: Some("no-cache"),
                    identity_len: None,
                };
            }
        },
    };
    let (body, gzipped) = maybe_gzip(data, gzip);
    Prepared {
        status: 200,
        body: Body::Owned(body),
        content_type: ctype,
        gzipped,
        cache_control: Some(cache),
        identity_len: None,
    }
}

/// gzip the body when the client accepts it and the payload is big enough to be worth
/// it (the tensor-tree JSON is tens of MB). Returns the body and whether it's encoded.
fn maybe_gzip(data: Vec<u8>, gzip: bool) -> (Vec<u8>, bool) {
    if gzip && data.len() > 1024 {
        gzip_bytes(&data).map_or((data, false), |compressed| (compressed, true))
    } else {
        (data, false)
    }
}

/// Send a body that is ALREADY in its final encoding — used for the cached
/// fixed-content responses, which are stored pre-gzipped so a repeat request costs
/// neither a re-serialise nor a re-compress.
fn send_encoded(
    req: tiny_http::Request,
    status: u16,
    body: &[u8],
    content_type: &str,
    gzipped: bool,
    cache_control: Option<&str>,
    identity_len: Option<u64>,
) {
    let mut headers = vec![header("Content-Type", content_type)];
    if let Some(cc) = cache_control {
        headers.push(header("Cache-Control", cc));
    }
    // Which build of the UI this binary serves, on **every** response.
    //
    // The tab already asks `/api/version` when it starts and whenever it is brought back to the front,
    // and that missed the case this project produces constantly: the tab is open and *watched* while
    // the server is reinstalled under it. Nothing brought it to the front, so nothing re-asked, and the
    // page went on running an older interface with no sign of it. A header costs no request — the first
    // thing the tab asks the new server for tells it.
    if let Some(id) = assets::build_id() {
        headers.push(header("X-App-Build", &id));
    }
    if gzipped {
        headers.push(header("Content-Encoding", "gzip"));
    }
    // What the body weighs *before* encoding. `Content-Length` describes the compressed
    // bytes while a browser reads the decoded stream, so this is the only denominator a
    // client can count against — see `CachedBody`.
    if let Some(len) = identity_len {
        headers.push(header("X-Uncompressed-Length", &len.to_string()));
    }
    let mut resp = tiny_http::Response::from_data(body).with_status_code(status);
    for h in headers {
        resp = resp.with_header(h);
    }
    let _ = req.respond(resp);
}

fn header(key: &str, value: &str) -> tiny_http::Header {
    // Every caller passes a literal header name and an ASCII value.
    #[allow(clippy::expect_used)]
    tiny_http::Header::from_bytes(key.as_bytes(), value.as_bytes()).expect("valid header")
}

fn gzip_bytes(data: &[u8]) -> std::io::Result<Vec<u8>> {
    use flate2::{Compression, write::GzEncoder};
    use std::io::Write;
    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(data)?;
    encoder.finish()
}

fn accepts_gzip(req: &tiny_http::Request) -> bool {
    req.headers().iter().any(|h| {
        h.field.equiv("Accept-Encoding") && h.value.as_str().to_ascii_lowercase().contains("gzip")
    })
}

/// This machine's fully-qualified hostname (`hostname -f`), for the reachable URL
/// when bound to all interfaces. `None` if it can't be determined.
fn fqdn() -> Option<String> {
    let out = std::process::Command::new("hostname")
        .arg("-f")
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let name = String::from_utf8(out.stdout).ok()?.trim().to_string();
    if name.is_empty() { None } else { Some(name) }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{Current, Prepared, maybe_gzip, parse_query, prepare};

    /// The request layer: routing, the JSON/asset split, caching, gzip and the panic
    /// boundary. Exercised through `prepare` — the same function `handle` calls — so
    /// these are the real responses a browser gets, minus the socket.
    fn serving() -> Current {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny.safetensors");
        let opts = crate::opening::Options::default();
        let opened = crate::opening::Target::from_paths(&[fixture], None, &opts)
            .expect("the fixture resolves")
            .read(
                crate::opening::Want::Model,
                &crate::hf::ReadProgress::default(),
            )
            .expect("the fixture reads");
        Current::new(
            opened,
            None,
            opts,
            std::net::IpAddr::from([127, 0, 0, 1]),
            // In-memory: the request-layer tests must not touch the user's config directory.
            crate::opening::Recents::default(),
        )
        .expect("the served state builds")
    }

    /// The served state, shared — jobs outlive their request, so the tests hold what the server holds.
    fn serving_shared() -> std::sync::Arc<Current> {
        std::sync::Arc::new(serving())
    }

    /// A GET, which is what every route but `/api/open` and the job starts take.
    ///
    /// `&Arc<Current>` because starting a job hands a clone to a thread that outlives the request.
    fn get(c: &std::sync::Arc<Current>, path: &str, query: &str, gzip: bool) -> Prepared {
        prepare(c, &tiny_http::Method::Get, path, query, gzip)
    }

    #[test]
    fn every_api_route_answers_with_json() {
        let c = serving_shared();
        let state = c.snapshot();
        let name = state.tensors[0].name.clone();
        // The layout endpoint takes a shard basename, which is what the tensors carry.
        let file = state.tensors[0]
            .source_path
            .rsplit('/')
            .next()
            .expect("a source path")
            .to_string();
        let cases = [
            ("/api/tree", String::new()),
            ("/api/files", String::new()),
            ("/api/stats", String::new()),
            ("/api/health", String::new()),
            ("/api/check", String::new()),
            ("/api/model", String::new()),
            ("/api/tensor", format!("name={name}")),
            ("/api/layout", format!("file={file}")),
            ("/api/filter", "q=dtype:F16".to_string()),
            ("/api/schema", "q=*".to_string()),
        ];
        for (path, query) in cases {
            let out = get(&c, path, &query, false);
            assert_eq!(out.status, 200, "{path}?{query} → {}", out.status);
            assert!(
                serde_json::from_slice::<serde_json::Value>(out.body.as_slice()).is_ok(),
                "{path} did not return JSON"
            );
        }
    }

    /// A parameter whose *value* is malformed is refused too, and the message says what is allowed.
    ///
    /// The name check above and this one are the same rule: a request the server cannot honour as
    /// asked gets an explanation, not a confident answer to a different question. `?rows=lots` used
    /// to sample 32 rows, `?mode=windwo` returned a grid rather than the window that was asked for,
    /// `?bins=many` chose the bin count itself, and `?full=yes` folded the families it was told to
    /// expand — every one of them a `200`.
    #[test]
    fn a_malformed_parameter_value_is_refused_with_a_reason() {
        let c = serving_shared();
        // A tensor the fixture really has: the name is resolved before the values are parsed, so a
        // made-up one would be a 404 and prove nothing about the numbers.
        let name = "model.embed_tokens.weight";
        for (path, query, wanted) in [
            // A count that is not a number, and one that cannot be a count.
            (
                "/api/tensor/sample",
                format!("name={name}&rows=lots"),
                "rows",
            ),
            (
                "/api/tensor/sample",
                format!("name={name}&rows=0"),
                "at least 1",
            ),
            // A mode that is not one of the four.
            (
                "/api/tensor/sample",
                format!("name={name}&mode=windwo"),
                "not one of",
            ),
            // A fraction outside the axis.
            (
                "/api/tensor/sample",
                format!("name={name}&mode=edges&row_tail=7"),
                "out of range",
            ),
            // A bin count that is not a number — "choose for me" is *absent*, not unreadable.
            (
                "/api/tensor/histogram",
                format!("name={name}&bins=many"),
                "bins",
            ),
            // A switch that is neither on nor off.
            ("/api/difftree", "id=1&full=yes".to_string(), "not a switch"),
            ("/api/diff", "id=1&swap=maybe".to_string(), "not a switch"),
            (
                "/api/diff",
                "id=1&only_tensors=please".to_string(),
                "not a switch",
            ),
        ] {
            let out = get(&c, path, &query, false);
            assert_eq!(out.status, 400, "{path}?{query} → {}", out.status);
            let body: serde_json::Value =
                serde_json::from_slice(out.body.as_slice()).expect("an error envelope is JSON");
            let msg = body["error"].as_str().unwrap_or_default();
            assert!(
                msg.contains(wanted),
                "{path}?{query} should explain the value: got {msg:?}"
            );
        }
    }

    /// The values these routes are *given* still work — a strict parser that refused a legitimate
    /// request would be a worse bug than the one it fixes.
    #[test]
    fn the_values_the_client_sends_are_all_accepted() {
        let c = serving_shared();
        // A tensor the fixture really has: the name is resolved before the values are parsed, so a
        // made-up one would be a 404 and prove nothing about the numbers.
        let name = "model.embed_tokens.weight";
        for query in [
            format!("name={name}"),
            format!("name={name}&rows=8&cols=8"),
            format!("name={name}&mode=window&row_off=1&col_off=2"),
            format!("name={name}&mode=edges&row_tail=0.25&col_tail=1"),
            format!("name={name}&mode=max"),
            format!("name={name}&mode=grid&raw=1"),
        ] {
            let out = get(&c, "/api/tensor/sample", &query, false);
            assert_eq!(out.status, 200, "sample?{query} → {}", out.status);
        }
        for query in [format!("name={name}"), format!("name={name}&bins=16")] {
            let out = get(&c, "/api/tensor/histogram", &query, false);
            assert_eq!(out.status, 200, "histogram?{query} → {}", out.status);
        }
    }

    /// A parameter an endpoint does not take is a `400` naming it, on every route.
    ///
    /// The reported bug: `&bogus=1` was ignored and the endpoint answered `200` with the full payload.
    /// That is harmless for `bogus` and not at all harmless for `nmae=model.layers.1.*`, which asks for
    /// nineteen tensors and is answered with all 117,664 — a mistyped filter reads as a filter that
    /// matched everything. The command line refuses an unknown `--flag`; so does this.
    #[test]
    fn an_unknown_query_parameter_is_refused_by_every_endpoint() {
        let c = serving_shared();
        for path in ROUTES.iter().map(|r| format!("/api/{r}")) {
            let path = path.as_str();
            let out = get(&c, path, "bogus=1", false);
            assert_eq!(out.status, 400, "{path}?bogus=1 → {}", out.status);
            let body: serde_json::Value =
                serde_json::from_slice(out.body.as_slice()).expect("an error envelope is JSON");
            let msg = body["error"].as_str().unwrap_or_default();
            assert!(
                msg.contains("bogus"),
                "the refusal should name the parameter: {msg}"
            );
        }
        // A near-miss of a real parameter, which is the case that matters: silently dropping this one
        // returns every tensor under a heading that says the filter was applied.
        let out = get(&c, "/api/diff", "id=1&nmae=layers.1", false);
        assert_eq!(out.status, 400);
        let body: serde_json::Value = serde_json::from_slice(out.body.as_slice()).expect("JSON");
        let msg = body["error"].as_str().unwrap_or_default();
        assert!(
            msg.contains("nmae") && msg.contains("name"),
            "the refusal should name the typo and list what is accepted: {msg}"
        );
    }

    /// Every route with a query allowlist — the refusal test walks it, and the fixture below is
    /// generated from it, so a new endpoint is covered by both the moment it is listed.
    const ROUTES: &[&str] = &[
        "tree",
        "files",
        "stats",
        "health",
        "check",
        "model",
        "tensor",
        "layout",
        "filter",
        "schema",
        "compact",
        "diff",
        "difftree",
        "diffnames",
        "subtrees",
        "command",
        "file",
        "tensor/stats",
        "tensor/sample",
        "tensor/histogram",
        "recents",
        "reading",
        "version",
        "open",
        "compare",
        "jobs/values",
        "jobs/verify-repack",
    ];

    /// The allowlist, written out for the browser to check itself against.
    ///
    /// **Why a fixture and not a list in this file.** The check this replaces held a hand-copied copy
    /// of the keys `web/src/lib/api.ts` and `diffscope.ts` put on the wire — and by the time it was
    /// looked at, that copy was missing `align_fused`, `subtree`, `subtree_new`, `full`, `names_list`
    /// and `map_json`. It passed anyway, because a stale copy of the client agrees with itself. A
    /// client parameter the server does not accept turns a working screen into a `400`, and the only
    /// way to catch it is to compare against what the client *actually sends*, which lives on the
    /// other side of a language boundary.
    ///
    /// So this generates `shared/parity/queryparams.json` from the allowlist itself — Rust is the
    /// reference, as with `shared/parity/format.json` — and `web/src/lib/queryparams.test.ts` drives
    /// the real `api.*` calls through a stubbed `fetch` and asserts every key it puts on a URL is in
    /// here. Neither side can drift without one of the two failing.
    ///
    /// Regenerate after an intentional change:
    ///
    /// ```text
    /// UPDATE_PARITY=1 cargo test --features hdf5 the_accepted_parameters
    /// ```
    #[test]
    fn the_accepted_parameters_are_published_for_the_client_to_check() {
        // Every route the table knows, with its scope parameters folded in — which is exactly what
        // `unknown_params` compares against, so the fixture cannot describe a different rule.
        let routes: std::collections::BTreeMap<String, Vec<&str>> = ROUTES
            .iter()
            .filter_map(|path| {
                let (own, scoped) = super::accepted_params(path)?;
                let mut keys: Vec<&str> = own.to_vec();
                match scoped {
                    super::Scoped::Yes => keys.extend(super::scope_params()),
                    super::Scoped::AndCheck => {
                        keys.extend(super::scope_params());
                        keys.extend(super::params::keys(super::params::CHECK));
                    }
                    super::Scoped::No => {}
                }
                keys.sort_unstable();
                Some(((*path).to_string(), keys))
            })
            .collect();
        let generated = serde_json::to_string_pretty(&serde_json::json!({
            "note": concat!(
                "Generated by `UPDATE_PARITY=1 cargo test the_accepted_parameters` ",
                "(src/web/mod.rs); checked by web/src/lib/queryparams.test.ts. Do not edit by hand."
            ),
            "scope": super::scope_params(),
            "routes": routes,
        }))
        .expect("the allowlist serializes")
            + "\n";

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shared/parity/queryparams.json");
        if std::env::var("UPDATE_PARITY").is_ok() {
            std::fs::write(&path, &generated).expect("the fixture is writable");
            return;
        }
        let committed = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(
            committed, generated,
            "the accepted parameters have changed — regenerate with \
             `UPDATE_PARITY=1 cargo test the_accepted_parameters`, and check the browser still \
             sends only what is in the new list"
        );
    }

    /// **The client's parameter table is generated from the server's.**
    ///
    /// The browser encodes and decodes these parameters itself — it owns its address bar — and used to
    /// hold its own list of the same strings. A contract test caught a key the *server* would refuse;
    /// it could not catch a key the client read back under the wrong name. So the list is generated
    /// instead of checked: rename a row in `params.rs`, regenerate, and TypeScript fails to compile
    /// wherever the old field name is used.
    ///
    /// Regenerate after an intentional change:
    ///
    /// ```text
    /// UPDATE_PARITY=1 cargo test --features hdf5 the_client_parameter_table
    /// ```
    #[test]
    fn the_client_parameter_table_is_generated_from_this_one() {
        let generated = super::params::typescript_table();
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("web/src/lib/params.generated.ts");
        if std::env::var("UPDATE_PARITY").is_ok() {
            std::fs::write(&path, &generated).expect("the generated module is writable");
            return;
        }
        let committed = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(
            committed, generated,
            "the parameter table has changed — regenerate with \
             `UPDATE_PARITY=1 cargo test the_client_parameter_table`"
        );
    }

    /// **The build id the server reports is the bundle it actually serves.**
    ///
    /// A tab compares this against the script it is running, and acts on a mismatch by telling someone
    /// to reload — so an id that drifts from the served assets is worse than no id at all. Derived from
    /// the `index.html` being served rather than stamped in at build time, and this is that property:
    /// the name it reports is a real asset, and it is the one the shell loads.
    #[test]
    fn the_reported_build_is_the_bundle_being_served() {
        let c = serving_shared();
        let out = get(&c, "/api/version", "", false);
        assert_eq!(out.status, 200);
        let body: serde_json::Value = serde_json::from_slice(out.body.as_slice()).expect("JSON");
        let assets = body["assets"].as_str().unwrap_or_default();
        // Vite's own naming, which is what both sides key on — the extension is fixed, not a
        // filesystem lookup, so it is compared literally.
        assert!(
            assets.starts_with("index-")
                && std::path::Path::new(assets).extension() == Some("js".as_ref()),
            "expected a hashed entry script, got {body}"
        );

        // It is in the shell…
        let shell = String::from_utf8_lossy(get(&c, "/", "", false).body.as_slice()).into_owned();
        assert!(shell.contains(assets), "the shell does not load {assets}");
        // …and it is a file this server will serve.
        let served = get(&c, &format!("/assets/{assets}"), "", false);
        assert_eq!(served.status, 200, "{assets} is not served");
    }

    #[test]
    fn an_unknown_api_route_is_a_404_with_an_error_envelope() {
        let c = serving_shared();
        let out = get(&c, "/api/nope", "", false);
        assert_eq!(out.status, 404);
        let body: serde_json::Value =
            serde_json::from_slice(out.body.as_slice()).expect("an error envelope is JSON");
        assert!(body.get("error").is_some(), "{body}");
    }

    #[test]
    fn a_missing_tensor_is_reported_not_guessed() {
        let c = serving_shared();
        let out = get(&c, "/api/tensor", "name=no.such.tensor", false);
        assert!(
            out.status >= 400,
            "expected an error status, got {}",
            out.status
        );
        let body: serde_json::Value = serde_json::from_slice(out.body.as_slice()).expect("JSON");
        assert!(body.get("error").is_some(), "{body}");
    }

    #[test]
    fn the_spa_is_served_for_the_root_and_for_unknown_paths() {
        // A client-routed URL (`/#detail?...` arrives as a path on reload) must get
        // index.html, not a 404 — otherwise a deep link breaks on refresh.
        for path in ["/", "/index.html", "/tree", "/some/client/route"] {
            let out = get(&serving_shared(), path, "", false);
            assert_eq!(out.status, 200, "{path} → {}", out.status);
            assert!(
                out.body.as_slice().starts_with(b"<!") || out.body.as_slice().starts_with(b"<html"),
                "{path} did not return the SPA shell"
            );
        }
    }

    #[test]
    fn assets_are_served_with_a_content_type_and_are_cacheable() {
        let c = serving_shared();
        let index = String::from_utf8_lossy(get(&c, "/", "", false).body.as_slice()).to_string();
        // Pull the hashed bundle names out of the shell and fetch them.
        for ext in ["js", "css"] {
            let needle = format!(".{ext}");
            let Some(pos) = index.find(&needle) else {
                continue;
            };
            let start = index[..pos].rfind("/assets/").expect("an asset path");
            let path = &index[start..pos + needle.len()];
            let out = get(&c, path, "", false);
            assert_eq!(out.status, 200, "{path} → {}", out.status);
            assert!(!out.body.as_slice().is_empty(), "{path} was empty");
        }
        // A missing asset is a 404, not the SPA shell (which would corrupt a script).
        let out = get(&c, "/assets/nope-12345678.js", "", false);
        assert_eq!(out.status, 404);
    }

    #[test]
    fn gzip_is_applied_only_when_the_client_asks_and_it_helps() {
        let c = serving_shared();
        // A large JSON body compresses, and the compressed form is what's sent.
        let plain = get(&c, "/api/model", "", false);
        let zipped = get(&c, "/api/model", "", true);
        assert_eq!(plain.status, 200);
        assert!(
            zipped.body.as_slice().len() <= plain.body.as_slice().len(),
            "gzip made the body bigger: {} vs {}",
            zipped.body.as_slice().len(),
            plain.body.as_slice().len()
        );
        // `maybe_gzip` reports whether it actually encoded, so the header can't lie.
        let (small, encoded) = maybe_gzip(b"tiny".to_vec(), true);
        assert!(
            !encoded || small.len() < 4,
            "a tiny body isn't worth encoding"
        );
        let (raw, encoded) = maybe_gzip(b"whatever".to_vec(), false);
        assert!(!encoded && raw == b"whatever");
    }

    #[test]
    fn the_precomputed_bodies_are_cached_and_shared() {
        let c = serving_shared();
        // The second request for a static endpoint must reuse the first body rather than
        // re-serialising it — an unbounded rebuild per request leaked 2 GB before.
        let first = get(&c, "/api/tree", "", false);
        let second = get(&c, "/api/tree", "", false);
        assert_eq!(first.body.as_slice(), second.body.as_slice());
        // The cache is keyed by endpoint name (see `STATIC_ENDPOINTS`), not by path. It lives
        // inside the served state, which is what makes a checkpoint switch discard it.
        let state = c.snapshot();
        assert!(
            state.cached_body("tree", false).is_some(),
            "the body was cached"
        );
        // Gzipped and plain are cached separately, so a mixed client set can't cross.
        let _ = get(&c, "/api/tree", "", true);
        assert!(state.cached_body("tree", true).is_some());
        // A parameterised endpoint is not cached under the static key.
        assert!(
            state.cached_body("tensor", false).is_none(),
            "a parameterised endpoint must not be cached"
        );
    }

    #[test]
    fn a_query_string_is_percent_decoded_per_value() {
        let q = parse_query("name=model.layers.0%2Fw&dtype=F16&flag=");
        assert_eq!(q.get("name").map(String::as_str), Some("model.layers.0/w"));
        assert_eq!(q.get("dtype").map(String::as_str), Some("F16"));
        assert_eq!(q.get("flag").map(String::as_str), Some(""));
        assert!(!q.contains_key("missing"));
    }

    #[test]
    fn decodes_encoded_tensor_name() {
        // The client sends encodeURIComponent("model.layers.0/mlp.weight").
        let q = parse_query("name=model.layers.0%2Fmlp.weight&dtype=f16&rows=8");
        assert_eq!(
            q.get("name").map(String::as_str),
            Some("model.layers.0/mlp.weight")
        );
        assert_eq!(q.get("dtype").map(String::as_str), Some("f16"));
        assert_eq!(q.get("rows").map(String::as_str), Some("8"));
    }

    #[test]
    fn empty_query_is_empty_map() {
        assert!(parse_query("").is_empty());
    }
}
