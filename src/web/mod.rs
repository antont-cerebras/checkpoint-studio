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
pub(crate) mod dto;
pub(crate) mod handlers;

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
    pub root: String,
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
    // UI can show — including `/api/diff?against=PATH`, which will read any checkpoint
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
         read this checkpoint — and any checkpoint path this user can read (see /api/diff). \
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

fn handle(current: &Current, req: tiny_http::Request) {
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
    current: &Current,
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

fn route_api(
    current: &Current,
    s: &WebState,
    method: &tiny_http::Method,
    path: &str,
    q: &Query,
) -> Reply {
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
    match path {
        "recents" => handlers::recents(current),
        "tree" => handlers::tree(s),
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
        "diff" => handlers::diff(s, q),
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

    /// A GET, which is what every route but `/api/open` takes.
    fn get(c: &Current, path: &str, query: &str, gzip: bool) -> Prepared {
        prepare(c, &tiny_http::Method::Get, path, query, gzip)
    }

    #[test]
    fn every_api_route_answers_with_json() {
        let c = serving();
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

    #[test]
    fn an_unknown_api_route_is_a_404_with_an_error_envelope() {
        let c = serving();
        let out = get(&c, "/api/nope", "", false);
        assert_eq!(out.status, 404);
        let body: serde_json::Value =
            serde_json::from_slice(out.body.as_slice()).expect("an error envelope is JSON");
        assert!(body.get("error").is_some(), "{body}");
    }

    #[test]
    fn a_missing_tensor_is_reported_not_guessed() {
        let c = serving();
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
            let out = get(&serving(), path, "", false);
            assert_eq!(out.status, 200, "{path} → {}", out.status);
            assert!(
                out.body.as_slice().starts_with(b"<!") || out.body.as_slice().starts_with(b"<html"),
                "{path} did not return the SPA shell"
            );
        }
    }

    #[test]
    fn assets_are_served_with_a_content_type_and_are_cacheable() {
        let c = serving();
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
        let c = serving();
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
        let c = serving();
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
