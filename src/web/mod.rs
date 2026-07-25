//! `--web`: a headless HTTP server (sync/blocking, no async runtime) that serves
//! the checkpoint as JSON — the **data** — plus the embedded Svelte UI, which owns
//! its own **view state**. Local checkpoints only for now.
//!
//! `WebState` is read once at startup and shared read-only across worker threads
//! (`Arc`); every derived view/report is precomputed so request handling needs no
//! `&mut` (only the on-demand tensor-data scans touch disk, behind a small cache).

mod assets;
pub mod dto;
pub mod handlers;

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::{check, filetree, filter, health, kernel, model, safelayout, sample, stats, tree};
use handlers::{Query, Reply};

/// Everything the API serves, computed once from a local read. Shared read-only.
pub struct WebState {
    pub root: String,
    /// The full serializable model (backs `/api/model`).
    pub checkpoint: model::Checkpoint,
    /// The tensor-tree hierarchy (client folds/selects/searches it).
    pub tree: Vec<tree::TreeNode>,
    pub file_tree: dto::WebFileNode,
    pub stats: stats::CheckpointStats,
    pub health: Vec<health::HealthReport>,
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

/// `(endpoint, gzipped)` -> the fully-encoded response body.
type StaticBodies = Mutex<HashMap<(&'static str, bool), Arc<Vec<u8>>>>;

impl WebState {
    /// Build the shared state from a local checkpoint read. `files`/`index_specs`
    /// are what `run_explore` already resolved (for the structural check + health).
    pub fn build(
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
        let tree = session.build_tree();
        let checkpoint_stats = session.stats_with_disk(disk).clone();

        let tensor_index = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        let schemas = sample::parse_packing_schemas(&tensors, &metadata);

        let file_tree =
            dto::WebFileNode::from_node(&filetree::build(Path::new(&root), 8), Path::new(&root));

        let health: Vec<health::HealthReport> = index_specs
            .iter()
            .map(|spec| health::check_loaded(spec, &tensors))
            .collect();

        // Structural check only (values = false → no byte scan at startup).
        let all = filter::NameFilter::parse(&[]).expect("empty NameFilter is valid");
        let check = Some(check::run(
            root.clone(),
            &tensors,
            &metadata,
            files,
            &health,
            config.as_ref(),
            &all,
            false,
            1,
        ));

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

        WebState {
            root,
            checkpoint,
            tree,
            file_tree,
            stats: checkpoint_stats,
            health,
            check,
            layouts,
            tensors,
            tensor_index,
            schemas,
            stats_cache: Mutex::new(HashMap::new()),
            static_bodies: Mutex::new(HashMap::new()),
        }
    }

    /// The encoded body for a fixed-content endpoint, building (and caching) it on
    /// first use. `None` for endpoints that aren't cacheable.
    fn cached_body(&self, api: &str, gzipped: bool) -> Option<Arc<Vec<u8>>> {
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
        let body = if gzipped {
            gzip_bytes(&json).ok()?
        } else {
            json
        };
        let body = Arc::new(body);
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
pub fn bind(host: IpAddr, port: u16) -> Result<tiny_http::Server> {
    match tiny_http::Server::http(SocketAddr::new(host, port)) {
        Ok(server) => Ok(server),
        Err(e) if port != 0 => {
            let server = tiny_http::Server::http(SocketAddr::new(host, 0))
                .map_err(|e2| anyhow::anyhow!("failed to start web server on {host}: {e2}"))?;
            let freed = server.server_addr().to_ip().map(|a| a.port()).unwrap_or(0);
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

/// Start the server and block until the process is stopped (Ctrl-C). Binds the
/// port immediately (see [`bind`]); for a remote read, bind first and pass the
/// server to [`serve_on`] so the port is held while the read runs.
pub fn serve(state: Arc<WebState>, host: IpAddr, port: u16) -> Result<()> {
    serve_on(bind(host, port)?, state, host)
}

/// Serve on an already-[`bind`]-ed socket and block until stopped. `host` is only
/// used to render a reachable URL (a wildcard bind isn't clickable).
pub fn serve_on(server: tiny_http::Server, state: Arc<WebState>, host: IpAddr) -> Result<()> {
    let bound = server.server_addr().to_ip().map(|a| a.port()).unwrap_or(0);
    // Print a URL the browser can actually reach: a wildcard bind (0.0.0.0 / ::)
    // isn't clickable, so show this host's FQDN instead of the bind address.
    let display = if host.is_unspecified() {
        fqdn().unwrap_or_else(|| "localhost".to_string())
    } else {
        host.to_string()
    };
    let url = format!("http://{display}:{bound}/");
    print_serve_banner(&url);

    // A small worker pool so a static-asset / metadata request stays responsive
    // while another worker is inside a multi-second tensor scan.
    let server = Arc::new(server);
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(2, 8);
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let server = Arc::clone(&server);
        let state = Arc::clone(&state);
        handles.push(std::thread::spawn(move || {
            while let Ok(req) = server.recv() {
                handle(&state, req);
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
fn print_serve_banner(url: &str) {
    use std::io::IsTerminal;
    let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    if !color {
        println!("checkpoint-studio web UI: {url}  (Ctrl-C to stop)");
        return;
    }
    // Raw ANSI via named consts + a terminal/NO_COLOR guard — the same convention
    // the other one-shot CLI status lines use (e.g. `sftp`'s retry notice, the
    // `diff: done` footer, `progress`'s bars). `yansi` is reserved for the TUI layer.
    const DIM: &str = "\x1b[2m";
    const URL: &str = "\x1b[1;4;96m"; // bold + underline + bright cyan (link-like)
    const RESET: &str = "\x1b[0m";
    // Blank line to break from the finished ✓ load bar above; the label dim, the URL
    // the one bright thing on the line so it wins the eye.
    println!(
        "\n  {DIM}checkpoint-studio web UI ▸{RESET}  {URL}{url}{RESET}\n  {DIM}Ctrl-C to stop{RESET}\n"
    );
}

const JSON_CT: &str = "application/json; charset=utf-8";

/// A response body, either freshly built or shared from the fixed-content cache. Shared
/// so a repeat `/api/tree` hands out an `Arc` instead of copying 14 MB.
enum Body {
    Owned(Vec<u8>),
    Shared(Arc<Vec<u8>>),
}

impl Body {
    fn as_slice(&self) -> &[u8] {
        match self {
            Body::Owned(v) => v,
            Body::Shared(a) => a,
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
}

fn handle(state: &WebState, req: tiny_http::Request) {
    let url = req.url().to_string();
    let gzip = accepts_gzip(&req);
    // Contain a panic. The worker pool is small (2-8 threads) and each worker loops on
    // `server.recv()`, so an unwinding handler would kill that worker permanently —
    // after a handful of bad requests the process would still accept connections and
    // answer none: alive, but silently hung. (We shipped exactly such a handler
    // earlier: a histogram bin-count overflow that indexed an empty vector.) Resolving
    // the response before touching the socket means a panic costs one 500, not a
    // worker.
    let prepared = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (path, query_str) = url.split_once('?').unwrap_or((url.as_str(), ""));
        prepare(state, path, query_str, gzip)
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
        }
    });
    send_encoded(
        req,
        prepared.status,
        prepared.body.as_slice(),
        prepared.content_type,
        prepared.gzipped,
        prepared.cache_control,
    );
}

/// Resolve a request to bytes. Pure: touches no socket, so it is safe to run inside the
/// panic boundary above.
fn prepare(state: &WebState, path: &str, query_str: &str, gzip: bool) -> Prepared {
    let Some(api) = path.strip_prefix("/api/") else {
        return prepare_asset(path, gzip);
    };
    let q = parse_query(query_str);
    // The API reflects one read-once checkpoint; a browser must never reuse a response
    // from a prior server run (a different checkpoint on the same port) — hence
    // `no-store` — but we can reuse it SERVER-side: a fixed-content endpoint is encoded
    // once and then handed out as bytes (see `cached_body`).
    if q.is_empty()
        && let Some(body) = state.cached_body(api, gzip)
    {
        return Prepared {
            status: 200,
            body: Body::Shared(body),
            content_type: JSON_CT,
            gzipped: gzip,
            cache_control: Some("no-store"),
        };
    }
    let (status, data) = route_api(state, api, &q);
    let (body, gzipped) = maybe_gzip(data, gzip);
    Prepared {
        status,
        body: Body::Owned(body),
        content_type: JSON_CT,
        gzipped,
        cache_control: Some("no-store"),
    }
}

fn route_api(s: &WebState, path: &str, q: &Query) -> Reply {
    match path {
        "tree" => handlers::tree(s),
        "files" => handlers::files(s),
        "filter" => handlers::filter(s, q),
        "schema" => handlers::schema(s, q),
        "stats" => handlers::stats(s),
        "health" => handlers::health(s),
        "check" => handlers::check(s),
        "model" => handlers::model(s),
        "tensor" => handlers::tensor(s, q),
        "layout" => handlers::layout(s, q),
        "file" => handlers::file(s, q),
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
    }
}

/// gzip the body when the client accepts it and the payload is big enough to be worth
/// it (the tensor-tree JSON is tens of MB). Returns the body and whether it's encoded.
fn maybe_gzip(data: Vec<u8>, gzip: bool) -> (Vec<u8>, bool) {
    if gzip && data.len() > 1024 {
        match gzip_bytes(&data) {
            Ok(compressed) => (compressed, true),
            Err(_) => (data, false),
        }
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
) {
    let mut headers = vec![header("Content-Type", content_type)];
    if let Some(cc) = cache_control {
        headers.push(header("Cache-Control", cc));
    }
    if gzipped {
        headers.push(header("Content-Encoding", "gzip"));
    }
    let mut resp = tiny_http::Response::from_data(body).with_status_code(status);
    for h in headers {
        resp = resp.with_header(h);
    }
    let _ = req.respond(resp);
}

fn header(key: &str, value: &str) -> tiny_http::Header {
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
    use super::parse_query;

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
