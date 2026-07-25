//! One function per API route. Each takes `&WebState` (+ the parsed query) and
//! returns `(status, json)` — no socket, so they're unit-testable directly. The
//! metadata/view routes read precomputed state (instant); the `/api/tensor/*`
//! data routes read tensor bytes on demand (local-only) via `crate::sample`.

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

use serde::Serialize;
use serde_json::{Value, json};

use super::WebState;
use crate::sample::{self, SampleMode, ViewDtype};
use crate::tree::TensorInfo;
use crate::web::dto::{self, HistogramDto, SampleDto, StatsDto};

pub type Query = HashMap<String, String>;
/// An HTTP status plus the response body ALREADY serialised to JSON bytes (not a
/// `serde_json::Value`) — see `ok`.
pub type Reply = (u16, Vec<u8>);

fn ok<T: Serialize>(v: T) -> Reply {
    // Serialise STRAIGHT to bytes. Going via `serde_json::to_value` first materialised
    // the whole response as a `Value` tree — for `/api/tree` on a 31k-tensor checkpoint
    // that was ~250 MB of transient allocation and a second full pass over the data,
    // every request.
    (
        200,
        serde_json::to_vec(&v).unwrap_or_else(|_| b"null".to_vec()),
    )
}

pub fn err(status: u16, msg: impl Into<String>) -> Reply {
    let body = json!({ "error": msg.into() });
    (
        status,
        serde_json::to_vec(&body)
            .unwrap_or_else(|_| br#"{"error":"serialisation failed"}"#.to_vec()),
    )
}

/// Data-value views need the tensor bytes locally; a remote (`--ssh-proxy`) source
/// only carries its structure. Returns a friendly 400 for a remote tensor (so the
/// UI shows a clear note instead of a cryptic open-file failure), else `None`.
fn require_local(t: &TensorInfo) -> Option<Reply> {
    crate::remote::is_remote_source(&t.source_path).then(|| {
        err(
            400,
            "This checkpoint was read remotely (--ssh-proxy): only its structure is available. \
             Data views (heatmap, values, histogram, statistics) need the file locally — copy \
             the checkpoint down to preview its values.",
        )
    })
}

// ---- metadata / derived-view routes (served from precomputed state) ----

pub fn tree(s: &WebState) -> Reply {
    // Wrap the forest in a single root node summarising the whole checkpoint, the
    // way the TUI's tree does (`▾ <name> (▦ N, P params, S)`), with the metadata
    // group (when present) among its children.
    let root = crate::tree::TreeNode::Group {
        name: basename(&s.root).to_string(),
        children: s.tree.clone(),
        expanded: true,
        tensor_count: s.tensors.len(),
        params: s.tensors.iter().map(|t| t.num_elements).sum(),
        total_size: s.tensors.iter().map(|t| t.size_bytes).sum(),
        stored_size: s.tensors.iter().map(|t| t.on_disk_size()).sum(),
    };
    ok(json!({
        "root": s.root,
        "tensor_count": s.tensors.len(),
        "tree": [root],
    }))
}

pub fn files(s: &WebState) -> Reply {
    ok(&s.file_tree)
}

/// Rich tensor filtering: parse the `?q=` text query (see [`crate::tensorfilter`])
/// with the shared matcher and return the names of the tensors that pass, so the
/// client masks its tree to them. `active:false` for an empty query (show all); a
/// malformed query is a `400` whose message the filter bar shows inline.
pub fn filter(s: &WebState, q: &Query) -> Reply {
    let query = q.get("q").map(String::as_str).unwrap_or("");
    match crate::tensorfilter::TensorFilter::parse(query) {
        Ok(f) if !f.is_active() => ok(json!({ "active": false })),
        Ok(f) => {
            let names: Vec<&str> = s
                .tensors
                .iter()
                .filter(|t| f.matches(t))
                .map(|t| t.name.as_str())
                .collect();
            ok(json!({ "active": true, "names": names }))
        }
        Err(e) => err(400, e.to_string()),
    }
}

/// Compact per-family listing: collapse the (optionally `?q=`-filtered) tensors into
/// index-templated families (`model.layers.{0-47}.…experts.{0-3}.down_proj.weight`)
/// with per-family count + uniform dtype/shape + total params/bytes — a "what's in
/// here, per layer / per expert" summary (same collapsing as `diff`).
pub fn schema(s: &WebState, q: &Query) -> Reply {
    let query = q.get("q").map(String::as_str).unwrap_or("");
    let filter = match crate::tensorfilter::TensorFilter::parse(query) {
        Ok(f) => f,
        Err(e) => return err(400, e.to_string()),
    };
    let families = if filter.is_active() {
        let matched: Vec<crate::tree::TensorInfo> = s
            .tensors
            .iter()
            .filter(|t| filter.matches(t))
            .cloned()
            .collect();
        crate::diff::tensor_families(&matched)
    } else {
        crate::diff::tensor_families(&s.tensors)
    };
    ok(json!({ "families": families }))
}

pub fn stats(s: &WebState) -> Reply {
    ok(&s.stats)
}

pub fn health(s: &WebState) -> Reply {
    ok(&s.health)
}

pub fn check(s: &WebState) -> Reply {
    match &s.check {
        Some(report) => ok(report.to_json(false)),
        None => ok(Value::Null),
    }
}

pub fn model(s: &WebState) -> Reply {
    ok(&s.checkpoint)
}

pub fn tensor(s: &WebState, q: &Query) -> Reply {
    match lookup(s, q) {
        Ok(t) => ok(t),
        Err(e) => e,
    }
}

/// Read a text/JSON file's content (capped) for the file browser's preview. Only
/// serves paths that are in the checkpoint's own file list — no path traversal.
pub fn file(s: &WebState, q: &Query) -> Reply {
    let Some(rel) = q.get("path") else {
        return err(400, "missing ?path=");
    };
    let Some(entry) = s
        .checkpoint
        .files
        .iter()
        .find(|f| f.rel_path == *rel && !f.is_dir())
    else {
        return err(404, format!("no such file: {rel}"));
    };
    let abs = std::path::Path::new(&s.root).join(&entry.rel_path);
    const CAP: usize = 1 << 20; // 1 MiB — enough for config/index/readme/merges
    match std::fs::read(&abs) {
        Ok(bytes) => {
            let truncated = bytes.len() > CAP;
            let text = String::from_utf8_lossy(&bytes[..bytes.len().min(CAP)]).into_owned();
            ok(json!({
                "path": rel,
                "name": entry.name,
                "size": entry.apparent(),
                "truncated": truncated,
                "text": text,
            }))
        }
        Err(e) => err(500, format!("read failed: {e}")),
    }
}

pub fn layout(s: &WebState, q: &Query) -> Reply {
    let Some(file) = q.get("file") else {
        return err(400, "missing ?file=");
    };
    match s
        .layouts
        .iter()
        .find(|l| l.name == *file || basename(&l.name) == file.as_str())
    {
        Some(l) => ok(l),
        None => err(404, format!("no layout for file: {file}")),
    }
}

// ---- on-demand tensor-data routes (read bytes; local only) ----

pub fn tensor_stats(s: &WebState, q: &Query) -> Reply {
    let t = match lookup(s, q) {
        Ok(t) => t,
        Err(e) => return e,
    };
    if let Some(e) = require_local(t) {
        return e;
    }
    let view = match view_of(q) {
        Ok(v) => v,
        Err(e) => return e,
    };
    match scan_stats(s, t, view) {
        Ok(dto) => ok(dto),
        Err(e) => e,
    }
}

pub fn tensor_sample(s: &WebState, q: &Query) -> Reply {
    let t = match lookup(s, q) {
        Ok(t) => t,
        Err(e) => return e,
    };
    if let Some(e) = require_local(t) {
        return e;
    }
    let view = match view_of(q) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let rows = num(q, "rows", 32);
    let cols = num(q, "cols", 32);
    let slice = num(q, "slice", 0);
    let mode = match q.get("mode").map(String::as_str) {
        Some("window") => SampleMode::Window {
            row_off: num(q, "row_off", 0),
            col_off: num(q, "col_off", 0),
        },
        Some("edges") => SampleMode::Edges {
            row_tail: fnum(q, "row_tail", 0.5),
            col_tail: fnum(q, "col_tail", 0.5),
        },
        Some("max") => SampleMode::GridMax,
        _ => SampleMode::Grid,
    };
    let schema = s.schemas.get(name_of(q));
    let include_raw = matches!(q.get("raw").map(String::as_str), Some("1") | Some("true"));
    match sample::sample_tensor(t, rows, cols, slice, view, mode, schema) {
        Ok(sample) => ok(SampleDto::from_sample(&sample, &t.dtype, include_raw)),
        Err(e) => err(500, e),
    }
}

pub fn tensor_histogram(s: &WebState, q: &Query) -> Reply {
    let t = match lookup(s, q) {
        Ok(t) => t,
        Err(e) => return e,
    };
    if let Some(e) = require_local(t) {
        return e;
    }
    let view = match view_of(q) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let bins = q.get("bins").and_then(|b| b.parse::<usize>().ok());

    // Float / wide-int bins need the value range; reuse the cached stats or scan.
    let range = match scan_stats(s, t, view) {
        Ok(dto) => Some((dto.min, dto.max)),
        Err(e) => return e,
    };
    let Some((hist_bins, n)) = sample::histogram_bins(view, &t.dtype, range, bins) else {
        return err(400, format!("no histogram for dtype {}", t.dtype));
    };
    let shared = sample::HistShared::new(n);
    let (cancel, pause) = (AtomicBool::new(false), AtomicBool::new(false));
    let schema = s.schemas.get(name_of(q));
    if let Err(e) = sample::tensor_histogram_into(
        t, view, schema, hist_bins, n, &shared, &cancel, &pause, None,
    ) {
        return err(500, e);
    }
    ok(HistogramDto::from(&shared.snapshot(hist_bins)))
}

// ---- helpers ----

/// Compute (or fetch the cached) whole-tensor stats for `(name, view)`.
fn scan_stats(s: &WebState, t: &TensorInfo, view: ViewDtype) -> Result<StatsDto, Reply> {
    let key = (t.name.clone(), dto::view_label(view));
    // `unwrap_or_else(into_inner)`: this is a pure memo, so a mutex poisoned by an
    // unrelated panic carries no broken invariant — but `.unwrap()` would turn that into
    // a permanent 500 for this endpoint for the rest of the process's life.
    let hit = s
        .stats_cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&key)
        .cloned();
    if let Some(hit) = hit {
        return Ok(hit.clone());
    }
    let (cancel, pause) = (AtomicBool::new(false), AtomicBool::new(false));
    let schema = s.schemas.get(&t.name);
    let stats =
        sample::tensor_stats(t, view, schema, &cancel, &pause, None).map_err(|e| err(500, e))?;
    let dto = StatsDto::from(&stats);
    s.stats_cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(key, dto.clone());
    Ok(dto)
}

fn lookup<'a>(s: &'a WebState, q: &Query) -> Result<&'a TensorInfo, Reply> {
    let name = q.get("name").ok_or_else(|| err(400, "missing ?name="))?;
    let idx = s
        .tensor_index
        .get(name)
        .ok_or_else(|| err(404, format!("unknown tensor: {name}")))?;
    Ok(&s.tensors[*idx])
}

fn view_of(q: &Query) -> Result<ViewDtype, Reply> {
    match q.get("dtype") {
        Some(d) => sample::parse_view_dtype(d).map_err(|e| err(400, e)),
        None => Ok(ViewDtype::Stored),
    }
}

fn name_of(q: &Query) -> &str {
    q.get("name").map(String::as_str).unwrap_or("")
}

fn num<T: std::str::FromStr>(q: &Query, key: &str, default: T) -> T {
    q.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn fnum(q: &Query, key: &str, default: f32) -> f32 {
    q.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn basename(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Build the shared state from a checked-in fixture, exactly as `run_web` does.
    /// These handlers had no automated coverage at all despite being the layer that
    /// changes most often, so this exercises every endpoint's success shape, its error
    /// path, and the fixed-content cache.
    fn state() -> crate::web::WebState {
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny.safetensors");
        let files = vec![fixture];
        let model = crate::readers::read_local(&files).expect("fixture reads");
        crate::web::WebState::build(model, &files, &[])
    }

    /// Parse a reply body back into JSON so tests assert on the values a client sees.
    fn json(reply: &Reply) -> serde_json::Value {
        serde_json::from_slice(&reply.1).unwrap_or_else(|e| {
            panic!(
                "reply body is not JSON ({e}): {}",
                String::from_utf8_lossy(&reply.1)
            )
        })
    }

    fn query(pairs: &[(&str, &str)]) -> Query {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    const TENSOR: &str = "model.layers.0.mlp.down_proj.weight";

    #[test]
    fn every_endpoint_answers_200_with_the_documented_shape() {
        let s = state();

        let tree = json(&tree(&s));
        assert!(tree["root"].is_string(), "tree exposes the checkpoint root");
        assert!(tree["tree"].is_array(), "tree exposes the node array");

        assert!(json(&files(&s))["kind"].is_string(), "files is an FsNode");
        let st = json(&stats(&s));
        assert_eq!(st["files"]["count"], 1, "one fixture shard");
        assert!(
            st["tensors"].is_object() || st["dtypes"].is_array(),
            "stats reports tensor facets: {st}"
        );
        assert!(json(&health(&s)).is_array(), "health is a list of reports");
        assert!(json(&check(&s))["summary"].is_object());
        assert!(json(&model(&s))["root"].is_string());

        // The whole point of `/api/tensor`: one tensor's metadata by exact name.
        let t = json(&tensor(&s, &query(&[("name", TENSOR)])));
        assert_eq!(t["name"], TENSOR);
        assert_eq!(t["dtype"], "U16");
        assert_eq!(t["shape"], serde_json::json!([3, 4, 5]));

        // Filtering is server-side so the web and TUI agree; check it actually filters.
        let f = json(&filter(&s, &query(&[("q", "dtype:F32")])));
        assert_eq!(f["active"], true);
        let names: Vec<&str> = f["names"]
            .as_array()
            .expect("names")
            .iter()
            .map(|n| n.as_str().unwrap_or(""))
            .collect();
        assert_eq!(
            names,
            ["model.layers.0.input_layernorm.weight", "model.norm.weight"]
        );

        // An empty query is "inactive" (show everything), not "match nothing".
        assert_eq!(json(&filter(&s, &query(&[("q", "")])))["active"], false);

        assert!(json(&schema(&s, &query(&[("q", "")])))["families"].is_array());
    }

    #[test]
    fn data_view_endpoints_return_the_requested_window() {
        let s = state();
        let sample = json(&tensor_sample(
            &s,
            &query(&[
                ("name", TENSOR),
                ("mode", "window"),
                ("rows", "2"),
                ("cols", "3"),
            ]),
        ));
        assert_eq!(sample["values"].as_array().map(Vec::len), Some(2));
        assert_eq!(sample["values"][0].as_array().map(Vec::len), Some(3));
        // U16 is an integer view, so the client is told to format from the raw bits.
        assert_eq!(sample["integer"], true);
        assert_eq!(sample["signed"], false);
        assert!(
            sample["raw"].is_array(),
            "integer views always ship raw bits"
        );

        let st = json(&tensor_stats(&s, &query(&[("name", TENSOR)])));
        assert_eq!(st["count"], 60); // 3*4*5
        assert!(st["min"].is_number() && st["max"].is_number());

        let h = json(&tensor_histogram(
            &s,
            &query(&[("name", TENSOR), ("bins", "8")]),
        ));
        assert!(h["counts"].as_array().is_some_and(|c| !c.is_empty()));

        let l = json(&layout(&s, &query(&[("file", "tiny.safetensors")])));
        assert!(l["segments"].is_array(), "byte-layout segments");
    }

    #[test]
    fn bad_input_is_a_4xx_with_a_message_never_a_panic() {
        let s = state();
        for (label, reply) in [
            ("unknown tensor", tensor(&s, &query(&[("name", "nope")]))),
            ("missing name", tensor(&s, &query(&[]))),
            (
                "unknown layout file",
                layout(&s, &query(&[("file", "nope.safetensors")])),
            ),
            ("missing file param", layout(&s, &query(&[]))),
            ("unknown file", file(&s, &query(&[("path", "nope.txt")]))),
            (
                "sample of unknown tensor",
                tensor_sample(&s, &query(&[("name", "nope")])),
            ),
            (
                "stats of unknown tensor",
                tensor_stats(&s, &query(&[("name", "nope")])),
            ),
            ("bad filter facet", filter(&s, &query(&[("q", "bogus:1")]))),
            (
                "bad filter number",
                filter(&s, &query(&[("q", "size:abc")])),
            ),
        ] {
            assert!(
                (400..500).contains(&reply.0),
                "{label}: expected a 4xx, got {}",
                reply.0
            );
            let msg = json(&reply)["error"].as_str().unwrap_or("").to_string();
            assert!(!msg.is_empty(), "{label}: a 4xx must explain itself");
        }
    }

    /// A dtype override must reinterpret the SAME bytes, not re-read the tensor: the
    /// packed 4-bit view yields more values than the stored U16 one.
    #[test]
    fn dtype_override_reinterprets_the_same_bytes() {
        let s = state();
        let stored = json(&tensor_sample(
            &s,
            &query(&[
                ("name", TENSOR),
                ("mode", "window"),
                ("rows", "1"),
                ("cols", "40"),
            ]),
        ));
        let as_u4 = json(&tensor_sample(
            &s,
            &query(&[
                ("name", TENSOR),
                ("dtype", "u4"),
                ("mode", "window"),
                ("rows", "1"),
                ("cols", "40"),
            ]),
        ));
        assert_eq!(stored["view"], "stored");
        assert_eq!(as_u4["view"], "u4");
        assert!(
            as_u4["total_cols"].as_u64() > stored["total_cols"].as_u64(),
            "unpacking 4-bit nibbles must widen the logical row"
        );
    }
}
