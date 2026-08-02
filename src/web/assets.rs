//! The embedded Svelte SPA (`web/dist`). rust-embed reads from disk in debug and
//! embeds the bytes in release builds, so `cargo run -- --web` always serves the
//! latest `npm run build` while a released binary is self-contained.

use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "web/dist"]
pub(super) struct WebAssets;

/// Which build of the UI this binary serves — the entry script's hashed name, e.g.
/// `index-c1322f20.js`.
///
/// **Read out of the served `index.html` rather than stamped in at compile time**, so it is the
/// identity of the bundle actually going over the wire: rust-embed reads from disk in debug, and a
/// stamp would then describe a bundle the server is no longer serving. Vite puts the content hash in
/// the filename, so this changes exactly when the UI changes.
///
/// A browser tab compares this against the script it is *running* (`web/src/lib/build.ts`). They differ
/// for one reason: the tab was loaded before the server was rebuilt under it — this project's own
/// workflow, which restarts the server after every batch. The consequence used to be silent and
/// arbitrarily bad; an old client reading a newer comparison shape declared two unrelated checkpoints
/// "structurally identical".
///
/// `None` when the shell holds no hashed entry script — a dev server proxying to Vite, where the tab's
/// module is `/src/main.ts` and there is nothing to compare.
pub(super) fn build_id() -> Option<String> {
    let shell = WebAssets::get("index.html")?;
    let html = std::str::from_utf8(shell.data.as_ref()).ok()?;
    // `<script type="module" crossorigin src="/assets/index-c1322f20.js">` — the src of the first
    // module script, reduced to its basename.
    let at = html.find("/assets/index-")?;
    let rest = html.get(at..)?;
    let end = rest.find(".js")? + 3;
    Some(rest.get("/assets/".len()..end)?.to_string())
}

/// Content-type for a served asset path, by extension (covers the SPA's outputs).
pub(super) fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("html") => "text/html; charset=utf-8",
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json" | "map") => "application/json; charset=utf-8",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        _ => "application/octet-stream",
    }
}
