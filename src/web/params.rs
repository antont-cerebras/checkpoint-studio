//! **One table of a comparison's parameters, and every spelling of them derived from it.**
//!
//! A comparison is described by the same set of facts wherever it appears: in a URL the browser holds,
//! in the query a request carries, and in the `checkpoint-studio diff …` invocation the UI offers so
//! the run can be repeated in a terminal. Those three used to be written out separately — a query
//! parser here, an allowlist there, a `cli_args` chain in a third place, and (worst) a command string
//! assembled by hand in the browser's Data view. Every one of them was a place to forget a parameter,
//! and the failures were silent in the direction that matters most: a comparison scoped to one tensor
//! offering a command that compares all 117,664, or a control the server accepted and the command
//! dropped.
//!
//! So the parameters are a table. Each row says the key — which is the query parameter *and* the URL's
//! parameter — and how that key renders as a command-line argument. From the table come:
//!
//! * the accepted-parameter allowlist (`web::mod`'s refusal of unknown keys, published to the browser
//!   through `shared/parity/queryparams.json`),
//! * the `diff` flags for any set of parameters ([`render`]), used by the report's offered command, by
//!   `GET /api/command`, and so by every surface that shows one.
//!
//! A new control is one row. Adding it makes it acceptable *and* renderable; there is no second place
//! to update and therefore no second place to forget.

use crate::web::handlers::Query;

/// Which side of a comparison an operand-borne parameter belongs to.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Side {
    Baseline,
    Candidate,
}

/// How a parameter renders on a command line.
#[derive(Clone, Copy)]
pub(crate) enum Cli {
    /// `--flag VALUE`, once per non-blank line of the field. `--name` is repeatable, and one glob per
    /// line is how the browser's box holds several.
    PerLine(&'static str),
    /// `--flag VALUE` with the value verbatim.
    Value(&'static str),
    /// `--flag VALUE` with the field's lines joined by commas — for a list whose flag takes one value.
    LinesAsList(&'static str),
    /// `--flag` when the switch is on, nothing when it is off.
    Switch(&'static str),
    /// Not a flag at all: it rides on the operand as `SOURCE#value` (`diff 'hf#language_model' conv`),
    /// which is how `diff` spells a subtree. [`subtrees`] hands these to the command builder.
    OnOperand(Side),
    /// Carried in the URL and on the wire, but not a `diff` argument: a *view* control (which tab, which
    /// sections are folded) or a knob the offered command has no equivalent for. Named explicitly so the
    /// table says "considered and deliberately nothing" rather than leaving a gap.
    NotAnArgument,
}

/// One parameter of a comparison, in every spelling it has.
pub(crate) struct Param {
    /// The query key — and the URL's parameter name, which is the same string by design.
    pub key: &'static str,
    pub cli: Cli,
    /// The browser's field name for it, when the browser holds one.
    ///
    /// `None` for a parameter no UI edits: `names_list` and `map_json` are paste-a-file forms the CLI
    /// has and the panel does not. The rest are generated into `web/src/lib/params.generated.ts`, so the
    /// client's own encode/decode is built from *these* rows rather than from a second list of the same
    /// strings — see the module note.
    ///
    /// Read by the generator ([`typescript_table`]), which runs as a test: the committed module is
    /// checked against it on every run and rewritten with `UPDATE_PARITY=1`. So in a non-test build this
    /// field is data nobody reads, which is exactly what it looks like.
    #[cfg_attr(not(test), allow(dead_code, reason = "read by the generator test"))]
    pub ts: Option<&'static str>,
}

const fn p(key: &'static str, cli: Cli, ts: Option<&'static str>) -> Param {
    Param { key, cli, ts }
}

/// **The selection**: what a comparison compares, and how the two sides are lined up first.
///
/// Accepted by every route that answers about a comparison, and rendered into the offered command.
pub(crate) const SCOPE: &[Param] = &[
    p("name", Cli::PerLine("--name"), Some("name")),
    p("names", Cli::LinesAsList("--names"), Some("names")),
    // `--names-from` takes a *path*; a pasted list is not one, so it folds into `--names` as well.
    p("names_list", Cli::LinesAsList("--names"), None),
    p("dtype_is", Cli::Value("--dtype-is"), Some("dtypeIs")),
    p("shape_is", Cli::Value("--shape-is"), Some("shapeIs")),
    p("map", Cli::PerLine("--map"), Some("map")),
    // A rename map pasted as JSON: the CLI takes `--map-json PATH`, and the content is not a path.
    // Its *rules* are rendered as `--map` above, from the same compiled map.
    p("map_json", Cli::NotAnArgument, None),
    p(
        "only_tensors",
        Cli::Switch("--only-tensors"),
        Some("onlyTensors"),
    ),
    p(
        "align_fused",
        Cli::Switch("--align-fused"),
        Some("alignFused"),
    ),
    p("subtree", Cli::OnOperand(Side::Baseline), Some("subtree")),
    p(
        "subtree_new",
        Cli::OnOperand(Side::Candidate),
        Some("subtreeNew"),
    ),
    // **How each side's packed tensors are decoded**, one row per side. Part of the scope rather than of
    // a particular check, because it is the same kind of thing as a rename rule or the fused alignment:
    // something applied to a side *before* anything is compared. The two sides are separate rows because
    // the two sides are exactly what differs — a sparse baseline (`[4]`) against a merged candidate
    // (`[3,3,3,3,3]`), which no single width describes.
    p(
        "repack_schema",
        Cli::Value("--repack-schema"),
        Some("repackSchema"),
    ),
    p(
        "repack_schema_new",
        Cli::Value("--repack-schema-new"),
        Some("repackSchemaNew"),
    ),
];

/// **The check**: which comparison to run over the selection, and how.
///
/// The structural report is the default and needs no flag; the rest are the Data view's jobs.
pub(crate) const CHECK: &[Param] = &[
    p("values", Cli::Switch("--values"), Some("values")),
    p("histogram", Cli::Switch("--histogram"), Some("histogram")),
    p("bins", Cli::Value("--bins"), Some("bins")),
    p(
        "verify_repack",
        Cli::Switch("--verify-repack"),
        Some("verifyRepack"),
    ),
    p(
        "repack_bits",
        Cli::Value("--repack-bits"),
        Some("repackBits"),
    ),
    p("tensor", Cli::Value("--tensor"), Some("tensor")),
    p("jobs", Cli::Value("--jobs"), Some("jobs")),
    // Every layer as its own row: it decides how the *structural report* prints, so it is an argument.
    p("full", Cli::Switch("--full"), Some("full")),
];

/// Every key of a table — the allowlist a route with that table accepts.
pub(crate) fn keys(table: &[Param]) -> Vec<&'static str> {
    table.iter().map(|p| p.key).collect()
}

/// The `#subtree` each side carries, from a query — `(baseline, candidate)`.
///
/// Read from the table rather than by name, so a second operand-borne parameter needs no change here.
pub(crate) fn subtrees(q: &Query) -> (Option<&str>, Option<&str>) {
    let side = |want: Side| {
        SCOPE
            .iter()
            .find(|p| matches!(p.cli, Cli::OnOperand(s) if s == want))
            .and_then(|p| q.get(p.key))
            .map(String::as_str)
            .filter(|v| !v.is_empty())
    };
    (side(Side::Baseline), side(Side::Candidate))
}

/// The `diff` arguments a query's parameters render to, in table order.
///
/// Reads the request rather than a compiled scope, which is the point: a parameter the server accepted
/// is rendered by the row that accepted it, so the two cannot disagree. Values are rendered raw —
/// quoting belongs to the command builder, which knows it is producing a shell line.
pub(crate) fn render(q: &Query, tables: &[&[Param]]) -> Vec<String> {
    let mut out = Vec::new();
    for param in tables.iter().copied().flatten() {
        let Some(raw) = q.get(param.key).map(String::as_str) else {
            continue;
        };
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        match param.cli {
            Cli::PerLine(flag) => {
                for line in raw.lines().map(str::trim).filter(|l| !l.is_empty()) {
                    out.push(flag.to_string());
                    out.push(line.to_string());
                }
            }
            Cli::Value(flag) => {
                out.push(flag.to_string());
                out.push(raw.to_string());
            }
            Cli::LinesAsList(flag) => {
                // Commas or newlines in, one comma-separated value out — the form the flag documents.
                let items: Vec<&str> = raw
                    .split([',', '\n'])
                    .map(str::trim)
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .collect();
                if !items.is_empty() {
                    out.push(flag.to_string());
                    out.push(items.join(","));
                }
            }
            Cli::Switch(flag) => {
                if matches!(raw, "1" | "true") {
                    out.push(flag.to_string());
                }
            }
            // Rendered onto the operands by the command builder, or deliberately not rendered.
            Cli::OnOperand(_) | Cli::NotAnArgument => {}
        }
    }
    out
}

/// The client's table, as TypeScript — generated from the rows above.
///
/// **Why generate rather than contract.** The browser has to encode and decode these parameters
/// itself: it owns its address bar. It held its own list of the same strings, kept in step by a test
/// that drove the real `api.*` calls and compared the keys they sent against the published allowlist
/// — which catches a key the *server* would refuse, and not a key the client reads back under the
/// wrong name. Generating the list removes the second copy instead of checking it: a row renamed here
/// renames the client's field, and TypeScript then fails to compile wherever that field is used by its
/// old name.
///
/// Written by `UPDATE_PARITY=1 cargo test the_client_parameter_table`, checked by that test otherwise.
#[cfg(test)]
pub(crate) fn typescript_table() -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    out.push_str(
        "// Generated from `src/web/params.rs` by `UPDATE_PARITY=1 cargo test \
         the_client_parameter_table`.\n// Do not edit: rename the row in Rust and regenerate, so the \
         wire, the URL, the CLI\n// rendering and this client agree by construction.\n\n",
    );
    let emit = |out: &mut String, name: &str, table: &[Param]| {
        let _ = writeln!(out, "export const {name} = [");
        for param in table.iter().filter(|p| p.ts.is_some()) {
            let field = param.ts.unwrap_or_default();
            // The client needs only "is it a switch": a text field round-trips as a string, a switch as
            // `1`/absent. Spelled out rather than wildcarded, so a new row kind has to be considered here.
            let kind = match param.cli {
                Cli::Switch(_) => "switch",
                Cli::PerLine(_)
                | Cli::Value(_)
                | Cli::LinesAsList(_)
                | Cli::OnOperand(_)
                | Cli::NotAnArgument => "text",
            };
            let _ = writeln!(
                out,
                "  {{ field: '{field}', key: '{key}', kind: '{kind}' }},",
                key = param.key
            );
        }
        out.push_str("] as const;\n\n");
    };
    emit(&mut out, "SCOPE_PARAMS", SCOPE);
    emit(&mut out, "CHECK_PARAMS", CHECK);
    out
}

#[cfg(test)]
mod tests {
    use super::{CHECK, Cli, Param, SCOPE, Side, render, subtrees};
    use crate::web::handlers::Query;

    fn q(pairs: &[(&str, &str)]) -> Query {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    /// A value that exercises the row's kind — a switch is on, everything else is a plausible input.
    fn sample(param: &Param) -> &'static str {
        match param.cli {
            Cli::Switch(_) => "1",
            Cli::PerLine(_) => "one\ntwo",
            Cli::LinesAsList(_) => "a.w\nb.w",
            Cli::Value(_) | Cli::OnOperand(_) | Cli::NotAnArgument => "value",
        }
    }

    /// **Every row of the table renders, or says why it does not.**
    ///
    /// The one property that matters here: a parameter the server accepts is a parameter the offered
    /// command carries. Both surfaces have shipped a command that quietly dropped a control — an exact
    /// name list, and then a whole selection — and both times the cause was a *second* place that had to
    /// list the parameter and did not. Walking the table means a new row is covered the moment it exists.
    #[test]
    fn every_parameter_either_renders_or_states_that_it_cannot() {
        for param in SCOPE.iter().chain(CHECK) {
            let query = q(&[(param.key, sample(param))]);
            let args = render(&query, &[SCOPE, CHECK]);
            match param.cli {
                Cli::PerLine(flag) | Cli::Value(flag) | Cli::LinesAsList(flag) => {
                    assert!(
                        args.contains(&flag.to_string()),
                        "{} must render as {flag}: {args:?}",
                        param.key
                    );
                    assert!(
                        args.len() >= 2,
                        "{} takes a value, so the flag cannot stand alone: {args:?}",
                        param.key
                    );
                }
                Cli::Switch(flag) => {
                    assert_eq!(args, [flag], "{} is a switch", param.key);
                }
                Cli::OnOperand(side) => {
                    // Not an argument: it rides on the operand, and `subtrees` is what hands it over.
                    assert!(args.is_empty(), "{} is not a flag: {args:?}", param.key);
                    let (baseline, candidate) = subtrees(&query);
                    let mine = if side == Side::Baseline {
                        baseline
                    } else {
                        candidate
                    };
                    assert_eq!(mine, Some("value"), "{} reaches the operand", param.key);
                }
                Cli::NotAnArgument => {
                    assert!(
                        args.is_empty(),
                        "{} is deliberately not an argument: {args:?}",
                        param.key
                    );
                }
            }
        }
    }

    /// An empty box is "unset", not "a pattern matching nothing" — a UI that always sends its fields
    /// would otherwise put an empty `--name` into every command it offers.
    #[test]
    fn blank_and_off_parameters_render_nothing() {
        let all_blank: Vec<(&str, &str)> = SCOPE
            .iter()
            .chain(CHECK)
            .map(|p| {
                (
                    p.key,
                    if matches!(p.cli, Cli::Switch(_)) {
                        "0"
                    } else {
                        ""
                    },
                )
            })
            .collect();
        assert!(render(&q(&all_blank), &[SCOPE, CHECK]).is_empty());
    }

    /// The keys are the *URL's* parameter names as well, so they have to be unique across the tables a
    /// route accepts together — one key cannot mean two things.
    #[test]
    fn the_keys_are_unique() {
        let mut keys: Vec<&str> = SCOPE.iter().chain(CHECK).map(|p| p.key).collect();
        keys.sort_unstable();
        let before = keys.len();
        keys.dedup();
        assert_eq!(
            before,
            keys.len(),
            "a duplicated key would be ambiguous: {keys:?}"
        );
    }
}
