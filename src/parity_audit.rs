//! The guard behind `docs/cli-web-parity.md`.
//!
//! The goal is that anything the CLI can do, the web UI can do. The risk is not that someone
//! *disagrees* with that — it is that the CLI grows a flag, the web quietly cannot do that thing, and
//! nobody writes it down. A prose promise cannot catch this; the flag list can.
//!
//! So: walk clap's own command tree, and require every long flag to appear in the ledger. Adding a CLI
//! flag then fails the build until its web status is recorded — as `yes`, `gap` or `n/a`. The ledger
//! stays a decision log rather than a snapshot that was true once.
//!
//! Same argument as `tests/parity.rs`, which makes the TUI/web *formatting* agreement a test for the
//! same reason: "comments saying 'mirrors the TUI' don't stop the two from drifting, and drift here is
//! the worst kind: silent."
//!
//! Clap is the source of truth, not `--help` output: the command tree is the definition, and parsing
//! help text would break on formatting.

#[cfg(test)]
mod tests {
    use clap::CommandFactory;

    /// Flags that are clap's, not ours — no web status to record.
    const CLAP_BUILTINS: &[&str] = &["help", "version"];

    fn ledger() -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/cli-web-parity.md");
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("the parity ledger must be readable at {path:?}: {e}"))
    }

    /// Every `(command path, long flag)` clap defines, recursing into subcommands.
    fn every_flag() -> Vec<(String, String)> {
        fn walk(cmd: &clap::Command, path: &str, out: &mut Vec<(String, String)>) {
            for arg in cmd.get_arguments() {
                if let Some(long) = arg.get_long() {
                    out.push((path.to_string(), long.to_string()));
                }
            }
            for sub in cmd.get_subcommands() {
                let name = sub.get_name();
                // `help` is clap's own generated subcommand.
                if name == "help" {
                    continue;
                }
                walk(sub, name, out);
            }
        }
        let mut out = Vec::new();
        walk(&crate::Cli::command(), BROWSING, &mut out);
        out
    }

    /// What the ledger calls the root command's section.
    const BROWSING: &str = "(browsing)";

    /// The flags each `##` section records, taken from the **first cell of its table rows** only.
    ///
    /// Rows, not the whole file: prose naturally names flags while explaining things — the overview
    /// paragraph mentions `--verify-repack` — and an earlier version of this check accepted that, so
    /// deleting a flag's row still passed. A row is the deliberate act; a sentence is not.
    ///
    /// Per section, not globally, so a flag cannot be recorded against the wrong subcommand: `--name`
    /// means three different things here, and each needs its own answer.
    fn recorded_flags(doc: &str) -> std::collections::HashMap<String, Vec<String>> {
        let mut out: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();
        let mut section = String::new();
        for line in doc.lines() {
            if let Some(heading) = line.strip_prefix("## ") {
                // `## `diff`` → `diff`; the root command's heading is prose, mapped to BROWSING.
                section = if heading.starts_with("Browsing") {
                    BROWSING.to_string()
                } else {
                    heading.trim().trim_matches('`').to_string()
                };
                continue;
            }
            let Some(first_cell) = line.strip_prefix('|').and_then(|r| r.split('|').next()) else {
                continue;
            };
            let flags = out.entry(section.clone()).or_default();
            let mut rest = first_cell;
            while let Some(at) = rest.find("--") {
                rest = &rest[at + 2..];
                let end = rest
                    .find(|c: char| !c.is_ascii_lowercase() && !c.is_ascii_digit() && c != '-')
                    .unwrap_or(rest.len());
                if end > 0 {
                    flags.push(rest[..end].to_string());
                }
                rest = &rest[end..];
            }
        }
        out
    }

    /// **Every CLI flag has a recorded web status, in its own command's table.**
    ///
    /// Fails naming the flag and the command, so the fix is to add a row rather than to work out what
    /// changed.
    #[test]
    fn every_cli_flag_is_accounted_for_in_the_parity_ledger() {
        let doc = ledger();
        let recorded = recorded_flags(&doc);
        let flags = every_flag();
        assert!(
            flags.len() > 60,
            "the walk should find the whole CLI surface, found {}",
            flags.len()
        );

        let missing: Vec<String> = flags
            .into_iter()
            .filter(|(_, long)| !CLAP_BUILTINS.contains(&long.as_str()))
            .filter(|(cmd, long)| {
                !recorded
                    .get(cmd)
                    .is_some_and(|rows| rows.iter().any(|r| r == long))
            })
            .map(|(cmd, long)| format!("{cmd}: --{long}"))
            .collect();
        assert!(
            missing.is_empty(),
            "these CLI flags have no recorded web status — add a row under the matching `##` heading \
             in docs/cli-web-parity.md saying `yes`, `gap` or `n/a`:\n  {}",
            missing.join("\n  ")
        );
    }

    /// **The ledger's route names still exist.**
    ///
    /// The other direction of rot: the ledger lists the web API's routes, and a renamed or deleted
    /// route would leave it describing a surface that is gone. Checked against the router's source,
    /// because the router is a `match` on string literals and there is no list to import.
    #[test]
    fn the_routes_the_ledger_names_still_exist() {
        let doc = ledger();
        let router = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/web/mod.rs"),
        )
        .expect("the router source is readable");

        // The routes named in the ledger's inventory paragraph, as `` `name` `` spans.
        let listed: Vec<&str> = doc
            .split("The web API has ")
            .nth(1)
            .expect("the ledger states its route inventory")
            .split("\n\n")
            .next()
            .expect("that paragraph ends")
            .split('`')
            .skip(1)
            .step_by(2)
            .collect();
        assert!(
            listed.len() > 10,
            "the ledger should name the whole API surface, found {listed:?}"
        );
        let gone: Vec<&&str> = listed
            .iter()
            .filter(|r| !router.contains(&format!("\"{r}\"")))
            .collect();
        assert!(
            gone.is_empty(),
            "docs/cli-web-parity.md names routes the router no longer has: {gone:?}"
        );
    }
}
