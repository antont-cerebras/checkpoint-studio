//! User CLI defaults, read once at startup from a small config file so common
//! flags (the SSH proxy, especially) needn't be retyped every invocation. Every
//! field is optional and an explicit CLI flag always wins (see `resolve_ssh_proxy`
//! in `main`). Parsed with a tiny `key = "value"` reader — a TOML-compatible subset
//! sufficient for a couple of string keys — so there's no config-parser dependency.

use std::path::PathBuf;

/// The CLI defaults a user can set in `config.toml`. All optional.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct CliConfig {
    /// Default `--ssh-proxy` host (`[USER@]HOST`) for reading remote / `s3://`
    /// checkpoints — the tedious-to-retype one.
    pub ssh_proxy: Option<String>,
    /// Default `--ssh-venv` (the cstorch virtualenv path on that host).
    pub ssh_venv: Option<String>,
}

impl CliConfig {
    /// Load from [`Self::path`], or return the defaults (all `None`) when the file
    /// is absent or unreadable — a missing/typo'd config is never fatal.
    pub(crate) fn load() -> Self {
        Self::path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map(|s| Self::parse(&s))
            .unwrap_or_default()
    }

    /// The config file path: `$XDG_CONFIG_HOME/checkpoint-studio/config.toml`, or
    /// `$HOME/.config/checkpoint-studio/config.toml`. `None` if neither var is set.
    pub(crate) fn path() -> Option<PathBuf> {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .filter(|p| !p.as_os_str().is_empty())
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
        Some(base.join("checkpoint-studio").join("config.toml"))
    }

    /// Parse `key = "value"` lines (a TOML subset): blank lines and `#` comments are
    /// ignored, values may be quoted or bare, and unknown keys are ignored (so a
    /// newer config doesn't break an older binary). Accepts `ssh_proxy`/`ssh-proxy`.
    pub(crate) fn parse(text: &str) -> Self {
        let mut cfg = Self::default();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, val)) = line.split_once('=') else {
                continue;
            };
            // Strip surrounding quotes, then any inner whitespace, then an inline
            // `# comment` on a bare (unquoted) value.
            let val = val.trim();
            let val = if let Some(inner) = val
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .or_else(|| val.strip_prefix('\'').and_then(|v| v.strip_suffix('\'')))
            {
                inner.to_string()
            } else {
                val.split('#').next().unwrap_or("").trim().to_string()
            };
            if val.is_empty() {
                continue;
            }
            match key.trim() {
                "ssh_proxy" | "ssh-proxy" => cfg.ssh_proxy = Some(val),
                "ssh_venv" | "ssh-venv" => cfg.ssh_venv = Some(val),
                _ => {}
            }
        }
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::CliConfig;

    #[test]
    fn parses_quoted_and_bare_values() {
        let cfg = CliConfig::parse(
            "# my defaults\nssh_proxy = \"lab@host.example.com\"\nssh_venv = ~/envs/cstorch\n",
        );
        assert_eq!(cfg.ssh_proxy.as_deref(), Some("lab@host.example.com"));
        assert_eq!(cfg.ssh_venv.as_deref(), Some("~/envs/cstorch"));
    }

    #[test]
    fn ignores_comments_blanks_and_unknown_keys() {
        let cfg = CliConfig::parse("\n\n  # comment\nunknown = 5\nssh_proxy = h  # inline note\n");
        assert_eq!(cfg.ssh_proxy.as_deref(), Some("h"));
        assert_eq!(cfg.ssh_venv, None);
    }

    #[test]
    fn empty_and_malformed_are_harmless() {
        assert_eq!(CliConfig::parse(""), CliConfig::default());
        // A line with no `=`, and an empty value, are skipped (not fatal).
        assert_eq!(
            CliConfig::parse("just some text\nssh_proxy =\n"),
            CliConfig::default()
        );
    }

    #[test]
    fn hyphen_key_aliases_are_accepted() {
        let cfg = CliConfig::parse("ssh-proxy = 'user@h'\n");
        assert_eq!(cfg.ssh_proxy.as_deref(), Some("user@h"));
    }
}
