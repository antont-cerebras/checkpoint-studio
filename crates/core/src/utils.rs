/// Standard (RFC 4648) base64 encoding. Used to wrap clipboard text in the
/// OSC 52 terminal escape; avoids pulling in a dependency for ~20 lines.
#[must_use]
pub fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    // The `& 0x3f` masks each sextet into `0..64`, which is exactly `ALPHABET`'s length —
    // so `get` can only be `None` if the table were the wrong size, and `=` (base64's own
    // padding) is the least surprising thing to emit if it ever were.
    let sextet = |n: u32, shift: u32| {
        ALPHABET
            .get(((n >> shift) & 0x3f) as usize)
            .map_or('=', |&b| b as char)
    };
    for chunk in input.chunks(3) {
        let byte = |i: usize| u32::from(chunk.get(i).copied().unwrap_or(0));
        let n = (byte(0) << 16) | (byte(1) << 8) | byte(2);
        out.push(sextet(n, 18));
        out.push(sextet(n, 12));
        // The last two characters are padding when the chunk was short.
        out.push(if chunk.len() > 1 { sextet(n, 6) } else { '=' });
        out.push(if chunk.len() > 2 { sextet(n, 0) } else { '=' });
    }
    out
}

/// Expand a leading `~` to `$HOME` — `~`, `~/`, and `~/rest` only, never `~user`.
///
/// A shell does this before a CLI ever sees the argument, which is exactly why paths
/// typed *inside* the app need it done here: the web UI's compare box and the TUI's
/// compare prompt both send the literal `~`, and `~/ws/model` then resolves to nothing.
/// `~user` is deliberately not handled — resolving another account's home needs passwd
/// lookups, and silently treating `~bob/x` as a relative directory called `~bob` would be
/// worse than the error.
#[must_use]
pub fn expand_tilde(path: &str) -> std::path::PathBuf {
    let Some(rest) = path.strip_prefix('~') else {
        return std::path::PathBuf::from(path);
    };
    // `~user/...` — not ours to expand; hand it back unchanged so the error names what
    // was actually typed.
    if !rest.is_empty() && !rest.starts_with('/') {
        return std::path::PathBuf::from(path);
    }
    let Some(home) = std::env::var_os("HOME") else {
        return std::path::PathBuf::from(path);
    };
    let home = std::path::PathBuf::from(home);
    match rest.strip_prefix('/') {
        None | Some("") => home,
        Some(tail) => home.join(tail),
    }
}

/// Parse a human size like `1G`, `256M`, `64K` (binary, ×1024) or a bare byte
/// count, returning the number of bytes.
pub fn parse_size(s: &str) -> Result<usize, String> {
    let s = s.trim();
    let (num, mult) = match s.chars().last() {
        Some('G' | 'g') => (&s[..s.len() - 1], 1usize << 30),
        Some('M' | 'm') => (&s[..s.len() - 1], 1usize << 20),
        Some('K' | 'k') => (&s[..s.len() - 1], 1usize << 10),
        _ => (s, 1),
    };
    num.trim()
        .parse::<usize>()
        .map(|n| n * mult)
        .map_err(|_| format!("invalid size '{s}' (use e.g. 64M, 256M, 1G)"))
}

/// Terminal width in columns, or `fallback` when not attached to a tty — the
/// core's frontend-free replacement for `crossterm::terminal::size()`, so output
/// formatters (diff, progress bars) can fit the terminal without depending on the
/// full crossterm terminal layer.
#[must_use]
pub fn term_width(fallback: usize) -> usize {
    terminal_size::terminal_size().map_or(fallback, |(w, _)| w.0 as usize)
}

// The display formatters below are duplicated in the web client
// (`web/src/lib/format.ts`), because a browser can't call into this crate. Their
// agreement is not left to comments: `tests/parity.rs` generates
// `shared/parity/format.json` from the functions here and the web's `parity.test.ts`
// asserts the TypeScript produces the same strings, so drift fails CI on both sides.
// Change one, regenerate, and the other side's test tells you what to change.

pub fn format_shape(shape: &[usize]) -> String {
    format!(
        "({})",
        shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

#[must_use]
pub fn format_size(bytes: usize) -> String {
    // Sizes are scaled by 1024, so use the binary (IEC) unit labels. Up to PiB: a
    // fleet-sized total shouldn't read as "5120.0 GiB".
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    // The loop above stops at `UNITS.len() - 1`, so the unit is always there; `"B"` is the
    // honest fallback for a table that had been emptied.
    let unit = UNITS.get(unit_idx).copied().unwrap_or("B");
    if unit_idx == 0 {
        format!("{bytes} {unit}")
    } else {
        format!("{size:.1} {unit}")
    }
}

#[must_use]
pub fn format_parameters(params: usize) -> String {
    if params < 1_000 {
        format!("{params}")
    } else if params < 1_000_000 {
        format!("{:.1}K", params as f64 / 1_000.0)
    } else if params < 1_000_000_000 {
        format!("{:.1}M", params as f64 / 1_000_000.0)
    } else if params < 1_000_000_000_000 {
        format!("{:.1}B", params as f64 / 1_000_000_000.0)
    } else {
        // Trillions: frontier checkpoints are already here, and "1500.0B" reads worse.
        format!("{:.1}T", params as f64 / 1_000_000_000_000.0)
    }
}

/// A fraction (0.0–1.0) as a percentage for display. An exact zero reads `0%`; a
/// tiny-but-nonzero fraction switches to scientific notation, so a checkpoint with a
/// handful of zeros among billions of values never displays a misleading `0.0%`.
///
/// `is_zero` comes from the true count rather than the fraction, so floating-point
/// dust can't masquerade as an exact zero.
#[must_use]
pub fn format_percent(fraction: f64, is_zero: bool) -> String {
    if is_zero {
        return "0%".to_string();
    }
    let pct = fraction * 100.0;
    if pct < 0.1 {
        format!("{pct:.1e}%")
    } else {
        format!("{pct:.1}%")
    }
}

#[cfg(test)]
mod tests {
    use super::expand_tilde;
    use std::path::PathBuf;

    #[test]
    fn tilde_expands_only_the_forms_a_shell_would() {
        // SAFETY-ish: tests in this module are the only ones touching HOME, and the value
        // is restored below, so no other test observes the change.
        let saved = std::env::var_os("HOME");
        unsafe { std::env::set_var("HOME", "/home/u") };

        assert_eq!(expand_tilde("~"), PathBuf::from("/home/u"));
        assert_eq!(expand_tilde("~/"), PathBuf::from("/home/u"));
        assert_eq!(
            expand_tilde("~/ws/model"),
            PathBuf::from("/home/u/ws/model")
        );
        // Not a tilde path at all.
        assert_eq!(expand_tilde("/abs/path"), PathBuf::from("/abs/path"));
        assert_eq!(expand_tilde("rel/path"), PathBuf::from("rel/path"));
        // `~user` is left alone rather than being mangled into a relative directory.
        assert_eq!(expand_tilde("~bob/x"), PathBuf::from("~bob/x"));
        // A tilde inside the path is just a character.
        assert_eq!(expand_tilde("/a/~/b"), PathBuf::from("/a/~/b"));

        match saved {
            Some(v) => unsafe { std::env::set_var("HOME", v) },
            None => unsafe { std::env::remove_var("HOME") },
        }
    }
}
