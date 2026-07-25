/// Standard (RFC 4648) base64 encoding. Used to wrap clipboard text in the
/// OSC 52 terminal escape; avoids pulling in a dependency for ~20 lines.
pub fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[((n >> 6) & 0x3f) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(n & 0x3f) as usize] as char
        } else {
            '='
        });
    }
    out
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
pub fn term_width(fallback: usize) -> usize {
    terminal_size::terminal_size()
        .map(|(w, _)| w.0 as usize)
        .unwrap_or(fallback)
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
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

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

    if unit_idx == 0 {
        format!("{} {}", bytes, UNITS[unit_idx])
    } else {
        format!("{:.1} {}", size, UNITS[unit_idx])
    }
}

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
