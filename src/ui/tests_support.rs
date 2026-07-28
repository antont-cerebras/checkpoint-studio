//! Helpers shared by the render tests across the `ui` module tree.

/// Drop CSI escape sequences (`\x1b[…<letter>`) so a colored string can be
/// compared against its plain text.
pub(super) fn strip_ansi_codes(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}
