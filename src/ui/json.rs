//! Metadata JSON rendering — a `serde_json` formatter that keeps scalar arrays
//! on one line, plus the syntax highlighter both frontends print through.

use ratatui::style::Color;
use ratatui::text::{Line, Span};

use super::palette;
use super::theme::to_yansi;

/// JSON highlighting styled from the app palette, so a metadata config reads in
/// the same colors as the rest of the UI: keys in the structural cyan accent
/// (like tree groups), numbers in the amber dtype color, strings green, and the
/// `{}`/`[]` brackets in the normal foreground — the same contrast as the commas
/// and other punctuation colored_json leaves unstyled — while the colons stay
/// dimmed so key/value separators recede behind the values.
fn json_styler() -> colored_json::Styler {
    let dim = to_yansi(palette::DIM).foreground();
    let bracket = to_yansi(Color::Reset).foreground();
    colored_json::Styler {
        object_brackets: bracket,
        object_colon: dim,
        array_brackets: bracket,
        key: to_yansi(palette::ACCENT).bold(),
        string_value: to_yansi(palette::SUCCESS).foreground(),
        integer_value: to_yansi(palette::DTYPE).foreground(),
        float_value: to_yansi(palette::DTYPE).foreground(),
        bool_value: to_yansi(palette::WARN).foreground(),
        nil_value: dim,
        string_include_quotation: true,
    }
}

/// If `raw` is a JSON object or array, pretty-print it with syntax highlighting
/// (via `colored_json`, styled from [`json_styler`]) and return one ANSI-colored
/// string per line; otherwise `None`, so the caller shows the raw text. Bare
/// scalars (a lone string/number) aren't worth reformatting, so they fall
/// through to the raw path too.
fn highlight_json(raw: &str, inline_arrays: bool) -> Option<Vec<String>> {
    let value: serde_json::Value = serde_json::from_str(raw.trim()).ok()?;
    if !value.is_object() && !value.is_array() {
        return None;
    }
    // `colored_json` paints via yansi, whose default condition drops the ANSI
    // codes when stdout isn't a detected TTY (which would also make the result
    // non-deterministic). We render into our own buffer and own the terminal, so
    // force coloring on. `inline_arrays` swaps the layout formatter so a
    // safetensors header's flat arrays (shape / data_offsets) stay on one line.
    yansi::enable();
    let styler = json_styler();
    let on = colored_json::ColorMode::On;
    let pretty = if inline_arrays {
        colored_json::ColoredFormatter::with_styler(ObjectPrettyArrayInline::default(), styler)
            .to_colored_json(&value, on)
            .ok()?
    } else {
        colored_json::ColoredFormatter::with_styler(colored_json::PrettyFormatter::new(), styler)
            .to_colored_json(&value, on)
            .ok()?
    };
    Some(pretty.split('\n').map(str::to_string).collect())
}

/// Pretty-print `raw` JSON with flat scalar arrays inline (like
/// [`highlight_json_lines_inline`]) but **without** colour — as plain Ratatui
/// lines. Far cheaper than the highlighted path (no `colored_json` ANSI + no
/// `ansi-to-tui` parse), so a huge safetensors header renders instantly. Returns
/// `None` for non-JSON.
pub fn plain_json_lines_inline(raw: &str) -> Option<Vec<Line<'static>>> {
    let value: serde_json::Value = serde_json::from_str(raw.trim()).ok()?;
    if !value.is_object() && !value.is_array() {
        return None;
    }
    // `ColorMode::Off` keeps our layout formatter but emits no ANSI codes.
    let pretty = colored_json::ColoredFormatter::with_styler(
        ObjectPrettyArrayInline::default(),
        json_styler(),
    )
    .to_colored_json(&value, colored_json::ColorMode::Off)
    .ok()?;
    Some(
        pretty
            .split('\n')
            .map(|l| Line::from(Span::raw(l.to_string())))
            .collect(),
    )
}

/// [`highlight_json`] parsed back into styled Ratatui lines (via `ansi-to-tui`),
/// or `None` for non-JSON. Shared by the metadata detail view and the copy-menu
/// preview so both show the same `colored_json` palette.
pub fn highlight_json_lines(raw: &str) -> Option<Vec<Line<'static>>> {
    json_to_lines(raw, false)
}

/// Like [`highlight_json_lines`], but flat scalar arrays stay on one line — for
/// the safetensors header preview (`shape` / `data_offsets` read as `[a, b]`).
pub fn highlight_json_lines_inline(raw: &str) -> Option<Vec<Line<'static>>> {
    json_to_lines(raw, true)
}

fn json_to_lines(raw: &str, inline_arrays: bool) -> Option<Vec<Line<'static>>> {
    use ansi_to_tui::IntoText;
    let mut lines = highlight_json(raw, inline_arrays)?
        .join("\n")
        .into_text()
        .ok()?
        .lines;
    // `colored_json`'s resets parse to an explicit `bg = Reset`, which would
    // paint the terminal's default background over a panel (e.g. the copy-menu
    // pop-up). Drop it so each span inherits whatever container draws it.
    for span in lines.iter_mut().flat_map(|line| line.spans.iter_mut()) {
        span.style.bg = None;
    }
    Some(lines)
}

/// A `serde_json` formatter that pretty-prints objects (one key per line) but
/// keeps arrays inline (`[1, 2, 3]`). safetensors headers only contain flat
/// scalar arrays (a tensor's `shape` / `data_offsets`), so this reads far better
/// than the default element-per-line arrays. Fed to `colored_json`, which colours
/// the values while this controls the layout.
#[derive(Default)]
struct ObjectPrettyArrayInline {
    indent: usize,
    has_value: bool,
}

impl serde_json::ser::Formatter for ObjectPrettyArrayInline {
    fn begin_object<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        self.indent += 1;
        self.has_value = false;
        w.write_all(b"{")
    }
    fn end_object<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        self.indent -= 1;
        if self.has_value {
            w.write_all(b"\n")?;
            json_indent(w, self.indent)?;
        }
        w.write_all(b"}")
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        w.write_all(if first { b"\n" } else { b",\n" })?;
        json_indent(w, self.indent)
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
    fn end_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        let _ = w;
        self.has_value = true;
        Ok(())
    }
    fn begin_array<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b"[")
    }
    fn end_array<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b"]")
    }
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn end_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        let _ = w;
        self.has_value = true;
        Ok(())
    }
}

fn json_indent<W: ?Sized + std::io::Write>(w: &mut W, levels: usize) -> std::io::Result<()> {
    for _ in 0..levels {
        w.write_all(b"  ")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tests_support::strip_ansi_codes;

    #[test]
    fn highlight_json_colors_objects_and_arrays_only() {
        // Non-JSON text and bare scalars fall through to the raw path.
        assert!(highlight_json("just some text", false).is_none());
        assert!(highlight_json("\"a lone string\"", false).is_none());
        assert!(highlight_json("42", false).is_none());

        let raw = r#"{"b":[true,null,"x"],"a":1}"#;
        let lines = highlight_json(raw, false).expect("an object is highlighted");
        let joined = lines.join("\n");
        // Styled from the app palette: keys in the ACCENT color, numbers in the
        // DTYPE color (256-color SGR `38;5;<n>`), not colored_json's defaults.
        assert!(
            joined.contains("38;5;81"),
            "expected keys in the ACCENT color (81)"
        );
        assert!(
            joined.contains("38;5;215"),
            "expected numbers in the DTYPE color (215)"
        );
        // Stripping the color recovers exactly serde_json's pretty layout, so the
        // highlighter only adds color and never alters the text itself.
        let value: serde_json::Value = serde_json::from_str(raw).unwrap();
        assert_eq!(
            strip_ansi_codes(&joined),
            serde_json::to_string_pretty(&value).unwrap()
        );
    }

    #[test]
    fn inline_arrays_keep_scalar_arrays_on_one_line() {
        // The safetensors-header variant renders a tensor's shape / data_offsets
        // inline — as they actually appear in the rendered (colored → Ratatui)
        // lines, not just in an isolated formatter.
        let raw = r#"{"w":{"dtype":"BF16","shape":[152064,4096],"data_offsets":[0,16]}}"#;
        let lines = highlight_json_lines_inline(raw).expect("json highlights");
        let text = |l: &Line| -> String { l.spans.iter().map(|s| s.content.as_ref()).collect() };
        // Some single rendered line carries the whole shape inline.
        assert!(
            lines.iter().any(|l| text(l).contains("[152064, 4096]")),
            "shape inline:\n{}",
            lines.iter().map(text).collect::<Vec<_>>().join("\n")
        );
        assert!(
            lines.iter().any(|l| text(l).contains("[0, 16]")),
            "offsets inline"
        );
        // The object is still expanded — dtype and shape land on different lines.
        assert!(lines.len() > 4, "object still multi-line: {}", lines.len());
    }

    #[test]
    fn plain_inline_matches_the_highlighted_layout_without_color() {
        // The fast (uncoloured) path used for large headers keeps arrays inline
        // and produces the same text as the highlighted path, minus the ANSI.
        let raw = r#"{"w":{"dtype":"BF16","shape":[152064,4096],"data_offsets":[0,16]}}"#;
        let text = |ls: &[Line]| {
            ls.iter()
                .map(|l| {
                    l.spans
                        .iter()
                        .map(|s| s.content.as_ref())
                        .collect::<String>()
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        let plain = plain_json_lines_inline(raw).expect("plain json");
        assert!(
            text(&plain).contains("[152064, 4096]"),
            "shape inline (plain)"
        );
        // No colour: every span uses the default (Reset) foreground.
        assert!(
            plain
                .iter()
                .flat_map(|l| l.spans.iter())
                .all(|s| s.style.fg.is_none() || s.style.fg == Some(Color::Reset)),
            "plain lines carry no colour"
        );
        // Same text as the highlighted variant (stripped of styling).
        assert_eq!(
            text(&plain),
            text(&highlight_json_lines_inline(raw).unwrap())
        );
    }
}
