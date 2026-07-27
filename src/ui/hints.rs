//! The footer key hints and the clickable chips they turn into. Every screen's
//! hint line is built here so the wording and the key order stay consistent, and
//! so a click on a hint can be replayed as the keypress it stands for.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use crate::sample::SampleMode;
use crate::viewstate::{NumBase, StripeMode};

use super::ChipRegions;
use super::palette;
use super::theme::key_span;

/// A clickable footer key-hint chip: where it sits within a hint block (line
/// index + column + width) and the key it stands for. The `render_*` functions
/// translate these to absolute screen [`Rect`]s and pair them with the key, so a
/// click can be turned into the equivalent keypress.
pub(crate) struct ChipHit {
    pub line: u16,
    pub col: u16,
    pub width: u16,
    pub key: KeyEvent,
}

/// A plain (no-modifier) key event — what clicking a single-letter hint stands for.
pub(super) fn hint_key(c: char) -> KeyEvent {
    KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
}

/// A piece of a footer chip's key text: either a clickable glyph paired with the
/// key it synthesizes, or a non-clickable separator (`/`, `Shift+`). The footer
/// builders emit one [`ChipHit`] per [`Seg::Key`] at its own sub-column, so each
/// half of a dual chip (`E/C`, `↑/↓`, `⌫/\`, …) is independently clickable.
pub(super) enum Seg {
    Key(&'static str, KeyEvent),
    Sep(&'static str),
}

impl Seg {
    fn text(&self) -> &'static str {
        match self {
            Self::Key(t, _) | Self::Sep(t) => t,
        }
    }
}

/// A chip for a plain single-character key, where the glyph *is* the key —
/// `letter("y")` rather than `Seg::Key("y", hint_key('y'))`.
///
/// Writing the character once removes the only interesting way one of these can be wrong:
/// a label that doesn't match the key it synthesizes, which nothing but a click would
/// reveal. Use `Seg::Key` directly where the label is a word (`Tab`, `Enter`) or a glyph
/// standing for a non-character key (`⌫`, `↵`).
fn letter(text: &'static str) -> Vec<Seg> {
    let c = text.chars().next().unwrap_or(' ');
    vec![Seg::Key(text, hint_key(c))]
}

/// A dual chip: two independently clickable keys joined by a non-clickable `/`.
fn pair(a: &'static str, ka: KeyEvent, b: &'static str, kb: KeyEvent) -> Vec<Seg> {
    vec![Seg::Key(a, ka), Seg::Sep("/"), Seg::Key(b, kb)]
}

// The chips that appear on more than one screen, each built once. These are the same
// bindings in the same order everywhere they show up, and spelling them out per screen
// meant the glyph and the key it synthesizes were re-paired by hand five or six times —
// a `↓` chip wired to `Up` is a bug you can only find by clicking it. They also have to
// match the web UI's footers (see the TUI/web parity note in the README), which is easier
// to check against one definition than against six copies.

/// `↑`/`↓` — move the selection by a row.
fn nav_updown() -> Vec<Seg> {
    let plain = KeyModifiers::NONE;
    pair(
        "↑",
        KeyEvent::new(KeyCode::Up, plain),
        "↓",
        KeyEvent::new(KeyCode::Down, plain),
    )
}

/// `←`/`→` — collapse/expand, or step by column.
fn nav_leftright() -> Vec<Seg> {
    let plain = KeyModifiers::NONE;
    pair(
        "←",
        KeyEvent::new(KeyCode::Left, plain),
        "→",
        KeyEvent::new(KeyCode::Right, plain),
    )
}

/// `PgUp`/`PgDn` — move by a screenful.
fn nav_pages() -> Vec<Seg> {
    let plain = KeyModifiers::NONE;
    pair(
        "PgUp",
        KeyEvent::new(KeyCode::PageUp, plain),
        "PgDn",
        KeyEvent::new(KeyCode::PageDown, plain),
    )
}

/// `Space`/`:` — open the command palette.
fn palette_keys() -> Vec<Seg> {
    pair("Space", hint_key(' '), ":", hint_key(':'))
}

/// `⌫`/`\` — step back and forward through the view history.
fn history_keys() -> Vec<Seg> {
    pair(
        "⌫",
        KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
        "\\",
        hint_key('\\'),
    )
}

/// Draw a `[×]` close control in the top-right corner and return its clickable
/// region paired with the key a click should synthesize (`q` to quit the tree,
/// `⌫` to step back from a sub-screen). No-op (empty region list) if too narrow.
pub(super) fn close_button(frame: &mut Frame, key: KeyEvent) -> Vec<(Rect, KeyEvent)> {
    let area = frame.area();
    if area.width < 3 {
        return Vec::new();
    }
    let rect = Rect {
        x: area.width - 3,
        y: 0,
        width: 3,
        height: 1,
    };
    frame
        .buffer_mut()
        .set_string(rect.x, rect.y, "[×]", Style::default().fg(palette::ACCENT));
    vec![(rect, key)]
}

/// Translate a data view's footer [`ChipHit`]s (lines relative to `footer_top`)
/// into absolute screen regions and append the top-right `[×]` (→ step back).
/// Shared by the heatmap and numeric-grid renderers, which lay out identically.
pub(super) fn data_view_regions(
    frame: &mut Frame,
    chips: &[ChipHit],
    footer_top: u16,
) -> Vec<(Rect, KeyEvent)> {
    let mut regions = chip_regions(chips, footer_top);
    regions.extend(close_button(
        frame,
        KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
    ));
    regions
}

/// True when `(col, row)` falls inside a clickable region.
pub(crate) fn region_hit(regions: &[(Rect, KeyEvent)], col: u16, row: u16) -> Option<KeyEvent> {
    region_at(regions, col, row).map(|(_, k)| k)
}

/// The clickable region (its rect and key) under `(col, row)`, if any — like
/// [`region_hit`] but keeps the rect too, so a hover can anchor a help bubble to
/// the chip it points at.
pub(crate) fn region_at(
    regions: &[(Rect, KeyEvent)],
    col: u16,
    row: u16,
) -> Option<(Rect, KeyEvent)> {
    regions
        .iter()
        .find(|(r, _)| col >= r.x && col < r.x + r.width && row >= r.y && row < r.y + r.height)
        .copied()
}

/// Map a footer's [`ChipHit`]s (line/col relative to their hint block) to absolute
/// screen [`Rect`]s paired with each chip's key. `base_row` is the block's first
/// screen row: `1` for the tree/files header hints, or `footer_top` for the modes
/// whose hints sit in the bottom footer. Replaces the per-mode remap that was
/// copy-pasted across every screen renderer.
pub(super) fn chip_regions(chips: &[ChipHit], base_row: u16) -> ChipRegions {
    chips
        .iter()
        .map(|c| {
            (
                Rect {
                    x: c.col,
                    y: base_row + c.line,
                    width: c.width,
                    height: 1,
                },
                c.key,
            )
        })
        .collect()
}

/// Which screen a footer shortcut sits on, so [`shortcut_help`] can disambiguate
/// keys that mean different things per screen (`h`, `b`, `r`, the arrows).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum HelpCtx {
    Tree,
    /// The compare screen (a structural diff against another checkpoint).
    Diff,
    Files,
    Layout,
    Detail,
    Data,
    Rename,
    Stats,
}

/// A one-line help description for a footer shortcut `key` on screen `ctx`, shown
/// as a bubble when the mouse hovers the chip. `None` for keys with no help.
pub(crate) fn shortcut_help(key: KeyEvent, ctx: HelpCtx) -> Option<&'static str> {
    use HelpCtx::{Data, Detail, Files, Layout, Rename, Stats, Tree};
    use KeyCode::{Backspace, Char, Down, Left, PageDown, PageUp, Right, Tab, Up};
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    let help = match (ctx, key.code) {
        // File browser.
        (Tree, Tab) => "Switch to the file browser — the checkpoint's directory.",
        (Files, Tab) => "Switch back to the tensor tree.",
        // safetensors layout map.
        (Layout, Char(' ' | ':')) => "Open the command palette — search and run any command.",
        (Layout, Tab) => "Switch back to the tensor tree.",
        (Layout, Up | Down) => "Move the selection to the previous / next segment.",
        (Layout, PageUp | PageDown) => "Move the selection by one screenful.",
        (Layout, KeyCode::Enter) => "Jump to the selected tensor's place in the tensor tree.",
        (Detail, Tab) => "Show this tensor in its file's byte-layout map.",
        (Files, Up | Down) => "Move the selection up / down one row.",
        (Files, Left | Right) => "Collapse a directory / expand it (or step to its parent).",
        (Files, PageUp | PageDown) => "Scroll the listing by one screenful.",
        (Files, KeyCode::Enter) => {
            "Expand a directory, open a checkpoint file, or preview a text / JSON sidecar."
        }
        (Files, Char(' ' | ':')) => "Open the command palette — search and run any command.",
        (Files, Char('f')) => "Copy the selected file's path.",
        // Tree navigation.
        (Tree, Up | Down) if shift => "Jump to the previous / next sibling at this depth.",
        (Tree, Up | Down) => "Move the selection up / down one row.",
        (Tree, Left | Right) => "Collapse to the parent group, or step into the child.",
        (Tree, PageUp | PageDown) => "Scroll the tree by one screenful.",
        (Tree, KeyCode::Enter) => "Open the selected tensor, or expand / collapse a group.",
        (Tree, Char(' ' | ':')) => "Open the command palette — search and run any command.",
        (Tree, Char('e' | 'E')) => "Expand every group in the tree.",
        (Tree, Char('c' | 'C')) => "Collapse every group in the tree.",
        (Tree, Char('/')) => "Search: filter tensors by name as you type.",
        (Tree, Char('h')) => "Run the checkpoint health checks and show the report.",
        (Tree, Char('s')) => {
            "Show overall checkpoint stats: sizes, params, dtype mix, layers, experts."
        }
        (Tree, Char('t')) => "Copy the tree or a flat tensor list — text or JSON (opens a menu).",
        (Tree, Char('f')) => "Copy the selected row's file path.",
        (Tree, Char('n')) => "Copy the selected tensor's name.",
        (Tree, Char('r')) => "Repack this HDF5 checkpoint into a new file with another codec.",
        (Tree, Char('R')) => {
            "Rename tensors in place (safetensors): rewrites shard headers and the index."
        }
        (Tree, Char('q')) => "Quit the explorer.",
        // Detail screen.
        (Detail, Char(' ' | ':')) => "Open the command palette — search and run any command.",
        (Detail | Data, Char('m')) => "Show the tensor as a heatmap.",
        (Detail | Data, Char('v')) => "Show the tensor as a grid of numeric values.",
        (Detail, Char('h')) => "Compute and show the value histogram.",
        (Detail, Char('b' | 'B')) => "Set the histogram's bucket count.",
        (Detail, Char('s')) => "Compute exact whole-tensor statistics (min/max, mean, std, …).",
        (Detail | Data, Char('d')) => "Reinterpret the stored dtype (e.g. u4, i4, bf16, f32).",
        (Detail | Data, Char('r' | 'R')) => "Reshape the tensor's dimensions (row-major).",
        // Data view.
        (Data, Char(' ' | ':')) => "Open the command palette — search and run any command.",
        (Data, Char('e' | 'E')) => "Cycle the layout: overview → abs-max → edges → window.",
        (Data, Char('z' | 'Z')) => "Cycle zebra striping: rows → columns → off.",
        (Data, Char('b' | 'B')) => "Cycle the numeral base: dec → hex → oct → bin.",
        (Data, Char(']' | '[')) => "Step to the next / previous slice.",
        (Data, Up | Down | Left | Right) => {
            "Pan the view (Shift = one screenful, Ctrl = to the edge)."
        }
        // Rename editor — palette commands, keyed by their registry sentinel char
        // (the palette maps each to `KeyCode::Char(sentinel)`; see `RENAME_COMMANDS`).
        (Rename, Char(' ' | ':')) => "Open the command palette — search and run any command.",
        (Rename, Char('r' | '\u{12}')) => {
            "Apply the rename in place (asks for confirmation first)."
        }
        (Rename, Char('\r')) => "Move to the next field (past the last field, add a new rule).",
        (Rename, Char('\u{e}')) => "Add another source → new-name rule.",
        (Rename, Char('\u{4}')) => "Remove the focused rule.",
        (Rename, Char('y' | '\u{19}')) => "Copy the CLI command that reopens this rename editor.",
        (Rename, Char('a' | '\u{1}')) => {
            "Copy the `convert --map` command that applies this rename non-interactively."
        }
        (Rename, Char('s' | '\u{13}')) => "Copy the whole screen's text to the clipboard.",
        (Rename, Char('l' | '\u{c}')) => "Show the legend for the rename editor's symbols.",
        (Rename, Char('\u{1b}')) => "Go back to the previous view.",
        (Rename, Char('\u{3}')) => "Quit the explorer.",
        // Checkpoint stats (full-screen).
        (Stats, Char(' ' | ':')) => "Open the command palette — search and run any command.",
        (Stats, Up | Down) => "Scroll the report up / down one line.",
        (Stats, PageUp | PageDown) => "Scroll the report by one screenful.",
        (Stats, Char('f')) => "Fold / expand the per-shard on-disk breakdown.",
        (Stats, Char('r')) => "Copy the stats report as plain text.",
        (Stats, Char('q')) => "Quit the explorer.",
        // Common to every screen.
        (_, Char('l')) => "Show the legend for this screen's symbols and keys.",
        (_, Char('c')) => "Copy the whole screen's text to the clipboard.",
        (_, Char('y')) => "Copy the CLI command that reopens this exact screen.",
        (_, Backspace) => "Step back through view history.",
        (_, Char('\\')) => "Step forward through view history.",
        _ => return None,
    };
    Some(help)
}

/// Footer for the stats popup: a "✓ copied …" flash, or the key hints.
/// The full-screen stats view's footer hint chips — the same borderless,
/// clickable, hover-aware footer every other view has. `can_fold` shows the
/// per-shard `f` toggle only when there's a breakdown; `shards_expanded` picks
/// its label. Every command key in [`STATS_COMMANDS`] appears here (enforced by
/// `every_static_mode_footer_shows_its_command_keys`).
pub(crate) fn stats_hint_lines(
    can_fold: bool,
    shards_expanded: bool,
    width: u16,
) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    use KeyCode::Backspace;
    let plain = KeyModifiers::NONE;
    let mut items: Vec<(Vec<Seg>, &str)> = vec![(nav_updown(), "scroll"), (nav_pages(), "page")];
    if can_fold {
        items.push((
            letter("f"),
            if shards_expanded {
                "fold shards"
            } else {
                "expand shards"
            },
        ));
    }
    items.push((letter("r"), "copy report"));
    items.push((letter("c"), "copy screen"));
    items.push((letter("y"), "copy command"));
    items.push((letter("l"), "legend"));
    items.push((vec![Seg::Key("⌫", KeyEvent::new(Backspace, plain))], "back"));
    items.push((letter("q"), "quit"));
    wrap_hint_items(items, width)
}

/// Footer for the data views: offers the other representation (`m`/`v` switch
/// in place, no trip back to the detail screen) and mentions slice navigation
/// only when there is more than one slice to move between. Keys highlighted.
/// The footer hint items for a data view — shared by the renderer and the
/// height calculation so the two can't drift. Depends only on values known
/// before sampling (layout mode, slice count, whether the dtype is overridable,
/// the representation, and the zebra/base toggles).
fn view_footer_items(
    mode: SampleMode,
    slices: usize,
    overridable: bool,
    heatmap: bool,
    stripe: StripeMode,
    base: NumBase,
) -> Vec<(Vec<Seg>, &'static str)> {
    use KeyCode::{Down, End, Home, Left, Right, Up};
    let plain = KeyModifiers::NONE;
    let shift = KeyModifiers::SHIFT;
    // The other representation to switch to (heatmap ⇆ numeric values).
    let switch = if heatmap {
        (letter("v"), "numeric values")
    } else {
        (letter("m"), "heatmap")
    };
    let mut items: Vec<(Vec<Seg>, &str)> = vec![switch];
    let edges = matches!(mode, SampleMode::Edges { .. });
    let window = matches!(mode, SampleMode::Window { .. });
    // In the edges view the arrows rebalance first vs. last (Shift snaps to one
    // end); in the window view they pan the block (Shift a screenful, Ctrl to an
    // edge). Either way slice stepping moves to `[`/`]` so the arrows are free.
    if edges {
        items.push((
            vec![
                Seg::Key("←", KeyEvent::new(Left, plain)),
                Seg::Sep(" "),
                Seg::Key("→", KeyEvent::new(Right, plain)),
            ],
            "first/last cols",
        ));
        items.push((
            vec![
                Seg::Key("↑", KeyEvent::new(Up, plain)),
                Seg::Sep(" "),
                Seg::Key("↓", KeyEvent::new(Down, plain)),
            ],
            "first/last rows",
        ));
        items.push((vec![Seg::Sep("+Shift")], "one end"));
    }
    if window {
        items.push((vec![Seg::Sep("←↑↓→")], "pan"));
        items.push((vec![Seg::Sep("+Shift")], "page"));
        items.push((
            vec![
                Seg::Key("Home", KeyEvent::new(Home, plain)),
                Seg::Sep("/"),
                Seg::Key("End", KeyEvent::new(End, plain)),
            ],
            "col edge",
        ));
        items.push((nav_pages(), "row edge"));
    }
    if slices > 1 {
        if edges || window {
            items.push((
                vec![
                    Seg::Key("[", hint_key('[')),
                    Seg::Sep(" "),
                    Seg::Key("]", hint_key(']')),
                ],
                "slice",
            ));
        } else {
            items.push((
                vec![
                    Seg::Key("←", KeyEvent::new(Left, plain)),
                    Seg::Sep(" "),
                    Seg::Key("→", KeyEvent::new(Right, plain)),
                ],
                "step",
            ));
            items.push((
                vec![
                    Seg::Sep("Shift+"),
                    Seg::Key("←", KeyEvent::new(Left, shift)),
                    Seg::Sep(" "),
                    Seg::Key("→", KeyEvent::new(Right, shift)),
                ],
                "jump 5%",
            ));
        }
        items.push((letter("/"), "index or %"));
    }
    if overridable {
        items.push((letter("d"), "dtype"));
        items.push((letter("r"), "reshape"));
    }
    // Cycle the layout overview → abs-max → edges → window → overview; the label
    // names the layout `e` switches to next.
    items.push((
        letter("e"),
        match mode {
            SampleMode::Grid => "abs-max",
            SampleMode::GridMax => "edges",
            SampleMode::Edges { .. } => "window",
            SampleMode::Window { .. } => "overview",
        },
    ));
    // Cycle the zebra striping / numeral base (numeric grid only).
    if !heatmap {
        items.push((
            letter("z"),
            match stripe {
                StripeMode::Rows => "zebra: rows",
                StripeMode::Cols => "zebra: cols",
                StripeMode::Off => "zebra: off",
            },
        ));
        items.push((
            letter("b"),
            match base {
                NumBase::Decimal => "base: dec",
                NumBase::Hex => "base: hex",
                NumBase::Octal => "base: oct",
                NumBase::Binary => "base: bin",
            },
        ));
    }
    items.push((letter("c"), "copy screen"));
    items.push((letter("y"), "copy cmd"));
    items.push((letter("l"), "legend"));
    items.push((palette_keys(), "commands"));
    items.push((history_keys(), "back/fwd"));
    items
}

/// Physical lines the data view footer occupies at `width`: the blank spacer row
/// above it plus the wrapped hint line(s). Used to size the grid so the header
/// (tensor name + file) never scrolls off. Shares [`wrap_hint_items`] with
/// [`data_view_footer_wrapped_lines`] so the reservation can't drift from what's
/// drawn.
pub(crate) fn data_view_footer_lines(
    mode: SampleMode,
    slices: usize,
    overridable: bool,
    heatmap: bool,
    stripe: StripeMode,
    base: NumBase,
    width: usize,
) -> usize {
    let items = view_footer_items(mode, slices, overridable, heatmap, stripe, base);
    1 + wrap_hint_items(items, width as u16).0.len().max(1)
}

/// A hint line `key label · key label · …` as styled spans, keys highlighted —
/// the Ratatui port of [`hint_line`]. An empty key writes the label plain; an
/// empty label writes just the key.
pub(super) fn hint_spans(items: &[(&str, &str)]) -> Vec<Span<'static>> {
    let dim = Style::default().fg(palette::DIM);
    let mut spans: Vec<Span> = Vec::new();
    for (i, (key, label)) in items.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(" · ", dim));
        }
        if key.is_empty() {
            spans.push(Span::raw(label.to_string()));
        } else {
            spans.push(key_span(key.to_string()));
            if !label.is_empty() {
                spans.push(Span::raw(format!(" {label}")));
            }
        }
    }
    spans
}

/// The data-view footer as styled [`Line`]s + clickable chips — the shared
/// ` · `-separated `key label` chip format via [`wrap_hint_items`], identical to
/// every other view's footer (so the coloring / wrapping / hit-testing can't drift).
pub(crate) fn data_view_footer_wrapped_lines(
    mode: SampleMode,
    slices: usize,
    overridable: bool,
    heatmap: bool,
    stripe: StripeMode,
    base: NumBase,
    width: usize,
) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    wrap_hint_items(
        view_footer_items(mode, slices, overridable, heatmap, stripe, base),
        width as u16,
    )
}

/// The tree browser's key-hint line(s), word-wrapped to `width` on the
/// ` · `-separated `key label` chips (the long hint spills onto a second line).
pub(crate) fn tree_hint_lines(
    can_repack: bool,
    can_rename: bool,
    width: u16,
) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    use KeyCode::{Down, Enter, Tab, Up};
    let plain = KeyModifiers::NONE;
    let shift = KeyModifiers::SHIFT;
    // Each chip's key text is a list of segments; a `Seg::Key` glyph is clickable
    // (and synthesizes its key), a `Seg::Sep` (`/`, `Shift+`) is not. Both halves
    // of a dual chip are thus independently clickable.
    let mut items: Vec<(Vec<Seg>, &str)> = vec![
        (nav_updown(), "navigate"),
        (nav_leftright(), "parent/child"),
        (
            vec![
                Seg::Sep("Shift+"),
                Seg::Key("↑", KeyEvent::new(Up, shift)),
                Seg::Sep("/"),
                Seg::Key("↓", KeyEvent::new(Down, shift)),
            ],
            "sibling",
        ),
        (nav_pages(), "page"),
        (vec![Seg::Key("Enter", KeyEvent::new(Enter, plain))], "open"),
        (vec![Seg::Key("Tab", KeyEvent::new(Tab, plain))], "files"),
        (palette_keys(), "commands"),
        (
            vec![
                Seg::Key("e", hint_key('e')),
                Seg::Sep("/"),
                Seg::Key("c", hint_key('c')),
            ],
            "expand/collapse all",
        ),
        (letter("/"), "search"),
        (letter("l"), "legend"),
        (letter("h"), "health"),
        (letter("s"), "stats"),
        (letter("d"), "compare"),
        (letter("t"), "copy tree"),
        (letter("f"), "copy file"),
        (letter("n"), "copy name"),
        (letter("y"), "copy command"),
        (history_keys(), "back/fwd"),
    ];
    if can_repack {
        items.push((letter("r"), "repack"));
    }
    if can_rename {
        items.push((letter("R"), "rename"));
    }
    items.push((letter("q"), "quit"));
    wrap_hint_items(items, width)
}

/// The in-place rename editor's footer hint chips — the same borderless,
/// clickable, hover-aware footer every other view has, opening with the common
/// `Space / :` command-palette chip. The `^N`/`^D`/`^Y` chips synthesize their
/// Ctrl combos; the rest are plain keys the editor loop already handles.
pub(crate) fn rename_hint_lines(
    width: u16,
    applicable: bool,
) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    use KeyCode::{Char, Enter, Esc, Tab};
    let plain = KeyModifiers::NONE;
    let ctrl = KeyModifiers::CONTROL;
    // The apply chip's label reflects readiness (`^R` is blocked until clean).
    let apply_label = if applicable {
        "apply"
    } else {
        "apply (fix issues)"
    };
    let items: Vec<(Vec<Seg>, &str)> = vec![
        (palette_keys(), "commands"),
        (vec![Seg::Key("Tab", KeyEvent::new(Tab, plain))], "complete"),
        (nav_updown(), "fields"),
        (
            vec![Seg::Key("↵", KeyEvent::new(Enter, plain))],
            "next field",
        ),
        (nav_leftright(), "caret"),
        (
            vec![Seg::Key("^N", KeyEvent::new(Char('n'), ctrl))],
            "add rule",
        ),
        (
            vec![Seg::Key("^D", KeyEvent::new(Char('d'), ctrl))],
            "remove",
        ),
        (nav_pages(), "scroll"),
        (
            vec![Seg::Key("^R", KeyEvent::new(Char('r'), ctrl))],
            apply_label,
        ),
        // The universal commands — bare `c`/`l`/`y` type into a field here, so they're
        // the Ctrl keys `^S`/`^L`/`^Y`, mirroring the non-editing modes' `c`/`l`/`y`.
        // `^A` copies the apply (`convert --map`) command.
        (
            vec![Seg::Key("^L", KeyEvent::new(Char('l'), ctrl))],
            "legend",
        ),
        (
            vec![Seg::Key("^S", KeyEvent::new(Char('s'), ctrl))],
            "copy screen",
        ),
        (
            vec![Seg::Key("^Y", KeyEvent::new(Char('y'), ctrl))],
            "copy command",
        ),
        (
            vec![Seg::Key("^A", KeyEvent::new(Char('a'), ctrl))],
            "copy apply cmd",
        ),
        (vec![Seg::Key("Esc", KeyEvent::new(Esc, plain))], "back"),
    ];
    wrap_hint_items(items, width)
}

/// Lay a list of key-hint chips (`key label`, ` · `-separated) into styled
/// [`Line`]s wrapped to `width`, tracking each clickable [`Seg::Key`]'s
/// [`ChipHit`] position. Shared by the tree ([`tree_hint_lines`]) and file
/// ([`files_hint_lines`]) footers so their wrapping and hit-testing match.
pub(super) fn wrap_hint_items(
    items: Vec<(Vec<Seg>, &str)>,
    width: u16,
) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    let width = width as usize;
    let key_style = Style::default()
        .fg(palette::KEY)
        .add_modifier(Modifier::BOLD);
    let sep_style = Style::default().fg(palette::DIM);
    let mut lines: Vec<Line> = Vec::new();
    let mut chips: Vec<ChipHit> = Vec::new();
    let mut spans: Vec<Span> = Vec::new();
    let mut col = 0usize;
    for (segs, label) in items {
        let key_text: String = segs.iter().map(Seg::text).collect();
        let item_w = key_text.chars().count() + 1 + label.chars().count();
        let has_prev = !spans.is_empty();
        if has_prev && col + 3 + item_w > width {
            lines.push(Line::from(std::mem::take(&mut spans)));
            col = 0;
        }
        if !spans.is_empty() {
            spans.push(Span::styled(" · ", sep_style));
            col += 3;
        }
        // A single-action chip is clickable across its whole "key label"; a dual
        // chip (two keys sharing a label) keeps one region per glyph, since each
        // glyph is a different action and the label between them is ambiguous.
        let key_count = segs.iter().filter(|s| matches!(s, Seg::Key(..))).count();
        if key_count == 1 {
            // Exactly one `Seg::Key` by the count above, so the search finds it.
            #[allow(clippy::expect_used, clippy::unwrap_used)]
            let key = segs
                .iter()
                .find_map(|s| match s {
                    Seg::Key(_, k) => Some(*k),
                    Seg::Sep(_) => None,
                })
                .unwrap();
            chips.push(ChipHit {
                line: lines.len() as u16,
                col: col as u16,
                width: item_w as u16,
                key,
            });
        } else {
            let mut off = 0usize;
            for seg in &segs {
                let n = seg.text().chars().count();
                if let Seg::Key(_, key) = seg {
                    chips.push(ChipHit {
                        line: lines.len() as u16,
                        col: (col + off) as u16,
                        width: n as u16,
                        key: *key,
                    });
                }
                off += n;
            }
        }
        spans.push(Span::styled(key_text, key_style));
        spans.push(Span::raw(format!(" {label}")));
        col += item_w;
    }
    if !spans.is_empty() {
        lines.push(Line::from(spans));
    }
    (lines, chips)
}

/// The file browser's key-hint line(s), wrapped to `width` like
/// [`tree_hint_lines`] — the same `key label · …` chips and clickable
/// [`ChipHit`]s, for the file-view footer.
/// The footer tail every non-editing screen ends with: the palette, the legend, the two
/// copies, history, quit. Extracted because it is genuinely the same list on every screen
/// — the compare screen made it a third copy, and a screen that quietly lost `y` from its
/// footer while still binding it is exactly the drift this removes.
fn common_hint_tail() -> Vec<(Vec<Seg>, &'static str)> {
    vec![
        (palette_keys(), "commands"),
        (letter("l"), "legend"),
        (letter("c"), "copy screen"),
        (letter("y"), "copy command"),
        (history_keys(), "back/fwd"),
        (letter("q"), "quit"),
    ]
}

/// The compare screen's key hints. No `Tab` (it is reached from the tree's palette, and
/// Backspace goes back), and it advertises the CLI command for the value comparison this
/// screen deliberately doesn't do.
pub(crate) fn diff_hint_lines(width: u16) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    let mut items: Vec<(Vec<Seg>, &str)> = vec![(nav_updown(), "scroll"), (nav_pages(), "page")];
    items.extend(common_hint_tail());
    wrap_hint_items(items, width)
}

pub(crate) fn files_hint_lines(width: u16) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    use KeyCode::{Enter, Tab};
    let plain = KeyModifiers::NONE;
    let items: Vec<(Vec<Seg>, &str)> = vec![
        (nav_updown(), "navigate"),
        (nav_leftright(), "collapse/expand"),
        (nav_pages(), "page"),
        (
            vec![Seg::Key("Enter", KeyEvent::new(Enter, plain))],
            "open/preview",
        ),
        (
            vec![Seg::Key("Tab", KeyEvent::new(Tab, plain))],
            "tensor tree",
        ),
        (palette_keys(), "commands"),
        (letter("l"), "legend"),
        (letter("f"), "copy path"),
        (letter("c"), "copy screen"),
        (letter("y"), "copy command"),
        (history_keys(), "back/fwd"),
        (letter("q"), "quit"),
    ];
    wrap_hint_items(items, width)
}

/// The layout map's footer hints (`↑↓ select · ↵ in tree · …`), wrapped to
/// `width` like the tree's, with clickable [`ChipHit`]s.
pub(crate) fn layout_hint_lines(width: u16) -> (Vec<Line<'static>>, Vec<ChipHit>) {
    use KeyCode::{Enter, Tab};
    let plain = KeyModifiers::NONE;
    let mut items: Vec<(Vec<Seg>, &str)> = vec![
        (nav_updown(), "select"),
        (nav_pages(), "page"),
        (vec![Seg::Key("↵", KeyEvent::new(Enter, plain))], "in tree"),
        (
            vec![Seg::Key("Tab", KeyEvent::new(Tab, plain))],
            "tensor tree",
        ),
    ];
    items.extend(common_hint_tail());
    wrap_hint_items(items, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortcut_help_is_context_aware() {
        // The same key means different things on different screens.
        assert_eq!(
            shortcut_help(hint_key('h'), HelpCtx::Tree),
            Some("Run the checkpoint health checks and show the report."),
        );
        assert_eq!(
            shortcut_help(hint_key('h'), HelpCtx::Detail),
            Some("Compute and show the value histogram."),
        );
        // A common key resolves on any screen; an unknown key has no bubble.
        assert!(shortcut_help(hint_key('l'), HelpCtx::Data).is_some());
        assert_eq!(shortcut_help(hint_key('☺'), HelpCtx::Tree), None);
    }
}
