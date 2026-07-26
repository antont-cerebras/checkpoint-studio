//! Small, frontend-free enums describing the **data-view** presentation choices —
//! the numeric-grid layout, zebra striping, and numeral base — plus their CLI
//! parsers. Kept in core (no ratatui) so the kernel's data-view state
//! (`kernel::DataViewState`) can own them and they round-trip through `y` / JSON.

use crate::sample::ViewDtype;

/// The numeric grid / heatmap layout: a downsampled **overview** of the whole
/// tensor, the same overview reduced by **abs-max** (each cell = max |value| over
/// its block, so no element is sampled away), the **edges** (corners) at full
/// resolution, or a scrollable **window** (for inspecting padding). Cycled with `e`.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug, serde::Serialize, serde::Deserialize)]
pub enum DataLayout {
    Overview,
    OverviewMax,
    #[default]
    Edges,
    Window,
}

impl DataLayout {
    /// The next layout in the `e` cycle: Overview → `OverviewMax` → Edges → Window →
    /// Overview.
    #[must_use]
    pub fn next(self) -> Self {
        match self {
            Self::Overview => Self::OverviewMax,
            Self::OverviewMax => Self::Edges,
            Self::Edges => Self::Window,
            Self::Window => Self::Overview,
        }
    }
}

/// The numeric grid's zebra striping: a subtle alternating background down the
/// rows, down the columns, or none. Cycled with `z`.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug, serde::Serialize, serde::Deserialize)]
pub enum StripeMode {
    #[default]
    Rows,
    Cols,
    Off,
}

impl StripeMode {
    /// The next mode in the `z` cycle: rows → cols → off → rows.
    #[must_use]
    pub fn next(self) -> Self {
        match self {
            Self::Rows => Self::Cols,
            Self::Cols => Self::Off,
            Self::Off => Self::Rows,
        }
    }
}

/// Parse a CLI `--zebra` value (`rows`, `cols`, or `off`) into a [`StripeMode`].
pub fn parse_stripe_mode(s: &str) -> Result<StripeMode, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "rows" | "row" => Ok(StripeMode::Rows),
        "cols" | "col" | "columns" | "column" => Ok(StripeMode::Cols),
        "off" | "none" => Ok(StripeMode::Off),
        _ => Err(format!(
            "unknown zebra mode '{s}'; expected rows, cols, or off"
        )),
    }
}

/// The numeral base the numeric grid prints values in. `Decimal` is the normal
/// human-readable form (floats in scientific notation, integers as signed
/// decimals); the other bases show each element's raw stored bit pattern,
/// zero-padded to the dtype's width. Cycled with `b`.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug, serde::Serialize, serde::Deserialize)]
pub enum NumBase {
    #[default]
    Decimal,
    Hex,
    Octal,
    Binary,
}

impl NumBase {
    /// The next base in the `b` cycle: dec → hex → oct → bin → dec.
    #[must_use]
    pub fn next(self) -> Self {
        match self {
            Self::Decimal => Self::Hex,
            Self::Hex => Self::Octal,
            Self::Octal => Self::Binary,
            Self::Binary => Self::Decimal,
        }
    }

    /// Short label for the footer/command (`dec`, `hex`, `oct`, `bin`).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Decimal => "dec",
            Self::Hex => "hex",
            Self::Octal => "oct",
            Self::Binary => "bin",
        }
    }

    /// Number of digits needed to print `width` bits in this base (raw-bit
    /// bases only; `Decimal` returns 0 since it sizes cells differently).
    #[must_use]
    pub fn digits(self, width: u32) -> usize {
        match self {
            Self::Decimal => 0,
            Self::Hex => width.div_ceil(4) as usize,
            Self::Octal => width.div_ceil(3) as usize,
            Self::Binary => width as usize,
        }
    }

    /// Display width (chars, incl. a 1-col gap) of one numeric-grid cell under
    /// this base, for the given `view`/`dtype`. Decimal sizes to the actual
    /// value `range` (small ints pack tighter); the raw-bit bases use the
    /// dtype's fixed digit count. Both the sampler (how many columns to fetch)
    /// and the renderer call this, so they can't disagree on the width.
    #[must_use]
    pub fn cell_width(self, view: ViewDtype, dtype: &str, range: Option<(f64, f64)>) -> usize {
        match self {
            Self::Decimal => view.cell_width(dtype, range),
            Self::Hex | Self::Octal | Self::Binary => self.digits(view.bit_width(dtype)) + 1,
        }
    }
}

/// Parse a CLI `--base` value into a [`NumBase`].
pub fn parse_num_base(s: &str) -> Result<NumBase, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "dec" | "decimal" | "10" => Ok(NumBase::Decimal),
        "hex" | "hexadecimal" | "16" => Ok(NumBase::Hex),
        "oct" | "octal" | "8" => Ok(NumBase::Octal),
        "bin" | "binary" | "2" => Ok(NumBase::Binary),
        _ => Err(format!(
            "unknown base '{s}'; expected dec, hex, oct, or bin"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsers_and_cycles_round_trip() {
        assert_eq!(parse_stripe_mode("cols").unwrap(), StripeMode::Cols);
        assert!(parse_stripe_mode("nope").is_err());
        assert_eq!(parse_num_base("hex").unwrap(), NumBase::Hex);
        assert_eq!(NumBase::Decimal.next().next(), NumBase::Octal);
        assert_eq!(DataLayout::Overview.next(), DataLayout::OverviewMax);
        assert_eq!(DataLayout::OverviewMax.next(), DataLayout::Edges);
        assert_eq!(NumBase::Binary.label(), "bin");
    }
}
