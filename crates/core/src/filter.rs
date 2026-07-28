//! Shared tensor-name filtering. A filter is a set of glob patterns; any pattern
//! may be **negated** with a leading `!` to *exclude*. The same rules back
//! `diff --name` and the `--print-tree` / `--print-tensors` `--name` option, so a
//! pattern behaves identically wherever it's accepted.

use anyhow::{Context, Result};
use glob::Pattern;

/// Expand shell-style brace alternation `{a,b,c}` in a glob into its concrete
/// patterns — the cartesian product over multiple groups — so
/// `layers.{1,31,60}.*` becomes three globs. No braces ⇒ the input unchanged.
/// Nesting isn't supported; an unbalanced `{` is left literal (glob compilation
/// then reports it like any other bad pattern). Shared by the `name:` facet in
/// [`crate::tensorfilter`] so `{…}` works everywhere a name glob is accepted.
pub(crate) fn expand_braces(pattern: &str) -> Vec<String> {
    let Some(open) = pattern.find('{') else {
        return vec![pattern.to_string()];
    };
    let Some(rel) = pattern[open + 1..].find('}') else {
        return vec![pattern.to_string()]; // unbalanced — leave literal
    };
    let close = open + 1 + rel;
    let (prefix, body, suffix) = (
        &pattern[..open],
        &pattern[open + 1..close],
        &pattern[close + 1..],
    );
    let mut out = Vec::new();
    for alt in body.split(',') {
        for rest in expand_braces(suffix) {
            out.push(format!("{prefix}{alt}{rest}"));
        }
    }
    out
}

/// A name filter: a name passes if it matches some **include** glob (or there are
/// none) and matches **no exclude** glob. An empty filter matches everything.
#[derive(Default, Clone)]
pub struct NameFilter {
    pub include: Vec<Pattern>,
    pub exclude: Vec<Pattern>,
}

impl NameFilter {
    /// Parse repeated `--name` values: a leading `!` marks a pattern as an
    /// exclude ("everything except …"), any other value is an include. Globs use
    /// the standard `*` / `?` / `[…]` plus brace alternation `{a,b,c}` (e.g.
    /// `layers.{1,31,60}.*`); a bad glob is an error.
    pub fn parse(patterns: &[String]) -> Result<Self> {
        let mut filter = Self::default();
        for pattern in patterns {
            let (bucket, glob) = match pattern.strip_prefix('!') {
                Some(rest) => (&mut filter.exclude, rest),
                None => (&mut filter.include, pattern.as_str()),
            };
            // `{a,b,c}` expands to several globs, matched as alternatives (include:
            // any; exclude: any) — so one `--name` can select a few layers.
            for expanded in expand_braces(glob) {
                bucket.push(
                    Pattern::new(&expanded)
                        .with_context(|| format!("invalid --name glob {glob:?}"))?,
                );
            }
        }
        Ok(filter)
    }

    /// Whether the filter constrains anything (so callers can skip work / drop
    /// metadata when it doesn't).
    #[must_use]
    pub fn is_active(&self) -> bool {
        !self.include.is_empty() || !self.exclude.is_empty()
    }

    /// Whether `name` passes: it matches at least one include (or there are no
    /// includes) and matches none of the excludes.
    #[must_use]
    pub fn matches(&self, name: &str) -> bool {
        if !self.include.is_empty() && !self.include.iter().any(|p| p.matches(name)) {
            return false;
        }
        !self.exclude.iter().any(|p| p.matches(name))
    }
}

#[cfg(test)]
mod tests {
    use super::NameFilter;

    #[test]
    fn empty_filter_matches_everything() {
        let f = NameFilter::parse(&[]).unwrap();
        assert!(!f.is_active());
        assert!(f.matches("anything.at.all"));
    }

    #[test]
    fn include_globs_match_any() {
        let f = NameFilter::parse(&["*.mlp.*".into(), "*.norm.weight".into()]).unwrap();
        assert!(f.is_active());
        assert!(f.matches("model.layers.0.mlp.down_proj.weight"));
        assert!(f.matches("model.norm.weight"));
        assert!(!f.matches("model.embed_tokens.weight"));
    }

    #[test]
    fn bare_exclude_is_all_except() {
        let f = NameFilter::parse(&["!*.bias".into()]).unwrap();
        assert!(f.is_active());
        assert!(f.matches("model.layers.0.mlp.down_proj.weight")); // kept
        assert!(!f.matches("model.layers.0.mlp.down_proj.bias")); // excluded
    }

    #[test]
    fn brace_alternation_expands_to_alternatives() {
        // `{1,31,60}` selects exactly those layers (and nothing else).
        let f = NameFilter::parse(&["model.layers.{1,31,60}.*".into()]).unwrap();
        assert!(f.matches("model.layers.1.self_attn.q_proj.weight"));
        assert!(f.matches("model.layers.31.mlp.down_proj.weight"));
        assert!(f.matches("model.layers.60.input_layernorm.weight"));
        assert!(!f.matches("model.layers.0.mlp.down_proj.weight"));
        assert!(!f.matches("model.layers.6.mlp.down_proj.weight")); // not a literal-substring match
        // Multiple groups expand as a cartesian product.
        assert_eq!(super::expand_braces("a.{1,2}.{x,y}").len(), 4);
        // No braces ⇒ unchanged; unbalanced ⇒ left literal.
        assert_eq!(super::expand_braces("a.*.b"), vec!["a.*.b".to_string()]);
        assert_eq!(super::expand_braces("a.{1,2"), vec!["a.{1,2".to_string()]);
    }

    #[test]
    fn include_minus_exclude() {
        let f = NameFilter::parse(&["*.weight".into(), "!*.norm.weight".into()]).unwrap();
        assert!(f.matches("model.layers.0.mlp.down_proj.weight"));
        assert!(!f.matches("model.norm.weight")); // matches include but excluded
        assert!(!f.matches("model.layers.0.mlp.down_proj.bias")); // no include match
    }
}
