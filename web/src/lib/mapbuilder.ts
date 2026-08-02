/** One ordered `--map` rewrite, before it is compiled as a regular expression. */
export interface MappingRule {
  pattern: string;
  replacement: string;
}

export interface ParsedMappings {
  rules: MappingRule[];
  /** True when switching to the builder would discard a comment or an incomplete line. */
  rawOnly: boolean;
}

/**
 * Read the CLI's `PATTERN=>REPLACEMENT` form without trying to compile its regexes.
 *
 * Splitting on the first separator mirrors Rust's `NameMap::parse_rules`. Blank lines are harmless;
 * comments and incomplete lines require the raw editor because the two-column builder cannot preserve
 * them honestly.
 */
export function parseMappingRules(text: string): ParsedMappings {
  const rules: MappingRule[] = [];
  let rawOnly = false;
  for (const source of text.split('\n')) {
    const line = source.trim();
    if (!line) continue;
    if (line.startsWith('#')) {
      rawOnly = true;
      continue;
    }
    const separator = line.indexOf('=>');
    if (separator < 0) {
      rawOnly = true;
      continue;
    }
    rules.push({
      pattern: line.slice(0, separator).trim(),
      replacement: line.slice(separator + 2).trim(),
    });
  }
  return { rules, rawOnly };
}

/** Render builder rows back to the exact line-oriented form accepted by the server. */
export function serializeMappingRules(rules: MappingRule[]): string {
  return rules
    .filter((rule) => rule.pattern.trim() || rule.replacement.trim())
    .map((rule) => `${rule.pattern.trim()}=>${rule.replacement.trim()}`)
    .join('\n');
}

