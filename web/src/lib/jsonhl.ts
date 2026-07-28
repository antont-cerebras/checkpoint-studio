/**
 * JSON syntax highlighting for the file preview — the browser's half of what the TUI
 * does with `colored_json` (`src/ui/json.rs`).
 *
 * The TUI pretty-prints a JSON sidecar and colours it from the app palette: keys in the
 * structural accent, strings green, numbers amber, colons dimmed. The web preview dumped
 * the raw bytes into a `<pre>`, so `model.safetensors.index.json` — the file where colour
 * matters most, because it is thousands of near-identical lines — read as a grey wall.
 *
 * This tokenizes the *pretty-printed* text rather than walking the parsed value, so the
 * output is a flat span list a `<pre>` can render as-is. Walking the value would mean
 * re-implementing the layout too, and the layout is `JSON.stringify`'s job.
 */

/** A highlighted run: the text, and which palette class paints it (`''` = unstyled). */
export type Token = [text: string, cls: '' | 'k' | 's' | 'n' | 'b' | 'p'];

/**
 * Pretty-print `raw` and tokenize it, or `null` when it isn't a JSON object/array.
 *
 * Bare scalars return `null` on purpose: a lone string or number is not worth
 * reformatting, which is the same call `highlight_json` makes in the TUI.
 */
export function highlightJson(raw: string): Token[] | null {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  if (value === null || typeof value !== 'object') return null;
  return tokenize(JSON.stringify(value, null, 2));
}

/**
 * Split pretty-printed JSON into palette runs.
 *
 * A string is scanned with escapes honoured (`"a\\"b"` is one token, not two), because
 * stopping at the first unescaped-looking quote would shift every colour after it on any
 * line containing an escape — and tokenizer keys in a checkpoint index are full of dots
 * and, in some exports, quotes.
 */
function tokenize(pretty: string): Token[] {
  const out: Token[] = [];
  let plain = '';
  const flush = () => {
    if (plain) out.push([plain, '']);
    plain = '';
  };
  let i = 0;
  while (i < pretty.length) {
    const c = pretty[i] as string;
    if (c === '"') {
      const start = i;
      i++;
      while (i < pretty.length && pretty[i] !== '"') {
        i += pretty[i] === '\\' ? 2 : 1;
      }
      i = Math.min(i + 1, pretty.length);
      const text = pretty.slice(start, i);
      // A string followed by a colon is a key; anything else is a value. That lookahead
      // is the whole distinction, and it holds because we control the layout.
      const rest = pretty.slice(i);
      flush();
      out.push([text, /^\s*:/.test(rest) ? 'k' : 's']);
      continue;
    }
    if (c === ':') {
      flush();
      out.push([':', 'p']);
      i++;
      continue;
    }
    const num = /^-?\d+(\.\d+)?([eE][-+]?\d+)?/.exec(pretty.slice(i));
    if (num) {
      flush();
      out.push([num[0], 'n']);
      i += num[0].length;
      continue;
    }
    const word = /^(true|false|null)/.exec(pretty.slice(i));
    if (word) {
      flush();
      out.push([word[0], 'b']);
      i += word[0].length;
      continue;
    }
    plain += c;
    i++;
  }
  flush();
  return out;
}
