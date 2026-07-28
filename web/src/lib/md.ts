/**
 * Markdown for the file preview: rendered as a document, with Shiki-highlighted code
 * blocks — plus the highlighted-source view behind a toggle.
 *
 * Checkpoint directories ship READMEs and model cards. A model card is prose with tables,
 * links and code, so *rendering* it is what makes it readable; a highlighted source dump
 * is the fallback, not the goal. Both are offered because the source is occasionally what
 * you want (checking exactly what a card claims, or reading raw frontmatter).
 *
 * **This is browser-only.** The TUI previews markdown as plain text; the terminal has no
 * equivalent library, and hand-writing one would produce a second implementation that
 * quietly disagreed with this one. See `preview_lines` in `src/explorer/mod.rs`.
 *
 * # Untrusted input
 *
 * A model card is a file from a checkpoint someone else built, and the server it is
 * served from has no access control. So the rendered HTML is **sanitized** with DOMPurify
 * before it reaches the DOM — real markdown uses raw HTML (every card on the Hub has
 * `<img>` badges), so refusing HTML outright would drop content, and passing it through
 * would hand a checkpoint author script execution in the viewer's browser.
 *
 * # Bundle size
 *
 * `web/dist` is committed *and* embedded in the release binary, so the whole markdown
 * path — marked, DOMPurify and Shiki — is imported dynamically and lands in chunks
 * fetched the first time someone opens a `.md`. Eager fine-grained Shiki imports alone
 * took the main bundle from 184 kB to 474 kB, for a pane most sessions never open.
 *
 * Fenced-code grammars are an explicit allow-list rather than a computed import path: a
 * template-literal import would put all ~200 of Shiki's languages into the build graph,
 * and everything in `dist` ends up inside the binary whether or not a browser fetches it.
 */

import type { HighlighterCore, LanguageRegistration } from 'shiki/core';
// Type-only, so it does not pull marked into the eager bundle.
import type { Tokens } from 'marked';

/** Parsed frontmatter: the `key: value` block many model cards open with. */
export type Frontmatter = [key: string, value: string][];

export interface Rendered {
  /** The leading `---` block, if any, for display as metadata rather than as content. */
  frontmatter: Frontmatter;
  /** Sanitized HTML, safe to inject. */
  html: string;
}

/**
 * Languages whose grammars ship, and the aliases a card is likely to write.
 *
 * Bounded on purpose (see the bundle note above). An unlisted language still renders as a
 * code block, just without colour — the failure is invisible rather than broken.
 */
type LangModule = { default: LanguageRegistration[] };

const LANGS: Record<string, () => Promise<LangModule>> = {
  python: () => import('@shikijs/langs/python'),
  bash: () => import('@shikijs/langs/bash'),
  json: () => import('@shikijs/langs/json'),
  yaml: () => import('@shikijs/langs/yaml'),
  markdown: () => import('@shikijs/langs/markdown'),
};

/** Aliases → a key in [`LANGS`]. */
const ALIASES: Record<string, string> = {
  py: 'python',
  python3: 'python',
  sh: 'bash',
  shell: 'bash',
  zsh: 'bash',
  console: 'bash',
  jsonc: 'json',
  yml: 'yaml',
  md: 'markdown',
};

/** Resolve a fence's info string to a shipped grammar, or `null` to leave it plain. */
function langKey(info: string): string | null {
  const name = info.trim().toLowerCase().split(/[\s,:]/)[0] ?? '';
  const key = ALIASES[name] ?? name;
  return key in LANGS ? key : null;
}

/** Built once and reused: loading grammars is the expensive part, not highlighting. */
const highlighters = new Map<string, Promise<HighlighterCore>>();

/**
 * A highlighter carrying exactly the grammars asked for.
 *
 * Keyed by the sorted language set so a card with three fences loads three grammars once,
 * rather than rebuilding an engine per code block.
 */
function highlighter(langs: string[]): Promise<HighlighterCore> {
  const wanted = [...new Set(langs)].sort();
  const key = wanted.join(',');
  let pending = highlighters.get(key);
  if (!pending) {
    pending = (async () => {
      const [{ createHighlighterCore }, { createJavaScriptRegexEngine }, dark, light, ...grammars] =
        await Promise.all([
          import('shiki/core'),
          import('shiki/engine/javascript'),
          import('@shikijs/themes/one-dark-pro'),
          import('@shikijs/themes/one-light'),
          ...wanted.map((l) => (LANGS[l] as () => Promise<LangModule>)()),
        ]);
      return createHighlighterCore({
        langs: grammars.flatMap((g) => g.default),
        themes: [dark.default, light.default],
        engine: createJavaScriptRegexEngine(),
      });
    })();
    highlighters.set(key, pending);
  }
  return pending;
}

/**
 * Shiki options shared by both views.
 *
 * Both themes are emitted as CSS variables (`--shiki-light` / `--shiki-dark`) rather than
 * baked colours, so the app's `data-theme` switch applies without re-highlighting. One
 * Dark / One Light are the pair the app's own palette derives from — `--ok`, `--warn` and
 * `--danger` are One Dark's green, yellow and red — so a code block sits in the UI rather
 * than on top of it.
 */
const THEMES = { light: 'one-light', dark: 'one-dark-pro' } as const;

/**
 * Highlight markdown source as source — the "view source" half of the preview.
 *
 * Safe to inject: Shiki tokenizes its input as text and escapes it, so a README
 * containing `<script>` yields an escaped code span, not a script tag.
 */
export async function highlightMarkdown(source: string): Promise<string> {
  const shiki = await highlighter(['markdown']);
  return shiki.codeToHtml(source, { lang: 'markdown', themes: THEMES, defaultColor: false });
}

/**
 * Split a leading `---` frontmatter block off the source.
 *
 * Deliberately not a YAML parser: this reads the top-level `key: value` lines and folds
 * `- item` continuations into the key above, which covers what model cards actually put
 * there (`license`, `pipeline_tag`, `tags`). Anything it can't read stays in the body, so
 * an unusual block is still visible rather than swallowed.
 */
export function splitFrontmatter(source: string): { frontmatter: Frontmatter; body: string } {
  const match = /^---\r?\n([\s\S]*?)\r?\n---[ \t]*(?:\r?\n|$)/.exec(source);
  if (!match) return { frontmatter: [], body: source };
  const frontmatter: Frontmatter = [];
  for (const raw of (match[1] ?? '').split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const item = /^-\s*(.*)$/.exec(line);
    const last = frontmatter[frontmatter.length - 1];
    if (item && last) {
      last[1] = last[1] ? `${last[1]}, ${item[1] ?? ''}` : (item[1] ?? '');
      continue;
    }
    const pair = /^([A-Za-z0-9_.-]+)\s*:\s*(.*)$/.exec(line);
    if (pair) frontmatter.push([pair[1] ?? '', (pair[2] ?? '').replace(/^["']|["']$/g, '')]);
  }
  // Only treat it as frontmatter if it read as key/value pairs; a document that opens
  // with a horizontal rule keeps its body.
  if (frontmatter.length === 0) return { frontmatter: [], body: source };
  return { frontmatter, body: source.slice(match[0].length) };
}

/**
 * Render markdown to sanitized HTML, with fenced code blocks highlighted.
 *
 * Two passes because Shiki is async and marked's renderer is not: the first walks the
 * token tree to collect and highlight every fence, the second renders with those results
 * substituted in.
 */
export async function renderMarkdown(source: string): Promise<Rendered> {
  const { frontmatter, body } = splitFrontmatter(source);
  const [{ Marked }, { default: DOMPurify }] = await Promise.all([
    import('marked'),
    import('dompurify'),
  ]);
  const purifier = DOMPurify as unknown as Purifier;

  const marked = new Marked({ gfm: true, breaks: false });
  // Pass 1: which grammars does this document need?
  const fences: { text: string; lang: string | null }[] = [];
  marked.use({
    async: true,
    walkTokens: (token) => {
      if (token.type === 'code') {
        const code = token as Tokens.Code;
        fences.push({ text: code.text, lang: langKey(code.lang ?? '') });
      }
    },
  });
  await marked.parse(body);

  // Pass 2: highlight them, then render with the results keyed by (lang, text).
  const wanted = fences.map((f) => f.lang).filter((l): l is string => l !== null);
  const done = new Map<string, string>();
  if (wanted.length > 0) {
    const shiki = await highlighter(wanted);
    for (const { text, lang } of fences) {
      if (lang !== null) {
        done.set(`${lang} ${text}`, shiki.codeToHtml(text, { lang, themes: THEMES, defaultColor: false }));
      }
    }
  }
  const renderer = new Marked({ gfm: true, breaks: false });
  renderer.use({
    renderer: {
      code({ text, lang }) {
        const key = langKey(lang ?? '');
        const hit = key === null ? undefined : done.get(`${key} ${text}`);
        // No grammar (or a fence that appeared only after highlighting) still renders as
        // a code block; `escape` keeps it text rather than markup.
        return hit ?? `<pre class="plain"><code>${escapeHtml(text)}</code></pre>`;
      },
    },
  });
  const raw = await renderer.parse(body);

  return { frontmatter, html: sanitize(purifier, raw) };
}

/** Escape text for a code block we render ourselves (no grammar available). */
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** The slice of DOMPurify's API used here. */
type Purifier = {
  sanitize: (dirty: string, cfg?: object) => string;
  addHook: (name: string, cb: (node: Element) => void) => void;
};

/** Hooks are global to the instance, so install them once. */
let hooked = false;

function installHooks(purify: Purifier): void {
  if (hooked) return;
  hooked = true;
  purify.addHook('afterSanitizeAttributes', (node) => {
    if (node.tagName !== 'A' || !node.hasAttribute('href')) return;
    node.setAttribute('target', '_blank');
    node.setAttribute('rel', 'noopener noreferrer');
  });
}

/**
 * Sanitize rendered markdown, and make its links safe to click.
 *
 * A card's links point off-site, so they open in a new tab with `rel="noopener
 * noreferrer"` — without `noopener` the opened page can reach back through
 * `window.opener`. Shiki's inline `style` attributes have to survive, which is why this
 * allows `style` rather than stripping it: they are the theme CSS variables, and
 * DOMPurify still filters the property values.
 */
function sanitize(purify: Purifier, html: string): string {
  installHooks(purify);
  return purify.sanitize(html, {
    ADD_ATTR: ['target', 'rel'],
    // `javascript:` and friends never survive this; `#` fragments and normal URLs do.
    ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto):|[^a-z]|[a-z+.-]+(?:[^a-z+.\-:]|$))/i,
    FORBID_TAGS: ['style', 'form', 'input', 'button'],
    FORBID_ATTR: ['srcset', 'formaction', 'ping'],
  });
}

/** Whether a file name is markdown, and so worth rendering. */
export function isMarkdown(name: string): boolean {
  return /\.(md|markdown|mdown|mkd)$/i.test(name);
}
