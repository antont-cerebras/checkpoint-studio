<script lang="ts">
  import { api } from '../lib/api';
  import { humanSize } from '../lib/format';
  import { highlightJson, type Token } from '../lib/jsonhl';
  import { renderMarkdown, highlightMarkdown, isMarkdown, type Rendered } from '../lib/md';
  import Spinner from './Spinner.svelte';

  export let path: string;
  export let name: string;

  let data: { text: string; truncated: boolean; size: number; cap?: number } | null = null;
  /* Highlighted runs, or null for a file that isn't JSON (or a truncated one, which no
     longer parses) — those keep the plain <pre>, exactly as in the TUI. */
  let tokens: Token[] | null;
  $: tokens = data && !data.truncated ? highlightJson(data.text) : null;
  /* A model card is prose with tables, links and code, so it is *rendered* — the
     highlighted source is behind the toggle for when you want to see the file itself.
     Both are async (marked, DOMPurify and Shiki load on demand), so these are promises
     the markup awaits, and a failure falls back to plain text rather than replacing the
     file with an error. */
  type MdView = 'rendered' | 'source';
  const MD_VIEWS: MdView[] = ['rendered', 'source'];
  let mdView: MdView = 'rendered';
  $: isMd = data !== null && isMarkdown(name);
  let doc: Promise<Rendered> | null;
  $: doc = isMd && data && mdView === 'rendered' ? renderMarkdown(data.text) : null;
  let mdSource: Promise<string> | null;
  $: mdSource = isMd && data && mdView === 'source' ? highlightMarkdown(data.text) : null;
  let err = '';
  let loading = true;

  $: void load(path);
  async function load(p: string) {
    loading = true;
    mdView = 'rendered';
    err = '';
    data = null;
    try {
      data = await api.file(p);
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }
</script>

<div class="preview">
  <div class="head">
    <span class="name">{name}</span>
    {#if data}
      <span class="dim"
        >· {humanSize(data.size)}{data.truncated && data.cap
          ? ` · truncated to ${humanSize(data.cap)}`
          : ''}</span
      >
    {/if}
    {#if isMd}
      <span class="toggle">
        {#each MD_VIEWS as v (v)}
          <button class:on={mdView === v} on:click={() => (mdView = v)}>{v}</button>
        {/each}
      </span>
    {/if}
  </div>
  {#if loading}
    <Spinner label="reading file…" />
  {:else if err}
    <p class="err">{err}</p>
  {:else if doc}
    {#await doc}
      <Spinner label="rendering…" />
    {:then rendered}
      <div class="md">
        {#if rendered.frontmatter.length > 0}
          <!-- The `---` block is metadata about the model, not part of the prose, so it
           reads as a field list rather than as a stray table at the top. -->
          <dl class="front">
            {#each rendered.frontmatter as [k, v] (k)}
              <dt>{k}</dt>
              <dd>{v}</dd>
            {/each}
          </dl>
        {/if}
        <!-- Sanitized by DOMPurify in lib/md.ts before it gets here; md.test.ts pins that
         scripts, event handlers and javascript: URLs do not survive. -->
        <!-- eslint-disable-next-line svelte/no-at-html-tags -->
        <div class="body">{@html rendered.html}</div>
      </div>
    {:catch}
      <pre>{data?.text ?? ''}</pre>
    {/await}
  {:else if mdSource}
    {#await mdSource then html}
      <!-- Shiki tokenizes its input as text and escapes it, so this is highlighted
       markup wrapped around escaped source, never markup from the file itself. -->
      <!-- eslint-disable-next-line svelte/no-at-html-tags -->
      <div class="md">{@html html}</div>
    {:catch}
      <pre>{data?.text ?? ''}</pre>
    {/await}
  {:else if tokens}
    <pre>{#each tokens as [text, cls], i (i)}{#if cls}<span class={cls}>{text}</span>{:else}{text}{/if}{/each}</pre>
  {:else if data}
    <pre>{data.text}</pre>
  {/if}
</div>

<style>
  .preview {
    height: 100%;
    display: flex;
    flex-direction: column;
  }
  .head {
    flex: 0 0 auto;
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 8px 14px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-panel);
  }
  .name {
    color: var(--accent);
  }
  pre {
    flex: 1 1 auto;
    margin: 0;
    padding: 12px 16px;
    overflow: auto;
    white-space: pre;
    tab-size: 2;
    font-size: 12px;
    line-height: 1.5;
  }
  /* Shiki emits both themes as variables (see lib/md.ts) so the app's theme switch
     applies without re-highlighting. `fallout` has no Shiki counterpart and takes the
     dark one. */
  .md :global(.shiki),
  .md :global(.shiki span) {
    color: var(--shiki-dark);
    background: transparent;
  }
  :global(:root[data-theme='light']) .md :global(.shiki),
  :global(:root[data-theme='light']) .md :global(.shiki span) {
    color: var(--shiki-light);
  }
  .md {
    flex: 1 1 auto;
    overflow: auto;
  }
  /* Direct child only: the source view's own <pre>, not the ones inside a rendered
     document. Markdown source is mostly prose, and a paragraph is one long line — without
     wrapping you read a model card by scrolling sideways. Code blocks in the *rendered*
     view still scroll, because wrapping a line of Python moves where it breaks. */
  .md > :global(pre) {
    margin: 0;
    padding: 12px 16px;
    font-size: 12px;
    line-height: 1.5;
    tab-size: 2;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
  }

  /* The rendered document. Prose wants a proportional face and a measure — the rest of
     the app is monospace tables, but a model card is something you read. */
  .body {
    max-width: 78ch;
    padding: 4px 16px 32px;
    font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
    font-size: 13.5px;
    line-height: 1.6;
  }
  .body :global(h1),
  .body :global(h2),
  .body :global(h3),
  .body :global(h4) {
    margin: 1.4em 0 0.5em;
    color: var(--accent);
    font-weight: 600;
    line-height: 1.3;
  }
  .body :global(h1) {
    font-size: 1.5em;
    padding-bottom: 0.2em;
    border-bottom: 1px solid var(--border);
  }
  .body :global(h2) {
    font-size: 1.25em;
  }
  .body :global(h3) {
    font-size: 1.1em;
  }
  .body :global(a) {
    color: var(--accent);
  }
  .body :global(p),
  .body :global(ul),
  .body :global(ol),
  .body :global(blockquote) {
    margin: 0.7em 0;
  }
  .body :global(ul),
  .body :global(ol) {
    padding-left: 1.4em;
  }
  .body :global(blockquote) {
    margin-left: 0;
    padding: 0.1em 0 0.1em 1em;
    border-left: 2px solid var(--border);
    color: var(--fg-dim);
  }
  /* Inline code and unhighlightable fences share the panel background, so a code block
     reads as a block whether or not a grammar shipped for it. */
  .body :global(code) {
    /* No horizontal padding: with it, `code`. renders as if there were a space before the
       period. The background alone separates it from the prose. */
    padding: 1px 0;
    border-radius: 3px;
    background: var(--bg-panel);
    font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.9em;
  }
  .body :global(pre) {
    margin: 0.9em 0;
    /* Its own padding: the source view's `pre` rule is a direct-child selector now, so a
       rendered code block no longer inherits one and sat flush against its border. */
    padding: 10px 14px;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: var(--bg-panel);
    overflow-x: auto;
  }
  .body :global(pre code) {
    padding: 0;
    background: none;
  }
  .body :global(table) {
    margin: 0.9em 0;
    border-collapse: collapse;
    font-size: 0.95em;
  }
  .body :global(th),
  .body :global(td) {
    padding: 4px 10px;
    border: 1px solid var(--border);
    text-align: left;
  }
  .body :global(th) {
    background: var(--bg-panel);
  }
  .body :global(img) {
    max-width: 100%;
  }
  .body :global(hr) {
    margin: 1.5em 0;
    border: 0;
    border-top: 1px solid var(--border);
  }

  /* Frontmatter: the model's own metadata, so it reads like the app's other field lists
     rather than like the document. */
  .front {
    display: grid;
    grid-template-columns: max-content 1fr;
    gap: 2px 12px;
    margin: 0;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-panel);
    font-size: 11.5px;
  }
  .front dt {
    color: var(--fg-dim);
  }
  .front dd {
    margin: 0;
  }

  .toggle {
    margin-left: auto;
    display: flex;
    gap: 2px;
  }
  .toggle button {
    padding: 1px 8px;
    border: 0;
    border-radius: 3px;
    background: transparent;
    color: var(--fg-dim);
    font: inherit;
    font-size: 11px;
    cursor: pointer;
  }
  .toggle button:hover {
    background: var(--bg-hover);
  }
  .toggle button.on {
    background: var(--bg-sel);
    color: var(--fg);
  }
  /* The same roles the TUI's json_styler paints: keys in the structural accent, strings
     green, numbers amber, colons dimmed behind their values. */
  pre :global(.k) {
    color: var(--accent);
    font-weight: 600;
  }
  pre :global(.s) {
    color: var(--ok);
  }
  pre :global(.n) {
    color: var(--dtype);
  }
  pre :global(.b) {
    color: var(--warn);
  }
  pre :global(.p) {
    color: var(--fg-dim);
  }
  .err {
    padding: 14px;
    color: var(--danger);
  }
</style>
