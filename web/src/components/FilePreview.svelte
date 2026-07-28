<script lang="ts">
  import { api } from '../lib/api';
  import { humanSize } from '../lib/format';
  import { highlightJson, type Token } from '../lib/jsonhl';
  import Spinner from './Spinner.svelte';

  export let path: string;
  export let name: string;

  let data: { text: string; truncated: boolean; size: number; cap?: number } | null = null;
  /* Highlighted runs, or null for a file that isn't JSON (or a truncated one, which no
     longer parses) — those keep the plain <pre>, exactly as in the TUI. */
  let tokens: Token[] | null;
  $: tokens = data && !data.truncated ? highlightJson(data.text) : null;
  let err = '';
  let loading = true;

  $: void load(path);
  async function load(p: string) {
    loading = true;
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
  </div>
  {#if loading}
    <Spinner label="reading file…" />
  {:else if err}
    <p class="err">{err}</p>
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
