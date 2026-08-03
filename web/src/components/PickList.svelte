<script context="module" lang="ts">
  /**
   * One row: the value to set, and the count that tells a real namespace from a typo.
   *
   * In a module block because a Svelte 4 component cannot export a *type* from its instance script —
   * the parser stops at `export interface`.
   */
  export interface Option {
    value: string;
    /** `312 tensors`, or absent when a count says nothing useful. */
    detail?: string;
  }
</script>

<script lang="ts">
  /**
   * A searchable list of things a scope field can be set to — tensor names, or one side's namespaces.
   *
   * Typing into these fields is a guess: a tensor name is one of 79,732, and a subtree prefix that
   * selects nothing is a refusal rather than a comparison. So each field offers what is actually there,
   * fetched from the server, searched with the matcher the tree screen uses, and clicked into the box.
   *
   * One component for both, because the two differ only in what a row *says*: the names picker is a
   * multi-select (a comma list) and the subtree pickers set a single value, which is the `multi` flag.
   */
  import { onMount } from 'svelte';
  import TextField from './TextField.svelte';

  /** Distinguishes this list's search box from the others on the panel. */
  export let id: string;
  /** What the field currently holds, so a row can show as chosen. */
  export let chosen: string[] = [];
  /** Fetch a page of options for a query — the caller owns which endpoint that is. */
  export let load: (query: string) => Promise<{ options: Option[]; total: number }>;
  /** Told what was clicked; the caller decides whether that adds to a list or replaces a value. */
  export let onPick: (value: string) => void;
  export let disabled = false;
  export let placeholder = 'search…';
  /** What the list is of, for the line under the box: `names`, `namespaces`. */
  export let noun = 'options';
  /** Said when the source has nothing to offer — the reason is different for each field. */
  export let emptyNote = 'Nothing to choose from yet.';

  let query = '';
  let options: Option[] = [];
  let total = 0;
  let error = '';
  let busy = false;
  /** Which request is current — a slower earlier answer must not land on a later query. */
  let seq = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;

  // Fetch as soon as the list exists. Asking the *opener* to fetch cannot work: the component is
  // created by the very `{#if}` the opener flips, so at that moment there is nothing to call — the
  // three lists all opened empty, saying the source had nothing to offer.
  onMount(() => refresh(0));

  /** The search box changed — a template cannot carry the cast this needs, so it lives here. */
  function typed(e: Event) {
    query = (e.currentTarget as HTMLInputElement | HTMLTextAreaElement).value;
    refresh();
  }

  /**
   * Ask, debounced.
   *
   * `refresh(0)` for the first open and for a change to what the list *depends on* (the alignment, for
   * the names list); the default delay is for typing.
   */
  export function refresh(delay = 200) {
    clearTimeout(timer);
    busy = true;
    const mine = ++seq;
    timer = setTimeout(() => {
      void load(query)
        .then((r) => {
          if (mine !== seq) return;
          options = r.options;
          total = r.total;
          error = '';
        })
        .catch((e: unknown) => {
          if (mine !== seq) return;
          options = [];
          error = e instanceof Error ? e.message : String(e);
        })
        .finally(() => {
          if (mine === seq) busy = false;
        });
    }, delay);
  }
</script>

<div class="picklist">
  <TextField
    variant="dense"
    rows={0}
    {id}
    value={query}
    on:input={typed}
    {placeholder}
    spellcheck="false"
    readonly={disabled}
  />
  <p class="count dim" role="status">
    {#if error}
      <span class="error">{error}</span>
    {:else if busy && options.length === 0}
      looking…
    {:else if total === 0}
      {emptyNote}
    {:else}
      {options.length.toLocaleString()} shown{total > options.length
        ? ` of ${total.toLocaleString()}`
        : ''} {noun}
    {/if}
  </p>
  {#if options.length}
    <ul>
      {#each options as o (o.value)}
        <li>
          <button
            type="button"
            class:on={chosen.includes(o.value)}
            aria-pressed={chosen.includes(o.value)}
            {disabled}
            on:click={() => onPick(o.value)}
          >
            <span class="tick" aria-hidden="true">{chosen.includes(o.value) ? '✓' : '+'}</span>
            <span class="value mono">{o.value}</span>
            {#if o.detail}<span class="detail dim">{o.detail}</span>{/if}
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .picklist {
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin-top: 2px;
    padding: 6px;
    border-radius: 4px;
    background: var(--bg-elev);
  }
  .count {
    margin: 0;
    font-size: 11px;
    font-variant-numeric: tabular-nums;
  }
  .error {
    color: var(--danger);
  }
  /* Capped and scrolling: the answer is up to a few hundred rows, and the panel is not a page. */
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    max-height: 180px;
    overflow: auto;
  }
  button {
    display: flex;
    gap: 6px;
    width: 100%;
    font: inherit;
    font-size: 11.5px;
    color: var(--fg-dim);
    background: none;
    border: none;
    padding: 2px 4px;
    border-radius: 3px;
    cursor: pointer;
    text-align: left;
  }
  button:hover:not(:disabled) {
    background: var(--bg-hover);
    color: var(--fg);
  }
  button.on {
    color: var(--fg);
  }
  .tick {
    flex: 0 0 1ch;
    color: var(--accent);
  }
  .value {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    word-break: break-all;
  }
  /* The count that tells a namespace from a typo — never squeezed out by a long prefix. */
  .detail {
    flex: 0 0 auto;
    margin-left: auto;
    font-variant-numeric: tabular-nums;
  }
</style>
