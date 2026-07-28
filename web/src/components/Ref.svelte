<!--
  A universal link to a checkpoint entity by name. Used everywhere a tensor or
  shard name is shown (stats, health, layout, …): a name that resolves to a known
  tensor opens its detail view; a known shard opens its byte-layout map. Unknown
  names render as plain text, so callers can pass anything without guarding.
-->
<script lang="ts">
  import { tensorNames, shardNames } from '../stores/server';
  import { openDetail, navigate } from '../stores/view';

  export let name: string;
  /** How to resolve `name`. 'auto' checks the tensor set first, then shards. Pass
   * 'tensor'/'file' to disambiguate when a name could match either. */
  export let kind: 'auto' | 'tensor' | 'file' = 'auto';
  let extra = '';
  export { extra as class };

  type Target = 'tensor' | 'file' | null;
  $: target = resolve(name, kind, $tensorNames, $shardNames);

  function resolve(n: string, k: typeof kind, tn: Set<string>, sn: Set<string>): Target {
    if (k === 'tensor') return tn.has(n) ? 'tensor' : null;
    if (k === 'file') return sn.has(n) ? 'file' : null;
    if (tn.has(n)) return 'tensor';
    if (sn.has(n)) return 'file';
    return null;
  }

  function go() {
    if (target === 'tensor') openDetail(name);
    else if (target === 'file') navigate({ kind: 'layout', file: name });
  }
</script>

{#if target}
  <button
    type="button"
    class="ref {extra}"
    on:click|stopPropagation={go}
    title={target === 'tensor' ? 'Open tensor detail' : 'Open byte layout'}
  >{name}</button>
{:else}
  <span class={extra}>{name}</span>
{/if}

<style>
  .ref {
    display: inline;
    margin: 0;
    padding: 0;
    border: none;
    background: none;
    font: inherit;
    color: var(--accent);
    text-align: left;
    cursor: pointer;
    border-radius: 2px;
  }
  .ref:hover {
    text-decoration: underline;
  }
  .ref:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 1px;
  }
</style>
