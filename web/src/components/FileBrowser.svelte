<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '../lib/api';
  import type { FileNode } from '../lib/types';
  import { humanSize, shardNote } from '../lib/format';
  import { openFile } from '../stores/view';
  import { flattenFiles, toggleDir } from '../lib/filerows';

  let root: FileNode | null = null;
  let err = '';
  let expanded = new Set<string>();

  onMount(async () => {
    try {
      root = await api.files();
      if (root) expanded = toggleDir(expanded, root.path);
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
    }
  });

  // Both arguments are named here so the compiler sees the fold set as a dependency.
  $: rows = flattenFiles(root, expanded);

  /** What the model reads out of this shard: sixteen equal-sized shards are otherwise
      sixteen indistinguishable rows. Empty for a sidecar, and for a listing nobody
      attributed (a remote browse root that isn't the open checkpoint). Narrowed here
      rather than in the markup, where the compiler's narrowing doesn't reach. */
  function shardSuffix(node: FileNode): string {
    return node.kind === 'file' && node.shard ? ` · ${shardNote(node.shard)}` : '';
  }

  function activate(node: FileNode) {
    if (node.kind === 'dir') {
      expanded = toggleDir(expanded, node.path);
    } else {
      openFile(node.path, node.name, node.file_kind);
    }
  }
</script>

<div class="files">
  {#if err}
    <p class="err">{err}</p>
  {:else if !root}
    <p class="dim">loading…</p>
  {:else}
    {#each rows as { node, depth } (node.path + node.name)}
      <div
        class="row {node.kind}"
        style="padding-left:{8 + depth * 16}px"
        role="button"
        tabindex="-1"
        on:click={() => activate(node)}
        on:keydown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            activate(node);
          }
        }}
      >
        <span class="caret">{node.kind === 'dir' ? (expanded.has(node.path) ? '▾' : '▸') : ''}</span>
        <span class="icon">{node.kind === 'dir' ? '📁' : fileIcon(node.file_kind)}</span>
        <span class="name">{node.name}</span>
        <span class="meta dim">
          {#if node.kind === 'dir'}{node.files} files · {humanSize(node.size)}
          {:else}{node.file_kind} · {humanSize(node.size)}{shardSuffix(node)}{/if}
        </span>
      </div>
    {/each}
  {/if}
</div>

<script context="module" lang="ts">
  function fileIcon(kind: string): string {
    switch (kind) {
      case 'Checkpoint':
        return '🧊';
      case 'Json':
        return '📋';
      case 'Text':
        return '📄';
      default:
        return '·';
    }
  }
</script>

<style>
  .files {
    height: 100%;
    overflow: auto;
    padding: 6px 0;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 6px;
    height: 24px;
    line-height: 24px;
    padding-right: 12px;
    white-space: nowrap;
    cursor: pointer;
  }
  .row:hover {
    background: var(--bg-hover);
  }
  .caret {
    flex: 0 0 12px;
    color: var(--fg-dim);
    text-align: center;
  }
  .icon {
    flex: 0 0 auto;
  }
  .name {
    flex: 0 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .row.dir .name {
    color: var(--group);
  }
  .meta {
    flex: 1 1 auto;
    text-align: right;
    font-size: 12px;
  }
  .err {
    color: var(--danger);
  }
</style>
