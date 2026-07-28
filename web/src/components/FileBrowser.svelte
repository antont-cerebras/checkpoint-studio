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
    return node.kind === 'file' && node.shard ? shardNote(node.shard) : '';
  }

  /** This file's share of the largest file, for the proportional bar; 0 for a
      directory, whose aggregate isn't a size to compare against its children. */
  function sizeShare(node: FileNode): number {
    return node.kind === 'file' ? node.size_share : 0;
  }

  /** True for a checkpoint file no index declares — a loader following only the index
      will not read it. Only the exception is marked: sixteen "in the index" notes
      would bury the one row that isn't. */
  function unlisted(node: FileNode): boolean {
    return node.kind === 'file' && node.index === 'unlisted';
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
        <span class="icon" title={node.kind === 'dir' ? 'Directory' : node.file_kind}
          >{node.kind === 'dir' ? '📁' : fileIcon(node.file_kind)}</span
        >
        <span class="name">{node.name}</span>
        <!-- The size and the bar are fixed-width columns, so they line up down the
         listing however long the names are — only the name absorbs the slack. -->
        <span class="size dim">{humanSize(node.size)}</span>
        <!-- Files only: a directory's size is its children's total, so a rail beside
         it would invite comparing it with them. The terminal leaves the column blank
         for a directory too. -->
        <span class="bar" class:rail={node.kind === 'file'} aria-hidden="true">
          {#if sizeShare(node) > 0}
            <i style="width:{(sizeShare(node) * 100).toFixed(2)}%"></i>
          {/if}
        </span>
        <span class="note dim">
          {#if node.kind === 'dir'}{node.files} {node.files === 1 ? 'file' : 'files'}
          {:else}{shardSuffix(node)}{#if unlisted(node)}<span class="extra"
                title="on disk but not listed in model.safetensors.index.json"
                >{shardSuffix(node) ? ' · ' : ''}✚ not in the index</span
              >{/if}{/if}
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
  /* The name is the only elastic column, so everything after it lines up down the
     listing — including across depths, since the row's indent eats into the name
     rather than shifting the columns. Mirrors the TUI's shared size column. */
  .name {
    flex: 1 1 auto;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .row.dir .name {
    color: var(--group);
  }
  .size {
    flex: 0 0 76px;
    text-align: right;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }
  /* Each file's size against the largest file in the tree (`size_share`, served so
     both UIs draw the same bar). The empty rail stays visible on a small file: that
     it has almost nothing filled in IS the reading. */
  .bar {
    flex: 0 0 84px;
    height: 4px;
    border-radius: 2px;
    overflow: hidden;
  }
  .bar.rail {
    background: var(--border);
  }
  .bar i {
    display: block;
    height: 100%;
    background: var(--accent);
  }
  /* Fixed, though it's the last column: a note whose width varied would change how
     much slack the name gets, and the columns before it would wander per row. Wide
     enough for the longest form — the shard note plus the index mark. */
  .note {
    flex: 0 0 50ch;
    font-size: 12px;
  }
  /* The same vivid red the terminal marks an unindexed file with (palette::UNINDEXED)
     — one signal, whichever UI you're in. */
  .note .extra {
    color: var(--unindexed);
  }
  .err {
    color: var(--danger);
  }
</style>
