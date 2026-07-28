<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '../lib/api';
  import type { FileNode } from '../lib/types';
  import { humanSize, shardNote } from '../lib/format';
  import { filterToShard, openFile, selectedSource } from '../stores/view';
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

  /** Names this file's bytes have, when there's more than one. Hardlinked files share
      their bytes, so deleting this name frees nothing and the sizes down the column
      sum to more than the checkpoint occupies. 0 when the question doesn't arise. */
  function sharedNames(node: FileNode): number {
    return node.kind === 'file' && node.links > 1 ? node.links : 0;
  }

  /** Why this file's header wouldn't parse. The loudest fact about a row when it's set:
      the read carried on without the file, so its tensors are missing from the tree, the
      stats and the parameter count. */
  function readError(node: FileNode): string {
    return node.kind === 'file' ? (node.read_error ?? '') : '';
  }

  /** Whether this row is the file the tree's selected tensor lives in. Matched on the
      file name: the browser's paths are checkpoint-relative and a `source_path` is
      absolute, which is the same reason the terminal's lookup falls back to the name. */
  function isCurrent(node: FileNode, source: string | null): boolean {
    if (node.kind !== 'file' || !source) return false;
    return source.split(/[/\\]/).pop() === node.name;
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
        class:current={isCurrent(node, $selectedSource)}
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
        <!-- The cross-link: narrow the tree to this file's tensors, as `t` does in the
         terminal. Its own column so the ones after it don't move from row to row, and
         `stopPropagation` so it doesn't also open the layout map. -->
        <span class="link-col">
          {#if node.kind === 'file' && node.file_kind === 'Checkpoint'}
            <button
              class="chip"
              title="Show this file's tensors in the tree (a shard: filter)"
              on:click|stopPropagation={() => filterToShard(node.name)}>tensors</button
            >
          {/if}
        </span>
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
          {#if readError(node)}<span class="broken" title={readError(node)}
              >✗ unreadable — see check</span
            >{:else if node.kind === 'dir'}{node.files} {node.files === 1
              ? 'file'
              : 'files'}{node.hardlinked
              ? ` · ${node.hardlinked} hardlinked`
              : ''}
          {:else}{shardSuffix(node)}{#if unlisted(node)}<span class="extra"
                title="on disk but not listed in model.safetensors.index.json"
                >{shardSuffix(node) ? ' · ' : ''}✚ not in the index</span
              >{/if}{#if sharedNames(node)}<span
                title="hardlinked: one copy of the bytes under {sharedNames(
                  node,
                )} names, so this size is shared"
                >{shardSuffix(node) || unlisted(node) ? ' · ' : ''}⧉ {sharedNames(node)} names</span
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
  /* The file the tree's selected tensor lives in — the same selection colour a selected
     row gets elsewhere, since that is what this is. */
  .row.current {
    background: var(--bg-sel);
  }
  .link-col {
    flex: 0 0 9ch;
  }
  .chip {
    padding: 0 6px;
    border: 0;
    border-radius: 3px;
    background: var(--bg-elev);
    color: var(--fg-dim);
    font: inherit;
    font-size: 11px;
    cursor: pointer;
  }
  .chip:hover {
    background: var(--bg-hover);
    color: var(--accent);
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
     enough for the longest form — the shard note, the index mark and the link count. */
  .note {
    flex: 0 0 64ch;
    font-size: 12px;
  }
  /* The same vivid red the terminal marks an unindexed file with (palette::UNINDEXED)
     — one signal, whichever UI you're in. */
  .note .extra {
    color: var(--unindexed);
  }
  /* A file that didn't read is wrong, not merely unusual — the app's error colour. */
  .note .broken {
    color: var(--danger);
  }
  .err {
    color: var(--danger);
  }
</style>
