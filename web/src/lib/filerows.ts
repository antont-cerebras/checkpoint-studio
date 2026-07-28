// Client-side file-tree flattening — the server sends the directory hierarchy, the
// client owns fold state. The tensor tree's counterpart is `flatten.ts`.

import type { FileNode } from './types';

export interface FileRow {
  node: FileNode;
  depth: number;
}

/**
 * Flatten a directory tree to visible rows, honoring `expanded` (a set of node paths).
 *
 * `expanded` is a *parameter* rather than something the walker closes over, and that is
 * the point: this ran as a nested function inside the component, so the `$: rows = …`
 * block only mentioned the tree. Svelte tracks the variables a reactive block references
 * directly, so it never saw the fold set — clicking a folder mutated `expanded`, `rows`
 * was not recomputed, and no triangle in the file browser folded anything. Taking it as
 * an argument makes the dependency visible to the compiler, and testable here.
 */
export function flattenFiles(root: FileNode | null, expanded: Set<string>): FileRow[] {
  const out: FileRow[] = [];
  const walk = (node: FileNode, depth: number) => {
    out.push({ node, depth });
    if (node.kind === 'dir' && expanded.has(node.path)) {
      for (const c of node.children) walk(c, depth + 1);
    }
  };
  if (root) walk(root, 0);
  return out;
}

/** Toggle one directory's fold state, returning a new set (so assignment is reactive). */
export function toggleDir(expanded: Set<string>, path: string): Set<string> {
  const next = new Set(expanded);
  if (!next.delete(path)) next.add(path);
  return next;
}
