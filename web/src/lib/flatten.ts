// Client-side tree flattening — the server sends the hierarchy, the client owns
// fold state (which mirrors the Rust `tree::flatten_tree`, but expansion lives here).

import type { TreeNode } from './types';

export interface Row {
  id: string;
  node: TreeNode;
  depth: number;
  hasChildren: boolean;
}

/** A stable id for a node from its path (groups) or unique name (leaves). */
export function nodeId(node: TreeNode, parentId: string): string {
  if (node.kind === 'group') return parentId ? `${parentId}/${node.name}` : node.name;
  if (node.kind === 'tensor') return `t:${node.info.name}`;
  return `m:${node.info.name}`;
}

/**
 * The ids of the groups the *server* sent expanded — the initial fold state.
 *
 * Every `TreeNode.Group` carries an `expanded` flag that the Rust flattener honors, so
 * it is what the terminal UI opens with (root plus, for example, a `model` group whose
 * only sibling chain collapses into it). This client owns fold state after load, but it
 * has to *start* from the same place, or the same checkpoint shows a different first
 * screen in the two UIs — which is exactly what it used to do: the seed was just the
 * root, so `model` came up collapsed here and expanded in the terminal.
 */
export function expandedIds(tree: TreeNode[]): Set<string> {
  const ids = new Set<string>();
  const walk = (nodes: TreeNode[], parentId: string) => {
    for (const node of nodes) {
      if (node.kind !== 'group') continue;
      const id = nodeId(node, parentId);
      if (node.expanded) ids.add(id);
      walk(node.children, id);
    }
  };
  walk(tree, '');
  return ids;
}

/** Flatten the tree to visible rows, honoring `expanded` (a set of group ids). */
export function flatten(tree: TreeNode[], expanded: Set<string>): Row[] {
  const out: Row[] = [];
  const walk = (nodes: TreeNode[], depth: number, parentId: string) => {
    for (const node of nodes) {
      const id = nodeId(node, parentId);
      const hasChildren = node.kind === 'group' && node.children.length > 0;
      out.push({ id, node, depth, hasChildren });
      if (node.kind === 'group' && hasChildren && expanded.has(id)) {
        walk(node.children, depth + 1, id);
      }
    }
  };
  walk(tree, 0, '');
  return out;
}
