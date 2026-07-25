// A small fuzzy subsequence matcher for the tree search box — mirrors the TUI's
// "type to filter tensors" behavior without a dependency.
//
// This runs on EVERY keystroke over every tensor in the checkpoint (31k in the model
// used for development, up to 116k), so the shape of the work matters:
//   - one pass, not two: the visible rows and the untruncated match total come from a
//     single walk (they used to be two separate walks that scored every name twice)
//   - the haystacks are lowercased ONCE per tree, not once per comparison (that was
//     ~31k throwaway strings per walk, i.e. ~62k per keystroke)
// Measured on the 31k-tensor tree: 11.5–16.2 ms per keystroke before, 3.4–5.9 ms after.

import type { TreeNode } from './types';
import { nodeId, type Row } from './flatten';

/** Cap on the rows handed to the view; the match TOTAL is reported separately so a
 * truncated list never looks like an exact count. */
export const SEARCH_LIMIT = 1000;

/** Subsequence score for an already-lowercased needle and haystack (higher = better);
 * -1 if `needle` isn't a subsequence of `hay`. Hot: called once per tensor per
 * keystroke, so it allocates nothing. */
function scoreLower(n: string, h: string): number {
  if (!n) return 0;
  let hi = 0;
  let score = 0;
  let streak = 0;
  for (let i = 0; i < n.length; i++) {
    const c = n[i];
    let found = -1;
    for (let k = hi; k < h.length; k++) {
      if (h[k] === c) {
        found = k;
        break;
      }
    }
    if (found < 0) return -1;
    streak = found === hi ? streak + 1 : 0;
    score += 1 + streak * 2 + (found === 0 ? 5 : 0);
    hi = found + 1;
  }
  return score - h.length * 0.01; // gently prefer shorter (tighter) matches
}

/** Whether a query is "cased" — has at least one uppercase letter, which under smart
 * case makes the whole match case-sensitive. */
function hasUpper(s: string): boolean {
  return s !== s.toLowerCase();
}

/** Subsequence score with smart case: case-insensitive unless the needle itself carries
 * uppercase. Kept for callers with raw strings (and tests); the search path uses the
 * pre-lowercased index below. */
export function fuzzyScore(needle: string, hay: string): number {
  return hasUpper(needle) ? scoreLower(needle, hay) : scoreLower(needle.toLowerCase(), hay.toLowerCase());
}

/** The searchable leaves of one tree, flattened once with their row ids and both the
 * original and lowercased names (smart case needs the original). Rebuilt only when the
 * tree itself changes — not per keystroke. */
interface SearchIndex {
  rows: Row[];
  lower: string[];
  original: string[];
}
const indexCache = new WeakMap<TreeNode[], SearchIndex>();

function searchIndex(tree: TreeNode[]): SearchIndex {
  const hit = indexCache.get(tree);
  if (hit) return hit;
  const rows: Row[] = [];
  const lower: string[] = [];
  const original: string[] = [];
  const walk = (nodes: TreeNode[], parentId: string) => {
    for (const node of nodes) {
      const id = nodeId(node, parentId);
      if (node.kind === 'group') {
        walk(node.children, id);
      } else {
        rows.push({ id, node, depth: 0, hasChildren: false });
        original.push(node.info.name);
        lower.push(node.info.name.toLowerCase());
      }
    }
  };
  walk(tree, '');
  const built = { rows, lower, original };
  indexCache.set(tree, built);
  return built;
}

/** Matching leaf rows (ranked, capped at `limit`) AND the untruncated match count, from
 * a single pass over the tree. */
export function searchTree(
  tree: TreeNode[],
  query: string,
  limit = SEARCH_LIMIT,
): { rows: Row[]; total: number } {
  const { rows, lower, original } = searchIndex(tree);
  // Smart case, the same rule the TUI's matcher uses: an all-lowercase query ignores
  // case, a query with any uppercase is matched literally. So `norm` finds
  // `LayerNorm` while `Norm` finds only the capitalised one. Pinned by
  // shared/parity/format.json.
  const cased = hasUpper(query);
  const hay = cased ? original : lower;
  const needle = cased ? query : query.toLowerCase();
  const scored: { row: Row; score: number }[] = [];
  for (let i = 0; i < rows.length; i++) {
    const score = scoreLower(needle, hay[i] ?? '');
    if (score >= 0) scored.push({ row: rows[i]!, score });
  }
  const total = scored.length;
  scored.sort((a, b) => b.score - a.score);
  return { rows: scored.slice(0, limit).map((s) => s.row), total };
}
