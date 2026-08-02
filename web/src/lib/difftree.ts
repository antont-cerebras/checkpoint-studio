// The aligned two-checkpoint tree, as the browser renders it. The server does the alignment
// (`checkpoint_studio_core::difftree`); this is the fold-aware flattening and the difference
// navigation on top of it.
//
// One tree, so folding and selection are shared between the two columns by construction — which is
// what "lockstep" means here. Nothing reconciles two scroll positions, because there is only one.

import type { TensorInfo } from './types';

/** How a row's two sides relate. */
export type DiffStatus = 'same' | 'changed' | 'only_old' | 'only_new';

/** What one side of a row holds; `null` on the side a row is missing from. */
export type DiffSide =
  | { kind: 'group'; tensor_count: number; params: number; total_size: number }
  /** `fold` is `256` when an unfused/fused alignment folded that many tensors onto this row — the fused
   * side has one of them, and the count is the answer to "did the conversion keep them all". `null` on
   * an ordinary one-to-one row, which is every row of an ordinary comparison. */
  | { kind: 'tensor'; info: TensorInfo; fold: number | null }
  | { kind: 'metadata'; name: string; value: string };

export interface AlignedNode {
  name: string;
  /** A tensor's full name (so a row can open its detail view); a stable key for groups. */
  path: string;
  old: DiffSide | null;
  new: DiffSide | null;
  status: { kind: DiffStatus };
  /** Differing tensors anywhere beneath this row — what a folded group reports. */
  differing: number;
  /** How many identical index-named sibling subtrees this row stands for: `1` ordinarily, `62` for a
   * family the server folded (`{0-61}`). The subtree below is one member — the template — so its sizes
   * describe one layer and this says how many there are. */
  members: number;
  children: AlignedNode[];
}

/**
 * What `POST /api/compare` answers with: the comparison's identity, and its two specs as the server
 * *resolved* them.
 *
 * The id is quoted on the follow-up `/api/difftree`. The specs are what the client checks the returned
 * tree against — resolved rather than as typed, because only the server performs that resolution
 * (`:/p` → `host:/p`, a glob → the directory), so comparing against what was typed would reject
 * correct answers.
 */
export interface ComparisonSet {
  id: number;
  left: string;
  right: string;
  recents: string[];
}

/** The headings for a comparison's two totals lines, worded by the server (see `DiffResponse`). */
export interface TotalsLabels {
  size: string;
  params: string;
}

/** One side of a comparison, as the server describes it. */
export interface DiffSideInfo {
  spec: string;
  root: string;
  tensor_count: number;
  /** Whether this side *is* the checkpoint the server is serving — the server's own answer, not a
   * string comparison of specs (which mis-answered for two spellings of one checkpoint). */
  served: boolean;
  /** This side's totals, so the view can head itself with the `size:` / `params:` lines the report
   * and the terminal show. It had neither, which made it the one view of a re-quantization that never
   * said the checkpoint got four times smaller.
   *
   * **Over the tensors in scope**: under a filter these cover the selected tensors, which is what the
   * rows below are — hence `totals_labels`, which says so. Summed over the deduped tensors, like the
   * report's; a test pins the two to each other. */
  params: number;
  bytes: number;
}

/**
 * How the comparison's rows fall out by status — the headline, counted by the server.
 *
 * Counted there, once, because two counters over two representations disagreed: the one-page report
 * and this view printed different totals for the same pair. `checkpoint_studio_core::difftree::Tally`
 * is the only thing that counts now, and a test pins it to the report's own sections.
 */
export interface DiffCounts {
  same: number;
  changed: number;
  only_old: number;
  only_new: number;
}

export interface DiffTally {
  tensors: DiffCounts;
  metadata: DiffCounts;
}

/**
 * Read a tally in the opposite direction.
 *
 * Matches and changes are symmetric, while one-sided rows are not: something added from A to B is
 * removed from B to A. Keep this next to [[swapSides]] so every summary and every row follows the
 * same direction when the browser swaps a comparison locally.
 */
export function swapTally(t: DiffTally): DiffTally {
  const swap = (c: DiffCounts): DiffCounts => ({
    same: c.same,
    changed: c.changed,
    only_old: c.only_new,
    only_new: c.only_old,
  });
  return { tensors: swap(t.tensors), metadata: swap(t.metadata) };
}

/**
 * The whole comparison, read the other way round.
 *
 * A pure transform of the server's answer, not a second request: which side is "old" is a question
 * about *reading* an alignment, and both checkpoints are already in it. Every part has to turn
 * together — the two side descriptions, the counts ([[swapTally]]) and every row ([[swapSides]]) — and
 * having one function do all three is what stops a future part from being forgotten, as the tally
 * once was: the rows read `+` where they had read `-` while the chips above them did not move.
 *
 * The *canonical* answer is kept as the server sent it and this is applied at the point of drawing, so
 * a flipped comparison is a way of looking at the same request rather than a different one.
 */
export function swapResponse(t: DiffTreeResponse): DiffTreeResponse {
  return {
    ...t,
    base: t.current,
    current: t.base,
    tally: swapTally(t.tally),
    rows: swapSides(t.rows),
  };
}

export interface DiffTreeResponse {
  base: DiffSideInfo;
  current: DiffSideInfo;
  tally: DiffTally;
  /** What a scope selected, as the CLI's `matched M of N`; `null` when nothing narrowed it. */
  matched: { selected: number; total: number } | null;
  /** What to call the two totals lines — `size (filtered subset)` when a filter narrowed them. */
  totals_labels: TotalsLabels;
  /** The differing rows in draw order, by path — precomputed server-side, over the rows below (so a
   * jump always lands on a row that exists in this tree, folded or not). */
  differences: string[];
  /** Whether these rows are every layer, or uniform families folded onto one row each. From the
   * server, because the checkbox and the tree on screen can be one request apart. */
  full: boolean;
  rows: AlignedNode[];
}

/**
 * Everything in one set of counters that is not a match.
 *
 * `Number(x) || 0` rather than a bare sum, because a *missing* counter used to poison the whole
 * comparison. A tab left open across a server upgrade read the new server's split tally
 * (`{tensors, metadata}`) with the old flat shape, so every counter came back `undefined` — and
 * `undefined + undefined` is `NaN`, which is not `> 0`, which made [[identicalNote]] declare two
 * checkpoints sharing no tensor name at all "structurally identical". A count that cannot be read is
 * zero here and caught by [[tallyIsReadable]] there; it is never silently the answer.
 */
export function differing(c: DiffCounts | undefined): number {
  if (!c) return 0;
  return count(c.changed) + count(c.only_old) + count(c.only_new);
}

/** One counter as a number, or 0 for anything that is not one. */
function count(n: number | undefined): number {
  return typeof n === 'number' && Number.isFinite(n) ? n : 0;
}

/**
 * Whether a tally is one this build understands.
 *
 * The shape is part of the API, and a browser tab outlives the server it was loaded from — this app's
 * own workflow restarts the server under open tabs. So the client checks rather than assumes: every
 * counter of both halves must be a real number. `false` means *this page is out of date*, which is a
 * thing to say, not a comparison to render.
 */
export function tallyIsReadable(t: DiffTally | null | undefined): boolean {
  const ok = (c: DiffCounts | undefined) =>
    !!c &&
    [c.same, c.changed, c.only_old, c.only_new].every(
      (n) => typeof n === 'number' && Number.isFinite(n),
    );
  return !!t && ok(t.tensors) && ok(t.metadata);
}

/** Everything that is not a match, of either kind — what "N differences" means. */
export function differingCount(t: DiffTally): number {
  return differing(t.tensors) + differing(t.metadata);
}

/**
 * Whether the two checkpoints share no row at all.
 *
 * Mirrors `Tally::disjoint`. Worth its own answer because it is a different situation from "many
 * differences", and the useful thing to say about it is one sentence rather than 187k rows: two
 * checkpoints with unrelated naming schemes align nothing, so every tensor of *both* is one-sided.
 */
export function isDisjoint(t: DiffTally): boolean {
  // Tensors alone: two checkpoints can share a `format` key and still have nothing to do with each
  // other, and "share no tensor names" is what the banner says.
  const x = t.tensors;
  return x.same === 0 && x.changed === 0 && x.only_old > 0 && x.only_new > 0;
}

/**
 * The report's own sentence for the tally:
 * `0 unchanged; 31,247 added, 1 removed, 4 changed, 2 metadata changes`.
 *
 * The same words, in the same order, with the same punctuation as `compare::verdict` — so the two views
 * of one comparison read as one. Commas rather than the ` · ` this header uses between its other facts:
 * the tally is a single phrase, and separating its parts the way the header separates unrelated facts
 * made it look like four of them.
 *
 * Thousands separators throughout — the count used to appear twice in one sentence, grouped in one
 * place and not the other (`31,255 differences · 1 of 31255`).
 */
export function tallyText(t: DiffTally): string {
  const parts: string[] = [];
  for (const [n, what] of [
    [t.tensors.only_new, 'added'],
    [t.tensors.only_old, 'removed'],
    [t.tensors.changed, 'changed'],
  ] as const) {
    if (n > 0) parts.push(`${n.toLocaleString()} ${what}`);
  }
  // Metadata gets its own phrase rather than being folded into "removed", which is how this view came
  // to say `3 removed` where the report said `1 removed, 2 metadata changes`. The totals always agreed;
  // the label did not.
  const meta = differing(t.metadata);
  if (meta > 0) {
    parts.push(`${meta.toLocaleString()} metadata change${meta === 1 ? '' : 's'}`);
  }
  // Empty when nothing differs: that case gets a banner of its own ([[identicalNote]]) rather than a
  // dim fragment at the end of a count line, so the phrase lives in exactly one place.
  if (parts.length === 0) return '';
  // `tensors.same`, not every matching leaf: `compare::verdict` counts unchanged *tensors*, and this is
  // that sentence. Metadata appears in it only as the "N metadata changes" clause.
  return `${t.tensors.same.toLocaleString()} unchanged; ${parts.join(', ')}`;
}

/** The banner for a comparison that found nothing, and what that does — and does not — claim. */
export interface IdenticalNote {
  headline: string;
  detail: string;
}

/**
 * The verdict when the two checkpoints match, stated as a banner with its meaning spelled out.
 *
 * Two reasons it is not just the words "structurally identical" in dim text. It is the one outcome
 * where the whole view is empty, so a quiet aside next to a count of zero reads as "nothing loaded"
 * rather than as an answer. And the phrase is easy to over-read: it means the *shapes of the two files*
 * agree, not that the weights do. This comparison never looks at a tensor's bytes — that is `--values`
 * on the CLI, which has a progress bar for it — so a reader who takes "identical" at face value would
 * conclude two differently-trained checkpoints are the same file.
 *
 * `null` when something differs, so the caller has nothing to decide.
 */
export function identicalNote(t: DiffTally): IdenticalNote | null {
  // An unreadable tally is not an identical one. This banner is the strongest claim the screen makes,
  // so it is the one that must never be reached by accident (see [[tallyIsReadable]]).
  if (!tallyIsReadable(t) || differingCount(t) > 0) return null;
  return {
    headline: 'Structurally identical',
    detail:
      'Every tensor name, dtype and shape matches, and so does the metadata. ' +
      'The numbers inside were not compared — for that, run the diff in a terminal with --values.',
  };
}

/**
 * What to say where the rows would be when there are none.
 *
 * With "differences only" on and two matching checkpoints, every row is filtered out and the pane goes
 * blank — which reads as a view that failed to load rather than as the answer "nothing differs". An
 * empty result still needs to say what it is.
 */
export function emptyRowsNote(rowCount: number, t: DiffTally): string {
  if (rowCount > 0) return '';
  return differingCount(t) === 0
    ? 'No differences.'
    : 'No rows match the current view — clear “Differences only” to see the whole tree.';
}

/** A visible row: the node, how deep it sits, and whether it is folded open. */
export interface DiffRow {
  node: AlignedNode;
  depth: number;
  expanded: boolean;
}

/**
 * The rows to draw, given which paths are expanded.
 *
 * Depth-first in the server's order, descending only into expanded groups — the same shape the
 * single-checkpoint tree uses, so the two views scroll and fold alike.
 */
export function flattenDiff(
  rows: AlignedNode[],
  expanded: Set<string>,
  differencesOnly = false,
  /**
   * Find rows whose name contains this, case-insensitively — the tree's equivalent of the report's
   * *Find in results*, and a different question from the scope: the scope decides what the server
   * *compares*, this decides what is on screen.
   *
   * A matching row brings its ancestors with it (they are the path to it) and its subtree with it (a
   * group named `layers.3` is a way of asking for what is in it). Matching is on the row's own name,
   * which is what the reader typed against — `qkv_proj` finds the leaf, `layers.3` finds the group.
   */
  find = '',
): DiffRow[] {
  const out: DiffRow[] = [];
  // An explicit stack, and one `push` per row.
  //
  // This was `out.push(...flattenDiff(children))`, which is a crash rather than an inefficiency:
  // spreading passes one *argument* per row, and past Chrome's ~65k argument limit the call throws
  // `RangeError: Maximum call stack size exceeded`. Since the load auto-expands every ancestor of
  // every difference, two unrelated checkpoints (117k differences) hit it every time — and the throw
  // landed mid-render, so the flush that would have taken the progress bar down never completed and
  // the view hung on a finished-looking spinner forever.
  //
  // Recursion depth was never the problem; the argument count was. The stack here is bounded by the
  // tree's depth, and nothing scales with the number of rows.
  const needle = find.trim().toLowerCase();
  // Which subtrees hold a match at all, so a search can skip the rest without walking it twice.
  const hit = needle === '' ? null : matching(rows, needle);
  const stack: { nodes: AlignedNode[]; at: number; depth: number; inMatch: boolean }[] = [
    { nodes: rows, at: 0, depth: 0, inMatch: false },
  ];
  while (stack.length > 0) {
    const frame = stack[stack.length - 1]!;
    if (frame.at >= frame.nodes.length) {
      stack.pop();
      continue;
    }
    const node = frame.nodes[frame.at++]!;
    const isGroup = node.children.length > 0;
    // A search shows the matches, the path down to them, and everything under one — and nothing else.
    const matched = frame.inMatch || node.name.toLowerCase().includes(needle);
    if (hit !== null && !matched && !hit.has(node.path)) continue;
    // "Differences only" hides what matches — a leaf that is `same`, and a group with nothing
    // differing anywhere beneath it. Skipping the group *and* its subtree is the point: descending
    // into a group known to contain no difference is the work this option exists to avoid.
    if (differencesOnly && node.differing === 0 && node.status.kind === 'same') continue;
    // A search unfolds its own way down: a match three groups deep is no use behind a closed one, and
    // a group that matched *is* the answer, so it opens too — every group drawn during a search is
    // either the path to a match or a match itself.
    const open = isGroup && (expanded.has(node.path) || hit !== null);
    out.push({ node, depth: frame.depth, expanded: open });
    if (open) {
      stack.push({ nodes: node.children, at: 0, depth: frame.depth + 1, inMatch: matched });
    }
  }
  return out;
}

/** The paths of the groups that contain a match somewhere beneath them. */
function matching(rows: AlignedNode[], needle: string): Set<string> {
  const holds = new Set<string>();
  const walk = (node: AlignedNode): boolean => {
    let found = node.name.toLowerCase().includes(needle);
    for (const child of node.children) {
      if (walk(child)) found = true;
    }
    if (found) holds.add(node.path);
    return found;
  };
  for (const node of rows) walk(node);
  return holds;
}

/**
 * How many differences may be unfolded automatically on load.
 *
 * Revealing every difference is the right default for a normal comparison — a change three groups
 * deep is no use folded. It is the wrong default for two checkpoints that share nothing, where
 * "every difference" is *every tensor of both* and the view arrives as a six-figure wall of rows with
 * one `+` per line and no way back. Past this many, the tree stays folded and each group reports its
 * own `(N differ)` count, which is the readable form of the same information; `n` still unfolds
 * whatever it lands in.
 */
export const REVEAL_LIMIT = 2000;

/** The paths to expand on load: every difference's ancestors, or none when there are too many. */
export function initialExpansion(rows: AlignedNode[], differences: number): Set<string> {
  return differences <= REVEAL_LIMIT ? expandToDifferences(rows) : new Set();
}

/** Every group's path — "expand all", as an explicit action rather than a default. */
export function allGroupPaths(rows: AlignedNode[]): Set<string> {
  const open = new Set<string>();
  const stack: AlignedNode[][] = [rows];
  while (stack.length > 0) {
    for (const n of stack.pop()!) {
      if (n.children.length > 0) {
        open.add(n.path);
        stack.push(n.children);
      }
    }
  }
  return open;
}

/**
 * The paths to expand so that every difference is visible.
 *
 * What "show me what changed" needs: a difference three groups deep is no use if all three are
 * folded. Only ancestors *of differences* are opened — expanding everything would bury the changes
 * in 31k unchanged rows, which is the problem the comparison exists to solve.
 */
export function expandToDifferences(rows: AlignedNode[]): Set<string> {
  const open = new Set<string>();
  // One mutable trail, pushed and popped, rather than `[...ancestors, n.path]` at every node: that
  // allocated two arrays per group, and this walks every node of both checkpoints (373k for the pair
  // that motivated this) on the critical path between "downloaded" and "on screen".
  const trail: string[] = [];
  const walk = (nodes: AlignedNode[]): void => {
    for (const n of nodes) {
      if (n.children.length > 0) {
        trail.push(n.path);
        // A group whose subtree contains a difference is on the path to one.
        if (n.differing > 0 || n.status.kind !== 'same') {
          for (const a of trail) open.add(a);
        }
        walk(n.children);
        trail.pop();
      } else if (n.status.kind !== 'same') {
        for (const a of trail) open.add(a);
      }
    }
  };
  walk(rows);
  return open;
}

/**
 * The next differing path after `from`, wrapping around; `null` when there are none.
 *
 * Wrapping rather than stopping at the end: with a handful of differences in a large checkpoint,
 * "next" that dead-ends makes you scroll back to the top by hand. `from` need not itself be a
 * difference — stepping from anywhere goes to the next one below it.
 */
export function nextDifference(
  differences: string[],
  from: string | null,
  direction: 1 | -1 = 1,
): string | null {
  if (differences.length === 0) return null;
  const at = from === null ? -1 : differences.indexOf(from);
  if (at === -1) {
    // Not on a difference: forwards starts at the first, backwards at the last.
    return direction === 1 ? differences[0]! : differences[differences.length - 1]!;
  }
  const n = differences.length;
  return differences[(at + direction + n) % n]!;
}

/** The ancestor paths of a row, so jumping to it can unfold the groups hiding it. */
export function ancestorsOf(rows: AlignedNode[], path: string): string[] {
  const found: string[] = [];
  const walk = (nodes: AlignedNode[], trail: string[]): boolean => {
    for (const n of nodes) {
      if (n.path === path && n.children.length === 0) {
        found.push(...trail);
        return true;
      }
      if (n.children.length > 0) {
        if (n.path === path) {
          found.push(...trail);
          return true;
        }
        if (walk(n.children, [...trail, n.path])) return true;
      }
    }
    return false;
  };
  walk(rows, []);
  return found;
}

/**
 * The same comparison the other way round.
 *
 * A pure transform, not a refetch: both checkpoints are already aligned, and which one is "old" is
 * only a question of which column a side is drawn in and which way `+`/`-` point. So flipping is
 * instant and needs nothing from the server. Row order is preserved deliberately — re-aligning
 * would reorder, and rows jumping under the cursor is a worse cost than column order.
 *
 * Mirrors `checkpoint_studio_core::difftree::swap`, so the terminal's `s` and this do the same
 * thing to the same model.
 */
export function swapSides(rows: AlignedNode[]): AlignedNode[] {
  const flip = (s: DiffStatus): DiffStatus =>
    s === 'only_new' ? 'only_old' : s === 'only_old' ? 'only_new' : s;
  return rows.map((r) => ({
    ...r,
    old: r.new,
    new: r.old,
    status: { kind: flip(r.status.kind) },
    children: swapSides(r.children),
  }));
}

/** Which pane a row was clicked in. */
export type Which = 'old' | 'new';

/** What clicking a row should do. */
export type ClickOutcome =
  /** Fold or unfold this group. */
  | { kind: 'toggle'; path: string }
  /** Open this tensor's detail view — only ever for the side that is actually served. */
  | { kind: 'open'; name: string }
  /** Nothing to do: a metadata row, a side with no row, or a tensor in the checkpoint this tab has
   * not loaded — the detail screen reads the served one, so there is nothing to show. */
  | { kind: 'none' };

/**
 * What a click on `row`'s `which` pane means.
 *
 * Pure, and in `lib/` rather than in the component, so the rule is covered by the same gate as the
 * rest of the model — and so the one decision that can show *wrong numbers* is not spelled out in a
 * template. The detail screen reads the served checkpoint, so a tensor on any other side cannot be
 * opened there.
 *
 * That case is simply not a click. It used to raise a note — "`X` lives in `Y`, which is not the
 * checkpoint this tab has loaded" with an *Open it* button — which is a paragraph, a state to clear
 * and a second way to switch checkpoints, all to explain why a click did nothing. The cell says it in
 * its tooltip instead.
 */
export function clickOutcome(
  node: AlignedNode,
  which: Which,
  sides: { base: DiffSideInfo; current: DiffSideInfo } | null,
): ClickOutcome {
  if (node.children.length > 0) return { kind: 'toggle', path: node.path };
  const side = which === 'old' ? node.old : node.new;
  if (side?.kind !== 'tensor') return { kind: 'none' };
  // A folded family of leaves stands for several tensors — `{0-61}.mlp.weight` is not the name of one,
  // so there is nothing to open. Turning family folding off gives every one of them its own row.
  if (node.members > 1) return { kind: 'none' };
  const info = which === 'old' ? sides?.base : sides?.current;
  return info?.served ? { kind: 'open', name: side.info.name } : { kind: 'none' };
}

/** The marker a row is drawn with — the same `+`/`-`/`~` the terminal and `diff` use. */
export function statusMark(status: DiffStatus): string {
  switch (status) {
    case 'only_new':
      return '+';
    case 'only_old':
      return '-';
    case 'changed':
      return '~';
    case 'same':
      return ' ';
  }
}

/** One side's signature, for the column: `F16 (4096, 4096)`, a metadata value, or a group's count. */
export function sideText(side: DiffSide | null): string {
  if (!side) return '';
  switch (side.kind) {
    case 'tensor':
      // `×256` after the signature, when an alignment folded several tensors onto this row: the shape
      // alone would read as an unexplained extra dimension against the fused side.
      return `${side.info.dtype} (${side.info.shape.join(', ')})${side.fold ? `  ×${side.fold}` : ''}`;
    case 'metadata':
      return side.value;
    case 'group':
      // A group that holds no tensors says nothing rather than "0 tensors". The `🔧 Metadata` group
      // is exactly that — it holds metadata entries — and "🔧 Metadata 0 tensors" read as a bug in
      // the tool rather than as a true statement about a group of a different kind.
      return side.tensor_count > 0
        ? `${side.tensor_count.toLocaleString()} tensor${side.tensor_count === 1 ? '' : 's'}`
        : '';
  }
}
