/**
 * What the app is doing while there is no checkpoint content to show.
 *
 * There used to be three of these, in three components, each with its own wording and its own
 * idea of what a wait looks like: `reading checkpoint structure` (the initial load),
 * `reading the checkpoint` (an open in flight) and `folding the tree` (the compact view). Three
 * screens for one situation — *waiting for this checkpoint* — and the differences between them
 * were accidents of where the code lived, not distinctions worth drawing.
 *
 * So there is one screen now, and the thing that varies is this value: which step, and what it
 * is working on. Naming the step is the point — "loading…" tells you nothing about whether the
 * server is reading sixteen shard headers or the browser is pulling 13 MB of JSON, and those
 * take very different amounts of time for very different reasons.
 */

import type { Progress } from './progress';

/** A distinguishable phase of getting a checkpoint on screen. */
export type LoadStep =
  /** The server is reading the checkpoint's shard headers, before it can answer anything. */
  | { kind: 'opening'; spec: string; progress: Progress | null }
  /** The tensor tree is downloading — the one wait with a real byte total. */
  | { kind: 'tree'; progress: Progress | null }
  /** Uniform layers and experts are being folded into families. */
  | { kind: 'folding'; progress: Progress | null };

/** What the inputs to [`currentStep`] are — the stores it reads, as plain values. */
export interface LoadInputs {
  /** Non-null while `POST /api/open` is in flight. */
  opening: Progress | null;
  /** What that open is reading, for the label. */
  openingSpec: string;
  /** Non-null while `/api/tree` is downloading. */
  tree: Progress | null;
  /** Whether the tensor tree has landed. */
  haveTree: boolean;
  /** Whether the tree load *failed*. A failure is not a wait: without this, "no tree yet"
   * reads as "still loading" and the error screen behind it is never reached. */
  treeError: boolean;
  /** Whether the compact (folded) view is the one showing. */
  compact: boolean;
  /** Whether the compact tree has landed. */
  haveCompact: boolean;
  /** Non-null while the compact tree is being fetched. */
  folding: Progress | null;
  /** A failed fold shows its own message, not a wait that never ends. */
  compactError: boolean;
}

/**
 * The step to show, or `null` when there is content to show instead.
 *
 * Ordered by what the user is waiting on *most*: an open in flight outranks everything, because
 * until it lands the tree on screen belongs to the checkpoint being replaced. Then the tree
 * itself. Then the fold, which only matters once a tree exists.
 */
export function currentStep(i: LoadInputs): LoadStep | null {
  if (i.opening) return { kind: 'opening', spec: i.openingSpec, progress: i.opening };
  // An error outranks every wait: something that failed is not something to wait for.
  if (i.treeError) return null;
  if (!i.haveTree) return { kind: 'tree', progress: i.tree };
  if (i.compact && !i.haveCompact && !i.compactError) {
    return { kind: 'folding', progress: i.folding };
  }
  return null;
}

/** The step, said plainly — one line naming the actor and the work. */
export function stepLabel(s: LoadStep): string {
  switch (s.kind) {
    case 'opening':
      return 'reading the checkpoint';
    case 'tree':
      return 'reading the tensor tree';
    case 'folding':
      return 'folding uniform layers into families';
  }
}

/**
 * The detail under the label: *who* is doing the work and *why* it takes as long as it does.
 *
 * Worth a second line because these three waits fail and drag for entirely different reasons —
 * a slow disk on the server, a slow link to the browser, or a big tally — and a wait you can
 * attribute is one you can act on.
 */
export function stepDetail(s: LoadStep): string {
  switch (s.kind) {
    case 'opening':
      return 'the server is reading every shard header';
    case 'tree':
      return 'downloading the tensor list from the server';
    case 'folding':
      return 'grouping tensors whose names differ only by an index';
  }
}

/** What the step applies to, when that isn't obvious: the path being opened. */
export function stepSubject(s: LoadStep): string {
  return s.kind === 'opening' ? s.spec : '';
}
