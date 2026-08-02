// How one row of the diff report reads: the parts of a tensor name, which dimensions actually changed,
// and what kind of change the row is.
//
// The report used to render each row as one string — `[F16, (3072, 1536)] → [U16, (256, 3072, 1540)]` —
// which puts the work of finding the difference on the reader, for every row of a list that can be
// thousands long. The three functions here are what let the markup point at it instead: the leaf name
// apart from its path, the dimensions that moved apart from the ones that did not, and one heading per
// kind of change so a re-quantization's 747 rows read as "747 of them are the same change".
//
// Pure, so they are tested rather than eyeballed.

import type { TensorChange, TensorSig } from './types';

/** A tensor name split for display: the dotted path, and the leaf that says what the tensor *is*. */
export interface NameParts {
  /** Everything up to and including the last `.`, or `''` for a top-level name. */
  path: string;
  /** The last segment — `weight`, `qscale`, `codebook`. */
  leaf: string;
}

/**
 * Split a name into its path and its leaf.
 *
 * Worth separating because in a checkpoint the leaf is the *kind* of tensor and the path is where it
 * lives, and a report is read leaf-first: with 62 layers × 6 names, the eye is looking for `qscale`
 * among a wall of identical prefixes.
 */
export function splitName(name: string): NameParts {
  const at = name.lastIndexOf('.');
  return at < 0 ? { path: '', leaf: name } : { path: name.slice(0, at + 1), leaf: name.slice(at + 1) };
}

/** Which dimensions differ, and which are new — per side, ready to style. */
export interface ShapeDiff {
  /** One flag per dimension of the old shape: `true` where it differs from its counterpart. */
  old: boolean[];
  /** Ditto for the new shape. A dimension with no counterpart counts as differing. */
  new: boolean[];
}

/**
 * Compare two shapes **from the right**, so an added leading dimension reads as one.
 *
 * A fused tensor is its unfused parts stacked along a new leading axis: `(3072, 1536)` becomes
 * `(256, 3072, 1540)`. Aligned from the left, every dimension differs and the row says nothing; aligned
 * from the right, `3072` matches `3072`, `1536`/`1540` is the real change, and the leading `256` is the
 * fold. That is the difference between a row you can read and a row you have to work out.
 */
export function shapeDiff(oldShape: number[], newShape: number[]): ShapeDiff {
  const out: ShapeDiff = { old: oldShape.map(() => true), new: newShape.map(() => true) };
  const common = Math.min(oldShape.length, newShape.length);
  for (let i = 1; i <= common; i += 1) {
    const same = oldShape[oldShape.length - i] === newShape[newShape.length - i];
    out.old[oldShape.length - i] = !same;
    out.new[newShape.length - i] = !same;
  }
  return out;
}

/** What kind of change a changed row is. */
export type ChangeKind = 'dtype' | 'shape' | 'both' | 'values';

/**
 * Classify a changed tensor.
 *
 * The four kinds are not decoration: a re-quantization changes the dtype of everything and the shape of
 * the expert tensors only, so grouping by kind turns 747 rows into "624 dtype only, 123 dtype and
 * shape" — which is the shape of what happened, visible without reading a row.
 */
export function changeKind(c: TensorChange): ChangeKind {
  const dtype = c.old.dtype !== c.new.dtype;
  const shape = c.old.shape.join() !== c.new.shape.join();
  if (dtype && shape) return 'both';
  if (dtype) return 'dtype';
  if (shape) return 'shape';
  // Same signature: the report only calls it changed when the *values* differ (`--values`).
  return 'values';
}

/** The heading for a group of changed rows. */
export function changeKindLabel(kind: ChangeKind): string {
  switch (kind) {
    case 'dtype':
      return 'dtype only';
    case 'shape':
      return 'shape only';
    case 'both':
      return 'dtype and shape';
    case 'values':
      return 'values only — same dtype and shape';
  }
}

/** Changed rows grouped by kind, in a fixed order so the report does not reshuffle between loads. */
export function byChangeKind(changes: TensorChange[]): { kind: ChangeKind; rows: TensorChange[] }[] {
  const order: ChangeKind[] = ['both', 'shape', 'dtype', 'values'];
  const buckets = new Map<ChangeKind, TensorChange[]>();
  for (const c of changes) {
    const kind = changeKind(c);
    const into = buckets.get(kind);
    if (into) into.push(c);
    else buckets.set(kind, [c]);
  }
  return order
    .filter((k) => (buckets.get(k)?.length ?? 0) > 0)
    .map((kind) => ({ kind, rows: buckets.get(kind) ?? [] }));
}

/** `[F16, (6, 4)]`, for the one-sided rows where there is nothing to compare against. */
export function sigText(s: TensorSig): string {
  return `${s.dtype} (${s.shape.join(', ')})`;
}

/** One piece of a signature, and whether it is the piece that differs. */
export interface SigCell {
  text: string;
  differs: boolean;
}

/** A signature split into the pieces a row can tint one at a time. */
export interface SigCells {
  dtype: SigCell;
  dims: SigCell[];
}

/**
 * Split one side's signature against the other's, marking only the pieces that actually differ.
 *
 * **Why not tint the whole side.** The comparison used to paint every changed row's old side red and
 * its new side green, which says "these two are not the same" — a thing the `~` in the margin already
 * said — while saying nothing about *what* is not the same. On a re-quantization, where every row
 * differs in dtype and only the expert rows differ in shape, that is the whole question. And it was
 * actively misleading on the parts that match: a group whose two sides both read `6 tensors` (it is
 * "changed" only because something beneath it is) had both copies of the same number highlighted.
 *
 * `other` is `null` on a one-sided row: nothing to compare against, so nothing is marked — the row's
 * band and its name already say it exists on one side only.
 */
export function sigCells(mine: TensorSig, other: TensorSig | null): SigCells {
  const dims = shapeDiff(mine.shape, other?.shape ?? []).old;
  return {
    dtype: { text: mine.dtype, differs: other != null && mine.dtype !== other.dtype },
    dims: mine.shape.map((d, i) => ({
      text: String(d),
      differs: other != null && (dims[i] ?? false),
    })),
  };
}
