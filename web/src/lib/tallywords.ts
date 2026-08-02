/**
 * What each count in a comparison actually counts — the tooltips both diff views hang off their chips.
 *
 * *Metadata* is the word that needed saying. A tensor has a dtype and a shape, and calling those
 * "metadata" is entirely reasonable — but this report does not: they are what makes a tensor
 * **changed**, and the metadata section is the checkpoint's own `__metadata__` header, the free-form
 * key/value strings a writer saves beside the tensors. Two things one word could mean, one of them
 * wrong, in a chip with no room to explain itself.
 *
 * Here rather than in either view because both draw the same strip, and a definition of "changed" that
 * differed between the summary and the aligned tree would be worse than none.
 */

/** The kinds a chip can stand for, as the report names them. */
export type TallyKind = 'unchanged' | 'added' | 'removed' | 'changed' | 'metadata';

/** One sentence per kind, in the vocabulary of `diff OLD NEW`: baseline (old) and candidate (new). */
export const TALLY_MEANS: Record<TallyKind, string> = {
  unchanged: 'Tensors in both checkpoints with the same dtype and shape',
  added: 'Tensors the candidate has and the baseline does not',
  removed: 'Tensors the baseline has and the candidate does not',
  changed: 'Tensors in both, whose dtype or shape differs',
  metadata:
    "The checkpoint's own metadata header — the free-form key/value strings saved beside the " +
    'tensors. A tensor\'s dtype and shape are not counted here; a tensor whose dtype or shape ' +
    'differs is a changed tensor.',
};

/** The tooltip for a chip: what it counts, and — when it leads somewhere — that it does. */
export function tallyTitle(kind: TallyKind, count: number, opens = false): string {
  // Trimmed of its own full stop first: `metadata` ends in one because it is three sentences, and
  // appending to it read `… a changed tensor.. None here.`
  const means = TALLY_MEANS[kind].replace(/\.$/, '');
  if (count === 0) return `${means}. None here.`;
  return opens ? `${means}. Click to show them.` : means;
}
