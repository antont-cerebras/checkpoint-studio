// The row glyphs, as the terminal's tree legend defines them — one place, so a tree in the browser and
// the same tree in the terminal are read with one set of habits.
//
// The legend (`src/ui/legend.rs`, `Legend::Tree`) is the contract:
//   ▾ ▸  a group, expanded / collapsed
//   ·    a tensor (a stored array)
//   ✚    an extra tensor on disk but not listed in the index (model.safetensors.index.json)
//   †    a metadata entry
//   ▦ N  number of tensors in the group / checkpoint
//   ≡ N  number of layers (numbered sub-groups) in the group
//
// The web tree used to leave a tensor's glyph slot *empty*, so the two surfaces disagreed about what a
// row even looks like, and its names started one glyph-width left of the terminal's.

/** Every glyph a row can carry, named. */
export const GLYPH = {
  expanded: '▾',
  collapsed: '▸',
  tensor: '·',
  /** On disk but absent from the index — the terminal tints this one too (`palette::UNINDEXED`). */
  unindexed: '✚',
  metadata: '†',
  tensorCount: '▦',
  layerCount: '≡',
} as const;

/** What a row is, as far as its glyph is concerned. */
export type RowShape =
  | { kind: 'group'; fold: 'open' | 'closed' }
  /** `listing` says whether the index mentions the tensor's file — an unlisted one is worth seeing. */
  | { kind: 'tensor'; listing: 'listed' | 'unlisted' }
  | { kind: 'metadata' };

/** The glyph that leads a row. */
export function rowGlyph(row: RowShape): string {
  switch (row.kind) {
    case 'group':
      return row.fold === 'open' ? GLYPH.expanded : GLYPH.collapsed;
    case 'tensor':
      return row.listing === 'unlisted' ? GLYPH.unindexed : GLYPH.tensor;
    case 'metadata':
      return GLYPH.metadata;
  }
}
