// The glyphs are a contract with the terminal, not decoration: these assert the exact characters its
// tree legend documents, so a change here has to be a deliberate change to both surfaces.

import { describe, expect, it } from 'vitest';
import { GLYPH, rowGlyph } from './glyphs';

describe('the row glyphs', () => {
  it('are the ones the terminal legend lists', () => {
    expect(GLYPH.expanded).toBe('▾');
    expect(GLYPH.collapsed).toBe('▸');
    expect(GLYPH.tensor).toBe('·');
    expect(GLYPH.unindexed).toBe('✚');
    expect(GLYPH.metadata).toBe('†');
    expect(GLYPH.tensorCount).toBe('▦');
    expect(GLYPH.layerCount).toBe('≡');
  });

  it('lead a row by what it is', () => {
    expect(rowGlyph({ kind: 'group', fold: 'open' })).toBe('▾');
    expect(rowGlyph({ kind: 'group', fold: 'closed' })).toBe('▸');
    expect(rowGlyph({ kind: 'tensor', listing: 'listed' })).toBe('·');
    expect(rowGlyph({ kind: 'tensor', listing: 'unlisted' })).toBe('✚');
    expect(rowGlyph({ kind: 'metadata' })).toBe('†');
  });

  // A tensor row's slot used to be empty, which is not "no glyph" but "a different tree": its name
  // started where the terminal's glyph is.
  it('never gives a row an empty glyph', () => {
    const every: Parameters<typeof rowGlyph>[0][] = [
      { kind: 'group', fold: 'open' },
      { kind: 'group', fold: 'closed' },
      { kind: 'tensor', listing: 'listed' },
      { kind: 'tensor', listing: 'unlisted' },
      { kind: 'metadata' },
    ];
    for (const row of every) expect(rowGlyph(row)).not.toBe('');
  });
});
