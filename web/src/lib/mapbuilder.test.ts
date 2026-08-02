import { describe, expect, it } from 'vitest';
import { parseMappingRules, serializeMappingRules } from './mapbuilder';

describe('mapping builder text', () => {
  it('round-trips ordered mapping rows', () => {
    const text = '^blocks\\.= >ignored\n^blocks\\.=>model.layers.\n(x)=>(left=>right)-$1';
    const parsed = parseMappingRules(text);
    expect(parsed.rawOnly).toBe(true);
    expect(parsed.rules).toEqual([
      { pattern: '^blocks\\.', replacement: 'model.layers.' },
      { pattern: '(x)', replacement: '(left=>right)-$1' },
    ]);
    expect(serializeMappingRules(parsed.rules)).toBe(
      '^blocks\\.=>model.layers.\n(x)=>(left=>right)-$1',
    );
  });

  it('marks comments and incomplete lines as raw-only', () => {
    expect(parseMappingRules('# explain\n^old\\.=>new.\nunfinished')).toEqual({
      rules: [{ pattern: '^old\\.', replacement: 'new.' }],
      rawOnly: true,
    });
  });

  it('ignores blank builder rows when serializing', () => {
    expect(
      serializeMappingRules([
        { pattern: '', replacement: '' },
        { pattern: ' ^old\\. ', replacement: ' new. ' },
      ]),
    ).toBe('^old\\.=>new.');
  });

  // Blank lines are how a rule list is kept readable; they are not rules, and they are not a reason
  // to force the raw editor.
  it('passes over blank lines without leaving the builder', () => {
    expect(parseMappingRules('\n^a\\.=>b.\n\n  \n^c\\.=>d.\n')).toEqual({
      rules: [
        { pattern: '^a\\.', replacement: 'b.' },
        { pattern: '^c\\.', replacement: 'd.' },
      ],
      rawOnly: false,
    });
  });

  it('has nothing to say about nothing', () => {
    expect(parseMappingRules('')).toEqual({ rules: [], rawOnly: false });
    expect(serializeMappingRules([])).toBe('');
  });
});
