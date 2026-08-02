// What the chips promise. The strip is five words and five numbers; these sentences are the only
// place the words are defined, and both diff views hang them off the same call.

import { describe, expect, it } from 'vitest';
import { TALLY_MEANS, tallyTitle } from './tallywords';

describe('what a count means', () => {
  // The report this exists for: `Metadata 0` beside `Changed 0`, with no way to tell that a tensor
  // whose dtype changed is counted in the second and not the first.
  it('says that a tensor’s dtype and shape are not what metadata means here', () => {
    expect(TALLY_MEANS.metadata).toMatch(/dtype and shape are not counted here/);
    expect(TALLY_MEANS.metadata).toMatch(/changed tensor/);
    expect(TALLY_MEANS.changed).toMatch(/dtype or shape differs/);
  });

  // Baseline and candidate, the words the two boxes above the report use — not "old"/"new", which
  // name the direction rather than the checkpoints.
  it('names the two sides the way the screen names them', () => {
    expect(TALLY_MEANS.added).toBe('Tensors the candidate has and the baseline does not');
    expect(TALLY_MEANS.removed).toBe('Tensors the baseline has and the candidate does not');
    expect(TALLY_MEANS.unchanged).toMatch(/same dtype and shape/);
  });
});

describe('the tooltip', () => {
  it('is the meaning alone when the chip is only a count', () => {
    expect(tallyTitle('added', 3)).toBe(TALLY_MEANS.added);
  });

  it('offers the click when there is one, and something to click for', () => {
    expect(tallyTitle('added', 3, true)).toBe(`${TALLY_MEANS.added}. Click to show them.`);
    // Nothing to show: an empty chip says so instead of inviting a click that reveals "none".
    expect(tallyTitle('added', 0, true)).toBe(`${TALLY_MEANS.added}. None here.`);
  });

  // A sentence that already ends in a full stop must not gain a second one.
  it('does not double the full stop of a meaning that has its own', () => {
    expect(tallyTitle('metadata', 0)).not.toMatch(/\.\./);
    expect(tallyTitle('metadata', 0)).toMatch(/None here\.$/);
  });
});
