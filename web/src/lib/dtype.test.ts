// The dtype explainer behind the badge's hover bubble. The table is prose, so what's
// worth testing is that every dtype the app can display gets an entry (a checkpoint
// full of `F8_E4M3` shouldn't hover as "stored numeric type") and that lookup is
// case-insensitive, since dtype strings arrive from several readers.

import { describe, expect, it } from 'vitest';
import { dtypeInfo } from './dtype';

// The safetensors dtypes the readers produce, plus the packed widths the quantized
// checkpoints use.
const KNOWN = [
  'BF16', 'F16', 'F32', 'F64', 'F8_E4M3', 'F8_E5M2',
  'I8', 'U8', 'I16', 'U16', 'I32', 'U32', 'I64', 'U64',
  'BOOL', 'U4', 'I4',
];

describe('dtypeInfo', () => {
  it.each(KNOWN)('explains %s', (d) => {
    const info = dtypeInfo(d);
    expect(info).not.toContain('stored numeric type');
    expect(info.length).toBeGreaterThan(20);
  });

  it('is case-insensitive', () => {
    expect(dtypeInfo('bf16')).toBe(dtypeInfo('BF16'));
    expect(dtypeInfo('f8_e4m3')).toBe(dtypeInfo('F8_E4M3'));
  });

  it('names the width and signedness of the integer types', () => {
    expect(dtypeInfo('I8')).toMatch(/8-bit signed/);
    expect(dtypeInfo('U64')).toMatch(/64-bit unsigned/);
  });

  it('falls back to a generic line for an unknown dtype, echoing it back', () => {
    expect(dtypeInfo('WEIRD9')).toBe('WEIRD9 — stored numeric type.');
    expect(dtypeInfo('')).toBe(' — stored numeric type.');
  });
});
