import { describe, expect, it } from 'vitest';
import { humanCount, humanSize, num, percent, pyShape, shape } from './format';

// These formatters exist on BOTH sides of the app: the TUI formats in Rust
// (`utils::format_size`, the stats/zeros display), the web UI formats here. Nothing
// makes them agree automatically, so pin the observable output — including the boundary
// cases where the two implementations could plausibly drift (unit thresholds, the
// <10 -> 2 decimals switch, exact zero vs a tiny fraction).
describe('humanSize', () => {
  it('stays in bytes below 1 KiB', () => {
    expect(humanSize(0)).toBe('0 B');
    expect(humanSize(1023)).toBe('1023 B');
  });
  it('switches unit exactly at each 1024 boundary', () => {
    expect(humanSize(1024)).toBe('1.00 KiB');
    expect(humanSize(1024 * 1024)).toBe('1.00 MiB');
    expect(humanSize(1024 ** 3)).toBe('1.00 GiB');
    expect(humanSize(1024 ** 4)).toBe('1.00 TiB');
  });
  it('uses 2 decimals below 10 and 1 above', () => {
    expect(humanSize(9.5 * 1024)).toBe('9.50 KiB');
    expect(humanSize(10 * 1024)).toBe('10.0 KiB');
  });
  it('saturates at the largest unit rather than inventing one', () => {
    expect(humanSize(1024 ** 6)).toMatch(/PiB$/);
  });
});

describe('humanCount', () => {
  it('is exact below 1000', () => {
    expect(humanCount(0)).toBe('0');
    expect(humanCount(999)).toBe('999');
  });
  it('uses decimal (not binary) steps — 1000, not 1024', () => {
    expect(humanCount(1000)).toBe('1.00K');
    expect(humanCount(1_000_000)).toBe('1.00M');
    expect(humanCount(1_000_000_000)).toBe('1.00B');
    expect(humanCount(1_000_000_000_000)).toBe('1.00T');
  });
  it('formats a realistic parameter count', () => {
    expect(humanCount(30_900_000_000)).toBe('30.9B');
  });
});

describe('num', () => {
  it('names the non-finite values instead of printing NaN/Infinity', () => {
    expect(num(Number.NaN)).toBe('NaN');
    expect(num(Number.POSITIVE_INFINITY)).toBe('+∞');
    expect(num(Number.NEGATIVE_INFINITY)).toBe('-∞');
  });
  it('prints an exact zero as a single character', () => {
    expect(num(0)).toBe('0');
  });
  it('goes exponential only outside [1e-4, 1e6)', () => {
    expect(num(1e6)).toBe('1.000e+6');
    expect(num(1e-5)).toBe('1.000e-5');
    expect(num(0.5)).toBe('0.5');
    expect(num(-0.000514984130859375)).toBe('-0.000514984');
  });
});

describe('percent', () => {
  // Matches the TUI: an exact zero is "0%", a tiny-but-nonzero fraction goes
  // scientific so it can never be shown as a misleading "0.00%".
  it('shows an exact zero as 0%', () => {
    expect(percent(0, true)).toBe('0%');
  });
  it('uses scientific notation below 0.1%', () => {
    expect(percent(1e-6, false)).toBe('1.0e-4%');
  });
  it('uses one decimal at or above 0.1%', () => {
    expect(percent(0.5, false)).toBe('50.0%');
    expect(percent(0.001, false)).toBe('0.1%');
  });
});

describe('shape', () => {
  it('renders dims for display and Python tuples for copy-paste', () => {
    expect(shape([768, 2048])).toBe('768 × 2048');
    expect(shape([])).toBe('scalar');
    expect(pyShape([768, 2048])).toBe('(768, 2048)');
    expect(pyShape([4])).toBe('(4,)');
    expect(pyShape([])).toBe('()');
  });
});
