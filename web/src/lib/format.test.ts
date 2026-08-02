import { describe, expect, it } from 'vitest';
import {
  humanCount,
  humanSize,
  middleTruncate,
  num,
  percent,
  pyShape,
  shape,
  specHelp,
} from './format';

// These formatters exist on BOTH sides of the app: the TUI formats in Rust
// (`utils::format_size`, `format_parameters`, `format_percent`), the web UI formats
// here. What makes them agree is `parity.test.ts`, which checks these against a fixture
// generated from the Rust functions — see shared/parity/README.md. The cases below are
// the local, behavioural ones: unit thresholds, saturation, exact zero vs a tiny
// fraction, and the display-vs-copy split for shapes.
describe('humanSize', () => {
  it('stays in bytes below 1 KiB', () => {
    expect(humanSize(0)).toBe('0 B');
    expect(humanSize(1023)).toBe('1023 B');
  });
  it('switches unit exactly at each 1024 boundary', () => {
    expect(humanSize(1024)).toBe('1.0 KiB');
    expect(humanSize(1024 * 1024)).toBe('1.0 MiB');
    expect(humanSize(1024 ** 3)).toBe('1.0 GiB');
    expect(humanSize(1024 ** 4)).toBe('1.0 TiB');
  });
  // One decimal at every magnitude, like the TUI. (This used to be two decimals below
  // 10, which is where the two UIs disagreed: `1.50 KiB` vs `1.5 KiB`.)
  it('uses one decimal above the byte range', () => {
    expect(humanSize(9.5 * 1024)).toBe('9.5 KiB');
    expect(humanSize(10 * 1024)).toBe('10.0 KiB');
  });
  // Sizes are power-of-two divisions, so exact ties are common: 1280 B is exactly
  // 1.25 KiB. Rust rounds ties to even, and so must this.
  it('rounds an exact tie to even, as Rust does', () => {
    expect(humanSize(1280)).toBe('1.2 KiB');
    expect(humanSize(1792)).toBe('1.8 KiB');
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
    expect(humanCount(1000)).toBe('1.0K');
    expect(humanCount(1_000_000)).toBe('1.0M');
    expect(humanCount(1_000_000_000)).toBe('1.0B');
    expect(humanCount(1_000_000_000_000)).toBe('1.0T');
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

describe('shortening a path for a narrow box', () => {
  // The tail is what distinguishes two checkpoints, and it is what an end-ellipsis throws away.
  it('keeps both ends and drops the middle', () => {
    const p = '/net/data/ws/models/kimi-k2.6/3bit-22s-kvh5/260724';
    const short = middleTruncate(p, 24);
    expect(short).toHaveLength(24);
    expect(short.startsWith('/net/data')).toBe(true);
    expect(short.endsWith('260724')).toBe(true);
  });

  it('tells apart two paths that differ only at the end', () => {
    const a = middleTruncate('/models/kimi/3bit-22s/260626', 20);
    const b = middleTruncate('/models/kimi/3bit-22s-kvh5/260724', 20);
    expect(a).not.toEqual(b);
  });

  it('leaves a string that already fits alone', () => {
    expect(middleTruncate('/short/path', 40)).toBe('/short/path');
    expect(middleTruncate('exact', 5)).toBe('exact');
  });

  it('degrades to an ellipsis rather than producing nonsense', () => {
    expect(middleTruncate('/a/long/path', 1)).toBe('…');
    expect(middleTruncate('x', 1)).toBe('x');
  });
});

// Every checkpoint-address box shows this one sentence, because they all resolve through the same
// `crate::opening::resolve` and accept the same set. The open prompt used to promise less than the
// comparison boxes did, which reads as a narrower feature rather than a shorter label.
describe('what an address box accepts', () => {
  it('names the proxy host, since only the server knows which one `:` means', () => {
    const help = specHelp(true, 'lab@build-host');
    expect(help).toContain(':/path on lab@build-host');
    expect(help).toContain('s3:// URI');
  });

  it('falls back to naming the proxy generically when the host is unknown', () => {
    expect(specHelp(true)).toContain(':/path on the ssh proxy');
  });

  it('drops the proxy forms entirely when this server has none', () => {
    const help = specHelp(false);
    // The shorthand is what goes away — `[user@]host:/path` carries its own host and stays.
    expect(help).not.toContain('ssh proxy');
    expect(help).toContain('[user@]host:/path');
  });
});
