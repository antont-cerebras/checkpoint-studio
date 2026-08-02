import { describe, expect, it } from 'vitest';
import { elapsedSeconds, fraction, startedNow, totalBytes, type Progress } from './progress';

const at = (received: number, total: number | null): Progress => ({
  received,
  total,
  startedAt: 1000,
});

describe('a wait with nothing to measure', () => {
  it('starts at zero with no total, so the bar stays indeterminate', () => {
    const before = performance.now();
    const p = startedNow();
    expect(p.received).toBe(0);
    // The server scans and *then* answers, so a byte total would be a fiction. Null is
    // what makes the component show a spinner and a timer instead of a bar at 0%.
    expect(p.total).toBeNull();
    expect(fraction(p)).toBeNull();
    expect(p.startedAt).toBeGreaterThanOrEqual(before);
  });
});

describe('the announced total', () => {
  it('prefers the uncompressed length, because that is what the stream yields', () => {
    // The bug this header exists for: `Content-Length` describes the compressed body while
    // `response.body` is decoded, so counting one against the other runs past 100%.
    const headers = new Headers({
      'Content-Length': '1000',
      'X-Uncompressed-Length': '14000000',
    });
    expect(totalBytes(headers)).toBe(14_000_000);
  });

  it('uses Content-Length when the response is not encoded', () => {
    expect(totalBytes(new Headers({ 'Content-Length': '2048' }))).toBe(2048);
  });

  it('is null when nothing is announced, or the value is useless', () => {
    expect(totalBytes(new Headers())).toBeNull();
    // A chunked response can legitimately announce nothing; 0 and garbage would both make
    // a denominator that produces Infinity or NaN.
    expect(totalBytes(new Headers({ 'Content-Length': '0' }))).toBeNull();
    expect(totalBytes(new Headers({ 'Content-Length': 'lots' }))).toBeNull();
  });
});

describe('the fraction', () => {
  it('is null while the total is unknown, so the bar stays indeterminate', () => {
    expect(fraction(at(500, null))).toBeNull();
  });

  it('measures received against the total', () => {
    expect(fraction(at(0, 100))).toBe(0);
    expect(fraction(at(25, 100))).toBe(0.25);
    expect(fraction(at(100, 100))).toBe(1);
  });

  // A zero denominator is not a fraction either: dividing by it wrote `0 B / 0 B · NaN%` across the
  // wait for a comparison whose size the server had not yet announced.
  it('is null for a zero total, not NaN', () => {
    expect(fraction(at(0, 0))).toBeNull();
  });

  it('clamps rather than overshooting', () => {
    // A proxy that re-encodes can make the decoded stream longer than what was announced.
    // A bar past 100% reads as a bug in the app rather than in the proxy.
    expect(fraction(at(150, 100))).toBe(1);
  });
});

describe('the elapsed timer', () => {
  it('reads like the terminal load screen, to one decimal', () => {
    expect(elapsedSeconds(at(0, null), 4400)).toBe('3.4s');
    expect(elapsedSeconds(at(0, null), 1000)).toBe('0.0s');
  });

  it('never goes backwards if the clock does', () => {
    expect(elapsedSeconds(at(0, null), 900)).toBe('0.0s');
  });
});
