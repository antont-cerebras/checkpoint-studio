// The heatmap colour ramp. Every cell of every heatmap goes through `viridis`, so a
// non-finite or out-of-range sample must still produce a valid CSS colour — one
// `rgb(NaN,…)` and the canvas silently paints nothing.

import { describe, expect, it, vi } from 'vitest';
import { cssVar, viridis } from './color';

const rgb = (s: string) => s.match(/^rgb\((\d+),(\d+),(\d+)\)$/)?.slice(1).map(Number);

describe('viridis', () => {
  it('anchors the ends of the ramp on the first and last stop', () => {
    expect(viridis(0)).toBe('rgb(68,1,84)');
    expect(viridis(1)).toBe('rgb(253,231,37)');
  });

  it('interpolates between stops', () => {
    // 0.5 lands exactly on the middle stop of five.
    expect(viridis(0.5)).toBe('rgb(33,145,140)');
    // A quarter of the way is between stops 1 and 2, so it matches neither.
    const q = viridis(0.25);
    expect(q).not.toBe(viridis(0));
    expect(q).not.toBe(viridis(0.5));
  });

  it('clamps out-of-range input instead of extrapolating', () => {
    expect(viridis(-3)).toBe(viridis(0));
    expect(viridis(42)).toBe(viridis(1));
    expect(viridis(Infinity)).toBe(viridis(0)); // non-finite → the low end
    expect(viridis(-Infinity)).toBe(viridis(0));
  });

  it('yields a valid colour for NaN rather than rgb(NaN,NaN,NaN)', () => {
    expect(viridis(NaN)).toBe(viridis(0));
    expect(rgb(viridis(NaN))).toEqual([68, 1, 84]);
  });

  it('always returns integer channels in 0..255', () => {
    for (let i = 0; i <= 40; i++) {
      const parts = rgb(viridis(i / 40));
      expect(parts).toBeDefined();
      for (const c of parts!) {
        expect(Number.isInteger(c)).toBe(true);
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(255);
      }
    }
  });

  it('rises monotonically in brightness across the ramp', () => {
    const lum = (t: number) => {
      const [r, g, b] = rgb(viridis(t))!;
      return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!;
    };
    for (let i = 0; i < 20; i++) expect(lum((i + 1) / 20)).toBeGreaterThan(lum(i / 20));
  });
});

describe('cssVar', () => {
  const stubComputedStyle = (value: string) => {
    vi.stubGlobal('document', { documentElement: {} });
    vi.stubGlobal('getComputedStyle', () => ({ getPropertyValue: () => value }));
  };

  it('reads the property off :root', () => {
    stubComputedStyle('  #1e1e1e ');
    expect(cssVar('--bg')).toBe('#1e1e1e');
    vi.unstubAllGlobals();
  });

  it('falls back when the property is unset, so the canvas is never drawn with ""', () => {
    stubComputedStyle('');
    expect(cssVar('--nope')).toBe('#888');
    expect(cssVar('--nope', '#fff')).toBe('#fff');
    vi.unstubAllGlobals();
  });
});
