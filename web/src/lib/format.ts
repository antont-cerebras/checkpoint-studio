// Human-readable formatting helpers. `humanSize`, `humanCount` and `percent` must
// produce byte-identical strings to the Rust side's `format_size`, `format_parameters`
// and `format_percent` (crates/core/src/utils.rs) — the same tensor has to report the
// same size in the TUI and in the browser. That's enforced, not hoped for:
// `shared/parity/format.json` is generated from the Rust functions and `parity.test.ts`
// checks these against it. See shared/parity/README.md.

/** One decimal place, rounded the way Rust's `{:.1}` rounds.
 *
 * `toFixed` already rounds the exact binary value, so it agrees with Rust everywhere
 * except on an exact tie — a value like `1.25`, which is representable exactly.
 * There `toFixed` rounds away from zero (`1.3`) while Rust rounds to even (`1.2`).
 * Ties are not exotic here: every size is a power-of-two division, so `1280 B` is
 * exactly `1.25 KiB`. An exact tie at one decimal is an odd multiple of 0.25. */
function fixed1(v: number): string {
  if (Number.isInteger(v * 4) && !Number.isInteger(v * 2)) {
    const truncated = Math.trunc(v * 10); // toward zero, e.g. 12 for 1.25
    const towardZero = truncated / 10;
    const even = truncated % 2 === 0 ? towardZero : towardZero + Math.sign(v) * 0.1;
    return even.toFixed(1);
  }
  return v.toFixed(1);
}

/** Byte size with IEC units, e.g. `593.5 MiB`. Bytes stay whole; anything larger gets
 * one decimal. Mirrors Rust `format_size`. */
export function humanSize(bytes: number): string {
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB'];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return i === 0 ? `${bytes} B` : `${fixed1(v)} ${units[i]}`;
}

/** Parameter count in decimal units, e.g. `30.9B`. Mirrors Rust `format_parameters`. */
export function humanCount(n: number): string {
  const units = ['', 'K', 'M', 'B', 'T'];
  let v = n;
  let i = 0;
  while (v >= 1000 && i < units.length - 1) {
    v /= 1000;
    i++;
  }
  return i === 0 ? String(n) : `${fixed1(v)}${units[i]}`;
}

export function shape(dims: number[]): string {
  return dims.length ? dims.join(' × ') : 'scalar';
}

/** Python-reusable tuple form: `(768, 2048)`, `(768,)` for 1D, `()` for scalar. */
export function pyShape(dims: number[]): string {
  if (dims.length === 0) return '()';
  if (dims.length === 1) return `(${dims[0]},)`;
  return `(${dims.join(', ')})`;
}

/** A compact number for grid cells / stats (trims noise, keeps precision). */
export function num(v: number): string {
  if (!Number.isFinite(v)) return v > 0 ? '+∞' : v < 0 ? '-∞' : 'NaN';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1e6 || a < 1e-4) return v.toExponential(3);
  return Number(v.toPrecision(6)).toString();
}

/** A fraction (0–1) as a percentage, the way the TUI shows it: an exact zero reads
 * "0%", a tiny-but-nonzero fraction uses scientific notation (so it never shows a
 * misleading "0.00%"), and the rest one decimal. Pass `isZero` from the true count so
 * floating-point dust never masquerades as an exact zero. */
export function percent(fraction: number, isZero: boolean): string {
  if (isZero) return '0%';
  const pct = fraction * 100;
  return pct < 0.1 ? `${pct.toExponential(1)}%` : `${pct.toFixed(1)}%`;
}
