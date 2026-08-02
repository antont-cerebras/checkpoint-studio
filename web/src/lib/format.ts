import type { ShardTensors } from './types';

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

/**
 * How one overall total changed: `size: 1.9 TiB → 451.8 GiB (-1.5 TiB, -77.0%)`.
 *
 * Mirrors Rust `diff::totals_line`, case-by-case in the parity fixture. The report used to print
 * `451.8 GiB → 32 B` and stop, leaving the reader to work out a difference the terminal states
 * outright — and the side-by-side showed neither size nor parameter count at all.
 *
 * `(unchanged)` when the two agree, and no percentage when the old side is zero: there is no
 * percentage of nothing. The percentage goes through [[fixed1]] for the same reason every other
 * one-decimal number here does — an exact tie (`12.25%`) rounds to even in Rust and away from zero
 * in a bare `toFixed`.
 */
export function totalsLine(
  label: string,
  oldTotal: number,
  newTotal: number,
  fmt: (n: number) => string,
): string {
  const p = totalsParts(oldTotal, newTotal, fmt);
  if (!p.delta) return `${label}: ${p.to} (unchanged)`;
  return `${label}: ${p.from} → ${p.to} (${p.delta}${p.percent ? `, ${p.percent}` : ''})`;
}

/** The same change, in pieces — for a layout that is not a line of text. */
export interface TotalsParts {
  from: string;
  to: string;
  /** `+72 B`, or `''` when the two are equal. */
  delta: string;
  /** `+78.3%`, or `''` when there is no baseline to be relative to (or nothing changed). */
  percent: string;
  /** Which way it went, for colour: `0` when unchanged. */
  direction: -1 | 0 | 1;
}

/**
 * The parts [[totalsLine]] assembles.
 *
 * Split out so a *screen* can lay them out — a label column, the two values, the delta as its own
 * chip — while the assembled string stays one implementation, pinned against Rust case by case in
 * `parity.test.ts`. Two functions computing one delta would be the drift that fixture exists to stop.
 */
export function totalsParts(
  oldTotal: number,
  newTotal: number,
  fmt: (n: number) => string,
): TotalsParts {
  const from = fmt(oldTotal);
  const to = fmt(newTotal);
  if (oldTotal === newTotal) return { from, to, delta: '', percent: '', direction: 0 };
  const diff = newTotal - oldTotal;
  const sign = diff >= 0 ? '+' : '-';
  return {
    from,
    to,
    delta: `${sign}${fmt(Math.abs(diff))}`,
    percent: oldTotal === 0 ? '' : `${sign}${fixed1((Math.abs(diff) / oldTotal) * 100)}%`,
    direction: diff >= 0 ? 1 : -1,
  };
}

/**
 * What a checkpoint-address box accepts, in one sentence.
 *
 * Every such box accepts the same set, because they all resolve through `crate::opening::resolve` —
 * so saying different things in different boxes is not a difference in the app, only in its labels.
 * The open prompt used to promise less than the two comparison boxes ("path on the ssh proxy, or an
 * s3:// prefix"), which reads as "globs and `hf://` are not for this box" when they always were.
 *
 * `proxyHost` names the host `:/path` resolves to, which only the server knows (`/api/recents`) —
 * "the ssh proxy" is only as informative as the reader's memory of their config file.
 */
export function specHelp(proxied: boolean, proxyHost = ''): string {
  return proxied
    ? `a path, glob, hf:// repo, s3:// URI, host:/path, or :/path on ${proxyHost || 'the ssh proxy'}`
    : 'a path, glob, hf:// repo, s3:// URI, or [user@]host:/path';
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

/** What a shard contributes, for a file-browser row: `1062 tensors · 6.4% of params`.
 * Mirrors `shard_note` in src/ui/files.rs — the same shard has to read the same way in
 * both browsers, which is why the wording lives in one function per side rather than
 * inline in a component. */
export function shardNote(shard: ShardTensors): string {
  const unit = shard.tensors === 1 ? 'tensor' : 'tensors';
  return `${shard.tensors} ${unit} · ${percent(shard.params_share, shard.params === 0)} of params`;
}

/**
 * Shorten a path by dropping the *middle*, keeping both ends: `/net/…/3bit-22s-kvh5/260724`.
 *
 * Checkpoint addresses are long and differ at the tail — `…/3bit-22s/260626` against
 * `…/3bit-22s-kvh5/260724` — so the CSS default of an ellipsis at the end removes precisely the part
 * that tells the two apart. At 600px both boxes of a comparison read `/net/antont-vm/srv,` and the
 * view became two identical labels over two different checkpoints.
 *
 * Returns the string unchanged when it already fits, and keeps slightly more of the tail than the
 * head when the budget is odd: the tail is the distinguishing end.
 */
export function middleTruncate(s: string, max: number): string {
  if (max <= 1) return s.length <= max ? s : '…';
  if (s.length <= max) return s;
  const keep = max - 1;
  const head = Math.floor(keep / 2);
  return `${s.slice(0, head)}…${s.slice(s.length - (keep - head))}`;
}
