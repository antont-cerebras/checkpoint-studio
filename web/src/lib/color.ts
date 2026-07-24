// A viridis-like perceptual gradient for the heatmap (no charting dependency).

const STOPS: [number, number, number][] = [
  [68, 1, 84],
  [59, 82, 139],
  [33, 145, 140],
  [94, 201, 98],
  [253, 231, 37],
];

/** Read a CSS custom property off :root (so canvas colors follow the theme). */
export function cssVar(name: string, fallback = '#888'): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/** Map t in [0,1] to an `rgb(...)` string along the gradient. */
export function viridis(t: number): string {
  const last = STOPS.length - 1;
  // NaN would slip past Math.max/min, so clamp it to 0 explicitly — a non-finite
  // sample value must still yield a valid colour rather than `rgb(NaN,…)`.
  const clamped = Number.isFinite(t) ? Math.max(0, Math.min(1, t)) : 0;
  const x = clamped * last;
  const i = Math.min(Math.floor(x), last);
  const f = x - i;
  // `i` is clamped to the array, so both lookups are present; `?? STOPS[0]!` keeps
  // that provable to the compiler without an assertion on the arithmetic.
  const a = STOPS[i] ?? STOPS[0]!;
  const b = STOPS[Math.min(i + 1, last)] ?? a;
  const r = Math.round(a[0] + (b[0] - a[0]) * f);
  const g = Math.round(a[1] + (b[1] - a[1]) * f);
  const bl = Math.round(a[2] + (b[2] - a[2]) * f);
  return `rgb(${r},${g},${bl})`;
}
