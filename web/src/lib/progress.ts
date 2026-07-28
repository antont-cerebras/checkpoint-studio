/**
 * Download progress for a streamed JSON response — what the loading bar counts.
 *
 * The terminal's load screen shows `reading checkpoint structure (3.4s)` over a
 * `{done}/{total} shards` gauge, because reading N shard headers is what it is doing. The
 * browser is doing something else: pulling one already-built response, tens of MB of it for
 * a 31k-tensor checkpoint. So the shape is the same — bar, count, elapsed — and the unit is
 * bytes, which is the honest denominator here. Calling them shards would describe work the
 * browser isn't doing.
 */

/** Bytes received so far, and the total when the server announced one. */
export interface Progress {
  received: number;
  /** `null` until known — see [[totalBytes]]. */
  total: number | null;
  /** `performance.now()` when the request started, for the elapsed timer. */
  startedAt: number;
}

/**
 * The body's length before encoding, from the server's headers.
 *
 * `Content-Length` is no use on its own: when the response is gzipped it describes the
 * compressed bytes while `response.body` yields the *decoded* stream, so counting one
 * against the other gives a fraction that runs past 100%. The server therefore sends
 * `X-Uncompressed-Length` whenever it encodes (see `CachedBody` in src/web/mod.rs), and
 * `Content-Length` is correct for everything else.
 *
 * `null` when neither is present — a chunked response with no announced size, which the bar
 * shows as indeterminate rather than as a wrong fraction.
 */
export function totalBytes(headers: Headers): number | null {
  const announced = headers.get('X-Uncompressed-Length') ?? headers.get('Content-Length');
  if (announced === null) return null;
  const n = Number(announced);
  return Number.isFinite(n) && n > 0 ? n : null;
}

/** `0..=1`, or `null` while the total is unknown. Clamped: a proxy that re-encodes could
 * make the decoded stream longer than the announced length, and a bar that overshoots reads
 * as a bug in the app rather than in the proxy. */
export function fraction(p: Progress): number | null {
  if (p.total === null) return null;
  return Math.min(1, p.received / p.total);
}

/** Seconds since the request started, one decimal — the terminal's `(3.4s)`. */
export function elapsedSeconds(p: Progress, now: number): string {
  return `${Math.max(0, (now - p.startedAt) / 1000).toFixed(1)}s`;
}
