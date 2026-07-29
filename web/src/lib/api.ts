// Typed fetch wrappers over the Rust `--web` JSON API. The server owns the data;
// these just fetch it. Errors surface the server's `{error}` envelope.

import { totalBytes } from './progress';
import type {
  FileNode,
  HistogramDto,
  LayoutMap,
  SampleDto,
  StatsDto,
  TensorInfo,
  CompactTree,
  DiffResponse,
  TreeResponse,
} from './types';

/**
 * Fetch JSON while reporting download progress — for the one response big enough to wait
 * on. Streams the body so `onProgress` can be called per chunk; everything else about the
 * result (including the error envelope) matches [[getJson]].
 *
 * Falls back to buffering whole when the browser gives no readable body.
 */
async function getJsonStreamed<T>(
  url: string,
  onProgress: (received: number, total: number | null) => void,
): Promise<T> {
  const res = await fetch(url);
  const total = totalBytes(res.headers);
  const reader = res.ok ? (res.body?.getReader() ?? null) : null;
  let text: string;
  if (reader === null) {
    text = await res.text();
  } else {
    const chunks: Uint8Array[] = [];
    let received = 0;
    onProgress(0, total);
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress(received, total);
    }
    text = new TextDecoder().decode(concat(chunks, received));
  }
  const body: unknown = text === '' ? null : (JSON.parse(text) as unknown);
  if (!res.ok) throw new Error(serverError(body, res.status));
  return body as T;
}

/** One buffer from many, without a `Blob` round-trip. */
function concat(chunks: Uint8Array[], total: number): Uint8Array {
  const out = new Uint8Array(total);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

/** The server's `{error}` envelope, or a bare status when it didn't send one. */
function serverError(body: unknown, status: number): string {
  const detail =
    typeof body === 'object' && body !== null && 'error' in body
      ? (body as { error?: unknown }).error
      : undefined;
  return typeof detail === 'string' ? detail : `HTTP ${status}`;
}

/** Told how many bytes have arrived, and the total when the server announced one. */
export type OnProgress = (received: number, total: number | null) => void;

/**
 * Fetch JSON, streaming it when the caller wants progress and buffering it otherwise.
 *
 * `typeof`, not truthiness: the generic accessor guard in api.test.ts calls every method
 * with one string, on the premise that a string satisfies any first parameter. A callback
 * is the one parameter it doesn't, and streaming to a non-function would throw — so a
 * caller with nothing to report simply gets the plain fetch.
 */
function fetchJson<T>(url: string, onProgress?: OnProgress): Promise<T> {
  return typeof onProgress === 'function'
    ? getJsonStreamed<T>(url, onProgress)
    : getJson<T>(url);
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  // `res.json()` is `any`; keep it `unknown` and narrow, so a malformed error
  // envelope can't smuggle an untyped value into the app.
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) throw new Error(serverError(body, res.status));
  return body as T;
}

const enc = encodeURIComponent;

// `?: T | undefined` throughout: callers build this object with every key present and
// leave the inapplicable ones `undefined` (`qs` drops those), which
// `exactOptionalPropertyTypes` would otherwise reject.
export interface SampleParams {
  mode?: 'grid' | 'max' | 'window' | 'edges' | undefined;
  rows?: number | undefined;
  cols?: number | undefined;
  slice?: number | undefined;
  dtype?: string | undefined;
  row_off?: number | undefined;
  col_off?: number | undefined;
  raw?: number | undefined;
}

function qs(params: Record<string, string | number | undefined>): string {
  return Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== '')
    .map(([k, v]) => `${k}=${enc(String(v))}`)
    .join('&');
}

/**
 * Ask the server to serve a different checkpoint.
 *
 * A POST, and the only one: this changes what every other endpoint answers, and a GET that
 * did that would be a URL a browser could follow on its own (a prefetch, a restored tab) and
 * swap the checkpoint with nobody having asked. The server replies when the new checkpoint is
 * *ready*, so there is no state to poll and reconcile — it either resolved or it didn't.
 */
async function postJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { method: 'POST' });
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) throw new Error(serverError(body, res.status));
  return body as T;
}

/** What `POST /api/open` answers with. */
export interface OpenResponse {
  root: string;
  tensor_count: number;
  opened: string;
  recents: string[];
}

export const api = {
  /** The tensor tree — the one response worth a progress bar (tens of MB for a 31k-tensor
   * checkpoint). `onProgress` is optional so every other caller stays a plain fetch. */
  tree: (onProgress?: OnProgress) => fetchJson<TreeResponse>('/api/tree', onProgress),
  files: () => getJson<FileNode>('/api/files'),
  filter: (q: string) => getJson<{ active: boolean; names?: string[] }>(`/api/filter?q=${enc(q)}`),
  schema: (q: string) =>
    getJson<{
      families: {
        name: string;
        count: number;
        dtype: string | null;
        shape: number[] | null;
        params: number;
        size_bytes: number;
      }[];
    }>(`/api/schema?q=${enc(q)}`),
  /** Structural diff against another checkpoint on the server's filesystem. Rejects
   * with the server's message (a 400) when the path is not a readable checkpoint. */
  diff: (against: string) => getJson<DiffResponse>(`/api/diff?against=${enc(against)}`),
  /** The compact (family-folded) tree, optionally scoped by the filter query. */
  compact: (q: string) => getJson<CompactTree>(`/api/compact?q=${enc(q)}`),
  /** Read another checkpoint and serve it instead. Rejects with the server's message for a
   * path that doesn't resolve, in which case the served checkpoint is unchanged. */
  open: (spec: string) => postJson<OpenResponse>(`/api/open?path=${enc(spec)}`),
  /** The checkpoints opened this run, most recent first, and whether this server reads over
   * an ssh proxy (which decides what kind of path the prompt accepts). */
  recents: () => getJson<{ recents: string[]; proxied: boolean }>('/api/recents'),
  stats: () => getJson<Record<string, unknown>>('/api/stats'),
  health: () => getJson<unknown[]>('/api/health'),
  check: () => getJson<Record<string, unknown> | null>('/api/check'),
  tensor: (name: string) => getJson<TensorInfo>(`/api/tensor?name=${enc(name)}`),
  /** A shard's byte layout. Streamed like the tree when a progress callback is given: a
   * 12k-tensor shard's segment list is megabytes. */
  layout: (file: string, onProgress?: OnProgress) =>
    fetchJson<LayoutMap>(`/api/layout?file=${enc(file)}`, onProgress),
  /** A sidecar's text, up to `PREVIEW_CAP` (4 MiB) — a real `model.safetensors.index.json`
   * is 1.7 MB of it, which is worth a bar. */
  file: (path: string, onProgress?: OnProgress) =>
    fetchJson<{ path: string; name: string; size: number; truncated: boolean; text: string }>(
      `/api/file?path=${enc(path)}`,
      onProgress,
    ),
  tensorStats: (name: string, dtype?: string) =>
    getJson<StatsDto>(`/api/tensor/stats?${qs({ name, dtype })}`),
  sample: (name: string, p: SampleParams) =>
    getJson<SampleDto>(`/api/tensor/sample?${qs({ name, ...p })}`),
  histogram: (name: string, bins?: number, dtype?: string) =>
    getJson<HistogramDto>(`/api/tensor/histogram?${qs({ name, bins, dtype })}`),
};
