// Typed fetch wrappers over the Rust `--web` JSON API. The server owns the data;
// these just fetch it. Errors surface the server's `{error}` envelope.

import { noteServedBuild } from './build';
import { CHECK_PARAMS } from './params.generated';
import { totalBytes } from './progress';
import { scopeToQuery, type DiffScopeParams } from './diffscope';
import { packingToQuery, type Packing } from './packing';
import type { JobStatus } from '../stores/jobs';
import type { ReadingProgress } from '../stores/reading';
import type { ComparisonSet, DiffTreeResponse } from './difftree';
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
  onDecoding?: () => void,
  signal?: AbortSignal,
): Promise<T> {
  const res = await fetch(url, signal ? { signal } : {});
  noteServedBuild(res.headers.get(BUILD_HEADER));
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
    // The last byte has arrived, and the slow part is about to start: decoding and parsing 91 MB
    // blocks the main thread for tens of seconds, with the bar stuck at 100% and its timer frozen
    // because nothing can re-render. Announcing the change of phase *and yielding a task* is what
    // lets the new label paint before the thread is taken.
    if (onDecoding) {
      onDecoding();
      await new Promise((resume) => setTimeout(resume, 0));
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

/**
 * The server is mid-read and refused this one, naming what it is busy with.
 *
 * A distinct error type, not a string to match on: the reply carries the running spec and how long it
 * has been going so a caller can *offer to stop it*, and prose is the wrong place to keep facts a
 * button needs. `postJson` raises this whenever the server says stopping is available.
 */
export class BusyError extends Error {
  readonly busyWith: string;
  readonly busyForSeconds: number;
  constructor(message: string, busyWith: string, busyForSeconds: number) {
    super(message);
    this.name = 'BusyError';
    this.busyWith = busyWith;
    this.busyForSeconds = busyForSeconds;
  }
}

/** Whether a refusal body is the server saying "something else is reading; you may stop it". */
function busyFrom(body: unknown, message: string): BusyError | null {
  if (typeof body !== 'object' || body === null || !('can_stop_other' in body)) return null;
  const b = body as { busy_with?: unknown; busy_for_seconds?: unknown };
  return new BusyError(
    message,
    typeof b.busy_with === 'string' ? b.busy_with : '',
    typeof b.busy_for_seconds === 'number' ? b.busy_for_seconds : 0,
  );
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
function fetchJson<T>(
  url: string,
  onProgress?: OnProgress,
  onDecoding?: () => void,
  signal?: AbortSignal,
): Promise<T> {
  return typeof onProgress === 'function'
    ? getJsonStreamed<T>(url, onProgress, onDecoding, signal)
    : getJson<T>(url);
}

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  noteServedBuild(res.headers.get(BUILD_HEADER));
  // `res.json()` is `any`; keep it `unknown` and narrow, so a malformed error
  // envelope can't smuggle an untyped value into the app.
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) throw new Error(serverError(body, res.status));
  return body as T;
}

/**
 * Which build of the UI the server serves, sent on every response.
 *
 * Read here rather than by asking, because this is the layer that already talks to the server: the
 * first request a tab makes after the server is reinstalled under it is the one that finds out
 * (`stores/version` explains the other two chances).
 */
const BUILD_HEADER = 'X-App-Build';

const enc = encodeURIComponent;

/**
 * Which comparison to render a command for, named as the generated table names it
 * (`params.generated.ts`, from `src/web/params.rs`).
 */
export type CheckKind = 'values' | 'histogram' | 'verifyRepack';

/**
 * The check as a query tail: `&values=1`, plus `&full=1` for the structural report's row fold.
 *
 * Keyed through the generated table rather than by writing the wire names here — the parameter is
 * `verify_repack` and the field is `verifyRepack`, and that pairing is what the table exists to hold in
 * one place.
 */
function checkTail(check: CheckKind | undefined, full: boolean): string {
  const parts: string[] = [];
  if (check) parts.push(`&${WIRE[check]}=1`);
  if (full) parts.push(`&${WIRE.full}=1`);
  return parts.join('');
}

/** Every check field's wire key, from the generated table — typed as total, so there is no
 * "unknown field" branch to fall back through: the fields *are* the table's. */
const WIRE = Object.fromEntries(CHECK_PARAMS.map((p) => [p.field, p.key])) as Record<
  (typeof CHECK_PARAMS)[number]['field'],
  string
>;

/** The packing as a query tail — only what is set, and only the verification reads it. */
function packingTail(packing: Packing | undefined): string {
  return packingToQuery(packing)
    .map(([k, v]) => `&${k}=${enc(v)}`)
    .join('');
}

/** A scope as a query tail, or nothing. One place, so the two diff routes cannot encode it differently. */
function scopeTail(scope: DiffScopeParams | undefined): string {
  return scope === undefined
    ? ''
    : scopeToQuery(scope)
        .map(([k, v]) => `&${k}=${enc(v)}`)
        .join('');
}

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
  noteServedBuild(res.headers.get(BUILD_HEADER));
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    const message = serverError(body, res.status);
    throw busyFrom(body, message) ?? new Error(message);
  }
  return body as T;
}

/** Drop one entry from the recents list. `DELETE`, because it removes one identified thing. */
async function deleteJson<T>(url: string): Promise<T> {
  const res = await fetch(url, { method: 'DELETE' });
  noteServedBuild(res.headers.get(BUILD_HEADER));
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
  /** Structural diff against another checkpoint. Rejects with the server's message (a 400) when the
   * spec is not a readable checkpoint, or when a scope's glob does not compile.
   *
   * `scope` is the CLI's selection flags — see `lib/diffscope`. Omitted means the whole comparison. */
  /** `swapped` turns the comparison round — the open checkpoint as the baseline. `full` says the reader
   * expanded the families, which the offered command has to carry. */
  /**
   * The one-page report for a comparison the server has set up.
   *
   * By `id`, like [[difftree]]: both views read the pair the comparison slot already holds, so
   * neither re-reads a checkpoint, and the two cannot end up describing different pairs — which is
   * what happened when this resolved its own baseline and compared it against whatever the server
   * had *open* rather than against the candidate the reader named.
   */
  diff: (id: number, scope?: DiffScopeParams, swapped = false, full = false) =>
    getJson<DiffResponse>(
      `/api/diff?id=${id}${swapped ? '&swap=1' : ''}${full ? '&full=1' : ''}${scopeTail(scope)}`,
    ),
  /** The compact (family-folded) tree, optionally scoped by the filter query. */
  compact: (q: string) => getJson<CompactTree>(`/api/compact?q=${enc(q)}`),
  /** Read another checkpoint and serve it instead. Rejects with the server's message for a
   * path that doesn't resolve, in which case the served checkpoint is unchanged. */
  /** `stopOther` asks the server to cancel a read already in progress and take its place — what the
   * "Stop it" offer on a [[BusyError]] does. */
  open: (spec: string, stopOther = false) =>
    postJson<OpenResponse>(`/api/open?path=${enc(spec)}${stopOther ? '&stop_other=1' : ''}`),
  /** Which build the server serves, so a tab can tell it has gone stale (see `lib/build`). */
  version: () =>
    getJson<{ app: string; assets: string | null; spec: string }>('/api/version'),
  /** How far the read in flight has got — the only view into a synchronous open (see
   * `stores/reading`). `{reading: null}` when the server is idle. */
  reading: () => getJson<{ reading: ReadingProgress | null }>('/api/reading'),
  /** The checkpoints opened this run, most recent first, and whether this server reads over
   * an ssh proxy (which decides what kind of path the prompt accepts). */
  recents: () =>
    getJson<{ recents: string[]; proxied: boolean; proxy_host: string | null }>('/api/recents'),
  /**
   * Set up a comparison. `right` empty means "the checkpoint that is open" — the common case, which
   * costs no second read. A read either way, so it can take seconds.
   *
   * Answers with the comparison's `id`, which [[difftree]] must quote, and the two specs as the
   * server *resolved* them — which is what the caller checks the returned tree against.
   */
  setComparison: (left: string, right: string, stopOther = false) =>
    postJson<ComparisonSet>(
      `/api/compare?left=${enc(left)}&right=${enc(right)}${stopOther ? '&stop_other=1' : ''}`,
    ),
  /**
   * Start a long-running comparison. Answers with the id to poll — see `stores/jobs`.
   *
   * `params` is built by the caller because the value modes take a dozen between them; encoding them
   * here would mean a second place that knows the flag names.
   */
  // `params = []` for the generic accessor guard in api.test.ts, which calls every method with a single
  // string on the premise that one argument satisfies any first parameter — same reason `fetchJson`
  // tests `typeof onProgress`.
  startJob: (kind: 'values' | 'verify-repack', params: [string, string][] = []) =>
    postJson<{ id: number }>(
      `/api/jobs/${kind}?${params.map(([k, v]) => `${k}=${enc(v)}`).join('&')}`,
    ),
  /** Where a job has got to, and what it has found so far. */
  jobStatus: (id: number) => getJson<JobStatus>(`/api/jobs/${id}`),
  /** Ask a job to stop. Cooperative: the state becomes `cancelled` once the work notices. */
  cancelJob: (id: number) => deleteJson<JobStatus>(`/api/jobs/${id}`),
  /** Drop the comparison, freeing whatever it held. */
  clearComparison: () => deleteJson<{ comparison: null }>('/api/compare'),
  /**
   * The two checkpoints aligned into one tree, for the comparison with this `id`.
   *
   * Streamed: a 31k-tensor comparison carries both sides, so it is the largest body this API serves.
   * `onDecoding` fires when the last byte has arrived and the parse is about to block the thread — the
   * phase that used to look like a hang (see `LoadStep`'s `building`).
   *
   * `id` is required by the server. There is one comparison slot per server, and this route used to
   * answer from whatever was in it — so two overlapping clients received each other's results with a
   * `200`. Quoting the id makes a lost race a `409` rather than a confident wrong answer.
   */
  difftree: (
    id: number,
    scope?: DiffScopeParams,
    onProgress?: OnProgress,
    onDecoding?: () => void,
    signal?: AbortSignal,
    /** `full`: every layer as its own row. Off means uniform families arrive folded, which is what
     * makes a 117,000-row comparison readable — the server does the folding. */
    full = false,
  ) =>
    fetchJson<DiffTreeResponse>(
      // The scope is applied at *align* time, so changing it re-aligns without re-reading either
      // checkpoint — the comparison id still identifies the pair.
      `/api/difftree?id=${id}${full ? '&full=1' : ''}${scopeTail(scope)}`,
      onProgress,
      onDecoding,
      signal,
    ),
  /**
   * The names a comparison's two sides **share**, for the exact-name picker.
   *
   * Only the *alignment* half of the scope changes the answer, but the whole scope goes on the wire:
   * the caller holds one object, and a client that sent half of it would be one refactor away from
   * sending the wrong half. `q` is a fuzzy search, ranked by the same matcher the tree screen uses;
   * `limit` caps the rows so a keystroke costs kilobytes rather than the 91 MB the aligned tree does.
   */
  diffNames: (id: number, scope?: DiffScopeParams, q = '', limit = 100) =>
    getJson<{ total: number; matched: number; names: string[] }>(
      `/api/diffnames?id=${id}&limit=${limit}${q ? `&q=${enc(q)}` : ''}${scopeTail(scope)}`,
    ),
  /**
   * One side's namespaces, with the number of tensors under each — for the subtree pickers.
   *
   * No scope: a re-root is applied *to* this answer, so applying one here would offer prefixes of
   * prefixes. `side` is which checkpoint, since the two have different namespaces — that is the whole
   * reason the field exists.
   */
  subtrees: (id: number, side: 'old' | 'new', q = '', limit = 100) =>
    getJson<{ total: number; subtrees: { prefix: string; tensors: number }[] }>(
      `/api/subtrees?id=${id}&side=${side}&limit=${limit}${q ? `&q=${enc(q)}` : ''}`,
    ),
  /**
   * The terminal invocation for a set of parameters — **the only way this client obtains one**.
   *
   * `check` says which comparison (`values`, `histogram`, `verify-repack`, or the structural default),
   * and the scope goes with it. Rendered server-side from the same table that decides which parameters
   * are accepted at all, so a control the panel sets cannot be one the command drops. The Data view used
   * to build its line here from the two addresses and silently dropped the whole selection.
   */
  command: (
    left: string,
    right: string,
    scope?: DiffScopeParams,
    check?: CheckKind,
    full = false,
    packing?: Packing,
  ) =>
    getJson<{ command: string | null }>(
      `/api/command?left=${enc(left)}&right=${enc(right)}${checkTail(check, full)}${packingTail(packing)}${scopeTail(scope)}`,
    ),
  /** Forget one checkpoint. Returns the list without it; rejects with a 404 if it wasn't there. */
  forgetRecent: (spec: string) =>
    deleteJson<{ forgot: string; recents: string[] }>(`/api/recents?path=${enc(spec)}`),
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
