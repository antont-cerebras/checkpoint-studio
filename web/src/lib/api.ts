// Typed fetch wrappers over the Rust `--web` JSON API. The server owns the data;
// these just fetch it. Errors surface the server's `{error}` envelope.

import type {
  FileNode,
  HistogramDto,
  LayoutMap,
  SampleDto,
  StatsDto,
  TensorInfo,
  DiffResponse,
  TreeResponse,
} from './types';

async function getJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  // `res.json()` is `any`; keep it `unknown` and narrow, so a malformed error
  // envelope can't smuggle an untyped value into the app.
  const body: unknown = await res.json().catch(() => null);
  if (!res.ok) {
    const detail =
      typeof body === 'object' && body !== null && 'error' in body
        ? (body as { error?: unknown }).error
        : undefined;
    throw new Error(typeof detail === 'string' ? detail : `HTTP ${res.status}`);
  }
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

export const api = {
  tree: () => getJson<TreeResponse>('/api/tree'),
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
  stats: () => getJson<Record<string, unknown>>('/api/stats'),
  health: () => getJson<unknown[]>('/api/health'),
  check: () => getJson<Record<string, unknown> | null>('/api/check'),
  tensor: (name: string) => getJson<TensorInfo>(`/api/tensor?name=${enc(name)}`),
  layout: (file: string) => getJson<LayoutMap>(`/api/layout?file=${enc(file)}`),
  file: (path: string) =>
    getJson<{ path: string; name: string; size: number; truncated: boolean; text: string }>(
      `/api/file?path=${enc(path)}`,
    ),
  tensorStats: (name: string, dtype?: string) =>
    getJson<StatsDto>(`/api/tensor/stats?${qs({ name, dtype })}`),
  sample: (name: string, p: SampleParams) =>
    getJson<SampleDto>(`/api/tensor/sample?${qs({ name, ...p })}`),
  histogram: (name: string, bins?: number, dtype?: string) =>
    getJson<HistogramDto>(`/api/tensor/histogram?${qs({ name, bins, dtype })}`),
};
