// TypeScript mirrors of the JSON the Rust `--web` server sends. Loose `any` is
// used for the deeply-nested report objects (stats/check) the UI renders generically.

export interface TensorInfo {
  name: string;
  dtype: string;
  shape: number[];
  size_bytes: number;
  num_elements: number;
  storage: unknown;
  source_path: string;
  layout: unknown;
}

export interface MetadataInfo {
  name: string;
  value: string;
  value_type: string;
}

export type TreeNode =
  | {
      kind: 'group';
      name: string;
      children: TreeNode[];
      expanded: boolean;
      tensor_count: number;
      params: number;
      total_size: number;
      stored_size: number;
    }
  | { kind: 'tensor'; info: TensorInfo; label: string | null }
  | { kind: 'metadata'; info: MetadataInfo };

export interface TreeResponse {
  root: string;
  tensor_count: number;
  tree: TreeNode[];
  /** The server's no-access-control caution, or null when it is bound to loopback and so
   * only reachable from the machine it runs on. The server sends the same sentence it
   * prints to its terminal at startup. */
  access_warning: string | null;
}

export type FileNode =
  | {
      kind: 'dir';
      name: string;
      path: string;
      size: number;
      files: number;
      children: FileNode[];
    }
  | { kind: 'file'; name: string; path: string; size: number; file_kind: string };

export interface StatsDto {
  count: number;
  min: number;
  max: number;
  mean: number;
  std: number;
  zeros: number;
  nonfinite: number;
  zero_fraction: number;
  elapsed_ms: number;
}

export interface SampleDto {
  rows: number[];
  cols: number[];
  values: number[][];
  min: number;
  max: number;
  total_rows: number;
  total_cols: number;
  slices: number;
  slice: number;
  display_shape: number[];
  view: string;
  mode: string;
  overridable: boolean;
  /** Whether the values are integers, and if so signed. JSON numbers are f64 and
   * cannot carry a 64-bit integer exactly, so an integer view's decimal cells are
   * formatted from `raw` via BigInt — `values` would round past 2^53. */
  integer: boolean;
  signed: boolean;
  /** Raw stored bits per cell as zero-padded hex (always for an integer view, else
   * only when ?raw=1); width in `raw_width`. */
  raw_width?: number;
  raw?: string[][];
}

export type HistBins =
  | { type: 'int'; start: number; step: number }
  | { type: 'range'; lo: number; hi: number };

export interface HistogramDto {
  bins: HistBins;
  counts: number[];
  total: number;
  nonfinite: number;
  elapsed_ms: number;
}

export type SegmentKind =
  | { kind: 'header' }
  | { kind: 'tensor'; dtype: string; shape: number[] }
  | { kind: 'gap' };

export interface Segment {
  name: string;
  start: number;
  end: number;
  kind: SegmentKind;
}

export interface LayoutMap {
  name: string;
  total_len: number;
  header_len: number;
  tensor_count: number;
  metadata: [string, string][];
  segments: Segment[];
}
