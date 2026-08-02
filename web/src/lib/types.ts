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

/** What the source supports — `crate::capability::Capabilities`, served so the client asks
 * one question instead of re-deriving availability from the source's shape. That
 * re-derivation is the bug the Rust module exists to prevent, and it is just as wrong here:
 * a Hub repo and an ssh-proxied directory both lack byte access for different reasons. */
export interface Capabilities {
  /** Tensor bytes are reachable — the heatmap, value grid, histogram and whole-tensor
   * scan. False for every non-local source today. */
  read_bytes: boolean;
  modify_in_place: boolean;
  repack: boolean;
  layout_map: boolean;
  browse_files: boolean;
  object_metadata: boolean;
  codec_info: boolean;
  reach: 'direct' | 'via_ssh_proxy';
}

export interface TreeResponse {
  /** The checkpoint's display root — a directory, even for a single-file checkpoint. */
  root: string;
  /** What to *address* this checkpoint by: what was opened, in a spelling that reopens it.
   * Distinct from `root`, which for a single file is its containing directory — and that
   * directory can hold several checkpoints. */
  spec: string;
  tensor_count: number;
  tree: TreeNode[];
  /** What this source can do — see [[Capabilities]]. */
  capabilities: Capabilities;
  format: string;
  location: string;
  /** Why the data views are unavailable, or null when they aren't. The server's own
   * sentence, so the disabled pane says what a 400 from it would have said. */
  data_view_note: string | null;
  /** `source_path`s of tensors on disk that `model.safetensors.index.json` doesn't
   * list. Sent with the tree because that's where they're marked, and keyed on
   * `source_path` so the test here is the same one the TUI makes. */
  unindexed: string[];
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
      /** Hardlinked files under here — how much of the directory shares its bytes.
       * 0 where unknown, which is every remote source (`st_nlink` needs a local stat). */
      hardlinked: number;
      children: FileNode[];
    }
  | {
      kind: 'file';
      name: string;
      path: string;
      size: number;
      file_kind: string;
      /** What the model reads out of this file; null for anything that isn't a shard. */
      shard: ShardTensors | null;
      /** This file's size as a fraction of the largest file in the tree, for the
       * proportional bar. Served rather than computed here so the bar means the same
       * thing in the terminal, and doesn't rescale as the tree is folded. */
      size_share: number;
      /** Whether `model.safetensors.index.json` declares this file; null when the
       * question can't apply (not a checkpoint file, or there is no index). */
      index: 'listed' | 'unlisted' | null;
      /** Names this file's bytes have (`st_nlink`). `>1` means hardlinked, so the size
       * is shared with another name rather than this file's own; `1` for an ordinary
       * file and for every remote source, which can't count names. */
      links: number;
      /** Why this file's header wouldn't parse, when it wouldn't — the read carried on
       * without it, so its tensors are absent from the tree and the stats. */
      read_error: string | null;
    };

/** A shard's contribution to the model (`filetree::ShardTensors`). */
export interface ShardTensors {
  tensors: number;
  params: number;
  /** Fraction of the checkpoint's parameters, 0–1. Served rather than divided here, so
   * the browser and the terminal cannot disagree about it. */
  params_share: number;
}

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

/** A tensor's comparable signature, as `diff::TensorSig` serializes it. */
export interface TensorSig {
  dtype: string;
  shape: number[];
}

/** One tensor present in both checkpoints but not identical. */
export interface TensorChange {
  name: string;
  old: TensorSig;
  new: TensorSig;
}

/** One metadata entry present in both but with a different value. */
export interface MetaChange {
  name: string;
  old: { value: string };
  new: { value: string };
}

/** The structural diff `diff::DiffReport` serializes. Added/removed entries are
 * `[name, value]` pairs (serde's representation of the Rust tuple). */
export interface DiffReport {
  tensors_added: [string, TensorSig][];
  tensors_removed: [string, TensorSig][];
  tensors_changed: TensorChange[];
  tensors_unchanged: number;
  meta_added: [string, { value: string }][];
  meta_removed: [string, { value: string }][];
  meta_changed: MetaChange[];
  meta_unchanged: number;
  old_bytes: number;
  new_bytes: number;
  old_params: number;
  new_params: number;
  /** Each side's last-modified — the newest object under the prefix. `null` unless both sides are
   * `s3://`, which is the only pair that has per-object timestamps. */
  old_modified: string | null;
  new_modified: string | null;
}

/** `/api/diff`'s envelope: the baseline it compared against, the shared one-line
 * verdict, the equivalent CLI command, and the report. */
export interface DiffResponse {
  against: string;
  /** Which way round the report reads: `true` means the open checkpoint is the baseline (`?swap=1`).
   * Answered by the server so the view labels the report it actually got, not the one it asked for. */
  swapped: boolean;
  verdict: string;
  /** The `diff` invocation that reproduces this report, scope included. */
  command: string;
  /** What a scope selected, as the CLI's `matched M of N`; `null` when nothing narrowed it. */
  matched: { selected: number; total: number; names: string[] } | null;
  /** Names two rename rules mapped onto one, so a tensor silently left the comparison. */
  rename_collisions: string[];
  /**
   * Why the metadata was *not* compared (`filtered subset` / `--only-tensors`), or `null` when it was.
   *
   * From the server, because the rule is the server's: any filter suppresses the metadata comparison,
   * not only `--only-tensors`. Without it an empty section reads as "nothing differs" — the opposite
   * of what happened.
   */
  metadata_note: string | null;
  /** What the S3 object comparison did, or why it did not happen. `null` for a non-`s3://` pair. */
  s3_note: string | null;
  /** The S3 object section as the terminal prints it. `null` unless both sides are `s3://`. */
  s3_lines: S3Line[] | null;
  /** `modified: OLD → NEW`, humanised server-side. `null` unless both sides carry timestamps. */
  modified_line: string | null;
  /** What to call the two totals lines. Under a filter they cover the *matched* tensors, so they read
   * `size (filtered subset)` — worded by the server, which owns the rule. */
  totals_labels: TotalsLabels;
  /**
   * What an unfused/fused alignment folded: `name → [old parts, new parts]`.
   *
   * `×256 → ×1` on a row means 256 per-expert tensors on the old side correspond to the one fused
   * tensor on the new side — the answer to "did the conversion keep every expert", on the row that
   * compares them. Empty unless `align_fused=1` folded something.
   */
  folded: Record<string, [number, number]>;
  /** Whether the unfused/fused alignment is in force. */
  aligns_fused: boolean;
  /** The same tensor sections with index-templated families collapsed onto one row each — what the
   * terminal prints by default. Grouped by the server because the templating rule lives there and
   * driving the terminal's output; a second implementation here would be a second answer. */
  grouped: GroupedReport;
  report: DiffReport;
}

/** One row of the grouped report: a name template and how many tensors it stands for. */
export interface GroupedEntry {
  /** `model.layers.{0-61}.inv_freq_default` — ranges filled in. */
  name: string;
  count: number;
  sig: TensorSig;
}

/** One grouped *changed* row. `fold` is `[old parts, new parts]` when an alignment folded them all. */
export interface GroupedChange {
  name: string;
  count: number;
  old: TensorSig;
  new: TensorSig;
  fold: [number, number] | null;
}

export interface GroupedReport {
  tensors_added: GroupedEntry[];
  tensors_removed: GroupedEntry[];
  tensors_changed: GroupedChange[];
}

/** The headings for the `size:` / `params:` lines — see `DiffResponse.totals_labels`. */
export interface TotalsLabels {
  size: string;
  params: string;
}

/** One line of the S3 object section, to be styled by kind — the words come from Rust
 * (`S3Diff::summary_lines`), because what a matching multipart ETag proves is not a claim to make
 * twice. */
export interface S3Line {
  kind: 'heading' | 'removed' | 'added' | 'changed' | 'note';
  text: string;
}

/** Which of a family's attributes disagree across its members. */
export interface Varying {
  dtype: boolean;
  shape: boolean;
}

/** `/api/compact`: the tensor tree with uniform layer / expert stacks folded into one
 * templated subtree each. `tree` is the same `TreeNode` shape the tensor tree uses, so it
 * flattens with the same `flatten`; a leaf is a *family*, and `counts` says how many real
 * tensors it stands for (keyed by the leaf's `info.name`, which is the template). */
export interface CompactTree {
  tree: TreeNode[];
  counts: Record<string, number>;
  varying: Record<string, Varying>;
  tensor_count: number;
}
