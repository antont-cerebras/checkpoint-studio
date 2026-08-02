/**
 * What the app is doing while there is no checkpoint content to show.
 *
 * There used to be three of these, in three components, each with its own wording and its own
 * idea of what a wait looks like: `reading checkpoint structure` (the initial load),
 * `reading the checkpoint` (an open in flight) and `folding the tree` (the compact view). Three
 * screens for one situation — *waiting for this checkpoint* — and the differences between them
 * were accidents of where the code lived, not distinctions worth drawing.
 *
 * So there is one screen now, and the thing that varies is this value: which step, and what it
 * is working on. Naming the step is the point — "loading…" tells you nothing about whether the
 * server is reading sixteen shard headers or the browser is pulling 13 MB of JSON, and those
 * take very different amounts of time for very different reasons.
 */

import type { Progress } from './progress';

/** A distinguishable phase of getting a checkpoint on screen. */
export type LoadStep =
  /** The server is reading the checkpoint's shard headers, before it can answer anything. */
  | { kind: 'opening'; spec: string; progress: Progress | null }
  /** The tensor tree is downloading — the one wait with a real byte total. */
  | { kind: 'tree'; progress: Progress | null }
  /** Uniform layers and experts are being folded into families. */
  | { kind: 'folding'; progress: Progress | null }
  /**
   * The server is reading a comparison's **two** checkpoints, before it can align them.
   *
   * Both specs, not one: they are read one after another and at wildly different speeds — a local
   * directory in a second, an `s3://` prefix in twenty — so a single bar named after the baseline said
   * nothing about which of the two you were waiting for. The screen draws a row per side.
   */
  | { kind: 'comparing'; spec: string; right: string; progress: Progress | null }
  /** The aligned tree is downloading — the largest body this API serves. */
  | { kind: 'difftree'; progress: Progress | null }
  /**
   * The browser is turning that response into rows.
   *
   * Its own step because it is the *slowest* one for a large comparison and used to be invisible:
   * the bar sat at `87.0 MiB / 87.0 MiB · 100%` with a frozen timer for tens of seconds while the
   * main thread parsed 91 MB of JSON and walked 373k nodes. A bar at 100% that keeps going reads as
   * a hung program, which is exactly what it was mistaken for.
   */
  | { kind: 'building'; progress: Progress | null };

/** What the inputs to [`currentStep`] are — the stores it reads, as plain values. */
export interface LoadInputs {
  /** Non-null while `POST /api/open` is in flight. */
  opening: Progress | null;
  /** What that open is reading, for the label. */
  openingSpec: string;
  /** Non-null while `/api/tree` is downloading. */
  tree: Progress | null;
  /** Whether the tensor tree has landed. */
  haveTree: boolean;
  /** Whether the tree load *failed*. A failure is not a wait: without this, "no tree yet"
   * reads as "still loading" and the error screen behind it is never reached. */
  treeError: boolean;
  /** Whether the compact (folded) view is the one showing. */
  compact: boolean;
  /** Whether the compact tree has landed. */
  haveCompact: boolean;
  /** Non-null while the compact tree is being fetched. */
  folding: Progress | null;
  /** A failed fold shows its own message, not a wait that never ends. */
  compactError: boolean;
}

/**
 * The step to show, or `null` when there is content to show instead.
 *
 * Ordered by what the user is waiting on *most*: an open in flight outranks everything, because
 * until it lands the tree on screen belongs to the checkpoint being replaced. Then the tree
 * itself. Then the fold, which only matters once a tree exists.
 */
export function currentStep(i: LoadInputs): LoadStep | null {
  if (i.opening) return { kind: 'opening', spec: i.openingSpec, progress: i.opening };
  // An error outranks every wait: something that failed is not something to wait for.
  if (i.treeError) return null;
  if (!i.haveTree) return { kind: 'tree', progress: i.tree };
  if (i.compact && !i.haveCompact && !i.compactError) {
    return { kind: 'folding', progress: i.folding };
  }
  return null;
}

/** The step, said plainly — one line naming the work. */
export function stepLabel(s: LoadStep): string {
  switch (s.kind) {
    case 'opening':
      return 'reading shard headers';
    case 'tree':
      return 'reading the tensor tree';
    case 'folding':
      return 'folding uniform layers into families';
    case 'comparing':
      return 'reading both checkpoints';
    case 'difftree':
      return 'reading the comparison';
    case 'building':
      return 'building the comparison';
  }
}

/**
 * The detail under the label: for the two *downloads*, which way the bytes are going.
 *
 * Opening a Hub repo shows two waits in a row, and they are not the same wait twice: the first
 * pulls shard headers from Hugging Face **to the server**, over the server's network; the second
 * pulls the assembled list from the server **to the browser**, over yours. They fail and drag for
 * unrelated reasons, and which one you are watching decides what you would do about it — so each
 * says its endpoints rather than both saying "downloading".
 */
export function stepDetail(s: LoadStep, proxyHost = ''): string {
  switch (s.kind) {
    case 'opening':
      return `${sourceName(s.spec, proxyHost)} → this server`;
    case 'tree':
      return 'this server → your browser';
    case 'folding':
      return 'grouping tensors whose names differ only by an index';
    case 'comparing':
      return `${sourceName(s.spec, proxyHost)} → this server`;
    case 'difftree':
      return 'this server → your browser';
    case 'building':
      return 'aligning both trees into rows, in this tab';
  }
}

/**
 * Where an open is pulling from, named the way the person who typed the spec would name it.
 *
 * Derived from the spec rather than asked of the server, because this is shown *while* the open
 * is in flight — the server cannot answer about a checkpoint it has not finished reading.
 */
function sourceName(spec: string, proxyHost = ''): string {
  const s = spec.trim();
  if (s.startsWith('hf://') || s.startsWith('https://huggingface.co/')) return 'Hugging Face';
  if (s.startsWith('s3://')) return 'S3';
  // `:/path` is the configured ssh proxy; `[user@]host:/path` names its own host. A colon before
  // any slash is what distinguishes both from a local path (the same rule `split_scp` uses).
  const colon = s.indexOf(':');
  // Name the host when the server has told us which one it is — "the ssh proxy" is only as
  // informative as the reader's memory of their config file.
  if (colon === 0) return proxyHost || 'the ssh proxy';
  if (colon > 0 && !s.slice(0, colon).includes('/')) return s.slice(0, colon);
  return "this server's disk";
}

/**
 * The spec as the address it resolves to: `:/path` shown as `host:/path`.
 *
 * The `:` shorthand means "on whatever `ssh_proxy` names", which is a fact about the *server's*
 * config — so the browser is told the host (`/api/recents`) rather than guessing. Showing the
 * shorthand back while a checkpoint reads names it only to someone who already remembers what
 * their config says; the resolved form names it to anyone.
 */
export function resolvedSpec(spec: string, proxyHost = ''): string {
  const s = spec.trim();
  return proxyHost && s.startsWith(':') ? `${proxyHost}:${s.slice(1)}` : s;
}

/**
 * The inverse, for a box you type in: `host:/path` shown as `:/path` when the host **is** the
 * configured proxy.
 *
 * Two different jobs, which is why both exist. A *wait* names the machine being read, because
 * "which host is this coming from" is the question then. An *address field* is a thing you edit and
 * retype, and there the host is 52 characters of the same answer on every line — it pushes the part
 * that differs off the end of the box and out of the dropdown.
 *
 * Display only. What gets **stored** stays the scp form (`opening::recorded_spec`), because the
 * shorthand resolves against a config file that can change and a stored entry has to name the same
 * checkpoint later. The two forms resolve identically, so what is typed back is what was meant.
 */
export function shortSpec(spec: string, proxyHost = ''): string {
  const s = spec.trim();
  if (!proxyHost) return s;
  return s.startsWith(`${proxyHost}:`) ? s.slice(proxyHost.length) : s;
}

/** What the step applies to, when that isn't obvious: the checkpoint being opened, as the
 * address it resolves to. */
export function stepSubject(s: LoadStep, proxyHost = ''): string {
  return s.kind === 'opening' || s.kind === 'comparing' ? resolvedSpec(s.spec, proxyHost) : '';
}
