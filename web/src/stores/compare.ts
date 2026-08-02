// The two-checkpoint comparison: which baseline is set, the aligned tree, and where the cursor is
// among the differences.
//
// Separate from `server.ts` because a comparison is a *second* checkpoint's worth of state with its
// own lifecycle — set up, browsed, torn down — and folding it into the current-checkpoint store
// would leave every consumer of that store asking "which checkpoint is this about".

import { get, writable } from 'svelte/store';
import { api, BusyError } from '../lib/api';
import { recents, tree } from './server';

// A different served checkpoint means every cached comparison keyed on "whatever is served" is
// stale. Invalidating here rather than from `forgetCheckpoint` keeps the dependency one-way:
// `server.ts` knows nothing about comparisons.
let servedSpec: string | null = null;
tree.subscribe((t) => {
  const spec = t?.spec ?? null;
  if (servedSpec !== null && servedSpec !== spec) loadedPair = null;
  servedSpec = spec;
});
import { startedNow } from '../lib/progress';
import type { LoadStep } from '../lib/loadstep';
import { initialExpansion, type DiffTreeResponse } from '../lib/difftree';
import type { DiffScopeParams } from '../lib/diffscope';

/**
 * The comparison the server has set up: its id and the two specs it resolved.
 *
 * **One read, every view.** Establishing the pair (`POST /api/compare`) reads both checkpoints —
 * seconds each, minutes over an ssh proxy for an `s3://` side. The summary used to resolve and read
 * its own baseline while the tree read the pair again through this slot, so switching views re-read
 * both checkpoints and, worse, the summary compared the baseline against whatever the server happened
 * to have *open* rather than against the candidate: two views of "one comparison" describing two
 * different pairs. Both views quote this id now.
 */
export const comparison = writable<{ id: number; left: string; right: string } | null>(null);

/** The aligned tree, or null when no comparison is set up. */
export const diffTree = writable<DiffTreeResponse | null>(null);
export const diffError = writable<string>('');
/**
 * Set when the server refused because it was already reading — with what, and for how long.
 *
 * Its own store rather than text in `diffError`, because the useful response is an *action*: this is
 * what lets the view offer "Stop it and compare anyway" instead of a sentence saying to wait, which on
 * a server that reads one checkpoint at a time was advice with nothing behind it.
 */
export const diffBusy = writable<{ spec: string; seconds: number } | null>(null);
/**
 * Which phase of loading a comparison is running, with its progress; null when none is.
 *
 * A `LoadStep`, the same type the checkpoint load uses, rather than a bare `Progress`: the three
 * phases here (the server reading both checkpoints, the aligned tree downloading, this tab building
 * rows from it) drag for unrelated reasons and take very different amounts of time. A single byte
 * counter could only describe the middle one — which is why the bar used to sit at 100% with a
 * frozen timer for the whole of the third and read as a hang.
 */
export const diffStep = writable<LoadStep | null>(null);
/** Which groups are unfolded. Shared by both columns — that is what makes them move together. */
export const diffExpanded = writable<Set<string>>(new Set());
/** The row the cursor is on, by path; `n`/`N` move it between differences. */
export const diffCursor = writable<string | null>(null);
/**
 * Which pair is loaded — so returning to the screen does not re-read it.
 *
 * A plain `let`, like every other load-bookkeeping flag in this app (`treeStarted`, `compactSeq`,
 * `seededExpand`): nothing renders from it, and a store here would advertise reactivity it does not
 * have and invite a component to couple to a private cache key.
 */
let loadedPair: string | null = null;
/**
 * The pair being read *right now*.
 *
 * Without it the same comparison could be asked for twice before the first landed — a reactive
 * statement re-running, a scope applied, a remount — and the second request found the server busy
 * with the *first one*. The offer that followed ("stop that read and compare these") therefore
 * proposed stopping the very read that was fetching what was being asked for, and accepting it threw
 * the work away and started it again. Asking twice is now a no-op instead.
 */
let inFlightPair: string | null = null;
/** The pair the server has set up, and the one being set up — see [[establishComparison]]. */
let establishedPair = '';
let establishing = '';
/** The set-up in flight, so a second view awaits it rather than starting another. */
let inFlightSetup: Promise<void> | undefined;
/**
 * Two counters, because there are two things to supersede and they have different lifetimes.
 *
 * `pairSeq` guards *setting the pair up* (the POST that reads both checkpoints); `treeSeq` guards
 * *fetching the aligned tree* from a pair already set up. One counter meaning both broke this twice:
 * a view asking for a tree bumped the counter and thereby told the set-up it was awaiting to discard
 * its own answer — so the pair never published, the tree never loaded, and the page stayed busy.
 */
let pairSeq = 0;
let treeSeq = 0;
/** Whether a tree fetch is in progress, so the set-up does not take the progress line down under it. */
let fetchingTree = false;
/**
 * Aborts the aligned tree's download when a comparison is cancelled or superseded.
 *
 * Superseding alone is not enough: this body is the largest the API serves (91 MB for two unrelated
 * checkpoints), so an abandoned request left to finish goes on using the link and the server's
 * memory for tens of seconds after nobody is waiting for it.
 */
let inFlight: AbortController | null = null;

/**
 * The cache key for a pair.
 *
 * The served checkpoint is part of it, because an empty `right` *means* "whatever is served" — so
 * the same URL denotes a different comparison once the served checkpoint changes, and a key that
 * ignored that kept a comparison against a checkpoint no longer loaded, with no way to refresh it.
 */
function slotKey(left: string, right: string): string {
  // The served checkpoint stands in for an empty `right`, because that is what the server does with
  // it — so the same URL denotes a different comparison once a different checkpoint is open.
  return `${left}\u0000${right || (get(tree)?.spec ?? '')}`;
}

function pairKey(c: Comparison): string {
  // The scope is part of the key: two different selections of one pair are two different comparisons,
  // and a key that ignored it would make narrowing a loaded comparison do nothing at all. So is `full`,
  // because the server folds the families — the same pair unfolded is a different tree.
  const sel = c.scope === undefined ? '' : JSON.stringify(c.scope);
  return `${slotKey(c.left, c.right ?? '')}\u0000${sel}\u0000${c.full ?? false ? 'full' : 'folded'}`;
}

/**
 * Which comparison to load — **named**, not positional.
 *
 * `compareAgainst(left, right, force, stopOther, scope, full)` was four trailing optionals, two of
 * them adjacent booleans, and it read the same at every call site whatever it meant. That is not a
 * style complaint: it is how two "try again" buttons came to re-run a comparison without its scope
 * and without its fold state, and how a swap re-keyed the cache to the wrong one. A field has to be
 * named to be omitted, so a missing one is visible in the diff that omits it.
 */
export interface Comparison {
  /** The baseline. */
  left: string;
  /** The candidate; empty (or absent) means the checkpoint the server has open. */
  right?: string;
  /** Re-run even if this exact comparison is already on screen — the Compare button and "try again". */
  force?: boolean;
  /**
   * Take the server's read slot, stopping whatever holds it.
   *
   * **On by default**, because asking for a comparison *is* the decision: the alternative was a refusal
   * ("the server is reading …; it reads one checkpoint at a time") with a button whose only sensible
   * answer was yes — a question between you and the thing you just asked for. The server still refuses
   * unless asked (`stop_other=1`), so the choice exists; this is a client that has already made it.
   */
  stopOther?: boolean;
  /** The selection to apply when the two trees are aligned. `undefined` for an unscoped comparison —
   * spelled out, because the callers hold exactly that (`exactOptionalPropertyTypes`). */
  scope?: DiffScopeParams | undefined;
  /** Every layer as its own row, rather than uniform families folded onto one each (`?full=1`). */
  full?: boolean;
}

/**
 * Set the comparison up on the server: read both checkpoints, and remember the pair by id.
 *
 * **Once per pair.** Every view of a comparison quotes this id — the summary, the aligned tree, the
 * data checks — so the two checkpoints are read once however many ways they are then read. They used
 * to be read per view, which on an ssh proxy is minutes each time the reader changed tab.
 *
 * Idempotent, and safe to call from two places at once: a second caller for the same pair waits on
 * the first rather than starting another.
 */
export async function establishComparison(c: Comparison): Promise<void> {
  await ensurePair(c);
  // Summary and Data set up only the pair; unlike Browse they have no tree fetch to take the progress
  // line down afterwards, and leaving it up disabled the Compare button, the checkpoint boxes and
  // every scope control for good. Not while a tree *is* being fetched: that wait is still running.
  if (!fetchingTree) diffStep.set(null);
}

/**
 * Make sure the server holds this pair, and say whether it does.
 *
 * Supersede-safe on its own counter: a set-up that finishes after the reader has asked for a
 * different pair does not publish. It is deliberately *not* the counter the tree fetch uses — a view
 * asking for a tree must not tell the set-up it is waiting for to throw its answer away.
 */
async function ensurePair(c: Comparison): Promise<boolean> {
  const left = c.left;
  const right = c.right ?? '';
  const force = c.force ?? false;
  if (!left) return false;
  const pair = slotKey(left, right);
  if (!force && establishedPair === pair && get(comparison) !== null) return true;
  // Being set up by someone else — wait for it rather than reading the same pair twice.
  if (!force && establishing === pair && inFlightSetup !== undefined) {
    await inFlightSetup;
    return get(comparison) !== null;
  }
  establishing = pair;
  const seq = ++pairSeq;
  diffError.set('');
  diffBusy.set(null);
  // Both sides are read one after another, at speeds that can differ by a factor of twenty, and the
  // screen draws a row for each.
  diffStep.set({
    kind: 'comparing',
    spec: left,
    right: right || get(tree)?.spec || '',
    progress: startedNow(),
  });
  const run = (async () => {
    try {
      // Both sides of a comparison are checkpoints you have now opened, so the server records them.
      // Take its list back rather than leaving ours stale: without this the paths you just compared
      // were missing from the pickers until something else happened to refetch them.
      const set = await api.setComparison(left, right, c.stopOther ?? true);
      if (seq !== pairSeq) return;
      recents.set(set.recents);
      comparison.set({ id: set.id, left: set.left, right: set.right });
      establishedPair = pair;
      // A fresh read of the pair invalidates everything derived from the old one. The report is keyed
      // by the comparison id and so re-fetches by itself; the tree is keyed by the *pair*, which has
      // not changed — so pressing Compare on an unchanged pair would re-read both checkpoints and go
      // on showing the tree from before them.
      loadedPair = null;
    } catch (e) {
      if (seq !== pairSeq) return;
      comparison.set(null);
      establishedPair = '';
      // The server reads one checkpoint at a time and said so. Recorded as a fact rather than as
      // prose, so the view can offer to stop that read instead of telling the reader to wait.
      if (e instanceof BusyError) {
        diffBusy.set({ spec: e.busyWith, seconds: e.busyForSeconds });
      } else {
        diffError.set(e instanceof Error ? e.message : String(e));
      }
    } finally {
      if (seq === pairSeq) establishing = '';
    }
  })();
  inFlightSetup = run;
  await run;
  return get(comparison) !== null;
}

/**
 * Set the baseline and load the aligned tree.
 *
 * Two requests, because they are two different waits: reading a checkpoint (seconds, on the server)
 * and downloading the tree (the largest body this API serves). Reporting them as one would make a
 * slow disk and a slow link look like the same problem.
 */
export async function compareAgainst(c: Comparison): Promise<void> {
  const { left, scope } = c;
  const right = c.right ?? '';
  const force = c.force ?? false;
  const stopOther = c.stopOther ?? true;
  const full = c.full ?? false;
  // Already showing this pair? Then leave it alone.
  //
  // Leaving the screen and coming back — opening a tensor's detail and pressing Back — remounts the
  // component, and re-running the comparison meant re-reading both checkpoints (seconds) and losing
  // the fold state and the cursor. Nothing about the pair changed, so there is nothing to redo.
  //
  // `force` is how the Compare button re-runs the *same* pair, which is the only way to pick up
  // checkpoints that changed on disk — and the only thing that made the button work again after
  // Stop, since navigating to an unchanged URL fires nothing.
  const pair = pairKey(c);
  if (!force && loadedPair === pair && get(diffTree) !== null) return;
  // Already being read — see `inFlightPair`. `force` still re-runs it, which is what the Compare
  // button and "try again" are for.
  if (!force && inFlightPair === pair) return;
  inFlightPair = pair;
  const seq = ++treeSeq;
  fetchingTree = true;
  // Whatever was still coming is no longer wanted.
  inFlight?.abort();
  const abort = new AbortController();
  inFlight = abort;
  diffError.set('');
  diffBusy.set(null);
  /**
   * Announce a read only when there is one.
   *
   * Switching from the summary to the tree does not re-read anything — the pair is already in the
   * server's slot — but this set the *reading both checkpoints* step regardless, so Browse opened on
   * a screen naming two checkpoints as `reading…` and stayed there for as long as the server took to
   * align them and send the tree. Nothing on it was true. When the pair is already established the
   * wait starts where it actually is: the comparison coming over the wire.
   */
  const reading = force || establishedPair !== slotKey(left, right) || get(comparison) === null;
  // Phase one, when there is one: the server reads both checkpoints, one after the other, at speeds
  // that can differ by a factor of twenty — so the screen draws a row for each.
  diffStep.set(
    reading
      ? {
          kind: 'comparing',
          spec: left,
          right: right || get(tree)?.spec || '',
          progress: startedNow(),
        }
      : { kind: 'difftree', progress: startedNow() },
  );
  try {
    // The pair, set up once and shared with every other view of it (see `establishComparison`).
    // Reading two checkpoints to draw a second view of the same comparison is the cost this split
    // exists to avoid.
    const ready = await ensurePair({ left, right, force, stopOther });
    if (seq !== treeSeq) return; // superseded while the server was reading
    // The set-up failed and has already said why; there is no pair to align.
    const set = get(comparison);
    if (!ready || set === null) return;
    const startedAt = performance.now();
    const t = await api.difftree(
      // The comparison this client set up, by id. There is one slot on the server, so without this
      // two overlapping clients received each other's trees with a 200; quoting it makes a lost race
      // a 409 that says so.
      set.id,
      // The selection, applied when the trees are aligned — so changing it re-aligns without either
      // checkpoint being read again.
      scope,
      // Phase two: the aligned tree, over the wire, with a real byte total.
      (received, total) =>
        diffStep.set({ kind: 'difftree', progress: { received, total, startedAt } }),
      // Phase three: the last byte has landed and this tab is about to spend tens of seconds
      // parsing it. A fresh timer, because it is a different wait, not more of the same one.
      () => diffStep.set({ kind: 'building', progress: startedNow() }),
      abort.signal,
      full,
    );
    // Supersede check *before* publishing: a comparison still in flight when you navigate back
    // would otherwise land on top of the newer one and leave the view describing a pair the URL no
    // longer names, with nothing left to re-fire.
    if (seq !== treeSeq) return;
    // And check the server answered about the pair we asked for.
    //
    // Belt and braces over the id: the id makes a swapped comparison a 409 rather than a 200, and this
    // catches anything that would still get through — a proxy serving a cached body, an id reused
    // after a restart. Compared against the specs the *server* resolved (`set.left`/`set.right`), not
    // the ones typed, because only the server expands `:/p` to `host:/p` or a glob to its directory,
    // and comparing against the typed form would reject correct answers.
    if (t.base.spec !== set.left || t.current.spec !== set.right) {
      throw new Error(
        `the server answered about a different comparison (${t.base.spec} ↔ ${t.current.spec}, ` +
          `asked for ${set.left} ↔ ${set.right}) — try again`,
      );
    }
    diffTree.set(t);
    // Open the way to every difference, and nothing else: a change three groups deep is no use
    // folded, and expanding everything would bury it in unchanged rows. Above `REVEAL_LIMIT` that
    // reasoning inverts — see `initialExpansion` — and the tree arrives folded with per-group counts.
    diffExpanded.set(initialExpansion(t.rows, t.differences.length));
    diffCursor.set(t.differences[0] ?? null);
    loadedPair = pair;
  } catch (e) {
    if (seq !== treeSeq) return;
    diffTree.set(null);
    loadedPair = null;
    // An abort is this app's own doing, not a failure to report: `cancelComparison` bumps the
    // sequence first, so the guard above catches the common case, and this covers a signal that
    // fires without one (a navigation away mid-download).
    if (e instanceof DOMException && e.name === 'AbortError') return;
    // No `BusyError` arm here: "the server is reading something else" can only come from the POST
    // that sets the pair up, which `ensurePair` owns — this is the GET that fetches the tree of a
    // pair the server already holds.
    diffError.set(e instanceof Error ? e.message : String(e));
  } finally {
    if (seq === treeSeq) {
      fetchingTree = false;
      diffStep.set(null);
      inFlight = null;
      inFlightPair = null;
    }
  }
}

/**
 * Abort a comparison that is still loading, leaving the boxes as they are.
 *
 * The other half of what one button used to do. While a comparison is in flight there is nothing to
 * clear — and no button was even rendered, so a mistyped baseline meant watching a 91 MB download to
 * the end. Cancelling keeps both paths in place, which is the point: the usual reason to stop is to
 * fix one character and go again.
 */
export function cancelComparison(): void {
  // Cleared here as well as in `finally`: a cancel bumps the sequence, so that block declines to touch
  // anything — and a pair left marked in-flight would swallow the next identical request.
  inFlightPair = null;
  // Bump first: the abort rejects the fetch, and the guard on `seq` is what stops that rejection
  // from being reported as a failure.
  pairSeq += 1;
  treeSeq += 1;
  inFlight?.abort();
  inFlight = null;
  diffStep.set(null);
  diffError.set('');
}

// No `swapComparison` / `noteSwapped` here any more, and that is the point.
//
// Flipping used to mutate the loaded tree *and* rewrite the URL's operands, which meant the cache key
// had to be corrected afterwards — and, worse, that the URL described a pair whose directional scope
// (`--map` rewrites the baseline; a `#subtree` belongs to one side) no longer matched what was on
// screen: reloading it asked the server for a different comparison. The pair the server is asked
// about is now always the canonical one, and which way round it is *read* is view state the URL
// carries as `swap=1` — so a flip touches nothing here, and the reload of any link reproduces exactly
// what its sender saw (`difftree::swapResponse`).

/**
 * Tear the comparison down, freeing the baseline on the server.
 *
 * Distinct from [`cancelComparison`]: this discards a *result*, and the caller resets the URL with
 * it — the address bar naming a comparison that is no longer on screen was the confusing part, since
 * a reload then brought back what had just been cleared.
 */
export async function stopComparing(): Promise<void> {
  // Bump the sequence too: a comparison still loading must not land after a Stop.
  pairSeq += 1;
  treeSeq += 1;
  inFlight?.abort();
  inFlight = null;
  diffStep.set(null);
  diffTree.set(null);
  comparison.set(null);
  loadedPair = null;
  diffError.set('');
  diffExpanded.set(new Set());
  diffCursor.set(null);
  try {
    await api.clearComparison();
  } catch {
    // The comparison is already gone from this client's point of view; a failed teardown only
    // means the server holds a checkpoint it will drop on the next comparison or restart.
  }
}
