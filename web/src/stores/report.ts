// The one-page structural report — the *Summary* view's data, held where the whole page can see it.
//
// It used to be three `let`s inside the view that renders it, which was fine while that view was a
// screen of its own. It is one of three readings of a comparison now, and the others need what it
// knows: the Data view sizes its run from the totals this response carries (they follow the scope,
// so they are the honest "how much will be read"), and the page's swap has to know whether *any*
// result is on screen, not just an aligned tree.
//
// Two responses per comparison is deliberate, not an oversight: the report is a small categorised
// answer and the aligned tree is the largest body this API serves (91 MB on a real pair). Fetching
// the tree to show a summary would make the cheap view pay for the expensive one, so each view loads
// what it needs and both are cached by their own key.

import { writable } from 'svelte/store';

import { api } from '../lib/api';
import type { DiffScopeParams } from '../lib/diffscope';
import { startedNow, type Progress } from '../lib/progress';
import type { DiffResponse } from '../lib/types';

/** The report on screen, or `null` when none has landed. Named like `diffTree`, its
 * counterpart for the Browse view. */
export const diffReport = writable<DiffResponse | null>(null);
/** When a read is in flight, the timer for its progress line; `null` when idle. */
export const reportWait = writable<Progress | null>(null);
export const reportError = writable<string | null>(null);

/** What a given report is *of*, so an unchanged request is not re-run. */
let loadedKey = '';
/** Which request is the current one — a slower earlier answer must not land on a later one. */
let seq = 0;

function key(id: number, scope: DiffScopeParams | undefined, swapped: boolean, full: boolean) {
  const sel = scope === undefined ? '' : JSON.stringify(scope);
  return `${id} ${sel} ${swapped ? 's' : ''}${full ? 'f' : ''}`;
}

/**
 * Load the report for this comparison, unless it is already the one showing.
 *
 * `id` is the comparison the server has set up (`stores/compare`'s `comparison`), not a spec: the
 * pair is read once and every view quotes it. The rest is in the URL, and each changes the answer:
 * the selection, which way round the pair is read, and — because the *offered command* has to match
 * what is on screen — whether families are collapsed.
 */
export async function loadReport(
  id: number | null,
  scope: DiffScopeParams | undefined,
  swapped: boolean,
  full: boolean,
): Promise<void> {
  if (id === null) {
    diffReport.set(null);
    loadedKey = '';
    return;
  }
  const want = key(id, scope, swapped, full);
  if (want === loadedKey) return;
  const mine = ++seq;
  reportError.set(null);
  reportWait.set(startedNow());
  try {
    const answer = await api.diff(id, scope, swapped, full);
    // A superseded request must not publish: the reader has asked for something else since.
    if (mine !== seq) return;
    diffReport.set(answer);
    loadedKey = want;
  } catch (e) {
    if (mine !== seq) return;
    diffReport.set(null);
    loadedKey = '';
    reportError.set(e instanceof Error ? e.message : String(e));
  } finally {
    if (mine === seq) reportWait.set(null);
  }
}

/** Forget the report — the pair was cleared, so nothing on screen describes it any more. */
export function clearReport(): void {
  seq += 1;
  diffReport.set(null);
  reportError.set(null);
  reportWait.set(null);
  loadedKey = '';
}
