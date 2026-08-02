// Has the server been rebuilt under this tab?
//
// Three chances to notice, because the answer only matters before the stale page is *used*:
//
//  - every response carries `X-App-Build`, so the first thing this tab asks the restarted server tells
//    it (`lib/build`'s `noteServedBuild`, called from `lib/api`) — no request of its own;
//  - `/api/version` when the app starts, and whenever the tab is brought back to the front, which is
//    the moment a tab that sat in the background is about to be used again;
//  - and a slow poll while the tab is *visible*, for the case that produced the report this exists
//    for: the page open and being watched while the server is reinstalled under it. Nothing was in
//    flight and nothing brought it to the front, so on the two triggers above it sat there silently
//    running an older interface. Visible-only, so a backgrounded tab asks nothing.

import { api } from '../lib/api';
import { currentBuild, isStale, staleBuild } from '../lib/build';

/** The store lives in `lib/build`, beside the rule that sets it; re-exported here because this is
 * where the rest of the app looks for it. */
export { staleBuild };

/** How often a visible tab re-asks. Slow: a build changes at most every few minutes even in this
 * project's workflow, and the response is under a hundred bytes. */
const POLL_MS = 15_000;

/** Ask once. Silent on failure: a failed check says nothing about the build — the server may simply be
 * restarting — and the request the user is waiting on reports for itself. */
export async function checkBuild(): Promise<void> {
  const mine = currentBuild();
  if (mine === '') return; // dev server: nothing to compare
  try {
    const served = await api.version();
    if (isStale(mine, served.assets)) staleBuild.set(true);
  } catch {
    // Nothing to say; see above.
  }
}

/**
 * Check now, whenever the tab is brought to the front, and slowly while it is in front.
 *
 * Returns the unsubscribe, for `onMount`.
 */
export function watchBuild(): () => void {
  let timer: ReturnType<typeof setInterval> | undefined;
  const start = () => {
    if (timer === undefined) timer = setInterval(() => void checkBuild(), POLL_MS);
  };
  const stop = () => {
    clearInterval(timer);
    timer = undefined;
  };
  void checkBuild();
  const onVisible = () => {
    if (document.visibilityState === 'visible') {
      void checkBuild();
      start();
    } else {
      stop();
    }
  };
  if (document.visibilityState === 'visible') start();
  document.addEventListener('visibilitychange', onVisible);
  return () => {
    stop();
    document.removeEventListener('visibilitychange', onVisible);
  };
}
