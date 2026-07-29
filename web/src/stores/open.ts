// Switching the served checkpoint: the one action, in the one order.
//
// A third module rather than a function on either store, because the switch spans both —
// `server.ts` owns the fetched data, `view.ts` owns where you are in it, and `view.ts` already
// imports `server.ts`. Putting this in `server.ts` would make that a cycle.
//
// It lives here at all because two places offer it (the address bar in the header and the open
// screen) and the *order* below is load-bearing — see the comments. Two copies of an ordering
// this subtle is two chances to get it wrong.

import { openCheckpoint, reloadCheckpoint } from './server';
import { navigate, resetViewForNewCheckpoint } from './view';

/**
 * Open `spec` and land on its tree.
 *
 * Rejects with the server's message when the path doesn't resolve, having changed nothing: the
 * server only swaps after a successful read, and nothing here is discarded until that
 * succeeded. Whatever you were reading is still on screen.
 */
export async function switchCheckpoint(spec: string): Promise<string> {
  const root = await openCheckpoint(spec);
  // 1. Reset the view *before* the new tree can arrive. The fold state is seeded by a
  //    subscription that fires once per checkpoint, so a tree that lands before this flag is
  //    cleared gets no initial expansion — the second checkpoint would open fully collapsed
  //    where the first opened expanded.
  resetViewForNewCheckpoint();
  // 2. Be on the tree while it loads, so the wait is shown where the content will appear.
  navigate({ kind: 'tree' });
  // 3. Then drop the caches and fetch.
  await reloadCheckpoint();
  return root;
}
