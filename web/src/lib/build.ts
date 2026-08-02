// Which build of the UI this tab is running, and whether the server still serves it.
//
// A browser tab outlives the server it was loaded from, and this project restarts the server under open
// tabs as a matter of routine. What that produced was not an error but a *wrong answer*: a tab holding
// the previous build read the newer comparison shape, found every counter missing, summed them to `NaN`
// — and since `NaN > 0` is false, announced that two checkpoints sharing no tensor name at all were
// "structurally identical". Reading the counters defensively fixes that one symptom. This is the
// question behind it: is this page still the page the server is serving?
//
// The identity is Vite's content-hashed entry filename (`index-c1322f20.js`), because it changes exactly
// when the UI changes and both sides can see it without a build-time stamp to keep in step: the server
// reads it out of the `index.html` it serves, and the tab reads it off its own module URL.
//
// The *store* lives here, next to the two functions that decide what goes in it, so the API layer can
// set it from a response header without importing a store that imports the API layer.

import { writable } from 'svelte/store';

/** A built entry script, as Vite names it. Anything else is a dev server. */
const BUILT = /^index-[a-z0-9]+\.js$/;

/**
 * The build this tab is running, or `''` when there is nothing to compare.
 *
 * `import.meta.url` is this module's own URL. In a built bundle that is `/assets/index-<hash>.js`; under
 * `npm run dev` it is `/src/lib/build.ts`, which is not a build id and must not be treated as one —
 * a dev session would otherwise report itself permanently stale.
 */
export function currentBuild(url: string = import.meta.url): string {
  const name = url.split('?')[0]?.split('/').pop() ?? '';
  return BUILT.test(name) ? name : '';
}

/**
 * Whether the tab should be reloaded: both sides know their build, and they differ.
 *
 * Deliberately conservative. An unknown build on either side (a dev server, an older binary that does
 * not answer `/api/version`) is *not* stale — a false alarm telling someone to reload a page that is
 * fine is its own kind of wrong, and would train them to ignore the real one.
 */
export function isStale(mine: string, served: string | null | undefined): boolean {
  return mine !== '' && !!served && mine !== served;
}

/** True once the server is known to serve a different build of the UI than this tab is running. */
export const staleBuild = writable(false);

/**
 * Record the build the server just said it serves — from `X-App-Build`, on any response.
 *
 * **Never un-set.** A tab that has seen a newer build is out of date until it reloads, and a later
 * answer (the server restarting again, a response from a cache) must not clear the warning.
 *
 * `mine` is a parameter with a default for the same reason [[currentBuild]]'s `url` is: under a test
 * runner this module's own URL is a source path, so the real answer is always "no build to compare"
 * and the rule could not be exercised at all.
 */
export function noteServedBuild(served: string | null | undefined, mine = currentBuild()): void {
  if (isStale(mine, served)) staleBuild.set(true);
}
