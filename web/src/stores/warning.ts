// Whether the no-access-control banner is showing.
//
// The condition it reports lasts as long as the server does, so the banner is not
// something you "acknowledge and move on" from — but it also shouldn't cost a row of
// screen for the whole session once you've read it. So: dismissible, persisted per
// browser, and always recoverable.
//
// Deliberately *not* dismissed-by-default and deliberately not forgotten on reload: the
// point is that whoever is looking at this page knows the server is open, and making them
// re-dismiss it every reload would only teach them to stop reading it.

import { writable } from 'svelte/store';

const KEY = 'ce-access-warning-dismissed';

function load(): boolean {
  try {
    return localStorage.getItem(KEY) === '1';
  } catch {
    // Private-mode / storage-disabled: show the banner rather than hide it on an error.
    return false;
  }
}

export const warningDismissed = writable<boolean>(load());

warningDismissed.subscribe((v) => {
  try {
    if (v) localStorage.setItem(KEY, '1');
    else localStorage.removeItem(KEY);
  } catch {
    // Nothing to do — the banner state is then per-page-load, which is safe.
  }
});
