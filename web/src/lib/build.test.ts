// Whether this tab is still the page the server serves. The rule has to be *conservative*: a false
// "reload me" trains people to ignore the true one, and the true one is the difference between a wrong
// answer and a correct one.

import { describe, expect, it } from 'vitest';
import { get } from 'svelte/store';
import { currentBuild, isStale, noteServedBuild, staleBuild } from './build';

describe('which build this tab is running', () => {
  it('is the hashed entry script it was loaded as', () => {
    expect(currentBuild('http://host:8080/assets/index-c1322f20.js')).toBe('index-c1322f20.js');
    // A cache-busting query is not part of the identity.
    expect(currentBuild('/assets/index-c1322f20.js?t=123')).toBe('index-c1322f20.js');
  });

  it('is nothing under a dev server, where there is no build to compare', () => {
    // `npm run dev` serves modules by source path; treating that as a build id would report every
    // dev session as permanently stale.
    expect(currentBuild('http://localhost:5173/src/lib/build.ts')).toBe('');
    expect(currentBuild('http://localhost:5173/@fs/x/y/main.ts')).toBe('');
    expect(currentBuild('')).toBe('');
  });
});

describe('deciding a tab is out of date', () => {
  it('says so only when both sides know their build and they differ', () => {
    expect(isStale('index-aaa.js', 'index-bbb.js')).toBe(true);
    expect(isStale('index-aaa.js', 'index-aaa.js')).toBe(false);
  });

  it('never cries wolf when either side cannot answer', () => {
    // A dev session (no id of its own), an older server that has no `/api/version`, and a server that
    // serves no hashed bundle: all unknown, none stale.
    expect(isStale('', 'index-bbb.js')).toBe(false);
    expect(isStale('index-aaa.js', null)).toBe(false);
    expect(isStale('index-aaa.js', undefined)).toBe(false);
    expect(isStale('index-aaa.js', '')).toBe(false);
  });
});

describe('what a response says about the build', () => {
  // Every response carries `X-App-Build`, so the first request a tab makes after the server was
  // reinstalled under it is the one that finds out — no poll, no request of its own.
  it('raises the flag for a build that is not this one, and never lowers it', () => {
    const mine = 'index-aaa.js';
    expect(get(staleBuild)).toBe(false);
    noteServedBuild(mine, mine);
    noteServedBuild(null, mine);
    expect(get(staleBuild), 'the same build, and no answer, are both fine').toBe(false);
    noteServedBuild('index-bbb.js', mine);
    expect(get(staleBuild)).toBe(true);
    // A later answer must not take it back: the tab is out of date until it reloads.
    noteServedBuild(mine, mine);
    noteServedBuild(null, mine);
    expect(get(staleBuild)).toBe(true);
  });

  // The dev server has no build to be stale against; saying otherwise would flag every session.
  it('says nothing when this tab has no build id', () => {
    expect(isStale('', 'index-bbb.js')).toBe(false);
  });
});
