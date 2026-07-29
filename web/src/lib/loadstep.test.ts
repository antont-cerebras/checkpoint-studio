// The one wait, and which step owns it. This file exists because the ordering is a judgement
// call with a wrong answer that is easy to ship: a checkpoint that failed to load must show its
// error, not a wait that never ends.

import { describe, expect, it } from 'vitest';
import {
  currentStep,
  resolvedSpec,
  stepDetail,
  stepLabel,
  stepSubject,
  type LoadInputs,
} from './loadstep';
import { startedNow } from './progress';

const IDLE: LoadInputs = {
  opening: null,
  openingSpec: '',
  tree: null,
  haveTree: true,
  treeError: false,
  compact: false,
  haveCompact: false,
  folding: null,
  compactError: false,
};

describe('which step owns the screen', () => {
  it('shows nothing once there is content', () => {
    expect(currentStep(IDLE)).toBeNull();
  });

  it('puts an open in flight above everything else', () => {
    // While the server reads a new checkpoint, the tree still on screen belongs to the one being
    // replaced — so even a fully-loaded tree does not outrank this.
    const step = currentStep({ ...IDLE, opening: startedNow(), openingSpec: '/models/other' });
    expect(step).toMatchObject({ kind: 'opening', spec: '/models/other' });
  });

  it('waits for the tree when there is none', () => {
    expect(currentStep({ ...IDLE, haveTree: false })).toMatchObject({ kind: 'tree' });
  });

  it('shows no wait when the tree FAILED, so the error is reachable', () => {
    // The bug this pins: with the old `!$tree` test alone, a failed load looked like a load
    // still in progress and the error screen behind it was never reached.
    expect(currentStep({ ...IDLE, haveTree: false, treeError: true })).toBeNull();
  });

  it('waits for the fold only while the compact view is the one showing', () => {
    expect(currentStep({ ...IDLE, compact: true })).toMatchObject({ kind: 'folding' });
    // Not folding: the full tree is showing, so an unfetched compact tree is nobody's wait.
    expect(currentStep({ ...IDLE, compact: false, haveCompact: false })).toBeNull();
    // Landed.
    expect(currentStep({ ...IDLE, compact: true, haveCompact: true })).toBeNull();
    // Failed — the compact pane says so itself.
    expect(currentStep({ ...IDLE, compact: true, compactError: true })).toBeNull();
  });

  it('carries the progress through, so the bar and timer have something to read', () => {
    const p = startedNow();
    expect(currentStep({ ...IDLE, haveTree: false, tree: p })?.progress).toBe(p);
    expect(currentStep({ ...IDLE, compact: true, folding: p })?.progress).toBe(p);
  });
});

describe('what each step says', () => {
  const steps = [
    { kind: 'opening', spec: '/models/x', progress: null },
    { kind: 'tree', progress: null },
    { kind: 'folding', progress: null },
  ] as const;

  it('names the work in a distinct way per step', () => {
    const labels = steps.map(stepLabel);
    expect(new Set(labels).size).toBe(steps.length);
    for (const l of labels) expect(l).not.toBe('');
  });

  it('says which way the bytes are going, for each of the two downloads', () => {
    // Opening a Hub repo shows both in a row, and they are not the same wait twice: one crosses
    // the server's network, the other crosses yours.
    expect(stepDetail({ kind: 'opening', spec: 'hf://owner/name', progress: null })).toBe(
      'Hugging Face → this server',
    );
    expect(stepDetail(steps[1])).toBe('this server → your browser');
    expect(stepDetail(steps[2])).toContain('index');
  });

  it.each([
    ['hf://owner/name', 'Hugging Face'],
    ['https://huggingface.co/owner/name', 'Hugging Face'],
    ['s3://bucket/prefix/', 'S3'],
    [':/opt/models/m', 'the ssh proxy'],
    ['lab@net004:/opt/models/m', 'lab@net004'],
    ['/models/local', "this server's disk"],
    ['./relative', "this server's disk"],
  ])('names %s as coming from %s', (spec, from) => {
    expect(stepDetail({ kind: 'opening', spec, progress: null })).toBe(`${from} → this server`);
  });

  it('names the checkpoint only while opening one', () => {
    expect(stepSubject(steps[0])).toBe('/models/x');
    expect(stepSubject(steps[1])).toBe('');
    expect(stepSubject(steps[2])).toBe('');
  });

  it('shows the `:` shorthand as the address it resolves to', () => {
    // `:/path` means "on whatever ssh_proxy names" — a fact about the *server's* config, which the
    // browser is told rather than guessing. Echoing the `:` back names the checkpoint only to
    // someone who already remembers what their config says.
    const step = { kind: 'opening', spec: ':/opt/models/Kimi', progress: null } as const;
    expect(stepSubject(step, 'lab@net004')).toBe('lab@net004:/opt/models/Kimi');
    expect(stepDetail(step, 'lab@net004')).toBe('lab@net004 → this server');
    // Without a known host it still says something true, just less specific.
    expect(stepSubject(step)).toBe(':/opt/models/Kimi');
    expect(stepDetail(step)).toBe('the ssh proxy → this server');
  });

  it.each([
    [':/opt/m', 'host', 'host:/opt/m'],
    ['/local/m', 'host', '/local/m'],
    ['hf://owner/name', 'host', 'hf://owner/name'],
    ['other@box:/opt/m', 'host', 'other@box:/opt/m'],
    [':/opt/m', '', ':/opt/m'],
  ])('resolves %s with proxy %s to %s', (spec, host, want) => {
    expect(resolvedSpec(spec, host)).toBe(want);
  });
});
