// The one wait, and which step owns it. This file exists because the ordering is a judgement
// call with a wrong answer that is easy to ship: a checkpoint that failed to load must show its
// error, not a wait that never ends.

import { describe, expect, it } from 'vitest';
import { currentStep, resolvedSpec, shortSpec, stepDetail, stepLabel, stepSubject, type LoadInputs } from './loadstep';
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

describe('the three phases of a comparison', () => {
  const p = null;

  // Each phase names *whose* work it is, because they drag for unrelated reasons: a slow remote
  // checkpoint, a slow link, or this tab parsing 91 MB. The bar used to sit at 100% for the whole of
  // the third with a frozen timer, which read as a hang.
  it('names each phase and whose work it is', () => {
    expect(stepLabel({ kind: 'comparing', spec: '/base', right: '/newer', progress: p })).toBe(
      'reading both checkpoints',
    );
    expect(stepLabel({ kind: 'difftree', progress: p })).toBe('reading the comparison');
    expect(stepLabel({ kind: 'building', progress: p })).toBe('building the comparison');

    expect(stepDetail({ kind: 'comparing', spec: 's3://b/k', right: '/newer', progress: p })).toBe('S3 → this server');
    expect(stepDetail({ kind: 'difftree', progress: p })).toBe('this server → your browser');
    expect(stepDetail({ kind: 'building', progress: p })).toBe(
      'aligning both trees into rows, in this tab',
    );
  });

  // The baseline is the side actually being fetched, so it is the one worth naming — resolved, so a
  // `:` shorthand reads as the host it means.
  it('shows the baseline being read, as the address it resolves to', () => {
    expect(stepSubject({ kind: 'comparing', spec: ':/models/ckpt', right: '/newer', progress: p }, 'lab@host')).toBe(
      'lab@host:/models/ckpt',
    );
    // The two downloads have no single subject: both sides are already named on screen.
    expect(stepSubject({ kind: 'difftree', progress: p })).toBe('');
    expect(stepSubject({ kind: 'building', progress: p })).toBe('');
  });
});

// An address field is a thing you edit, and there the proxy host is the same 52 characters on every
// line — it pushes the part that differs out of the box. A *wait* is the other way round (see
// `resolvedSpec`): naming the machine is the point then.
describe('shortening a remote address for a box you type in', () => {
  const HOST = 'lab@build-host.example.com';

  it('drops the host when it is the configured proxy', () => {
    expect(shortSpec(`${HOST}:/opt/models/m`, HOST)).toBe(':/opt/models/m');
  });

  it('leaves another host alone — that one is not "the proxy"', () => {
    expect(shortSpec('other@elsewhere:/opt/models/m', HOST)).toBe('other@elsewhere:/opt/models/m');
  });

  it('leaves local paths, URIs and the shorthand itself alone', () => {
    expect(shortSpec('/net/models/m', HOST)).toBe('/net/models/m');
    expect(shortSpec('s3://bucket/key', HOST)).toBe('s3://bucket/key');
    expect(shortSpec(':/opt/models/m', HOST)).toBe(':/opt/models/m');
  });

  it('changes nothing when the server has no proxy to be short about', () => {
    expect(shortSpec(`${HOST}:/opt/models/m`)).toBe(`${HOST}:/opt/models/m`);
  });

  it('round-trips with resolvedSpec', () => {
    const full = `${HOST}:/opt/models/m`;
    expect(resolvedSpec(shortSpec(full, HOST), HOST)).toBe(full);
  });
});
