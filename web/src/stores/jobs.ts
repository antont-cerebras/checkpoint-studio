// Long-running comparisons: start, poll, stop.
//
// The value-reading diff modes (`--values`, `--histogram`, `--tensor`, `--verify-repack`) read every
// selected tensor on both sides, so the server runs them as jobs and this polls. Polling rather than
// streaming was the deliberate choice — see `src/web/jobs.rs`: the job outlives the tab, so a reload
// picks a run back up instead of losing it.

import { get, writable } from 'svelte/store';
import { api } from '../lib/api';
import { scopeToQuery, type DiffScopeParams } from '../lib/diffscope';

/** What a poll reports. Mirrors `jobs::Job::snapshot`. */
export interface JobStatus {
  id: number;
  kind: string;
  state: 'running' | 'done' | 'cancelled' | 'failed';
  done: number;
  /** 0 until the work knows how many items there are — a spinner, not a bar at zero. */
  total: number;
  bytes: number;
  current: string;
  elapsed_s: number;
  findings: JobFinding[];
  error: string | null;
}

/** How one tensor's element values compare (`--values`). */
export interface ValueFinding {
  differing: number;
  elements: number;
  max_abs: number;
  mean_abs: number;
  nonfinite_mismatch: number;
}

/** How one tensor's distribution compares (`--histogram`). */
export interface HistFinding {
  tvd: number;
  bins: number;
}

/**
 * One finding. `kind` distinguishes the per-item results from the closing verdict.
 *
 * Spelled out rather than left as an index signature: a template cannot carry a TypeScript cast, so an
 * untyped bag forces `{@const x = f['k'] as T}` in the markup — which does not parse. Third time that
 * has bitten in this codebase, hence the fields.
 */
export interface JobFinding {
  kind: 'tensor' | 'verdict';
  name?: string;
  /** `--values` / `--histogram`, per tensor. */
  values?: ValueFinding | null;
  histogram?: HistFinding | null;
  /**
   * Why this tensor could **not** be compared — a shape the two sides do not share, a name one side
   * lacks, a fold the alignment made.
   *
   * Rendered, unlike when it was only in the payload: a run that compared nothing showed a list of bare
   * names under `0 of 0 compared tensor(s) differ`, which reads as "nothing differs" rather than as
   * "nothing happened".
   */
  error?: string;
  /** `--verify-repack`, per tensor: decoded indices rather than element values. */
  elements?: number;
  differing?: number;
  max_delta?: number;
  differing_gt1?: number;
  sparse_bad?: number;
  dense_bad?: number;
  /** The closing verdict — repack. */
  equivalent?: boolean;
  pairs?: number;
  bits?: number;
  /** How each side was decoded, from the server — `at 4-bit` is false of a `[3,4,4,4]` candidate. */
  packing?: string;
  other_differs?: boolean;
  /** The closing verdict — values / histogram. */
  compared?: number;
  differ?: number;
  verdict?: string;
}

/** The job on screen, or null. One at a time: the server reads one checkpoint at a time anyway. */
export const job = writable<JobStatus | null>(null);
export const jobError = writable<string>('');

/** How often to poll. Half a second is under the threshold where progress feels stalled, and costs one
 * small request — the trade named in the design. */
const POLL_MS = 500;

let timer: ReturnType<typeof setInterval> | undefined;

function stopPolling() {
  if (timer !== undefined) {
    clearInterval(timer);
    timer = undefined;
  }
}

/** Poll until the job stops running, then leave its final state on screen. */
function watch(id: number) {
  stopPolling();
  timer = setInterval(() => {
    void api
      .jobStatus(id)
      .then((s) => {
        job.set(s);
        // Stop polling once it has settled — but keep the result, which is what the reader came for.
        if (s.state !== 'running') stopPolling();
      })
      .catch((e: unknown) => {
        // A job evicted from the registry, or a server restart. Reported rather than polled forever.
        stopPolling();
        jobError.set(e instanceof Error ? e.message : String(e));
      });
  }, POLL_MS);
}

/** The modes a run can be, as the UI offers them. */
export type JobKind = 'values' | 'histogram' | 'verify-repack';

/**
 * Start a run and begin polling.
 *
 * `scope` is the same selection the structural views use, so "these nineteen tensors" means one thing
 * across the screen — and reading 117k tensors when nineteen were asked for is the mistake this
 * prevents.
 */
export async function startJob(
  kind: JobKind,
  left: string,
  right: string,
  scope: DiffScopeParams | undefined,
  tensor?: string,
): Promise<void> {
  stopPolling();
  jobError.set('');
  job.set(null);
  const params: [string, string][] = [
    ['left', left],
    ['right', right],
    ...(scope ? scopeToQuery(scope) : []),
  ];
  if (kind === 'values') params.push(['values', '1']);
  if (kind === 'histogram') params.push(['histogram', '1']);
  if (tensor) params.push(['tensor', tensor]);
  try {
    const started = await api.startJob(kind === 'verify-repack' ? 'verify-repack' : 'values', params);
    watch(started.id);
  } catch (e) {
    jobError.set(e instanceof Error ? e.message : String(e));
  }
}

/** Ask the running job to stop. Cooperative, so the state becomes `cancelled` when it notices. */
export async function cancelJob(): Promise<void> {
  const current = get(job);
  if (!current) return;
  try {
    job.set(await api.cancelJob(current.id));
  } catch (e) {
    jobError.set(e instanceof Error ? e.message : String(e));
  }
}

/** Drop the panel's contents, leaving any server-side run alone. */
export function clearJob(): void {
  stopPolling();
  job.set(null);
  jobError.set('');
}
