<script lang="ts">
  import { onMount } from 'svelte';
  import { api } from '../lib/api';
  import { humanCount } from '../lib/format';
  import Spinner from './Spinner.svelte';
  import Ref from './Ref.svelte';

  interface Finding {
    severity: string;
    subject: string;
    message: string;
  }
  interface Check {
    id: string;
    title: string;
    /** "what passing means" — shown as the per-check hover explanation. */
    note: string;
    status: string;
    findings: Finding[];
  }
  interface CheckReport {
    /** Checkpoint format (safetensors / hdf5 / numpy / gguf / other) — gates
     * format-specific sections (index reconciliation is safetensors-only). */
    format: string;
    summary: { files: number; tensors: number; params: number; errors: number; warnings: number };
    checks: Check[];
    healthy: boolean;
  }
  interface Health {
    index_path: string;
    missing_files: string[];
    extra_files: string[];
    missing_tensors: string[];
    extra_tensors: string[];
    /** `s3://` only: the checkpoint index and the tensor's own object metadata
     * describe the same tensor and disagree. Absent on an older server. */
    mismatched_tensors?: string[];
    /** `s3://` only: tensors the cross-check could not verify either way. */
    unverified_tensors?: string[];
  }

  let check: CheckReport | null = null;
  let health: Health[] = [];
  let err = '';
  let loading = true;

  onMount(async () => {
    try {
      const [c, h] = await Promise.all([api.check(), api.health()]);
      check = c as unknown as CheckReport;
      health = (h as unknown as Health[]) ?? [];
    } catch (e) {
      err = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  });

  interface StatusInfo {
    icon: string;
    cls: string;
    label: string;
  }
  const NA: StatusInfo = { icon: '–', cls: 'na', label: 'n/a' };
  const STATUS: Record<string, StatusInfo> = {
    pass: { icon: '✓', cls: 'ok', label: 'pass' },
    warn: { icon: '⚠', cls: 'warn', label: 'warning' },
    fail: { icon: '✗', cls: 'fail', label: 'fail' },
    na: NA,
  };
  const SEV: Record<string, string> = { error: 'fail', warning: 'warn', info: 'na' };

  /** Badge for a check status / finding severity, falling back to n/a for anything the
   * server sends that we don't know — so an unrecognised value renders instead of
   * blowing up on an undefined lookup. */
  const statusOf = (s: string | undefined): StatusInfo => (s ? (STATUS[s] ?? NA) : NA);
  const sevOf = (s: string): StatusInfo => statusOf(SEV[s]);

  /** The four index-reconciliation lists for one shard, as typed tuples. */
  type IdxList = [heading: string, items: string[], cls: string, refKind: 'file' | 'tensor'];
  const idxLists = (h: Health): IdxList[] => [
    ['Missing files', h.missing_files, 'fail', 'file'],
    ['Extra files (on disk, not in index)', h.extra_files, 'warn', 'file'],
    ['Missing tensors', h.missing_tensors, 'fail', 'tensor'],
    ['Extra tensors', h.extra_tensors, 'warn', 'tensor'],
    ['Index disagrees with the object metadata', h.mismatched_tensors ?? [], 'fail', 'tensor'],
    ['Not cross-checked against the object metadata', h.unverified_tensors ?? [], 'warn', 'tensor'],
  ];

  // Findings-first: checks with problems on top, then passing, then n/a.
  const rank = (s: string) => (s === 'fail' ? 0 : s === 'warn' ? 1 : s === 'pass' ? 2 : 3);
  $: checks = check ? [...check.checks].sort((a, b) => rank(a.status) - rank(b.status)) : [];
  $: indexIssues = health.filter((h) => idxLists(h).some(([, items]) => items.length));
</script>

<div class="health">
  <div class="inner">
  {#if loading}
    <Spinner label="running checks…" />
  {:else if err}
    <p class="err">{err}</p>
  {:else if check}
    <!-- overall banner -->
    <div class="banner {check.healthy ? (check.summary.warnings ? 'warn' : 'ok') : 'fail'}">
      <span class="big">{check.healthy ? (check.summary.warnings ? '⚠' : '✓') : '✗'}</span>
      <span class="msg">
        {#if !check.healthy}{check.summary.errors} error{check.summary.errors === 1 ? '' : 's'}
        {:else if check.summary.warnings}{check.summary.warnings} warning{check.summary.warnings === 1 ? '' : 's'} — no errors
        {:else}Healthy — all checks passed{/if}
      </span>
      <span class="sub dim">
        {check.summary.files} files · {check.summary.tensors.toLocaleString()} tensors · {humanCount(check.summary.params)} params
      </span>
    </div>

    <!-- structural checks -->
    <section>
      <h3 title="Header-only checks (no tensor data read): each row explains what passing verifies — hover it.">Structural checks</h3>
      <ul class="checks">
        {#each checks as c (c.id)}
          {@const st = statusOf(c.status)}
          <li>
            <div class="checkhead">
              <span class="badge {st.cls}" title={st.label}>{st.icon}</span>
              <span class="ctitle" title={c.note}>{c.title}</span>
              <span class="what dim" title={c.note}>ⓘ</span>
              {#if c.findings.length}<span class="dim">· {c.findings.length}</span>{/if}
            </div>
            {#if c.findings.length}
              <ul class="findings">
                {#each c.findings as f, fi (fi)}
                  <li>
                    <span class="badge sm {sevOf(f.severity).cls}">{sevOf(f.severity).icon}</span>
                    <Ref name={f.subject} />
                    <span class="fmsg dim">{f.message}</span>
                  </li>
                {/each}
              </ul>
            {/if}
          </li>
        {/each}
      </ul>
    </section>

    <!-- index reconciliation — safetensors only (other formats have no index.json) -->
    {#if check.format === 'safetensors'}
    <section>
      <h3 title="Cross-checks model.safetensors.index.json against the shards on disk: files and tensors the index lists vs. what's actually present.">Index health</h3>
      {#if !health.length}
        <p class="dim">No <code>model.safetensors.index.json</code> to reconcile.</p>
      {:else if !indexIssues.length}
        <p class="ok-line"><span class="badge ok">✓</span> Index matches the files on disk.</p>
      {:else}
        {#each health as h (h.index_path)}
          {@const lists = idxLists(h)}
          {#each lists as [heading, items, cls, refKind] (heading)}
            {#if items.length}
              <div class="idxgroup">
                <div class="idxhead"><span class="badge sm {cls}">{cls === 'fail' ? '✗' : '⚠'}</span> {heading} <span class="dim">({items.length})</span></div>
                <ul class="idxlist">
                  {#each items as it (it)}<li class="mono"><Ref name={it} kind={refKind} /></li>{/each}
                </ul>
              </div>
            {/if}
          {/each}
        {/each}
      {/if}
    </section>
    {/if}
  {/if}
  </div>
</div>

<style>
  .health {
    height: 100%;
    overflow: auto;
  }
  .inner {
    max-width: 900px;
    margin: 0 auto;
    padding: 18px 22px;
  }
  .banner {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    border: 1px solid var(--border);
    border-left-width: 4px;
    border-radius: 8px;
    background: var(--bg-panel);
    margin-bottom: 22px;
  }
  .banner.ok {
    border-left-color: var(--ok);
  }
  .banner.warn {
    border-left-color: var(--warn);
  }
  .banner.fail {
    border-left-color: var(--danger);
  }
  .banner .big {
    font-size: 26px;
  }
  .banner.ok .big {
    color: var(--ok);
  }
  .banner.warn .big {
    color: var(--warn);
  }
  .banner.fail .big {
    color: var(--danger);
  }
  .banner .msg {
    font-size: 15px;
  }
  .banner .sub {
    margin-left: auto;
    font-size: 12px;
  }
  h3 {
    margin: 0 0 10px;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--fg-dim);
  }
  section {
    margin-bottom: 26px;
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .checks > li {
    padding: 8px 0;
    border-top: 1px solid var(--border);
  }
  .checkhead {
    display: flex;
    align-items: center;
    gap: 9px;
  }
  .ctitle {
    color: var(--fg);
  }
  .what {
    font-size: 11px;
    cursor: help;
  }
  .badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    border-radius: 4px;
    font-size: 12px;
    flex: 0 0 auto;
  }
  .badge.sm {
    width: 15px;
    height: 15px;
    font-size: 10px;
  }
  .badge.ok {
    background: color-mix(in srgb, var(--ok) 22%, transparent);
    color: var(--ok);
  }
  .badge.warn {
    background: color-mix(in srgb, var(--warn) 22%, transparent);
    color: var(--warn);
  }
  .badge.fail {
    background: color-mix(in srgb, var(--danger) 22%, transparent);
    color: var(--danger);
  }
  .badge.na {
    background: var(--bg-hover);
    color: var(--fg-dim);
  }
  .findings {
    margin: 6px 0 2px 27px;
  }
  .findings li {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 2px 0;
  }
  .fmsg {
    font-size: 12px;
  }
  .idxgroup {
    margin-bottom: 12px;
  }
  .idxhead {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }
  .idxlist {
    margin-left: 24px;
  }
  .idxlist li {
    padding: 1px 0;
    color: var(--fg-dim);
  }
  .ok-line {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .err {
    color: var(--danger);
  }
</style>
