<script lang="ts">
  /**
   * Comparing two checkpoints: one screen, three ways of reading the result.
   *
   * **Why this exists.** There were two screens — a one-page report and an aligned side-by-side tree
   * — reached from two places (the palette's *Diff report*, the header's *Compare*). They took the
   * same pair and the same scope and answered the same question in two shapes, so the reader had to
   * choose a representation *before seeing the result*, and switching meant naming the pair again.
   * The pair, its direction and its scope are the comparison; **Summary**, **Browse** and **Data**
   * are views over it, and they live in the URL like every other piece of view state.
   *
   * What is here is what belongs to the comparison rather than to one reading of it: the two
   * checkpoints, the swap, the scope, and the view switch. Each view keeps its own body.
   */
  import { onMount } from 'svelte';
  import {
    cancelComparison,
    compareAgainst,
    comparison,
    diffBusy,
    diffError,
    diffStep,
    diffTree,
    establishComparison,
    stopComparing,
  } from '../stores/compare';
  import { clearReport, diffReport, loadReport } from '../stores/report';
  import { loadRecents, proxied, proxyHost, tree } from '../stores/server';
  import { navigate } from '../stores/view';
  import { middleTruncate, specHelp } from '../lib/format';
  import { isEditable } from '../lib/keys';
  import { emptyScope, type DiffScopeParams } from '../lib/diffscope';
  import type { CompareScreen, CompareView } from '../lib/hash';
  import CheckpointPicker from './CheckpointPicker.svelte';
  import SwapButton from './SwapButton.svelte';
  import ScopeBar from './ScopeBar.svelte';
  import DiffView from './DiffView.svelte';
  // The aligned tree, under the name the tabs give it. (Its file keeps the older name.)
  import BrowseView from './CompareView.svelte';
  import CompareData from './CompareData.svelte';
  import LoadingScreen from './LoadingScreen.svelte';

  /** The pair, as the server is asked about it — always in this order; `swapped` says which way it
   * is drawn. Empty `rhs` means the checkpoint the server has open. */
  export let lhs: string;
  export let rhs: string;
  export let view: CompareView = 'summary';
  export let scope: DiffScopeParams | undefined = undefined;
  export let full = false;
  export let swapped = false;
  export let closed: string[] = [];

  onMount(() => {
    void loadRecents();
  });

  /**
   * The boxes, in the order they are *drawn* — the pair, flipped when `swapped`.
   *
   * An empty candidate means "the checkpoint that is open", so it is shown as that checkpoint
   * wherever it is drawn. As a one-shot fill of whichever box happened to be empty at mount, a swap
   * left the *baseline* box blank: the candidate had moved into it, and "the open one" had already
   * been spent on the other side.
   */
  $: served = $tree?.spec ?? '';
  /**
   * Nothing is being compared, so the boxes are an empty form rather than a resolved pair.
   *
   * *Clear* leaves both operands empty, and an empty candidate *means* "the checkpoint that is open" —
   * so the resolution below filled the box with the served checkpoint the moment the comparison was
   * discarded: `/tmp/mapfix/new.safetensors` sitting in a form you had just emptied. What an empty box
   * means is the placeholder's job, which already says it.
   */
  $: blank = lhs === '';
  $: shownBase = blank ? '' : swapped ? rhs || served : lhs;
  $: shownRight = blank ? '' : swapped ? lhs : rhs || served;
  let draftBase = '';
  let draftRight = '';
  // Fields of an object rather than plain `let`s: each holds the value of the *previous* run, read on
  // the next one, which no static analysis can see (`no-useless-assignment` calls each dead).
  //
  // Tracking what was last *shown* is also what lets a box be emptied: clearing it deliberately does
  // not change `shownBase`, so nothing puts the text back.
  const applied = { base: '', right: '' };
  $: if (shownBase !== applied.base) {
    applied.base = shownBase;
    draftBase = shownBase;
  }
  $: if (shownRight !== applied.right) {
    applied.right = shownRight;
    draftRight = shownRight;
  }

  /** A comparison is being read; everything that would start another is inert until it lands. */
  // **The pair is set up here, once.** Both result views read it by id — the summary and the tree
  // used to each read the two checkpoints themselves, so switching views re-read them (seconds, or
  // minutes over an ssh proxy) and the summary compared against whatever the server had *open*
  // rather than against the candidate.
  $: if (lhs) void establishComparison({ left: lhs, right: rhs });
  // **The Data view sizes its run from the report**, whose totals follow the scope — so it needs one
  // even though it does not draw it. Only the Summary asked for it, so arriving straight at the Data
  // view (a link with `view=data`, or the tab) left it sizing a run from nothing: `0 tensors selected ·
  // about 0 B to read`, which is a claim about the pair rather than an admission of not knowing yet.
  // The same four arguments the Summary passes, so the two share one cached answer and switching views
  // costs no request.
  $: if (view === 'data') void loadReport($comparison?.id ?? null, scope, swapped, full);

  $: busy = $diffStep !== null;
  /** What an address box accepts, and which host `:PATH` resolves to — named rather than left to be
   * discovered, since the `:PATH` shorthand is documented nowhere else. */
  $: help = specHelp($proxied, $proxyHost ?? '');

  /** Go to the same comparison, changing one thing about how it is read. */
  function go(change: Partial<Omit<CompareScreen, 'kind'>>, replace = false) {
    navigate({ kind: 'compare', lhs, rhs, view, scope, full, swapped, closed, ...change }, replace);
  }

  function submit() {
    const base = draftBase.trim();
    if (!base) return;
    const other = draftRight.trim();
    // Omitted when it is just the open checkpoint, so the common case stays a short URL.
    const candidate = other === served ? '' : other;
    // The boxes are the pair *as drawn*, so submitting also settles the orientation.
    const unchanged = lhs === base && rhs === candidate && !swapped;
    go({ lhs: base, rhs: candidate, swapped: false });
    // Navigating to an identical hash emits no event. Compare still means "read these now" — it is
    // how a checkpoint rewritten on disk is refreshed, and otherwise the button appears dead.
    //
    // What that takes depends on the view: Browse has to fetch its tree again as well, since the
    // rows on screen came from the read being replaced. Summary and Data re-fetch themselves, being
    // keyed by the comparison id, which a fresh set-up changes.
    if (unchanged) {
      if (view === 'browse') {
        void compareAgainst({ left: base, right: candidate, force: true, scope, full });
      } else {
        void establishComparison({ left: base, right: candidate, force: true });
      }
    }
  }

  /**
   * Swap the two sides — before a comparison has been run, or after.
   *
   * With a result on screen this is one bit in the URL: the pair and its scope stay as the server was
   * asked for them and the drawing turns round, so nothing is refetched and the link still means what
   * it shows. With nothing loaded there is no comparison to turn round, so the boxes exchange text.
   */
  function swapBoth() {
    if (!$comparison) {
      [draftBase, draftRight] = [draftRight, draftBase];
      return;
    }
    go({ swapped: !swapped }, true);
  }

  /**
   * Discard the comparison *and* the URL that names it.
   *
   * Leaving `#compare?lhs=…` in the address bar meant the app claimed a comparison that was no
   * longer on screen, and a reload brought back exactly what had just been cleared. Distinct from
   * `cancelComparison`, which stops a read and keeps both paths for a retry.
   */
  async function clear() {
    clearReport();
    await stopComparing();
    navigate({ kind: 'compare', lhs: '', rhs: '' }, true);
  }

  /**
   * `s` swaps and `k` folds the families, wherever the caret is not.
   *
   * Both on the page rather than in a view, because both are facts about the *comparison* — the pair's
   * direction and whether uniform layers are one row or sixty-two — and both have a control up here.
   * `k` used to be handled by the aligned tree alone, so on the summary, which draws the very checkbox
   * it toggles, the key did nothing. `n`/`N` belong to a view with rows, and each of those handles them.
   */
  function onKeydown(e: KeyboardEvent) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (isEditable(e.target)) return;
    if (e.key === 's') {
      e.preventDefault();
      swapBoth();
    } else if (e.key === 'k') {
      e.preventDefault();
      go({ full: !full });
    }
  }

  /** Apply a scope from the panel. Named and typed: a callback prop's parameter has no type inside a
   * template, and a template expression cannot carry an annotation to give it one. */
  function applyScope(next: DiffScopeParams) {
    go({ scope: next });
  }

  /**
   * A fix offered from inside a result: apply it, or — with `null` — open the panel where it is set.
   *
   * `matchingOpen` is bumped rather than set, because the panel may already be open and the reader may
   * have closed it: a counter always changes, so the panel always responds.
   */
  let matchingOpen = 0;
  function fix(next: DiffScopeParams | null) {
    if (next) go({ scope: next });
    else matchingOpen += 1;
  }

  const VIEWS: { id: CompareView; label: string; hint: string }[] = [
    {
      id: 'summary',
      label: 'Summary',
      hint: 'What was added, removed and changed, by section',
    },
    { id: 'browse', label: 'Browse', hint: 'The two checkpoints as one aligned tree, in lockstep' },
    { id: 'data', label: 'Data', hint: 'Compare the numbers, not just the shapes' },
  ];
</script>

<svelte:window on:keydown={onKeydown} />

<div class="page">
  <!-- One address above the other, not side by side.
       A row of two boxes gave each half the window, which is not enough for the addresses people
       actually compare — an `s3://` URI and a `host:/opt/…` path are 60-odd characters each, so both
       were scrolled and neither was readable. Stacked, each gets the full width and the pair reads
       as what it is: old over new, in the order the report states them. -->
  <form class="pick" on:submit|preventDefault={submit}>
    <div class="pair">
      <div class="line">
        <label for="cmp-base">Baseline</label>
        <CheckpointPicker
          id="cmp-base"
          bind:value={draftBase}
          {busy}
          ariaLabel="baseline"
          placeholder="baseline — {help}"
          title={help}
          onEnter={() => submit()}
        />
      </div>
      <!-- "Candidate", not "newer": any two checkpoints can be compared, and the app enforces no
           chronology between them. The *report* still says old/new, which is what `diff OLD NEW`
           prints and what the terminal shows — that is the direction of the diff, not a claim about
           which was made first. -->
      <div class="line">
        <label for="cmp-right">Candidate</label>
        <CheckpointPicker
          id="cmp-right"
          bind:value={draftRight}
          {busy}
          ariaLabel="the candidate"
          placeholder={$tree?.spec ?? 'the open checkpoint'}
          title={help}
          onEnter={() => submit()}
        />
      </div>
    </div>
    <!-- Between the two rows it belongs to, and the only one on the page: the report used to carry a
         second copy of this button over a second copy of the two addresses. -->
    <SwapButton onSwap={swapBoth} disabled={busy} title="Swap the two sides (s)" />
    <div class="go">
      <button type="submit" disabled={!draftBase.trim() || busy}>Compare</button>
      <!-- Two buttons, because they are two different actions on two different states. One "Stop"
           used to mean "discard the result" and appeared only *after* the read finished — so the
           phase you would actually want to stop was the one with no button at all. -->
      {#if busy}
        <button type="button" class="quiet" on:click={cancelComparison}>Cancel</button>
      {:else if $comparison}
        <button type="button" class="quiet" on:click={() => void clear()}>Clear</button>
      {/if}
    </div>
  </form>

  {#if lhs}
    <ScopeBar
      scope={scope ?? emptyScope()}
      onApply={applyScope}
      matched={$diffTree?.matched ?? $diffReport?.matched ?? null}
      {busy}
      openMatching={matchingOpen}
    />

    <!-- The three readings of one comparison. A tab, not a link to another screen: everything that
         makes this *this* comparison — both sides, the direction, the scope, the family fold — is
         held here, so switching cannot change it. -->
    <nav class="views" aria-label="How to read this comparison">
      {#each VIEWS as v (v.id)}
        <button
          type="button"
          class="view"
          class:on={view === v.id}
          aria-pressed={view === v.id}
          title={v.hint}
          on:click={() => go({ view: v.id })}>{v.label}</button
        >
      {/each}
    </nav>
  {/if}

  <div class="body">
    <!-- Browse owns the setup/tree's richer three-stage wait. Summary and Data used to render none of
         the shared pair setup: Compare produced a blank body while reading, and a failed POST left a
         blank body forever. Every view now visibly accounts for that shared prerequisite. -->
    {#if view !== 'browse' && $diffStep}
      <LoadingScreen step={$diffStep} />
    {:else if view !== 'browse' && $diffBusy}
      <div class="setup-message" role="status">
        <p>
          The server is still reading
          <span class="mono" title={$diffBusy.spec}>{middleTruncate($diffBusy.spec, 60)}</span>
          ({Math.round($diffBusy.seconds)}s so far).
        </p>
        <button type="button" on:click={() => void establishComparison({ left: lhs, right: rhs, force: true })}>Try again</button>
      </div>
    {:else if view !== 'browse' && $diffError}
      <div class="setup-message error" role="alert">
        <p>{$diffError}</p>
        <button type="button" on:click={() => void establishComparison({ left: lhs, right: rhs, force: true })}>Try again</button>
      </div>
    {:else if view === 'browse'}
      <BrowseView {lhs} {rhs} {scope} {full} {swapped} onFix={fix} onNavigate={go} />
    {:else if view === 'data'}
      <CompareData left={lhs} right={rhs || $tree?.spec || ''} {scope} />
    {:else}
      <DiffView {scope} {full} {swapped} {closed} onNavigate={go} />
    {/if}
  </div>
</div>

<style>
  .page {
    height: 100%;
    display: flex;
    flex-direction: column;
    padding: 12px 16px;
    gap: 10px;
    min-height: 0;
  }
  .pick {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  /* The two addresses, stacked, sharing one column so the boxes line up under each other. */
  .pair {
    flex: 1 1 auto;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .line {
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 0;
  }
  /* A fixed label column, so the two boxes start at the same x. */
  .pick label {
    flex: 0 0 9ch;
    color: var(--fg-dim);
  }
  /* Both actions in a column of their own, beside the pair rather than trailing the second row —
     where "Compare" read as belonging to the candidate alone. */
  .go {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .go button {
    width: 100%;
  }
  .quiet {
    color: var(--fg-dim);
    background: none;
    border: 1px solid var(--border);
    border-radius: 4px;
    font: inherit;
    font-size: 12px;
    padding: 3px 9px;
    cursor: pointer;
  }
  .setup-message {
    padding: 10px;
    border-radius: 4px;
    background: var(--bg-elev);
    font-size: 12px;
  }
  .setup-message p {
    margin: 0 0 8px;
  }
  .setup-message.error {
    color: var(--danger);
  }
  .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  /* Tabs, outlined by their fill rather than by a box — the treatment the palette and the scope bar
     use. The selected one is the only one that looks pressed. */
  .views {
    flex: 0 0 auto;
    display: flex;
    gap: 4px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
  }
  .view {
    font: inherit;
    font-size: 12.5px;
    color: var(--fg-dim);
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 5px 12px;
    cursor: pointer;
  }
  .view:hover {
    color: var(--fg);
    background: var(--bg-hover);
  }
  .view.on {
    color: var(--fg);
    border-bottom-color: var(--accent);
    font-weight: 600;
  }
  /* The view's own body scrolls; the pair, the scope and the tabs stay put. */
  .body {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
</style>
