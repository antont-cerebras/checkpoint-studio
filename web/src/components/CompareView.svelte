<script lang="ts">
  // Side-by-side comparison of two checkpoints, browsed in lockstep.
  //
  // One tree with two columns, not two trees: the server aligns them
  // (`checkpoint_studio_core::difftree`), so a row is a name with each side's content beside it and
  // a gap where a side has nothing. Folding and the cursor are therefore shared by construction —
  // there is no second scroll position to keep in step, which is the part that would drift.
  //
  // `n`/`N` step between differences, wrapping; `s` swaps the sides. Clicking a tensor opens its
  // detail — but only in the pane that is the *served* checkpoint, because that is the one the detail
  // screen reads; on any other side a click does nothing rather than show a different checkpoint's
  // numbers under that name, and the cell's tooltip says why (see `clickOutcome`, `cellTitle`).
  import { get } from 'svelte/store';
  import { tick } from 'svelte';
  import {
    compareAgainst,
    diffBusy,
    diffCursor,
    diffError,
    diffExpanded,
    diffStep,
    diffTree,
  } from '../stores/compare';
  import { proxyHost } from '../stores/server';
  import { openDetail } from '../stores/view';
  import type { CompareScreen } from '../lib/hash';
  import {
    allGroupPaths,
    ancestorsOf,
    clickOutcome,
    differing,
    differingCount,
    emptyRowsNote,
    flattenDiff,
    identicalNote,
    isDisjoint,
    tallyIsReadable,
    nextDifference,
    sideText,
    statusMark,
    swapResponse,
    type AlignedNode,
    type DiffRow,
    type DiffTally,
    type DiffSide,
    type Which,
  } from '../lib/difftree';
  // The same rule the one-page report's rows use: mark the piece that differs, not the whole side.
  import { sigCells } from '../lib/difflines';
  import { rowGlyph } from '../lib/glyphs';
  import type { TensorSig } from '../lib/types';
  import { humanCount, humanSize, middleTruncate, totalsParts } from '../lib/format';
  import { isEditable } from '../lib/keys';
  import LoadingScreen from './LoadingScreen.svelte';
  import { emptyScope, scopeSummary, type DiffScopeParams } from '../lib/diffscope';
  import { shortSpec, stepLabel } from '../lib/loadstep';
  import TextField from './TextField.svelte';
  import FamilyToggle from './FamilyToggle.svelte';
  import DiffChip from './DiffChip.svelte';
  import { tallyTitle, type TallyKind } from '../lib/tallywords';

  /** The comparison's two sides, from the URL — so a comparison is a shareable link. */
  export let lhs: string;
  export let rhs: string;
  /** The selection, from the URL — the same parameters the report takes. */
  export let scope: DiffScopeParams | undefined = undefined;
  /**
   * `--full`: every layer as its own row, rather than uniform index families folded onto one each.
   *
   * Off by default, as in the report and on the CLI, and for the same reason: a re-quantization aligns
   * to 117,000 rows of which 116,000 say what the row above said. Folded, the *irregular* layer — an
   * extra tensor, a dtype its siblings don't share — is one of a handful of rows on screen. The server
   * does the folding (`difftree::fold_families`), so the terminal and the browser fold alike.
   */
  export let full = false;
  /**
   * Which way round the pair is being read — **a view state, not a different comparison**.
   *
   * `lhs` and `rhs` are the pair the *server* is asked about, always in that order; this says
   * which of them is drawn as the baseline. Flipping used to rewrite the two operands instead, which
   * looked right on screen (the loaded rows were transformed in memory) and was wrong in the URL: the
   * scope is directional — `--map` rewrites the baseline's names and cannot be inverted, and a
   * `#subtree` belongs to one side — so reloading the flipped URL asked the server to apply the old
   * scope to the reversed operands, and answered a different comparison than the one that produced
   * the link. Kept canonical, a flip is free (`swapResponse`), needs no refetch, and any link
   * reproduces exactly what its sender saw. The report screen has always modelled it this way.
   */
  export let swapped = false;
  /**
   * Apply a scope from inside the result — or, with `null`, just open the panel where it is set.
   *
   * The tree is where "nothing lines up" becomes visible, and the fix for it lives in a panel above.
   * Handing the fix to the page rather than describing where to find it is the difference between an
   * explanation and a way out.
   */
  export let onFix: (scope: DiffScopeParams | null) => void = () => {};
  /**
   * Change one thing about how this comparison is read.
   *
   * Injected by `ComparePage`, like the summary's: a view that navigates on its own has to re-state
   * the whole comparison, and the one that did dropped the view it was in — so folding the families
   * from the tree landed back on the summary.
   */
  export let onNavigate: (change: Partial<Omit<CompareScreen, 'kind'>>, replace?: boolean) => void;

  // The server's answer is canonical; `swapped` decides which way it is drawn. One transform, applied
  // where the screen reads it, so the rows, the two side descriptions and the counts cannot disagree
  // about the direction — the tally once did.
  $: data = $diffTree === null ? null : swapped ? swapResponse($diffTree) : $diffTree;
  $: tally = data?.tally ?? null;
  /** The matching-checkpoints banner, or null when something differs. */
  $: identical = tally ? identicalNote(tally) : null;
  /** The active selection in one line, for the header — empty when nothing narrows the comparison.
   * The same summary the scope bar shows, said where the counts are, since a narrowed row count with
   * no stated scope is a number you cannot check. */
  $: scopeText = scope ? scopeSummary(scope) : '';

  /** An unreadable tally reads as no differences rather than as `NaN` — see `tallyIsReadable`. */
  const EMPTY = { same: 0, changed: 0, only_old: 0, only_new: 0 };

  // Load whenever the URL's pair or scope changes, including back/forward — this view is addressable,
  // so arriving at it twice with different parameters has to show different comparisons. `swapped` is
  // deliberately *not* a dependency: the pair the server is asked for is the same either way round,
  // so flipping costs nothing.
  //
  // Lazy, and only here: the aligned tree is the largest body this API serves (91 MB on a real pair),
  // so the Summary view must not pay for it. That is why the page does not load it centrally.
  $: if (lhs) void compareAgainst({ left: lhs, right: rhs, scope, full });

  /** Show only rows that differ — see [`flattenDiff`]. Off by default: the point of the two panes is
   * that a difference is visible *in context*, and hiding the matches loses the context. */
  let differencesOnly = false;
  /** *Find in results*: narrow the rows on screen to the names that contain this. Not the scope —
   * the scope changes what the server compared. */
  let find = '';


  // Virtualized, the same way the tensor tree is (`TreeView.svelte`): two unrelated checkpoints
  // produce a row per tensor on each side — 117k differences were measured — and putting all of them
  // in the DOM makes the screen unusable rather than merely slow. Only the visible slice is rendered,
  // with spacers standing in for the rest so the scrollbar still describes the whole comparison.
  const ROW_H = 22;
  let scrollEl: HTMLElement | undefined;
  let scrollTop = 0;
  let viewportH = 600;
  $: first = Math.max(0, Math.floor(scrollTop / ROW_H) - 6);
  $: slice = rows.slice(first, first + Math.ceil(viewportH / ROW_H) + 12);

  /**
   * The rows to draw, and the reason there are none when there are none.
   *
   * Guarded, because a throw in here does not fail *this*: it aborts the whole Svelte flush, so the
   * DOM keeps whatever it had — which is how a `RangeError` in the flattener left a finished download
   * showing a progress bar at 100% and a frozen timer, with no error anywhere and no way out but a
   * reload. A view that cannot render is a message, not a hang.
   */
  function derive(
    d: typeof data,
    expanded: Set<string>,
    only: boolean,
    needle: string,
  ): { rows: DiffRow[]; failed: string } {
    if (!d) return { rows: [], failed: '' };
    try {
      return { rows: flattenDiff(d.rows, expanded, only, needle), failed: '' };
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      return {
        rows: [],
        failed: `This comparison could not be drawn: ${msg}. It has ${d.differences.length.toLocaleString()} differences — try “differences only”, or compare a narrower pair.`,
      };
    }
  }

  $: derived = derive(data, $diffExpanded, differencesOnly, find);
  /** What to put where the rows would be when the filter leaves none — a blank pane reads as a
   * failed load. */
  $: emptyNote = tally ? emptyRowsNote(rows.length, tally) : '';
  $: rows = derived.rows;
  $: differences = data?.differences ?? [];
  $: cursorAt = $diffCursor === null ? -1 : differences.indexOf($diffCursor);

  function toggle(path: string) {
    diffExpanded.update((s) => {
      const next = new Set(s);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }

  /** Move to the next/previous difference, unfolding whatever hides it and scrolling it into view. */
  async function step(direction: 1 | -1) {
    if (!data) return;
    const to = nextDifference(differences, $diffCursor, direction);
    if (to === null) return;
    diffCursor.set(to);
    // Unfold the ancestors first: jumping to a row inside a folded group would move the cursor to
    // something not on screen, which reads as nothing happening.
    //
    // Only the ones that are actually folded. This used to write a fresh `Set` unconditionally, which
    // invalidated the store on every press and re-flattened the whole tree — ~640 ms per keypress on
    // a 50k-row comparison, for a fold state that had not changed. After the initial reveal the
    // ancestors are already open, so the common case now costs nothing.
    const open = get(diffExpanded);
    const need = ancestorsOf(data.rows, to).filter((p) => !open.has(p));
    if (need.length > 0) {
      diffExpanded.update((s) => new Set([...s, ...need]));
      await tick();
    }
    document
      .querySelector(`[data-diff-path="${CSS.escape(to)}"]`)
      ?.scrollIntoView({ block: 'center' });
  }

  /** Unfold every group, or fold them all back. The load's own choice is neither — see
   * `initialExpansion` — and before this there was no way to change it. */
  function foldAll(open: boolean) {
    diffExpanded.set(open && data ? allGroupPaths(data.rows) : new Set());
  }

  /** Fold or unfold the families — through the page, which holds the rest of what makes this *this*
   * comparison. Navigating from here dropped `view`, so folding sent the reader back to the summary. */
  function setFull(everything: boolean) {
    onNavigate({ full: everything });
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    // The shared rule (`lib/keys`), not a hand-written tag list: this one checked `INPUT` and `SELECT`
    // and not `TEXTAREA`, so `s` typed into a scope box swapped the two checkpoints.
    if (isEditable(e.target)) return;
    // No `k` here: the page handles it, for every view of the comparison. Handled in both, the two
    // listeners fired on one keypress and the fold toggled twice — back to where it started.
    if (e.key === 'n') {
      e.preventDefault();
      void step(1);
    } else if (e.key === 'N') {
      e.preventDefault();
      void step(-1);
    }
  }

  /** The two panes of a row, typed once so the call site needs no cast. */
  function columnsOf(row: DiffRow): { side: DiffSide | null; which: Which }[] {
    return [
      { side: row.node.old, which: 'old' },
      { side: row.node.new, which: 'new' },
    ];
  }

  /**
   * The signature in the *other* column, for marking one side against it.
   *
   * `null` when the other side is missing or is not a tensor — a name that is a tensor here and a
   * group there is a change of kind, and there is no dimension-by-dimension story to tell about it.
   */
  function otherSig(node: AlignedNode, which: Which): TensorSig | null {
    const other = which === 'old' ? node.new : node.old;
    return other?.kind === 'tensor' ? other.info : null;
  }

  /** The glyph this row leads with, in both columns — the terminal's set (`lib/glyphs`). */
  function glyphOf(row: DiffRow): string {
    if (row.node.children.length > 0) {
      return rowGlyph({ kind: 'group', fold: row.expanded ? 'open' : 'closed' });
    }
    const side = row.node.old ?? row.node.new;
    return side?.kind === 'metadata'
      ? rowGlyph({ kind: 'metadata' })
      : rowGlyph({ kind: 'tensor', listing: 'listed' });
  }

  /** `11 differences · row 1 of 8 ·` — the headline count (tensors, which folding does not change),
   * then where the cursor is among the rows that differ, worded as the terminal words it. */
  $: navLine = (() => {
    const total = differingCount(tally ?? { tensors: EMPTY, metadata: EMPTY });
    const plural = total === 1 ? '' : 's';
    const at =
      cursorAt >= 0
        ? ` · row ${(cursorAt + 1).toLocaleString()} of ${differences.length.toLocaleString()}`
        : '';
    return `${total.toLocaleString()} difference${plural}${at} · `;
  })();

  /** The counts worth a chip, in report order — the empty ones say nothing worth the space.
   *
   * `label` doubles as the kind whose meaning the tooltip states (`lib/tallywords`), so this strip and
   * the summary's answer "what counts as metadata" with one sentence rather than two. */
  function countChips(
    t: DiffTally,
  ): { tone: 'added' | 'removed' | 'changed' | 'meta'; count: number; label: TallyKind }[] {
    return [
      { tone: 'added' as const, count: t.tensors.only_new, label: 'added' as const },
      { tone: 'removed' as const, count: t.tensors.only_old, label: 'removed' as const },
      { tone: 'changed' as const, count: t.tensors.changed, label: 'changed' as const },
      { tone: 'meta' as const, count: differing(t.metadata), label: 'metadata' as const },
    ].filter((c) => c.count > 0);
  }

  /** Whether this cell does anything when pressed — a group folds, a served tensor opens. */
  function pressable(row: DiffRow, which: Which): boolean {
    const sides = data ? { base: data.base, current: data.current } : null;
    return clickOutcome(row.node, which, sides).kind !== 'none';
  }

  /** The first column that has anything in it — where a fact about the whole row is drawn. */
  function leadColumn(node: AlignedNode): Which {
    return node.old ? 'old' : 'new';
  }

  /** Whether a row's two sides say the same thing — a group "changed" only by its children reads
   * `6 tensors` on both, and highlighting both copies of that says nothing. */
  function sidesRead(node: AlignedNode): 'alike' | 'differently' {
    if (!node.old || !node.new) return 'alike';
    return sideText(node.old) === sideText(node.new) ? 'alike' : 'differently';
  }

  /** Act on a click: fold a group, or open a tensor of the checkpoint this tab has loaded. The
   * decision itself is `clickOutcome` in lib/, where it is covered and where it cannot be lost in a
   * template — including the case that is deliberately nothing (see `cellTitle`). */
  function activate(row: DiffRow, which: Which) {
    const outcome = clickOutcome(row.node, which, data ? { base: data.base, current: data.current } : null);
    switch (outcome.kind) {
      case 'toggle':
        toggle(outcome.path);
        return;
      case 'open':
        openDetail(outcome.name);
        return;
      case 'none':
        return;
    }
  }

  /**
   * A cell's tooltip: what the row is, and — where a click does nothing — why.
   *
   * The "why" was a paragraph above the tree with an *Open it* button, tagged with the comparison and
   * the view state it was raised against so it could not outlive them. All of that to explain a click
   * that did nothing. On hover it costs no space at all.
   */
  function cellTitle(row: DiffRow, which: Which): string {
    const what = `${row.node.name} — ${sideText(which === 'old' ? row.node.old : row.node.new)}`;
    if (row.node.children.length > 0) return what;
    const info = which === 'old' ? data?.base : data?.current;
    if (!info || info.served) return what;
    return `${what}\nin ${info.spec} — not the checkpoint this tab has loaded, so its data cannot be opened here`;
  }
</script>

<svelte:window on:keydown={onKeydown} />

<div class="compare">
  <!-- No pair boxes, no scope bar, no data panel: those belong to the *comparison*, and
       `ComparePage` owns them. This is one reading of it — the aligned tree. -->
  <!-- The tree stays while a *fold* is re-aligned.
       Pressing `k` asks the server to redraw the same comparison with families collapsed — nothing is
       re-read, and both checkpoints are already in its slot — but the whole tree used to vanish behind
       the loading screen while the new one came over, which reads exactly like starting again. A first
       tree has nothing to keep and gets the screen; one already on display stays, dimmed, with a line
       saying what is coming. -->
  {#if $diffStep && !data}
    <LoadingScreen step={$diffStep} />
  {:else if $diffBusy}
    <!-- Only reachable when the takeover itself failed: asking for a comparison already asks the server
         to stop whatever else it is reading (`compareAgainst`'s `stopOther`), so being told "it reads one
         checkpoint at a time" and offered a button whose only sensible answer was yes was a question
         between you and the thing you had just asked for. -->
    <div class="busy" role="status">
      <p>
        The server is still reading
        <span class="mono" title={$diffBusy.spec}>{middleTruncate($diffBusy.spec, 60)}</span>
        ({Math.round($diffBusy.seconds)}s so far) and would not let go. It reads one checkpoint at a
        time.
      </p>
      <div class="acts">
        <button
          type="button"
          on:click={() =>
            void compareAgainst({ left: lhs, right: rhs, force: true, scope, full })}
        >
          Try again
        </button>
      </div>
    </div>
  {:else if $diffError}
    <p class="err" role="alert">{$diffError}</p>
    <!-- A read that failed because the server was busy with another one is a "try again", and the
         only thing missing was something to try again *with*. -->
    <p>
      <button
        type="button"
        class="quiet"
        on:click={() => void compareAgainst({ left: lhs, right: rhs, force: true, scope, full })}
      >
        Try again
      </button>
    </p>
  {:else if derived.failed}
    <p class="err" role="alert">{derived.failed}</p>
  {:else if data && tally && !tallyIsReadable(tally)}
    <!-- The server answered in a shape this build does not understand — which happens for exactly one
         reason: this tab was loaded before the server was upgraded under it, and its JavaScript is
         still the old one. Said outright, because the alternative is what it used to do: read every
         counter as `undefined`, total them to `NaN`, and — since `NaN > 0` is false — announce that two
         checkpoints sharing no tensor name at all were "structurally identical". -->
    <div class="stale" role="alert">
      <strong>This page is out of date</strong>
      <span
        >The server sent a comparison in a newer shape than this tab can read, so nothing below would
        be trustworthy. Reload to pick up the current version.</span
      >
      <button type="button" on:click={() => location.reload()}>Reload</button>
    </div>
  {:else if data && tally}
    <div class="head" class:updating={$diffStep !== null}>
      {#if $diffStep}
        <!-- What is coming, over the tree that is still readable. `stepLabel` rather than a second
             wording: it is the same wait, drawn smaller. -->
        <p class="updating-note dim" role="status">{stepLabel($diffStep)}…</p>
      {/if}
      <!-- In full. A fixed 64-character cut threw away the end of the path — `…/Kimi-K2.6-3bit` — which
           is the part that says *which* checkpoint this column is, and there is room for it: each
           column is half a window wide. Wraps rather than truncating when there genuinely is not. -->
      <!-- `:/path` rather than `host:/path` when the host is this server's own proxy: on every line it
           is the same fifty characters, and it pushes the part that differs out of the column. -->
      <span class="side old" title={data.base.spec}>{shortSpec(data.base.spec, $proxyHost ?? '')}</span>
      <span class="side new" title={data.current.spec}
        >{shortSpec(data.current.spec, $proxyHost ?? '')}</span>
      <!-- The summary, laid out rather than printed.
           These were four dim monospace lines and a run-on count — the terminal's own output pasted
           into a page, where the reader has to parse `size: A → B (+Δ, +x%)` to find the number that
           matters. The facts are the same and the assembled strings are still the contracted ones
           (`lib/format`); what changed is that each has a place. -->
      <div class="summary">
        <div class="counts">
          <!-- The same chips the report's strip is made of, in the same colours — they were two sets
               with two greens. Plain here: these describe the tree already on screen, so there is
               nowhere for a click to go. -->
          <DiffChip
            tone="same"
            count={tally.tensors.same.toLocaleString()}
            label="unchanged"
            title={tallyTitle('unchanged', tally.tensors.same)}
          />
          {#each countChips(tally) as chip (chip.label)}
            <DiffChip
              tone={chip.tone}
              count={chip.count.toLocaleString()}
              label={chip.label}
              title={tallyTitle(chip.label, chip.count)}
            />
          {/each}
          {#if differences.length}
            <span class="nav dim">
              <!-- The total counts differing *tensors*; the position counts the rows `n`/`N` stop on.
                   Assembled in the script: a `{#if}` in the middle of a sentence loses the space in
                   front of it, which read as `11 differences· row 1 of 8`. -->
              {navLine}
              <kbd>n</kbd>/<kbd>N</kbd> step · <kbd>s</kbd> swap · <kbd>k</kbd> families
            </span>
          {/if}
        </div>

        <!-- Two stats rather than two sentences: the label, the two values, and the change as its own
             piece — which is the one you came for. -->
        <dl class="stats">
          {#each [{ label: data.totals_labels.size, parts: totalsParts(data.base.bytes, data.current.bytes, humanSize) }, { label: data.totals_labels.params, parts: totalsParts(data.base.params, data.current.params, humanCount) }] as stat (stat.label)}
            <div class="stat">
              <dt>{stat.label}</dt>
              <dd>
                <span class="from">{stat.parts.from}</span>
                {#if stat.parts.delta}
                  <span class="arrow" aria-hidden="true">→</span>
                  <span class="to">{stat.parts.to}</span>
                  <span class="delta" class:up={stat.parts.direction > 0} class:down={stat.parts.direction < 0}>
                    {stat.parts.delta}{stat.parts.percent ? ` (${stat.parts.percent})` : ''}
                  </span>
                {:else}
                  <span class="delta same">unchanged</span>
                {/if}
              </dd>
            </div>
          {/each}
        </dl>

        <!-- What was compared, and what the gutter marks mean. This view is nothing but marks and two
             columns, and it explained neither; the scope belongs here too, because "why am I looking at
             nineteen rows" is a header question. -->
        <p class="caption dim">
          structure only — names, dtypes and shapes; values not compared{scopeText
            ? ` · ${scopeText}`
            : ''}
          <span class="key"
            ><span class="removed">−</span> removed <span class="added">+</span> added
            <span class="changed">~</span> changed</span>
        </p>
      </div>
      {#if identical}
        <!-- A banner, because this is the one outcome where the tree below is empty: the same words in
             dim text beside a count of zero read as "nothing loaded" rather than as the answer. The
             detail is always visible rather than on hover — it is the part that stops "identical" being
             over-read as "the weights are the same", and a tooltip is no use on a touch screen. -->
        <div class="identical" role="status">
          <strong>{identical.headline}</strong>
          <span class="what">{identical.detail}</span>
        </div>
      {/if}
      {#if isDisjoint(tally)}
        <!-- The one sentence that explains the whole comparison, when it applies — with the things
             that fix it. Two checkpoints with unrelated naming schemes align nothing, so every tensor
             of both is one-sided and the difference count is just their sum: a number that describes
             nothing. Naming the cause and leaving the reader to find `--align-fused` in a panel of
             nine inputs is a diagnosis without a treatment. -->
        <div class="disjoint" role="status">
          <p>
            These two checkpoints share no tensor names — different naming schemes, so nothing lines
            up. {tally.tensors.only_old.toLocaleString()} only in the baseline,
            {tally.tensors.only_new.toLocaleString()} only in the candidate.
          </p>
          <div class="fixes">
            <button type="button" on:click={() => onFix({ ...(scope ?? emptyScope()), alignFused: true })}>
              Try fused ↔ unfused alignment
            </button>
            <button type="button" on:click={() => onFix(null)}>Choose matching subtrees…</button>
            <span class="dim">or add a custom rule under <em>Match different names</em>.</span>
          </div>
        </div>
      {/if}
    </div>

    <!-- Fold and filter. The load either reveals every difference or (past `REVEAL_LIMIT`) none, and
         before this there was no way to change its mind. -->
    <div class="controls">
      <button type="button" class="quiet" on:click={() => foldAll(true)}>Expand all</button>
      <button type="button" class="quiet" on:click={() => foldAll(false)}>Collapse all</button>
      <label class="only">
        <input type="checkbox" bind:checked={differencesOnly} />
        Differences only
      </label>
      <!-- The report's control, worded the same, and in the URL for the same reason. Through
           `navigate` rather than a local flag: the server folds, so this is a different request. -->
      <FamilyToggle {full} onChange={setFull} />
      <!-- *Find in results*, the report's control, on the tree. A different question from the scope
           above: the scope decides what the server compares, this decides what is on screen — and the
           tree had no way at all to ask "where is qkv_proj" in 117,000 rows. -->
      <label class="only" for="cmp-find">Find in results</label>
      <TextField
        id="cmp-find"
        bind:value={find}
        grow={false}
        width="24ch"
        placeholder="part of a tensor name"
        spellcheck="false"
        autocomplete="off"
      />
      {#if find.trim()}
        <button type="button" class="quiet" on:click={() => (find = '')}>clear</button>
      {/if}
      <span class="dim rowcount">{rows.length.toLocaleString()} rows shown</span>
    </div>

    <!-- `treegrid`, not a pile of divs: this is a tree whose rows have two cells, which is exactly
         what the role describes, and it is what lets a screen reader announce the fold state and the
         position instead of reading 187,000 anonymous buttons. -->
    <div
      class="rows"
      role="treegrid"
      aria-label="Aligned comparison of the two checkpoints"
      aria-rowcount={rows.length}
      bind:this={scrollEl}
      bind:clientHeight={viewportH}
      on:scroll={() => (scrollTop = scrollEl?.scrollTop ?? 0)}
    >
      <!-- The rows above and below the window, as height rather than elements. -->
      <div style="height:{first * ROW_H}px"></div>
      {#each slice as row, i (`${row.node.path}@${row.depth}`)}
        <div
          class="row {row.node.status.kind}"
          class:cursor={$diffCursor === row.node.path}
          data-diff-path={row.node.path}
          role="row"
          aria-level={row.depth + 1}
          aria-rowindex={first + i + 1}
          aria-expanded={row.node.children.length ? row.expanded : undefined}
        >
          <span class="mark" role="gridcell">{statusMark(row.node.status.kind)}</span>
          <!-- Two real tree columns, each drawing its own side: indent, twisty, name, signature.
               A side that has no row draws nothing, which is what makes an addition or a removal
               obvious. One aligned row per line means the two trees scroll and fold in step by
               construction — there is no second scroll position to keep. -->
          {#each columnsOf(row) as col (col.which)}
            <!-- A side with no row here is a *gap*, not a control: it was rendering as an empty
                 disabled `<button>`, which a screen reader announces as a nameless button and there
                 were two or three per row. Nothing to press means nothing to render. -->
            {#if col.side}
              <!-- A cell is a control when there is something to press: a group folds, and a tensor of
                   the checkpoint this tab has loaded opens its detail. A tensor of the *other* side
                   opens nothing — the detail screen reads the served checkpoint — so it is text, not a
                   button that quietly does nothing. `svelte:element` keeps one set of children. -->
              <svelte:element
                this={pressable(row, col.which) ? 'button' : 'span'}
                type={pressable(row, col.which) ? 'button' : undefined}
                class="cell {col.which}"
                class:inert={!pressable(row, col.which)}
                role="gridcell"
                tabindex="-1"
                style="padding-left:{row.depth * 14}px"
                title={cellTitle(row, col.which)}
                on:click={() => activate(row, col.which)}
              >
                <!-- The terminal's glyphs, from the one place that defines them (`lib/glyphs`):
                     ▾ / ▸ for a group, · for a tensor, † for a metadata entry. -->
                <span class="caret">{glyphOf(row)}</span>
                <span class="nm">{row.node.name}</span>
                <!-- What a folded family stands for. The subtree below it is one member, so this is the
                     number that says the row is not one layer. Once per row, not once per column: how
                     many layers a row stands for is a fact about the row, unlike the `×256` of a
                     rename fold, which is a fact about one side. -->
                {#if row.node.members > 1 && col.which === leadColumn(row.node)}
                  <span class="family" title="{row.node.members.toLocaleString()} identical layers folded onto this row — untick Collapse families for all of them">×{row.node.members.toLocaleString()}</span>
                {/if}
                <!-- **Only the piece that differs is tinted.** A tensor's signature is drawn as dtype
                     and dimensions, each marked against its counterpart (`sigCells`), so a
                     re-quantization's rows show a coloured dtype and untouched shapes, and a fold shows
                     the leading axis it gained. Anything else — a group's count, a metadata value — is
                     marked only when the two sides really read differently, which is why
                     `6 tensors` / `6 tensors` is now plain on both sides. -->
                {#if col.side.kind === 'tensor'}
                  {@const cells = sigCells(col.side.info, otherSig(row.node, col.which))}
                  <span class="sig"
                    ><span class:hl={cells.dtype.differs}>{cells.dtype.text}</span> ({#each cells.dims as d, di (di)}<span
                      class:hl={d.differs}>{d.text}</span
                    >{#if di < cells.dims.length - 1}, {/if}{/each}){#if col.side.fold}
                      <span class="fold">×{col.side.fold}</span>{/if}</span
                  >
                {:else}
                  <span class="sig" class:hl={sidesRead(row.node) === 'differently'}
                    >{sideText(col.side)}</span
                  >
                {/if}
                {#if row.node.children.length && !row.expanded && row.node.differing}
                  <span class="differing">({row.node.differing.toLocaleString()} differ)</span>
                {/if}
              </svelte:element>
            {:else}
              <span class="cell gap" role="gridcell" aria-label="not in this checkpoint"></span>
            {/if}
          {/each}
        </div>
      {/each}
      <div style="height:{Math.max(0, (rows.length - first - slice.length) * ROW_H)}px"></div>
      {#if emptyNote}
        <p class="none dim" role="status">{emptyNote}</p>
      {/if}
    </div>
  {/if}
</div>

<style>
  .compare {
    height: 100%;
    display: flex;
    flex-direction: column;
    padding: 12px 16px;
    gap: 10px;
    min-height: 0;
  }
  /* No `.picker` / dropdown-`.caret` rules here: the box and its popup are `CheckpointPicker`, which
     owns both. The pair that outlived the extraction was not merely dead — `.caret` is also the tree
     rows' twisty, so every ▾ / ▸ / · in the comparison was drawn in a 1ch-wide bordered box with its
     glyph pushed out of it. */
  /* The boxes are `TextField`, which owns their look — see that component. */
  .pick {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .pick label {
    color: var(--fg-dim);
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
  .err {
    flex: 0 0 auto;
    margin: 0;
    font-size: 12px;
  }
  .err {
    color: var(--danger);
    white-space: pre-wrap;
  }
  /* The two column headers sit over the columns they name. */
  /* A tree being replaced stays readable — it is the same comparison, folded the other way. */
  .head.updating {
    opacity: 0.7;
  }
  .updating-note {
    grid-column: 1 / -1;
    margin: 0 0 4px;
    font-size: 11px;
  }
  .head {
    flex: 0 0 auto;
    display: grid;
    grid-template-columns: 1.6em minmax(0, 1fr) minmax(0, 1fr);
    gap: 8px;
    align-items: baseline;
    font-size: 12px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--border);
  }
  .head .side {
    grid-column: span 1;
    min-width: 0;
    word-break: break-all;
  }
  .head .old {
    grid-column: 2;
    color: var(--danger);
  }
  .head .new {
    grid-column: 3;
    color: var(--ok, #3fb950);
  }
  /* The summary spans both columns: it is about the pair, not about either side. */
  .head .summary {
    grid-column: 1 / -1;
    display: flex;
    flex-direction: column;
    gap: 7px;
    margin: 6px 0 2px;
  }
  .counts {
    display: flex;
    align-items: baseline;
    gap: 6px;
    flex-wrap: wrap;
  }
  .nav {
    font-size: 12px;
  }
  /* The "nothing lines up" banner: the sentence, then the two things that fix it. Background fill,
     no border — the treatment every other panel here uses. */
  .disjoint {
    margin: 6px 0 2px;
    padding: 7px 10px;
    border-radius: 4px;
    background: color-mix(in srgb, var(--warn) 12%, var(--bg-panel));
  }
  .disjoint p {
    margin: 0 0 6px;
    font-weight: 600;
  }
  .fixes {
    display: flex;
    align-items: baseline;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 12px;
  }
  .fixes button {
    font: inherit;
    font-size: 12px;
    color: var(--fg);
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 9px;
    cursor: pointer;
  }
  .fixes button:hover {
    border-color: var(--accent);
    color: var(--accent);
  }
  /* Two stats, aligned: label, the two values, and the change as its own piece. */
  .stats {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 28px;
    margin: 0;
  }
  .stat {
    display: flex;
    align-items: baseline;
    gap: 8px;
  }
  .stats dt {
    color: var(--fg-dim);
    font-size: 12px;
  }
  .stats dd {
    margin: 0;
    display: flex;
    align-items: baseline;
    gap: 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }
  .stats .from,
  .stats .arrow {
    color: var(--fg-dim);
  }
  .stats .to {
    color: var(--fg);
  }
  /* The number you came for. Neutral-bright rather than green/red: a checkpoint growing is neither
     good nor bad, and colouring it as if it were would be an opinion. */
  .stats .delta {
    color: var(--accent);
    font-weight: 600;
  }
  .stats .delta.same {
    color: var(--fg-dim);
    font-weight: 400;
  }
  .caption {
    margin: 0;
    font-size: 12px;
  }
  .caption .key {
    margin-left: 10px;
    white-space: nowrap;
  }
  .caption .removed {
    color: var(--danger);
    font-weight: 600;
  }
  .caption .added {
    color: var(--ok, #3fb950);
    font-weight: 600;
  }
  .caption .changed {
    color: var(--warn);
    font-weight: 600;
  }
  /* The one sentence that explains a comparison of two unrelated checkpoints. Spans both columns,
     because it is about the pair rather than either side. */
  .head .verdict {
    grid-column: 1 / -1;
    margin: 4px 0 0;
    font-size: 12.5px;
    color: var(--warn);
  }
  /* Background fill, no border — the same treatment as the other banners here. */
  .stale {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
    margin: 8px 0;
    padding: 10px 12px;
    border-radius: 4px;
    background: color-mix(in srgb, var(--warn) 14%, var(--bg-panel));
    font-size: 12.5px;
  }
  .stale strong {
    color: var(--warn);
    font-size: 13px;
  }
  /* Background fill, no border — the treatment the palette and the filter bar use. */
  .identical {
    grid-column: 1 / -1;
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin: 6px 0 2px;
    padding: 7px 10px;
    border-radius: 4px;
    background: color-mix(in srgb, var(--ok) 14%, var(--bg-panel));
  }
  .identical strong {
    color: var(--ok);
    font-size: 13px;
  }
  /* Always shown, not a tooltip: this is the sentence that keeps "identical" from being read as
     "the weights match". */
  .identical .what {
    color: var(--fg-dim);
    font-size: 12px;
  }
  .none {
    margin: 10px 2px;
    font-size: 12.5px;
  }
  .head .aside {
    grid-column: 1 / -1;
    margin: 2px 0 0;
    font-size: 12px;
  }
  /* Background fill, no border — the treatment used for the palette and the identical banner. */
  .busy {
    flex: 0 0 auto;
    padding: 9px 11px;
    border-radius: 4px;
    background: color-mix(in srgb, var(--warn) 12%, var(--bg-panel));
    font-size: 12.5px;
  }
  .busy p {
    margin: 0 0 7px;
  }
  .busy .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    color: var(--fg);
  }
  .acts {
    display: flex;
    gap: 8px;
  }
  .controls {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    font-size: 12px;
  }
  .controls .only {
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--fg-dim);
    cursor: pointer;
  }
  .rowcount {
    margin-left: auto;
    font-variant-numeric: tabular-nums;
  }
  /* An action inside a sentence: the message names a checkpoint, this opens it. */
  .link {
    font: inherit;
    font-size: 12px;
    color: var(--accent);
    background: none;
    border: none;
    padding: 0 2px;
    text-decoration: underline;
    cursor: pointer;
  }
  .rows {
    flex: 1 1 auto;
    min-height: 0;
    overflow: auto;
  }
  /* One grid per row, so the two side columns line up down the whole list — the alignment is the
     point of the view, and letting each row size its own columns would lose it. */
  /* Gutter, then two equal panes — neither side is the one being read, so neither gets more room. */
  .row {
    /* Fixed, because the virtualization above computes positions from it. */
    height: 22px;
    display: grid;
    grid-template-columns: 1.6em minmax(0, 1fr) minmax(0, 1fr);
    gap: 8px;
    align-items: baseline;
    font-size: 13px;
  }
  .row.cursor {
    background: var(--bg-hover);
  }
  .mark {
    text-align: center;
    color: var(--fg-dim);
  }
  /* **A one-sided row is loud; a changed row is quiet.**
     In a re-quantization almost every row is `~`, so a `+` or `-` among them — the rows that say a
     tensor is *missing from one side*, which is the thing you least want to miss — was one dim glyph in
     a column of them. The rare case gets a tinted band and a coloured edge; the common one keeps its
     mark and nothing more, or the whole tree would be striped. */
  .row.only_new {
    background: color-mix(in srgb, var(--ok, #3fb950) 12%, transparent);
    box-shadow: inset 2px 0 var(--ok, #3fb950);
  }
  .row.only_old {
    background: color-mix(in srgb, var(--danger) 12%, transparent);
    box-shadow: inset 2px 0 var(--danger);
  }
  .row.only_new .mark,
  .row.only_new .cell.new .nm {
    color: var(--ok, #3fb950);
    font-weight: 600;
  }
  .row.only_old .mark,
  .row.only_old .cell.old .nm {
    color: var(--danger);
    font-weight: 600;
  }
  .row.changed .mark {
    color: var(--warn);
  }
  /* One tree row: indent comes from the inline padding, then twisty, name, signature. */
  .cell {
    display: flex;
    align-items: baseline;
    gap: 6px;
    min-width: 0;
    text-align: left;
    font: inherit;
    color: var(--fg);
    background: none;
    border: none;
    padding: 1px 0;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
  }
  /* A side with no row here: an empty cell holding the column open, with nothing to press. */
  .cell.gap,
  .cell.inert {
    cursor: default;
  }
  .nm {
    flex: 0 1 auto;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .sig {
    flex: 0 0 auto;
    font-size: 12px;
    color: var(--fg-dim);
  }
  /* **Only what differs is coloured**, and in its side's colour, so a pair reads as before → after
     *and* points at the part that moved. Everything else in the signature stays dim: a row whose two
     sides read the same has nothing to point at. */
  .cell.old .hl {
    color: var(--danger);
    font-weight: 600;
  }
  .cell.new .hl {
    color: var(--ok, #3fb950);
    font-weight: 600;
  }
  .row.only_old .cell.old .nm {
    color: var(--danger);
  }
  .row.only_new .cell.new .nm {
    color: var(--ok, #3fb950);
  }
  /* What a fold stands for (`×256`): a note about the row, not part of the signature. */
  .fold {
    color: var(--warn);
  }
  /* How many identical layers a family row stands for — the same `×N` shape, since it answers the
     same question about a different kind of fold. */
  .family {
    flex: 0 0 auto;
    font-size: 12px;
    color: var(--accent);
  }
  /* The tree row's twisty (▾ / ▸) or tensor dot (·) — the terminal's glyphs, in a fixed slot of the
     same width as the main tree's, so the names down a column start in the same place. */
  .caret {
    flex: 0 0 12px;
    text-align: center;
    color: var(--fg-dim);
  }
  .differing {
    color: var(--warn);
    font-size: 12px;
  }
  kbd {
    font: inherit;
    font-size: 11px;
    padding: 0 3px;
    border: 1px solid var(--border);
    border-radius: 3px;
  }
</style>
