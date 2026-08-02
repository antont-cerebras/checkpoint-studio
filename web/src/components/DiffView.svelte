<script lang="ts">
  // The compare screen: a structural diff of the served checkpoint against another one
  // on the server's filesystem. Mirrors the terminal UI's `ui/diff.rs` — same sections in
  // the same order, same +/-/~ markers, same green/red/yellow, same one-line verdict (the
  // server computes that string, so the two UIs cannot word it differently).
  //
  // Structure only: names, dtypes and shapes. The value comparison reads every byte of
  // both checkpoints, so it stays on the CLI where it has a progress bar — the footer
  // hands over the exact command, extended with --values.
  import { diffReport, loadReport, reportError, reportWait } from '../stores/report';
  import { comparison } from '../stores/compare';
  import { humanCount, humanSize, totalsLine } from '../lib/format';
  import { openDetail } from '../stores/view';
  import type { CompareScreen } from '../lib/hash';
  import { copyText } from '../lib/clipboard';
  import { identicalNote } from '../lib/difftree';
  import Section from './Section.svelte';
  import DiffRow from './DiffRow.svelte';
  import { byChangeKind, changeKindLabel } from '../lib/difflines';
  import type { DiffScopeParams } from '../lib/diffscope';
  import LoadingBar from './LoadingBar.svelte';
  import TextField from './TextField.svelte';
  import FamilyToggle from './FamilyToggle.svelte';
  import MoreRows from './MoreRows.svelte';
  import { TALLY_MEANS, tallyTitle } from '../lib/tallywords';
  import { isEditable } from '../lib/keys';
  import DiffChip from './DiffChip.svelte';

  /** The selection, from the URL — so a scoped report is a link you can send. */
  export let scope: DiffScopeParams | undefined = undefined;
  /**
   * Which way round to read the pair: `false` compares the baseline (old) with the open checkpoint
   * (new), `true` turns it round.
   *
   * A diff is directional — what was added one way is removed the other — and the side-by-side view has
   * always had a swap. This one had none, so seeing the same pair the other way meant editing the URL
   * by hand. In the URL, like every other view state here, so a swapped report is still a link.
   */
  export let swapped = false;
  /**
   * `--full`: every tensor, rather than index-templated families collapsed onto one row.
   *
   * Off by default, which is the terminal's default too: 62 rows differing only by a layer number say
   * less than one row saying `model.layers.{0-61}.inv_freq_default (×62)`. In the URL because it changes
   * what the report shows — and because the command offered at the bottom has to carry it.
   */
  export let full = false;
  /** Sections folded away, by key — from the URL, so a reload lands on the report as it was read. */
  export let closed: string[] = [];
  /**
   * Change one thing about how this comparison is read.
   *
   * Injected by `ComparePage`, which holds the pair, the direction and the scope: a view that
   * navigated on its own would have to re-state all of them, and the one that did dropped three —
   * so *Browse these two side by side* arrived at a different comparison than the one being read.
   */
  export let onNavigate: (change: Partial<Omit<CompareScreen, 'kind'>>, replace?: boolean) => void;

  /**
   * How many rows of a section to draw before offering the rest on request.
   *
   * A re-quantization adds tens of thousands of tensors, and every one of them was in the DOM:
   * 125,081 nodes and 5.8 s to paint for one report. Nobody reads 31,247 names in a flat list — the
   * first screenful tells you what happened, and the count in the heading tells you how much of it
   * there is. The side-by-side view is where a comparison that size is actually navigable, so the
   * header links to it.
   */
  const SECTION_CAP = 200;
  /** Past this many withheld rows, offer the tree instead of a longer flat list. */
  const BROWSE_AT = 2000;
  /** Which sections the reader has asked to see in full, by heading. */
  let showAll: Record<string, boolean> = {};

  /** Keyed by section, and every key present: `bind:open` needs a boolean, not `boolean | undefined`. */
  type SectionKey = 'tensors_added' | 'tensors_removed' | 'tensors_changed' | 'metadata' | 's3';
  const SECTION_KEYS: SectionKey[] = [
    'tensors_added',
    'tensors_removed',
    'tensors_changed',
    'metadata',
    's3',
  ];

  /**
   * Which sections are open — **from the URL**, so folding survives a reload and travels in a link.
   *
   * Folding a 31,247-row section away and having it spring back on refresh is the same complaint as a
   * filter that does not survive one: it is state you set deliberately. Only the *closed* keys are
   * carried, so an untouched report has a clean URL.
   */
  $: openSections = Object.fromEntries(
    SECTION_KEYS.map((k) => [k, !closed.includes(k)]),
  ) as Record<SectionKey, boolean>;

  /** Fold or unfold a section, through the URL — which is what makes it survive a reload. */
  function setOpen(key: SectionKey, open: boolean) {
    const next = open ? closed.filter((k) => k !== key) : [...closed.filter((k) => k !== key), key];
    onNavigate({ closed: next });
  }

  /** The count strip's entries, in report order. */
  const SECTIONS = [
    { key: 'tensors_added', title: 'Added', tone: 'added', means: 'added' },
    { key: 'tensors_removed', title: 'Removed', tone: 'removed', means: 'removed' },
    { key: 'tensors_changed', title: 'Changed', tone: 'changed', means: 'changed' },
    { key: 'metadata', title: 'Metadata', tone: 'meta', means: 'metadata' },
  ] as const;

  /** How many rows a chip is standing for — the *shown* count, so it agrees with the heading it
   * scrolls to while a filter is on. */
  function sectionCount(
    key: (typeof SECTIONS)[number]['key'],
    _report: unknown,
    shown: { added: unknown[]; removed: unknown[]; changed: unknown[]; metaShown: number },
  ): number {
    switch (key) {
      case 'tensors_added':
        return shown.added.length;
      case 'tensors_removed':
        return shown.removed.length;
      case 'tensors_changed':
        return shown.changed.length;
      case 'metadata':
        return shown.metaShown;
    }
  }

  /**
   * Open a section from its chip, so a chip is a way *in* rather than a decoration.
   *
   * Through the URL, like the heading's own toggle. It used to assign to `openSections`, which is
   * *derived* from the URL's `closed` list — so the chip's effect lasted until anything else
   * recomputed it (a filter, a swap, the families checkbox) and vanished on reload, while the heading
   * beside it survived both.
   */
  function reveal(key: SectionKey) {
    setOpen(key, true);
  }

  /**
   * The heading's toggle for one section, as a typed function.
   *
   * Curried rather than written inline at each `<Section>`: a callback prop's parameter has no type
   * inside the template (a `.svelte` import carries none outside svelte-check), so `(v) => setOpen(k, v)`
   * passes an `any` to a `boolean`, and a template expression cannot carry an annotation to fix it.
   */
  function fold(key: SectionKey): (open: boolean) => void {
    return (open: boolean) => setOpen(key, open);
  }

  /**
   * The grouped sections, narrowed by the same needle as the flat ones.
   *
   * Filtering the *grouped* rows by name means a needle matches a family's display name
   * (`model.layers.{0-61}.…`), which is what is on screen — filtering the flat rows and re-grouping
   * would show families whose visible name does not contain what was typed.
   */
  $: groupedAdded = keep(result?.grouped.tensors_added ?? [], (g) => g.name, match);
  $: groupedRemoved = keep(result?.grouped.tensors_removed ?? [], (g) => g.name, match);
  $: groupedChanged = keep(result?.grouped.tensors_changed ?? [], (g) => g.name, match);
  /** What the collapse is worth, said plainly: `18 rows for 809 tensors`. */
  $: familyRows = groupedAdded.length + groupedRemoved.length + groupedChanged.length;
  $: flatRows = added.length + removed.length + changed.length;

  /** How many kinds of change the changed section holds — one kind needs no sub-headings. */
  $: kinds = byChangeKind(changed).length;
  /** The rows actually on screen, by kind: the groups' *totals* come from the whole filtered section, so
   * a heading cannot claim the cap's 200 rows are all there is. */
  $: shownByKind = new Map(
    byChangeKind(capped(changed, 'tensors_changed', showAll)).map((g) => [g.kind, g.rows]),
  );

  /**
   * Narrow every section to names containing this, case-insensitively.
   *
   * A 31,247-row list with no way to ask "what happened to `layers.3`" is a list you scroll past. The
   * side-by-side view has fold and filter controls; this one had none, which made the report the worse
   * of the two for exactly the comparisons big enough to need help.
   */
  let needle = '';
  $: match = needle.trim().toLowerCase();
  /**
   * `m` is a parameter, not a closure over `match` — and that is the whole bug this signature exists to
   * prevent.
   *
   * Svelte works out a reactive statement's dependencies from the *identifiers written in it*. Reading
   * `match` inside this function's body is invisible to that analysis, so `$: added = keep(list, name)`
   * depended on `list` alone and never recomputed when the box was typed in. The filter looked live —
   * the clear button appeared on the first keystroke — and changed nothing, which is worse than having
   * no filter at all. Taking the needle as an argument puts it in the statement, where the compiler can
   * see it.
   */
  function keep<T>(items: T[], nameOf: (x: T) => string, m: string): T[] {
    return m ? items.filter((x) => nameOf(x).toLowerCase().includes(m)) : items;
  }
  /** `12 of 31,247` while filtering, plain `31,247` otherwise — so a narrowed count never reads as
   * the whole truth about the checkpoint. */
  function countLabel(shown: number, total: number): string {
    return shown === total
      ? total.toLocaleString()
      : `${shown.toLocaleString()} of ${total.toLocaleString()}`;
  }
  $: added = keep(report?.tensors_added ?? [], ([n]) => n, match);
  $: removed = keep(report?.tensors_removed ?? [], ([n]) => n, match);
  $: changed = keep(report?.tensors_changed ?? [], (c) => c.name, match);
  // Metadata is filtered too. It was not, so a needle matching no tensor still left the metadata rows
  // on screen — which made the filter look like it had missed something rather than found nothing.
  $: metaAdded = keep(report?.meta_added ?? [], ([n]) => n, match);
  $: metaRemoved = keep(report?.meta_removed ?? [], ([n]) => n, match);
  $: metaChanged = keep(report?.meta_changed ?? [], (c) => c.name, match);
  $: metaShown = metaAdded.length + metaRemoved.length + metaChanged.length;

  let copied = false;

  // Re-run whenever the URL's baseline, scope *or direction* changes (including back/forward), not
  // just on mount — the screen is addressable, so arriving at it twice with different parameters has
  // to show different reports. Each of the three is in the dependency list because each is in the URL.
  // `full` is a dependency because the *command* the server offers depends on it; `closed` is not,
  // since folding is a display choice this component makes on data it already has.
  //
  // The result lives in a store (`stores/report`) rather than here: it is one of three readings of a
  // comparison now, and the Data view sizes its run from the totals this response carries.
  $: void loadReport($comparison?.id ?? null, scope, swapped, full);
  $: result = $diffReport;
  $: error = $reportError;
  $: waitStarted = $reportWait;
  $: loading = $reportWait !== null;

  /** Collapse families, or show every tensor. Through the URL, like the fold state. */
  function setFull(everything: boolean) {
    onNavigate({ full: everything });
  }

  /**
   * `×256 → ×1` for a row an alignment folded, or `''`.
   *
   * The fold is the point of the alignment, so it belongs on the row rather than in a note above the
   * list: 256 per-expert tensors *are* the one fused tensor beside them, and a shape that gains a
   * leading dimension is otherwise an unexplained change of rank.
   */
  function foldNote(name: string, folded: Record<string, [number, number]>): string {
    const parts = folded[name];
    if (!parts || parts[0] === parts[1]) return '';
    return `×${parts[0].toLocaleString()} → ×${parts[1].toLocaleString()}`;
  }

  function copyCommand() {
    if (result && copyText(result.command)) {
      copied = true;
      setTimeout(() => (copied = false), 1500);
    }
  }

  $: report = result?.report ?? null;
  /**
   * The matching-checkpoints banner, from the same function the side-by-side view uses.
   *
   * The report's sections map one-to-one onto the tally's two halves — the correspondence a differential
   * test asserts — so this is not a second definition of "identical".
   */
  $: identical = report
    ? identicalNote({
        tensors: {
          same: report.tensors_unchanged,
          changed: report.tensors_changed.length,
          only_old: report.tensors_removed.length,
          only_new: report.tensors_added.length,
        },
        metadata: {
          same: report.meta_unchanged,
          changed: report.meta_changed.length,
          only_old: report.meta_removed.length,
          only_new: report.meta_added.length,
        },
      })
    : null;
  $: metaTotal = report
    ? report.meta_added.length + report.meta_removed.length + report.meta_changed.length
    : 0;

  /**
   * The first `SECTION_CAP` rows of a section, unless the reader asked for all of them.
   *
   * `all` is passed in for the same reason `keep` takes its needle: a template expression is re-run when
   * an identifier *in it* changes, and `showAll` read from inside the body was not one — so pressing
   * the tail's button updated the state and redrew nothing.
   *
   * `items` is the already-filtered list, so both the cap and the count below are relative to what the
   * filter matched rather than to the whole section.
   */
  function capped<T>(items: T[], key: string, all: Record<string, boolean>): T[] {
    return all[key] ? items : items.slice(0, SECTION_CAP);
  }

  /**
   * How many **rows** a section is holding back.
   *
   * A count of rows, not of tensors. Folded, a section draws one row per family and this counted the
   * tensors behind them — five rows on screen under "show the remaining 79,532", and pressing it added
   * nothing, because there were no more rows to add. What is offered has to be what arrives.
   */
  function withheldRows(rows: number, key: string, all: Record<string, boolean>): number {
    return all[key] ? 0 : Math.max(0, rows - SECTION_CAP);
  }
  /** Per section, of the list it actually draws: families when folded, tensors under `--full`. */
  $: moreAdded = withheldRows(full ? added.length : groupedAdded.length, 'tensors_added', showAll);
  $: moreRemoved = withheldRows(
    full ? removed.length : groupedRemoved.length,
    'tensors_removed',
    showAll,
  );
  $: moreChangedFolded = withheldRows(groupedChanged.length, 'tensors_changed', showAll);
  $: moreChangedFull = withheldRows(changed.length, 'tensors_changed', showAll);
  /** Show every row of a section — the tail's own button. */
  function revealAll(key: SectionKey) {
    showAll = { ...showAll, [key]: true };
  }

  /**
   * The rows `n`/`N` step through: every difference **drawn**, in the order it is drawn.
   *
   * Drawn, not merely present: a folded section shows one row per family and a closed one shows none,
   * and a cursor that moved to a row nobody can see would scroll the page to nothing. So the list
   * follows the screen — the fold, the filter, the cap and the folds of the sections themselves.
   *
   * The footer has advertised `n/N next/prev difference` on this screen from the start and only the
   * aligned tree obeyed it, which is the report this closes: on the summary the keys did nothing.
   */
  // A flat row is identified by its tensor name (unique in a checkpoint); a *grouped* row by its
  // position, because two of them can share a display name — see the each-block below.
  $: cursorRows = [
    ...(openSections.tensors_added
      ? full
        ? capped(added, 'tensors_added', showAll).map(([n]) => `tensors_added:${n}`)
        : capped(groupedAdded, 'tensors_added', showAll).map((_, i) => `tensors_added:${i}`)
      : []),
    ...(openSections.tensors_removed
      ? full
        ? capped(removed, 'tensors_removed', showAll).map(([n]) => `tensors_removed:${n}`)
        : capped(groupedRemoved, 'tensors_removed', showAll).map((_, i) => `tensors_removed:${i}`)
      : []),
    ...(openSections.tensors_changed
      ? full
        ? byChangeKind(changed)
            .flatMap((g) => shownByKind.get(g.kind) ?? [])
            .map((c) => `tensors_changed:${c.name}`)
        : capped(groupedChanged, 'tensors_changed', showAll).map((_, i) => `tensors_changed:${i}`)
      : []),
  ];
  /** Which of them the cursor is on; `''` for none yet. Kept as the row's own id rather than an index,
   * so folding a section or narrowing the filter moves the cursor with the row instead of onto a
   * different one. */
  let cursorRow = '';
  /** Step to the next difference (`+1`) or the previous one (`-1`), wrapping at the ends. */
  function step(delta: 1 | -1) {
    if (cursorRows.length === 0) return;
    const at = cursorRows.indexOf(cursorRow);
    // From nowhere, `n` starts at the first row and `N` at the last — what the terminal does.
    const next =
      at < 0
        ? delta === 1
          ? 0
          : cursorRows.length - 1
        : (at + delta + cursorRows.length) % cursorRows.length;
    cursorRow = cursorRows[next] ?? '';
    // Centred rather than merely "into view": a row that lands under the sticky controls is a row you
    // have to scroll to anyway.
    document
      .querySelector(`[data-row="${CSS.escape(cursorRow)}"]`)
      ?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  /**
   * `n`/`N` step between differences, and `k` folds the families.
   *
   * On this view as well as on the aligned tree, because both are readings of one comparison and the
   * footer promises the keys for the screen rather than for one of its tabs. `s` (swap) belongs to the
   * page, which owns the pair; `k` is *also* the page's state, but it is handled here and there for the
   * same reason `n`/`N` are — the view that has rows is the one that knows what the keys mean.
   */
  function onKeydown(e: KeyboardEvent) {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (isEditable(e.target)) return;
    if (e.key === 'n') {
      e.preventDefault();
      step(1);
    } else if (e.key === 'N') {
      e.preventDefault();
      step(-1);
    }
  }


  /**
   * Open a tensor's detail view from a report row.
   *
   * Only for names on the *new* side. The detail screen reads the served checkpoint, and this report
   * compares the baseline against it — so an added or changed tensor is there to be opened, while a
   * removed one exists only in the baseline and would show a different checkpoint's numbers under its
   * name. Same rule as the side-by-side view's `clickOutcome`.
   */
  function open(name: string) {
    openDetail(name);
  }

</script>

<svelte:window on:keydown={onKeydown} />

<div class="diff">
  <!-- No pair box and no scope bar here: they belong to the *comparison*, and `ComparePage` owns
       them. This is one reading of it — the categorised summary. -->
  {#if result && report}
    <div class="controls">
      <!-- *Find in results*, not "filter": the scope bar above decides what the server compared, and
           two controls both called some kind of filter is how a reader comes to believe this one
           changed the comparison. -->
      <label for="diff-filter" class="dim">Find in results</label>
      <TextField
        id="diff-filter"
        bind:value={needle}
        grow={false}
        width="28ch"
        placeholder="part of a tensor name"
        spellcheck="false"
        autocomplete="off"
      />
      {#if match}
        <button type="button" class="quiet" on:click={() => (needle = '')}>clear</button>
      {/if}
      <!-- The terminal's default is collapsed, and for a reason: 62 rows differing only by a layer
           number say less than one row saying `model.layers.{0-61}.…  (×62)`. `--full` is the switch,
           and it is in the URL, so a link shows what the sender was looking at. -->
      <FamilyToggle {full} onChange={setFull} />
      <span class="dim rowcount">
        · {full ? 'every tensor' : `${familyRows.toLocaleString()} rows for ${flatRows.toLocaleString()} tensors`}
      </span>
    </div>
  {/if}

  <!-- **Nothing is read here.** Both checkpoints are already in the comparison the server holds; this
       is the server re-deriving the report from them, which is milliseconds. It used to say "reading
       the baseline" and replace the whole report with a progress bar — so turning the pair round, or
       narrowing it, looked like starting over on something that had just been read.
       A first report has nothing to show yet and gets the bar; a report already on screen stays, with
       a word to say a newer one is coming. -->
  {#if loading && !result}
    <LoadingBar label="building the report" progress={waitStarted} />
  {:else if error}
    <p class="error" role="alert">{error}</p>
  {:else if result && report}
    <header class:updating={loading}>
      {#if loading}
        <p class="updating-note dim" role="status">updating…</p>
      {/if}
      <!-- No `old …` / `new …` block and no swap button here.
           Both were a second copy of what the page already shows: the two address boxes are the pair,
           in the order it is being read, with the one swap control between them. Repeating the
           addresses under them cost four lines of the report to say what was two inches above it, and
           the second swap button meant two ways to flip one thing. What is *not* duplicated — which
           side the server actually read as old — is still enforced here: the labels come from the
           server's `swapped`, and the boxes above are drawn from the same bit. -->
      {#if identical}
        <div class="identical" role="status">
          <strong>{identical.headline}</strong>
          <span class="what">{identical.detail}</span>
        </div>
      {/if}
      <!-- What this report is, before its numbers: names, dtypes and shapes — not values. It used to
           be the first line of the footer, below every section, which is the one place a reader who
           has taken the counts at face value will never look. -->
      <p class="scope-note">
        Structure only — names, dtypes and shapes.
        <button type="button" class="link" on:click={() => onNavigate({ view: 'data' })}>
          Compare the numbers →
        </button>
      </p>
      <!-- The two overall totals, worded as the terminal words them — including the delta and its
           percentage, which this line used to leave the reader to work out from `451.8 GiB → 32 B`.
           `totalsLine` is contracted against the Rust original case by case (lib/parity.test.ts).
           The *labels* come from the server: under a filter these numbers cover the matched tensors,
           and `size:` would then read as the checkpoint's size. -->
      <p class="delta dim mono">
        {totalsLine(result.totals_labels.size, report.old_bytes, report.new_bytes, humanSize)}
      </p>
      <p class="delta dim mono">
        {totalsLine(result.totals_labels.params, report.old_params, report.new_params, humanCount)}
      </p>
      <!-- When each side has one (an s3-vs-s3 pair), which is newer. The server's line, humanised
           by the rule the terminal uses rather than by a second one here. -->
      {#if result.modified_line}<p class="delta dim mono">{result.modified_line}</p>{/if}
    </header>

    <!-- A strip of counts, each a link to its section: the shape of the comparison in one line, and a
         way into the part you care about. On a 31,247-row report the alternative is scrolling to find
         out whether anything was removed. -->
    <!-- The verdict used to be a line of prose above this: `0 unchanged; 79,732 added, 558 removed,
         375 changed`, immediately followed by chips saying the same four numbers. The strip is the
         better of the two — each count is a way *into* its section — so it carries `unchanged` too
         now, and the terminal's sentence stays as the strip's tooltip. -->
    <nav class="tally" aria-label="Sections" title={result.verdict}>
      <!-- Not a button: there is no "unchanged" section to go to. It is here because a comparison
           with 79,732 changes and 0 unchanged tensors is a different thing from one with 80,000
           unchanged, and the chips are where that is read. -->
      <DiffChip
        tone="same"
        label="Unchanged"
        order="label-first"
        count={report.tensors_unchanged.toLocaleString()}
        empty={report.tensors_unchanged === 0}
        title={tallyTitle('unchanged', report.tensors_unchanged)}
      />
      {#each SECTIONS as sec (sec.key)}
        {@const n = sectionCount(sec.key, report, { added, removed, changed, metaShown })}
        <DiffChip
          tone={sec.tone}
          label={sec.title}
          order="label-first"
          count={n.toLocaleString()}
          empty={n === 0}
          title={tallyTitle(sec.means, n, true)}
          onPick={() => reveal(sec.key)}
        />
      {/each}
    </nav>

    <Section
      title="Tensors added"
      titleHint={TALLY_MEANS.added}
      count={countLabel(added.length, report.tensors_added.length)}
      tone="added"
      open={openSections.tensors_added}
      onToggle={fold('tensors_added')}
    >
      {#if !added.length}<p class="none dim">none</p>{/if}
      {#if full}
        {#each capped(added, 'tensors_added', showAll) as [name, s] (name)}
          <DiffRow
            mark="+"
            {name}
            neu={s}
            onOpen={open}
            rowId="tensors_added:{name}"
            cursor={cursorRow === `tensors_added:${name}`}
          />
        {/each}
      {:else}
        <!-- A family row stands for many tensors, so it opens none of them: `Collapse families` off is
             the way to a single tensor's detail. -->
        {#each capped(groupedAdded, 'tensors_added', showAll) as g, i (i)}
          <DiffRow
            mark="+"
            name={g.name}
            neu={g.sig}
            count={g.count}
            rowId="tensors_added:{i}"
            cursor={cursorRow === `tensors_added:${i}`}
          />
        {/each}
      {/if}
      <MoreRows
        n={moreAdded}
        onShowAll={() => revealAll('tensors_added')}
        onBrowse={moreAdded > BROWSE_AT ? () => onNavigate({ view: 'browse' }) : null}
      />
    </Section>

    <Section
      title="Tensors removed"
      titleHint={TALLY_MEANS.removed}
      count={countLabel(removed.length, report.tensors_removed.length)}
      tone="removed"
      open={openSections.tensors_removed}
      onToggle={fold('tensors_removed')}
    >
      {#if !removed.length}<p class="none dim">none</p>{/if}
      {#if full}
        {#each capped(removed, 'tensors_removed', showAll) as [name, s] (name)}
          <DiffRow
            mark="-"
            {name}
            old={s}
            why="only in the baseline, so there is nothing here to open"
            rowId="tensors_removed:{name}"
            cursor={cursorRow === `tensors_removed:${name}`}
          />
        {/each}
      {:else}
        {#each capped(groupedRemoved, 'tensors_removed', showAll) as g, i (i)}
          <DiffRow
            mark="-"
            name={g.name}
            old={g.sig}
            count={g.count}
            why="only in the baseline, so there is nothing here to open"
            rowId="tensors_removed:{i}"
            cursor={cursorRow === `tensors_removed:${i}`}
          />
        {/each}
      {/if}
      <MoreRows
        n={moreRemoved}
        onShowAll={() => revealAll('tensors_removed')}
        onBrowse={moreRemoved > BROWSE_AT ? () => onNavigate({ view: 'browse' }) : null}
      />
    </Section>

    <Section
      title="Tensors changed"
      titleHint={TALLY_MEANS.changed}
      count={countLabel(changed.length, report.tensors_changed.length)}
      tone="changed"
      open={openSections.tensors_changed}
      onToggle={fold('tensors_changed')}
    >
      {#if !changed.length}<p class="none dim">none</p>{/if}
      <!-- Grouped by *what* changed. A re-quantization changes the dtype of everything and the shape of
           the expert tensors only, so "624 dtype only, 123 dtype and shape" is the shape of what
           happened — visible without reading a row. One group is not worth a sub-heading. -->
      {#if !full}
        <!-- Grouped: one row per family, already sorted by the server. Kinds are not sub-headed here —
             a family row *is* the summary, and 18 of them need no further grouping. -->
        <!-- Keyed by position, not by name: two grouped rows can carry the *same* display name when one
             name template holds two signatures — `model.layers.{0-1}.experts.{0-1}.w  F32 → F16` beside
             `… F32 → BF16`, which is what a re-quantization that split a family looks like. A keyed
             `{#each}` on a value the server does not promise to be unique is a broken update waiting
             to happen. -->
        {#each capped(groupedChanged, 'tensors_changed', showAll) as g, i (i)}
          <DiffRow
            mark="~"
            name={g.name}
            old={g.old}
            neu={g.new}
            count={g.count}
            fold={g.fold ? `×${g.fold[0].toLocaleString()} → ×${g.fold[1].toLocaleString()}` : ''}
            rowId="tensors_changed:{i}"
            cursor={cursorRow === `tensors_changed:${i}`}
          />
        {/each}
        <MoreRows
          n={moreChangedFolded}
          onShowAll={() => revealAll('tensors_changed')}
          onBrowse={moreChangedFolded > BROWSE_AT ? () => onNavigate({ view: 'browse' }) : null}
        />
      {/if}
      {#each full ? byChangeKind(changed) : [] as group (group.kind)}
        {@const rows = shownByKind.get(group.kind) ?? []}
        {#if rows.length}
          {#if kinds > 1}
            <!-- `shown of total` when the cap is holding rows back, so a group's count never reads as
                 the whole of it — the same rule the section headings use. -->
            <p class="kind dim">
              {changeKindLabel(group.kind)}
              <span class="n">({countLabel(rows.length, group.rows.length)})</span>
            </p>
          {/if}
          {#each rows as c (c.name)}
            <DiffRow
              mark="~"
              name={c.name}
              old={c.old}
              neu={c.new}
              fold={foldNote(c.name, result.folded)}
              onOpen={open}
              rowId="tensors_changed:{c.name}"
              cursor={cursorRow === `tensors_changed:${c.name}`}
            />
          {/each}
        {/if}
      {/each}
      <!-- Inside `{#if full}`, like the rows it belongs to. Rendered unconditionally, a *folded* section
           drew this one as well as its own — two buttons, the second counting the 1,000 tensors behind
           the 500 rows on screen. Clicking it revealed nothing (the rows it was counting are not the
           rows being drawn) and the button vanished, which is how it was reported. -->
      {#if full}
        <MoreRows
          n={moreChangedFull}
          onShowAll={() => revealAll('tensors_changed')}
          onBrowse={moreChangedFull > BROWSE_AT ? () => onNavigate({ view: 'browse' }) : null}
        />
      {/if}
    </Section>

    <Section
      title="Metadata"
      titleHint={TALLY_MEANS.metadata}
      count={result.metadata_note ? '' : countLabel(metaShown, metaTotal)}
      note={result.metadata_note ? `not compared (${result.metadata_note})` : ''}
      tone="meta"
      open={openSections.metadata}
      onToggle={fold('metadata')}
    >
      {#if !metaShown && !result.metadata_note}<p class="none dim">none</p>{/if}
      {#each metaAdded as [name, v] (name)}
        <div class="row"><span class="mark added">+</span><span class="name mono">{name}</span
          ><span class="detail dim">{v.value}</span></div>
      {/each}
      {#each metaRemoved as [name, v] (name)}
        <div class="row"><span class="mark removed">-</span><span class="name mono">{name}</span
          ><span class="detail dim">{v.value}</span></div>
      {/each}
      {#each metaChanged as c (c.name)}
        <div class="row"><span class="mark changed">~</span><span class="name mono">{c.name}</span
          ><span class="detail dim">{c.old.value} → {c.new.value}</span></div>
      {/each}
      {#if report.meta_unchanged}
        <p class="none dim">{report.meta_unchanged} metadata entries unchanged</p>
      {/if}
    </Section>

    <!-- The objects themselves, for an s3-vs-s3 pair: ETag, size, checksums, tags. Rendered from the
         server's own lines (`S3Diff::summary_lines`), so the browser states exactly what the terminal
         states about what "unchanged" is worth here — and nothing about evidence is worded twice. -->
    {#if result.s3_lines?.length}
      {@const heading = result.s3_lines.find((l) => l.kind === 'heading')}
      <Section
        title={heading?.text ?? 'S3 objects'}
        tone="meta"
        open={openSections.s3}
      onToggle={fold('s3')}
      >
        {#each result.s3_lines.slice(0, SECTION_CAP) as line, i (i)}
          {#if line.kind !== 'heading'}
            <div class="row"><span class="s3 {line.kind}">{line.text}</span></div>
          {/if}
        {/each}
        {#if result.s3_lines.length > SECTION_CAP}
          <p class="none dim">
            … and {(result.s3_lines.length - SECTION_CAP).toLocaleString()} more object lines
          </p>
        {/if}
      </Section>
    {:else if result.s3_note}
      <!-- A note about *how the comparison was made*, not a result of it. Among the counts it read as
           a finding — and it was there twice, once here and once above the totals. -->
      <p class="method">{result.s3_note}</p>
    {/if}

    <footer>
      <!-- This is structure: names, dtypes and shapes. Comparing the *numbers* is the Data view —
           which used to say "run this in a terminal", while the other screen ran it in the browser.
           Both sentences were about the same pair and one of them was false. -->
      <!-- Only the command is left down here. What this report *is* — a structural comparison, with
           the numbers a tab away — belongs above the numbers it qualifies, not under 31,247 rows. -->
      {#if result.command}
        <p class="dim">The same comparison in a terminal:</p>
        <div class="cmd">
          <code>{result.command}</code>
          <button on:click={copyCommand}>{copied ? '✓ copied' : 'copy'}</button>
        </div>
      {/if}
    </footer>
  {/if}
</div>

<style>
  .diff {
    padding: 10px 14px;
    overflow: auto;
  }
  .controls {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
    font-size: 12px;
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
  header {
    margin-bottom: 14px;
  }
  /* A report being replaced stays readable — it is the same comparison, one turn behind. */
  header.updating {
    opacity: 0.7;
  }
  .updating-note {
    margin: 0 0 4px;
    font-size: 11px;
  }
  /* What the report covers, stated before its numbers. Quiet, but not dim-to-invisible: it is the
     difference between "these two are the same" and "the same in every way this looked at". */
  .scope-note {
    margin: 0 0 8px;
    padding: 5px 9px;
    border-radius: 4px;
    background: var(--bg-elev);
    color: var(--fg-dim);
    font-size: 12px;
  }
  /* A note about method, set apart from the findings by having a shape of its own. */
  .method {
    margin: 8px 0 0;
    padding: 5px 9px;
    border-radius: 4px;
    background: var(--bg-elev);
    color: var(--fg-dim);
    font-size: 12px;
  }
  /* Background fill, no border — matching the side-by-side view's banner and the command palette. */
  .identical {
    display: flex;
    flex-direction: column;
    gap: 2px;
    margin: 8px 0 2px;
    padding: 7px 10px;
    border-radius: 4px;
    background: color-mix(in srgb, var(--ok) 14%, var(--bg-panel));
  }
  .identical strong {
    color: var(--ok);
    font-size: 13px;
  }
  .identical .what {
    color: var(--fg-dim);
    font-size: 12px;
  }
  .delta {
    margin: 2px 0 0;
    font-size: 12px;
  }
  /* Tabular, so the two totals lines' numbers line up under each other as they do in a terminal. */
  .mono {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-variant-numeric: tabular-nums;
  }
  .row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 1px 0 1px 8px;
    font-size: 12.5px;
  }
  /* A row that leads somewhere. The report was a wall of inert spans: you could see that a tensor
     changed and had no way to go look at it. */
  /* The rest of a capped section, on request — see `SECTION_CAP`. */
  .more,
  .link {
    font: inherit;
    font-size: 12px;
    color: var(--accent);
    background: none;
    border: none;
    padding: 2px 0 2px 8px;
    text-decoration: underline;
    cursor: pointer;
  }
  .none {
    margin: 2px 0 2px 8px;
    font-size: 12px;
  }
  /* Green / red / yellow, matching both the terminal UI's palette and `diff`'s own
     ANSI output, so the same change is the same colour everywhere. */
  /* What a folded row stands for — the alignment's whole point, so it reads as a fact about the row
     rather than dim trailing detail. */
  /* An S3 object line, coloured by its kind exactly as the terminal paints it. */
  .s3 {
    word-break: break-all;
  }
  .s3.note {
    color: var(--fg-dim);
  }
  .s3.removed {
    color: var(--err, #e05c5c);
  }
  .s3.added {
    color: var(--ok, #4ec94e);
  }
  /* The count strip: the shape of the comparison in one line, and a way into each part. */
  .tally {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
    margin: 10px 0 12px;
  }
  /* One sub-heading per kind of change, inside the changed section. */
  .kind {
    margin: 8px 0 2px 8px;
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .kind .n {
    font-variant-numeric: tabular-nums;
    text-transform: none;
    letter-spacing: 0;
  }
  .row {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 1px 0 1px 8px;
    font-size: 12.5px;
  }
  .row .mark {
    flex: none;
    width: 1em;
    font-weight: 600;
  }
  .row .mark.added {
    color: var(--ok, #4ec94e);
  }
  .row .mark.removed {
    color: var(--err, #e05c5c);
  }
  .row .mark.changed {
    color: var(--warn, #d8b530);
  }
  .row .name {
    flex: 1 1 auto;
    min-width: 0;
    word-break: break-all;
  }
  .row .detail {
    white-space: nowrap;
  }
  .error {
    color: var(--err, #e05c5c);
  }
  .dim {
    color: var(--fg-dim);
  }
  footer {
    margin-top: 18px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
    font-size: 12px;
  }
  .cmd {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .cmd code {
    flex: 1;
    min-width: 0;
    padding: 4px 8px;
    overflow-x: auto;
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 3px;
    white-space: pre;
  }
</style>
