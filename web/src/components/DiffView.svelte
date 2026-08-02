<script lang="ts">
  // The compare screen: a structural diff of the served checkpoint against another one
  // on the server's filesystem. Mirrors the terminal UI's `ui/diff.rs` — same sections in
  // the same order, same +/-/~ markers, same green/red/yellow, same one-line verdict (the
  // server computes that string, so the two UIs cannot word it differently).
  //
  // Structure only: names, dtypes and shapes. The value comparison reads every byte of
  // both checkpoints, so it stays on the CLI where it has a progress bar — the footer
  // hands over the exact command, extended with --values.
  import { api } from '../lib/api';
  import { humanCount, humanSize, specHelp, totalsLine } from '../lib/format';
  import type { DiffResponse } from '../lib/types';
  import { navigate, openDetail } from '../stores/view';
  import { copyText } from '../lib/clipboard';
  import { identicalNote } from '../lib/difftree';
  import ScopeBar from './ScopeBar.svelte';
  import Section from './Section.svelte';
  import DiffRow from './DiffRow.svelte';
  import { byChangeKind, changeKindLabel } from '../lib/difflines';
  import { emptyScope, type DiffScopeParams } from '../lib/diffscope';
  import LoadingBar from './LoadingBar.svelte';
  import { startedNow, type Progress } from '../lib/progress';
  import { proxied, proxyHost, tree } from '../stores/server';
  import { shortSpec } from '../lib/loadstep';
  import TextField from './TextField.svelte';
  import SwapButton from './SwapButton.svelte';
  import FamilyToggle from './FamilyToggle.svelte';
  import DiffChip from './DiffChip.svelte';
  // The server reads the baseline checkpoint's headers before it can answer.
  let waitStarted: Progress | null = null;

  export let against: string;
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
   * How many rows of a section to draw before offering the rest on request.
   *
   * A re-quantization adds tens of thousands of tensors, and every one of them was in the DOM:
   * 125,081 nodes and 5.8 s to paint for one report. Nobody reads 31,247 names in a flat list — the
   * first screenful tells you what happened, and the count in the heading tells you how much of it
   * there is. The side-by-side view is where a comparison that size is actually navigable, so the
   * header links to it.
   */
  const SECTION_CAP = 200;
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
    navigate({ kind: 'diff', against, scope, swapped, full, closed: next });
  }

  /** The count strip's entries, in report order. */
  const SECTIONS = [
    { key: 'tensors_added', title: 'Added', tone: 'added' },
    { key: 'tensors_removed', title: 'Removed', tone: 'removed' },
    { key: 'tensors_changed', title: 'Changed', tone: 'changed' },
    { key: 'metadata', title: 'Metadata', tone: 'meta' },
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

  let result: DiffResponse | null = null;
  let error: string | null = null;
  let loading = false;
  /** The path in the input, which may differ from the one being shown. */
  let draft = against;
  let copied = false;

  // Re-run whenever the URL's baseline, scope *or direction* changes (including back/forward), not
  // just on mount — the screen is addressable, so arriving at it twice with different parameters has
  // to show different reports. Each of the three is in the dependency list because each is in the URL.
  // `full` is a dependency because the *command* the server offers depends on it; `closed` is not, since
  // folding is a display choice this component makes on data it already has.
  $: void load(against, scope, swapped, full);

  async function load(
    path: string,
    sel: DiffScopeParams | undefined,
    flip: boolean,
    everything: boolean,
  ) {
    if (!path) return;
    draft = path;
    loading = true;
    waitStarted = startedNow();
    error = null;
    try {
      result = await api.diff(path, sel, flip, everything);
    } catch (e) {
      result = null;
      error = e instanceof Error ? e.message : String(e);
    } finally {
      loading = false;
    }
  }

  function submit() {
    const p = draft.trim();
    // Navigating rather than loading directly keeps the URL the source of truth, so the
    // comparison is shareable and survives a reload.
    if (p) navigate({ kind: 'diff', against: p, scope, swapped, full, closed });
  }

  /** Apply a scope by navigating: the URL is the source of truth, so the reactive load above picks it
   * up and the narrowed report is a link. */
  function applyScope(s: DiffScopeParams) {
    navigate({ kind: 'diff', against, scope: s, swapped, full, closed });
  }

  /** Read the same pair the other way round. Through the URL, so it lands in the history and a
   * swapped report can be sent to someone. */
  function swap() {
    navigate({ kind: 'diff', against, scope, swapped: !swapped, full, closed });
  }

  /** The same comparison, in the view that can be navigated — with everything that makes it *this*
   * comparison: both sides, the direction, the scope and the family fold. */
  function browse() {
    navigate({ kind: 'compare', against, right: '', scope, full, swapped });
  }

  /** Collapse families, or show every tensor. Through the URL, like the fold state. */
  function setFull(everything: boolean) {
    navigate({ kind: 'diff', against, scope, swapped, full: everything, closed });
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
   * The name of this checkpoint, for the side that is not the baseline.
   *
   * The *spec* — what was opened — not `root`, which for a single-file checkpoint is the directory
   * holding it: the header read `new  tests/fixtures` against `old  tests/fixtures/diff_new.safetensors`,
   * naming a directory as one side of a comparison of two files. The server's own answer, so the two
   * lines are the two things being compared.
   */
  $: thisOne = $tree?.spec ?? ($$props.root as string | undefined) ?? 'this checkpoint';
  // Which spec sits on which side, from the *server's* answer rather than from the prop: while a
  // swapped report is loading, the old one is still on screen, and labels that flipped early would
  // caption it wrongly.
  $: oldSide = result?.swapped ? thisOne : (result?.against ?? '');
  $: newSide = result?.swapped ? (result?.against ?? '') : thisOne;
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
   * an identifier *in it* changes, and `showAll` read from inside the body was not one — so clicking
   * "show the remaining" updated the state and redrew nothing.
   *
   * `items` is the already-filtered list, so both the cap and the withheld count are relative to what
   * the filter matched rather than to the whole section.
   */
  function capped<T>(items: T[], key: string, all: Record<string, boolean>): T[] {
    return all[key] ? items : items.slice(0, SECTION_CAP);
  }

  /** How many rows this section is holding back, out of those the filter matched. */
  function withheld(items: unknown[], key: string, all: Record<string, boolean>): number {
    return all[key] ? 0 : Math.max(0, items.length - SECTION_CAP);
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

  /** What this box accepts — the same sentence the side-by-side view shows, including the `:PATH`
   * shorthand and which host it resolves to. */
  $: help = specHelp($proxied, $proxyHost ?? '');
</script>

<div class="diff">
  <form
    class="pick"
    on:submit|preventDefault={submit}
  >
    <label for="diff-against">Compare against</label>
    <TextField
      id="diff-against"
      bind:value={draft}
      placeholder="baseline — {help}"
      title={help}
      spellcheck="false"
    />
    <button type="submit" disabled={!draft.trim() || draft.trim() === against}>Compare</button>
  </form>

  {#if against}
    <ScopeBar
      scope={scope ?? emptyScope()}
      onApply={applyScope}
      matched={result?.matched ?? null}
      busy={loading}
    />
  {/if}

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
        {full ? 'every tensor' : `${familyRows.toLocaleString()} rows for ${flatRows.toLocaleString()} tensors`}
      </span>
    </div>
  {/if}

  {#if loading}
    <!-- Not `reading {against}`: that puts an unbounded filesystem path into a label sized
         for a phrase, so a 48-char path wrapped to three lines and stranded the timer alone
         on the first. Nothing is lost — the path is on screen twice already, in the
         breadcrumb and the box directly above — and "baseline" is what the TUI's legend
         calls this side of a diff. -->
    <LoadingBar label="reading the baseline" progress={waitStarted} />
  {:else if error}
    <p class="error" role="alert">{error}</p>
  {:else if result && report}
    <header>
      <!-- Which checkpoint is on which side comes from the server's `swapped`, not from this
           component's copy of the parameter: the report below was built one way round, and the labels
           have to name that one. -->
      <!-- The addresses in full: they run the width of the pane, and what a cut removes is the end —
           `…/Kimi-K2.6-3bit` — which is the part that says which checkpoint it is. `.path` wraps. -->
      <!-- `:/path` rather than `host:/path` when the host is this server's own proxy — the same fifty
           characters on every line otherwise. The full form is the tooltip. -->
      <!-- The pair as one block, with the control in a column of its own.
           The button used to sit *after* the newer side's path, inside a `word-break: break-all` span —
           so where it landed depended on how long that path was. With a remote address it ended up
           mid-line in the middle of a wall of monospace, and was reported as missing. A grid column
           puts it in the same place every time, whatever the two addresses are. -->
      <div class="pair">
        <span class="side old">old</span><span class="path" title={oldSide}
          >{shortSpec(oldSide, $proxyHost ?? '')}</span>
        <span class="side new">new</span><span class="path" title={newSide}
          >{shortSpec(newSide, $proxyHost ?? '')}</span>
        <span class="swapslot">
          <SwapButton
            onSwap={swap}
            title="Swap the two sides — the same pair, compared the other way (the terminal's `s`)"
          />
        </span>
      </div>
      {#if identical}
        <div class="identical" role="status">
          <strong>{identical.headline}</strong>
          <span class="what">{identical.detail}</span>
        </div>
      {:else}
        <p class="verdict">{result.verdict}</p>
      {/if}
      <!-- This page is a summary; the side-by-side view is where a comparison is navigable — folding,
           lockstep, `n`/`N` stepping. Linking to it beats reproducing all of that here.
           **Carrying the whole comparison**, not just the baseline: the scope, the direction and the
           family fold are what make this *these two*, and arriving at the other view with a different
           selection reads as the link having changed the comparison. -->
      <p class="alt">
        <button type="button" class="link" on:click={browse}>
          Browse these two side by side →
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
      {#if result.s3_note}<p class="delta dim">{result.s3_note}</p>{/if}
    </header>

    <!-- A strip of counts, each a link to its section: the shape of the comparison in one line, and a
         way into the part you care about. On a 31,247-row report the alternative is scrolling to find
         out whether anything was removed. -->
    <nav class="tally" aria-label="Sections">
      {#each SECTIONS as sec (sec.key)}
        {@const n = sectionCount(sec.key, report, { added, removed, changed, metaShown })}
        <DiffChip
          tone={sec.tone}
          label={sec.title}
          order="label-first"
          count={n.toLocaleString()}
          empty={n === 0}
          title={n === 0 ? `${sec.title}: none` : `Show ${sec.title.toLowerCase()}`}
          onPick={() => reveal(sec.key)}
        />
      {/each}
    </nav>

    <Section
      title="Tensors added"
      count={countLabel(added.length, report.tensors_added.length)}
      tone="added"
      open={openSections.tensors_added}
      onToggle={fold('tensors_added')}
    >
      {#if !added.length}<p class="none dim">none</p>{/if}
      {#if full}
        {#each capped(added, 'tensors_added', showAll) as [name, s] (name)}
          <DiffRow mark="+" {name} neu={s} onOpen={open} />
        {/each}
      {:else}
        <!-- A family row stands for many tensors, so it opens none of them: `Collapse families` off is
             the way to a single tensor's detail. -->
        {#each capped(groupedAdded, 'tensors_added', showAll) as g (g.name)}
          <DiffRow mark="+" name={g.name} neu={g.sig} count={g.count} />
        {/each}
      {/if}
      {#if withheld(added, 'tensors_added', showAll)}
        <button type="button" class="more" on:click={() => (showAll = { ...showAll, tensors_added: true })}>
          show the remaining {withheld(added, 'tensors_added', showAll).toLocaleString()}
        </button>
      {/if}
    </Section>

    <Section
      title="Tensors removed"
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
          />
        {/each}
      {:else}
        {#each capped(groupedRemoved, 'tensors_removed', showAll) as g (g.name)}
          <DiffRow
            mark="-"
            name={g.name}
            old={g.sig}
            count={g.count}
            why="only in the baseline, so there is nothing here to open"
          />
        {/each}
      {/if}
      {#if withheld(removed, 'tensors_removed', showAll)}
        <button type="button" class="more" on:click={() => (showAll = { ...showAll, tensors_removed: true })}>
          show the remaining {withheld(removed, 'tensors_removed', showAll).toLocaleString()}
        </button>
      {/if}
    </Section>

    <Section
      title="Tensors changed"
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
        {#each capped(groupedChanged, 'tensors_changed', showAll) as g (g.name)}
          <DiffRow
            mark="~"
            name={g.name}
            old={g.old}
            neu={g.new}
            count={g.count}
            fold={g.fold ? `×${g.fold[0].toLocaleString()} → ×${g.fold[1].toLocaleString()}` : ''}
          />
        {/each}
        {#if withheld(groupedChanged, 'tensors_changed', showAll)}
          <button type="button" class="more" on:click={() => (showAll = { ...showAll, tensors_changed: true })}>
            show the remaining {withheld(groupedChanged, 'tensors_changed', showAll).toLocaleString()}
          </button>
        {/if}
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
            />
          {/each}
        {/if}
      {/each}
      {#if withheld(changed, 'tensors_changed', showAll)}
        <button type="button" class="more" on:click={() => (showAll = { ...showAll, tensors_changed: true })}>
          show the remaining {withheld(changed, 'tensors_changed', showAll).toLocaleString()}
        </button>
      {/if}
    </Section>

    <!-- Its own heading rather than loose beneath the changed list, where it read as a stray footnote to
         that section instead of a count of its own. Nothing to fold: the count *is* the section. -->
    <p class="unchanged dim">
      Tensors unchanged <b>{report.tensors_unchanged.toLocaleString()}</b>
    </p>

    <Section
      title="Metadata"
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
      <p class="none dim">{result.s3_note}</p>
    {/if}

    <footer>
      <p class="dim">
        Structure only. To compare the numbers, run this in a terminal with
        <code>--values</code>:
      </p>
      <div class="cmd">
        <code>{result.command}</code>
        <button on:click={copyCommand}>{copied ? '✓ copied' : 'copy'}</button>
      </div>
    </footer>
  {/if}
</div>

<style>
  .diff {
    padding: 10px 14px;
    overflow: auto;
  }
  /* The boxes are `TextField`, which owns their look — see that component. */
  .pick {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
  }
  .pick label {
    font-size: 12px;
    color: var(--fg-dim);
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
  /* Label, address, control — three columns, so the button's place does not depend on the length of a
     path. The addresses still wrap (`.path`), and the button stays beside them either way. */
  /* The control's place in the header grid; what it *is* lives in `SwapButton`. */
  .swapslot {
    grid-column: 3;
    grid-row: 1 / span 2;
    align-self: center;
    justify-self: start;
    margin-left: 12px;
  }
  .pair {
    display: grid;
    /* The address column takes what it needs and no more (`max-content`, capped by the room there is),
       so the control sits beside the pair rather than out at the window's edge. */
    grid-template-columns: 2.2em minmax(0, max-content) auto;
    align-items: baseline;
    column-gap: 8px;
    row-gap: 2px;
    font-size: 12.5px;
  }
  .side {
    font-weight: 600;
  }
  .path {
    word-break: break-all;
  }
  .verdict {
    margin: 6px 0 0;
    font-weight: 600;
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
  .alt {
    margin: 4px 0 0;
  }
  .alt .link {
    padding-left: 0;
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
  /* The unchanged count is a fact, not a section: no caret, nothing to open. */
  .unchanged {
    margin: 0 0 12px;
    font-size: 12.5px;
  }
  .unchanged b {
    color: var(--fg);
    font-variant-numeric: tabular-nums;
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
