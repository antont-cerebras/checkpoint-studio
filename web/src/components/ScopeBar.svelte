<script lang="ts">
  // How a diff is narrowed and how two naming schemes are lined up — the CLI's selection flags.
  //
  // Shared by both diff views, because a scope that meant different things on the two screens would
  // recreate exactly the divergence this whole effort has been closing. The parameters and their
  // semantics are `lib/diffscope`; the server applies them with the CLI's own filter builders
  // (`src/web/diffscope.rs`), so what you narrow here is what `checkpoint-studio diff` narrows.
  //
  // **Two sections, not one row of boxes.** These controls answer two unrelated questions — *which
  // tensors are compared* and *how the two sides' names are matched up* — and side by side in one strip
  // they read as seven interchangeable boxes. The rename rules and the fused alignment in particular are
  // not filters: they change what "the same tensor" means. Each field says what it does and gives
  // examples, in the CLI's own words, because "dtype `--dtype-is`" is a label only for someone who has
  // read the help.
  //
  // Collapsed by default. It is eight controls that most comparisons do not need, and the summary line
  // means an *active* scope is never hidden behind a fold.
  import { emptyScope, isScopeActive, scopeSummary, type DiffScopeParams } from '../lib/diffscope';
  import TextField from './TextField.svelte';
  import Section from './Section.svelte';

  /** The scope in force. Changing a box does not apply it — [`apply`] does, so a half-typed glob
   * never triggers a comparison. */
  export let scope: DiffScopeParams = emptyScope();
  /** Called with the edited scope when the reader asks for it. */
  export let onApply: (s: DiffScopeParams) => void;
  /** `matched M of N`, when the server reported one. */
  export let matched: { selected: number; total: number } | null = null;
  /** Inert while a comparison is being read, like the path boxes above it. */
  export let busy = false;

  /** A text field of the scope: which key it edits, what it is, and what one looks like. */
  interface Field {
    key: 'name' | 'names' | 'dtypeIs' | 'shapeIs' | 'map' | 'subtree' | 'subtreeNew';
    label: string;
    /** The CLI flag it is, so the two surfaces are one thing to learn. */
    flag: string;
    /** `0` for a single-line box. */
    rows: number;
    /** Narrow, for the boxes that hold a dtype or a shape rather than a list. */
    narrow?: boolean;
    placeholder: string;
    /** What it selects — one sentence, in the flag's own terms. */
    hint: string;
    /** Concrete ones. The point of an example here is that these grammars are guessable *only* from
     * examples: nobody derives `**,2048` from a prose description of `--shape-is`. */
    examples: string[];
  }

  /** Which tensors are compared at all. */
  const FILTERS: Field[] = [
    {
      key: 'name',
      label: 'Name globs',
      flag: '--name',
      rows: 2,
      placeholder: 'model.layers.1.*\n!*.bias',
      hint: 'One glob per line; a tensor is kept if it matches any of them. A line starting with ! excludes instead. {a,b} tries each alternative.',
      examples: ['model.layers.1.*', '*.mlp.down_proj.weight', 'model.layers.{0,31,60}.*', '!*.bias'],
    },
    {
      key: 'names',
      label: 'Exact names',
      flag: '--names',
      rows: 2,
      placeholder: 'lm_head.weight, model.norm.weight',
      hint: 'Comma-separated names, matched literally — for a list produced somewhere else (the CLI also reads one from a file: --names-from). Narrows within the globs above: a tensor has to pass both.',
      examples: ['lm_head.weight, model.norm.weight'],
    },
    {
      key: 'dtypeIs',
      label: 'dtype',
      flag: '--dtype-is',
      rows: 0,
      narrow: true,
      placeholder: 'F*',
      hint: "A glob against the tensor's stored dtype, case-insensitive. Kept if either side matches, so a re-quantized tensor still counts.",
      examples: ['BF16', 'F*', 'U8', 'I*'],
    },
    {
      key: 'shapeIs',
      label: 'shape',
      flag: '--shape-is',
      rows: 0,
      narrow: true,
      placeholder: '768,**',
      hint: 'Dimensions, comma- or x-separated: * is one dimension, ** any number of them. Kept if either side matches.',
      examples: ['768,2048', '768,*', '*,2048', '**,2048'],
    },
  ];

  /** How the two sides' names are matched up. Not filters: these change what "the same tensor" is. */
  const ALIGNMENT: Field[] = [
    {
      key: 'map',
      label: 'Rename rules',
      flag: '--map',
      rows: 2,
      placeholder: '^blocks\\.=>model.layers.',
      hint: "PATTERN=>REPLACEMENT per line, applied to the baseline's names before comparing so two naming schemes line up instead of reading as added+removed. Regex, with $1 captures; rules apply in order.",
      examples: ['^blocks\\.=>model.layers.', '\\.mlp\\.experts\\.=>.block_sparse_moe.experts.'],
    },
    {
      key: 'subtree',
      label: 'Baseline subtree',
      flag: 'OLD#subtree',
      rows: 0,
      narrow: true,
      placeholder: 'language_model',
      hint: "Compare from inside this subtree of the baseline: its tensors are keyed by their sub-path, so a multimodal checkpoint's language_model.model.… lines up with a converted model.…, and the siblings (vision_tower.…) are out of scope rather than removed.",
      examples: ['language_model', 'model.layers'],
    },
    {
      key: 'subtreeNew',
      label: 'Newer subtree',
      flag: 'NEW#subtree',
      rows: 0,
      narrow: true,
      placeholder: 'model',
      hint: 'The same, on the newer side. Either side, or both — whichever one hangs under an extra namespace.',
      examples: ['language_model', 'model'],
    },
  ];

  let open = false;
  /** The two sections' fold state. Local: it is how this panel is arranged, not what it selects — the
   * selection is the thing that belongs in the URL, and it is all in `scope`. */
  let openFilters = true;
  let openAlignment = true;

  // Named, not inline: a template expression cannot carry a type annotation (the Svelte parser rejects
  // it), and without one the callback's argument is `any`.
  function foldFilters(o: boolean) {
    openFilters = o;
  }
  function foldAlignment(o: boolean) {
    openAlignment = o;
  }

  // A working copy, so a partly-typed glob is not applied on every keystroke.
  //
  // Re-seeded from the applied scope's *content*, via a string. `$: draft = { ...scope }` looks right
  // and is not: Svelte 4's `safe_not_equal` counts any object prop as changed on every update, so that
  // re-ran on each keystroke and put the old text straight back — the boxes were unfillable. Depending
  // on a `string` instead means value comparison, so this runs only when the scope really differs.
  //
  // Third time this trap has been sprung in this directory: once with `$: draft = prop || draft`, once
  // with an `on:input` flag that lost a race, now with object identity.
  let draft: DiffScopeParams = { ...scope };
  $: applied = JSON.stringify(scope);
  $: draft = reseed(applied);

  /** The applied scope, parsed back — using `applied` rather than closing over `scope` is what keeps
   * this a function of its argument, and so re-run only when that argument changes. */
  function reseed(json: string): DiffScopeParams {
    return JSON.parse(json) as DiffScopeParams;
  }

  $: active = isScopeActive(scope);
  $: summary = scopeSummary(scope);
  // Open it automatically when a link arrives already scoped: the boxes are the explanation for a
  // comparison that shows nineteen rows out of 117,664.
  $: if (active && !open && summary !== '') open = true;

  function apply() {
    onApply({ ...draft });
    open = false;
  }

  function clear() {
    draft = emptyScope();
    onApply(emptyScope());
  }
</script>

<div class="scope">
  <div class="head">
    <button type="button" class="toggle" aria-expanded={open} on:click={() => (open = !open)}>
      <span class="caret">{open ? '▾' : '▸'}</span> Limit what is compared
    </button>
    {#if summary}
      <span class="what" title={summary}>{summary}</span>
    {/if}
    {#if matched}
      <!-- The CLI's own context line: the selected count is only meaningful against the total. -->
      <span class="matched dim">
        matched {matched.selected.toLocaleString()} of {matched.total.toLocaleString()}
      </span>
    {/if}
    {#if active}
      <button type="button" class="quiet" disabled={busy} on:click={clear}>clear scope</button>
    {/if}
  </div>

  {#if open}
    <div class="body">
      <Section
        title="Filtering"
        note="which tensors are compared"
        open={openFilters}
        onToggle={foldFilters}
      >
        <div class="fields">
          {#each FILTERS as f (f.key)}
            <label for="scope-{f.key}" class:narrow={f.narrow}>
              <span class="what"
                >{f.label} <code>{f.flag}</code></span
              >
              <TextField
                variant="dense"
                rows={f.rows}
                id="scope-{f.key}"
                bind:value={draft[f.key]}
                placeholder={f.placeholder}
                spellcheck="false"
                readonly={busy}
              />
              <small class="dim">{f.hint}</small>
              <small class="eg">
                <span class="dim">e.g.</span>
                {#each f.examples as e (e)}<code>{e}</code>{/each}
              </small>
            </label>
          {/each}
        </div>
        <label class="check">
          <input type="checkbox" bind:checked={draft.onlyTensors} disabled={busy} />
          Tensors only <code>--only-tensors</code>
          <small class="dim"
            >— skip the metadata entries. Any filter above already skips them: no glob can select
            one.</small
          >
        </label>
      </Section>

      <Section
        title="Alignment"
        note="how the two sides' names are matched up"
        open={openAlignment}
        onToggle={foldAlignment}
      >
        <div class="fields">
          {#each ALIGNMENT as f (f.key)}
            <label for="scope-{f.key}" class:narrow={f.narrow}>
              <span class="what"
                >{f.label} <code>{f.flag}</code></span
              >
              <TextField
                variant="dense"
                rows={f.rows}
                id="scope-{f.key}"
                bind:value={draft[f.key]}
                placeholder={f.placeholder}
                spellcheck="false"
                readonly={busy}
              />
              <small class="dim">{f.hint}</small>
              <small class="eg">
                <span class="dim">e.g.</span>
                {#each f.examples as e (e)}<code>{e}</code>{/each}
              </small>
            </label>
          {/each}
        </div>
        <!-- The one-click answer for "these are the same model in two layouts". Without it the two sides
             share no tensor name, so the comparison reports every tensor of both as one-sided — 80,107
             against 933 — which is true and useless. -->
        <label class="check">
          <input type="checkbox" bind:checked={draft.alignFused} disabled={busy} />
          Align unfused ↔ fused <code>--align-fused</code>
          <small class="dim"
            >— drops the per-expert index, so the 256 tensors of one expert group fold onto the single
            fused tensor that holds them (shown as <code>×256</code>), and applies the standard layout
            synonyms: <code>w1</code>/<code>w3</code> ↔ <code>gate_up_proj</code>, <code>w2</code> ↔
            <code>down_proj</code>, q/k/v ↔ <code>qkv_proj</code>, <code>.weight.qscale</code> ↔
            <code>.qscale</code>, <code>e_score_correction_bias</code> ↔ <code>gate.bias</code>.
            Applied to both sides; each rule is a no-op on a side that is already fused.</small
          >
        </label>
      </Section>

      <div class="acts">
        <button type="button" disabled={busy} on:click={apply}>Apply scope</button>
      </div>
    </div>
  {/if}
</div>

<style>
  /* Background fill, no border — the treatment the palette, the identical banner and the busy panel
     all use. */
  .scope {
    flex: 0 0 auto;
    padding: 6px 9px;
    border-radius: 4px;
    background: var(--bg-elev);
    font-size: 12px;
  }
  .head {
    display: flex;
    align-items: baseline;
    gap: 9px;
    flex-wrap: wrap;
  }
  .toggle {
    font: inherit;
    font-size: 12px;
    color: var(--fg);
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
  }
  .caret {
    color: var(--fg-dim);
  }
  /* The reason the row count is what it is — never hidden behind the fold. */
  .head .what {
    color: var(--accent);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 60ch;
  }
  .matched {
    font-variant-numeric: tabular-nums;
  }
  .quiet {
    margin-left: auto;
    color: var(--fg-dim);
    background: none;
    border: 1px solid var(--border);
    border-radius: 4px;
    font: inherit;
    font-size: 12px;
    padding: 2px 8px;
    cursor: pointer;
  }
  .body {
    margin-top: 8px;
  }
  .fields {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    flex-wrap: wrap;
  }
  /* A field is a column: what it is, the box, what it means, an example. Wide enough that the hint
     reads as a sentence rather than as one word per line. */
  .fields label {
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1 1 30ch;
    min-width: 0;
    max-width: 46ch;
  }
  .fields label.narrow {
    flex: 1 1 20ch;
    max-width: 26ch;
  }
  .fields label .what {
    color: var(--fg-dim);
  }
  /* The switches: one per line, so the sentence after them has room. */
  .check {
    display: flex;
    align-items: baseline;
    gap: 6px;
    flex-wrap: wrap;
    margin-top: 8px;
  }
  .check small {
    flex: 1 1 30ch;
  }
  /* The boxes are `TextField` (variant `dense`), which owns their look. */
  small {
    font-size: 11px;
    line-height: 1.35;
  }
  /* The examples as separate chips, wrapping — run together on one line they read as one long glob,
     which is the opposite of an example. */
  .eg {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 3px 8px;
    color: var(--fg-dim);
  }
  .eg code {
    color: var(--accent);
  }
  code {
    font-size: 11px;
  }
  .acts {
    display: flex;
    justify-content: flex-end;
    margin-top: 4px;
  }
</style>
