<script lang="ts">
  // How a diff is narrowed and how two naming schemes are lined up — the CLI's selection flags.
  //
  // Shared by both diff views, because a scope that meant different things on the two screens would
  // recreate exactly the divergence this whole effort has been closing. The parameters and their
  // semantics are `lib/diffscope`; the server applies them with the CLI's own filter builders
  // (`src/web/diffscope.rs`), so what you narrow here is what `checkpoint-studio diff` narrows.
  //
  // **Three jobs, not basic versus advanced.** These controls answer three questions: which tensors
  // enter the comparison, whether either model sits below an extra namespace, and how equivalent names
  // line up. Exact-name selection is not more "advanced" than a glob, and a subtree is not the same kind
  // of knob as a rename rule. The layout follows those jobs instead of ranking unrelated CLI flags by
  // presumed difficulty.
  //
  // Collapsed by default. It is several controls that most comparisons do not need, and the summary line
  // means an *active* scope is never hidden behind a fold.
  import { emptyScope, isScopeActive, scopeSummary, type DiffScopeParams } from '../lib/diffscope';
  import { parseMappingRules, serializeMappingRules, type MappingRule } from '../lib/mapbuilder';
  import TextField from './TextField.svelte';
  import { api } from '../lib/api';
  import { comparison } from '../stores/compare';
  import PickList from './PickList.svelte';

  /** The scope in force. Changing a box does not apply it — [`apply`] does, so a half-typed glob
   * never triggers a comparison. */
  export let scope: DiffScopeParams = emptyScope();
  /** Called with the edited scope when the reader asks for it. */
  export let onApply: (s: DiffScopeParams) => void;
  /** `matched M of N`, when the server reported one. */
  export let matched: { selected: number; total: number } | null = null;
  /** Inert while a comparison is being read, like the path boxes above it. */
  export let busy = false;
  /**
   * Bumped by a caller that wants the name-matching controls in view — the "nothing lines up" banner's
   * *Choose matching subtrees…*. A counter, not a boolean: the panel may already be open, and a
   * boolean that is already `true` cannot ask for anything.
   */
  export let openMatching = 0;
  // A field rather than a `let`: it holds the value of the *previous* run, read on the next one,
  // which no static analysis can see (`no-useless-assignment` calls the assignment dead).
  const responded = { to: -1 };
  let rootsCard: HTMLElement | null = null;
  $: if (openMatching !== responded.to) {
    responded.to = openMatching;
    if (openMatching > 0) {
      open = true;
      // The caller asked specifically for the subtree controls. Wait for the folded panel to exist,
      // then put that card in view; merely opening a three-card form still leaves the reader hunting.
      requestAnimationFrame(() => rootsCard?.scrollIntoView({ block: 'nearest' }));
    }
  }

  /** A text field of the scope: which key it edits, what it is, and what one looks like. */
  interface Field {
    key: 'name' | 'names' | 'dtypeIs' | 'shapeIs' | 'subtree' | 'subtreeNew' | 'repackSchema' | 'repackSchemaNew';
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

  /** Everything that changes which tensors enter the comparison. */
  const SELECTION: Field[] = [
    {
      key: 'name',
      label: 'Name globs',
      flag: '--name',
      rows: 2,
      placeholder: 'model.layers.1.*\n!*.bias',
      hint: 'One glob per line — not comma-separated, because the comma belongs to {a,b}. A tensor is kept if it matches any line; a line starting with ! excludes instead.',
      examples: ['model.layers.1.*', '*.mlp.down_proj.weight', 'model.layers.{0,31,60}.*', '!*.bias'],
    },
    {
      key: 'names',
      label: 'Exact names',
      flag: '--names',
      rows: 2,
      placeholder: 'lm_head.weight\nmodel.norm.weight',
      hint: 'One name per line, or comma-separated — whichever you have. When name globs are also set, a tensor must pass both.',
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

  /** Remove a wrapper namespace before attempting to match tensor names. */
  const ROOTS: Field[] = [
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
      label: 'Candidate subtree',
      flag: 'NEW#subtree',
      rows: 0,
      narrow: true,
      placeholder: 'model',
      hint: 'The same, on the candidate. Either side, or both — whichever one hangs under an extra namespace.',
      examples: ['language_model', 'model'],
    },
  ];

  /**
   * How each side's packed tensors are **decoded**, before anything is compared.
   *
   * Here rather than beside the *Verify repack* button because that is what it is: a mapping applied to
   * certain tensors ahead of a comparison, like the rename rules above and the fused alignment — not an
   * option of one run. A packed expert weight stores indices into a codebook, several to a 16-bit word,
   * and **the checkpoint says nowhere how**: `[4]` is the sparse encoding (one index per word, four low
   * bits used) and `[3,3,3,3,3]` is five consecutive experts per word, each shifted three bits. A width
   * guessed from the fold ratio describes a uniform merge and says nothing about a sparse side, so a
   * sparse-vs-merged pair read as *not equivalent* with no way to say otherwise.
   */
  const PACKING: Field[] = [
    {
      key: 'repackSchema',
      label: 'Baseline packing',
      flag: '--repack-schema',
      rows: 0,
      narrow: true,
      placeholder: '[4]',
      hint: "Bit widths per 16-bit word, least-significant field first. Leave blank to infer it from the two shapes. Used when verifying a repack.",
      examples: ['[4]', '3,3,3,3,3', 'u3×5'],
    },
    {
      key: 'repackSchemaNew',
      label: 'Candidate packing',
      flag: '--repack-schema-new',
      rows: 0,
      narrow: true,
      placeholder: '[3,3,3,3,3]',
      hint: 'The same, for the candidate. The two sides are separate fields because a repack is exactly a change of packing — one side sparse, the other merged.',
      examples: ['[3,3,3,3,3]', '[4,4,4,4]', '4'],
    },
  ];

  let open = false;
  // A working copy, so a partly-typed glob is not applied on every keystroke.
  //
  // Re-seeded from the applied scope's serialized *content*, and guarded against the previous serialized
  // value below. Object identity is not a usable boundary in Svelte 4: a parent can hand down an equal
  // fresh object on any update, which must not erase text currently being edited.
  /** Parse a serialized scope into an independent working copy. */
  function reseed(json: string): DiffScopeParams {
    return JSON.parse(json) as DiffScopeParams;
  }

  interface EditableMapping extends MappingRule {
    id: number;
  }

  let mappingId = 0;
  function editableMappings(rules: MappingRule[]): EditableMapping[] {
    return rules.map((rule) => ({ ...rule, id: ++mappingId }));
  }

  /**
   * Seed the form only when the *applied scope's serialized value* actually changes.
   *
   * Assigning `draft = reseed(applied)` as an unconditional reactive statement looked equivalent, but
   * Svelte invalidated the object while a nested field was being bound. The statement then replaced the
   * draft during the same update: checkboxes unticked themselves and typed characters disappeared. This
   * explicit previous-value guard is the boundary between parent-owned applied state and editable state.
   */
  const initialApplied = JSON.stringify(scope);
  const seeded = { applied: initialApplied };
  let applied = initialApplied;
  let draft: DiffScopeParams = reseed(initialApplied);
  const initialMappings = parseMappingRules(draft.map);
  let mappings = editableMappings(initialMappings.rules);
  let mappingMode: 'builder' | 'raw' = initialMappings.rawOnly ? 'raw' : 'builder';
  let mappingIssue = '';
  $: incomingScope = JSON.stringify(scope);
  $: if (incomingScope !== seeded.applied) {
    seeded.applied = incomingScope;
    applied = incomingScope;
    draft = reseed(incomingScope);
    const parsed = parseMappingRules(draft.map);
    mappings = editableMappings(parsed.rules);
    mappingMode = parsed.rawOnly ? 'raw' : 'builder';
    mappingIssue = '';
  }

  type TextScopeKey = Field['key'];

  function eventValue(e: Event): string {
    return (e.currentTarget as HTMLInputElement | HTMLTextAreaElement).value;
  }

  function eventChecked(e: Event): boolean {
    return (e.currentTarget as HTMLInputElement).checked;
  }

  function setText(key: TextScopeKey, value: string) {
    draft = { ...draft, [key]: value };
  }

  function setFlag(key: 'onlyTensors' | 'alignFused', value: boolean) {
    draft = { ...draft, [key]: value };
  }

  function setRawMapping(value: string) {
    draft = { ...draft, map: value };
    mappingIssue = '';
  }

  /** Publish builder rows into the ordinary `map` text field — the API remains unchanged. */
  function commitMappings(next: EditableMapping[]) {
    mappings = next;
    draft = { ...draft, map: serializeMappingRules(next) };
    mappingIssue = '';
  }

  function addMapping() {
    commitMappings([...mappings, { id: ++mappingId, pattern: '', replacement: '' }]);
  }

  function updateMapping(id: number, key: keyof MappingRule, value: string) {
    commitMappings(mappings.map((rule) => (rule.id === id ? { ...rule, [key]: value } : rule)));
  }

  function removeMapping(id: number) {
    commitMappings(mappings.filter((rule) => rule.id !== id));
  }

  function moveMapping(index: number, by: -1 | 1) {
    const to = index + by;
    if (to < 0 || to >= mappings.length) return;
    const next = [...mappings];
    const current = next[index];
    const target = next[to];
    if (!current || !target) return;
    next[index] = target;
    next[to] = current;
    commitMappings(next);
  }

  function editRawMappings() {
    commitMappings(mappings);
    mappingMode = 'raw';
  }

  function useMappingBuilder() {
    const parsed = parseMappingRules(draft.map);
    if (parsed.rawOnly) {
      mappingIssue = 'Remove comments or complete every PATTERN=>REPLACEMENT line before switching to the builder.';
      return;
    }
    mappings = editableMappings(parsed.rules);
    mappingMode = 'builder';
    mappingIssue = '';
  }

  $: active = isScopeActive(scope);
  $: summary = scopeSummary(scope);
  // Open it automatically when a link *arrives* already scoped: the boxes are the explanation for a
  // comparison that shows nineteen rows out of 117,664.
  //
  // Once, on the first scoped render — not on every change. As a standing rule it re-opened the panel
  // the moment `apply` closed it (applying makes the scope active, which is the condition), so a
  // panel this tall sat permanently over the result it had just narrowed.
  const greeted = { done: false };
  $: if (!greeted.done && active && summary !== '') {
    greeted.done = true;
    open = true;
  }
  function apply() {
    onApply({ ...draft });
    open = false;
  }

  /**
   * Close without changing anything — and put the boxes back to what is actually in force.
   *
   * *Close*, not *Cancel*: nothing has been started that could be cancelled, and the commonest reason
   * to open this panel is to read what is set. Discarding the edits is what makes closing safe rather
   * than what it is for — leaving a half-typed glob in a closed panel is a trap, since next time it
   * opens it looks like the selection and *Apply settings* would narrow the comparison to something
   * nobody asked for.
   */
  function closePanel() {
    draft = reseed(applied);
    const parsed = parseMappingRules(draft.map);
    mappings = editableMappings(parsed.rules);
    mappingMode = parsed.rawOnly ? 'raw' : 'builder';
    mappingIssue = '';
    open = false;
  }

  /**
   * Escape closes the panel, wherever the caret is — the same as *Close*.
   *
   * On the window and in the *capture* phase, because `TextField` stops keydowns from reaching any
   * ancestor (that is what keeps `s` from swapping a comparison while you type into a box), and
   * because Escape otherwise means "go back a screen": a reader dismissing a panel does not expect
   * to leave the comparison.
   */
  function onWindowKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && open) {
      e.preventDefault();
      e.stopPropagation();
      closePanel();
    }
  }

  /**
   * Picking values from the lists the two checkpoints actually contain.
   *
   * Typing into these fields is a guess. A tensor name is one of tens of thousands, and a subtree prefix
   * that selects nothing is a refusal rather than a comparison — so each field offers what is there:
   * the names both sides share once the alignment is applied (`/api/diffnames`), and each side's
   * namespaces with the number of tensors under them (`/api/subtrees`).
   *
   * The names list carries the **draft** alignment, not the applied one: a reader writes a rename rule
   * and picks from what it produces, without applying anything first. The subtree lists carry no
   * alignment at all — re-rooting is applied *to* them.
   */
  let picking: '' | 'names' | 'subtree' | 'subtreeNew' = '';
  /**
   * A handle for the *names* list alone: it is the one that has to re-ask when something else on the
   * panel changes (the alignment). The subtree lists depend on nothing but their side.
   *
   * Typed structurally rather than as the component: a Svelte 4 component's exported functions are not
   * part of the type its class carries, so `PickList` here would be an `any` call at the one site that
   * matters.
   */
  let namesList: { refresh: (delay?: number) => void } | undefined;

  /** The names already in the field, in the CLI's own separated form. */
  $: chosenNames = draft.names
    .split(/[,\n]/)
    .map((n) => n.trim())
    .filter(Boolean);

  /** No pair, no lists: the options are facts about two checkpoints the server has read. */
  $: pairId = $comparison?.id;

  /** The names both sides share under the alignment being edited. */
  async function loadNames(query: string) {
    if (pairId === undefined) throw new Error('Compare a pair first — the list is what its two sides share.');
    const r = await api.diffNames(pairId, { ...draft }, query, 200);
    return { options: r.names.map((value) => ({ value })), total: r.matched };
  }

  /** One side's namespaces, with what is under each — the count is what tells a wrapper from a typo. */
  function loadSubtrees(side: 'old' | 'new') {
    return async (query: string) => {
      if (pairId === undefined) throw new Error('Compare a pair first — these are the namespaces it found.');
      const r = await api.subtrees(pairId, side, query, 200);
      return {
        options: r.subtrees.map((o) => ({
          value: o.prefix,
          detail: `${o.tensors.toLocaleString()} ${o.tensors === 1 ? 'tensor' : 'tensors'}`,
        })),
        total: r.total,
      };
    };
  }

  /** Open one list at a time: three open at once is a panel of scrolling boxes.
   *
   * No fetch here — each list asks on mount, since it is *created* by this flag and cannot be called
   * before it exists. */
  function togglePicker(which: 'names' | 'subtree' | 'subtreeNew') {
    picking = picking === which ? '' : which;
  }

  /** A name adds to (or comes out of) the list; a subtree is one value, so it replaces. */
  function pickName(name: string) {
    const next = chosenNames.includes(name)
      ? chosenNames.filter((n) => n !== name)
      : [...chosenNames, name];
    // The separator already in the box, so a pasted column stays a column and a typed list stays a list.
    setText('names', next.join(draft.names.includes('\n') ? '\n' : ', '));
  }

  function pickSubtree(key: 'subtree' | 'subtreeNew', prefix: string) {
    setText(key, draft[key] === prefix ? '' : prefix);
  }
  // Named, because a template expression cannot carry the annotation that would type its parameter —
  // and an untyped one is an `any` flowing into a typed prop.
  const pickBaselineSubtree = (prefix: string) => pickSubtree('subtree', prefix);
  const pickCandidateSubtree = (prefix: string) => pickSubtree('subtreeNew', prefix);
  /** The two loaders, made once rather than per render. */
  const loadBaselineSubtrees = loadSubtrees('old');
  const loadCandidateSubtrees = loadSubtrees('new');

  /** The alignment decides what lines up, so an edit to it re-asks — while the list is open. */
  $: alignmentKey = `${draft.map}\u0000${draft.subtree}\u0000${draft.subtreeNew}\u0000${draft.alignFused}`;
  $: if (picking === 'names' && alignmentKey) namesList?.refresh();

  function clear() {
    draft = emptyScope();
    mappings = [];
    mappingMode = 'builder';
    mappingIssue = '';
    onApply(emptyScope());
  }
</script>

<svelte:window on:keydown|capture={onWindowKeydown} />

<div class="scope">
  <div class="head">
    <button type="button" class="toggle" aria-expanded={open} on:click={() => (open = !open)}>
      <span class="caret">{open ? '▾' : '▸'}</span> Comparison settings
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
  </div>

  {#if open}
    <div class="body">
      <!-- **Alignment first, selection last.** The names you can select *are* the names the alignment
           produces: with two naming schemes and no rules, the two sides share nothing and there is
           nothing to pick. So the panel reads in the order the work happens — drop a wrapper
           namespace, line the names up, then choose among what lines up (and the picker in step 3
           lists exactly that). It used to open with the selection, which is the step that depends on
           the other two. -->
      <div class="cards">
        <section class="card roots" bind:this={rootsCard}>
          <header class="card-head">
            <span class="step">1 · Re-root</span>
            <div>
              <h3>Line up submodels</h3>
              <p>Remove an extra wrapper namespace from either checkpoint.</p>
            </div>
          </header>
          <div class="fields paired">
          {#each ROOTS as f (f.key)}
            <label for="scope-{f.key}" class:narrow={f.narrow}>
              <span class="what"
                >{f.label} <code>{f.flag}</code></span
              >
              <TextField
                variant="dense"
                rows={f.rows}
                id="scope-{f.key}"
                value={draft[f.key]}
                on:input={(e) => setText(f.key, eventValue(e))}
                placeholder={f.placeholder}
                spellcheck="false"
                readonly={busy}
              />
              <small class="dim">{f.hint}</small>
              <small class="eg">
                <span class="dim">e.g.</span>
                {#each f.examples as e (e)}<code>{e}</code>{/each}
              </small>
              <!-- Each side's own namespaces: the two checkpoints have different ones, which is the
                   whole reason there are two fields. A prefix that selects nothing is a refusal, so the
                   list carries the tensor count that tells `language_model` from a typo. -->
              <button
                type="button"
                class="picklink"
                disabled={busy}
                on:click={() => togglePicker(f.key === 'subtree' ? 'subtree' : 'subtreeNew')}
              >
                {picking === f.key ? 'Hide the list' : 'Pick a namespace…'}
              </button>
              {#if picking === 'subtree' && f.key === 'subtree'}
                <PickList
                  id="scope-subtree-search"
                  chosen={draft.subtree ? [draft.subtree] : []}
                  load={loadBaselineSubtrees}
                  onPick={pickBaselineSubtree}
                  disabled={busy}
                  placeholder="search the baseline's namespaces…"
                  noun="namespaces in the baseline"
                  emptyNote="No namespace to drop — the baseline's tensors sit at the root."
                />
              {:else if picking === 'subtreeNew' && f.key === 'subtreeNew'}
                <PickList
                  id="scope-subtree-new-search"
                  chosen={draft.subtreeNew ? [draft.subtreeNew] : []}
                  load={loadCandidateSubtrees}
                  onPick={pickCandidateSubtree}
                  disabled={busy}
                  placeholder="search the candidate's namespaces…"
                  noun="namespaces in the candidate"
                  emptyNote="No namespace to drop — the candidate's tensors sit at the root."
                />
              {/if}
            </label>
          {/each}
          </div>
          <p class="tip dim">Example: compare <code>language_model.model.…</code> with <code>model.…</code> by entering <code>language_model</code> on the wrapped side.</p>
        </section>

        <section class="card matching">
          <header class="card-head">
            <span class="step">2 · Match</span>
            <div>
              <h3>Match different names</h3>
              <p>Describe tensors that correspond but use different naming or packing layouts.</p>
            </div>
          </header>
          <label class="choice">
            <input
              type="checkbox"
              checked={draft.alignFused}
              disabled={busy}
              on:change={(e) => setFlag('alignFused', eventChecked(e))}
            />
            <span>
              <strong>Known fused ↔ unfused layouts</strong> <code>--align-fused</code>
              <small class="dim">Use built-in expert, projection, scale, and gate-name correspondences.</small>
            </span>
          </label>
          <div class="mapping-head">
            <span><strong>Custom mappings</strong> <code>--map</code></span>
            <div class="editor-tabs" aria-label="Mapping editor">
              <button
                type="button"
                class:on={mappingMode === 'builder'}
                aria-pressed={mappingMode === 'builder'}
                on:click={useMappingBuilder}>Builder</button
              >
              <button
                type="button"
                class:on={mappingMode === 'raw'}
                aria-pressed={mappingMode === 'raw'}
                on:click={editRawMappings}>Raw rules</button
              >
            </div>
          </div>
          {#if mappingMode === 'builder'}
            <p class="mapping-help dim">
              Rules run from top to bottom. The baseline pattern is a regex; use <code>$1</code> in the
              candidate name for captured text.
            </p>
            <div class="mapping-builder">
              {#each mappings as rule, i (rule.id)}
                <div class="mapping-row">
                  <span class="rule-number" aria-hidden="true">{i + 1}</span>
                  <TextField
                    variant="dense"
                    value={rule.pattern}
                    placeholder="baseline regex, e.g. ^blocks\\."
                    aria-label="Mapping {i + 1} baseline regex"
                    readonly={busy}
                    on:input={(e) => updateMapping(rule.id, 'pattern', eventValue(e))}
                  />
                  <span class="map-arrow" aria-hidden="true">→</span>
                  <TextField
                    variant="dense"
                    value={rule.replacement}
                    placeholder="candidate name, e.g. model.layers."
                    aria-label="Mapping {i + 1} candidate replacement"
                    readonly={busy}
                    on:input={(e) => updateMapping(rule.id, 'replacement', eventValue(e))}
                  />
                  <div class="rule-actions">
                    <button type="button" disabled={busy || i === 0} title="Move mapping up" aria-label="Move mapping {i + 1} up" on:click={() => moveMapping(i, -1)}>↑</button>
                    <button type="button" disabled={busy || i === mappings.length - 1} title="Move mapping down" aria-label="Move mapping {i + 1} down" on:click={() => moveMapping(i, 1)}>↓</button>
                    <button type="button" disabled={busy} title="Remove mapping" aria-label="Remove mapping {i + 1}" on:click={() => removeMapping(rule.id)}>×</button>
                  </div>
                </div>
              {/each}
              {#if mappings.length === 0}
                <p class="empty-mappings dim">No custom mappings. Add one when equivalent tensors use different names.</p>
              {/if}
              <button type="button" class="add-mapping" disabled={busy} on:click={addMapping}>+ Add mapping</button>
            </div>
          {:else}
            <div class="fields rename">
              <label for="scope-map">
                <span class="what">One <code>PATTERN=&gt;REPLACEMENT</code> rule per line</span>
                <TextField
                  variant="dense"
                  rows={4}
                  id="scope-map"
                  value={draft.map}
                  placeholder="^blocks\\.=>model.layers."
                  spellcheck="false"
                  readonly={busy}
                  on:input={(e) => setRawMapping(eventValue(e))}
                />
                <small class="dim">Comments and blank lines are accepted. Regex captures such as <code>$1</code> are supported.</small>
              </label>
            </div>
          {/if}
          {#if mappingIssue}<p class="mapping-issue" role="alert">{mappingIssue}</p>{/if}
        </section>
        <section class="card selection">
          <header class="card-head">
            <span class="step">3 · Select</span>
            <div>
              <h3>Choose tensors</h3>
              <p>Limit which tensors enter the comparison.</p>
            </div>
          </header>
          <div class="fields">
          {#each SELECTION as f (f.key)}
            <label for="scope-{f.key}" class:narrow={f.narrow}>
              <span class="what"
                >{f.label} <code>{f.flag}</code></span
              >
              <TextField
                variant="dense"
                rows={f.rows}
                id="scope-{f.key}"
                value={draft[f.key]}
                on:input={(e) => setText(f.key, eventValue(e))}
                placeholder={f.placeholder}
                spellcheck="false"
                readonly={busy}
              />
              <small class="dim">{f.hint}</small>
              <small class="eg">
                <span class="dim">e.g.</span>
                {#each f.examples as e (e)}<code>{e}</code>{/each}
              </small>
              {#if f.key === 'names'}
                <!-- The list rather than the guess. Typing an exact name is a guess among tens of
                     thousands, and a name neither side has after the alignment scopes the comparison to
                     nothing — so what is on offer is exactly what the two sides share, under the rules
                     in the two steps above. -->
                <button
                  type="button"
                  class="picklink"
                  disabled={busy}
                  on:click={() => togglePicker('names')}
                >
                  {picking === 'names' ? 'Hide the list' : 'Pick from both checkpoints…'}
                </button>
                {#if picking === 'names'}
                  <PickList
                    bind:this={namesList}
                    id="scope-names-search"
                    chosen={chosenNames}
                    load={loadNames}
                    onPick={pickName}
                    disabled={busy}
                    placeholder="search the names that line up…"
                    noun="that line up on both sides"
                    emptyNote="Nothing lines up yet — the two sides share no tensor name. Try the steps above."
                  />
                {/if}
              {/if}
            </label>
          {/each}
          </div>
          <label class="check">
            <input
              type="checkbox"
              checked={draft.onlyTensors}
              disabled={busy}
              on:change={(e) => setFlag('onlyTensors', eventChecked(e))}
            />
            <span><strong>Tensors only</strong> <code>--only-tensors</code></span>
            <small class="dim">Skip checkpoint metadata. Applying any tensor filter also leaves metadata out.</small>
          </label>
        </section>

        <!-- **Decode last, because it is about the bytes.** The three steps above decide which tensors
             are compared; this one decides how the packed ones are *read* when a comparison reaches
             their bytes. It changes what a verification answers, so it lives with the settings and
             travels in the link, rather than being an option of the button that starts the run. -->
        <section class="card packing">
          <header class="card-head">
            <span class="step">4 · Decode</span>
            <div>
              <h3>Unpack quantized experts</h3>
              <p>How each side packs expert indices into a 16-bit word.</p>
            </div>
          </header>
          <div class="fields paired">
            {#each PACKING as f (f.key)}
              <label for="scope-{f.key}" class:narrow={f.narrow}>
                <span class="what">{f.label} <code>{f.flag}</code></span>
                <TextField
                  variant="dense"
                  rows={f.rows}
                  id="scope-{f.key}"
                  value={draft[f.key]}
                  on:input={(e) => setText(f.key, eventValue(e))}
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
          <p class="tip dim">Nothing here declares its packing, so it is said: a sparse <code>[4]</code> baseline against a merged <code>[3,3,3,3,3]</code> candidate is one repack of the same weights.</p>
        </section>

      </div>

      <!-- Every action on this panel, in one place. *Clear* used to sit at the top-right of the
           header and *Apply* at the bottom-right of the body — the two things you can do to a
           selection, at opposite ends of a panel tall enough to scroll. And there was no way to say
           "nothing, thanks": the header toggle closes it, but a fold control is not an answer to a
           form, and closing it left whatever had been typed sitting in the boxes. *Close* rather than
           *Cancel*, because there is nothing running to cancel — the panel is most often opened just
           to see what is set. -->
      <div class="acts">
        <button type="button" class="quiet" on:click={closePanel}>Close</button>
        {#if active}
          <button type="button" class="quiet" disabled={busy} on:click={clear}>Reset settings</button>
        {/if}
        <button type="button" class="go" disabled={busy} on:click={apply}>Apply settings</button>
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
    max-height: min(70vh, 680px);
    overflow: auto;
    padding-right: 3px;
  }
  /* Selection is the broad job and gets the tall left rail. Re-rooting and name correspondence are
     independent, shorter jobs on the right. At narrow widths they become an ordinary reading order. */
  /* Two rows, not two columns of unequal height: the two *alignment* steps sit side by side at the top
     and the selection spans the width below them, so the panel reads in the order the work happens.
     Laid out as columns, the tallest card (the selection, four fields and a name picker) took the whole
     left side and read as step one whatever its number said. */
  .cards {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 10px;
    align-items: start;
  }
  .card {
    min-width: 0;
    padding: 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg-panel);
  }
  /* The width, under the two steps it depends on. */
  .selection {
    grid-column: 1 / -1;
  }
  /* Its four fields laid out along that width rather than stacked in a column. */
  .selection .fields {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 10px;
    align-items: start;
  }
  .card-head {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    margin-bottom: 9px;
  }
  .card-head h3,
  .card-head p {
    margin: 0;
  }
  .card-head h3 {
    font-size: 12.5px;
    line-height: 1.3;
  }
  .card-head p {
    margin-top: 1px;
    color: var(--fg-dim);
    font-size: 11px;
    line-height: 1.35;
  }
  .step {
    flex: 0 0 auto;
    min-width: 4.2em;
    padding: 2px 5px;
    border-radius: 3px;
    color: var(--accent);
    background: color-mix(in srgb, var(--accent) 12%, transparent);
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-align: center;
    text-transform: uppercase;
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
  .paired label.narrow,
  .rename label {
    max-width: none;
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
  .check strong,
  .choice strong {
    font-weight: 600;
  }
  .choice {
    display: flex;
    align-items: flex-start;
    gap: 7px;
    padding: 7px 8px;
    border-radius: 4px;
    background: var(--bg-elev);
  }
  .choice input {
    margin-top: 2px;
  }
  .choice span {
    min-width: 0;
  }
  .choice small {
    display: block;
    margin-top: 2px;
  }
  .rename {
    margin-top: 9px;
  }
  .mapping-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-top: 10px;
  }
  .editor-tabs {
    display: inline-flex;
    padding: 2px;
    border-radius: 4px;
    background: var(--bg-elev);
  }
  .editor-tabs button {
    border: none;
    border-radius: 3px;
    padding: 2px 7px;
    color: var(--fg-dim);
    background: none;
    font: inherit;
    font-size: 10.5px;
    cursor: pointer;
  }
  .editor-tabs button.on {
    color: var(--fg);
    background: var(--bg-hover);
  }
  .mapping-help,
  .empty-mappings {
    margin: 6px 0;
    font-size: 11px;
    line-height: 1.4;
  }
  .mapping-builder {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 7px;
  }
  .mapping-row {
    display: grid;
    grid-template-columns: 1.5em minmax(10ch, 1fr) auto minmax(10ch, 1fr) auto;
    align-items: center;
    gap: 5px;
  }
  .rule-number,
  .map-arrow {
    color: var(--fg-dim);
    text-align: center;
  }
  .rule-actions {
    display: inline-flex;
    gap: 2px;
  }
  .rule-actions button,
  .add-mapping {
    border: 1px solid var(--border);
    border-radius: 3px;
    color: var(--fg-dim);
    background: none;
    font: inherit;
    cursor: pointer;
  }
  .rule-actions button {
    width: 22px;
    height: 24px;
    padding: 0;
  }
  .rule-actions button:hover:not(:disabled),
  .add-mapping:hover:not(:disabled) {
    color: var(--fg);
    border-color: var(--accent);
  }
  .add-mapping {
    align-self: flex-start;
    padding: 3px 8px;
    font-size: 11px;
  }
  .mapping-issue {
    margin: 6px 0 0;
    color: var(--danger);
    font-size: 11px;
  }
  .tip {
    margin: 8px 0 0;
    font-size: 11px;
    line-height: 1.4;
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
  /* The link that opens a picker under a field. */
  .picklink {
    align-self: start;
    font: inherit;
    font-size: 11.5px;
    color: var(--accent);
    background: none;
    border: none;
    padding: 2px 0;
    cursor: pointer;
    text-decoration: underline;
  }
  /* All three together, at the end of the panel: close, reset, apply. */
  .acts {
    position: sticky;
    bottom: 0;
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
    background: var(--bg-elev);
  }
  .acts .go {
    font: inherit;
    font-size: 12px;
    color: var(--fg);
    background: var(--bg-elev);
    border: 1px solid var(--accent);
    border-radius: 4px;
    padding: 4px 11px;
    cursor: pointer;
  }
  .acts .go:hover:not(:disabled) {
    background: var(--bg-hover);
  }
  @media (max-width: 920px) {
    .cards {
      grid-template-columns: minmax(0, 1fr);
    }
    .selection {
      grid-column: auto;
    }
  }
  @media (max-width: 620px) {
    .mapping-row {
      grid-template-columns: 1.5em minmax(0, 1fr) auto;
    }
    .mapping-row .map-arrow {
      display: none;
    }
    .mapping-row :global(.field:nth-of-type(2)) {
      grid-column: 2;
    }
    .rule-actions {
      grid-column: 3;
      grid-row: 1 / span 2;
      flex-direction: column;
    }
  }
</style>
