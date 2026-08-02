// How a diff is narrowed, as the browser holds it — the CLI's selection flags, in a URL.
//
// `checkpoint-studio diff --name 'model.layers.1.*'` scopes a comparison to nineteen tensors of
// 117,664. The server side of this is `src/web/diffscope.rs`, which shares the CLI's own filter
// builders; this is the shape the UI edits and the URL carries.
//
// It lives in the hash rather than in component state because that is this app's rule for view state:
// a scoped comparison is a thing you send someone, and a reload should land on the same nineteen
// tensors rather than on all 117,664.

/** The selection, exactly the parameters the two diff routes accept. */
export interface DiffScopeParams {
  /** `--name`: one glob per line, `!` to exclude. Newlines because a repeated query key would
   * collapse to its last value in the server's `HashMap` query. */
  name: string;
  /** `--names`: exact names, comma-separated. */
  names: string;
  /** `--dtype-is`: a glob against the dtype, case-insensitive. */
  dtypeIs: string;
  /** `--shape-is`: a glob over comma- or `x`-separated dims. */
  shapeIs: string;
  /** `--map`: `PATTERN=>REPLACEMENT` rename rules, one per line, applied to the baseline's names
   * *before* comparing — so two naming schemes line up instead of reading as added+removed.
   *
   * Both views apply them. The side-by-side rebuilds the baseline's tree from the renamed names, since
   * groups are named from name segments and rewriting a leaf alone leaves the path above it describing
   * the name it used to have. */
  map: string;
  /** `--only-tensors`: skip the metadata comparison. Note that *any* filter also suppresses it — the
   * CLI's rule, since no glob can select a metadata entry. */
  onlyTensors: boolean;
  /**
   * `--align-fused`: line an **unfused** checkpoint up with its **fused** counterpart.
   *
   * Two layouts of one model share no tensor name, so a plain comparison reports every tensor of both
   * sides as one-sided — 80,107 against 933, "nothing lines up", which answers nothing. This drops the
   * per-expert index (256 tensors fold onto the one fused tensor that holds them, shown as `×256`) and
   * applies the standard layout synonyms. Applied to both sides; each rule is a no-op on a side that is
   * already fused.
   */
  alignFused: boolean;
  /**
   * `SOURCE#subtree`: compare *from inside* a subtree, per side.
   *
   * A Hugging Face multimodal checkpoint keeps its language model under `language_model.…`; the
   * converted one has it at the root. Without this the two share no tensor name and the comparison
   * reports every tensor of both sides as one-sided. Re-rooting a side keys its tensors by their
   * sub-path — so `language_model.model.layers.0.w` lines up with `model.layers.0.w` — and leaves the
   * siblings (`vision_tower.…`) out of scope rather than calling them removed.
   *
   * The CLI spells this on the operand (`diff 'hf#language_model' converted`), which is what the
   * offered command shows; here it is a field per side, because the address boxes are already full of
   * path.
   */
  subtree: string;
  subtreeNew: string;
}

/** No narrowing — the whole comparison. */
export function emptyScope(): DiffScopeParams {
  return {
    name: '',
    names: '',
    dtypeIs: '',
    shapeIs: '',
    map: '',
    onlyTensors: false,
    alignFused: false,
    subtree: '',
    subtreeNew: '',
  };
}

/** Whether anything narrows the comparison. Drives whether the bar shows a "clear" and whether the
 * request carries any scope at all. */
export function isScopeActive(s: DiffScopeParams): boolean {
  return (
    s.name.trim() !== '' ||
    s.names.trim() !== '' ||
    s.dtypeIs.trim() !== '' ||
    s.shapeIs.trim() !== '' ||
    s.map.trim() !== '' ||
    s.onlyTensors ||
    s.alignFused ||
    s.subtree.trim() !== '' ||
    s.subtreeNew.trim() !== ''
  );
}

/** The fields, paired with their URL and API key — one list, so encoding and decoding cannot drift. */
const TEXT_FIELDS = [
  ['name', 'name'],
  ['names', 'names'],
  ['dtypeIs', 'dtype_is'],
  ['shapeIs', 'shape_is'],
  ['map', 'map'],
  ['subtree', 'subtree'],
  ['subtreeNew', 'subtree_new'],
] as const;

/**
 * The scope as query parameters, for both the API and the URL hash.
 *
 * Only what is set: an empty box must not be sent, because the server reads a present-but-empty
 * parameter as "unset" only by explicit care, and a URL full of `&names=&dtype_is=` is unreadable.
 */
export function scopeToQuery(s: DiffScopeParams): [string, string][] {
  const out: [string, string][] = [];
  for (const [field, key] of TEXT_FIELDS) {
    const v = s[field].trim();
    if (v !== '') out.push([key, v]);
  }
  if (s.onlyTensors) out.push(['only_tensors', '1']);
  if (s.alignFused) out.push(['align_fused', '1']);
  return out;
}

/** Read a scope back out of a `URLSearchParams` — the inverse of [[scopeToQuery]]. */
export function scopeFromQuery(q: URLSearchParams): DiffScopeParams {
  const s = emptyScope();
  for (const [field, key] of TEXT_FIELDS) {
    s[field] = q.get(key) ?? '';
  }
  s.onlyTensors = q.get('only_tensors') === '1';
  s.alignFused = q.get('align_fused') === '1';
  return s;
}

/** Whether two scopes describe the same selection — so a re-render can skip a refetch. */
export function sameScope(a: DiffScopeParams, b: DiffScopeParams): boolean {
  return (
    TEXT_FIELDS.every(([field]) => a[field].trim() === b[field].trim()) &&
    a.onlyTensors === b.onlyTensors &&
    a.alignFused === b.alignFused
  );
}

/**
 * The scope said in one line, for the header: `name model.layers.1.* · dtype F* · tensors only`.
 *
 * Short and readable rather than a parameter dump: it sits next to the `matched M of N` count, and its
 * job is to answer "why am I looking at nineteen rows".
 */
export function scopeSummary(s: DiffScopeParams): string {
  const parts: string[] = [];
  const names = s.name
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l !== '');
  if (names.length > 0) parts.push(`name ${names.join(' ')}`);
  if (s.names.trim() !== '') {
    const n = s.names.split(',').filter((x) => x.trim() !== '').length;
    parts.push(`${n} exact name${n === 1 ? '' : 's'}`);
  }
  if (s.dtypeIs.trim() !== '') parts.push(`dtype ${s.dtypeIs.trim()}`);
  if (s.shapeIs.trim() !== '') parts.push(`shape ${s.shapeIs.trim()}`);
  const rules = s.map.split('\n').filter((l) => l.trim() !== '').length;
  if (rules > 0) parts.push(`${rules} rename rule${rules === 1 ? '' : 's'}`);
  if (s.onlyTensors) parts.push('tensors only');
  if (s.alignFused) parts.push('unfused ↔ fused aligned');
  if (s.subtree.trim() !== '') parts.push(`baseline from #${s.subtree.trim()}`);
  if (s.subtreeNew.trim() !== '') parts.push(`candidate from #${s.subtreeNew.trim()}`);
  return parts.join(' · ');
}
