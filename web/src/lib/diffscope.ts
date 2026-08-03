import { SCOPE_PARAMS } from './params.generated';

// How a diff is narrowed, as the browser holds it — the CLI's selection flags, in a URL.
//
// `checkpoint-studio diff --name 'model.layers.1.*'` scopes a comparison to nineteen tensors of
// 117,664. The server side of this is `src/web/diffscope.rs`, which shares the CLI's own filter
// builders; this is the shape the UI edits and the URL carries.
//
// It lives in the hash rather than in component state because that is this app's rule for view state:
// a scoped comparison is a thing you send someone, and a reload should land on the same nineteen
// tensors rather than on all 117,664.

/**
 * The selection, exactly the parameters the two diff routes accept.
 *
 * **Derived from the generated table** (`params.generated.ts`, written from `src/web/params.rs`), so a
 * parameter renamed on the server renames the field here and TypeScript fails to compile wherever the
 * old name is used. This was a hand-written interface beside a hand-written list of query keys; the
 * *keys* were contract-tested against the server's allowlist, which catches a key the server would
 * refuse and not one this client reads back under the wrong name.
 *
 * The doc comments for what each parameter *means* stay here — the generated file carries names, not
 * explanations:
 *
 * - `name` — `--name`: one glob per line, `!` to exclude. Newlines because a repeated query key would
 *   collapse to its last value in the server's `HashMap` query.
 * - `names` — `--names`: exact names, one per line or comma-separated.
 * - `dtypeIs` / `shapeIs` — `--dtype-is` / `--shape-is`: globs over the dtype and the dimensions.
 * - `map` — `--map`: `PATTERN=>REPLACEMENT` rename rules, one per line, applied to the baseline's names
 *   *before* comparing, so two naming schemes line up instead of reading as added+removed.
 * - `onlyTensors` — `--only-tensors`: skip the metadata comparison. *Any* filter also suppresses it,
 *   which is the CLI's rule, since no glob can select a metadata entry.
 * - `alignFused` — `--align-fused`: line an **unfused** checkpoint up with its **fused** counterpart.
 *   Two layouts of one model share no tensor name, so a plain comparison reports every tensor of both
 *   sides as one-sided; this drops the per-expert index (256 tensors fold onto the one fused tensor,
 *   shown as `×256`) and applies the standard layout synonyms, to both sides.
 * - `subtree` / `subtreeNew` — `SOURCE#subtree`, per side: compare *from inside* a subtree, so a
 *   multimodal checkpoint's `language_model.model.…` lines up with a converted `model.…` and the
 *   siblings (`vision_tower.…`) are out of scope rather than removed. The CLI spells this on the
 *   operand, which is what the offered command shows.
 */
export type DiffScopeParams = {
  [P in TextParam as P['field']]: string;
} & {
  [P in SwitchParam as P['field']]: boolean;
};

/** One row of the generated table, by kind. */
type ScopeParam = (typeof SCOPE_PARAMS)[number];
type TextParam = Extract<ScopeParam, { kind: 'text' }>;
type SwitchParam = Extract<ScopeParam, { kind: 'switch' }>;

/** The text fields and the switches, as lists to walk. Narrowed from the generated rows, so a new row
 * joins the right list by its `kind` and nothing here changes. */
const TEXTS = SCOPE_PARAMS.filter((p): p is TextParam => p.kind === 'text');
const SWITCHES = SCOPE_PARAMS.filter((p): p is SwitchParam => p.kind === 'switch');

/** No narrowing — the whole comparison. */
export function emptyScope(): DiffScopeParams {
  const s = {} as Record<string, string | boolean>;
  for (const p of TEXTS) s[p.field] = '';
  for (const p of SWITCHES) s[p.field] = false;
  return s as DiffScopeParams;
}

/** Whether anything narrows the comparison. Drives whether the bar shows a "clear" and whether the
 * request carries any scope at all. */
export function isScopeActive(s: DiffScopeParams): boolean {
  return (
    TEXTS.some((p) => s[p.field].trim() !== '') || SWITCHES.some((p) => s[p.field])
  );
}

/**
 * The scope as query parameters, for both the API and the URL hash.
 *
 * Only what is set: an empty box must not be sent, because the server reads a present-but-empty
 * parameter as "unset" only by explicit care, and a URL full of `&names=&dtype_is=` is unreadable.
 */
export function scopeToQuery(s: DiffScopeParams): [string, string][] {
  const out: [string, string][] = [];
  for (const p of TEXTS) {
    const v = s[p.field].trim();
    if (v !== '') out.push([p.key, v]);
  }
  for (const p of SWITCHES) {
    if (s[p.field]) out.push([p.key, '1']);
  }
  return out;
}

/** Read a scope back out of a `URLSearchParams` — the inverse of [[scopeToQuery]]. */
export function scopeFromQuery(q: URLSearchParams): DiffScopeParams {
  const s = emptyScope();
  for (const p of TEXTS) s[p.field] = q.get(p.key) ?? '';
  for (const p of SWITCHES) s[p.field] = q.get(p.key) === '1';
  return s;
}

/** Whether two scopes describe the same selection — so a re-render can skip a refetch. */
export function sameScope(a: DiffScopeParams, b: DiffScopeParams): boolean {
  return (
    TEXTS.every((p) => a[p.field].trim() === b[p.field].trim()) &&
    SWITCHES.every((p) => a[p.field] === b[p.field])
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
