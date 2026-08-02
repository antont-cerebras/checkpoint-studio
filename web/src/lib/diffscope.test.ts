// The diff scope's URL round trip. The server side of the same contract is `src/web/diffscope.rs`,
// whose own tests cover what each parameter *selects*; these cover the shape the browser carries.

import { describe, expect, it } from "vitest";
import {
  emptyScope,
  isScopeActive,
  sameScope,
  scopeFromQuery,
  scopeSummary,
  scopeToQuery,
  type DiffScopeParams,
} from "./diffscope";

const scope = (over: Partial<DiffScopeParams> = {}): DiffScopeParams => ({
  ...emptyScope(),
  ...over,
});

describe("an empty scope", () => {
  it("narrows nothing", () => {
    expect(isScopeActive(emptyScope())).toBe(false);
    expect(scopeToQuery(emptyScope())).toEqual([]);
  });

  // A UI that always sends its boxes would otherwise put `&names=&dtype_is=` on every URL, and lean on
  // the server treating blank as unset.
  it("sends nothing for whitespace-only boxes", () => {
    expect(scopeToQuery(scope({ name: "   \n  ", names: " " }))).toEqual([]);
    expect(isScopeActive(scope({ name: "  " }))).toBe(false);
  });
});

describe("the URL round trip", () => {
  it("carries every field back unchanged", () => {
    const s = scope({
      name: "model.layers.1.*\n!*.bias",
      names: "lm_head.weight,model.norm.weight",
      dtypeIs: "F*",
      shapeIs: "768,**",
      map: "^blocks\\.=>model.layers.",
      onlyTensors: true,
      alignFused: false,
      subtree: "",
      subtreeNew: "",
    });
    const q = new URLSearchParams(scopeToQuery(s));
    expect(scopeFromQuery(q)).toEqual(s);
  });

  it("emits only what is set", () => {
    expect(scopeToQuery(scope({ dtypeIs: "BF16" }))).toEqual([
      ["dtype_is", "BF16"],
    ]);
    expect(
      scopeToQuery(
        scope({
          onlyTensors: true,
          alignFused: false,
          subtree: "",
          subtreeNew: "",
        }),
      ),
    ).toEqual([["only_tensors", "1"]]);
  });

  it("reads a scope out of a URL that has none", () => {
    expect(scopeFromQuery(new URLSearchParams("against=%2Fa"))).toEqual(
      emptyScope(),
    );
  });

  // `--name` is repeatable on the command line; a repeated query key collapses to its last value in the
  // server's map, so the list travels as newlines. It must survive encoding.
  it("survives percent-encoding of the newline-separated globs", () => {
    const s = scope({ name: "a.*\n!b.*" });
    const round = scopeFromQuery(
      new URLSearchParams(new URLSearchParams(scopeToQuery(s)).toString()),
    );
    expect(round.name).toBe("a.*\n!b.*");
  });
});

describe("comparing two scopes", () => {
  it("ignores surrounding whitespace, so an edit that changes nothing refetches nothing", () => {
    expect(sameScope(scope({ name: "a.*" }), scope({ name: "  a.*  " }))).toBe(
      true,
    );
    expect(sameScope(scope({ name: "a.*" }), scope({ name: "b.*" }))).toBe(
      false,
    );
    expect(
      sameScope(
        scope({
          onlyTensors: true,
          alignFused: false,
          subtree: "",
          subtreeNew: "",
        }),
        scope(),
      ),
    ).toBe(false);
  });
});

describe("saying the scope in one line", () => {
  it("reads as a reason for the row count, not a parameter dump", () => {
    expect(
      scopeSummary(
        scope({
          name: "model.layers.1.*\n!*.bias",
          dtypeIs: "F*",
          onlyTensors: true,
          alignFused: false,
          subtree: "",
          subtreeNew: "",
        }),
      ),
    ).toBe("name model.layers.1.* !*.bias · dtype F* · tensors only");
  });

  it("counts exact names rather than listing them", () => {
    expect(scopeSummary(scope({ names: "a.w,b.w,c.w" }))).toBe("3 exact names");
    expect(scopeSummary(scope({ names: "a.w" }))).toBe("1 exact name");
  });

  it("says nothing when nothing is set", () => {
    expect(scopeSummary(emptyScope())).toBe("");
  });

  it("reports a shape glob", () => {
    expect(scopeSummary(scope({ shapeIs: "768,*" }))).toBe("shape 768,*");
  });

  it("counts rename rules rather than printing them", () => {
    expect(scopeSummary(scope({ map: "^a\\.=>b.\n^c\\.=>d." }))).toBe(
      "2 rename rules",
    );
    expect(scopeSummary(scope({ map: "^a\\.=>b." }))).toBe("1 rename rule");
  });

  // A rename rule alone narrows nothing but *changes* the comparison, so it counts as active — the bar
  // must offer to clear it.
  it("treats a rename rule as an active scope", () => {
    expect(isScopeActive(scope({ map: "^a=>b" }))).toBe(true);
  });
});

describe("the subtree scope", () => {
  it("travels as a query parameter per side, and back", () => {
    const s = {
      ...emptyScope(),
      subtree: "language_model",
      subtreeNew: "model",
    };
    expect(scopeToQuery(s)).toEqual([
      ["subtree", "language_model"],
      ["subtree_new", "model"],
    ]);
    const back = scopeFromQuery(
      new URLSearchParams("subtree=language_model&subtree_new=model"),
    );
    expect(back.subtree).toBe("language_model");
    expect(back.subtreeNew).toBe("model");
    expect(sameScope(s, back)).toBe(true);
  });

  it("counts as narrowing the comparison, so the bar offers to clear it", () => {
    expect(isScopeActive({ ...emptyScope(), subtree: "language_model" })).toBe(
      true,
    );
    expect(isScopeActive({ ...emptyScope(), subtreeNew: "model" })).toBe(true);
    expect(isScopeActive(emptyScope())).toBe(false);
  });

  it("says which side each subtree applies to", () => {
    expect(scopeSummary({ ...emptyScope(), subtree: "language_model" })).toBe(
      "baseline from #language_model",
    );
    expect(scopeSummary({ ...emptyScope(), subtreeNew: " model " })).toBe(
      "newer side from #model",
    );
    expect(
      scopeSummary({
        ...emptyScope(),
        subtree: "language_model",
        subtreeNew: "model",
      }),
    ).toBe("baseline from #language_model · newer side from #model");
  });
});
