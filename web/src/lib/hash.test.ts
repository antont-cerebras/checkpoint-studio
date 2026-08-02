// The hash is the whole view state, so what matters is that it ROUND-TRIPS: a link
// someone shares (or the browser's back button) must reopen the exact view. These
// tests drive that in both directions, and pin the fallbacks that keep a
// hand-edited or stale link from stranding the app on a blank screen.

import { describe, expect, it } from "vitest";
import {
  DV_KEYS,
  globalQuery,
  hashFor,
  parseGlobals,
  parseScreen,
  screenToHash,
  type Globals,
  type Screen,
} from "./hash";
import { emptyScope } from "./diffscope";

const DEFAULTS: Globals = {
  ckpt: "",
  filter: "",
  sortKey: "none",
  sortDir: "asc",
  compact: false,
  searching: false,
  search: "",
};

const SCREENS: Screen[] = [
  { kind: "tree" },
  { kind: "files" },
  { kind: "stats" },
  { kind: "health" },
  { kind: "layout" },
  { kind: "layout", file: "model-00001-of-00002.safetensors" },
  {
    kind: "detail",
    tensor: "model.layers.0.mlp.gate_proj.weight",
    tab: "info",
  },
  { kind: "detail", tensor: "lm_head.weight", tab: "heatmap" },
  { kind: "preview", path: "/ckpt/config.json", name: "config.json" },
  { kind: "diff", against: "/models/checkpoint_1000" },
  { kind: "open" },
  { kind: "compare", against: "/models/checkpoint_1000", right: "" },
  { kind: "compare", against: "/models/base", right: "/models/other" },
  // A path with characters that must survive percent-encoding in the hash.
  { kind: "diff", against: "/models/a b/model-00001-of-00002.safetensors" },
  // Turned round: the open checkpoint as the baseline. It changes what the report *says* — added and
  // removed trade places — so it has to be in the link, like every other piece of view state.
  { kind: "diff", against: "/models/checkpoint_1000", swapped: true },
  // Every tensor rather than collapsed families, and the sections folded away: both are what the
  // reader set deliberately, so a reload — and a link — has to land on them.
  { kind: "diff", against: "/models/base", full: true },
  {
    kind: "diff",
    against: "/models/base",
    closed: ["tensors_added", "metadata"],
  },
  {
    kind: "diff",
    against: "/models/base",
    swapped: true,
    full: true,
    closed: ["tensors_changed"],
  },
  // A *scoped* comparison is a link you send someone, so the selection round-trips like every other
  // piece of view state. Both screens carry it.
  {
    kind: "diff",
    against: "/models/base",
    scope: {
      name: "model.layers.1.*\n!*.bias",
      names: "",
      dtypeIs: "F*",
      shapeIs: "",
      map: "",
      onlyTensors: true,
      alignFused: false,
      subtree: "",
      subtreeNew: "",
    },
  },
  {
    kind: "compare",
    against: "/models/base",
    right: "/models/other",
    scope: {
      name: "",
      names: "lm_head.weight",
      dtypeIs: "",
      shapeIs: "768,**",
      map: "",
      onlyTensors: false,
      alignFused: false,
      subtree: "",
      subtreeNew: "",
    },
  },
];

describe("screen round-trip", () => {
  it.each(SCREENS)("survives hash → parse → hash for %j", (s) => {
    const parsed = parseScreen(`#${screenToHash(s)}`);
    expect(parsed).toEqual({
      ...s,
      ...(s.kind === "detail" ? { dv: undefined } : {}),
    });
    expect(screenToHash(parsed)).toBe(screenToHash(s));
  });

  it("carries every data-view param through unchanged", () => {
    const dv = Object.fromEntries(DV_KEYS.map((k, i) => [k, String(i)]));
    const s: Screen = { kind: "detail", tensor: "w", tab: "values", dv };
    expect(parseScreen(`#${screenToHash(s)}`)).toEqual(s);
  });

  it("omits the dv object entirely when no param is set", () => {
    const h = screenToHash({ kind: "detail", tensor: "w", tab: "info" });
    expect(h).toBe("detail?t=w&tab=info");
    expect(parseScreen(h)).toEqual({
      kind: "detail",
      tensor: "w",
      tab: "info",
      dv: undefined,
    });
  });

  // Tensor names carry `/` and `.`, and metadata keys have carried `&`/`?`/`#`/spaces.
  // A single round of encoding has to survive each of them, or the link opens the
  // wrong tensor (or no tensor).
  it.each([
    "model/layers.0/weight",
    "a&b=c?d#e",
    "name with spaces",
    "unicode·näme",
    "plus+sign",
    "100%",
  ])("encodes and recovers the awkward name %s", (name) => {
    expect(
      parseScreen(
        `#${screenToHash({ kind: "detail", tensor: name, tab: "info" })}`,
      ),
    ).toEqual({
      kind: "detail",
      tensor: name,
      tab: "info",
      dv: undefined,
    });
  });

  it("a file path with a hash character still opens the preview it names", () => {
    const s: Screen = {
      kind: "preview",
      path: "/ckpt/a#b.json",
      name: "a#b.json",
    };
    expect(parseScreen(`#${screenToHash(s)}`)).toEqual(s);
  });
});

describe("screen fallbacks", () => {
  it('falls back to the tree for an unknown screen, an empty hash, or bare "#"', () => {
    for (const h of ["", "#", "#nope", "#detail"] /* detail without ?t= */) {
      expect(parseScreen(h)).toEqual({ kind: "tree" });
    }
  });

  it("falls back to the tree when a preview has no path", () => {
    expect(parseScreen("#preview?name=x")).toEqual({ kind: "tree" });
  });

  it("names a preview after its path when the name is missing", () => {
    expect(parseScreen("#preview?path=/a/b.json")).toEqual({
      kind: "preview",
      path: "/a/b.json",
      name: "/a/b.json",
    });
  });

  it("falls back to the info tab for an unknown tab", () => {
    expect(parseScreen("#detail?t=w&tab=bogus")).toMatchObject({ tab: "info" });
    expect(parseScreen("#detail?t=w")).toMatchObject({ tab: "info" });
  });

  it("treats a layout with no file as the layout screen (the app picks a shard)", () => {
    expect(parseScreen("#layout")).toEqual({ kind: "layout", file: undefined });
  });
});

describe("global state round-trip", () => {
  it("is empty for the default view, so a plain URL stays plain", () => {
    expect(globalQuery(DEFAULTS)).toBe("");
    expect(hashFor({ kind: "tree" }, DEFAULTS)).toBe("tree");
  });

  it("round-trips filter, sort, compact and search together", () => {
    const g: Globals = {
      ckpt: "/models/checkpoint_1000",
      filter: "dtype:BF16 shape:(2048,2048)",
      sortKey: "size",
      sortDir: "desc",
      compact: true,
      searching: true,
      search: "gate_proj",
    };
    expect(parseGlobals(`#tree?${globalQuery(g)}`)).toEqual(g);
  });

  it("keeps search MODE with an empty query distinct from no search", () => {
    const open = { ...DEFAULTS, searching: true, search: "" };
    expect(globalQuery(open)).toBe("q=");
    expect(parseGlobals("#tree?q=")).toEqual(open);
    expect(parseGlobals("#tree")).toEqual(DEFAULTS);
  });

  it("trims the filter, so trailing whitespace never lands in the URL", () => {
    expect(globalQuery({ ...DEFAULTS, filter: "  dtype:F32  " })).toBe(
      "filter=dtype%3AF32",
    );
  });

  it("drops an unknown sort key and defaults the direction to ascending", () => {
    expect(parseGlobals("#tree?sort=bogus.desc")).toMatchObject({
      sortKey: "none",
    });
    expect(parseGlobals("#tree?sort=name")).toMatchObject({
      sortKey: "name",
      sortDir: "asc",
    });
    expect(parseGlobals("#tree?sort=name.sideways")).toMatchObject({
      sortDir: "asc",
    });
  });

  it('reads compact only from an exact "1"', () => {
    expect(parseGlobals("#tree?compact=1").compact).toBe(true);
    expect(parseGlobals("#tree?compact=0").compact).toBe(false);
    expect(parseGlobals("#tree?compact=true").compact).toBe(false);
  });
});

describe("hashFor", () => {
  const g: Globals = { ...DEFAULTS, filter: "w", compact: true };

  it("joins the global state with & when the screen already has a query", () => {
    expect(hashFor({ kind: "detail", tensor: "w", tab: "info" }, g)).toBe(
      "detail?t=w&tab=info&filter=w&compact=1",
    );
  });

  it("joins it with ? when the screen has none", () => {
    expect(hashFor({ kind: "stats" }, g)).toBe("stats?filter=w&compact=1");
  });

  it("produces a hash both halves can be read back out of", () => {
    const s: Screen = {
      kind: "detail",
      tensor: "x/y.w",
      tab: "values",
      dv: { mode: "window", rows: "16" },
    };
    const full = `#${hashFor(s, g)}`;
    expect(parseScreen(full)).toEqual(s);
    expect(parseGlobals(full)).toEqual(g);
  });
});

describe("the checkpoint a link names", () => {
  // A URL that named a screen and a filter but not the checkpoint described a view of whatever
  // happened to be loaded — the same link could look right on a different checkpoint.
  it("rides in the hash and comes back out", () => {
    const g: Globals = { ...DEFAULTS, ckpt: "/models/ckpt-1000" };
    expect(globalQuery(g)).toContain("ckpt=%2Fmodels%2Fckpt-1000");
    expect(parseGlobals(`#tree?${globalQuery(g)}`)).toEqual(g);
  });

  it("is omitted while nothing is loaded, so a bare view stays a bare URL", () => {
    expect(globalQuery(DEFAULTS)).toBe("");
    expect(parseGlobals("#tree")).toMatchObject({ ckpt: "" });
  });

  // Paths carry spaces, `#`, `&` and `?`; each has to survive one round of encoding or the
  // link opens the wrong checkpoint (or nothing).
  it.each([
    "/models/a b/ckpt",
    "/models/ckpt#2",
    "/models/a&b?c",
    "hf://owner/name",
    "s3://bucket/prefix/",
  ])("survives the awkward path %s", (ckpt) => {
    const parsed = parseGlobals(`#tree?${globalQuery({ ...DEFAULTS, ckpt })}`);
    expect(parsed.ckpt).toBe(ckpt);
  });
});

describe("the comparison screen", () => {
  // A comparison is two checkpoints, and both are in the URL: the baseline here, the other side in
  // the `ckpt` global. So a link reproduces the whole comparison, not half of it.
  it("carries its baseline, and survives an awkward one", () => {
    expect(
      screenToHash({ kind: "compare", against: "/models/a b/ckpt", right: "" }),
    ).toBe("compare?against=%2Fmodels%2Fa%20b%2Fckpt");
    expect(parseScreen("#compare?against=%2Fmodels%2Fa%20b%2Fckpt")).toEqual({
      kind: "compare",
      against: "/models/a b/ckpt",
      right: "",
    });
  });

  // Both sides ride in the URL, so a comparison of any two checkpoints is shareable — neither has
  // to be the one that is open.
  it("carries an overridden newer side too", () => {
    const s: Screen = {
      kind: "compare",
      against: "/models/base",
      right: "/models/other",
    };
    expect(screenToHash(s)).toBe(
      "compare?against=%2Fmodels%2Fbase&right=%2Fmodels%2Fother",
    );
    expect(parseScreen(`#${screenToHash(s)}`)).toEqual(s);
  });

  // Omitted when it is just the open checkpoint, so the common case stays a short URL.
  it("omits the newer side when it is the open checkpoint", () => {
    expect(screenToHash({ kind: "compare", against: "/a", right: "" })).toBe(
      "compare?against=%2Fa",
    );
  });

  // Unlike the diff report screen, which falls through to the tree without a baseline: this screen
  // is also where you *pick* one, so it opens empty rather than bouncing you away.
  it("opens empty when the link names no baseline", () => {
    expect(parseScreen("#compare")).toEqual({
      kind: "compare",
      against: "",
      right: "",
    });
  });
});

describe("the open screen", () => {
  // Deliberately state-free. What the screen *does* is change the server, and a URL cannot
  // capture that — a bookmark that re-pointed the server on load would be a link with a side
  // effect. So the hash records only "the prompt was open", and a reload lands back on the
  // prompt rather than on a blank screen.
  it("round-trips as a bare kind, carrying no path", () => {
    expect(screenToHash({ kind: "open" })).toBe("open");
    expect(parseScreen("#open")).toEqual({ kind: "open" });
  });

  it("ignores a path someone appends by hand, rather than opening it", () => {
    expect(parseScreen("#open?path=/somewhere/else")).toEqual({ kind: "open" });
  });
});

describe("the diff report screen", () => {
  // It navigates with no baseline from the palette, so falling through to the tree made the entry do
  // nothing — you asked for a screen and stayed on the one you were already on. It carries its own
  // path box, so an empty baseline is a usable state.
  it("opens with an empty baseline", () => {
    expect(parseScreen("#diff")).toEqual({ kind: "diff", against: "" });
    expect(parseScreen("#diff?against=")).toEqual({
      kind: "diff",
      against: "",
    });
  });

  it("percent-encodes the path so a space or hash cannot break the URL", () => {
    const h = screenToHash({ kind: "diff", against: "/a b/c#d" });
    expect(h).not.toContain(" ");
    expect(h.slice("diff?against=".length)).not.toContain("#");
    expect(parseScreen(`#${h}`)).toEqual({ kind: "diff", against: "/a b/c#d" });
  });
});

describe("the side-by-side comparison in a URL", () => {
  it("carries the family fold state, and only when it is off", () => {
    expect(screenToHash({ kind: "compare", against: "/a", right: "" })).toBe(
      "compare?against=%2Fa",
    );
    expect(
      screenToHash({ kind: "compare", against: "/a", right: "", full: true }),
    ).toBe("compare?against=%2Fa&full=1");
    expect(
      screenToHash({ kind: "compare", against: "/a", right: "/b", full: true }),
    ).toBe("compare?against=%2Fa&right=%2Fb&full=1");
  });

  it("reads it back — folded is the default, so nothing in the URL means folded", () => {
    expect(parseScreen("compare?against=%2Fa&full=1")).toEqual({
      kind: "compare",
      against: "/a",
      right: "",
      full: true,
    });
    expect(parseScreen("compare?against=%2Fa")).toEqual({
      kind: "compare",
      against: "/a",
      right: "",
    });
  });
});

describe("the orientation of a comparison", () => {
  // The bug this models away: flipping used to rewrite the two operands, and the scope is
  // *directional* — `--map` rewrites the baseline's names and cannot be inverted, a `#subtree`
  // belongs to one side. Reloading the flipped URL therefore asked the server to apply the old scope
  // to the reversed operands, which is a different comparison from the one on screen.
  it("keeps the pair and its scope canonical, and carries only the direction", () => {
    const scope = {
      ...emptyScope(),
      subtree: "language_model",
      map: "^blocks\\.=>model.layers.",
    };
    const shown = {
      kind: "compare" as const,
      against: "/hf",
      right: "/converted",
      scope,
      swapped: true,
    };
    const hash = screenToHash(shown);
    // The operands are in the order the server is asked about them, whichever way it is drawn.
    expect(hash).toContain("against=%2Fhf");
    expect(hash).toContain("right=%2Fconverted");
    expect(hash).toContain("swap=1");
    expect(hash).toContain("subtree=language_model");

    const back = parseScreen(hash);
    expect(back).toEqual(shown);
    if (back.kind !== "compare")
      throw new Error("the hash names the comparison screen");
    // The reload asks for the same comparison, with the subtree still on the side it belongs to and
    // the rename rules still against the baseline they were written for.
    expect(back.scope?.subtree).toBe("language_model");
    expect(back.scope?.subtreeNew).toBe("");
    expect(back.scope?.map).toBe("^blocks\\.=>model.layers.");
  });

  it("says nothing about direction when the pair is read the way it was asked for", () => {
    expect(
      screenToHash({ kind: "compare", against: "/a", right: "" }),
    ).not.toContain("swap");
    expect(parseScreen("compare?against=%2Fa")).toEqual({
      kind: "compare",
      against: "/a",
      right: "",
    });
  });
});
