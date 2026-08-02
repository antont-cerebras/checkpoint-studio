// The comparison view's client-side model. The alignment itself is the server's and is tested
// there; what is tested here is what the browser does with it — folding, revealing differences,
// and stepping between them.

import { describe, expect, it } from "vitest";
import {
  allGroupPaths,
  ancestorsOf,
  differingCount,
  initialExpansion,
  emptyRowsNote,
  identicalNote,
  tallyIsReadable,
  isDisjoint,
  REVEAL_LIMIT,
  tallyText,
  expandToDifferences,
  flattenDiff,
  nextDifference,
  sideText,
  clickOutcome,
  statusMark,
  swapResponse,
  swapSides,
  swapTally,
  type AlignedNode,
  type DiffStatus,
} from "./difftree";
import type { TensorInfo } from "./types";

const info = (dtype: string, shape: number[]): TensorInfo =>
  ({ name: "x", dtype, shape, size_bytes: 4, num_elements: 1 }) as TensorInfo;

const leaf = (name: string, status: DiffStatus, path = name): AlignedNode => ({
  name,
  path,
  old:
    status === "only_new"
      ? null
      : { kind: "tensor", info: info("F16", [4]), fold: null },
  new:
    status === "only_old"
      ? null
      : { kind: "tensor", info: info("U16", [4]), fold: null },
  status: { kind: status },
  differing: status === "same" ? 0 : 1,
  members: 1,
  children: [],
});

const group = (
  name: string,
  children: AlignedNode[],
  path = name,
): AlignedNode => ({
  name,
  path,
  old: {
    kind: "group",
    tensor_count: children.length,
    params: 0,
    total_size: 0,
  },
  new: {
    kind: "group",
    tensor_count: children.length,
    params: 0,
    total_size: 0,
  },
  status: {
    kind: children.some((c) => c.status.kind !== "same") ? "changed" : "same",
  },
  differing: children.reduce((n, c) => n + c.differing, 0),
  members: 1,
  children,
});

const TREE: AlignedNode[] = [
  group("model", [
    leaf("same.weight", "same", "model.same.weight"),
    group(
      "layers.0",
      [leaf("k.weight", "changed", "model.layers.0.k.weight")],
      "model.layers.0",
    ),
    leaf("added.weight", "only_new", "model.added.weight"),
  ]),
  leaf("gone.weight", "only_old"),
];

describe("flattening", () => {
  it("shows only the rows inside expanded groups", () => {
    const collapsed = flattenDiff(TREE, new Set());
    expect(collapsed.map((r) => r.node.name)).toEqual(["model", "gone.weight"]);

    const open = flattenDiff(TREE, new Set(["model"]));
    expect(open.map((r) => r.node.name)).toEqual([
      "model",
      "same.weight",
      "layers.0",
      "added.weight",
      "gone.weight",
    ]);
  });

  it("reports depth, so both columns indent alike", () => {
    const rows = flattenDiff(TREE, new Set(["model", "model.layers.0"]));
    const deep = rows.find((r) => r.node.name === "k.weight");
    expect(deep?.depth).toBe(2);
  });

  // Two checkpoints with different naming schemes share no names, so every tensor of both becomes a
  // one-sided row and the load unfolds all of them. `out.push(...recurse)` spreads those into
  // *arguments*, and past ~65k of them V8 throws `RangeError: Maximum call stack size exceeded` —
  // mid-render, so the flush that would have removed the progress bar never ran and the view hung on
  // a spinner at 100% forever. The size is the whole point of this test.
  it("flattens more rows than an argument list can hold", () => {
    const many: AlignedNode[] = Array.from({ length: 200_000 }, (_, i) =>
      leaf(`t${i}`, "only_new", `g.t${i}`),
    );
    const tree = [group("g", many)];
    expect(flattenDiff(tree, new Set(["g"]))).toHaveLength(200_001);
  });

  // One tree, one fold state: that is what makes the two columns move together. If this ever
  // returned per-side rows, lockstep would become two scroll positions to reconcile.
  it("produces one row per name, carrying both sides", () => {
    const rows = flattenDiff(TREE, new Set(["model"]));
    const added = rows.find((r) => r.node.name === "added.weight");
    expect(added?.node.old).toBeNull();
    expect(added?.node.new).not.toBeNull();
  });
});

describe("revealing the differences", () => {
  it("opens the ancestors of every difference and nothing else", () => {
    const open = expandToDifferences(TREE);
    // `model` and `layers.0` are on the way to changes…
    expect(open.has("model")).toBe(true);
    expect(open.has("model.layers.0")).toBe(true);
    // …and every difference is now visible.
    const visible = flattenDiff(TREE, open).map((r) => r.node.path);
    expect(visible).toContain("model.layers.0.k.weight");
    expect(visible).toContain("model.added.weight");
  });

  it("leaves a subtree with nothing changed folded", () => {
    const quiet: AlignedNode[] = [
      group("quiet", [leaf("a", "same", "quiet.a")]),
    ];
    expect(expandToDifferences(quiet).size).toBe(0);
  });
});

describe("stepping between differences", () => {
  const diffs = ["a", "b", "c"];

  it("starts at the first going forwards, the last going backwards", () => {
    expect(nextDifference(diffs, null)).toBe("a");
    expect(nextDifference(diffs, null, -1)).toBe("c");
  });

  it("advances and retreats", () => {
    expect(nextDifference(diffs, "a")).toBe("b");
    expect(nextDifference(diffs, "b", -1)).toBe("a");
  });

  // Wrapping rather than dead-ending: with a handful of changes in a 31k-tensor checkpoint, a
  // "next" that stops at the end makes you scroll back to the top by hand.
  it("wraps at both ends", () => {
    expect(nextDifference(diffs, "c")).toBe("a");
    expect(nextDifference(diffs, "a", -1)).toBe("c");
  });

  it("steps from a row that is not itself a difference", () => {
    expect(nextDifference(diffs, "not-in-the-list")).toBe("a");
  });

  it("has nowhere to go when the two checkpoints match", () => {
    expect(nextDifference([], null)).toBeNull();
    expect(nextDifference([], "a")).toBeNull();
  });
});

describe("jumping to a row", () => {
  it("names the groups that have to unfold to reveal it", () => {
    expect(ancestorsOf(TREE, "model.layers.0.k.weight")).toEqual([
      "model",
      "model.layers.0",
    ]);
    expect(ancestorsOf(TREE, "gone.weight")).toEqual([]);
  });

  it("names a group's own ancestors when the target is a group", () => {
    expect(ancestorsOf(TREE, "model.layers.0")).toEqual(["model"]);
  });

  it("returns nothing for a path that is not in the tree", () => {
    expect(ancestorsOf(TREE, "nope")).toEqual([]);
  });
});

describe("swapping the sides", () => {
  // Instant and local: both checkpoints are already aligned, so reading the comparison the other way
  // round is a transform, not a refetch.
  it("trades the sides and inverts added/removed", () => {
    const flipped = swapSides(TREE);
    const find = (rows: AlignedNode[], name: string) =>
      rows.find((r) => r.name === name);

    expect(find(flipped, "gone.weight")?.status.kind).toBe("only_new");
    const model = flipped.find((r) => r.name === "model");
    expect(
      model?.children.find((c) => c.name === "added.weight")?.status.kind,
    ).toBe("only_old");
    // A change is a change either way round; so is a match.
    expect(
      model?.children.find((c) => c.name === "same.weight")?.status.kind,
    ).toBe("same");
  });

  it("really moves the content, not just the labels", () => {
    const flipped = swapSides(TREE);
    const added = flipped
      .find((r) => r.name === "model")
      ?.children.find((c) => c.name === "added.weight");
    // It was new-only; now it is old-only, with its content on the old side.
    expect(added?.old).not.toBeNull();
    expect(added?.new).toBeNull();
  });

  // Twice is the identity, so the button is safe to lean on.
  it("is its own inverse", () => {
    expect(swapSides(swapSides(TREE))).toEqual(TREE);
  });

  // The whole response turns together — the two side descriptions, the counts and every row. Written
  // as one function for that reason: the tally was once left behind, so the rows read `+` where they
  // had read `-` while the chips above them did not move.
  it("turns the sides, the counts and the rows over together", () => {
    const t = {
      base: {
        spec: "/a",
        root: "/a",
        tensor_count: 1,
        served: false,
        params: 1,
        bytes: 4,
      },
      current: {
        spec: "/b",
        root: "/b",
        tensor_count: 2,
        served: true,
        params: 2,
        bytes: 8,
      },
      tally: {
        tensors: { same: 1, changed: 2, only_old: 3, only_new: 4 },
        metadata: { same: 0, changed: 0, only_old: 0, only_new: 0 },
      },
      matched: null,
      totals_labels: { size: "size", params: "params" },
      differences: ["x"],
      full: false,
      rows: TREE,
    };
    const f = swapResponse(t);
    expect(f.base.spec).toBe("/b");
    expect(f.current.spec).toBe("/a");
    expect(f.tally.tensors).toEqual({
      same: 1,
      changed: 2,
      only_old: 4,
      only_new: 3,
    });
    expect(f.rows).toEqual(swapSides(TREE));
    // The jump list is by path, and paths do not move — so it survives untouched.
    expect(f.differences).toEqual(["x"]);
    expect(swapResponse(f)).toEqual(t);
  });

  it("reverses added and removed counts for tensors and metadata", () => {
    const tally = {
      tensors: { same: 10, changed: 2, only_old: 3, only_new: 4 },
      metadata: { same: 1, changed: 5, only_old: 6, only_new: 7 },
    };
    expect(swapTally(tally)).toEqual({
      tensors: { same: 10, changed: 2, only_old: 4, only_new: 3 },
      metadata: { same: 1, changed: 5, only_old: 7, only_new: 6 },
    });
    expect(swapTally(swapTally(tally))).toEqual(tally);
  });
});

describe("what a click means", () => {
  const sides = (baseServed: boolean, currentServed: boolean) => ({
    base: {
      spec: "/base",
      root: "/base",
      tensor_count: 1,
      served: baseServed,
      params: 4,
      bytes: 8,
    },
    current: {
      spec: "/cur",
      root: "/cur",
      tensor_count: 1,
      served: currentServed,
      params: 4,
      bytes: 8,
    },
  });

  it("folds a group, whichever pane is clicked", () => {
    const g = TREE[0]!;
    expect(clickOutcome(g, "old", sides(false, true))).toEqual({
      kind: "toggle",
      path: "model",
    });
    expect(clickOutcome(g, "new", sides(false, true))).toEqual({
      kind: "toggle",
      path: "model",
    });
  });

  it("opens a tensor only from the side the server is serving", () => {
    const leaf = TREE[1]!; // gone.weight — old side only
    // The tensor's *real* name, not the row's display label — that is what a detail view opens.
    expect(clickOutcome(leaf, "old", sides(true, false))).toEqual({
      kind: "open",
      name: "x",
    });
  });

  // The bug this rule exists for: the detail screen reads the *served* checkpoint, so opening from
  // any other side showed a different checkpoint's numbers under this row's name. It is not a click —
  // the cell's tooltip carries the explanation, where it costs no space.
  it("does nothing when that side is not the served checkpoint", () => {
    const leaf = TREE[1]!;
    expect(clickOutcome(leaf, "old", sides(false, true))).toEqual({
      kind: "none",
    });
  });

  // `served` comes from the server, so two spellings of one checkpoint cannot make it wrong — which
  // string-comparing specs did.
  it("trusts the server's answer rather than comparing paths", () => {
    const leaf = TREE[1]!;
    const aliased = {
      base: {
        spec: "/models/ckpt/",
        root: "/models/ckpt",
        tensor_count: 1,
        served: true,
        params: 4,
        bytes: 8,
      },
      current: {
        spec: "/models/ckpt",
        root: "/models/ckpt",
        tensor_count: 1,
        served: false,
        params: 4,
        bytes: 8,
      },
    };
    expect(clickOutcome(leaf, "old", aliased).kind).toBe("open");
  });

  it("does nothing for a side with no row, or a metadata row", () => {
    const added = TREE[0]!.children.find((c) => c.name === "added.weight")!;
    // `added` exists only on the new side, so a click in the old pane has no tensor.
    expect(clickOutcome(added, "old", sides(true, true))).toEqual({
      kind: "none",
    });
  });

  // A folded family is several tensors on one row, so `{0-1,3}.mlp.weight` is not the name of one and
  // there is nothing to open. Untick Collapse families and every member gets its own row.
  it("does nothing for a folded family of leaves, even on the served side", () => {
    const family = { ...TREE[1]!, name: "{0-1,3}.mlp.weight", members: 3 };
    expect(clickOutcome(family, "old", sides(true, true))).toEqual({
      kind: "none",
    });
    // The same row unfolded is an ordinary tensor again.
    expect(
      clickOutcome({ ...family, members: 1 }, "old", sides(true, true)).kind,
    ).toBe("open");
  });

  // With no comparison loaded there is nothing that says either side is served, so a click cannot be
  // an open — the safe answer, not a guess.
  it("does nothing when no comparison is loaded", () => {
    const leaf = TREE[1]!;
    expect(clickOutcome(leaf, "old", null)).toEqual({ kind: "none" });
  });
});

describe("how a row reads", () => {
  it("uses the same markers as the terminal and `diff`", () => {
    expect(statusMark("only_new")).toBe("+");
    expect(statusMark("only_old")).toBe("-");
    expect(statusMark("changed")).toBe("~");
    expect(statusMark("same")).toBe(" ");
  });

  it("renders each side as its signature, and a missing side as nothing", () => {
    expect(
      sideText({ kind: "tensor", info: info("BF16", [6, 4]), fold: null }),
    ).toBe("BF16 (6, 4)");
    // A folded row says what it stands for, so a leading expert dimension on the other side reads as
    // the fold it is rather than as an unexplained change of rank.
    expect(
      sideText({ kind: "tensor", info: info("U8", [4, 2]), fold: 256 }),
    ).toBe("U8 (4, 2)  ×256");
    expect(sideText({ kind: "metadata", name: "format", value: "pt" })).toBe(
      "pt",
    );
    expect(
      sideText({ kind: "group", tensor_count: 1234, params: 0, total_size: 0 }),
    ).toBe("1,234 tensors");
    expect(
      sideText({ kind: "group", tensor_count: 1, params: 0, total_size: 0 }),
    ).toBe("1 tensor");
    expect(sideText(null)).toBe("");
  });

  // `🔧 Metadata` holds metadata entries, not tensors. "🔧 Metadata 0 tensors" is a true sentence
  // that reads as a broken one.
  it('says nothing rather than "0 tensors" for a group that holds none', () => {
    expect(
      sideText({ kind: "group", tensor_count: 0, params: 0, total_size: 0 }),
    ).toBe("");
  });
});

describe("the headline tally", () => {
  const tally = (
    same: number,
    changed: number,
    only_old: number,
    only_new: number,
    meta = 0,
  ) => ({
    tensors: { same, changed, only_old, only_new },
    metadata: { same: 0, changed: meta, only_old: 0, only_new: 0 },
  });

  it("counts everything that is not a match", () => {
    expect(differingCount(tally(10, 2, 3, 4))).toBe(9);
    expect(differingCount(tally(10, 0, 0, 0))).toBe(0);
  });

  // The same words, in the same order, as the one-page report — two views of one comparison.
  // Word for word and comma for comma what `compare::verdict` prints, so the side-by-side and the
  // one-page report read as one comparison rather than two descriptions of it.
  it("reads exactly as the report does", () => {
    expect(tallyText(tally(0, 4, 1, 31247))).toBe(
      "0 unchanged; 31,247 added, 1 removed, 4 changed",
    );
    // The prefix carries the real figure, and counts tensors — as `compare::verdict` does.
    expect(tallyText(tally(1200, 0, 0, 3))).toBe("1,200 unchanged; 3 added");
  });

  // The residue that made the two views disagree in *wording* while agreeing on the total: metadata
  // folded into "removed", so this said `3 removed` where the report said `1 removed, 2 metadata
  // changes`.
  it("names metadata changes rather than folding them into removed tensors", () => {
    expect(tallyText(tally(0, 4, 1, 31247, 2))).toBe(
      "0 unchanged; 31,247 added, 1 removed, 4 changed, 2 metadata changes",
    );
    expect(tallyText(tally(0, 0, 0, 0, 1))).toBe(
      "0 unchanged; 1 metadata change",
    );
    // And they still count towards the headline total.
    expect(differingCount(tally(0, 4, 1, 31247, 2))).toBe(31254);
  });

  // The matching case is a banner, not a fragment at the end of a count line, so the phrase lives in
  // exactly one place.
  it("says nothing when nothing differs", () => {
    expect(tallyText(tally(5, 0, 0, 0))).toBe("");
  });

  it("omits the sections that are empty", () => {
    expect(tallyText(tally(0, 0, 0, 7))).toBe("0 unchanged; 7 added");
    expect(tallyText(tally(0, 2, 0, 0))).toBe("0 unchanged; 2 changed");
  });

  // Two checkpoints with unrelated naming schemes: the fact worth leading with is that nothing
  // aligned, not that there are 117,664 differences.
  it("recognises a pair that shares nothing", () => {
    expect(isDisjoint(tally(0, 0, 116510, 1154))).toBe(true);
    // A shared `format` key is not a shared architecture: the judgement is about tensors.
    expect(
      isDisjoint({
        tensors: { same: 0, changed: 0, only_old: 116510, only_new: 1154 },
        metadata: { same: 2, changed: 0, only_old: 0, only_new: 0 },
      }),
    ).toBe(true);
    // One match, or one changed row, means the schemes do correspond.
    expect(isDisjoint(tally(1, 0, 116510, 1154))).toBe(false);
    expect(isDisjoint(tally(0, 1, 116510, 1154))).toBe(false);
    // Everything on one side only is a different situation: nothing to be disjoint *from*.
    expect(isDisjoint(tally(0, 0, 12, 0))).toBe(false);
  });
});

describe("what a comparison unfolds on arrival", () => {
  it("reveals every difference when there are few enough to read", () => {
    expect(initialExpansion(TREE, 3).size).toBeGreaterThan(0);
  });

  // The 117k-difference case: unfolding every ancestor of every difference *is* unfolding
  // everything, and the result is a wall of one-sided rows with no way to fold it back.
  it("stays folded when there are too many to be worth revealing", () => {
    expect(initialExpansion(TREE, REVEAL_LIMIT + 1).size).toBe(0);
  });

  it("can name every group for an explicit expand-all", () => {
    expect(allGroupPaths(TREE)).toEqual(new Set(["model", "model.layers.0"]));
    expect(allGroupPaths([])).toEqual(new Set());
  });
});

describe("showing only what differs", () => {
  it("hides matching leaves and groups with nothing beneath them", () => {
    const open = new Set(["model", "model.layers.0", "quiet"]);
    const tree: AlignedNode[] = [
      ...TREE,
      group("quiet", [leaf("untouched", "same", "quiet.untouched")], "quiet"),
    ];
    const names = flattenDiff(tree, open, true).map((r) => r.node.name);
    expect(names).not.toContain("same.weight");
    expect(names).not.toContain("quiet");
    expect(names).not.toContain("untouched");
    // …and keeps the differences, and the groups on the way to them.
    expect(names).toContain("model");
    expect(names).toContain("layers.0");
    expect(names).toContain("k.weight");
    expect(names).toContain("added.weight");
    expect(names).toContain("gone.weight");
  });

  it("shows everything when the filter is off", () => {
    const open = new Set(["model", "model.layers.0"]);
    expect(flattenDiff(TREE, open, false).map((r) => r.node.name)).toContain(
      "same.weight",
    );
  });
});

describe("when the two checkpoints match", () => {
  const tally = (
    same: number,
    changed = 0,
    only_old = 0,
    only_new = 0,
    meta = 0,
  ) => ({
    tensors: { same, changed, only_old, only_new },
    metadata: { same: 0, changed: meta, only_old: 0, only_new: 0 },
  });

  it("states the verdict as a headline", () => {
    expect(identicalNote(tally(120))?.headline).toBe("Structurally identical");
  });

  /**
   * **A tally this build cannot read is not an identical one.**
   *
   * Reported from a real session: a tab left open across a server upgrade read the new split tally
   * (`{tensors, metadata}`) with the old flat shape, so every counter came back `undefined`. The sum
   * was `NaN`, `NaN > 0` is false, and the screen announced that two checkpoints sharing *no tensor
   * name at all* — 116,510 on one side, 933 on the other — were "structurally identical", above a
   * count line reading `NaN differences`.
   */
  it("refuses to call an unreadable tally identical", () => {
    const flat = { only_new: 933, only_old: 116510 } as unknown as Parameters<
      typeof identicalNote
    >[0];
    expect(tallyIsReadable(flat)).toBe(false);
    expect(identicalNote(flat)).toBeNull();
    // And the count it would have shown is a number, not NaN — nothing downstream renders `NaN`.
    expect(Number.isNaN(differingCount(flat))).toBe(false);
    // A counter that is a number but not a *finite* one is no better than a missing one.
    const poisoned = {
      tensors: { same: 1, changed: NaN, only_old: 0, only_new: 0 },
      metadata: { same: 0, changed: 0, only_old: 0, only_new: 0 },
    };
    expect(tallyIsReadable(poisoned)).toBe(false);
    expect(identicalNote(poisoned)).toBeNull();
    expect(differingCount(poisoned)).toBe(0);
    // A well-formed tally is still readable, and still identical when it says so.
    expect(tallyIsReadable(tally(120))).toBe(true);
    expect(identicalNote(tally(120))).not.toBeNull();
  });

  // The phrase is easy to over-read. This comparison never looks at a tensor's bytes, so two
  // differently-trained checkpoints of one architecture are "structurally identical" — and a reader who
  // takes that at face value concludes they are the same file.
  it("says what it compared, and what it did not", () => {
    const detail = identicalNote(tally(120))?.detail ?? "";
    expect(detail).toMatch(/name/);
    expect(detail).toMatch(/dtype/);
    expect(detail).toMatch(/shape/);
    expect(detail).toMatch(/metadata/);
    expect(detail).toMatch(/not compared/);
    expect(detail).toContain("--values");
  });

  it("has nothing to say when something differs", () => {
    expect(identicalNote(tally(120, 1))).toBeNull();
    expect(identicalNote(tally(0, 0, 3, 0))).toBeNull();
    // Including when the only difference is a metadata entry.
    expect(identicalNote(tally(120, 0, 0, 0, 1))).toBeNull();
  });
});

describe("an empty row list", () => {
  const tally = (
    same: number,
    changed = 0,
    only_old = 0,
    only_new = 0,
    meta = 0,
  ) => ({
    tensors: { same, changed, only_old, only_new },
    metadata: { same: 0, changed: meta, only_old: 0, only_new: 0 },
  });

  // "Differences only" over two matching checkpoints filters every row out, and a blank pane reads as
  // a failed load rather than as the answer.
  it("says so explicitly when there is nothing to differ", () => {
    expect(emptyRowsNote(0, tally(120))).toBe("No differences.");
  });

  it("points at the filter when differences exist but none are shown", () => {
    expect(emptyRowsNote(0, tally(0, 2))).toMatch(/Differences only/);
  });

  it("says nothing at all when there are rows", () => {
    expect(emptyRowsNote(12, tally(120))).toBe("");
    expect(emptyRowsNote(12, tally(0, 2))).toBe("");
  });
});

describe("finding rows by name", () => {
  // The tree had difference stepping and folding and no way to ask "where is qkv_proj" — which is a
  // different question from the scope: the scope decides what the server *compared*.
  it("shows the matches, the path down to them, and what is under them", () => {
    const names = flattenDiff(TREE, new Set(), false, "added").map(
      (r) => r.node.name,
    );
    expect(names).toContain("added.weight");
    expect(names).toContain("model");
    expect(names).not.toContain("x");
  });

  it("takes a group's whole subtree when the group itself matches", () => {
    const names = flattenDiff(TREE, new Set(), false, "model").map(
      (r) => r.node.name,
    );
    expect(names).toContain("model");
    expect(names).toContain("added.weight");
  });

  it("is case-insensitive, and an empty needle is not a search", () => {
    expect(
      flattenDiff(TREE, new Set(), false, "ADDED").map((r) => r.node.name),
    ).toContain("added.weight");
    expect(flattenDiff(TREE, new Set(), false, "   ")).toEqual(
      flattenDiff(TREE, new Set()),
    );
  });

  it("says nothing matched by showing nothing", () => {
    expect(flattenDiff(TREE, new Set(), false, "no-such-tensor")).toEqual([]);
  });
});
