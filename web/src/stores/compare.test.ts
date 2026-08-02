// The comparison store's cache and supersede logic.
//
// This file exists because the guard that stops a revisit re-reading both checkpoints is exactly the
// kind of stateful shortcut that goes wrong quietly: it can suppress a *needed* refetch, or let a
// superseded response land on top of a newer one. Both were found by review rather than by use, which
// is the argument for pinning them here.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { get } from "svelte/store";

/** A comparison response the store can accept, with `n` differences. */
const tree = (left: string, right: string, n = 1) => ({
  base: {
    spec: left,
    root: left,
    tensor_count: 1,
    served: false,
    params: 1,
    bytes: 4,
  },
  current: {
    spec: right,
    root: right,
    tensor_count: 1,
    served: true,
    params: 1,
    bytes: 4,
  },
  tally: {
    tensors: { same: 0, changed: n, only_old: 2, only_new: 3 },
    metadata: { same: 1, changed: 0, only_old: 4, only_new: 5 },
  },
  matched: null,
  totals_labels: { size: "size", params: "params" },
  differences: Array.from({ length: n }, (_, i) => `d${i}`),
  full: false,
  rows: [],
});

/**
 * What `POST /api/compare` answers: an identity, and the two specs as resolved.
 *
 * The store quotes the id on the follow-up request and checks the tree it gets back describes these
 * two specs — so a stub that omitted them would (correctly) be rejected as someone else's comparison.
 */
let nextId = 1;
const comparisonSet = (url: string) => {
  const q = new URLSearchParams(url.slice(url.indexOf("?") + 1));
  const left = q.get("left") ?? "";
  const right = q.get("right") ?? "";
  return { id: nextId++, left, right, recents: [] };
};

/** Stub `fetch`, recording every URL and optionally delaying `/api/difftree`.
 *
 * Assertions count `"/api/compare?left="` rather than `"/api/compare"`: Stop issues a `DELETE` to
 * the same path, and counting that as a read made a passing test out of a wrong number. */
function stubFetch(body: (url: string) => unknown, delayDiff = 0) {
  const calls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => {
      calls.push(url);
      const payload = body(url);
      const text = JSON.stringify(payload);
      const bytes = new TextEncoder().encode(text);
      if (delayDiff && url.includes("/api/difftree")) {
        await new Promise((r) => setTimeout(r, delayDiff));
      }
      return {
        ok: true,
        status: 200,
        headers: new Headers({ "Content-Length": String(bytes.length) }),
        json: () => Promise.resolve(payload),
        text: () => Promise.resolve(text),
        body: {
          getReader() {
            let sent = false;
            return {
              read: () =>
                Promise.resolve(
                  sent
                    ? { done: true, value: undefined }
                    : ((sent = true), { done: false, value: bytes }),
                ),
            };
          },
        },
      };
    }),
  );
  return calls;
}

async function load() {
  vi.resetModules();
  return { c: await import("./compare"), s: await import("./server") };
}

/** Stub `fetch` so every request fails with the server's `{error}` envelope (plus any extra fields). */
function stubFailure(
  status: number,
  message: string,
  extra: Record<string, unknown> = {},
) {
  const payload = { error: message, ...extra };
  vi.stubGlobal(
    "fetch",
    vi.fn(() =>
      Promise.resolve({
        ok: false,
        status,
        headers: new Headers(),
        json: () => Promise.resolve(payload),
        text: () => Promise.resolve(JSON.stringify(payload)),
        body: null,
      }),
    ),
  );
}

/** The 409 a busy server sends: what it is reading, and that stopping it is on offer. */
function stubBusy(spec: string, seconds: number) {
  stubFailure(409, `${spec} is being read (${seconds}s so far)`, {
    busy_with: spec,
    busy_for_seconds: seconds,
    can_stop_other: true,
  });
}

beforeEach(() => vi.resetModules());
afterEach(() => vi.unstubAllGlobals());

describe("not re-reading a pair that is already loaded", () => {
  it("skips the second request for the same pair", async () => {
    const calls = stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    const { c } = await load();
    await c.compareAgainst({ left: "/a", right: "/b" });
    await c.compareAgainst({ left: "/a", right: "/b" });
    expect(calls.filter((u) => u.includes("/api/compare?left="))).toHaveLength(
      1,
    );
  });

  // What made the Compare button dead after Stop, and what a checkpoint rewritten on disk needs.
  it("re-reads when forced, even for the same pair", async () => {
    const calls = stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    const { c } = await load();
    await c.compareAgainst({ left: "/a", right: "/b" });
    await c.compareAgainst({ left: "/a", right: "/b", force: true });
    expect(calls.filter((u) => u.includes("/api/compare?left="))).toHaveLength(
      2,
    );
  });

  it("re-reads a different pair", async () => {
    const calls = stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/c") : comparisonSet(u),
    );
    const { c } = await load();
    await c.compareAgainst({ left: "/a", right: "/b" });
    await c.compareAgainst({ left: "/a", right: "/c" });
    expect(calls.filter((u) => u.includes("/api/compare?left="))).toHaveLength(
      2,
    );
  });

  // An empty `right` *means* "whatever is served", so the same URL denotes a different comparison
  // once the served checkpoint changes. Keying on the URL alone kept a comparison against a
  // checkpoint no longer loaded, with no interaction able to refresh it.
  it("re-reads when the served checkpoint changed under an implicit right side", async () => {
    const calls = stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/served") : comparisonSet(u),
    );
    const { c, s } = await load();
    s.tree.set({ spec: "/served-1", tree: [] } as never);
    await c.compareAgainst({ left: "/a", right: "" });
    s.tree.set({ spec: "/served-2", tree: [] } as never);
    await c.compareAgainst({ left: "/a", right: "" });
    expect(calls.filter((u) => u.includes("/api/compare?left="))).toHaveLength(
      2,
    );
  });

  it("forgets the pair on Stop, so comparing it again re-reads", async () => {
    const calls = stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    const { c } = await load();
    await c.compareAgainst({ left: "/a", right: "/b" });
    await c.stopComparing();
    expect(get(c.diffTree)).toBeNull();
    await c.compareAgainst({ left: "/a", right: "/b" });
    expect(calls.filter((u) => u.includes("/api/compare?left="))).toHaveLength(
      2,
    );
  });
});

describe("a superseded comparison", () => {
  // The failure this prevents: a slow pair landing after you have navigated back, leaving the view
  // describing a pair the URL no longer names and nothing left to re-fire.
  it("does not publish over a newer one", async () => {
    const calls = stubFetch(
      (u) =>
        u.includes("difftree")
          ? tree(u.includes("slow") ? "/slow" : "/fast", "/b")
          : comparisonSet(u),
      30,
    );
    const { c } = await load();
    const slow = c.compareAgainst({ left: "/slow", right: "/b" });
    // Start a second comparison before the first resolves; it supersedes.
    await c.compareAgainst({ left: "/fast", right: "/b", force: true });
    await slow;
    expect(get(c.diffTree)?.base.spec).toBe("/fast");
    expect(calls.filter((u) => u.includes("/api/compare?left="))).toHaveLength(
      2,
    );
  });

  it("does not clear a newer comparison’s load step", async () => {
    stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    const { c } = await load();
    await c.compareAgainst({ left: "/a", right: "/b" });
    expect(get(c.diffStep)).toBeNull();
  });

  // Cancelling keeps both paths so one character can be fixed and the read retried, and it must not
  // leave the aborted fetch's rejection on screen as if the address were wrong.
  it("cancelling stops the read without reporting an error", async () => {
    stubFetch(
      (u) => (u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u)),
      30,
    );
    const { c } = await load();
    const pending = c.compareAgainst({ left: "/a", right: "/b" });
    c.cancelComparison();
    expect(get(c.diffStep)).toBeNull();
    await pending;
    expect(get(c.diffError)).toBe("");
    expect(get(c.diffTree)).toBeNull();
  });
});

describe("being handed the wrong comparison", () => {
  // The server has one comparison slot. Before it required an id, two overlapping clients received
  // each other's trees with a 200: A set up its pair, B replaced it, A's GET returned B's. The id makes
  // that a 409; this assertion is the backstop for anything that still gets through — a cached body, an
  // id reused across a restart.
  it("refuses a tree that describes a pair it did not ask for", async () => {
    stubFetch((u) =>
      u.includes("difftree")
        ? tree("/someone-elses", "/other-side", 34)
        : comparisonSet(u),
    );
    const { c } = await load();
    await c.compareAgainst({ left: "/a", right: "/b" });
    expect(get(c.diffTree)).toBeNull();
    expect(get(c.diffError)).toMatch(/different comparison/);
    // And it names both what arrived and what was asked for, so the report is actionable.
    expect(get(c.diffError)).toContain("/someone-elses");
    expect(get(c.diffError)).toContain("/a");
  });

  it("quotes the comparison id it was given", async () => {
    const calls = stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    const { c } = await load();
    await c.compareAgainst({ left: "/a", right: "/b" });
    const asked = calls.find((u) => u.includes("/api/difftree"));
    expect(asked).toMatch(/[?&]id=\d+/);
  });
});

describe("when a read is refused", () => {
  // The server reads one checkpoint at a time and refuses the rest. That is a "try again in a
  // moment", not a bad address, and saying which it is decides whether the reader re-checks the path
  // they typed or simply waits.
  // Recorded as a fact, not as prose. The view needs the running spec to *offer to stop it*; on a server
  // that reads one checkpoint at a time, a sentence saying to wait was advice with nothing behind it.
  it("records what the server is busy with, rather than an error to read", async () => {
    stubBusy("s3://bucket/other-checkpoint", 4);
    const { c } = await load();
    await c.compareAgainst({ left: "/a", right: "/b" });
    expect(get(c.diffTree)).toBeNull();
    expect(get(c.diffBusy)).toEqual({
      spec: "s3://bucket/other-checkpoint",
      seconds: 4,
    });
    // Not duplicated into the error slot, which would render the same thing twice.
    expect(get(c.diffError)).toBe("");
  });

  /**
   * **Asking for a comparison takes the read slot.**
   *
   * It used to ask politely and report the refusal, which produced a question between the reader and
   * the thing they had just asked for — "the server is reading …; stop it and compare these?" — whose
   * only sensible answer was yes. The server still refuses unless asked (`stop_other=1` is what asks),
   * so the choice exists; this is a client that has already made it.
   */
  it("asks the server to stop whatever else it is reading", async () => {
    const calls = stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    const { c } = await load();
    await c.compareAgainst({ left: "/a", right: "/b" });
    expect(calls.some((u) => u.includes("stop_other=1"))).toBe(true);

    // …and a caller can still decline to, which is what the polite path was.
    const later = calls.length;
    await c.compareAgainst({
      left: "/a",
      right: "/c",
      force: true,
      stopOther: false,
    });
    expect(calls.slice(later).some((u) => u.includes("stop_other=1"))).toBe(
      false,
    );
  });

  /**
   * **The same comparison, asked for twice, is asked for once.**
   *
   * A reactive statement re-running or a scope applied mid-read used to fire a second request, which
   * found the server busy with the *first* — so the refusal offered to stop the very read that was
   * fetching what was being asked for, and accepting it threw the work away and started again.
   */
  it("does not ask twice for a comparison already being read", async () => {
    const calls = stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    const { c } = await load();
    const first = c.compareAgainst({ left: "/a", right: "/b" });
    const second = c.compareAgainst({ left: "/a", right: "/b" });
    await Promise.all([first, second]);
    expect(calls.filter((u) => u.includes("/api/compare")).length).toBe(1);
  });

  it("clears a previous refusal when a new attempt starts", async () => {
    stubBusy("/busy", 2);
    const { c } = await load();
    await c.compareAgainst({ left: "/a", right: "/b" });
    expect(get(c.diffBusy)).not.toBeNull();

    stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    await c.compareAgainst({
      left: "/a",
      right: "/b",
      force: true,
      stopOther: true,
    });
    expect(get(c.diffBusy)).toBeNull();
    expect(get(c.diffTree)).not.toBeNull();
  });

  it("passes an ordinary failure through as the server worded it", async () => {
    stubFailure(400, "opening /nope: no checkpoint files found at /nope");
    const { c } = await load();
    await c.compareAgainst({ left: "/nope", right: "" });
    expect(get(c.diffError)).toBe(
      "opening /nope: no checkpoint files found at /nope",
    );
    expect(get(c.diffError)).not.toContain("Nothing here is wrong");
  });

  // Clearing is local: what the server does with its copy cannot leave this tab showing a comparison
  // it has already discarded.
  it("tears the comparison down even when the server refuses to let go", async () => {
    stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    const { c } = await load();
    await c.compareAgainst({ left: "/a", right: "/b" });
    expect(get(c.diffTree)).not.toBeNull();

    stubFailure(500, "the server is unwell");
    await c.stopComparing();
    expect(get(c.diffTree)).toBeNull();
    expect(get(c.diffExpanded).size).toBe(0);
    expect(get(c.diffCursor)).toBeNull();
  });
});

describe("orientation", () => {
  // Flipping is view state now — `#compare?…&swap=1`, applied at the point of drawing
  // (`difftree::swapResponse`). The store's part of that contract is simply that it does not treat
  // the flipped pair as a different comparison: the same pair is fetched once, whichever way round
  // the screen is reading it.
  it("does not refetch a pair that is already loaded, whichever way it is read", async () => {
    const calls = stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    const { c } = await load();
    await c.compareAgainst({ left: "/a", right: "/b" });
    await c.compareAgainst({ left: "/a", right: "/b" });
    expect(calls.filter((u) => u.includes("/api/compare?left="))).toHaveLength(
      1,
    );
  });

  // The reverse *pair* is a different comparison — the scope is directional, and the server applies
  // `--map` to whichever side it is told is the baseline — so asking for it really does read again.
  // That is why the screen no longer rewrites the operands to flip.
  it("treats the reversed pair as a comparison of its own", async () => {
    const calls = stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    const { c } = await load();
    await c.compareAgainst({ left: "/a", right: "/b" });
    await c.compareAgainst({ left: "/b", right: "/a" });
    expect(calls.filter((u) => u.includes("/api/compare?left="))).toHaveLength(
      2,
    );
  });

  // The fold state is part of the tree's key but not the pair's: folding the families re-aligns what
  // the server already holds. Re-reading two checkpoints — seconds each, minutes over an ssh proxy —
  // to draw the same pair a second way is the cost the comparison slot exists to avoid.
  it("re-aligns when the family fold changes, without re-reading the pair", async () => {
    const calls = stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    const { c } = await load();
    await c.compareAgainst({ left: "/a", right: "/b" });
    await c.compareAgainst({ left: "/a", right: "/b", full: true });
    expect(calls.filter((u) => u.includes("/api/compare?left="))).toHaveLength(
      1,
    );
    expect(calls.filter((u) => u.includes("/api/difftree"))).toHaveLength(2);
  });

  // Switching to the aligned tree reads nothing: the pair is in the server's slot, and what is being
  // waited for is the comparison coming over the wire. It used to raise the *reading both checkpoints*
  // step regardless — so Browse opened naming two checkpoints as `reading…`, for as long as the server
  // took to align and send them, with nothing on screen that was true.
  it("announces a read only when there is one to announce", async () => {
    const calls = stubFetch(
      (u) => (u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u)),
      20,
    );
    const { c } = await load();
    const steps: (string | null)[] = [];
    const stop = c.diffStep.subscribe((s) => steps.push(s?.kind ?? null));
    await c.compareAgainst({ left: "/a", right: "/b" });
    expect(steps).toContain("comparing");
    steps.length = 0;
    // The same pair again, from another view of it.
    await c.compareAgainst({ left: "/a", right: "/b", full: true });
    expect(steps).not.toContain("comparing");
    expect(steps).toContain("difftree");
    stop();
    expect(calls.filter((u) => u.includes("/api/compare?left="))).toHaveLength(1);
  });

  // `force` is how the Compare button re-runs a pair that is already set up — the only way to pick up
  // a checkpoint rewritten on disk.
  it("re-reads the pair when forced", async () => {
    const calls = stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    const { c } = await load();
    await c.establishComparison({ left: "/a", right: "/b" });
    expect(get(c.diffStep)).toBeNull();
    await c.establishComparison({ left: "/a", right: "/b", force: true });
    expect(calls.filter((u) => u.includes("/api/compare?left="))).toHaveLength(
      2,
    );
  });

  // An empty baseline is not a comparison: nothing to set up, nothing to ask the server.
  it("asks nothing for an empty baseline", async () => {
    const calls = stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    const { c } = await load();
    await c.establishComparison({ left: "", right: "/b" });
    expect(calls).toHaveLength(0);
    expect(get(c.comparison)).toBeNull();
  });

  // Two views mounting at once must not each read the pair — the second waits for the first.
  it("does not set the same pair up twice at once", async () => {
    const calls = stubFetch(
      (u) => (u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u)),
      5,
    );
    const { c } = await load();
    const both = [
      c.establishComparison({ left: "/a", right: "/b" }),
      c.establishComparison({ left: "/a", right: "/b" }),
    ];
    await Promise.all(both);
    expect(get(c.comparison)).not.toBeNull();
    expect(get(c.diffStep)).toBeNull();
    expect(calls.filter((u) => u.includes("/api/compare?left="))).toHaveLength(
      1,
    );
  });

  // A pair that cannot be set up leaves nothing claiming to be one: the views read `comparison`, and
  // a stale id there would have them asking the server about a comparison it does not have.
  it("holds no comparison when the pair cannot be read", async () => {
    stubFailure(400, "no checkpoint files found at /nope");
    const { c } = await load();
    await c.establishComparison({ left: "/nope", right: "/b" });
    expect(get(c.comparison)).toBeNull();
    expect(get(c.diffError)).toContain("no checkpoint files found");
  });

  // The server reads one checkpoint at a time and says so; that is an offer to stop the other read,
  // not an error to print.
  it("reports a busy server as something to act on", async () => {
    stubFailure(409, "the server is reading /other", {
      can_stop_other: true,
      busy_with: "/other",
      busy_for_seconds: 3,
    });
    const { c } = await load();
    await c.establishComparison({ left: "/a", right: "/b" });
    expect(get(c.diffBusy)).toEqual({ spec: "/other", seconds: 3 });
    expect(get(c.comparison)).toBeNull();
  });

  // The same for a view that needs no tree at all: establishing the pair is what costs, and it
  // happens once however many views read it.
  it("sets the pair up once, however many views read it", async () => {
    const calls = stubFetch((u) =>
      u.includes("difftree") ? tree("/a", "/b") : comparisonSet(u),
    );
    const { c } = await load();
    await c.establishComparison({ left: "/a", right: "/b" });
    await c.compareAgainst({ left: "/a", right: "/b" });
    await c.establishComparison({ left: "/a", right: "/b" });
    expect(calls.filter((u) => u.includes("/api/compare?left="))).toHaveLength(
      1,
    );
  });
});
