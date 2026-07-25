import { defineConfig } from 'vitest/config';

// Unit tests for the pure TypeScript modules under `src/lib/`: the fuzzy matcher, the
// formatters, the tree flattener, the row/cursor rules, the URL ↔ view-state hash and
// the fetch layer. That's where the real logic lives, and several of those must agree
// with the Rust side — so they're worth pinning down precisely.
//
// `environment: 'node'` on purpose. Everything testable was moved OUT of the DOM's way
// (`stores/view.ts` kept only the store wiring and the `location`/`history` plumbing;
// its logic is in `lib/hash.ts` + `lib/rows.ts`), so nothing here needs jsdom. The few
// browser globals the leaf modules touch — `fetch`, `document`, `navigator` — are
// stubbed per test with `vi.stubGlobal`, which is both faster and more explicit than a
// full DOM.
//
// What that leaves uncovered is `src/stores/*` and the components: store subscriptions
// and Svelte reactivity, whose failures are visual and are caught by svelte-check plus
// the browser review pass. They stay in the coverage `include` rather than being
// excluded, so the gap keeps showing up in the report instead of being hidden by it.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
    coverage: {
      provider: 'v8',
      include: ['src/lib/**/*.ts', 'src/stores/**/*.ts'],
      // Type-only module: it compiles to nothing, so v8 reports it as 0% forever.
      exclude: ['src/lib/types.ts'],
      reporter: ['text', 'json-summary'],
    },
  },
});
