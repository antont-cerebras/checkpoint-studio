import { defineConfig } from 'vitest/config';

// Unit tests for the pure TypeScript modules (`src/lib/*`): the fuzzy matcher, the
// number/size formatters and the tree flattener. Those hold real logic that until now
// had no automated coverage at all, and they're the parts that must agree with the Rust
// side — so they're worth pinning down precisely.
//
// Component (.svelte) tests are deliberately out of scope for now: they'd need
// jsdom + @testing-library/svelte and would mostly re-test what svelte-check and the
// browser review already cover.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
    coverage: {
      provider: 'v8',
      include: ['src/lib/**/*.ts', 'src/stores/**/*.ts'],
      reporter: ['text', 'json-summary'],
    },
  },
});
