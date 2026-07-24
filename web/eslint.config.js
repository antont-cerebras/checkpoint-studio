// ESLint (flat config) for the Svelte UI.
//
// `svelte-check` already covers types; this adds the rules a type checker won't give
// you. The reason it's TYPE-AWARE (`projectService: true`) rather than syntax-only is
// that the rules which pay for themselves here need the checker: this UI is mostly
// async fetches feeding stores, and an unhandled promise is how it ends up showing
// stale or missing data instead of an error (exactly the class of bug that produced
// the blank data pane and the sticky rejected-request cache).

import js from '@eslint/js';
import ts from 'typescript-eslint';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';

export default ts.config(
  // Build output, deps, and tooling that isn't part of the typed app project
  // (`scripts/` is plain Node ESM run by npm, not compiled by tsconfig).
  { ignores: ['dist/', 'node_modules/', 'scripts/', '*.config.js', '*.config.ts'] },
  js.configs.recommended,
  ...ts.configs.recommendedTypeChecked,
  ...svelte.configs['flat/recommended'],
  {
    languageOptions: {
      // `projectService` resolves each file to its tsconfig, which is what enables the
      // type-aware rules below. `extraFileExtensions` lets it see .svelte files.
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
        extraFileExtensions: ['.svelte'],
      },
      globals: { ...globals.browser },
    },
    rules: {
      // The high-value type-aware rules for this codebase.
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/no-misused-promises': 'error',

      // `catch (e)` is typed `unknown`; we narrow with `instanceof Error` everywhere,
      // which these rules are happy with — keep them on to enforce that habit.
      '@typescript-eslint/no-unsafe-assignment': 'error',
      '@typescript-eslint/no-unsafe-member-access': 'error',

      // The API returns JSON the server owns; a few call sites legitimately cast the
      // parsed shape. Those are explicit `as unknown as T` and reviewed, so allow the
      // assertion but never a silent `any`.
      '@typescript-eslint/no-explicit-any': 'error',

      // tsconfig already reports unused locals/params with better messages.
      '@typescript-eslint/no-unused-vars': 'off',

      // Disagrees with the actual Svelte compiler: it calls two of our
      // `svelte-ignore a11y-…` comments unused, but removing them makes `svelte-check`
      // (i.e. the compiler) re-raise the warning. The compiler is the authority on
      // which of its own warnings are suppressed, so don't let the plugin's separate
      // a11y implementation delete working suppressions.
      'svelte/no-unused-svelte-ignore': 'off',
    },
  },
  {
    // Svelte components: parse <script lang="ts"> with the TS parser.
    files: ['**/*.svelte', '**/*.svelte.ts'],
    languageOptions: {
      parserOptions: { parser: ts.parser },
    },
  },
);
