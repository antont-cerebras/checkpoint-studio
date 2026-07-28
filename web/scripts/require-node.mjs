// Fail fast, with a useful message, when the running Node is too old.
//
// `engine-strict=true` in .npmrc already blocks `npm install`/`npm ci`, but that only
// covers installing. This guard covers the case that actually costs time: an existing
// `node_modules` plus an old Node on PATH. `web/dist` is COMMITTED and embedded into
// the Rust binary, and CI re-runs the build and diffs the result — so building on the
// wrong toolchain produces a confusing "dist is stale" failure rather than an obvious
// one. Better to refuse up front and say why.

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const webDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const wanted = Number(readFileSync(join(webDir, '.nvmrc'), 'utf8').trim().replace(/^v/, ''));
const actual = Number(process.versions.node.split('.')[0]);

if (Number.isNaN(wanted)) {
  console.error('require-node: could not parse web/.nvmrc');
  process.exit(1);
}

if (actual < wanted) {
  console.error(`
This project needs Node ${wanted}+ — found ${process.version}.

  web/dist is committed and embedded in the binary, and CI rebuilds it on Node
  ${wanted} (from web/.nvmrc) and fails if the output differs. ESLint also refuses to
  run on Node 16.

Pick one:
  nvm use                       # reads web/.nvmrc
  fnm use                       # reads web/.node-version
  export PATH="$HOME/.local/node20/bin:$PATH"
`);
  process.exit(1);
}
