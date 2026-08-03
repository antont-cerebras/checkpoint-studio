// Generated from `src/web/params.rs` by `UPDATE_PARITY=1 cargo test the_client_parameter_table`.
// Do not edit: rename the row in Rust and regenerate, so the wire, the URL, the CLI
// rendering and this client agree by construction.

export const SCOPE_PARAMS = [
  { field: 'name', key: 'name', kind: 'text' },
  { field: 'names', key: 'names', kind: 'text' },
  { field: 'dtypeIs', key: 'dtype_is', kind: 'text' },
  { field: 'shapeIs', key: 'shape_is', kind: 'text' },
  { field: 'map', key: 'map', kind: 'text' },
  { field: 'onlyTensors', key: 'only_tensors', kind: 'switch' },
  { field: 'alignFused', key: 'align_fused', kind: 'switch' },
  { field: 'subtree', key: 'subtree', kind: 'text' },
  { field: 'subtreeNew', key: 'subtree_new', kind: 'text' },
  { field: 'repackSchema', key: 'repack_schema', kind: 'text' },
  { field: 'repackSchemaNew', key: 'repack_schema_new', kind: 'text' },
] as const;

export const CHECK_PARAMS = [
  { field: 'values', key: 'values', kind: 'switch' },
  { field: 'histogram', key: 'histogram', kind: 'switch' },
  { field: 'bins', key: 'bins', kind: 'text' },
  { field: 'verifyRepack', key: 'verify_repack', kind: 'switch' },
  { field: 'repackBits', key: 'repack_bits', kind: 'text' },
  { field: 'tensor', key: 'tensor', kind: 'text' },
  { field: 'jobs', key: 'jobs', kind: 'text' },
  { field: 'full', key: 'full', kind: 'switch' },
] as const;

