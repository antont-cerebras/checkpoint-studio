import { describe, it, expect } from 'vitest';
import type { FileNode } from './types';
import { flattenFiles, toggleDir } from './filerows';

const file = (name: string, path: string): FileNode =>
  ({ kind: 'file', name, path, size: 1, file_kind: 'Json' }) as unknown as FileNode;
const dir = (name: string, path: string, children: FileNode[]): FileNode =>
  ({ kind: 'dir', name, path, children, size: 1, files: children.length }) as unknown as FileNode;

// The real shape: the checkpoint folder is the root and its path is the empty string,
// because paths are served relative to it.
const tree = dir('ckpt', '', [
  file('config.json', 'config.json'),
  dir('sub', 'sub', [file('a.txt', 'sub/a.txt')]),
]);

describe('flattenFiles', () => {
  it('shows only the root when nothing is expanded', () => {
    expect(flattenFiles(tree, new Set()).map((r) => r.node.name)).toEqual(['ckpt']);
  });

  it("expands the root under its empty-string path", () => {
    // An empty-string key is a real Set member; the root must not be a special case.
    expect(flattenFiles(tree, new Set([''])).map((r) => r.node.name)).toEqual([
      'ckpt',
      'config.json',
      'sub',
    ]);
  });

  it('nests deeper folders and reports depth', () => {
    const rows = flattenFiles(tree, new Set(['', 'sub']));
    expect(rows.map((r) => [r.node.name, r.depth])).toEqual([
      ['ckpt', 0],
      ['config.json', 1],
      ['sub', 1],
      ['a.txt', 2],
    ]);
  });

  it('folds the checkpoint folder back up — the regression', () => {
    // Collapsing the root hides everything under it. This is what the browser could not
    // do: the triangle flipped, the rows did not.
    const collapsed = toggleDir(new Set(['']), '');
    expect(flattenFiles(tree, collapsed).map((r) => r.node.name)).toEqual(['ckpt']);
  });

  it('handles no tree at all', () => {
    expect(flattenFiles(null, new Set([''])).length).toBe(0);
  });
});

describe('toggleDir', () => {
  it('returns a new set, so assigning it is reactive', () => {
    const before = new Set(['']);
    const after = toggleDir(before, 'sub');
    expect(after).not.toBe(before);
    expect(before.has('sub')).toBe(false);
    expect(after.has('sub')).toBe(true);
  });

  it('round-trips', () => {
    expect(toggleDir(toggleDir(new Set(['']), ''), '')).toEqual(new Set(['']));
  });
});
