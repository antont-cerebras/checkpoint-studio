// @vitest-environment jsdom
//
// The only test file here that needs a DOM: DOMPurify sanitizes by parsing into one, so
// there is no way to exercise the security-relevant path without it. Everything else in
// `lib/` stays on the `node` environment (see vitest.config.ts).
import { describe, it, expect } from 'vitest';
import { renderMarkdown, highlightMarkdown, splitFrontmatter, isMarkdown } from './md';

describe('isMarkdown', () => {
  it('recognises the markdown extensions and nothing else', () => {
    for (const n of ['README.md', 'card.MARKDOWN', 'a.mkd', 'b.mdown']) {
      expect(isMarkdown(n), n).toBe(true);
    }
    for (const n of ['config.json', 'notes.txt', 'model.safetensors', 'md', 'a.md.bak']) {
      expect(isMarkdown(n), n).toBe(false);
    }
  });
});

describe('splitFrontmatter', () => {
  it('lifts the key/value block model cards open with', () => {
    const { frontmatter, body } = splitFrontmatter(
      '---\nlibrary_name: transformers\nlicense: apache-2.0\n---\n# Title\n',
    );
    expect(frontmatter).toEqual([
      ['library_name', 'transformers'],
      ['license', 'apache-2.0'],
    ]);
    expect(body).toBe('# Title\n');
  });

  it('folds list items into the key above them', () => {
    const { frontmatter } = splitFrontmatter('---\ntags:\n- code\n- moe\n---\nbody');
    expect(frontmatter).toEqual([['tags', 'code, moe']]);
  });

  it('strips quotes from values', () => {
    expect(splitFrontmatter('---\na: "b"\n---\n').frontmatter).toEqual([['a', 'b']]);
  });

  it('leaves a document that merely opens with a rule alone', () => {
    // `---` then prose is a horizontal rule, not frontmatter; swallowing it would delete
    // the top of the document.
    const src = '---\njust prose here\n---\nmore';
    expect(splitFrontmatter(src)).toEqual({ frontmatter: [], body: src });
  });

  it('handles a document with no frontmatter at all', () => {
    expect(splitFrontmatter('# Title\n')).toEqual({ frontmatter: [], body: '# Title\n' });
  });

  it('skips comments and blank lines inside the block', () => {
    const { frontmatter } = splitFrontmatter('---\n# a comment\n\nlicense: mit\n---\nx');
    expect(frontmatter).toEqual([['license', 'mit']]);
  });

  it('keeps a key whose value is empty', () => {
    expect(splitFrontmatter('---\nbase_model:\n---\nx').frontmatter).toEqual([
      ['base_model', ''],
    ]);
  });

  it('ignores a leading list item with no key above it', () => {
    // `- x` as the first line has nothing to fold into; it must not crash or invent a key.
    expect(splitFrontmatter('---\n- orphan\nlicense: mit\n---\nx').frontmatter).toEqual([
      ['license', 'mit'],
    ]);
  });

  it('folds list items into a key that started empty', () => {
    expect(splitFrontmatter('---\ntags:\n- a\n- b\n---\nx').frontmatter).toEqual([
      ['tags', 'a, b'],
    ]);
  });

  it('accepts a frontmatter block that ends at end of file', () => {
    expect(splitFrontmatter('---\nlicense: mit\n---').frontmatter).toEqual([['license', 'mit']]);
  });

  it('handles CRLF line endings', () => {
    expect(splitFrontmatter('---\r\nlicense: mit\r\n---\r\nx').frontmatter).toEqual([
      ['license', 'mit'],
    ]);
  });
});

describe('renderMarkdown', () => {
  it('renders real document structure', async () => {
    const { html } = await renderMarkdown(
      '# H1\n\n## H2\n\nsome **bold** text\n\n- a\n- b\n\n| x | y |\n|---|---|\n| 1 | 2 |\n',
    );
    expect(html).toContain('<h1');
    expect(html).toContain('<h2');
    expect(html).toContain('<strong>bold</strong>');
    expect(html).toContain('<ul>');
    expect(html).toContain('<table>');
    expect(html).toContain('<td>1</td>');
  });

  it('highlights a fenced block whose grammar ships', async () => {
    const { html } = await renderMarkdown('```python\nx = 1\n```\n');
    expect(html).toContain('shiki');
    expect(html).toContain('--shiki-dark');
    const colours = new Set(html.match(/--shiki-dark:#[0-9a-fA-F]+/g) ?? []);
    expect(colours.size, 'tokenized rather than one flat run').toBeGreaterThan(1);
  });

  it('still renders a fence whose grammar does not ship, as escaped text', async () => {
    const { html } = await renderMarkdown('```brainfuck\n+++<script>\n```\n');
    expect(html).toContain('<pre');
    expect(html).not.toContain('<script>');
    expect(html).toContain('&lt;script&gt;');
  });

  it('resolves language aliases', async () => {
    const { html } = await renderMarkdown('```py\nx = 1\n```\n');
    expect(html).toContain('shiki');
  });

  it('highlights every grammar it ships, under each alias', async () => {
    // One case per entry in LANGS: a grammar that is listed but never loaded would be
    // dead weight in the binary, and an alias that resolves to nothing fails silently.
    const cases: [string, string][] = [
      ['python', 'x = 1'],
      ['py', 'x = 1'],
      ['python3', 'x = 1'],
      ['bash', 'echo hi'],
      ['sh', 'echo hi'],
      ['shell', 'echo hi'],
      ['zsh', 'echo hi'],
      ['console', 'echo hi'],
      ['json', '{"a": 1}'],
      ['jsonc', '{"a": 1}'],
      ['yaml', 'a: 1'],
      ['yml', 'a: 1'],
      ['markdown', '# h'],
      ['md', '# h'],
    ];
    for (const [lang, code] of cases) {
      const { html } = await renderMarkdown(`\`\`\`${lang}\n${code}\n\`\`\`\n`);
      expect(html, lang).toContain('shiki');
      expect(html, lang).not.toContain('class="plain"');
    }
  });

  it('ignores the info string after the language name', async () => {
    // Cards write ```python title="x" and ```bash,copy — the first word is the language.
    for (const info of ['python title="run.py"', 'python,copy', 'python:run.py']) {
      const { html } = await renderMarkdown(`\`\`\`${info}\nx = 1\n\`\`\`\n`);
      expect(html, info).toContain('shiki');
    }
  });

  it('renders a fence with no language at all as plain', async () => {
    const { html } = await renderMarkdown('```\nsome output\n```\n');
    expect(html).toContain('class="plain"');
    expect(html).toContain('some output');
  });

  // The security-relevant behaviour. The preview injects this HTML with {@html}, and the
  // source is a file from a checkpoint someone else built.
  describe('sanitization', () => {
    it('strips script tags', async () => {
      const { html } = await renderMarkdown('<script>alert(1)</script>\n\ntext\n');
      expect(html).not.toContain('<script');
      expect(html).not.toContain('alert(1)');
    });

    it('strips event handlers from passed-through HTML', async () => {
      const { html } = await renderMarkdown('<img src="x" onerror="alert(1)">\n');
      expect(html).not.toContain('onerror');
      expect(html).toContain('<img');
    });

    it('drops javascript: links but keeps http ones', async () => {
      const bad = await renderMarkdown('[click](javascript:alert(1))\n');
      expect(bad.html).not.toContain('javascript:');
      const good = await renderMarkdown('[card](https://huggingface.co/x)\n');
      expect(good.html).toContain('https://huggingface.co/x');
    });

    it('opens links in a new tab without leaking window.opener', async () => {
      const { html } = await renderMarkdown('[a](https://x.test/y)\n');
      expect(html).toContain('target="_blank"');
      expect(html).toContain('rel="noopener noreferrer"');
    });

    it('strips style tags, which could restyle the whole app', async () => {
      const { html } = await renderMarkdown('<style>body{display:none}</style>\n\nhi\n');
      expect(html).not.toContain('<style');
    });

    it('keeps the inline style attributes Shiki needs', async () => {
      // These carry the theme CSS variables; stripping `style` outright would leave code
      // blocks uncoloured.
      const { html } = await renderMarkdown('```json\n{"a":1}\n```\n');
      expect(html).toContain('style=');
    });
  });

  it('separates frontmatter from the rendered body', async () => {
    const { frontmatter, html } = await renderMarkdown('---\nlicense: mit\n---\n# T\n');
    expect(frontmatter).toEqual([['license', 'mit']]);
    expect(html).toContain('<h1');
    expect(html).not.toContain('license');
  });

  it('handles an empty document', async () => {
    await expect(renderMarkdown('')).resolves.toEqual({ frontmatter: [], html: '' });
  });
});

describe('highlightMarkdown', () => {
  it('highlights source and escapes HTML in it', async () => {
    const html = await highlightMarkdown('# T\n<script>alert(1)</script>\n');
    expect(html).toContain('shiki');
    expect(html).toContain('--shiki-light');
    expect(html).not.toContain('<script>');
  });

  it('reuses one highlighter across calls', async () => {
    expect(await highlightMarkdown('# a\n')).toBe(await highlightMarkdown('# a\n'));
  });
});
