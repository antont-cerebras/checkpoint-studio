import { describe, it, expect } from 'vitest';
import { highlightJson } from './jsonhl';

describe('highlightJson', () => {
  it('paints keys, strings and numbers as distinct roles', () => {
    const tokens = highlightJson('{"a":1,"b":"x","c":true,"d":null}');
    expect(tokens).not.toBeNull();
    const roles = (cls: string) =>
      tokens!.filter((t) => t[1] === cls).map((t) => t[0]);
    expect(roles('k')).toEqual(['"a"', '"b"', '"c"', '"d"']);
    expect(roles('s')).toEqual(['"x"']);
    expect(roles('n')).toEqual(['1']);
    expect(roles('b')).toEqual(['true', 'null']);
    expect(roles('p')).toHaveLength(4);
  });

  it('keeps an escaped quote inside its string', () => {
    // Stopping at the first `"` after the escape would split this into two tokens and
    // shift every colour after it on the line.
    const tokens = highlightJson('{"k":"a\\"b"}');
    expect(tokens!.filter((t) => t[1] === 's').map((t) => t[0])).toEqual(['"a\\"b"']);
  });

  it('round-trips the text verbatim, so nothing is dropped or duplicated', () => {
    const raw = '{"weight_map":{"m.0.w":"model-00001-of-00002.safetensors"},"n":-1.5e3}';
    const tokens = highlightJson(raw)!;
    expect(tokens.map((t) => t[0]).join('')).toBe(
      JSON.stringify(JSON.parse(raw), null, 2),
    );
  });

  it('declines anything that is not a JSON object or array', () => {
    expect(highlightJson('not json')).toBeNull();
    expect(highlightJson('"a bare string"')).toBeNull();
    expect(highlightJson('42')).toBeNull();
    expect(highlightJson('null')).toBeNull();
  });

  it('handles an index-sized document', () => {
    const map: Record<string, string> = {};
    for (let i = 0; i < 20000; i++) map[`model.layers.${i}.mlp.w`] = 'model-1.safetensors';
    const tokens = highlightJson(JSON.stringify({ weight_map: map }))!;
    expect(tokens.filter((t) => t[1] === 'k')).toHaveLength(20001);
  });
});
