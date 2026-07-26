//! Parsing for the NumPy `.npy` array format (also the payload of each entry in
//! a `.npz` archive). A `.npy` file is a small header — a 6-byte magic, a 2-byte
//! version, a 2- or 4-byte header length, then an ASCII Python-dict describing
//! the array — followed by the raw elements in C (row-major) order, exactly the
//! little-endian layout the rest of the explorer already decodes.
//!
//! See <https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html>.

use std::io::Read;

/// The decoded header of a `.npy` stream.
#[derive(Debug)]
pub struct NpyHeader {
    /// The explorer dtype name (`F32`, `I16`, …) the descriptor maps to.
    pub dtype: String,
    /// Logical shape (row-major). Reversed for Fortran-order arrays so the raw
    /// bytes still read correctly as a contiguous row-major buffer.
    pub shape: Vec<usize>,
    /// Bytes from the start of the stream to the first data element.
    pub data_offset: usize,
}

/// Read and parse a `.npy` header from the start of `r`. Leaves `r` positioned
/// at the first data byte. Errors on a bad magic or an unsupported dtype.
pub fn parse_header(r: &mut impl Read) -> Result<NpyHeader, String> {
    let mut magic = [0u8; 8]; // 6-byte magic + 2-byte version
    r.read_exact(&mut magic)
        .map_err(|e| format!("reading .npy magic: {e}"))?;
    if &magic[..6] != b"\x93NUMPY" {
        return Err("not a .npy stream (bad magic)".to_string());
    }
    let major = magic[6];
    // v1 uses a 2-byte header length; v2+ widened it to 4 bytes.
    let (header_len, len_field) = if major >= 2 {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)
            .map_err(|e| format!("reading .npy header length: {e}"))?;
        (u32::from_le_bytes(b) as usize, 4)
    } else {
        let mut b = [0u8; 2];
        r.read_exact(&mut b)
            .map_err(|e| format!("reading .npy header length: {e}"))?;
        (u16::from_le_bytes(b) as usize, 2)
    };
    let mut buf = vec![0u8; header_len];
    r.read_exact(&mut buf)
        .map_err(|e| format!("reading .npy header: {e}"))?;
    let header = String::from_utf8_lossy(&buf);

    let descr = dict_string(&header, "descr")?;
    let dtype = map_descr(&descr)?;
    let mut shape = dict_shape(&header)?;
    if dict_bool(&header, "fortran_order")? {
        // Column-major bytes are the row-major bytes of the transposed shape;
        // reversing the dims lets the row-major readers serve correct values.
        shape.reverse();
    }
    Ok(NpyHeader {
        dtype,
        shape,
        data_offset: 8 + len_field + header_len,
    })
}

/// Map a NumPy dtype descriptor (array-interface `typestr`, e.g. `<f4`, `|u1`,
/// `=i8`) to the explorer's dtype name. Rejects big-endian multi-byte types
/// (the decoders assume little-endian) and non-numeric kinds.
pub fn map_descr(descr: &str) -> Result<String, String> {
    let (order, rest) = match descr.as_bytes().first() {
        Some(b'<' | b'=' | b'>' | b'|') => (descr.as_bytes()[0], &descr[1..]),
        _ => (b'=', descr),
    };
    let kind = rest.chars().next().ok_or("empty dtype descriptor")?;
    let size: usize = rest[kind.len_utf8()..]
        .parse()
        .map_err(|_| format!("unsupported dtype: {descr}"))?;
    if order == b'>' && size > 1 {
        return Err(format!("big-endian dtype not supported: {descr}"));
    }
    let name = match (kind, size) {
        ('f', 8) => "F64",
        ('f', 4) => "F32",
        ('f', 2) => "F16",
        ('i', 8) => "I64",
        ('i', 4) => "I32",
        ('i', 2) => "I16",
        ('i', 1) => "I8",
        ('u', 8) => "U64",
        ('u', 4) => "U32",
        ('u', 2) => "U16",
        ('u', 1) => "U8",
        ('b', 1) => "BOOL",
        _ => return Err(format!("unsupported dtype: {descr}")),
    };
    Ok(name.to_string())
}

/// The single-quoted string value of `'key'` in the header dict.
fn dict_string(header: &str, key: &str) -> Result<String, String> {
    let rest = after_key(header, key)?;
    let open = rest.find('\'').ok_or_else(|| missing(key))?;
    let tail = &rest[open + 1..];
    let close = tail.find('\'').ok_or_else(|| missing(key))?;
    Ok(tail[..close].to_string())
}

/// The `True`/`False` value of `'key'` in the header dict.
fn dict_bool(header: &str, key: &str) -> Result<bool, String> {
    let rest = after_key(header, key)?;
    let t = rest.find("True");
    let f = rest.find("False");
    match (t, f) {
        (Some(ti), f) if f.is_none_or(|fi| ti < fi) => Ok(true),
        (_, Some(_)) => Ok(false),
        _ => Err(format!("malformed '{key}' in .npy header")),
    }
}

/// The `'shape'` tuple, e.g. `(4, 5)` → `[4, 5]`, `(5,)` → `[5]`, `()` → `[]`.
fn dict_shape(header: &str) -> Result<Vec<usize>, String> {
    let rest = after_key(header, "shape")?;
    let open = rest.find('(').ok_or_else(|| missing("shape"))?;
    let close = rest[open..].find(')').ok_or_else(|| missing("shape"))? + open;
    rest[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| format!("bad dimension in .npy shape: {s}"))
        })
        .collect()
}

/// The slice of the header following `'key':`.
///
/// The colon has to come straight after the key. Taking the *next* colon anywhere in the
/// header instead walks into the following entry, and then the value read belongs to that
/// one — a header missing a colon after `'descr'` reported `unsupported dtype: shape`,
/// naming a key from further along, rather than saying the header was malformed.
fn after_key(header: &str, key: &str) -> Result<String, String> {
    let pat = format!("'{key}'");
    let at = header.find(&pat).ok_or_else(|| missing(key))?;
    let after = header[at + pat.len()..].trim_start();
    let rest = after.strip_prefix(':').ok_or_else(|| missing(key))?;
    Ok(rest.to_string())
}

fn missing(key: &str) -> String {
    format!("missing '{key}' in .npy header")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_dtype_descriptors() {
        assert_eq!(map_descr("<f4").unwrap(), "F32");
        assert_eq!(map_descr("<f8").unwrap(), "F64");
        assert_eq!(map_descr("<f2").unwrap(), "F16");
        assert_eq!(map_descr("<i2").unwrap(), "I16");
        assert_eq!(map_descr("|i1").unwrap(), "I8");
        assert_eq!(map_descr("|u1").unwrap(), "U8");
        assert_eq!(map_descr("<u4").unwrap(), "U32");
        assert_eq!(map_descr("|b1").unwrap(), "BOOL");
        assert_eq!(map_descr("=i8").unwrap(), "I64");
        // Big-endian multi-byte and exotic kinds are rejected.
        assert!(map_descr(">f4").is_err());
        assert!(map_descr("<c8").is_err());
        assert!(map_descr("<U5").is_err());
    }

    #[test]
    fn parses_a_v1_header() {
        // A real v1.0 header for a 4×5 little-endian f32 C-order array.
        let dict = b"{'descr': '<f4', 'fortran_order': False, 'shape': (4, 5), }";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
        bytes.extend_from_slice(&(dict.len() as u16).to_le_bytes());
        bytes.extend_from_slice(dict);
        let mut cur = std::io::Cursor::new(&bytes);
        let h = parse_header(&mut cur).unwrap();
        assert_eq!(h.dtype, "F32");
        assert_eq!(h.shape, vec![4, 5]);
        assert_eq!(h.data_offset, bytes.len());
    }

    #[test]
    fn fortran_order_reverses_the_shape() {
        let dict = b"{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3), }";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
        bytes.extend_from_slice(&(dict.len() as u16).to_le_bytes());
        bytes.extend_from_slice(dict);
        let h = parse_header(&mut std::io::Cursor::new(&bytes)).unwrap();
        assert_eq!(h.shape, vec![3, 2]);
    }

    #[test]
    fn parses_scalar_and_1d_shapes() {
        assert_eq!(dict_shape("'shape': (), ").unwrap(), Vec::<usize>::new());
        assert_eq!(dict_shape("'shape': (7,), ").unwrap(), vec![7]);
        assert_eq!(dict_shape("'shape': (2, 3, 4), ").unwrap(), vec![2, 3, 4]);
    }

    /// numpy has written v2 (4-byte header length) since 1.9 for headers over 64 KiB —
    /// a real possibility for an array with many dimensions or a long structured dtype.
    /// The `data_offset` differs by the two extra length bytes, so getting the version
    /// wrong shifts every value read afterwards.
    #[test]
    fn parses_a_v2_header_with_its_wider_length_field() {
        let dict = b"{'descr': '<i2', 'fortran_order': False, 'shape': (3, 2), }";
        let mut bytes = b"\x93NUMPY\x02\x00".to_vec();
        bytes.extend((dict.len() as u32).to_le_bytes());
        bytes.extend(dict);
        let h = parse_header(&mut std::io::Cursor::new(&bytes)).unwrap();
        assert_eq!((h.dtype.as_str(), h.shape.clone()), ("I16", vec![3, 2]));
        assert_eq!(
            h.data_offset,
            8 + 4 + dict.len(),
            "v2 reserves 4 length bytes"
        );
        // The same dict as v1 sits two bytes earlier.
        let mut v1 = b"\x93NUMPY\x01\x00".to_vec();
        v1.extend((dict.len() as u16).to_le_bytes());
        v1.extend(dict);
        let h1 = parse_header(&mut std::io::Cursor::new(&v1)).unwrap();
        assert_eq!(h1.data_offset + 2, h.data_offset);
    }

    /// Every dtype the reader claims to map, in one place — a missing arm silently
    /// rejects a whole class of file, which is a support gap rather than a crash.
    #[test]
    fn maps_every_supported_width_and_kind() {
        for (descr, want) in [
            ("<f8", "F64"),
            ("<f4", "F32"),
            ("<f2", "F16"),
            ("<i8", "I64"),
            ("<i4", "I32"),
            ("<i2", "I16"),
            ("|i1", "I8"),
            ("<u8", "U64"),
            ("<u4", "U32"),
            ("<u2", "U16"),
            ("|u1", "U8"),
            ("|b1", "BOOL"),
        ] {
            assert_eq!(map_descr(descr).unwrap(), want, "{descr}");
        }
        // A single-byte big-endian descriptor has no byte order to get wrong, so it's
        // accepted where a multi-byte one is refused.
        assert_eq!(map_descr(">u1").unwrap(), "U8");
        assert!(map_descr(">u2").is_err());
        // No order prefix at all is native order, which is the little-endian we assume.
        assert_eq!(map_descr("f4").unwrap(), "F32");
        // Widths that exist in numpy but that we have no decoder for.
        assert!(map_descr("<f16").is_err()); // long double
        assert!(map_descr("<i16").is_err());
        assert!(map_descr("<f").is_err()); // no width at all
        assert!(map_descr("").is_err());
    }

    /// The header is parsed as text, so malformed input must produce a message rather
    /// than a panic on a slice — these files come off other people's machines.
    #[test]
    fn a_malformed_header_is_an_error_naming_what_is_missing() {
        let header = |dict: &str| {
            let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
            bytes.extend((dict.len() as u16).to_le_bytes());
            bytes.extend(dict.as_bytes());
            parse_header(&mut std::io::Cursor::new(bytes))
        };
        // Each key in turn: absent, or present with nothing usable after it.
        let e = header("{'fortran_order': False, 'shape': (1,), }").unwrap_err();
        assert!(e.contains("missing 'descr'"), "{e}");
        let e = header("{'descr': '<f4', 'shape': (1,), }").unwrap_err();
        assert!(e.contains("missing 'fortran_order'"), "{e}");
        let e = header("{'descr': '<f4', 'fortran_order': False, }").unwrap_err();
        assert!(e.contains("missing 'shape'"), "{e}");
        // Present but unparseable.
        let e = header("{'descr' '<f4', 'fortran_order': False, 'shape': (1,), }").unwrap_err();
        assert!(e.contains("missing 'descr'"), "no colon after the key: {e}");
        // An unquoted `descr` value: the string scan runs on to the next quoted token, so
        // this one surfaces as an unsupported dtype rather than a missing key. Still an
        // error naming the descriptor, which is the part the user has to fix.
        let e = header("{'descr': <f4, 'fortran_order': False, 'shape': (1,), }").unwrap_err();
        assert!(e.contains("unsupported dtype"), "unquoted value: {e}");
        let e = header("{'descr': '<f4', 'fortran_order': Maybe, 'shape': (1,), }").unwrap_err();
        assert!(e.contains("malformed 'fortran_order'"), "{e}");
        let e = header("{'descr': '<f4', 'fortran_order': False, 'shape': 4, }").unwrap_err();
        assert!(e.contains("missing 'shape'"), "no tuple: {e}");
        let e = header("{'descr': '<f4', 'fortran_order': False, 'shape': (2, x), }").unwrap_err();
        assert!(e.contains("bad dimension"), "{e}");
        // `False` before `True` in the text is still False (the note is a real one: the
        // *first* keyword after the key wins, not whichever appears anywhere).
        let dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), 'x': 'True'}";
        assert_eq!(header(dict).unwrap().shape, vec![2, 3], "not reversed");
    }

    /// Truncation at each stage of the read — the magic, the length field, the dict —
    /// must be reported, not read past.
    #[test]
    fn a_truncated_stream_errors_at_every_stage() {
        let dict = b"{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }";
        let mut full = b"\x93NUMPY\x01\x00".to_vec();
        full.extend((dict.len() as u16).to_le_bytes());
        full.extend(dict);
        for cut in 0..full.len() {
            let e = parse_header(&mut std::io::Cursor::new(&full[..cut]))
                .expect_err("a truncated header must not parse");
            assert!(e.contains(".npy"), "cut at {cut}: {e}");
        }
        assert!(parse_header(&mut std::io::Cursor::new(&full)).is_ok());
        // Right length, wrong file.
        let e = parse_header(&mut std::io::Cursor::new(b"PK\x03\x04zipfile")).unwrap_err();
        assert!(e.contains("bad magic"), "{e}");
    }
}
