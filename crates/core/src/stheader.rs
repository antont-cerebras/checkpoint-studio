//! Pure parsing of a safetensors **header** (the JSON blob after the 8-byte
//! length) into [`TensorInfo`]/[`MetadataInfo`]. Header-only — never touches the
//! tensor data. Shared by the local file reader, the remote SFTP reader
//! (`crate::sftp`), and the `--ssh-proxy` path, so the one parse lives in the
//! core crate with no dependency on the TUI/`Explorer`.

use anyhow::{Context, Result};

use crate::tree::{Layout, MetadataInfo, Storage, TensorInfo};

/// Validate a safetensors header length against a sane ceiling (guards a
/// corrupt / non-safetensors file claiming a huge header).
pub fn header_len(raw: u64, source: &str) -> Result<usize> {
    const MAX_HEADER_SIZE: u64 = 100_000_000;
    if raw > MAX_HEADER_SIZE {
        anyhow::bail!("SafeTensors header too large ({raw} bytes): {source}");
    }
    Ok(raw as usize)
}

/// Top-level keys the raw header declares **more than once**, in name order.
///
/// The JSON parse cannot tell you this: it goes through `serde_json::Map`, which keeps
/// the *last* of two identical keys. A header that declares `w` twice therefore parses as
/// one tensor, and the first declaration — its dtype, its shape, its byte span — is
/// discarded without a word. Every consumer then agrees with every other consumer about a
/// file that says two things.
///
/// The format has no notion of a repeated key and no writer should emit one, so a file
/// that does is one whose writer and reader disagree about its contents — worth an error
/// even though nothing will crash.
///
/// Walks the header with a visitor that keeps each entry as it comes instead of folding
/// it into a map, and discards the values: only the key sequence matters here.
fn duplicate_keys(header_buf: &[u8]) -> Result<Vec<String>> {
    let keys: KeySequence = serde_json::from_slice(header_buf)
        .with_context(|| "Failed to re-scan the SafeTensors header for repeated keys")?;
    let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    for key in keys.0 {
        *counts.entry(key).or_default() += 1;
    }
    Ok(counts
        .into_iter()
        .filter(|&(_, n)| n > 1)
        .map(|(k, _)| k)
        .collect())
}

/// Every top-level key of a JSON object, in file order, duplicates kept.
struct KeySequence(Vec<String>);

impl<'de> serde::Deserialize<'de> for KeySequence {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct Keys;
        impl<'de> serde::de::Visitor<'de> for Keys {
            type Value = KeySequence;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a safetensors header object")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<KeySequence, A::Error> {
                let mut keys = Vec::new();
                while let Some(key) = map.next_key::<String>()? {
                    // Required by `MapAccess`, and the values are genuinely not wanted:
                    // `IgnoredAny` walks each one without allocating it.
                    map.next_value::<serde::de::IgnoredAny>()?;
                    keys.push(key);
                }
                Ok(KeySequence(keys))
            }
        }
        d.deserialize_map(Keys)
    }
}

/// The JSON type name of a value, for [`MetadataInfo::value_type`].
fn json_type(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::String(_) => "string",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
        serde_json::Value::Null => "null",
    }
}

/// What one safetensors header says — everything a reader learns from the bytes it
/// already holds.
///
/// A struct rather than a tuple because the third field is the point: [`duplicate_keys`]
/// can only be known *while* parsing, and returning it here is what makes it known for
/// every source. Detecting it afterwards needs the header text again, which only a local
/// file can give — that asymmetry was a fact about where the check lived, not about the
/// checkpoint.
pub struct ParsedHeader {
    pub tensors: Vec<TensorInfo>,
    pub metadata: Vec<MetadataInfo>,
    /// Top-level keys the header declares more than once — see [`duplicate_keys`].
    pub duplicate_keys: Vec<String>,
}

/// Parse a safetensors header (the JSON blob after the 8-byte length) into
/// tensors + metadata, plus any repeated key. `source` is the tensors' `source_path` (a
/// local path or a remote marker). Every non-`__metadata__` entry describes a tensor.
pub fn parse_header(header_buf: &[u8], source: &str) -> Result<ParsedHeader> {
    let mut tensors: Vec<TensorInfo> = Vec::new();
    let mut metadata: Vec<MetadataInfo> = Vec::new();
    let source_path = source.to_string();

    let header: serde_json::Value = serde_json::from_slice(header_buf)
        .with_context(|| format!("Failed to parse SafeTensors header: {source}"))?;

    let obj = header
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Invalid SafeTensors header: {source}"))?;

    for (key, value) in obj {
        // The `__metadata__` entry holds free-form string key/value pairs.
        if key == "__metadata__" {
            if let Some(meta_obj) = value.as_object() {
                for (meta_key, meta_value) in meta_obj {
                    metadata.push(MetadataInfo {
                        name: meta_key.clone(),
                        value: meta_value
                            .as_str()
                            .map_or_else(|| meta_value.to_string(), ToString::to_string),
                        // The value's REAL JSON type, not "string" for everything. The
                        // spec makes `__metadata__` a string→string map, so anything
                        // else is a non-conforming writer — and reporting it as a string
                        // is how that went unnoticed. `check_headers` flags it.
                        value_type: json_type(meta_value).to_string(),
                    });
                }
            }
            continue;
        }

        // Every other entry describes a tensor.
        let dtype = value
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        let shape: Vec<usize> = value
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        let data_offsets = value
            .get("data_offsets")
            .and_then(|v| v.as_array())
            // Exactly the `[start, end]` pair safetensors defines — matched as a slice so
            // the length check and the reads are one operation.
            .and_then(|offsets| match offsets.as_slice() {
                [start, end] => Some((start.as_u64()?, end.as_u64()?)),
                _ => None,
            });
        let size_bytes = data_offsets.map_or(0, |(start, end)| end.saturating_sub(start) as usize);
        let layout = match data_offsets {
            Some((start, end)) => Layout::ByteRange { start, end },
            None => Layout::None,
        };
        let num_elements = shape.iter().product::<usize>();

        tensors.push(TensorInfo {
            name: key.clone(),
            dtype,
            shape,
            size_bytes,
            num_elements,
            storage: Storage::Unknown,
            source_path: source_path.clone(),
            layout,
        });
    }

    Ok(ParsedHeader {
        tensors,
        metadata,
        // The same buffer, walked a second time without allocating its values. Cheap, and
        // it happens here so every reader gets the answer — the JSON parse above has
        // already folded the repeats away, so this is the only moment they exist.
        // Infallible in practice: the parse above succeeded on these bytes.
        duplicate_keys: duplicate_keys(header_buf).unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicate_keys_sees_what_serde_json_folds_away() {
        // The whole reason this function exists: parsing keeps the LAST of two identical
        // keys, so the header below describes one tensor as far as every consumer is
        // concerned — and the first description of it is gone.
        let dup = br#"{"w": {"dtype":"F32","shape":[4],"data_offsets":[0,16]},
                      "w": {"dtype":"F16","shape":[8],"data_offsets":[0,16]}}"#;
        let h = parse_header(dup, "mem.safetensors").expect("it parses");
        assert_eq!(h.tensors.len(), 1, "one tensor survives the fold");
        assert_eq!(
            h.tensors[0].dtype, "F16",
            "and it is the second declaration"
        );
        // …and the parse reports it, so every reader knows without re-reading anything.
        assert_eq!(h.duplicate_keys, vec!["w".to_string()]);
    }

    #[test]
    fn duplicate_keys_is_quiet_on_an_ordinary_header() {
        let ok = br#"{"a": {"dtype":"F32","shape":[1],"data_offsets":[0,4]},
                     "b": {"dtype":"F32","shape":[1],"data_offsets":[4,8]},
                     "__metadata__": {"format":"pt"}}"#;
        assert!(duplicate_keys(ok).expect("valid JSON").is_empty());
        assert!(
            parse_header(ok, "mem.safetensors")
                .expect("parses")
                .duplicate_keys
                .is_empty()
        );
        // Repeats *inside* a value are not repeated tensor names — only the top level
        // decides what the file declares.
        let nested = br#"{"a": {"dtype":"F32","dtype":"F16","shape":[1],"data_offsets":[0,4]}}"#;
        assert!(duplicate_keys(nested).expect("valid JSON").is_empty());
    }

    #[test]
    fn a_damaged_header_is_an_error_not_a_panic() {
        // Every shape of damage a header can arrive in. None of these may panic: the
        // callers turn the error into a message (a popup, a check finding, a non-zero
        // exit), and a panic would take the whole TUI down instead.
        for bad in [
            &b""[..],                  // empty
            &b"{"[..],                 // truncated object
            &b"\xff\xfe not json"[..], // not JSON at all, and not UTF-8
            &b"[1, 2, 3]"[..],         // valid JSON, wrong shape
            &b"\"a string\""[..],      // ditto
            &b"null"[..],
        ] {
            assert!(parse_header(bad, "mem.safetensors").is_err(), "{bad:?}");
            assert!(duplicate_keys(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn header_len_rejects_an_absurd_prefix() {
        // A corrupt / non-safetensors file whose first 8 bytes read as a huge number:
        // refuse before allocating it, with the file named.
        let err = header_len(1 << 40, "/c/x.safetensors").expect_err("rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("too large") && msg.contains("/c/x.safetensors"),
            "{msg}"
        );
        assert_eq!(header_len(1024, "/c/x.safetensors").expect("sane"), 1024);
    }

    #[test]
    fn metadata_values_keep_their_real_json_type() {
        // The spec says string→string. A writer that puts a number there gets reported
        // as a number, so `check_headers` can say so — this used to claim "string".
        let h =
            br#"{"__metadata__": {"format":"pt","count":7,"flag":true,"nested":{},"nil":null}}"#;
        let meta = parse_header(h, "mem.safetensors")
            .expect("it parses")
            .metadata;
        let ty = |name: &str| {
            meta.iter()
                .find(|m| m.name == name)
                .map(|m| m.value_type.as_str())
        };
        assert_eq!(ty("format"), Some("string"));
        assert_eq!(ty("count"), Some("number"));
        assert_eq!(ty("flag"), Some("bool"));
        assert_eq!(ty("nested"), Some("object"));
        assert_eq!(ty("nil"), Some("null"));
    }
}
