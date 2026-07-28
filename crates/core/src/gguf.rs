// The GGML type names are the format's own (`Q4_K`, `IQ2_XXS`), so they keep their spelling
// rather than being re-cased into something a reader can't match against the spec.
#![allow(non_camel_case_types)]

use anyhow::Result;
use std::collections::HashMap;
use std::io::{Cursor, Read};

/// GGUF file format parser
/// Based on llama.cpp GGUF specification
pub struct GGUFFile {
    pub header: GGUFHeader,
    pub metadata: HashMap<String, GGUFValue>,
    pub tensors: Vec<GGUFTensorInfo>,
}

#[derive(Debug, Clone)]
pub struct GGUFHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GGMLType,
    pub offset: u64,
}

#[derive(Debug, Clone)]
pub enum GGUFValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Self>),
}

/// GGML tensor types from llama.cpp
/// Includes all quantization formats
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    GGML_TYPE_Q1_58 = 36,
}

impl GGMLType {
    #[must_use]
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            16 => Some(Self::IQ2_XXS),
            17 => Some(Self::IQ2_XS),
            18 => Some(Self::IQ3_XXS),
            19 => Some(Self::IQ1_S),
            20 => Some(Self::IQ4_NL),
            21 => Some(Self::IQ3_S),
            22 => Some(Self::IQ2_S),
            23 => Some(Self::IQ4_XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1_M),
            30 => Some(Self::BF16),
            36 => Some(Self::GGML_TYPE_Q1_58),
            _ => None,
        }
    }

    /// This type's storage block as `(bytes_per_block, elements_per_block)`.
    ///
    /// GGUF quantizes in whole blocks, so a tensor's size is
    /// `ceil(elements / elements_per_block) * bytes_per_block`. Kept as an exact
    /// integer ratio rather than a bytes-per-element `f32`: that had only 24 mantissa
    /// bits, so large tensors came out wrong (an F32 tensor of 20,000,001 elements
    /// measured 80,000,000 instead of 80,000,004), and truncating the product also
    /// dropped the final partial block.
    #[must_use]
    pub fn block_layout(&self) -> (usize, usize) {
        match self {
            Self::F32 | Self::I32 => (4, 1),
            Self::F16 | Self::BF16 | Self::I16 => (2, 1),
            Self::F64 | Self::I64 => (8, 1),
            Self::I8 => (1, 1),

            // Legacy Q‑quants (block of 32 weights)
            Self::Q4_0 => (18, 32),
            Self::Q4_1 => (20, 32),
            Self::Q5_0 => (22, 32),
            Self::Q5_1 => (24, 32),
            Self::Q8_0 => (34, 32),
            Self::Q8_1 => (36, 32),

            // K‑quants (super‑block of 256 weights); bytes = bpw * 256 / 8
            Self::Q2_K => (84, 256), // 2.625  bpw
            // Q3_K and IQ3_S share a block layout (3.4375 bpw), hence one arm.
            Self::Q3_K | Self::IQ3_S => (110, 256),
            Self::Q4_K => (144, 256), // 4.5    bpw
            Self::Q5_K => (176, 256), // 5.5    bpw
            Self::Q6_K => (210, 256), // 6.5625 bpw
            Self::Q8_K => (292, 256), // 9.125  bpw

            // Importance‑quants (IQ‑family, super‑block 256 except IQ4_NL's 32)
            Self::IQ1_S => (50, 256),   // 1.5625 bpw
            Self::IQ1_M => (56, 256),   // 1.75   bpw
            Self::IQ2_XXS => (66, 256), // 2.0625 bpw
            Self::IQ2_XS => (74, 256),  // 2.3125 bpw
            Self::IQ2_S => (80, 256),   // 2.5    bpw
            Self::IQ3_XXS => (98, 256), // 3.0625 bpw
            Self::IQ4_NL => (17, 32),   // 4.25   bpw
            Self::IQ4_XS => (136, 256), // 4.25   bpw
            // 1.58 bpw doesn't land on a whole byte per 256, so keep the exact ratio
            // this type has always been measured with (0.1975 B/elem = 79/400).
            Self::GGML_TYPE_Q1_58 => (79, 400),
        }
    }

    /// Exact stored size of `elements` values of this type, rounded up to whole
    /// blocks (see [`GGMLType::block_layout`]).
    #[must_use]
    pub fn stored_size(&self, elements: usize) -> usize {
        let (bytes, per_block) = self.block_layout();
        elements.div_ceil(per_block.max(1)).saturating_mul(bytes)
    }

    /// Every type, so the tests can sweep all of them instead of whichever ones someone
    /// remembered to list. (It's still hand-maintained: a *new* variant has to be added
    /// here as well as to `from_u32` and `Display`. What the sweep does catch is the
    /// likelier mistake — a wrong discriminant or a copy-pasted name among the 30 that
    /// are already here.)
    #[cfg(test)]
    pub(crate) const ALL: [Self; 30] = [
        Self::F32,
        Self::F16,
        Self::F64,
        Self::BF16,
        Self::I8,
        Self::I16,
        Self::I32,
        Self::I64,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Q8_0,
        Self::Q8_1,
        Self::Q2_K,
        Self::Q3_K,
        Self::Q4_K,
        Self::Q5_K,
        Self::Q6_K,
        Self::Q8_K,
        Self::IQ1_S,
        Self::IQ1_M,
        Self::IQ2_XXS,
        Self::IQ2_XS,
        Self::IQ2_S,
        Self::IQ3_XXS,
        Self::IQ3_S,
        Self::IQ4_NL,
        Self::IQ4_XS,
        Self::GGML_TYPE_Q1_58,
    ];
}

impl std::fmt::Display for GGMLType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::F64 => "F64",
            Self::BF16 => "BF16",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ3_S => "IQ3_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ4_XS => "IQ4_XS",
            Self::IQ1_M => "IQ1_M",
            Self::GGML_TYPE_Q1_58 => "Q1_58",
        };
        write!(f, "{s}")
    }
}

impl std::fmt::Display for GGUFValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::U8(v) => write!(f, "{v}"),
            Self::I8(v) => write!(f, "{v}"),
            Self::U16(v) => write!(f, "{v}"),
            Self::I16(v) => write!(f, "{v}"),
            Self::U32(v) => write!(f, "{v}"),
            Self::I32(v) => write!(f, "{v}"),
            Self::F32(v) => write!(f, "{v}"),
            Self::U64(v) => write!(f, "{v}"),
            Self::I64(v) => write!(f, "{v}"),
            Self::F64(v) => write!(f, "{v}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::String(v) => write!(f, "\"{v}\""),
            Self::Array(arr) => {
                if arr.len() <= 5 {
                    // Show small arrays in full
                    write!(f, "[")?;
                    for (i, item) in arr.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{item}")?;
                    }
                    write!(f, "]")
                } else {
                    // Show truncated arrays
                    write!(
                        f,
                        "[{}, {}, ..., {} ({})]",
                        // `len() > 5` on this branch, so all three are present.
                        arr.first().map_or_else(String::new, ToString::to_string),
                        arr.get(1).map_or_else(String::new, ToString::to_string),
                        arr.last().map_or_else(String::new, ToString::to_string),
                        arr.len()
                    )
                }
            }
        }
    }
}

impl GGUFFile {
    pub fn read(data: &[u8]) -> Result<Self> {
        let mut cursor = Cursor::new(data);

        // Read header
        let header = Self::read_header(&mut cursor)?;

        // Validate magic number
        if header.magic != 0x4655_4747 {
            return Err(anyhow::anyhow!("Invalid GGUF magic number"));
        }

        // Read metadata
        let metadata = Self::read_metadata(&mut cursor, header.metadata_kv_count)?;

        // Read tensor info
        let tensors = Self::read_tensor_info(&mut cursor, header.tensor_count)?;

        Ok(Self {
            header,
            metadata,
            tensors,
        })
    }

    fn read_header(cursor: &mut Cursor<&[u8]>) -> Result<GGUFHeader> {
        let magic = Self::read_u32(cursor)?;
        let version = Self::read_u32(cursor)?;
        let tensor_count = Self::read_u64(cursor)?;
        let metadata_kv_count = Self::read_u64(cursor)?;

        Ok(GGUFHeader {
            magic,
            version,
            tensor_count,
            metadata_kv_count,
        })
    }

    fn read_metadata(cursor: &mut Cursor<&[u8]>, count: u64) -> Result<HashMap<String, GGUFValue>> {
        let mut metadata = HashMap::new();

        for _ in 0..count {
            let key = Self::read_string(cursor)?;
            let value_type = Self::read_u32(cursor)?;
            let value = Self::read_value(cursor, value_type)?;
            metadata.insert(key, value);
        }

        Ok(metadata)
    }

    fn read_tensor_info(cursor: &mut Cursor<&[u8]>, count: u64) -> Result<Vec<GGUFTensorInfo>> {
        let mut tensors = Vec::new();

        for _ in 0..count {
            let name = Self::read_string(cursor)?;
            let n_dimensions = Self::read_u32(cursor)?;
            let mut dimensions = Vec::new();

            for _ in 0..n_dimensions {
                dimensions.push(Self::read_u64(cursor)?);
            }

            let tensor_type_u32 = Self::read_u32(cursor)?;
            let tensor_type = GGMLType::from_u32(tensor_type_u32)
                .ok_or_else(|| anyhow::anyhow!("Unknown tensor type: {}", tensor_type_u32))?;

            let offset = Self::read_u64(cursor)?;

            tensors.push(GGUFTensorInfo {
                name,
                dimensions,
                tensor_type,
                offset,
            });
        }

        Ok(tensors)
    }

    fn read_value(cursor: &mut Cursor<&[u8]>, value_type: u32) -> Result<GGUFValue> {
        match value_type {
            0 => Ok(GGUFValue::U8(Self::read_u8(cursor)?)),
            1 => Ok(GGUFValue::I8(Self::read_i8(cursor)?)),
            2 => Ok(GGUFValue::U16(Self::read_u16(cursor)?)),
            3 => Ok(GGUFValue::I16(Self::read_i16(cursor)?)),
            4 => Ok(GGUFValue::U32(Self::read_u32(cursor)?)),
            5 => Ok(GGUFValue::I32(Self::read_i32(cursor)?)),
            6 => Ok(GGUFValue::F32(Self::read_f32(cursor)?)),
            7 => Ok(GGUFValue::Bool(Self::read_u8(cursor)? != 0)),
            8 => Ok(GGUFValue::String(Self::read_string(cursor)?)),
            9 => {
                let array_type = Self::read_u32(cursor)?;
                let array_len = Self::read_u64(cursor)?;
                let mut array = Vec::new();
                for _ in 0..array_len {
                    array.push(Self::read_value(cursor, array_type)?);
                }
                Ok(GGUFValue::Array(array))
            }
            10 => Ok(GGUFValue::U64(Self::read_u64(cursor)?)),
            11 => Ok(GGUFValue::I64(Self::read_i64(cursor)?)),
            12 => Ok(GGUFValue::F64(Self::read_f64(cursor)?)),
            _ => Err(anyhow::anyhow!("Unknown value type: {}", value_type)),
        }
    }

    /// Read a length-prefixed string.
    ///
    /// The length is read *from the file*, so a corrupt or truncated GGUF can claim a
    /// string of exabytes — and `vec![0u8; len]` on that aborts the process (an
    /// allocation failure isn't a catchable error). GGUF files arrive as user downloads
    /// from `HuggingFace`, so a half-downloaded one is ordinary input and must produce an
    /// error, not a dead process. Check the claim against the bytes actually left
    /// before allocating.
    fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
        let len = Self::read_u64(cursor)?;
        let remaining = (cursor.get_ref().len() as u64).saturating_sub(cursor.position());
        if len > remaining {
            return Err(anyhow::anyhow!(
                "GGUF string at offset {} claims {len} bytes but only {remaining} remain \
                 (truncated or corrupt file)",
                cursor.position()
            ));
        }
        let mut bytes = vec![0u8; len as usize];
        cursor.read_exact(&mut bytes)?;
        Ok(String::from_utf8(bytes)?)
    }

    fn read_u8(cursor: &mut Cursor<&[u8]>) -> Result<u8> {
        let mut buf = [0u8; 1];
        cursor.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_i8(cursor: &mut Cursor<&[u8]>) -> Result<i8> {
        Ok(Self::read_u8(cursor)? as i8)
    }

    fn read_u16(cursor: &mut Cursor<&[u8]>) -> Result<u16> {
        let mut buf = [0u8; 2];
        cursor.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(cursor: &mut Cursor<&[u8]>) -> Result<i16> {
        let mut buf = [0u8; 2];
        cursor.read_exact(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
        let mut buf = [0u8; 4];
        cursor.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(cursor: &mut Cursor<&[u8]>) -> Result<i32> {
        let mut buf = [0u8; 4];
        cursor.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_f32(cursor: &mut Cursor<&[u8]>) -> Result<f32> {
        let mut buf = [0u8; 4];
        cursor.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_u64(cursor: &mut Cursor<&[u8]>) -> Result<u64> {
        let mut buf = [0u8; 8];
        cursor.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(cursor: &mut Cursor<&[u8]>) -> Result<i64> {
        let mut buf = [0u8; 8];
        cursor.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f64(cursor: &mut Cursor<&[u8]>) -> Result<f64> {
        let mut buf = [0u8; 8];
        cursor.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }
}

/// A GGUF writer, so the reader can be tested against bytes rather than only against
/// whatever real file happens to be at hand. Little-endian throughout, as the format
/// specifies.
///
/// `pub(crate)` because `readers.rs` needs it too: its `read_gguf` is the layer above
/// this one, and testing it means putting a real GGUF file on disk.
#[cfg(test)]
pub(crate) mod testing {
    #[derive(Default)]
    pub(crate) struct Gguf {
        body: Vec<u8>,
        tensors: u64,
        kvs: u64,
    }

    impl Gguf {
        pub(crate) fn str(&mut self, s: &str) -> &mut Self {
            self.body.extend((s.len() as u64).to_le_bytes());
            self.body.extend(s.as_bytes());
            self
        }
        pub(crate) fn u32(&mut self, v: u32) -> &mut Self {
            self.body.extend(v.to_le_bytes());
            self
        }
        pub(crate) fn u64(&mut self, v: u64) -> &mut Self {
            self.body.extend(v.to_le_bytes());
            self
        }
        /// One metadata entry: key, value-type tag, then the value's bytes.
        pub(crate) fn kv(&mut self, key: &str, ty: u32, value: &[u8]) -> &mut Self {
            self.kvs += 1;
            self.str(key).u32(ty);
            self.body.extend(value);
            self
        }
        pub(crate) fn tensor(
            &mut self,
            name: &str,
            dims: &[u64],
            ty: u32,
            offset: u64,
        ) -> &mut Self {
            self.tensors += 1;
            self.str(name).u32(dims.len() as u32);
            for d in dims {
                self.u64(*d);
            }
            self.u32(ty).u64(offset);
            self
        }
        /// Header (magic `GGUF`, version, counts) followed by the body.
        pub(crate) fn finish(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend(0x4655_4747u32.to_le_bytes());
            out.extend(3u32.to_le_bytes());
            out.extend(self.tensors.to_le_bytes());
            out.extend(self.kvs.to_le_bytes());
            out.extend(&self.body);
            out
        }
    }

    /// A GGUF string value's payload (length-prefixed, as `kv` expects for type 8).
    pub(crate) fn gguf_str(s: &str) -> Vec<u8> {
        let mut v = (s.len() as u64).to_le_bytes().to_vec();
        v.extend(s.as_bytes());
        v
    }

    pub(crate) fn le_u64(v: u64) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{Gguf, le_u64};
    use super::*;

    /// Sizes were computed as `elements as f32 * bytes_per_element`, which has only 24
    /// mantissa bits and truncated the trailing partial block. Both errors are visible on
    /// realistic tensor sizes, and the result feeds the displayed size, the checkpoint
    /// totals, and the exact `size>N` filter.

    #[test]
    fn reads_a_whole_file_header_metadata_and_tensors() {
        let mut g = Gguf::default();
        g.kv("general.architecture", 8, &{
            let mut v = Vec::new();
            v.extend(5u64.to_le_bytes());
            v.extend(b"llama");
            v
        })
        .kv("block_count", 4, &32u32.to_le_bytes())
        .kv("rope.freq_base", 6, &10000.0f32.to_le_bytes())
        .kv("use_parallel", 7, &[1])
        .kv("context_length", 10, &le_u64(4096))
        .tensor("token_embd.weight", &[4096, 32000], 0, 0)
        .tensor("blk.0.attn_q.weight", &[4096, 4096], 8, 512);
        let file = GGUFFile::read(&g.finish()).expect("a well-formed file reads");

        assert_eq!(file.header.version, 3);
        assert_eq!(file.header.tensor_count, 2);
        assert_eq!(file.metadata.len(), 5);
        assert!(matches!(
            file.metadata.get("general.architecture"),
            Some(GGUFValue::String(s)) if s == "llama"
        ));
        assert!(matches!(
            file.metadata.get("block_count"),
            Some(GGUFValue::U32(32))
        ));
        assert!(matches!(
            file.metadata.get("use_parallel"),
            Some(GGUFValue::Bool(true))
        ));
        assert!(matches!(
            file.metadata.get("context_length"),
            Some(GGUFValue::U64(4096))
        ));

        assert_eq!(file.tensors[0].name, "token_embd.weight");
        assert_eq!(file.tensors[0].dimensions, vec![4096, 32000]);
        assert_eq!(file.tensors[0].tensor_type, GGMLType::F32);
        assert_eq!(file.tensors[1].tensor_type, GGMLType::Q8_0);
        assert_eq!(file.tensors[1].offset, 512);
    }

    #[test]
    fn reads_a_nested_array_value() {
        // Arrays carry their element type once, then the elements — and can hold
        // strings, which is how tokenizer vocabularies arrive.
        let mut elems = Vec::new();
        elems.extend(8u32.to_le_bytes()); // element type: string
        elems.extend(2u64.to_le_bytes()); // length
        for s in ["<s>", "</s>"] {
            elems.extend((s.len() as u64).to_le_bytes());
            elems.extend(s.as_bytes());
        }
        let mut g = Gguf::default();
        g.kv("tokenizer.ggml.tokens", 9, &elems);
        let file = GGUFFile::read(&g.finish()).expect("an array value reads");
        match file.metadata.get("tokenizer.ggml.tokens") {
            Some(GGUFValue::Array(items)) => {
                assert_eq!(items.len(), 2);
                assert!(matches!(&items[0], GGUFValue::String(s) if s == "<s>"));
            }
            other => panic!("expected an array, got {other:?}"),
        }
    }

    #[test]
    fn rejects_a_file_that_is_not_gguf() {
        let mut bytes = Gguf::default().finish();
        bytes[0..4].copy_from_slice(b"XXXX");
        let err = GGUFFile::read(&bytes)
            .err()
            .expect("bad magic must be refused");
        assert!(format!("{err}").contains("magic"), "{err}");
    }

    #[test]
    fn rejects_an_unknown_value_type_instead_of_guessing() {
        let mut g = Gguf::default();
        g.kv("weird", 99, &[0, 0, 0, 0]);
        let err = GGUFFile::read(&g.finish())
            .err()
            .expect("an unknown tag must be refused");
        assert!(format!("{err}").contains("Unknown value type"), "{err}");
    }

    #[test]
    fn rejects_an_unknown_tensor_type() {
        let mut g = Gguf::default();
        g.tensor("t", &[4], 4242, 0);
        let err = GGUFFile::read(&g.finish())
            .err()
            .expect("an unknown tensor type must be refused");
        assert!(format!("{err}").contains("Unknown tensor type"), "{err}");
    }

    /// A truncated file must produce an error at every cut point — not a panic. These
    /// files come from users and from `HuggingFace`; a half-downloaded one is ordinary.
    #[test]
    fn a_truncated_file_errors_at_every_cut_point() {
        let mut g = Gguf::default();
        g.kv("general.architecture", 8, &{
            let mut v = Vec::new();
            v.extend(5u64.to_le_bytes());
            v.extend(b"llama");
            v
        })
        .tensor("token_embd.weight", &[4096, 32000], 0, 0);
        let full = g.finish();
        for cut in 0..full.len() {
            let result = GGUFFile::read(&full[..cut]);
            assert!(result.is_err(), "truncating to {cut} bytes should error");
        }
        assert!(
            GGUFFile::read(&full).is_ok(),
            "the untruncated file still reads"
        );
    }

    #[test]
    fn rejects_a_non_utf8_string() {
        let mut g = Gguf::default();
        g.kv("bad", 8, &{
            let mut v = Vec::new();
            v.extend(2u64.to_le_bytes());
            v.extend([0xff, 0xfe]); // not valid UTF-8
            v
        });
        assert!(GGUFFile::read(&g.finish()).is_err());
    }

    /// A length field is attacker-controlled: a corrupt file can claim a string is
    /// exabytes long. Reading it must fail on the missing bytes rather than trying to
    /// allocate that much first.
    #[test]
    fn an_absurd_string_length_fails_without_allocating_it() {
        let mut bytes = Vec::new();
        bytes.extend(0x4655_4747u32.to_le_bytes());
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes()); // no tensors
        bytes.extend(1u64.to_le_bytes()); // one kv…
        bytes.extend((u64::MAX / 2).to_le_bytes()); // …whose key claims to be enormous
        assert!(GGUFFile::read(&bytes).is_err());
    }

    #[test]
    fn stored_size_is_exact_and_rounds_up_to_whole_blocks() {
        // f32 rounding: 20_000_001 * 4.0 lost the last increment.
        assert_eq!(GGMLType::F32.stored_size(20_000_001), 80_000_004);
        assert_eq!(GGMLType::F16.stored_size(20_000_001), 40_000_002);
        // Whole-block rounding: 24_990_001 elements is 780_938 blocks of 32 (the last
        // holding a single value), not the truncated 780_937.
        assert_eq!(GGMLType::Q4_0.stored_size(24_990_001), 780_938 * 18);
        assert_eq!(GGMLType::Q4_0.stored_size(24_990_001), 14_056_884);
        // Exact multiples stay exact.
        assert_eq!(GGMLType::Q4_0.stored_size(32), 18);
        assert_eq!(GGMLType::Q4_K.stored_size(256), 144);
        assert_eq!(GGMLType::Q4_K.stored_size(257), 288); // spills into a second block
        // Empty tensors have no blocks.
        assert_eq!(GGMLType::F32.stored_size(0), 0);
        assert_eq!(GGMLType::Q6_K.stored_size(0), 0);
        // Every type reports a usable block (no zero divisor, no zero-byte block).
        for t in GGMLType::ALL {
            let (bytes, per_block) = t.block_layout();
            assert!(bytes > 0 && per_block > 0, "{t} has an empty block layout");
            assert!(t.stored_size(1) > 0, "{t} sized one element as 0 bytes");
            // The documented formula, checked against the implementation across a range
            // that crosses block boundaries in both directions.
            for n in [1usize, per_block - 1, per_block, per_block + 1, 10_000_003] {
                assert_eq!(
                    t.stored_size(n),
                    n.div_ceil(per_block) * bytes,
                    "{t} sized {n} elements wrong"
                );
            }
        }
    }

    /// The type table is three parallel matches — the discriminant, `from_u32`, and the
    /// displayed name — and nothing tied them together. A wrong number in `from_u32`
    /// silently reads every tensor of that type as some other type.
    #[test]
    fn every_ggml_type_round_trips_through_its_tag_and_has_its_own_name() {
        let mut names: Vec<&'static str> = Vec::new();
        for t in GGMLType::ALL {
            let tag = t as u32;
            assert_eq!(
                GGMLType::from_u32(tag),
                Some(t),
                "{t} (tag {tag}) does not come back from its own tag"
            );
            let name = t.to_string();
            assert!(!name.is_empty(), "{t:?} has no display name");
            assert!(
                !names.contains(&name.as_str()),
                "two types both display as {name:?}"
            );
            names.push(Box::leak(name.into_boxed_str()));
        }
        // Tags nobody has defined stay unknown rather than mapping to a neighbour — the
        // reader refuses the file instead of misreading its tensors.
        for tag in [4u32, 5, 31, 35, 37, 100, u32::MAX] {
            assert!(
                GGMLType::from_u32(tag).is_none(),
                "tag {tag} should be unknown"
            );
        }
    }

    /// Bits per weight, derived from the block layout, must land near the name — the
    /// quant tables are dense columns of numbers where a transposed digit is invisible
    /// by eye but silently misreports every size in the checkpoint.
    #[test]
    fn quantized_block_layouts_match_the_width_in_their_name() {
        for t in GGMLType::ALL {
            let (bytes, per_block) = t.block_layout();
            let bpw = bytes as f64 * 8.0 / per_block as f64;
            let name = t.to_string();
            // `Q4_K` / `IQ2_XS` / `Q1_58`: the digit after the leading Q or IQ is the
            // nominal bit width, and the real one is that plus block overhead — never
            // less, and never more than ~1.5 bits of it. (Matching only `Q`/`IQ` and not
            // a bare leading `I`, or `I16` would read as a 1-bit quant.)
            let Some(nominal) = name
                .strip_prefix("IQ")
                .or_else(|| name.strip_prefix('Q'))
                .and_then(|rest| rest.chars().next())
                .and_then(|c| c.to_digit(10))
            else {
                // F32/F16/BF16/I8… — exact widths, no blocking.
                assert_eq!(per_block, 1, "{name} should not be blocked");
                assert_eq!(bpw, bytes as f64 * 8.0);
                continue;
            };
            let nominal = f64::from(nominal);
            assert!(
                bpw >= nominal && bpw <= nominal + 1.6,
                "{name} stores {bpw:.4} bits per weight, which is not ~{nominal}"
            );
        }
    }

    /// Metadata values are shown as text on the detail screen; long arrays are elided so
    /// one 32k-token vocabulary doesn't push everything else off the panel.
    #[test]
    fn values_render_as_text_with_long_arrays_elided() {
        use GGUFValue::{Array, Bool, F32, I32, String as Str, U32};
        assert_eq!(U32(7).to_string(), "7");
        assert_eq!(I32(-7).to_string(), "-7");
        assert_eq!(F32(0.5).to_string(), "0.5");
        assert_eq!(Bool(true).to_string(), "true");
        // Strings are quoted, so `"32"` can't be mistaken for the number.
        assert_eq!(Str("llama".into()).to_string(), "\"llama\"");
        // Up to five elements show in full.
        let small = Array(vec![U32(1), U32(2), U32(3)]);
        assert_eq!(small.to_string(), "[1, 2, 3]");
        let five = Array((1..=5).map(U32).collect());
        assert_eq!(five.to_string(), "[1, 2, 3, 4, 5]");
        // Six or more elide the middle, keeping the ends and the count — enough to tell
        // what the array is without printing a vocabulary.
        let six = Array((1..=6).map(U32).collect());
        assert_eq!(six.to_string(), "[1, 2, ..., 6 (6)]");
        let tokens = Array((0..32_000).map(|i| Str(format!("tok{i}"))).collect());
        assert_eq!(
            tokens.to_string(),
            "[\"tok0\", \"tok1\", ..., \"tok31999\" (32000)]"
        );
        // Nested arrays render through the same rule.
        assert_eq!(
            Array(vec![small, six]).to_string(),
            "[[1, 2, 3], [1, 2, ..., 6 (6)]]"
        );
    }
}
