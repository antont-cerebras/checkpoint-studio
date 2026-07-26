#![allow(unused, non_camel_case_types)]

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
    Array(Vec<GGUFValue>),
}

/// GGML tensor types from llama.cpp
/// Includes all quantization formats
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
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
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(GGMLType::F32),
            1 => Some(GGMLType::F16),
            2 => Some(GGMLType::Q4_0),
            3 => Some(GGMLType::Q4_1),
            6 => Some(GGMLType::Q5_0),
            7 => Some(GGMLType::Q5_1),
            8 => Some(GGMLType::Q8_0),
            9 => Some(GGMLType::Q8_1),
            10 => Some(GGMLType::Q2_K),
            11 => Some(GGMLType::Q3_K),
            12 => Some(GGMLType::Q4_K),
            13 => Some(GGMLType::Q5_K),
            14 => Some(GGMLType::Q6_K),
            15 => Some(GGMLType::Q8_K),
            16 => Some(GGMLType::IQ2_XXS),
            17 => Some(GGMLType::IQ2_XS),
            18 => Some(GGMLType::IQ3_XXS),
            19 => Some(GGMLType::IQ1_S),
            20 => Some(GGMLType::IQ4_NL),
            21 => Some(GGMLType::IQ3_S),
            22 => Some(GGMLType::IQ2_S),
            23 => Some(GGMLType::IQ4_XS),
            24 => Some(GGMLType::I8),
            25 => Some(GGMLType::I16),
            26 => Some(GGMLType::I32),
            27 => Some(GGMLType::I64),
            28 => Some(GGMLType::F64),
            29 => Some(GGMLType::IQ1_M),
            30 => Some(GGMLType::BF16),
            36 => Some(GGMLType::GGML_TYPE_Q1_58),
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
    pub fn block_layout(&self) -> (usize, usize) {
        match self {
            GGMLType::F32 | GGMLType::I32 => (4, 1),
            GGMLType::F16 | GGMLType::BF16 | GGMLType::I16 => (2, 1),
            GGMLType::F64 | GGMLType::I64 => (8, 1),
            GGMLType::I8 => (1, 1),

            // Legacy Q‑quants (block of 32 weights)
            GGMLType::Q4_0 => (18, 32),
            GGMLType::Q4_1 => (20, 32),
            GGMLType::Q5_0 => (22, 32),
            GGMLType::Q5_1 => (24, 32),
            GGMLType::Q8_0 => (34, 32),
            GGMLType::Q8_1 => (36, 32),

            // K‑quants (super‑block of 256 weights); bytes = bpw * 256 / 8
            GGMLType::Q2_K => (84, 256),  // 2.625  bpw
            GGMLType::Q3_K => (110, 256), // 3.4375 bpw
            GGMLType::Q4_K => (144, 256), // 4.5    bpw
            GGMLType::Q5_K => (176, 256), // 5.5    bpw
            GGMLType::Q6_K => (210, 256), // 6.5625 bpw
            GGMLType::Q8_K => (292, 256), // 9.125  bpw

            // Importance‑quants (IQ‑family, super‑block 256 except IQ4_NL's 32)
            GGMLType::IQ1_S => (50, 256),   // 1.5625 bpw
            GGMLType::IQ1_M => (56, 256),   // 1.75   bpw
            GGMLType::IQ2_XXS => (66, 256), // 2.0625 bpw
            GGMLType::IQ2_XS => (74, 256),  // 2.3125 bpw
            GGMLType::IQ2_S => (80, 256),   // 2.5    bpw
            GGMLType::IQ3_XXS => (98, 256), // 3.0625 bpw
            GGMLType::IQ3_S => (110, 256),  // 3.4375 bpw
            GGMLType::IQ4_NL => (17, 32),   // 4.25   bpw
            GGMLType::IQ4_XS => (136, 256), // 4.25   bpw
            // 1.58 bpw doesn't land on a whole byte per 256, so keep the exact ratio
            // this type has always been measured with (0.1975 B/elem = 79/400).
            GGMLType::GGML_TYPE_Q1_58 => (79, 400),
        }
    }

    /// Exact stored size of `elements` values of this type, rounded up to whole
    /// blocks (see [`GGMLType::block_layout`]).
    pub fn stored_size(&self, elements: usize) -> usize {
        let (bytes, per_block) = self.block_layout();
        elements.div_ceil(per_block.max(1)).saturating_mul(bytes)
    }
}

impl std::fmt::Display for GGMLType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            GGMLType::F32 => "F32",
            GGMLType::F16 => "F16",
            GGMLType::F64 => "F64",
            GGMLType::BF16 => "BF16",
            GGMLType::I8 => "I8",
            GGMLType::I16 => "I16",
            GGMLType::I32 => "I32",
            GGMLType::I64 => "I64",
            GGMLType::Q4_0 => "Q4_0",
            GGMLType::Q4_1 => "Q4_1",
            GGMLType::Q5_0 => "Q5_0",
            GGMLType::Q5_1 => "Q5_1",
            GGMLType::Q8_0 => "Q8_0",
            GGMLType::Q8_1 => "Q8_1",
            GGMLType::Q2_K => "Q2_K",
            GGMLType::Q3_K => "Q3_K",
            GGMLType::Q4_K => "Q4_K",
            GGMLType::Q5_K => "Q5_K",
            GGMLType::Q6_K => "Q6_K",
            GGMLType::Q8_K => "Q8_K",
            GGMLType::IQ2_XXS => "IQ2_XXS",
            GGMLType::IQ2_XS => "IQ2_XS",
            GGMLType::IQ3_XXS => "IQ3_XXS",
            GGMLType::IQ1_S => "IQ1_S",
            GGMLType::IQ4_NL => "IQ4_NL",
            GGMLType::IQ3_S => "IQ3_S",
            GGMLType::IQ2_S => "IQ2_S",
            GGMLType::IQ4_XS => "IQ4_XS",
            GGMLType::IQ1_M => "IQ1_M",
            GGMLType::GGML_TYPE_Q1_58 => "Q1_58",
        };
        write!(f, "{s}")
    }
}

impl std::fmt::Display for GGUFValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GGUFValue::U8(v) => write!(f, "{v}"),
            GGUFValue::I8(v) => write!(f, "{v}"),
            GGUFValue::U16(v) => write!(f, "{v}"),
            GGUFValue::I16(v) => write!(f, "{v}"),
            GGUFValue::U32(v) => write!(f, "{v}"),
            GGUFValue::I32(v) => write!(f, "{v}"),
            GGUFValue::F32(v) => write!(f, "{v}"),
            GGUFValue::U64(v) => write!(f, "{v}"),
            GGUFValue::I64(v) => write!(f, "{v}"),
            GGUFValue::F64(v) => write!(f, "{v}"),
            GGUFValue::Bool(v) => write!(f, "{v}"),
            GGUFValue::String(v) => write!(f, "\"{v}\""),
            GGUFValue::Array(arr) => {
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
                        arr[0],
                        arr[1],
                        arr[arr.len() - 1],
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
        if header.magic != 0x46554747 {
            return Err(anyhow::anyhow!("Invalid GGUF magic number"));
        }

        // Read metadata
        let metadata = Self::read_metadata(&mut cursor, header.metadata_kv_count)?;

        // Read tensor info
        let tensors = Self::read_tensor_info(&mut cursor, header.tensor_count)?;

        Ok(GGUFFile {
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
    /// from HuggingFace, so a half-downloaded one is ordinary input and must produce an
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Sizes were computed as `elements as f32 * bytes_per_element`, which has only 24
    /// mantissa bits and truncated the trailing partial block. Both errors are visible
    /// on realistic tensor sizes, and the result feeds the displayed size, the
    /// checkpoint totals, and the exact `size>N` filter.
    /// A GGUF writer, so the reader can be tested against bytes rather than only
    /// against whatever real file happens to be at hand. Little-endian throughout, as
    /// the format specifies.
    #[derive(Default)]
    struct Gguf {
        body: Vec<u8>,
        tensors: u64,
        kvs: u64,
    }

    impl Gguf {
        fn str(&mut self, s: &str) -> &mut Self {
            self.body.extend((s.len() as u64).to_le_bytes());
            self.body.extend(s.as_bytes());
            self
        }
        fn u32(&mut self, v: u32) -> &mut Self {
            self.body.extend(v.to_le_bytes());
            self
        }
        fn u64(&mut self, v: u64) -> &mut Self {
            self.body.extend(v.to_le_bytes());
            self
        }
        /// One metadata entry: key, value-type tag, then the value's bytes.
        fn kv(&mut self, key: &str, ty: u32, value: &[u8]) -> &mut Self {
            self.kvs += 1;
            self.str(key).u32(ty);
            self.body.extend(value);
            self
        }
        fn tensor(&mut self, name: &str, dims: &[u64], ty: u32, offset: u64) -> &mut Self {
            self.tensors += 1;
            self.str(name).u32(dims.len() as u32);
            for d in dims {
                self.u64(*d);
            }
            self.u32(ty).u64(offset);
            self
        }
        /// Header (magic `GGUF`, version, counts) followed by the body.
        fn finish(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend(0x4655_4747u32.to_le_bytes());
            out.extend(3u32.to_le_bytes());
            out.extend(self.tensors.to_le_bytes());
            out.extend(self.kvs.to_le_bytes());
            out.extend(&self.body);
            out
        }
    }

    fn le_u64(v: u64) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

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
    /// files come from users and from HuggingFace; a half-downloaded one is ordinary.
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
    fn ggml_types_round_trip_their_tags_and_reject_unknown_ones() {
        for tag in [0u32, 1, 8, 10, 12] {
            let ty = GGMLType::from_u32(tag).unwrap_or_else(|| panic!("tag {tag} should be known"));
            // Every known type must describe itself and report a usable block layout.
            let (elems, bytes) = ty.block_layout();
            assert!(elems > 0 && bytes > 0, "{ty:?} has an empty block layout");
            assert!(!format!("{ty}").is_empty());
        }
        assert!(GGMLType::from_u32(9999).is_none());
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
        for t in [
            GGMLType::F32,
            GGMLType::F16,
            GGMLType::F64,
            GGMLType::BF16,
            GGMLType::I8,
            GGMLType::I16,
            GGMLType::I32,
            GGMLType::I64,
            GGMLType::Q4_0,
            GGMLType::Q4_1,
            GGMLType::Q5_0,
            GGMLType::Q5_1,
            GGMLType::Q8_0,
            GGMLType::Q8_1,
            GGMLType::Q2_K,
            GGMLType::Q3_K,
            GGMLType::Q4_K,
            GGMLType::Q5_K,
            GGMLType::Q6_K,
            GGMLType::Q8_K,
            GGMLType::IQ1_S,
            GGMLType::IQ1_M,
            GGMLType::IQ2_XXS,
            GGMLType::IQ2_XS,
            GGMLType::IQ2_S,
            GGMLType::IQ3_XXS,
            GGMLType::IQ3_S,
            GGMLType::IQ4_NL,
            GGMLType::IQ4_XS,
            GGMLType::GGML_TYPE_Q1_58,
        ] {
            let (bytes, per_block) = t.block_layout();
            assert!(bytes > 0 && per_block > 0, "{t} has an empty block layout");
            assert!(t.stored_size(1) > 0, "{t} sized one element as 0 bytes");
        }
    }
}
