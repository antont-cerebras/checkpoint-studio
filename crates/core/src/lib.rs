//! Frontend-free core of checkpoint-studio.
//!
//! Holds the serializable checkpoint model, the readers (local + SSH/S3) that
//! fill it, the derived views (tensor tree, file tree, byte layout), and the
//! reports (stats, health check, diff) — with **no** terminal / TUI / CLI
//! dependency. Frontends (the interactive terminal, and future web-server / MCP
//! bins) drive it and render its serializable outputs.

// An unwrap in a test IS the assertion — the panic is the failure report, and rewriting
// hundreds of them into `?` would make the tests harder to read for no gain. So
// `unwrap_used`/`expect_used` (denied for product code in Cargo.toml) are allowed in test
// builds only.
//
// `float_cmp` likewise: a test that computes a value and asserts it *exactly* is checking
// the arithmetic, which is the whole point (`assert_eq!(stats.max, 3.5)`). An epsilon there
// would weaken the test to hide a lint. Product code that needs an exact float comparison
// still has to say so at the site, with a reason.
#![cfg_attr(
    test,
    allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)
)]

// This tool memory-maps multi-gigabyte checkpoints and converts 64-bit header offsets
// and element counts to `usize` throughout. That is only sound on a 64-bit target, so
// state it as a compile-time requirement instead of leaving it implied.
const _: () = assert!(
    usize::BITS >= 64,
    "checkpoint-studio requires a 64-bit target: file offsets and element counts are \
     converted to usize"
);

pub mod check;
pub mod codec;
pub mod config;
pub mod diff;
pub mod filetree;
pub mod filter;
pub mod gguf;
pub mod health;
pub mod kernel;
pub mod model;
pub mod npy;
pub mod progress;
pub mod readers;
pub mod remote;
pub mod rename;
pub mod repack;
pub mod s3;
pub mod safelayout;
pub mod sample;
pub mod sftp;
pub mod stats;
pub mod stheader;
pub mod tensorfilter;
pub mod tree;
pub mod utils;
pub mod viewstate;

#[cfg(feature = "hdf5")]
pub mod convert;
#[cfg(feature = "hdf5")]
pub mod hdf5;
#[cfg(feature = "hdf5")]
pub mod hdf5_filter;
#[cfg(feature = "hdf5")]
pub mod hdf5_lz4;
#[cfg(feature = "hdf5")]
pub mod hdf5_zstd;
