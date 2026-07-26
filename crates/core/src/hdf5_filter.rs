//! The one piece the in-process HDF5 filters share: handing a freshly compressed or
//! decompressed buffer back to libhdf5.
//!
//! [`crate::hdf5_lz4`] and [`crate::hdf5_zstd`] are otherwise independent — different
//! framing, different registry ids, one decode-only and one both ways — but they meet
//! libhdf5 through the same buffer-ownership contract.

use std::os::raw::c_void;

use hdf5_metno_sys::h5::{H5allocate_memory, H5free_memory};

/// Replace `*buf` with a fresh, HDF5-owned buffer holding `out`; returns its length
/// (0 on allocation failure, which is how a filter callback reports one).
///
/// Shared by the LZ4 and zstd filters: libhdf5's filter contract is that the callback
/// frees the buffer it was given and hands back one allocated with `H5allocate_memory`,
/// which is the same six unsafe lines whatever the codec — and not something to keep two
/// copies of, since getting it wrong corrupts the heap rather than failing a test.
///
/// # Safety
///
/// `buf` and `buf_size` must be the pointers libhdf5 passed to the filter callback:
/// `*buf` an `H5allocate_memory` allocation it owns, `buf_size` its size. On success
/// ownership of the old allocation is released and `*buf` points at the new one.
pub(crate) unsafe fn install_output(
    buf: *mut *mut c_void,
    buf_size: *mut usize,
    out: &[u8],
) -> usize {
    unsafe {
        let p = H5allocate_memory(out.len().max(1), 0) as *mut u8;
        if p.is_null() {
            return 0;
        }
        std::ptr::copy_nonoverlapping(out.as_ptr(), p, out.len());
        H5free_memory(*buf);
        *buf = p as *mut c_void;
        *buf_size = out.len();
        out.len()
    }
}
