//! The title's waits and wake-ups in the frame timeline (diagnostics builds,
//! and only with `VICTORY_TIMELINE`, see `victory_gpu::timeline`): which
//! title thread waits for which handle, and which thread sets it. Each call
//! goes to the system's function unchanged.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;

use victory_gpu::timeline::ev;
use windows_sys::Win32::System::Threading as host;

use crate::{HleModule, hle_export};

pub extern "system" fn WaitForSingleObject(handle: *mut c_void, ms: u32) -> u32 {
    ev("wait>", handle as u64, u64::from(ms));
    // SAFETY: the title's own call, passed on.
    let r = unsafe { host::WaitForSingleObject(handle, ms) };
    ev("wait<", handle as u64, u64::from(r));
    r
}

pub extern "system" fn WaitForSingleObjectEx(handle: *mut c_void, ms: u32, alertable: i32) -> u32 {
    ev("wait>", handle as u64, u64::from(ms));
    // SAFETY: the title's own call, passed on.
    let r = unsafe { host::WaitForSingleObjectEx(handle, ms, alertable) };
    ev("wait<", handle as u64, u64::from(r));
    r
}

pub extern "system" fn WaitForMultipleObjects(count: u32, handles: *const *mut c_void, all: i32, ms: u32) -> u32 {
    // SAFETY: the title passes `count` handles.
    let first = if count > 0 && !handles.is_null() { unsafe { *handles } } else { std::ptr::null_mut() };
    ev("waitm>", first as u64, u64::from(count));
    // SAFETY: the title's own call, passed on.
    let r = unsafe { host::WaitForMultipleObjects(count, handles, all, ms) };
    ev("waitm<", first as u64, u64::from(r));
    r
}

pub extern "system" fn WaitForMultipleObjectsEx(
    count: u32,
    handles: *const *mut c_void,
    all: i32,
    ms: u32,
    alertable: i32,
) -> u32 {
    // SAFETY: the title passes `count` handles.
    let first = if count > 0 && !handles.is_null() { unsafe { *handles } } else { std::ptr::null_mut() };
    ev("waitm>", first as u64, u64::from(count));
    // SAFETY: the title's own call, passed on.
    let r = unsafe { host::WaitForMultipleObjectsEx(count, handles, all, ms, alertable) };
    ev("waitm<", first as u64, u64::from(r));
    r
}

pub extern "system" fn SetEvent(handle: *mut c_void) -> i32 {
    ev("set", handle as u64, 0);
    // SAFETY: the title's own call, passed on.
    unsafe { host::SetEvent(handle) }
}

pub extern "system" fn ReleaseSemaphore(handle: *mut c_void, count: i32, previous: *mut i32) -> i32 {
    ev("sem", handle as u64, count as u64);
    // SAFETY: the title's own call, passed on.
    unsafe { host::ReleaseSemaphore(handle, count, previous) }
}

pub(crate) fn module() -> Option<HleModule> {
    std::env::var_os("VICTORY_TIMELINE")?;
    Some(HleModule {
        name: "waits",
        dlls: &["kernel32.dll", "kernelbase.dll", "api-ms-win-core-synch-*"],
        exports: vec![
            hle_export!(WaitForSingleObject),
            hle_export!(WaitForSingleObjectEx),
            hle_export!(WaitForMultipleObjects),
            hle_export!(WaitForMultipleObjectsEx),
            hle_export!(SetEvent),
            hle_export!(ReleaseSemaphore),
        ],
    })
}
