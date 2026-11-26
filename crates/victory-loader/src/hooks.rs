//! Opt-in hooks in the code of known title builds.
//!
//! Only per-build offsets and short byte signatures live here (asset
//! policy: nothing of the title itself). Every hook checks the build's hash
//! and the exact bytes it replaces, and does nothing otherwise.

use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// First 16 hex digits of the BLAKE3 of the Series X 1.0.2.1 executable.
const BUILD_1_0_2_1: &str = "b99f876b74d4a77f";

/// The switch that turns the watchdog hooks on.
pub(crate) const NO_WATCHDOG_ENV: &str = "VICTORY_NO_WATCHDOG";

/// What a patched place must hold before anything is written there.
pub(crate) enum Expect {
    /// These exact bytes (a short stretch: the policy allows signatures).
    Bytes(&'static [u8]),
    /// A longer stretch, kept as its length and the BLAKE3 of its bytes
    /// (64 hex digits), so none of the title's code is stored here.
    Hash { len: usize, blake3: &'static str },
}

impl Expect {
    /// How many bytes are compared.
    pub(crate) fn len(&self) -> usize {
        match self {
            Expect::Bytes(b) => b.len(),
            Expect::Hash { len, .. } => *len,
        }
    }

    /// Whether `now` is the stretch that was expected.
    pub(crate) fn holds(&self, now: &[u8]) -> bool {
        match self {
            Expect::Bytes(b) => now == *b,
            Expect::Hash { len, blake3 } => now.len() == *len && blake3::hash(now).to_hex().as_str() == *blake3,
        }
    }
}

/// One code patch: `rva` must hold exactly `expect` before `new` is written.
struct Patch {
    what: &'static str,
    rva: usize,
    expect: Expect,
    new: &'static [u8],
}

/// The title's watchdog thread ends the process on purpose when its main
/// thread has not advanced for about 30 one-second ticks (a world frame or a
/// pipeline wait of 30 s here is enough). Two places do it: the handler it
/// calls (`mov dword ptr [0], imm32`, the fault at `+0x23cf683`) and, when
/// the handler returns or is unset, a fallback right after the call (`mov
/// dword ptr [0], 0; int3`). Jumping over the whole block returns the loop
/// to its normal path; the second patch makes the handler harmless if
/// something else calls it. Stopping the watchdog thread itself is wrong:
/// meeyao/VIctory tried it, the kills stopped and the title never drew (that
/// thread does work the title needs before it draws).
const WATCHDOG: &[Patch] = &[
    Patch {
        what: "watchdog: skip the call and the fallback fault",
        rva: 0x2215c27,
        // mov rax, [rip + disp32] (the handler pointer)
        expect: Expect::Bytes(&[0x48, 0x8B, 0x05, 0x1A, 0x07, 0xFA, 0x04]),
        // jmp +0x18 (to the loop's normal continuation), then NOPs
        new: &[0xEB, 0x18, 0x90, 0x90, 0x90, 0x90, 0x90],
    },
    Patch {
        what: "watchdog: the handler's deliberate fault",
        rva: 0x23cf683,
        expect: Expect::Bytes(&[0xC7, 0x04, 0x25, 0, 0, 0, 0, 0x21, 0xD4, 0x8B, 0x4D]),
        // an eleven-byte NOP
        new: &[0x66, 0x66, 0x2E, 0x0F, 0x1F, 0x84, 0, 0, 0, 0, 0],
    },
];

/// Applies the watchdog patches whose bytes match; how many applied. None
/// on another build.
///
/// # Safety
/// `base` is the mapped image whose original BLAKE3 is `hash`, its `.text`
/// covers the patches' offsets on that build, and no title code runs yet.
pub(crate) unsafe fn disarm_watchdog(base: usize, hash: &blake3::Hash) -> usize {
    if !is_known_build(hash) {
        tracing::warn!("title watchdog hooks: not the build they are for; nothing patched");
        return 0;
    }
    WATCHDOG
        .iter()
        // SAFETY: forwarded from this function's contract.
        .filter(|p| unsafe { apply(base, p) })
        .count()
}

/// Whether `hash` is the build the offsets in this crate's hooks are for.
pub(crate) fn is_known_build(hash: &blake3::Hash) -> bool {
    hash.to_hex().starts_with(BUILD_1_0_2_1)
}

/// # Safety
/// See [`disarm_watchdog`].
unsafe fn apply(base: usize, p: &Patch) -> bool {
    // SAFETY: forwarded from this function's contract.
    unsafe { apply_code(base, p.what, p.rva, &p.expect, p.new) }
}

/// Writes `new` at `rva` when the bytes there are the `expect`ed ones and
/// `new` is as long; says so in the log if not. Whether it was written.
///
/// # Safety
/// See [`disarm_watchdog`]; the build is a known one.
pub(crate) unsafe fn apply_code(base: usize, what: &str, rva: usize, expect: &Expect, new: &[u8]) -> bool {
    let at = (base + rva) as *mut u8;
    // SAFETY: the offset is inside the mapped .text of this exact build.
    let now = unsafe { std::slice::from_raw_parts(at, expect.len()) };
    if new.len() != expect.len() || !expect.holds(now) {
        tracing::warn!(what, "hook skipped: the bytes are not the expected ones");
        return false;
    }
    let mut old = 0;
    // SAFETY: changes the protection of those bytes only, restored below.
    if unsafe { VirtualProtect(at as _, new.len(), PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
        return false;
    }
    // SAFETY: writable now; the bytes are the ones checked above.
    unsafe {
        std::ptr::copy_nonoverlapping(new.as_ptr(), at, new.len());
        VirtualProtect(at as _, new.len(), old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), at as _, new.len());
    }
    tracing::info!(what, "hook applied");
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each patch replaces as many bytes as it checks, and the hooks touch
    /// nothing on another build.
    #[test]
    fn patches_are_their_own_size_and_keyed_by_build() {
        assert!(WATCHDOG.iter().all(|p| p.new.len() == p.expect.len()));
        let other = blake3::hash(b"not the build");
        // SAFETY: another build's hash: nothing is read or written.
        assert_eq!(unsafe { disarm_watchdog(0, &other) }, 0);
        assert!(!is_known_build(&other));
    }

    /// A stretch kept as a hash is found by its length and its BLAKE3, and
    /// by nothing else; a patch over it is written once and not over other
    /// bytes. Synthetic bytes in memory of our own.
    #[test]
    fn a_hashed_stretch_is_checked_and_patched() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, VirtualAlloc, VirtualFree};
        let stretch: Vec<u8> = (0..40u8).map(|i| i.wrapping_mul(7).wrapping_add(3)).collect();
        let hex = blake3::hash(&stretch).to_hex().to_string();
        let expect = Expect::Hash {
            len: 40,
            blake3: Box::leak(hex.into_boxed_str()),
        };
        assert_eq!(expect.len(), 40);
        assert!(expect.holds(&stretch));
        let mut other = stretch.clone();
        other[17] ^= 1;
        assert!(!expect.holds(&other));
        assert!(!expect.holds(&stretch[..39]));
        let bytes = Expect::Bytes(&[1, 2, 3]);
        assert!(bytes.holds(&[1, 2, 3]) && !bytes.holds(&[1, 2, 4]) && !bytes.holds(&[1, 2]));

        // SAFETY: fresh read-write memory of our own, freed below.
        unsafe {
            let place = VirtualAlloc(std::ptr::null(), 0x1000, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE) as *mut u8;
            assert!(!place.is_null());
            let base = place as usize;
            std::ptr::copy_nonoverlapping(stretch.as_ptr(), place.add(0x100), 40);
            let new = [0x90u8; 40];
            assert!(!apply_code(base, "test", 0x100, &expect, &new[..39]), "length differs");
            assert!(apply_code(base, "test", 0x100, &expect, &new));
            assert_eq!(std::slice::from_raw_parts(place.add(0x100), 40), new);
            assert!(!apply_code(base, "test", 0x100, &expect, &new), "already changed");
            VirtualFree(place.cast(), 0, MEM_RELEASE);
        }
    }
}
