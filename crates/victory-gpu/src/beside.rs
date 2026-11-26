//! What shaders wrote lands in title memory beside the guards
//! (`VICTORY_LAND_BESIDE=1`; off by default).
//!
//! A landing writes the bytes shaders changed into title memory at the
//! title's own address. Where that is a view of physical pages the blocks
//! are guarded, read-only, so the landing opened them first
//! (`crate::watch::writing_as`): a change of protection for each, and one
//! more to guard the block again at its next check, on the submitting
//! thread. In the first room every one of the 781 guarded blocks opened a
//! frame was opened by a landing (the census of 2026-11-26: "guarded blocks
//! opened by the layer's own writes"), none by the title; the profile of
//! that day had 3.8 ms a frame of the landing thread in opening them and
//! 8.3 ms of the submitting thread in the guards, 4.6 of it changing
//! protection.
//!
//! With the switch the landing writes through a view of the layer's own of
//! the same section, which no guard covers: the same physical pages, so the
//! title's views show the bytes, and they stay read-only. The blocks are
//! counted written where they are (`crate::watch::written_beside`, as a
//! write through the runtime's own view is), so every host copy over them
//! takes them again at its next check, as after an opening; a later write
//! of the title's faults as it would have. Memory that is not such a view
//! (private memory, which the kernel watches), a range no one view holds,
//! and a page that is not there go the old way.
//!
//! Off by default because a view of ours that showed other pages than the
//! title's would put the bytes where the title never reads them: `=2`
//! reads every such write back through the title's address and counts the
//! ones that are not there (and writes those the old way).
//!
//! The idea is avatarmanz/VIctory-7900XTX's landing beside the guards
//! (0d5e4c8cb) and Marvisak/VIctory pull request 3069; the code is this
//! tree's own, on its guards' count of blocks written while read-only.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// 0: landings open the guards, as before; 1: they write beside them; 2:
/// and every such write is read back through the title's address.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_LAND_BESIDE").as_deref().map(str::trim) {
        Ok("1") => 1,
        Ok("2") => 2,
        _ => 0,
    })
}

/// Our own view of each section a landing wrote into, by the section's
/// handle: where it is and its bytes; `None`: it could not be mapped.
static OWN: Mutex<Option<HashMap<usize, Option<(usize, u64)>>>> = Mutex::new(None);

/// Since the last frame line: writes that went beside the guards, and
/// their bytes; pieces left to the old way because their address is no
/// view of a section, because no one view of ours holds them; writes with
/// a page that was not there; writes read back, and those of them that
/// were not at the title's address.
static STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// The runtime made a section with this handle: a view of ours of an
/// earlier section that had the handle is no view of this one.
pub(crate) fn section_made(section: usize) {
    use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
    let old = OWN.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|own| own.remove(&section)).flatten();
    if let Some((at, _)) = old {
        // SAFETY: a view this module mapped; a landing writing through it
        // just now survives the page going (`crate::watch::copy_title`).
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: at as *mut _ }) };
    }
}

/// Our view of `section`, mapped whole the first time it is asked for.
fn own(section: usize) -> Option<(usize, u64)> {
    use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
    let mut own = OWN.lock().unwrap_or_else(|e| e.into_inner());
    *own.get_or_insert_with(HashMap::new).entry(section).or_insert_with(|| {
        let bytes = crate::inplace::section_bytes(section)?;
        // SAFETY: a section handle of the runtime's; the whole section, at
        // an address the system picks.
        let at = unsafe { MapViewOfFile(section as _, FILE_MAP_WRITE, 0, 0, 0) }.Value as usize;
        if at == 0 {
            tracing::warn!(
                section = format_args!("{section:#x}"),
                bytes,
                error = %std::io::Error::last_os_error(),
                "no view of our own of a section of the title's physical pages: landings there open the guards (VICTORY_LAND_BESIDE)"
            );
            return None;
        }
        Some((at, bytes))
    })
}

/// Where the first bytes of title memory `[at, at + len)` lie in a view of
/// our own, and how many of them one title view holds; why not, as a place
/// in [`STATS`].
fn place(at: usize, len: usize) -> Result<(usize, usize), usize> {
    let (base, bytes, section, offset) = crate::inplace::view_at(at).ok_or(2usize)?;
    let n = len.min(base.saturating_add(bytes).saturating_sub(at));
    let (own, held) = own(section).ok_or(3usize)?;
    let from = offset.checked_add((at - base) as u64).ok_or(3usize)?;
    if n == 0 || from.checked_add(n as u64).is_none_or(|end| end > held) {
        return Err(3);
    }
    Ok((own.checked_add(from as usize).ok_or(3usize)?, n))
}

/// A stretch of title memory one title view holds, and where it lies in a
/// view of our own: looked up once for the several writes of a landing
/// into it ([`find`]).
pub(crate) struct Place {
    title: usize,
    ours: usize,
    len: usize,
}

/// Title memory `[at, at + len)` in a view of our own, if one title view
/// holds all of it and a view of ours shows it; counted for the frame line
/// if not (the caller then writes as before).
pub(crate) fn find(at: usize, len: usize) -> Option<Place> {
    match place(at, len) {
        Ok((ours, held)) if held >= len => Some(Place { title: at, ours, len }),
        // Two title views, each with its own place: left to the old way.
        Ok(_) => {
            STATS[3].fetch_add(1, Relaxed);
            None
        }
        Err(why) => {
            STATS[why].fetch_add(1, Relaxed);
            None
        }
    }
}

impl Place {
    /// Writes `bytes` to title memory at `at` through our view: how many
    /// bytes from the first went that way (all of them, unless a page is
    /// not there or they reach past the stretch; none for bytes that begin
    /// outside it). The rest is the caller's to write as before; the
    /// caller counts what went here as written for the guards
    /// (`crate::watch::written_beside`).
    pub(crate) fn write(&self, at: usize, bytes: &[u8]) -> usize {
        match at.checked_sub(self.title) {
            Some(off) if off < self.len => {
                let n = bytes.len().min(self.len - off);
                put(self.ours + off, at, &bytes[..n])
            }
            _ => 0,
        }
    }
}

/// `part` into our view at `to`, which shows title memory at `at`: how
/// many bytes went there (all of them or, where a page is not there or the
/// check does not find them, fewer).
fn put(to: usize, at: usize, part: &[u8]) -> usize {
    let n = part.len();
    // SAFETY: inside our view of the section, which nothing of ours reads;
    // the source is the caller's buffer, apart from it; the copy survives
    // a page that is not there.
    let done = unsafe { crate::watch::copy_title(to as *mut u8, part.as_ptr(), n) };
    if done < n {
        // (Up to 64 bytes past it may be written: the caller writes them
        // again.)
        STATS[4].fetch_add(1, Relaxed);
        return done;
    }
    if mode() == 2 && !there(at, part) {
        return 0;
    }
    STATS[0].fetch_add(1, Relaxed);
    STATS[1].fetch_add(n as u64, Relaxed);
    n
}

/// Whether title memory at `at` holds `bytes` (the check of
/// `VICTORY_LAND_BESIDE=2`); counted, and said the first times it does not.
fn there(at: usize, bytes: &[u8]) -> bool {
    let mut now = vec![0u8; bytes.len()];
    STATS[5].fetch_add(1, Relaxed);
    if crate::host::read_into(at as *const u8, &mut now) && now == bytes {
        return true;
    }
    static SAID: AtomicU64 = AtomicU64::new(0);
    STATS[6].fetch_add(1, Relaxed);
    if SAID.fetch_add(1, Relaxed) < 8 {
        tracing::warn!(
            at = format_args!("{at:#x}"),
            len = bytes.len(),
            "a landing beside the guards is not at the title's address (or the title wrote there at once): written the old way (VICTORY_LAND_BESIDE=2)"
        );
    }
    false
}

/// With the frame line.
pub(crate) fn line(frames: f64) {
    if mode() == 0 {
        return;
    }
    let [writes, bytes, no_view, none_of_ours, not_there, checked, missed] = [0, 1, 2, 3, 4, 5, 6].map(|i| STATS[i].swap(0, Relaxed));
    tracing::info!(
        "landings beside the guards per frame (VICTORY_LAND_BESIDE={}): {:.0} writes, {:.2} MB, through views of our own; pieces \
         left to the old way: {:.0} at addresses that are no view of a section (private memory), {:.1} that no one view of ours \
         holds; {:.1} writes with a page not there; read back since the last line: {checked}, not at the title's address: {missed}",
        mode(),
        writes as f64 / frames,
        bytes as f64 / 1e6 / frames,
        no_view as f64 / frames,
        none_of_ours as f64 / frames,
        not_there as f64 / frames,
    );
}

#[cfg(test)]
mod tests {
    use super::{find, place, section_made};
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READONLY, PAGE_READWRITE,
        UnmapViewOfFile, VirtualProtect,
    };

    const K64: usize = 64 << 10;

    /// A write beside lands in the pages a read-only title view shows, at
    /// the view's offset in its section; past the view, and in memory that
    /// is no view, nothing is written and the caller is told how far it
    /// went.
    #[test]
    fn a_write_beside_shows_at_the_title_view_and_leaves_it_read_only() {
        // SAFETY: a section and views of the test's own, let go at the end.
        unsafe {
            let section = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, (4 * K64) as u32, std::ptr::null());
            assert!(!section.is_null());
            let handle = section as usize;
            // The "title's" view: the section's second and third 64 KiB.
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, K64 as u32, 2 * K64).Value as usize;
            assert_ne!(view, 0);
            crate::inplace::title_memory_section(handle, (4 * K64) as u64, true);
            crate::inplace::title_memory_view(view, 2 * K64, handle, K64 as u64);
            let title = |at: usize, n: usize| std::slice::from_raw_parts((view + at) as *const u8, n).to_vec();
            (view as *mut u8).add(0x1000).write_bytes(0x11, 0x40);
            let mut old = 0;
            assert_ne!(VirtualProtect(view as *const _, 2 * K64, PAGE_READONLY, &mut old), 0);
            let write = |at: usize, bytes: &[u8]| find(at, bytes.len()).map_or(0, |found| found.write(at, bytes));
            // Inside the view: all of it, and the view stays read-only.
            assert_eq!(write(view + 0x1010, &[0xa5; 0x20]), 0x20);
            assert_eq!(title(0x1000, 0x40), [vec![0x11; 0x10], vec![0xa5; 0x20], vec![0x11; 0x10]].concat());
            let mut now = 0;
            assert_ne!(VirtualProtect(view as *const _, 2 * K64, PAGE_READONLY, &mut now), 0);
            assert_eq!(now, PAGE_READONLY, "no protection was changed");
            // The view's first byte is the section's 64 KiB, not its start.
            let (ours, n) = place(view, 16).expect("a place in our view");
            assert_eq!((ours as *const u8).add(0x1010).read(), 0xa5);
            assert_eq!(n, 16);
            // A place looked up once: writes inside it, up to its end, and
            // none that begins outside it.
            let found = find(view + 0x3000, 0x100).expect("a place");
            assert_eq!(found.write(view + 0x3010, &[9; 8]), 8);
            assert_eq!(found.write(view + 0x30fc, &[9; 8]), 4);
            assert_eq!(found.write(view + 0x3100, &[9; 8]), 0);
            assert_eq!(found.write(view + 0x2ff8, &[9; 8]), 0);
            assert_eq!(title(0x3010, 8), [9; 8]);
            assert_eq!(title(0x30fc, 8), [9, 9, 9, 9, 0, 0, 0, 0]);
            // Across the view's end: no place, and nothing written.
            let before = title(2 * K64 - 8, 8);
            assert!(find(view + 2 * K64 - 8, 32).is_none());
            assert_eq!(title(2 * K64 - 8, 8), before);
            // Memory that is no view of a section: nothing.
            let mut private = vec![0u8; 64];
            assert_eq!(write(private.as_mut_ptr() as usize, &[1; 64]), 0);
            assert_eq!(private, [0u8; 64]);
            // The handle given to a section made later: the old view of
            // ours is let go, and the next write maps the new one.
            section_made(handle);
            assert_eq!(write(view + 0x2000, &[0x77; 8]), 8);
            assert_eq!(title(0x2000, 8), [0x77; 8]);
            section_made(handle);
            crate::title_memory_remapped(view, 2 * K64);
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ });
            CloseHandle(section);
        }
    }
}
