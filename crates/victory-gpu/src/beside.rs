//! What shaders wrote lands in title memory beside the guards
//! (`VICTORY_LAND_BESIDE`, on; `=0`: the guards are opened, as before;
//! `=2`: on, and every such write is read back).
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
//! Now the landing writes through a view of the layer's own of
//! the same section, which no guard covers: the same physical pages, so the
//! title's views show the bytes, and they stay read-only. The blocks are
//! counted written where they are (`crate::watch::written_beside`, as a
//! write through the runtime's own view is), so every host copy over them
//! takes them again at its next check, as after an opening; a later write
//! of the title's faults as it would have. Memory that is not such a view
//! (private memory, which the kernel watches) and a page that is not there
//! go the old way. A range that lies across several of the title's views
//! goes beside view by view ([`Place::more`]): until 2026-11-27 it went the
//! old way, and those were the blocks landings still opened (the first
//! room: 135 such pieces a frame, 360 guarded blocks opened by landings and
//! guarded again by the next check, of about 1,000 changes of protection a
//! frame on the submitting thread).
//!
//! A view of ours that showed other pages than the title's would put the
//! bytes where the title never reads them: `=2` reads every such write
//! back through the title's address and counts the ones that are not
//! there (and writes those the old way). Measured on 2026-11-26 with that
//! check, the prologue's first room and the open world's bedroom: 2,580
//! and 3,856 writes a frame, none not there in 187 and 138 frame lines;
//! the first room 75.5 to 68.7 ms a frame (preparing 55.1 to 42.2, copying
//! back 36.0 to 29.8), the faces clean. On since then.
//!
//! What the check does not see is the other half, the count for the
//! guards: a block whose bytes landed and that no later check reports
//! written leaves every other host copy over those pages with the bytes
//! from before, and reading the title's address back says nothing of it.
//! That half is the guards' own (`mark_with_views`, in use for writes
//! through the runtime's view since before this) and has its test in
//! `crate::watch`.
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
        Ok("0") => 0,
        Ok("2") => 2,
        _ => 1,
    })
}

/// Our own view of each section a landing wrote into, by the section's
/// handle: where it is and its bytes; `None`: it could not be mapped.
static OWN: Mutex<Option<HashMap<usize, Option<(usize, u64)>>>> = Mutex::new(None);

/// Since the last frame line: writes that went beside the guards, and
/// their bytes; pieces left to the old way because their address is no
/// view of a section, because views of ours do not hold all of them; writes
/// with a page that was not there; writes read back, and those of them that
/// were not at the title's address; ranges that went beside across more
/// than one of the title's views.
static STATS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];

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

/// A stretch of title memory and where it lies in views of our own:
/// looked up once for the several writes of a landing into it ([`find`]).
pub(crate) struct Place {
    title: usize,
    ours: usize,
    /// The bytes from `title` on that its first title view holds.
    len: usize,
    /// What follows in further title views, in order and without a gap:
    /// title address, where in a view of ours, bytes. Empty for a stretch
    /// one title view holds, which most are.
    more: Vec<(usize, usize, usize)>,
}

/// The place of `[at, at + len)`, whose first `held` bytes (fewer than
/// `len`) lie at `ours`: the rest view by view. `None`, counted, if a part
/// of it is in no view of a section or in none of ours.
fn across(at: usize, len: usize, ours: usize, held: usize) -> Option<Place> {
    let mut more = Vec::new();
    let mut done = held;
    while done < len {
        match place(at + done, len - done) {
            Ok((ours, n)) if n > 0 => {
                more.push((at + done, ours, n));
                done += n;
            }
            _ => {
                STATS[3].fetch_add(1, Relaxed);
                return None;
            }
        }
    }
    STATS[7].fetch_add(1, Relaxed);
    Some(Place { title: at, ours, len: held, more })
}

/// Title memory `[at, at + len)` in views of our own, if every byte of it
/// lies in a title view that a view of ours shows; counted for the frame
/// line if not (the caller then writes as before).
pub(crate) fn find(at: usize, len: usize) -> Option<Place> {
    match place(at, len) {
        Ok((ours, held)) if held >= len => Some(Place { title: at, ours, len, more: Vec::new() }),
        // Several title views, each with its own place.
        Ok((ours, held)) => across(at, len, ours, held),
        Err(why) => {
            STATS[why].fetch_add(1, Relaxed);
            None
        }
    }
}

/// The places of one landed piece's stretches: one for all of them where
/// one title view holds the piece, else one for each stretch.
pub(crate) struct Places {
    whole: Option<Place>,
    each: Vec<Place>,
}

impl Places {
    /// The place of the piece's `k`-th stretch.
    pub(crate) fn of(&self, k: usize) -> Option<&Place> {
        self.whole.as_ref().or_else(|| self.each.get(k))
    }
}

/// [`find`] for a piece `[at, at + len)` that is written as `stretches`
/// (address, bytes; inside the piece): if no one title view holds the
/// piece, each stretch by itself. A piece of a copy is 64 KiB from where
/// the copy begins, the title maps its physical pages 64 KiB at a time
/// from where they begin, and so a piece of most copies lies across two
/// of the title's views. `None`, counted for the frame line, if the piece
/// begins outside every view (private memory, which has no guard to open)
/// or a part of a stretch lies in no view of ours.
pub(crate) fn find_all(at: usize, len: usize, stretches: &[(usize, usize)]) -> Option<Places> {
    match place(at, len) {
        Ok((ours, held)) if held >= len => {
            return Some(Places { whole: Some(Place { title: at, ours, len, more: Vec::new() }), each: Vec::new() });
        }
        Ok(_) => {}
        Err(why) => {
            STATS[why].fetch_add(1, Relaxed);
            return None;
        }
    }
    let mut each = Vec::with_capacity(stretches.len());
    for &(at, len) in stretches {
        each.push(match place(at, len) {
            Ok((ours, held)) if held >= len => Place { title: at, ours, len, more: Vec::new() },
            // The stretch itself lies across views.
            Ok((ours, held)) => across(at, len, ours, held)?,
            Err(_) => {
                STATS[3].fetch_add(1, Relaxed);
                return None;
            }
        });
    }
    Some(Places { whole: None, each })
}

impl Place {
    /// Writes `bytes` to title memory at `at` through our view: how many
    /// bytes from the first went that way (all of them, unless a page is
    /// not there or they reach past the stretch; none for bytes that begin
    /// outside it). The rest is the caller's to write as before; the
    /// caller counts what went here as written for the guards
    /// (`crate::watch::written_beside`).
    pub(crate) fn write(&self, at: usize, bytes: &[u8]) -> usize {
        let mut done = 0;
        // View by view, from the first byte on: what a view does not take
        // (a page not there, the check's doubt) ends it.
        for &(title, ours, len) in std::iter::once(&(self.title, self.ours, self.len)).chain(&self.more) {
            if done == bytes.len() {
                break;
            }
            let Some(off) = (at + done).checked_sub(title).filter(|&off| off < len) else {
                // Before this view (a write that begins outside the
                // place), or all of it behind: the next one.
                if at + done < title {
                    break;
                }
                continue;
            };
            let n = (bytes.len() - done).min(len - off);
            let put = put(ours + off, at + done, &bytes[done..done + n]);
            done += put;
            if put < n {
                break;
            }
        }
        done
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
    let [writes, bytes, no_view, none_of_ours, not_there, checked, missed, several] =
        [0, 1, 2, 3, 4, 5, 6, 7].map(|i| STATS[i].swap(0, Relaxed));
    tracing::info!(
        "landings beside the guards per frame (VICTORY_LAND_BESIDE={}): {:.0} writes, {:.2} MB, through views of our own, {:.1} \
         ranges of them across more than one of the title's views; pieces left to the old way: {:.0} at addresses that are no \
         view of a section (private memory: no guard there), {:.1} with a part no view of ours holds; {:.1} writes with a page \
         not there; read back since the last line: {checked}, not at the title's address: {missed}",
        mode(),
        writes as f64 / frames,
        bytes as f64 / 1e6 / frames,
        several as f64 / frames,
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

    /// An address in no view of a section.
    fn private_at() -> usize {
        static PRIVATE: [u8; 64] = [0; 64];
        PRIVATE.as_ptr() as usize
    }

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
            // A piece across two title views: each stretch has its place,
            // unless one lies across the two itself. (Two views side by
            // side in the table at addresses nothing is mapped at: a title
            // address is only looked up, the bytes go through our view.
            // The first shows the section's first 64 KiB, the second its
            // third, which the real view above shows from 64 KiB on.)
            const SIDE: usize = 0x7_c000_0000;
            crate::inplace::title_memory_view(SIDE, K64, handle, 0);
            crate::inplace::title_memory_view(SIDE + K64, K64, handle, 2 * K64 as u64);
            let piece = SIDE + K64 - 0x100;
            // A range across the two: one place, written view by view (the
            // bytes behind the first view's end go into the section's
            // third 64 KiB, which the real view shows from 64 KiB on).
            let both = find(piece + 0xc0, 0x80).expect("a place across the two views");
            let bytes: Vec<u8> = (0..0x80u8).collect();
            assert_eq!(both.write(piece + 0xc0, &bytes), 0x80);
            let (end_of_first, _) = place(SIDE + K64 - 0x40, 0x40).expect("a place in our view");
            assert_eq!(std::slice::from_raw_parts(end_of_first as *const u8, 0x40), &bytes[..0x40]);
            assert_eq!(title(K64, 0x40), &bytes[0x40..]);
            // A write into it that begins in its second view, and one that
            // begins before it.
            assert_eq!(both.write(piece + 0x110, &[8; 0x10]), 0x10);
            assert_eq!(title(K64 + 0x10, 0x10), [8; 0x10]);
            assert_eq!(both.write(piece + 0xb0, &[8; 0x20]), 0);
            // Past the last view the table knows: no place.
            assert!(find(SIDE + 2 * K64 - 8, 32).is_none());
            let stretches = [(piece + 0x10, 0x20), (piece + 0x180, 0x20)];
            let places = super::find_all(piece, 0x200, &stretches).expect("a place for each stretch");
            assert_eq!(places.of(0).unwrap().write(stretches[0].0, &[3; 0x20]), 0x20);
            assert_eq!(places.of(1).unwrap().write(stretches[1].0, &[4; 0x20]), 0x20);
            // The second stretch is 0x80 into the section's third 64 KiB.
            assert_eq!(title(K64 + 0x80, 0x20), [4; 0x20]);
            // The first is 0x110 before the end of its first 64 KiB.
            let (first, _) = place(SIDE + K64 - 0xf0, 0x20).expect("a place in our view");
            assert_eq!(std::slice::from_raw_parts(first as *const u8, 0x20), [3; 0x20]);
            let across = super::find_all(piece, 0x200, &[(piece + 0xf0, 0x20)]).expect("a stretch across the two has its place");
            assert_eq!(across.of(0).unwrap().write(piece + 0xf0, &[5; 0x20]), 0x20);
            let (end_of_first, _) = place(SIDE + K64 - 0x10, 0x10).expect("a place in our view");
            assert_eq!(std::slice::from_raw_parts(end_of_first as *const u8, 0x10), [5; 0x10]);
            assert_eq!(title(K64, 0x10), [5; 0x10]);
            let whole = super::find_all(view + 0x100, 0x100, &[(view + 0x110, 8)]).expect("one place");
            assert_eq!(whole.of(0).unwrap().write(view + 0x110, &[6; 8]), 8);
            assert_eq!(title(0x110, 8), [6; 8]);
            assert!(super::find_all(private_at(), 16, &[]).is_none());
            crate::inplace::views_gone(SIDE, 2 * K64);
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
