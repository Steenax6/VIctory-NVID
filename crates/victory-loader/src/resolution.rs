//! Opt-in output size and render scale (`VICTORY_OUTPUT`,
//! `VICTORY_RENDER_SCALE`). Both change a few places of the title's code at
//! load; without the switch no byte is touched. Origin of the offsets and of
//! the idea: the avatarmanz fork (AMD) and its resolution patches.
//!
//! - **Output size.** The size the title was set to is a pair of numbers in
//!   its live settings (`+0x204`, `+0x208` of the settings block), applied by
//!   one block that hands them to the screen-size setter and, if the height
//!   changed, tells the rest. That block becomes: store the size asked for
//!   into the settings, then the same calls. Everything that reads the
//!   settings afterwards (device parameters, render size) sees that size. The
//!   title only has display modes for five sizes, so only those are taken.
//!   The pool the back buffers are made in is checked to be nearly used up
//!   (an `int3` otherwise); it is sized for 3840x2160, so for any other size
//!   that check is jumped over.
//! - **Render scale.** The title's own upscaler renders at a fraction of the
//!   output size, set once to 2/3 by a store of an immediate; the switch
//!   replaces that immediate.
//!
//! Offsets are image offsets of the one build `hooks` knows. Each place is
//! compared first (short ones by bytes, the long block by length and BLAKE3,
//! so none of the title's code is kept here) and is left alone, with a
//! warning, if it differs.

use crate::hooks::{Expect, apply_code, is_known_build};

/// Output sizes the title has display modes for.
const MODES: [(u32, u32); 5] = [(1280, 720), (1664, 936), (1920, 1080), (2560, 1440), (3840, 2160)];

/// The size the title writes and renders at without the switch.
const DEFAULT_OUTPUT: (u32, u32) = (3840, 2160);

/// Image offsets the new block calls or reads.
const SET_SCREEN_SIZE: usize = 0x2478130;
const SCREEN_HEIGHT: usize = 0xa9438d0;
const HEIGHT_CHANGED: usize = 0xd8c3a0;

/// Where the settings' size is applied, and the length of that block.
const OUTPUT_AT: usize = 0x224352b;
const OUTPUT_LEN: usize = 61;
/// BLAKE3 of the block's original bytes.
const OUTPUT_BLAKE3: &str = "f25c4444265c7e41c845ebd73928cab5d62069ebfa4ed6e90a59053d65ab0963";

/// The back buffer pool's left-over check: compare, branch, `int3`.
const POOL_AT: usize = 0x247af25;
const POOL_CHECK: [u8; 9] = [0x48, 0x3d, 0xff, 0xff, 0x1f, 0x00, 0x76, 0x01, 0xcc];

/// The own upscaler's store of its scale: `mov dword [rcx+90h], 2/3`.
const SCALE_AT: usize = 0x28f1e78;
const SCALE_STORE: [u8; 10] = [0xc7, 0x81, 0x90, 0x00, 0x00, 0x00, 0xab, 0xaa, 0x2a, 0x3f];

/// `VICTORY_OUTPUT=<w>x<h>`: one of the title's five modes.
fn parse_output(spec: &str) -> Option<(u32, u32)> {
    let (w, h) = spec.trim().split_once(['x', 'X'])?;
    let size = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    MODES.contains(&size).then_some(size)
}

/// `VICTORY_RENDER_SCALE=<s>`, 0.5 to 1.0.
fn parse_scale(spec: &str) -> Option<f32> {
    spec.trim().parse::<f32>().ok().filter(|s| (0.5..=1.0).contains(s))
}

/// The switch `name`, if set and allowed; says why in the log if it is set
/// and not.
fn asked<T>(name: &str, parse: fn(&str) -> Option<T>, allowed: &str) -> Option<T> {
    let spec = std::env::var(name).ok()?;
    let value = parse(&spec);
    if value.is_none() {
        tracing::warn!("{name}={spec}: not allowed (expected {allowed}); nothing changed");
    }
    value
}

/// `rel32` from the end of an instruction at image offset `next` to `to`.
fn rel(to: usize, next: usize) -> [u8; 4] {
    ((to as i64 - next as i64) as i32).to_le_bytes()
}

/// What goes at [`OUTPUT_AT`] for `width` x `height`, [`OUTPUT_LEN`] bytes.
fn output_code(width: u32, height: u32) -> Vec<u8> {
    let mut c = Vec::with_capacity(OUTPUT_LEN);
    c.push(0xb8); // mov eax, width
    c.extend(width.to_le_bytes());
    c.extend([0x89, 0x86, 0x04, 0x02, 0x00, 0x00]); // mov [rsi+204h], eax
    c.push(0xbf); // mov edi, height
    c.extend(height.to_le_bytes());
    c.extend([0x89, 0xbe, 0x08, 0x02, 0x00, 0x00]); // mov [rsi+208h], edi
    c.extend([0x8b, 0x2d]); // mov ebp, [rip+screen height]
    c.extend(rel(SCREEN_HEIGHT, OUTPUT_AT + c.len() + 4));
    c.extend([0x89, 0xc1, 0x89, 0xfa]); // mov ecx, eax ; mov edx, edi
    c.push(0xe8); // call the screen-size setter
    c.extend(rel(SET_SCREEN_SIZE, OUTPUT_AT + c.len() + 4));
    c.extend([0x39, 0xfd, 0x74, (OUTPUT_LEN - (c.len() + 4)) as u8]); // cmp ebp, edi ; je end
    c.extend([0xb1, 0x01, 0xe8]); // mov cl, 1 ; call the height-changed handler
    c.extend(rel(HEIGHT_CHANGED, OUTPUT_AT + c.len() + 4));
    c.extend([0xeb, (OUTPUT_LEN - (c.len() + 2)) as u8]); // jmp end
    c.resize(OUTPUT_LEN, 0xcc);
    c
}

/// What goes at [`POOL_AT`]: the same, with `jmp +1` in place of `jbe +1`.
fn pool_code() -> [u8; 9] {
    let mut c = POOL_CHECK;
    c[6] = 0xeb;
    c
}

/// What goes at [`SCALE_AT`] for `scale`.
fn scale_code(scale: f32) -> [u8; 10] {
    let mut c = SCALE_STORE;
    c[6..].copy_from_slice(&scale.to_le_bytes());
    c
}

/// Applies the switches that are set. Nothing at all happens (no log line
/// either) when neither is.
///
/// # Safety
/// `base` is the mapped image whose original BLAKE3 is `hash`, its `.text`
/// covers the patches' offsets on that build, and no title code runs yet.
pub(crate) unsafe fn apply_asked(base: usize, hash: &blake3::Hash) {
    let output = asked("VICTORY_OUTPUT", parse_output, "one of 1280x720, 1664x936, 1920x1080, 2560x1440, 3840x2160");
    let scale = asked("VICTORY_RENDER_SCALE", parse_scale, "0.5 to 1.0");
    if output.is_none() && scale.is_none() {
        return;
    }
    if !is_known_build(hash) {
        tracing::warn!("output size and render scale: not the build the patches are for; nothing patched");
        return;
    }
    let block = Expect::Hash {
        len: OUTPUT_LEN,
        blake3: OUTPUT_BLAKE3,
    };
    if let Some((w, h)) = output {
        let what = "output size: the settings' size applied";
        // SAFETY: forwarded from this function's contract.
        if unsafe { apply_code(base, what, OUTPUT_AT, &block, &output_code(w, h)) } {
            tracing::info!("VICTORY_OUTPUT: output size {w}x{h}");
        }
        if (w, h) != DEFAULT_OUTPUT {
            let what = "output size: the back buffer pool's check";
            // SAFETY: as above.
            unsafe { apply_code(base, what, POOL_AT, &Expect::Bytes(&POOL_CHECK), &pool_code()) };
        }
    }
    if let Some(s) = scale {
        let what = "render scale: the own upscaler's scale";
        // SAFETY: as above.
        if unsafe { apply_code(base, what, SCALE_AT, &Expect::Bytes(&SCALE_STORE), &scale_code(s)) } {
            tracing::info!("VICTORY_RENDER_SCALE: render scale {s}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_values_parse() {
        assert_eq!(parse_output("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_output(" 2560X1440 "), Some((2560, 1440)));
        assert_eq!(parse_output("3840x2160"), Some(DEFAULT_OUTPUT));
        assert_eq!(parse_output("1600x900"), None);
        assert_eq!(parse_output("1080p"), None);
        assert_eq!(parse_output(""), None);
        assert_eq!(parse_scale("0.5"), Some(0.5));
        assert_eq!(parse_scale("0.67"), Some(0.67));
        assert_eq!(parse_scale("1"), Some(1.0));
        assert_eq!(parse_scale("0.4"), None);
        assert_eq!(parse_scale("1.5"), None);
        assert_eq!(parse_scale("NaN"), None);
        assert_eq!(parse_scale("half"), None);
    }

    #[test]
    fn every_patch_is_as_long_as_what_it_checks() {
        assert_eq!(output_code(1920, 1080).len(), OUTPUT_LEN);
        assert_eq!(pool_code().len(), POOL_CHECK.len());
        assert_eq!(scale_code(0.5).len(), SCALE_STORE.len());
        assert_eq!(OUTPUT_BLAKE3.len(), 64);
        assert!(OUTPUT_BLAKE3.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
        // The stored default is the one a title without the switch has.
        assert_eq!(SCALE_STORE[6..], (2.0f32 / 3.0).to_le_bytes());
        // Only the branch and the scale differ from what was compared.
        assert_eq!(pool_code()[..6], POOL_CHECK[..6]);
        assert_eq!(pool_code()[7..], POOL_CHECK[7..]);
        assert_eq!(scale_code(0.5)[..6], SCALE_STORE[..6]);
        assert_eq!(scale_code(0.5)[6..], 0.5f32.to_le_bytes());
    }

    #[test]
    fn the_output_block_stores_the_size_and_calls_what_the_title_does() {
        let c = output_code(1920, 1080);
        assert_eq!(c[0], 0xb8);
        assert_eq!(c[1..5], 1920u32.to_le_bytes());
        assert_eq!(c[5..11], [0x89, 0x86, 0x04, 0x02, 0x00, 0x00]);
        assert_eq!(c[11], 0xbf);
        assert_eq!(c[12..16], 1080u32.to_le_bytes());
        assert_eq!(c[16..22], [0x89, 0xbe, 0x08, 0x02, 0x00, 0x00]);
        let target = |op_end: usize| (OUTPUT_AT + op_end) as i64 + i64::from(i32::from_le_bytes(c[op_end - 4..op_end].try_into().unwrap()));
        assert_eq!(c[22..24], [0x8b, 0x2d]);
        assert_eq!(target(28), SCREEN_HEIGHT as i64);
        assert_eq!(c[32], 0xe8);
        assert_eq!(target(37), SET_SCREEN_SIZE as i64);
        assert_eq!(c[37..40], [0x39, 0xfd, 0x74]);
        assert_eq!(41 + usize::from(c[40]), OUTPUT_LEN, "je lands at the end");
        assert_eq!(c[41..44], [0xb1, 0x01, 0xe8]);
        assert_eq!(target(48), HEIGHT_CHANGED as i64);
        assert_eq!(c[48], 0xeb);
        assert_eq!(50 + usize::from(c[49]), OUTPUT_LEN, "jmp lands at the end");
        assert!(c[50..].iter().all(|&x| x == 0xcc));
    }

    /// Without either switch nothing is read or written: the base is not
    /// even a mapped address.
    #[test]
    fn nothing_is_touched_without_a_switch() {
        // Both unset in this process's environment unless the test runner
        // was given them; then there is nothing to say here.
        if std::env::var_os("VICTORY_OUTPUT").is_some() || std::env::var_os("VICTORY_RENDER_SCALE").is_some() {
            return;
        }
        let build = blake3::hash(b"any");
        // SAFETY: no switch is set, so nothing is dereferenced.
        unsafe { apply_asked(0, &build) };
    }
}
