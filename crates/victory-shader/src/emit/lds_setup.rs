//! A thread group's set-up of its LDS by its first thread, written whether that thread's lane
//! still runs there or not.
//!
//! The compilers' `if (threadIndex == 0) shared = init;` comes out as
//!
//! ```text
//! v_cmpx_eq_u32 exec, 0, <thread id>       ; x, or x | y | z by v_or_b32
//! v_mov_b32 <address>, <constant>
//! v_mov_b32 <data>, <constant>
//! ds_write_b32 / ds_write2_b32 <address>, <data> ...
//! ```
//!
//! and the console runs that write only where the first thread's lane still runs. The water's
//! transmission volumetrics (`CS_TransmissionVolumetrics_Tiled` and `_Tiled_LocalLights`, one
//! console wave a group) have thread (0,0) set the tile's greatest and least depth (dwords 0
//! and 1, to 0 and 0xFFFFFFFF) after the lanes with nothing to shade have left (off the target,
//! or no water there); the lanes left take their depths into them with `ds_max_u32` and
//! `ds_min_u32`. A tile whose first pixel has nothing to shade starts from what LDS held: on
//! the console what the group before left, on the host anything. Here the first thread always
//! writes the set-up (`Lower::ds`), and the tile works from its own depths.
//!
//! Only where no one would write at all: the lanes missing at the set-up must have left for
//! good, so that none of them uses LDS before it or after it ([`lanes_missing_never_use_lds`]).
//! A set-up in one arm of a per-lane if/else whose other arm uses LDS too stays the lane's
//! write: there the console's second arm starts from the first arm's finished values when the
//! first thread went the other way, and that is the console's result.
//!
//! The title's shader archive has the pattern in 19 compute programs; 15 are taken. In 13 of
//! them EXEC is whole at the compare (at the program's start, or past uniform branches and
//! regions that have closed), so the first thread runs the write on the console too and what
//! they compute does not change. Only the two above change. Left as they were:
//! `CS_TransmissionVolumetrics` and `_VolumePortalsOnly` (a set-up in each arm of a per-lane
//! if/else), and two tessellation programs whose EXEC is whole at the compare but whose earlier
//! LDS writes run under masks the rule does not follow back.
//! (`VICTORY_LDS_FIRST_INIT=0`: the lane's write everywhere, as the console has it.)

use crate::gfx10::{Fields, Operand};
use crate::ir::{EXEC, Program, Reg, Value};
use crate::native::{NativeProgram, StageIo};
use crate::structure::Structured;

/// Whether set-ups are written by the first thread alone (`VICTORY_LDS_FIRST_INIT=0`: no).
pub(super) fn on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_ON.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LDS_FIRST_INIT").is_none_or(|v| v != "0"))
}

#[cfg(test)]
thread_local! {
    /// A test's choice for [`on`], for its thread only.
    pub(super) static TEST_ON: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// The set-ups the lowerings on this thread wrote by the first thread alone.
    pub(super) static TEST_SETUPS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// The LDS write at instruction `i` of block `b` (its address operand `addr`, data operands
/// `data`, and (byte offset, dwords) per element) when it is the set-up: `v_cmpx_eq_u32 exec,
/// 0, <place>` leaves EXEC as the write has it, the place being the thread's id in the group
/// as the program entered with it (x, or ids ORed together by `v_or_b32` in the same block
/// under the same guard, every dimension of more than one thread among them), and the address
/// and each dword written are constants moved in by `v_mov_b32` after the compare. Gives each
/// dword's index in LDS and its value; `None` for any other write, or with the switch off.
pub(super) fn find(
    p: &Program,
    st: &Structured,
    native: &NativeProgram,
    (b, i): (usize, usize),
    addr: Option<&Operand>,
    data: &[Operand],
    elements: &[(u32, u32)],
) -> Option<Vec<(u32, u32)>> {
    if !on() {
        return None;
    }
    let StageIo::Cs(cs) = native.io else {
        return None;
    };
    let insts = &p.blocks[b].insts;
    let guards = &st.guards[b];
    // the compare that made EXEC what it is at the write
    let j = (0..i).rev().find(|&j| insts[j].write(Reg::S(EXEC)).is_some())?;
    let Fields::Valu(cmp) = &insts[j].fields else {
        return None;
    };
    if insts[j].name() != "v_cmpx_eq_u32" || cmp.sdwa.is_some() || cmp.dpp.is_some() {
        return None;
    }
    let place = match cmp.src[..] {
        [Operand::Const(128), Operand::Vector { reg, .. }] | [Operand::Vector { reg, .. }, Operand::Const(128)] => {
            insts[j].read(Reg::V(reg))?
        }
        _ => return None,
    };
    // the place: thread ids (inputs v0..v2), ORed together in this block before the compare
    let id = |v: Value| {
        p.inputs.iter().find(|input| input.1 == v).and_then(|input| match input.0 {
            Reg::V(c @ 0..=2) => Some(c),
            _ => None,
        })
    };
    let mut ids: Vec<u8> = Vec::new();
    let mut open = vec![place];
    while let Some(v) = open.pop() {
        if let Some(c) = id(v) {
            if ids.contains(&c) {
                return None;
            }
            ids.push(c);
            continue;
        }
        let k = (0..j).find(|&k| insts[k].writes.iter().any(|w| w.1 == v))?;
        let Fields::Valu(or) = &insts[k].fields else {
            return None;
        };
        if insts[k].name() != "v_or_b32" || or.sdwa.is_some() || or.dpp.is_some() || guards[k] != guards[j] {
            return None;
        }
        for op in &or.src {
            let Operand::Vector { reg, .. } = *op else {
                return None;
            };
            open.push(insts[k].read(Reg::V(reg))?);
        }
        if ids.len() + open.len() > 3 {
            return None;
        }
    }
    if (0..3).any(|c| cs.threads[c] > 1 && !ids.contains(&(c as u8))) {
        return None;
    }
    // a register as the write reads it: a constant moved in after the compare
    let constant = |op: &Operand| -> Option<u32> {
        let Operand::Vector { reg, .. } = *op else {
            return None;
        };
        let v = insts[i].read(Reg::V(reg))?;
        let k = (j + 1..i).find(|&k| insts[k].writes.iter().any(|w| w.1 == v))?;
        let Fields::Valu(mov) = &insts[k].fields else {
            return None;
        };
        if insts[k].name() != "v_mov_b32" || mov.sdwa.is_some() || mov.dpp.is_some() {
            return None;
        }
        match mov.src[..] {
            [Operand::Const(c)] => crate::resource::inline(c),
            [Operand::Literal(x)] => Some(x),
            _ => None,
        }
    };
    if !lanes_missing_never_use_lds(p, &insts[j]) {
        return None;
    }
    let at = constant(addr?)?;
    let mut words = Vec::new();
    for (e, &(offset, dwords)) in elements.iter().enumerate() {
        if dwords != 1 {
            return None;
        }
        let byte = at.checked_add(offset)?;
        if byte % 4 != 0 || byte / 4 >= cs.lds_bytes() / 4 {
            return None;
        }
        words.push((byte / 4, constant(data.get(e)?)?));
    }
    Some(words)
}

/// Whether every LDS access of `p` runs under an EXEC that holds only lanes of the EXEC the
/// set-up's compare `cmp` starts from: the lanes missing at the set-up have left for good and
/// touch LDS neither before it nor after it. Not so where the set-up sits in one arm of a
/// per-lane if/else whose other arm uses LDS too: there the console's second arm starts from
/// what the first left when the first thread went the other way, and that is kept.
///
/// An EXEC value holds only such lanes when it is the compare's, or narrows one that does:
/// a copy (`s_mov`), an AND with one (`s_and`, `s_andn2` of one, `s_and_saveexec` and a
/// lane compare into EXEC), the EXEC a saveexec saved from one, or a phi of such values
/// (taken as so around loops until a way in says otherwise). Anything else (a restore of a
/// mask saved before the compare, an OR) holds lanes the compare did not have.
fn lanes_missing_never_use_lds(p: &Program, cmp: &crate::ir::Inst) -> bool {
    use std::collections::{HashMap, HashSet};
    let halves: u8 = if p.wave64 { 2 } else { 1 };
    let start: Vec<Value> = (0..halves).filter_map(|h| cmp.read(Reg::S(EXEC + h))).collect();
    if start.len() != usize::from(halves) {
        return false;
    }
    // how each value of a scalar half comes about
    enum Def<'p> {
        Phi(&'p [Value]),
        Inst(&'p crate::ir::Inst, Reg),
    }
    let mut defs: HashMap<Value, Def> = HashMap::new();
    for block in &p.blocks {
        for phi in &block.phis {
            defs.insert(phi.value, Def::Phi(&phi.args));
        }
        for inst in &block.insts {
            for &(reg, v) in &inst.writes {
                defs.insert(v, Def::Inst(inst, reg));
            }
        }
    }
    // the values an operand reads in half `h`
    let half = |inst: &crate::ir::Inst, op: Option<&Operand>, h: u8| match op {
        Some(Operand::Scalar { code, .. }) => inst.read(Reg::S(code + h)),
        _ => None,
    };
    // Greatest fixpoint: every value is taken to hold only the compare's lanes until its way
    // of coming about says otherwise.
    let mut out: HashSet<Value> = HashSet::new();
    let inside = |v: Option<Value>, out: &HashSet<Value>| v.is_some_and(|v| start.contains(&v) || defs.contains_key(&v) && !out.contains(&v));
    loop {
        let mut changed = false;
        for (&v, def) in &defs {
            if out.contains(&v) || start.contains(&v) {
                continue;
            }
            let holds = match *def {
                Def::Phi(args) => args.iter().all(|&a| inside(Some(a), &out)),
                Def::Inst(inst, Reg::S(code)) => holds_inside(inst, code, &half, &|x| inside(x, &out)),
                Def::Inst(..) => false,
            };
            if !holds {
                out.insert(v);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    p.blocks.iter().flat_map(|b| &b.insts).filter(|i| uses_lds(i)).all(|i| {
        (0..halves).all(|h| inside(i.read(Reg::S(EXEC + h)), &out))
    })
}

/// An LDS access (not a lane swizzle or permute, which touch no LDS).
fn uses_lds(i: &crate::ir::Inst) -> bool {
    matches!(i.fields, Fields::Ds { .. }) && !i.name().starts_with("ds_swizzle") && !i.name().contains("permute")
}

/// Whether what `inst` writes to `s<code>` holds only the compare's lanes, given `inside` for
/// the values it reads (`half`: an operand's value in the low or high dword).
fn holds_inside(
    inst: &crate::ir::Inst,
    code: u8,
    half: &dyn Fn(&crate::ir::Inst, Option<&Operand>, u8) -> Option<Value>,
    inside: &dyn Fn(Option<Value>) -> bool,
) -> bool {
    let name = inst.name();
    let exec = code.checked_sub(EXEC).filter(|&h| h < 2);
    let old_exec = |h: u8| inside(inst.read(Reg::S(EXEC + h)));
    match &inst.fields {
        // a lane compare into EXEC: the lanes it had, where the compare holds
        Fields::Valu(_) => exec.is_some_and(|h| name.starts_with("v_cmpx_") && old_exec(h)),
        Fields::Sop { dst: Some(Operand::Scalar { code: d, .. }), src, .. } => {
            let base = name.trim_end_matches("_b64").trim_end_matches("_b32");
            match (base, exec) {
                ("s_and_saveexec", Some(h)) => old_exec(h) || inside(half(inst, src.first(), h)),
                ("s_andn2_saveexec", Some(h)) => inside(half(inst, src.first(), h)),
                // the EXEC a saveexec saved
                (b, None) if b.ends_with("_saveexec") => code.checked_sub(*d).is_some_and(|h| h < 2 && old_exec(h)),
                (b, _) => {
                    let Some(h) = code.checked_sub(*d).filter(|&h| h < 2) else {
                        return false;
                    };
                    let (a, c) = (half(inst, src.first(), h), half(inst, src.get(1), h));
                    match b {
                        "s_mov" => inside(a),
                        "s_and" => inside(a) || inside(c),
                        "s_andn2" => inside(a),
                        _ => false,
                    }
                }
            }
        }
        _ => false,
    }
}
