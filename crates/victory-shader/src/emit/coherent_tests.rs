//! Programs that order memory between thread groups ([`crate::ir::orders_across_groups`]):
//! which they are, and that they get globally coherent views and device fences.

use super::{lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

fn build(body: &[u32]) -> ir::Program {
    let mut words = body.to_vec();
    words.push(S_ENDPGM);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    ir::build(&code, false, 8).unwrap()
}

fn orders(body: &[u32]) -> bool {
    let p = build(body);
    ir::orders_across_groups(p.blocks.iter().flat_map(|b| &b.insts).map(|i| (i.name(), i.fields.clone())))
}

/// Which programs order memory between groups: loads and stores marked `glc` or `dlc`, scalar
/// loads so marked, cache invalidations; not plain accesses, not an atomic's `glc` (the old
/// value back), not `slc`, not LDS atomics (after fxceless/victory-fork #914's test). Only
/// there does `s_waitcnt_vscnt` stay in the IR.
#[test]
fn programs_that_order_memory_between_groups_are_known_by_their_cache_bits() {
    // buffer_load_dword v1, v0, s[4:7], 0 offen; buffer_store_dword v1, v0, s[4:7], 0 offen
    assert!(!orders(&[0xE030_1000, 0x8001_0100, 0xE070_1000, 0x8001_0100]));
    // ds_add_u32 v0, v1
    assert!(!orders(&[0xD800_0000, 0x0000_0100]));
    // buffer_atomic_add v1, v0, s[4:7], 0 offen glc
    assert!(!orders(&[0xE0C8_5000, 0x8001_0100]));
    // the load with slc (dword 1 bit 22)
    assert!(!orders(&[0xE030_1000, 0x8041_0100]));
    // the load with glc (bit 14), dlc (bit 15)
    assert!(orders(&[0xE030_5000, 0x8001_0100]));
    assert!(orders(&[0xE030_9000, 0x8001_0100]));
    // buffer_gl0_inv
    assert!(orders(&[0xE1C4_0000, 0x0000_0000]));
    // s_load_dword s8, s[4:5], 0x0 glc, and without
    assert!(orders(&[0xF401_0202, 0xFA00_0000]));
    assert!(!orders(&[0xF400_0202, 0xFA00_0000]));
    // a store with glc, s_waitcnt_vscnt null, 0x0, an atomic: the wait stays
    let kept = |body: &[u32]| build(body).blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "s_waitcnt_vscnt");
    assert!(kept(&[0xE070_5000, 0x8001_0100, 0xBBFD_0000, 0xE0C8_1000, 0x8001_0100]));
    assert!(!kept(&[0xE070_1000, 0x8001_0100, 0xBBFD_0000, 0xE0C8_1000, 0x8001_0100]));
}

/// A one-thread compute program whose table at s0 holds one UAV, lowered.
fn lowered(words: &[u32]) -> lower::Lowered {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
    .expect("lower")
}

/// A program that orders memory between groups has its UAVs globally coherent and a device
/// fence at the wait for its stores; the same code without `glc` has neither.
#[test]
fn a_program_that_orders_memory_between_groups_has_coherent_views_and_fences() {
    // s_mov_b32 s4, s0; s_mov_b32 s5, 0; s_load_dwordx4 s[8:11], s[4:5]; a store (glc or not)
    // of v1 at v0 through it, s_waitcnt_vscnt null, 0x0, buffer_atomic_add likewise
    let program = |store: u32| {
        [
            0xBE84_0300,
            0xBE85_0380,
            0xF408_0202,
            0xFA00_0000,
            store,
            0x8002_0100,
            0xBBFD_0000,
            0xE0C8_1000,
            0x8002_0100,
            S_ENDPGM,
        ]
    };
    let plain = lowered(&program(0xE070_1000));
    let marked = lowered(&program(0xE070_5000));
    assert!(!plain.coherent && marked.coherent);
    // the views' property and the fence are in the module
    assert_ne!(plain.dxil, marked.dxil);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut plain.dxil.clone()).expect("DXIL validation failed");
        v.validate(&mut marked.dxil.clone()).expect("DXIL validation failed");
    }
}
