//! The copy back of shader-written textures (`crate::texback`) through the
//! draw path's view handling, on the software adapter: only the levels and
//! layers the bound views expose land in title memory, and a copy whose
//! memory was remapped meanwhile does not land. From BradRoland's
//! fxceless/victory-fork PR #1152.

use super::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_FLOAT, DXGI_SAMPLE_DESC};
use windows::core::IUnknown;

fn texture(slices: u16, mips: u16) -> IUnknown {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 8,
        Height: 8,
        DepthOrArraySize: slices,
        MipLevels: mips,
        Format: DXGI_FORMAT_R32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut texture = std::ptr::null();
    assert_eq!(
        crate::resource::create_committed(
            std::ptr::null(),
            (&raw const heap).cast(),
            0,
            (&raw const desc).cast(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32,
            std::ptr::null(),
            std::ptr::from_ref(&ID3D12Resource::IID).cast(),
            &mut texture,
        ),
        crate::object::S_OK
    );
    // SAFETY: create_committed handed us an owned reference.
    unsafe { IUnknown::from_raw(texture.cast_mut().cast()) }
}

fn array_view(mip: u32, first: u32, count: u32) -> [u32; 10] {
    [
        41,
        D3D12_UAV_DIMENSION_TEXTURE2DARRAY.0 as u32,
        mip,
        first,
        count,
        0,
        0,
        0,
        0,
        0,
    ]
}

fn two_d_view(mip: u32) -> [u32; 10] {
    [
        41,
        D3D12_UAV_DIMENSION_TEXTURE2D.0 as u32,
        mip,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
}

/// Clears the host texture of `title` through each of `views` to its value
/// in one list; with `bound`, each view is first handed to the draw path as
/// a shader's unordered view.
fn clear(
    host: &crate::host::Host,
    title: This,
    views: &[(Option<[u32; 10]>, f32)],
    bound: bool,
    before_run: impl FnOnce(),
) {
    let texture = crate::resource::host_copy(title).unwrap();
    let list = host.new_list().unwrap();
    let mut heaps = Vec::new();
    for (words, value) in views {
        if bound {
            readable(
                &list,
                ResourceType::Uav,
                &View::Unordered(title as usize, 0, *words),
            );
        }
        // SAFETY: the views describe this live texture; both heaps are kept
        // until the list has run.
        unsafe {
            let heap = |flags| -> ID3D12DescriptorHeap {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: 1,
                        Flags: flags,
                        NodeMask: 0,
                    })
                    .unwrap()
            };
            let cpu = heap(D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
            let gpu = heap(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
            let native = words.as_ref().map(|w| {
                w.as_ptr()
                    .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
                    .read_unaligned()
            });
            let handle = cpu.GetCPUDescriptorHandleForHeapStart();
            host.device.CreateUnorderedAccessView(
                &texture.texture,
                None::<&ID3D12Resource>,
                native.as_ref().map(std::ptr::from_ref),
                handle,
            );
            host.device.CopyDescriptorsSimple(
                1,
                gpu.GetCPUDescriptorHandleForHeapStart(),
                handle,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            );
            list.list.SetDescriptorHeaps(&[Some(gpu.clone())]);
            list.list.ClearUnorderedAccessViewFloat(
                gpu.GetGPUDescriptorHandleForHeapStart(),
                handle,
                &texture.texture,
                &[*value; 4],
                &[],
            );
            heaps.push((cpu, gpu));
        }
    }
    list.close();
    before_run();
    host.execute(&[&list]);
}

fn verify(host: &crate::host::Host, slices: u16) {
    let owner = texture(slices, 3);
    let title = owner.as_raw().cast();
    let (layout, at, size) = crate::resource::title_texture(title).unwrap();
    // Every host texel 1, through views not handed to the draw path.
    let all: Vec<_> = (0..3)
        .map(|mip| {
            (
                Some(if slices == 1 {
                    two_d_view(mip)
                } else {
                    array_view(mip, 0, u32::from(slices))
                }),
                1.0,
            )
        })
        .collect();
    clear(host, title, &all, false, || {});
    // SAFETY: our own allocation of `size` bytes; nothing runs.
    unsafe { (at as *mut u8).write_bytes(0x4a, size) };
    let bound = if slices == 1 {
        vec![(None, 4.0), (Some(two_d_view(1)), 5.0)]
    } else {
        vec![
            (None, 4.0),
            (Some(array_view(1, 1, 2)), 5.0),
            (Some(array_view(1, 2, u32::MAX)), 5.0),
            (Some(two_d_view(2)), 6.0),
        ]
    };
    clear(host, title, &bound, true, || {});
    let mut expected = vec![0x4a; size];
    for mip in 0..3 {
        for slice in 0..u32::from(slices) {
            let value = match (mip, slice) {
                (0, _) => 4f32.to_bits(),
                (1, _) if slices == 1 || slice >= 1 => 5f32.to_bits(),
                (2, 0) if slices > 1 => 6f32.to_bits(),
                _ => 0x4a4a_4a4a,
            };
            for y in 0..(8 >> mip) {
                for x in 0..(8 >> mip) {
                    let offset = layout.offset(mip, x, y, slice).unwrap() as usize;
                    expected[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
        }
    }
    // SAFETY: our allocation; the execute waited for the copy back.
    let actual = unsafe { std::slice::from_raw_parts(at as *const u8, size) };
    if let Some(offset) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
        panic!(
            "{slices} layers: byte {offset} is {:#x}, expected {:#x}",
            actual[offset], expected[offset]
        );
    }
    // A copy recorded before its memory is remapped does not land.
    // SAFETY: as above.
    unsafe { (at as *mut u8).write_bytes(0x4a, size) };
    clear(host, title, &[(None, 7.0)], true, || {
        crate::texback::remapped(at, size)
    });
    // SAFETY: as above.
    let after = unsafe { std::slice::from_raw_parts(at as *const u8, size) };
    assert!(
        after.iter().all(|&b| b == 0x4a),
        "a copy back landed in remapped memory"
    );
}

#[test]
fn copies_back_only_what_bound_views_expose() {
    const CHILD: &str = "VICTORY_TEST_TEXBACK_VIEWS";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                std::thread::current().name().unwrap(),
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("VICTORY_D3D12_WARP", "1")
            .env("VICTORY_TEXTURE_BACK", "8x8f41")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    let Some(host) = crate::host::host() else {
        eprintln!("texture copy-back test skipped: no software adapter");
        return;
    };
    verify(host, 4);
    verify(host, 1);
    released_before_landing(host);
    state_after_the_copy(host);
}

/// A texture the title releases between the list's close and its run: its
/// copy does not land (its memory is the title's no more), and nothing
/// faults.
fn released_before_landing(host: &crate::host::Host) {
    let owner = texture(1, 1);
    let title: This = owner.as_raw().cast();
    let before = crate::texback::totals();
    let list = host.new_list().unwrap();
    readable(&list, ResourceType::Uav, &View::Unordered(title as usize, 0, None));
    list.close();
    drop(owner);
    host.execute(&[&list]);
    let after = crate::texback::totals();
    assert_eq!(after[0], before[0], "a copy landed in released memory");
    assert_eq!(after[1], before[1] + 1, "the copy was not dropped");
}

/// The copy leaves the texture in the state the list left it in, not in
/// the copy's (fxceless/victory-fork pull request #1170).
fn state_after_the_copy(host: &crate::host::Host) {
    const UNORDERED_ACCESS: u32 = 0x8;
    let owner = texture(1, 1);
    let title: This = owner.as_raw().cast();
    let list = host.new_list().unwrap();
    readable(&list, ResourceType::Uav, &View::Unordered(title as usize, 0, None));
    list.close();
    let copy = crate::resource::host_copy(title).unwrap();
    assert_eq!(list.end_state(&copy), Some(UNORDERED_ACCESS));
    host.execute(&[&list]);
}
