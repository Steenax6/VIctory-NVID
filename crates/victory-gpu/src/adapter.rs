//! Which graphics card the host device is made on.
//!
//! `D3D12CreateDevice(None, ..)` takes the adapter DXGI lists first, which on
//! a machine with a second, weaker adapter (the processor's own graphics, a
//! laptop's integrated chip) need not be the card the player means. The
//! device is made on the adapter the system ranks highest for performance
//! instead, never on a software one
//! (Marvisak/VIctory pull request #3073, ported to this tree's helper).
//!
//! `VICTORY_D3D12_ADAPTER=<n or text>` takes another one: a number is the
//! place in the list the system ranks (0 is the first choice), text is part
//! of the adapter's name, in any case.

use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL;
use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device};
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

/// What the choice looks at of an adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Candidate {
    name: String,
    /// Dedicated video memory, bytes.
    dedicated: usize,
}

/// An adapter's null-terminated UTF-16 description as text.
fn adapter_name(desc: &[u16]) -> String {
    let len = desc.iter().position(|&c| c == 0).unwrap_or(desc.len());
    String::from_utf16_lossy(&desc[..len])
}

/// The place in `list` (hardware adapters, best first) that `filter` names:
/// a number is a place, text is part of a name. `None` if it names none.
fn pick(list: &[Candidate], filter: &str) -> Option<usize> {
    let filter = filter.trim();
    if let Ok(place) = filter.parse::<usize>() {
        return (place < list.len()).then_some(place);
    }
    let filter = filter.to_lowercase();
    list.iter().position(|c| c.name.to_lowercase().contains(&filter))
}

/// The hardware adapters (no software one, see `DXGI_ADAPTER_FLAG_SOFTWARE`),
/// best first: as the system ranks them for performance, or, where it does
/// not (before Windows 10 1803), by dedicated video memory.
fn hardware_adapters() -> Vec<(IDXGIAdapter1, Candidate, u32)> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return Vec::new();
        };
        let mut found: Vec<(IDXGIAdapter1, DXGI_ADAPTER_DESC1)> = Vec::new();
        let mut keep = |adapter: IDXGIAdapter1| {
            if let Ok(desc) = adapter.GetDesc1()
                && desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0
            {
                found.push((adapter, desc));
            }
        };
        if let Ok(factory6) = factory.cast::<IDXGIFactory6>() {
            let mut i = 0;
            while let Ok(adapter) = factory6
                .EnumAdapterByGpuPreference::<IDXGIAdapter1>(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
            {
                i += 1;
                keep(adapter);
            }
        } else {
            let mut i = 0;
            while let Ok(adapter) = factory.EnumAdapters1(i) {
                i += 1;
                keep(adapter);
            }
            found.sort_by_key(|(_, desc)| std::cmp::Reverse(desc.DedicatedVideoMemory));
        }
        found
            .into_iter()
            .map(|(adapter, desc)| {
                let candidate = Candidate {
                    name: adapter_name(&desc.Description),
                    dedicated: desc.DedicatedVideoMemory,
                };
                (adapter, candidate, desc.VendorId)
            })
            .collect()
    }
}

/// The adapter the host device is made on, with its description for the log;
/// `None` where DXGI lists no hardware adapter (the default is then left to
/// the runtime).
fn chosen() -> Option<(IDXGIAdapter1, Candidate, u32)> {
    let mut list = hardware_adapters();
    if list.is_empty() {
        return None;
    }
    let mut place = 0;
    if let Ok(filter) = std::env::var("VICTORY_D3D12_ADAPTER") {
        let names: Vec<Candidate> = list.iter().map(|(_, c, _)| c.clone()).collect();
        match pick(&names, &filter) {
            Some(i) => place = i,
            None => tracing::warn!(
                "VICTORY_D3D12_ADAPTER={filter} names none of the {} hardware adapters ({}): the first choice is taken",
                names.len(),
                names.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", ")
            ),
        }
    }
    Some(list.swap_remove(place))
}

/// Makes `device` on the adapter the choice names (logged once: its name,
/// vendor and dedicated video memory), or on the runtime's default one where
/// there is no choice or the device cannot be made on it.
pub(crate) fn create_device(
    level: D3D_FEATURE_LEVEL,
    device: &mut Option<ID3D12Device>,
) -> windows::core::Result<()> {
    static LOGGED: OnceLock<()> = OnceLock::new();
    // SAFETY: device creation on an adapter DXGI listed.
    unsafe {
        if let Some((adapter, candidate, vendor)) = chosen() {
            match D3D12CreateDevice(&adapter, level, device) {
                Ok(()) => {
                    LOGGED.get_or_init(|| {
                        tracing::info!(
                            "host adapter: {} (vendor {vendor:#06x}, {} MiB dedicated video memory)",
                            candidate.name,
                            candidate.dedicated >> 20
                        );
                    });
                    return Ok(());
                }
                Err(e) => tracing::warn!(
                    "no device on the adapter {} ({e}): the runtime's default one is taken",
                    candidate.name
                ),
            }
        }
        D3D12CreateDevice(None, level, device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list() -> Vec<Candidate> {
        ["NVIDIA GeForce RTX 4070 Ti", "AMD Radeon(TM) Graphics"]
            .into_iter()
            .map(|name| Candidate { name: name.into(), dedicated: 0 })
            .collect()
    }

    #[test]
    fn a_name_ends_at_its_terminator() {
        assert_eq!(adapter_name(&[b'G' as u16, b'P' as u16, 0, b'X' as u16]), "GP");
        assert_eq!(adapter_name(&[b'A' as u16, b'B' as u16]), "AB");
    }

    #[test]
    fn the_filter_names_a_place_or_part_of_a_name() {
        assert_eq!(pick(&list(), "1"), Some(1));
        assert_eq!(pick(&list(), "2"), None);
        assert_eq!(pick(&list(), "radeon"), Some(1));
        assert_eq!(pick(&list(), " RTX "), Some(0));
        assert_eq!(pick(&list(), "intel"), None);
    }

    #[test]
    fn the_listed_adapters_are_not_software_ones() {
        for (adapter, _, _) in hardware_adapters() {
            // SAFETY: plain DXGI query.
            let desc = unsafe { adapter.GetDesc1().unwrap() };
            assert_eq!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32, 0);
        }
    }
}
