//! `gameinput.dll`: input devices from the host. Gamepads come from XInput
//! (host slots 0-3). The keyboard and mouse play as a pad (`controls`): on
//! the first connected host pad, or on one stand-in gamepad while none is
//! connected. The keyboard is a device of its own too. A poll thread turns
//! host state into readings (a new one per change) and runs device
//! callbacks on connect and disconnect; a title's pad read makes one too,
//! carrying the mouse motion since its last read.
#![allow(non_snake_case)]
// Title code calls these through the objects, passing pointers the real API
// would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::VecDeque;
use std::ffi::c_void;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, PoisonError};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use super::controls::{Controls, Counts, Settings};
use super::xasync::{E_INVALIDARG, E_NOTIMPL};
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleExport, HleModule};

/// The documented errors.
pub const DEVICE_DISCONNECTED: HRESULT = 0x838A_0001_u32 as i32;
pub const DEVICE_NOT_FOUND: HRESULT = 0x838A_0002_u32 as i32;
pub const READING_NOT_FOUND: HRESULT = 0x838A_0003_u32 as i32;
pub const REFERENCE_TOO_OLD: HRESULT = 0x838A_0004_u32 as i32;
const E_NOINTERFACE: HRESULT = 0x8000_4002_u32 as i32;

/// Input kinds (bits) our devices have.
const KEYBOARD_KIND: u32 = 0x10;
const GAMEPAD_KIND: u32 = 0x4_0000;
/// Device status of a connected device: connected, input enabled.
const STATUS_CONNECTED: u32 = 0x3;
const ASYNC_ENUMERATION: u32 = 1;

/// Gamepad buttons; the title maps each bit to one of its pad buttons.
pub(super) const MENU: u32 = 0x1;
pub(super) const VIEW: u32 = 0x2;
pub(super) const A: u32 = 0x4;
pub(super) const B: u32 = 0x8;
pub(super) const X: u32 = 0x10;
pub(super) const Y: u32 = 0x20;
pub(super) const UP: u32 = 0x40;
pub(super) const DOWN: u32 = 0x80;
pub(super) const LEFT: u32 = 0x100;
pub(super) const RIGHT: u32 = 0x200;
pub(super) const LEFT_SHOULDER: u32 = 0x400;
pub(super) const RIGHT_SHOULDER: u32 = 0x800;
pub(super) const LEFT_STICK: u32 = 0x1000;
pub(super) const RIGHT_STICK: u32 = 0x2000;

/// Devices: host pad slots, the keyboard's stand-in pad, the keyboard.
const PADS: usize = 4;
const FALLBACK: usize = PADS;
const KEYBOARD: usize = PADS + 1;
const DEVICES: usize = PADS + 2;

/// Readings kept per device for next/previous lookups.
const HISTORY: usize = 256;
const POLL: Duration = Duration::from_millis(8);

/// A gamepad reading: triggers 0 to 1, sticks -1 to 1 (up positive).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct PadState {
    pub(super) buttons: u32,
    pub(super) left_trigger: f32,
    pub(super) right_trigger: f32,
    pub(super) left_x: f32,
    pub(super) left_y: f32,
    pub(super) right_x: f32,
    pub(super) right_y: f32,
}

/// One key held down: scan code (0xE0xx when extended), character, key.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct KeyState {
    scan_code: u32,
    code_point: u32,
    virtual_key: u8,
    dead_key: bool,
}

const _: () = assert!(size_of::<KeyState>() == 12 && size_of::<PadState>() == 28);

/// Host input, or a fake in tests.
trait Source: Send {
    /// The pad in host slot `slot`, `None` when the slot is empty.
    fn pad(&mut self, slot: usize) -> Option<PadState>;
    /// Keys held down; none while the title's window lacks focus.
    fn keys(&mut self) -> Vec<KeyState>;
    /// Pad buttons pressed for the player (the automatic presses), with or
    /// without focus and whatever the key bindings.
    fn pressed(&mut self) -> u32 {
        0
    }
    /// Mouse buttons held down (`controls::MOUSE_BUTTONS`); none while the
    /// cursor is not captured.
    fn mouse_buttons(&mut self) -> Vec<u8> {
        Vec::new()
    }
    /// Motor speeds: low frequency (left), high frequency (right).
    fn rumble(&mut self, slot: usize, low: u16, high: u16);
}

#[repr(C)]
struct Table<const N: usize>([*const (); N]);
// SAFETY: function addresses only.
unsafe impl<const N: usize> Sync for Table<N> {}

/// The input object; one per hub.
#[repr(C)]
pub struct Object {
    vtable: &'static Table<22>,
    hub: *const Hub,
}

#[repr(C)]
pub struct Device {
    vtable: &'static Table<21>,
    hub: *const Hub,
    index: usize,
    info: DeviceInfo,
}

/// What a device says about itself. The title reads `id` (+0x20); the rest
/// follows the public layout and is zero where we have nothing to say.
#[repr(C)]
#[derive(Default)]
#[allow(dead_code)] // read by title code
struct DeviceInfo {
    size: u32,
    vendor: u16,
    product: u16,
    revision: u16,
    interface: u8,
    collection: u8,
    usage: [u16; 2],
    versions: [u16; 8],
    id: [u8; 32],
    root_id: [u8; 32],
    family: i32,
    capabilities: u32,
    supported_input: u32,
    rumble_motors: u32,
    /// Counts and optional detail pointers.
    rest: [u64; 26],
}

const _: () = assert!(std::mem::offset_of!(DeviceInfo, id) == 0x20);

fn kind(index: usize) -> u32 {
    if index == KEYBOARD {
        KEYBOARD_KIND
    } else {
        GAMEPAD_KIND
    }
}

impl Device {
    fn new(index: usize) -> Device {
        let mut id = [0; 32];
        id[..7].copy_from_slice(b"victory");
        id[31] = index as u8 + 1;
        let info = DeviceInfo {
            size: size_of::<DeviceInfo>() as u32,
            id,
            root_id: id,
            supported_input: kind(index),
            // Low and high frequency.
            rumble_motors: if index < PADS { 0x3 } else { 0 },
            ..Default::default()
        };
        Device {
            vtable: &DEVICE,
            hub: std::ptr::null(),
            index,
            info,
        }
    }

    fn hub(&self) -> &'static Hub {
        // SAFETY: set by `Hub::new`; hubs are never freed.
        unsafe { &*self.hub }
    }
}

#[derive(Clone, PartialEq)]
enum Value {
    Pad(PadState),
    Keys(Vec<KeyState>),
}

#[repr(C)]
pub struct Reading {
    vtable: &'static Table<25>,
    refs: AtomicU32,
    device: *const Device,
    sequence: u64,
    timestamp: u64,
    value: Value,
}

impl Reading {
    fn kind(&self) -> u32 {
        match self.value {
            Value::Pad(_) => GAMEPAD_KIND,
            Value::Keys(_) => KEYBOARD_KIND,
        }
    }
}

/// A counted reference to a reading.
struct Ref(NonNull<Reading>);

impl Ref {
    fn new(reading: Reading) -> Ref {
        Ref(NonNull::from(Box::leak(Box::new(reading))))
    }

    /// Another reference, for the title.
    fn share(&self) -> *const Reading {
        reading_add_ref(self.0.as_ptr());
        self.0.as_ptr()
    }
}

impl Deref for Ref {
    type Target = Reading;
    fn deref(&self) -> &Reading {
        // SAFETY: we hold a reference, so it is alive.
        unsafe { self.0.as_ref() }
    }
}

impl Drop for Ref {
    fn drop(&mut self) {
        reading_release(self.0.as_ptr());
    }
}

type DeviceCallback = extern "system" fn(u64, *mut c_void, *const Device, u64, u32, u32);

#[derive(Clone, Copy)]
struct Registration {
    token: u64,
    /// One device's address, or 0 for any.
    device: usize,
    kind: u32,
    status: u32,
    context: usize,
    callback: DeviceCallback,
}

impl Registration {
    fn matches(&self, devices: &[Device; DEVICES], index: usize) -> bool {
        kind(index) & self.kind != 0
            && (self.device == 0 || self.device == &devices[index] as *const Device as usize)
    }

    fn event(&self, device: usize, timestamp: u64, current: u32, previous: u32) -> Event {
        Event {
            registration: *self,
            device,
            timestamp,
            current,
            previous,
        }
    }
}

struct Event {
    registration: Registration,
    device: usize,
    timestamp: u64,
    current: u32,
    previous: u32,
}

#[derive(Default)]
struct Slot {
    connected: bool,
    sequence: u64,
    history: VecDeque<Ref>,
    latch: Latch,
}

/// Whether presses are latched ([`Latch`]): yes, unless
/// `VICTORY_INPUT_LATCH=0`.
fn latch_enabled() -> bool {
    latch_asked(std::env::var_os("VICTORY_INPUT_LATCH").as_deref())
}

fn latch_asked(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|v| v != "0")
}

/// Presses are latched only on a device the title read the current state
/// of within this many µs.
const LATCH_IDLE: u64 = 2_000_000;
/// The most presses a latch holds for reads to come.
const LATCH_QUEUE: usize = 16;

/// A title that reads only the current state once a frame misses a tap
/// that begins and ends between two frames: at our frame rates (a frame
/// every 90 to 160 ms) that is most taps, and a key had to be held or
/// hammered. So a button or key pressed since the title last read this
/// device's current state stays pressed in what we record until such a
/// read has seen it (presses from different polls one per read, in order,
/// so taps in sequence are no chord); that read then records the host's own
/// state, so the release follows in the next read. Only presses are held:
/// a release of a press the title already saw is recorded at once, and
/// sticks and triggers are always the latest value. Readers of the history
/// see every press and release as before, the release of a held press only
/// later (at the read).
///
/// This departs from what the real `GetCurrentReading` returns, which is
/// why it can be turned off (`VICTORY_INPUT_LATCH=0`); a device the title
/// has not read that way for [`LATCH_IDLE`] holds nothing.
#[derive(Default)]
struct Latch {
    /// When a current-state read last handed out this device's reading.
    read: Option<u64>,
    /// The host's last state, without held presses.
    raw: Option<Value>,
    /// Presses since that read, oldest first, each the buttons or keys that
    /// went down in one poll: each read takes the oldest.
    buttons: VecDeque<u32>,
    keys: VecDeque<Vec<KeyState>>,
}

fn same_key(a: &KeyState, b: &KeyState) -> bool {
    a.virtual_key == b.virtual_key && a.scan_code == b.scan_code
}

impl Latch {
    /// `value` from the host, with the presses since the last read held.
    fn hold(&mut self, value: Value, now: u64, enabled: bool) -> Value {
        let reader = self.read.is_some_and(|t| now.saturating_sub(t) < LATCH_IDLE);
        if !enabled || !reader {
            self.buttons.clear();
            self.keys.clear();
        } else {
            match (&value, &self.raw) {
                (Value::Pad(pad), raw) => {
                    let before = match raw {
                        Some(Value::Pad(raw)) => raw.buttons,
                        _ => 0,
                    };
                    let pressed = pad.buttons & !before;
                    if pressed != 0 && self.buttons.len() < LATCH_QUEUE {
                        self.buttons.push_back(pressed);
                    }
                }
                (Value::Keys(keys), raw) => {
                    let before: &[KeyState] = match raw {
                        Some(Value::Keys(raw)) => raw,
                        _ => &[],
                    };
                    let pressed: Vec<KeyState> = keys
                        .iter()
                        .filter(|k| !before.iter().any(|b| same_key(b, k)))
                        .copied()
                        .collect();
                    if !pressed.is_empty() && self.keys.len() < LATCH_QUEUE {
                        self.keys.push_back(pressed);
                    }
                }
            }
        }
        self.raw = Some(value.clone());
        self.apply(value)
    }

    /// `value` as the next read is to see it: the oldest held press added,
    /// presses that came after it left for later reads.
    fn apply(&self, value: Value) -> Value {
        match value {
            Value::Pad(mut pad) => {
                let later = self.buttons.iter().skip(1).fold(0, |a, b| a | b);
                let first = self.buttons.front().copied().unwrap_or(0);
                pad.buttons = (pad.buttons & !(later & !first)) | first;
                Value::Pad(pad)
            }
            Value::Keys(mut keys) => {
                let first: &[KeyState] = self.keys.front().map_or(&[], |k| k);
                let later = self.keys.iter().skip(1).flatten();
                let later: Vec<&KeyState> =
                    later.filter(|l| !first.iter().any(|f| same_key(f, l))).collect();
                keys.retain(|k| !later.iter().any(|l| same_key(l, k)));
                for key in first {
                    if !keys.iter().any(|k| same_key(k, key)) {
                        keys.push(*key);
                    }
                }
                Value::Keys(keys)
            }
        }
    }

    /// A current-state read saw the device: the oldest held press is let
    /// go, and the reading the next read is to see is returned when it
    /// differs from what was recorded.
    fn seen(&mut self, now: u64, recorded: &Value) -> Option<Value> {
        self.read = Some(now);
        // Diagnostics: presses held only by the latch (released again
        // before this read), which the read would have missed without it.
        #[cfg(feature = "diagnostics")]
        match &self.raw {
            Some(Value::Pad(raw)) => {
                if let Some(&first) = self.buttons.front()
                    && first & !raw.buttons != 0
                {
                    let buttons = format_args!("{:#x}", first & !raw.buttons);
                    tracing::info!(buttons, "a tap kept for the title's read (released before it)");
                }
            }
            Some(Value::Keys(raw)) => {
                let tapped = self.keys.front().into_iter().flatten();
                for key in tapped.filter(|k| !raw.iter().any(|r| same_key(r, k))) {
                    let virtual_key = key.virtual_key;
                    tracing::info!(virtual_key, "a key tap kept for the title's read (released before it)");
                }
            }
            None => {}
        }
        self.buttons.pop_front();
        self.keys.pop_front();
        let next = self.raw.clone().map(|raw| self.apply(raw))?;
        (&next != recorded).then_some(next)
    }
}

impl Slot {
    fn record(&mut self, devices: &[Device; DEVICES], index: usize, value: Value, now: u64) {
        self.sequence += 1;
        if self.history.len() == HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(Ref::new(Reading {
            vtable: &READING,
            refs: AtomicU32::new(1),
            device: &devices[index],
            sequence: self.sequence,
            timestamp: now,
            value,
        }));
    }

    /// A current-state read handed out this device's newest reading: a
    /// press held for it is let go now.
    fn seen(&mut self, devices: &[Device; DEVICES], index: usize) {
        let now = now();
        let Some(newest) = self.history.back() else {
            return;
        };
        if let Some(raw) = self.latch.seen(now, &newest.value) {
            self.record(devices, index, raw, now);
        }
    }
}

struct State {
    slots: [Slot; DEVICES],
    /// Host slots whose connection has reported input.
    seen_input: [bool; PADS],
    callbacks: Vec<Registration>,
    /// Asynchronous enumerations, delivered by the next poll.
    pending: Vec<Event>,
    /// Device callbacks running now: token and thread.
    running: Vec<(u64, ThreadId)>,
    /// Presses are latched ([`Latch`]).
    latch: bool,
    /// The keyboard and mouse as a pad.
    controls: Controls,
    /// The last poll's host pads (those that reported input) and the
    /// buttons pressed for the player, for pad reads in between.
    host_pads: [Option<PadState>; PADS],
    pressed: u32,
}

pub struct Hub {
    object: Object,
    devices: [Device; DEVICES],
    /// Apart from the state: XInput can be slow, and the title reads
    /// readings every frame.
    source: Mutex<Box<dyn Source>>,
    /// Mouse motion and wheel steps, added by the host's raw input thread.
    counts: Arc<Counts>,
    state: Mutex<State>,
    /// A device callback returned.
    returned: Condvar,
}

fn lock<T: ?Sized>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

// SAFETY: the back pointers are set before the hub is shared and never
// change; what changes sits behind the mutex or is atomic.
unsafe impl Send for Hub {}
// SAFETY: as above.
unsafe impl Sync for Hub {}

static START: LazyLock<Instant> = LazyLock::new(Instant::now);

/// Microseconds, the clock of every timestamp we hand out.
fn now() -> u64 {
    START.elapsed().as_micros() as u64
}

impl Hub {
    fn new(source: Box<dyn Source>, counts: Arc<Counts>, settings: Settings) -> &'static Hub {
        cursor::ENABLED.store(settings.capture, Ordering::Relaxed);
        let hub = Box::into_raw(Box::new(Hub {
            object: Object {
                vtable: &INPUT,
                hub: std::ptr::null(),
            },
            devices: std::array::from_fn(Device::new),
            source: Mutex::new(source),
            counts,
            state: Mutex::new(State {
                slots: Default::default(),
                seen_input: [false; PADS],
                callbacks: Vec::new(),
                pending: Vec::new(),
                running: Vec::new(),
                latch: latch_enabled(),
                controls: Controls::new(settings),
                host_pads: [None; PADS],
                pressed: 0,
            }),
            returned: Condvar::new(),
        }));
        // SAFETY: just allocated and never freed; nothing else sees it yet.
        unsafe {
            (*hub).object.hub = hub;
            for device in &mut (*hub).devices {
                device.hub = hub;
            }
            &*hub
        }
    }

    /// A hub over the host's input, polled from now on.
    fn start(source: Box<dyn Source>, counts: Arc<Counts>, settings: Settings) -> &'static Hub {
        let hub = Hub::new(source, counts, settings);
        hub.poll();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(POLL);
                hub.poll();
            }
        });
        hub
    }

    fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// Reads the host once; device callbacks run on the calling thread.
    fn poll(&self) {
        let (pads, keys, buttons, pressed) = {
            let mut source = lock(&self.source);
            let pads: [_; PADS] = std::array::from_fn(|slot| source.pad(slot));
            (pads, source.keys(), source.mouse_buttons(), source.pressed())
        };
        let events = {
            let mut state = self.state();
            let polled = Polled {
                pads,
                keys,
                buttons: &buttons,
                pressed,
            };
            let events = state.poll(&self.devices, polled, &self.counts);
            if std::mem::take(&mut state.controls.toggle_capture) {
                cursor::toggle();
            }
            events
        };
        self.deliver(events);
    }

    /// Runs callbacks, never under our lock: they call back into us. One
    /// unregistered since the event was taken does not run, and
    /// `UnregisterCallback` waits for one that runs.
    fn deliver(&self, events: Vec<Event>) {
        let me = std::thread::current().id();
        for e in events {
            let r = e.registration;
            {
                let mut state = self.state();
                if !state.callbacks.iter().any(|c| c.token == r.token) {
                    continue;
                }
                state.running.push((r.token, me));
            }
            (r.callback)(
                r.token,
                r.context as *mut c_void,
                &self.devices[e.device],
                e.timestamp,
                e.current,
                e.previous,
            );
            let mut state = self.state();
            if let Some(i) = state.running.iter().position(|&x| x == (r.token, me)) {
                state.running.swap_remove(i);
            }
            self.returned.notify_all();
        }
    }

    fn index(&self, device: *const Device) -> Option<usize> {
        self.devices.iter().position(|d| std::ptr::eq(d, device))
    }
}

/// What one poll read from the host.
struct Polled<'a> {
    pads: [Option<PadState>; PADS],
    keys: Vec<KeyState>,
    buttons: &'a [u8],
    pressed: u32,
}

impl State {
    fn poll(&mut self, devices: &[Device; DEVICES], host: Polled, counts: &Counts) -> Vec<Event> {
        let now = now();
        let mut events = std::mem::take(&mut self.pending);
        // A wireless receiver answers for a switched-off pad with an all-zero
        // state: a slot counts as connected once its connection has reported
        // input, so such a receiver leaves the stand-in pad in place.
        self.host_pads = std::array::from_fn(|slot| {
            let Some(pad) = host.pads[slot] else {
                self.seen_input[slot] = false;
                return None;
            };
            self.seen_input[slot] |= pad != PadState::default();
            self.seen_input[slot].then_some(pad)
        });
        self.pressed = host.pressed;
        let by_position = self.controls.s.by_position;
        let held: Vec<u8> = host
            .keys
            .iter()
            .map(|k| {
                by_position
                    .then(|| super::controls::us_key(k.scan_code))
                    .flatten()
                    .unwrap_or(k.virtual_key)
            })
            .chain(host.buttons.iter().copied())
            .collect();
        self.controls.feed(&held, counts);
        self.publish(devices, now, None, &mut events);
        self.update(devices, KEYBOARD, Some(Value::Keys(host.keys)), now, &mut events);
        events
    }

    /// The pad devices' readings: the host pads, with the keyboard and mouse
    /// (and the buttons pressed for the player) on the first connected one,
    /// or else on the stand-in pad. `take`: a title read, which turns the
    /// mouse counts since the last one into the right stick.
    fn publish(
        &mut self,
        devices: &[Device; DEVICES],
        now: u64,
        take: Option<&Counts>,
        events: &mut Vec<Event>,
    ) {
        let pads = self.host_pads;
        let first = pads.iter().position(Option::is_some);
        let mut values = [None; PADS + 1];
        values[..PADS].copy_from_slice(&pads);
        let base = first.and_then(|i| pads[i]);
        let mut pad = match take {
            Some(counts) => self.controls.take(now, base, counts),
            None => self.controls.peek(base),
        };
        pad.buttons |= self.pressed;
        values[first.unwrap_or(FALLBACK)] = Some(pad);
        for (index, value) in values.into_iter().enumerate() {
            self.update(devices, index, value.map(Value::Pad), now, events);
        }
    }

    /// A title read of a pad: a new reading with the mouse motion since the
    /// last read.
    fn title_read(&mut self, devices: &[Device; DEVICES], counts: &Counts) {
        let mut events = Vec::new();
        self.publish(devices, now(), Some(counts), &mut events);
        // Connections only change on host polls; keep anything for the next.
        self.pending.append(&mut events);
    }

    /// Records a device's state: a reading when it changed, callbacks when
    /// the device came or went.
    fn update(
        &mut self,
        devices: &[Device; DEVICES],
        index: usize,
        value: Option<Value>,
        now: u64,
        events: &mut Vec<Event>,
    ) {
        let latch = self.latch;
        let slot = &mut self.slots[index];
        let was = slot.connected;
        slot.connected = value.is_some();
        if !was {
            slot.latch = Latch::default();
        }
        if let Some(value) = value.map(|v| slot.latch.hold(v, now, latch))
            && (!was || slot.history.back().is_none_or(|r| r.value != value))
        {
            slot.record(devices, index, value, now);
        }
        if was != slot.connected {
            tracing::info!(device = index, connected = !was, "input device");
            let (current, previous) = if was {
                (0, STATUS_CONNECTED)
            } else {
                (STATUS_CONNECTED, 0)
            };
            events.extend(
                self.callbacks
                    .iter()
                    .filter(|r| r.matches(devices, index) && r.status & STATUS_CONNECTED != 0)
                    .map(|r| r.event(index, now, current, previous)),
            );
        }
    }
}

#[cfg(windows)]
fn host(counts: &Arc<Counts>, settings: &Settings) -> Box<dyn Source> {
    if settings.capture {
        host::raw_mouse(counts.clone());
    }
    Box::new(host::Host::new(counts.clone()))
}

#[cfg(not(windows))]
fn host(_: &Arc<Counts>, _: &Settings) -> Box<dyn Source> {
    struct Nothing;
    impl Source for Nothing {
        fn pad(&mut self, _: usize) -> Option<PadState> {
            None
        }
        fn keys(&mut self) -> Vec<KeyState> {
            Vec::new()
        }
        fn rumble(&mut self, _: usize, _: u16, _: u16) {}
    }
    Box::new(Nothing)
}

static HUB: LazyLock<&'static Hub> = LazyLock::new(|| {
    let settings = super::controls::settings();
    let counts = Arc::new(Counts::default());
    Hub::start(host(&counts, &settings), counts, settings)
});

/// The mouse cursor while the title window has focus: captured (hidden,
/// held inside the window, its motion turns the camera) or free. A click
/// in the window captures it (focus alone does not, so starting the title
/// or switching to it leaves the cursor alone); Esc frees it and captures
/// it again, losing focus or sizing the window frees it. The host poll and
/// the title window's procedure (`lifecycle`) keep it.
pub(crate) mod cursor {
    use std::sync::atomic::{AtomicBool, Ordering};

    /// Capture is wanted at all (`[controls] mouse_capture`).
    pub(crate) static ENABLED: AtomicBool = AtomicBool::new(false);
    pub(crate) static CAPTURED: AtomicBool = AtomicBool::new(false);
    /// The title window has focus (kept by the host poll).
    pub(crate) static FOCUSED: AtomicBool = AtomicBool::new(false);
    /// The user is sizing or moving the window (between `WM_ENTERSIZEMOVE`
    /// and `WM_EXITSIZEMOVE`): the cursor is theirs.
    static SIZING: AtomicBool = AtomicBool::new(false);
    /// How far inside the client area a captured cursor is held, so it
    /// cannot reach the window's borders and start a size loop.
    const INSET: i32 = 16;

    pub(crate) fn captured() -> bool {
        CAPTURED.load(Ordering::Relaxed)
    }

    fn sizing() -> bool {
        SIZING.load(Ordering::Relaxed)
    }

    /// The window entered (`true`) or left a size or move loop. A captured
    /// cursor is freed first: clipped to a client area that shrinks under
    /// it, it would drag the window down to its title bar. Afterwards a
    /// click in the window captures it again.
    pub(crate) fn size_move(entered: bool) {
        if entered {
            release();
        }
        SIZING.store(entered, Ordering::Relaxed);
    }

    /// Whether the cursor is held in the window now.
    pub(crate) fn clips(focused: bool) -> bool {
        focused && captured() && !sizing()
    }

    /// Where a captured cursor is held: the client rectangle (left, top,
    /// right, bottom) shrunk by [`INSET`]; `None` for one too small for that.
    pub(crate) fn clip_rect(client: (i32, i32, i32, i32)) -> Option<(i32, i32, i32, i32)> {
        let (l, t, r, b) = client;
        let rect = (l + INSET, t + INSET, r - INSET, b - INSET);
        (rect.2 > rect.0 && rect.3 > rect.1).then_some(rect)
    }

    /// Whether a capture may happen now: when capture is wanted at all
    /// (`enabled`) and the window is not being sized or moved.
    fn may_capture(enabled: bool) -> bool {
        enabled && !sizing()
    }

    fn capture() {
        if may_capture(ENABLED.load(Ordering::Relaxed)) && !CAPTURED.swap(true, Ordering::Relaxed) {
            tracing::info!("mouse cursor captured");
        }
    }

    pub(crate) fn release() {
        if CAPTURED.swap(false, Ordering::Relaxed) {
            tracing::info!("mouse cursor released");
        }
    }

    /// A click in the title window's client area captures. Buttons held
    /// when it is captured are no shots until they are let go (`host`).
    pub(crate) fn click() {
        capture();
    }

    /// The release key.
    pub(crate) fn toggle() {
        if captured() { release() } else { capture() }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The cursor's rules around a size or move loop: dragging a
        /// border with a captured cursor collapsed the window in the
        /// source, the clip following the shrinking client area.
        #[test]
        fn a_size_or_move_loop_frees_the_cursor() {
            assert_eq!(clip_rect((100, 50, 900, 650)), Some((116, 66, 884, 634)));
            assert_eq!(clip_rect((0, 0, 1339, 30)), None, "a strip: no clip");
            // The capture flag is set directly: other tests' hubs set
            // whether capture is wanted at all.
            CAPTURED.store(true, Ordering::Relaxed);
            assert!(clips(true) && may_capture(true));
            assert!(!clips(false), "no clip without focus");
            size_move(true);
            assert!(!captured() && !clips(true));
            assert!(!may_capture(true), "no capture during the loop");
            size_move(false);
            assert!(!captured(), "still free after the loop");
            assert!(may_capture(true), "a click in the window captures again");
            assert!(!may_capture(false));
        }
    }
}

type This = *const Object;
type Out = *mut *const c_void;

/// The DLL's only export (ordinal 1): the input object, static and shared.
pub extern "system" fn GameInputCreate(out: *mut This) -> HRESULT {
    tracing::info!("GameInputCreate");
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null; the title passes room for a pointer.
    unsafe { out.write(&HUB.object) };
    S_OK
}

fn hub(this: This) -> &'static Hub {
    // SAFETY: `this` is a hub's object; hubs are never freed.
    unsafe { &*(*this).hub }
}

/// Clears a COM out pointer and returns `hr`.
fn fail(out: Out, hr: HRESULT) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a pointer.
        unsafe { out.write(std::ptr::null()) };
    }
    hr
}

/// Gives the title a reference to `reading`, or fails as with none.
fn hand(out: Out, reading: Option<&Ref>) -> HRESULT {
    match reading {
        _ if out.is_null() => E_INVALIDARG,
        Some(reading) => {
            // SAFETY: checked non-null; the title passes room for a pointer.
            unsafe { out.write(reading.share().cast()) };
            S_OK
        }
        None => fail(out, READING_NOT_FOUND),
    }
}

extern "system" fn no_interface(_: *const c_void, _iid: *const c_void, out: Out) -> HRESULT {
    tracing::warn!("input QueryInterface not implemented");
    fail(out, E_NOINTERFACE)
}
/// The input object and devices live as long as the process.
extern "system" fn add_ref(_: *const c_void) -> u32 {
    1
}

extern "system" fn timestamp(_: This) -> u64 {
    now()
}

extern "system" fn current_reading(
    this: This,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let hub = hub(this);
    let mut state = hub.state();
    if kind & GAMEPAD_KIND != 0 {
        state.title_read(&hub.devices, &hub.counts);
    }
    let newest = |state: &State, i: usize| {
        state.slots[i]
            .history
            .back()
            .is_some_and(|r| r.kind() & kind != 0)
            .then_some(i)
    };
    let found = if device.is_null() {
        (0..DEVICES)
            .filter(|&i| state.slots[i].connected)
            .filter_map(|i| newest(&state, i))
            .max_by_key(|&i| state.slots[i].history.back().map(|r| r.timestamp))
    } else {
        match hub.index(device) {
            None => return fail(out, E_INVALIDARG),
            Some(i) if !state.slots[i].connected => return fail(out, DEVICE_DISCONNECTED),
            Some(i) => newest(&state, i),
        }
    };
    tracing::trace!(
        kind = format_args!("{kind:#x}"),
        device = ?hub.index(device),
        sequence = found.and_then(|i| state.slots[i].history.back()).map(|r| r.sequence),
        "current reading"
    );
    let Some(index) = found else {
        return hand(out, None);
    };
    let result = hand(out, state.slots[index].history.back());
    if result == S_OK {
        // What the title has now seen is let go of.
        state.slots[index].seen(&hub.devices, index);
    }
    result
}

/// The reading after (`forward`) or before `reference`: on `device`, or with
/// none given on any device, in time order (ties by device, then sequence).
fn relative(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
    forward: bool,
) -> HRESULT {
    if reference.is_null() {
        return fail(out, E_INVALIDARG);
    }
    let hub = hub(this);
    // SAFETY: a reading we handed out and the title still holds.
    let reference = unsafe { &*reference };
    let Some(index) = hub.index(reference.device) else {
        return fail(out, E_INVALIDARG);
    };
    if !device.is_null() && !std::ptr::eq(device, reference.device) {
        return fail(out, E_INVALIDARG);
    }
    let state = hub.state();
    if state.slots[index]
        .history
        .front()
        .is_none_or(|oldest| reference.sequence < oldest.sequence)
    {
        return fail(out, REFERENCE_TOO_OLD);
    }
    // On one device this order is its sequence order.
    let at = (reference.timestamp, index, reference.sequence);
    let devices = if device.is_null() {
        0..DEVICES
    } else {
        index..index + 1
    };
    let candidates = devices
        .flat_map(|i| {
            let history = &state.slots[i].history;
            history
                .iter()
                .map(move |r| ((r.timestamp, i, r.sequence), r))
        })
        .filter(|(_, r)| r.kind() & kind != 0);
    let found = if forward {
        candidates
            .filter(|(key, _)| *key > at)
            .min_by_key(|(key, _)| *key)
    } else {
        candidates
            .filter(|(key, _)| *key < at)
            .max_by_key(|(key, _)| *key)
    };
    hand(out, found.map(|(_, r)| r))
}
extern "system" fn next_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    relative(this, reference, kind, device, out, true)
}
extern "system" fn previous_reading(
    this: This,
    reference: *const Reading,
    kind: u32,
    device: *const Device,
    out: Out,
) -> HRESULT {
    relative(this, reference, kind, device, out, false)
}
/// The device's reading at `time`, among those still kept.
extern "system" fn temporal_reading(
    this: This,
    time: u64,
    device: *const Device,
    out: Out,
) -> HRESULT {
    let hub = hub(this);
    let Some(index) = hub.index(device) else {
        return fail(out, E_INVALIDARG);
    };
    let state = hub.state();
    let history = &state.slots[index].history;
    hand(out, history.iter().rev().find(|r| r.timestamp <= time))
}

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

fn token(out: *mut u64) -> HRESULT {
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(NEXT_TOKEN.fetch_add(1, Ordering::Relaxed)) };
    }
    S_OK
}

// shortcut: reading callbacks never run; the title registers none
// (it polls).
extern "system" fn register_reading(
    _: This,
    _device: *const c_void,
    _kind: u32,
    _threshold: f32,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    tracing::warn!("input reading callbacks are never called");
    token(out)
}

/// Device callbacks run on our poll thread when a device comes or goes.
/// Enumeration reports the devices already there: blocking on the caller's
/// thread before we return (the title registers holding a lock its callback
/// takes again), asynchronous on the next poll.
extern "system" fn register_device(
    this: This,
    device: *const Device,
    kind: u32,
    status: u32,
    enumeration: u32,
    context: *mut c_void,
    callback: Option<DeviceCallback>,
    out: *mut u64,
) -> HRESULT {
    let Some(callback) = callback else {
        return E_INVALIDARG;
    };
    let hub = hub(this);
    let registration = Registration {
        token: NEXT_TOKEN.fetch_add(1, Ordering::Relaxed),
        device: device as usize,
        kind,
        status,
        context: context as usize,
        callback,
    };
    let mut found = Vec::new();
    {
        let mut state = hub.state();
        if enumeration != 0 && status & STATUS_CONNECTED != 0 {
            let now = now();
            found = (0..DEVICES)
                .filter(|&i| state.slots[i].connected && registration.matches(&hub.devices, i))
                .map(|i| registration.event(i, now, STATUS_CONNECTED, 0))
                .collect();
        }
        tracing::info!(
            kind = format_args!("{kind:#x}"),
            enumeration,
            devices = found.len(),
            "input device callback"
        );
        state.callbacks.push(registration);
        if enumeration == ASYNC_ENUMERATION {
            state.pending.append(&mut found);
        }
    }
    if !out.is_null() {
        // SAFETY: checked non-null.
        unsafe { out.write(registration.token) };
    }
    hub.deliver(found);
    S_OK
}
/// Guide button and keyboard layout callbacks: neither ever changes here.
extern "system" fn register_other(
    _: This,
    _device: *const c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    token(out)
}

fn forget(this: This, token: u64) {
    let mut state = hub(this).state();
    state.callbacks.retain(|r| r.token != token);
    state.pending.retain(|e| e.registration.token != token);
}
extern "system" fn stop_callback(this: This, token: u64) {
    forget(this, token);
}
/// Stops the callback, then waits up to `timeout` µs for a call of it that
/// is running; false on time-out, or from inside that callback (it cannot
/// wait for itself), as documented. Either way it is never called again.
extern "system" fn unregister_callback(this: This, token: u64, timeout: u64) -> bool {
    forget(this, token);
    let hub = hub(this);
    let me = std::thread::current().id();
    let deadline = Instant::now().checked_add(Duration::from_micros(timeout));
    let mut state = hub.state();
    loop {
        let running = state.running.iter().filter(|r| r.0 == token);
        if running.clone().any(|r| r.1 == me) {
            return false;
        }
        if running.count() == 0 {
            return true;
        }
        state = match deadline {
            None => hub.returned.wait(state).unwrap_or_else(PoisonError::into_inner),
            Some(deadline) => {
                let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                    return false;
                };
                let waited = hub.returned.wait_timeout(state, left);
                waited.unwrap_or_else(PoisonError::into_inner).0
            }
        };
    }
}

/// Dispatcher and aggregate device: no title of ours asks for them.
extern "system" fn create_dispatcher(_: This, out: Out) -> HRESULT {
    tracing::warn!("input dispatcher not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn create_aggregate(_: This, _kind: u32, out: Out) -> HRESULT {
    tracing::warn!("aggregate input device not implemented");
    fail(out, E_NOTIMPL)
}
extern "system" fn find_by_id(this: This, id: *const [u8; 32], out: Out) -> HRESULT {
    if id.is_null() || out.is_null() {
        return fail(out, E_INVALIDARG);
    }
    // SAFETY: checked non-null; the title passes a device id.
    let id = unsafe { &*id };
    match hub(this).devices.iter().find(|d| d.info.id == *id) {
        Some(device) => {
            // SAFETY: checked non-null.
            unsafe { out.write((device as *const Device).cast()) };
            S_OK
        }
        None => fail(out, DEVICE_NOT_FOUND),
    }
}
/// Lookup by object, platform handle or platform string: our devices have
/// none of these.
extern "system" fn find_device(_: This, _value: *const c_void, out: Out) -> HRESULT {
    fail(out, DEVICE_NOT_FOUND)
}
extern "system" fn enable_oem_device(
    _: This,
    _vendor: u16,
    _product: u16,
    _interface: u8,
    _collection: u8,
) -> HRESULT {
    S_OK
}
extern "system" fn set_focus_policy(_: This, _policy: u32) {}

static INPUT: Table<22> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    timestamp as *const (),           // 0x18
    current_reading as *const (),     // 0x20
    next_reading as *const (),        // 0x28
    previous_reading as *const (),    // 0x30
    temporal_reading as *const (),    // 0x38
    register_reading as *const (),    // 0x40
    register_device as *const (),     // 0x48
    register_other as *const (),      // 0x50 guide button
    register_other as *const (),      // 0x58 keyboard layout
    stop_callback as *const (),       // 0x60
    unregister_callback as *const (), // 0x68
    create_dispatcher as *const (),   // 0x70
    create_aggregate as *const (),    // 0x78
    find_by_id as *const (),          // 0x80
    find_device as *const (),         // 0x88 by object
    find_device as *const (),         // 0x90 by platform handle
    find_device as *const (),         // 0x98 by platform string
    enable_oem_device as *const (),   // 0xa0
    set_focus_policy as *const (),    // 0xa8
]);

extern "system" fn device_info(this: *const Device) -> *const DeviceInfo {
    // SAFETY: one of our devices.
    unsafe { &(*this).info }
}
extern "system" fn device_status(this: *const Device) -> u32 {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    if device.hub().state().slots[device.index].connected {
        STATUS_CONNECTED
    } else {
        0
    }
}
// shortcut: trigger motors are dropped (XInput has none). The host API that
// drives them without a redistributable is Windows.Gaming.Input (WinRT
// `Gamepad.Vibration`), which needs WinRT bindings this crate lacks outside
// diagnostics builds; the title sends trigger levels in slot 0x50.
extern "system" fn set_rumble(this: *const Device, params: *const [f32; 4]) {
    // SAFETY: one of our devices.
    let device = unsafe { &*this };
    if device.index >= PADS {
        return;
    }
    let [low, high, ..] = if params.is_null() {
        [0.0; 4]
    } else {
        // SAFETY: checked non-null; four motor levels.
        unsafe { *params }
    };
    let speed = |level: f32| (level.clamp(0.0, 1.0) * f32::from(u16::MAX)) as u16;
    lock(&device.hub().source).rumble(device.index, speed(low), speed(high));
}
extern "system" fn unsupported<const SLOT: usize>(_: *const c_void) -> HRESULT {
    tracing::warn!(
        slot = format_args!("{SLOT:#x}"),
        "input device method not implemented"
    );
    E_NOTIMPL
}

static DEVICE: Table<21> = Table([
    no_interface as *const (),        // 0x00
    add_ref as *const (),             // 0x08 AddRef
    add_ref as *const (),             // 0x10 Release
    device_info as *const (),         // 0x18
    device_status as *const (),       // 0x20
    unsupported::<0x28> as *const (), // battery
    unsupported::<0x30> as *const (), // force feedback
    unsupported::<0x38> as *const (), //
    unsupported::<0x40> as *const (), //
    unsupported::<0x48> as *const (), // haptics
    set_rumble as *const (),          // 0x50
    unsupported::<0x58> as *const (), // input synchronization
    unsupported::<0x60> as *const (), //
    unsupported::<0x68> as *const (), // power off
    unsupported::<0x70> as *const (), // raw device access
    unsupported::<0x78> as *const (), //
    unsupported::<0x80> as *const (), //
    unsupported::<0x88> as *const (), //
    unsupported::<0x90> as *const (), //
    unsupported::<0x98> as *const (), //
    unsupported::<0xa0> as *const (), //
]);

fn reading<'a>(this: *const Reading) -> &'a Reading {
    // SAFETY: a reading we handed out and the title still holds.
    unsafe { &*this }
}
extern "system" fn reading_add_ref(this: *const Reading) -> u32 {
    reading(this).refs.fetch_add(1, Ordering::Relaxed) + 1
}
extern "system" fn reading_release(this: *const Reading) -> u32 {
    let left = reading(this).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if left == 0 {
        // SAFETY: the last reference is gone; `Ref::new` boxed it.
        drop(unsafe { Box::from_raw(this.cast_mut()) });
    }
    left
}
extern "system" fn input_kind(this: *const Reading) -> u32 {
    reading(this).kind()
}
extern "system" fn sequence_number(this: *const Reading, kind: u32) -> u64 {
    let reading = reading(this);
    if reading.kind() & kind != 0 {
        reading.sequence
    } else {
        0
    }
}
extern "system" fn reading_timestamp(this: *const Reading) -> u64 {
    reading(this).timestamp
}
extern "system" fn reading_device(this: *const Reading, out: *mut *const Device) {
    if !out.is_null() {
        // SAFETY: checked non-null; devices need no reference.
        unsafe { out.write(reading(this).device) };
    }
}
extern "system" fn key_count(this: *const Reading) -> u32 {
    match &reading(this).value {
        Value::Keys(keys) => keys.len() as u32,
        Value::Pad(_) => 0,
    }
}
extern "system" fn key_state(this: *const Reading, count: u32, out: *mut KeyState) -> u32 {
    let Value::Keys(keys) = &reading(this).value else {
        return 0;
    };
    if out.is_null() {
        return 0;
    }
    let n = keys.len().min(count as usize);
    // SAFETY: the title passes room for `count` keys.
    unsafe { std::ptr::copy_nonoverlapping(keys.as_ptr(), out, n) };
    n as u32
}
extern "system" fn gamepad_state(this: *const Reading, out: *mut PadState) -> bool {
    match &reading(this).value {
        Value::Pad(state) if !out.is_null() => {
            log_title_read(reading(this), state);
            // SAFETY: checked non-null.
            unsafe { out.write(*state) };
            true
        }
        _ => false,
    }
}
/// What the title reads, when it differs from what it read before:
/// `VICTORY_LOG=victory_hle::modules::gameinput=debug`.
fn log_title_read(reading: &Reading, state: &PadState) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    static LAST: Mutex<Option<PadState>> = Mutex::new(None);
    let mut last = lock(&LAST);
    if *last != Some(*state) {
        *last = Some(*state);
        // SAFETY: readings point at one of their hub's devices.
        let device = unsafe { (*reading.device).index };
        tracing::debug!(
            device,
            sequence = reading.sequence,
            buttons = format_args!("{:#06x}", state.buttons),
            lt = state.left_trigger,
            rt = state.right_trigger,
            left = format_args!("{:+.3} {:+.3}", state.left_x, state.left_y),
            right = format_args!("{:+.3} {:+.3}", state.right_x, state.right_y),
            "title read pad"
        );
    }
}
/// Kinds our devices do not have: no count, no state.
extern "system" fn nothing(_: *const c_void) -> usize {
    0
}

static READING: Table<25> = Table([
    no_interface as *const (),      // 0x00
    reading_add_ref as *const (),   // 0x08
    reading_release as *const (),   // 0x10
    input_kind as *const (),        // 0x18
    sequence_number as *const (),   // 0x20
    reading_timestamp as *const (), // 0x28
    reading_device as *const (),    // 0x30
    nothing as *const (),           // 0x38 raw report
    nothing as *const (),           // 0x40 controller axes, buttons, switches
    nothing as *const (),           // 0x48
    nothing as *const (),           // 0x50
    nothing as *const (),           // 0x58
    nothing as *const (),           // 0x60
    nothing as *const (),           // 0x68
    key_count as *const (),         // 0x70
    key_state as *const (),         // 0x78
    nothing as *const (),           // 0x80 mouse
    nothing as *const (),           // 0x88 touch
    nothing as *const (),           // 0x90
    nothing as *const (),           // 0x98 motion
    nothing as *const (),           // 0xa0 arcade stick
    nothing as *const (),           // 0xa8 flight stick
    gamepad_state as *const (),     // 0xb0
    nothing as *const (),           // 0xb8 racing wheel
    nothing as *const (),           // 0xc0 UI navigation
]);

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "gameinput",
        dlls: &["gameinput.dll"],
        // Titles import it by ordinal only.
        exports: vec![HleExport {
            name: "#1",
            address: GameInputCreate as *const () as usize,
        }],
    }
}

/// `VICTORY_AUTO_START=1`: the start screen's "press A" is answered for the
/// player. The title adds its user ([`user_added`], `XUserAddAsync`) just
/// before it shows that screen; from then on A (Enter on the keyboard) is
/// pressed for 0.2 s every 2 s, until the title opens its next file
/// ([`title_moved_on`]: it left the screen) or 30 s have passed.
///
/// The title (2026) ignores a press before the screen is up (presses 2 and
/// 4 s after the user was added were read and dropped, the one at 6 s was
/// taken), and presses during the film that follows change nothing (it ran
/// its 30 s either way). Its first-start menus (settings, privacy) come only
/// after the screen and that film, so no press reaches them.
/// `VICTORY_AUTO_SKIP=<n>`: holds A through the first `n` scenes a title
/// lets the player skip that way (a title that shows "skip" while A is held
/// and skips after about three seconds).
///
/// Nothing here knows the title: a scene is frames of 500 draws and more for
/// four seconds, as the frame meter counts them
/// (`victory_gpu::frame::recent_draws`). Then A is held for four seconds; if
/// the draws fall under 100 a frame within the next eight (the cut to black
/// a skip makes), that scene counts as skipped, and the next one is waited
/// for. A hold that changes nothing is tried again twice, eight seconds
/// apart, and then the holding ends for the run: where A does something else
/// (a scene the player walks in), three presses are all it gets.
pub(crate) mod auto_skip {
    use std::sync::{Mutex, OnceLock};
    use std::time::{Duration, Instant};

    const SCENE_DRAWS: u32 = 500;
    const BLACK_DRAWS: u32 = 100;
    const SCENE_FOR: Duration = Duration::from_secs(4);
    const HOLD: Duration = Duration::from_secs(4);
    const WATCH: Duration = Duration::from_secs(8);
    const TRIES: u32 = 3;

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub(super) enum State {
        /// No scene yet, or one for less than [`SCENE_FOR`] (since when).
        Waiting(Option<Instant>),
        /// A is down since then; the hold's number in this scene.
        Holding(Instant, u32),
        /// A is up since then: did the scene end?
        Watching(Instant, u32),
        Done,
    }

    struct Skip {
        state: State,
        left: u32,
    }

    /// How many scenes to skip (0: none, the switch is off).
    fn scenes() -> u32 {
        static N: OnceLock<u32> = OnceLock::new();
        *N.get_or_init(|| std::env::var("VICTORY_AUTO_SKIP").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0))
    }

    /// The next state and whether a scene was skipped, from the draws a
    /// frame now (`None`: no frames shown yet).
    pub(super) fn step(state: State, draws: Option<u32>, now: Instant) -> (State, bool) {
        let scene = draws.is_some_and(|d| d >= SCENE_DRAWS);
        match state {
            State::Waiting(since) => match (scene, since) {
                (false, _) => (State::Waiting(None), false),
                (true, None) => (State::Waiting(Some(now)), false),
                (true, Some(since)) if now.duration_since(since) >= SCENE_FOR => (State::Holding(now, 1), false),
                (true, since) => (State::Waiting(since), false),
            },
            State::Holding(since, n) if now.duration_since(since) >= HOLD => (State::Watching(now, n), false),
            State::Holding(..) => (state, false),
            State::Watching(_, _) if draws.is_some_and(|d| d < BLACK_DRAWS) => (State::Waiting(None), true),
            State::Watching(since, n) if now.duration_since(since) >= WATCH => {
                if n >= TRIES {
                    (State::Done, false)
                } else {
                    (State::Holding(now, n + 1), false)
                }
            }
            State::Watching(..) | State::Done => (state, false),
        }
    }

    /// Whether A is down now.
    pub(super) fn pressing() -> bool {
        if scenes() == 0 {
            return false;
        }
        static SKIP: Mutex<Option<Skip>> = Mutex::new(None);
        let mut skip = SKIP.lock().unwrap_or_else(|e| e.into_inner());
        let skip = skip.get_or_insert_with(|| {
            tracing::info!(scenes = scenes(), "auto skip: A is held through scenes that can be skipped (VICTORY_AUTO_SKIP)");
            Skip { state: State::Waiting(None), left: scenes() }
        });
        if skip.state == State::Done {
            return false;
        }
        let before = skip.state;
        let (state, skipped) = step(before, victory_gpu::frame::recent_draws(), Instant::now());
        skip.state = state;
        if skipped {
            skip.left -= 1;
            tracing::info!(left = skip.left, "auto skip: a scene was skipped");
            if skip.left == 0 {
                skip.state = State::Done;
            }
        }
        match (before, skip.state) {
            (State::Waiting(_), State::Holding(..)) => tracing::info!("auto skip: holding A"),
            (State::Watching(..), State::Done) if !skipped => {
                tracing::info!("auto skip: holding A changed nothing, no more holds")
            }
            _ => {}
        }
        matches!(skip.state, State::Holding(..))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn a_scene_is_held_through_and_a_cut_to_black_counts() {
            let t0 = Instant::now();
            let at = |s: u64| t0 + Duration::from_secs(s);
            // Loading screens: nothing.
            let (s, _) = step(State::Waiting(None), Some(11), at(0));
            assert_eq!(s, State::Waiting(None));
            // A scene begins; four seconds of it start the hold.
            let (s, _) = step(s, Some(2600), at(1));
            assert_eq!(s, State::Waiting(Some(at(1))));
            let (s, _) = step(s, Some(2600), at(3));
            assert_eq!(s, State::Waiting(Some(at(1))));
            let (s, _) = step(s, Some(2600), at(5));
            assert_eq!(s, State::Holding(at(5), 1));
            // Held four seconds, then watched; the cut to black counts.
            let (s, _) = step(s, Some(2600), at(7));
            assert_eq!(s, State::Holding(at(5), 1));
            let (s, _) = step(s, Some(2600), at(9));
            assert_eq!(s, State::Watching(at(9), 1));
            let (s, skipped) = step(s, Some(1), at(10));
            assert_eq!((s, skipped), (State::Waiting(None), true));
        }

        #[test]
        fn holds_that_change_nothing_end_after_three() {
            let t0 = Instant::now();
            let at = |s: u64| t0 + Duration::from_secs(s);
            let mut s = State::Holding(at(0), 1);
            let mut holds = 1;
            for second in 1..60 {
                let (next, skipped) = step(s, Some(5000), at(second));
                assert!(!skipped);
                if matches!(next, State::Holding(..)) && !matches!(s, State::Holding(..)) {
                    holds += 1;
                }
                s = next;
            }
            assert_eq!((s, holds), (State::Done, 3));
        }

        #[test]
        fn a_scene_that_ends_by_itself_starts_the_wait_again() {
            let t0 = Instant::now();
            let at = |s: u64| t0 + Duration::from_secs(s);
            let (s, _) = step(State::Waiting(Some(at(0))), Some(3), at(2));
            assert_eq!(s, State::Waiting(None));
        }
    }
}

pub(crate) mod auto_start {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    const FIRST: Duration = Duration::from_secs(1);
    const EVERY: Duration = Duration::from_secs(2);
    const HELD: Duration = Duration::from_millis(200);
    const LIMIT: Duration = Duration::from_secs(30);

    static ADDED: OnceLock<Instant> = OnceLock::new();
    static DONE: AtomicBool = AtomicBool::new(false);

    fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("VICTORY_AUTO_START").is_ok_and(|v| v.trim() == "1"))
    }

    /// The title added its user: the presses start (once per run).
    pub(crate) fn user_added() {
        if enabled() && ADDED.set(Instant::now()).is_ok() {
            tracing::info!("auto start: pressing A until the title moves on");
        }
    }

    /// The title opened a file: once the presses have started, they end.
    pub(crate) fn title_moved_on() {
        if ADDED.get().is_some() && !DONE.swap(true, Ordering::Relaxed) {
            tracing::info!("auto start: the title moved on, no more presses");
        }
    }

    /// Whether A is down now.
    pub(super) fn pressing() -> bool {
        let Some(added) = ADDED.get() else {
            return false;
        };
        if DONE.load(Ordering::Relaxed) {
            return false;
        }
        let elapsed = added.elapsed();
        if elapsed >= LIMIT {
            if !DONE.swap(true, Ordering::Relaxed) {
                tracing::info!("auto start: no more presses after 30 s");
            }
            return false;
        }
        down(elapsed)
    }

    /// The press schedule: held from 1 s on, 0.2 s of every 2 s.
    fn down(elapsed: Duration) -> bool {
        elapsed
            .checked_sub(FIRST)
            .is_some_and(|t| t.as_millis() % EVERY.as_millis() < HELD.as_millis())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn presses_are_short_and_two_seconds_apart() {
            let at = |ms| down(Duration::from_millis(ms));
            assert!(!at(0) && !at(999));
            assert!(at(1000) && at(1199));
            assert!(!at(1200) && !at(2999));
            assert!(at(3000) && at(3100));
            assert!(!at(3200));
            assert!(at(29_000));
        }
    }
}

/// Host input: XInput pads, the keyboard while our process has focus.
#[cfg(windows)]
mod host {
    use std::time::{Duration, Instant};

    use windows_sys::Win32::Foundation::{POINT, RECT};
    use windows_sys::Win32::Graphics::Gdi::ClientToScreen;
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        GetAsyncKeyState, MAPVK_VK_TO_VSC_EX, MapVirtualKeyW,
    };
    use windows_sys::Win32::UI::Input::{
        GetRawInputData, HRAWINPUT, RAWINPUT, RAWINPUTDEVICE, RAWINPUTHEADER, RID_INPUT,
        RIDEV_INPUTSINK, RIM_TYPEMOUSE, RegisterRawInputDevices,
    };
    use windows_sys::Win32::UI::Input::XboxController::{
        XINPUT_GAMEPAD, XINPUT_GAMEPAD_A, XINPUT_GAMEPAD_B, XINPUT_GAMEPAD_BACK,
        XINPUT_GAMEPAD_DPAD_DOWN, XINPUT_GAMEPAD_DPAD_LEFT, XINPUT_GAMEPAD_DPAD_RIGHT,
        XINPUT_GAMEPAD_DPAD_UP, XINPUT_GAMEPAD_LEFT_SHOULDER, XINPUT_GAMEPAD_LEFT_THUMB,
        XINPUT_GAMEPAD_RIGHT_SHOULDER, XINPUT_GAMEPAD_RIGHT_THUMB, XINPUT_GAMEPAD_START,
        XINPUT_GAMEPAD_X, XINPUT_GAMEPAD_Y, XINPUT_STATE, XINPUT_VIBRATION, XInputGetState,
        XInputSetState,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        ClipCursor, CreateWindowExW, DefWindowProcW, DispatchMessageW, GetClientRect,
        GetForegroundWindow, GetMessageW, GetWindowThreadProcessId, HWND_MESSAGE, MSG,
        RegisterClassW, WM_INPUT, WNDCLASSW,
    };

    use super::super::controls::MOUSE_BUTTONS;
    use super::*;

    /// Asking XInput about an empty slot is slow; look again this often.
    const EMPTY_SLOT_RECHECK: Duration = Duration::from_secs(1);

    const BUTTONS: [(u16, u32); 14] = [
        (XINPUT_GAMEPAD_START, MENU),
        (XINPUT_GAMEPAD_BACK, VIEW),
        (XINPUT_GAMEPAD_A, A),
        (XINPUT_GAMEPAD_B, B),
        (XINPUT_GAMEPAD_X, X),
        (XINPUT_GAMEPAD_Y, Y),
        (XINPUT_GAMEPAD_DPAD_UP, UP),
        (XINPUT_GAMEPAD_DPAD_DOWN, DOWN),
        (XINPUT_GAMEPAD_DPAD_LEFT, LEFT),
        (XINPUT_GAMEPAD_DPAD_RIGHT, RIGHT),
        (XINPUT_GAMEPAD_LEFT_SHOULDER, LEFT_SHOULDER),
        (XINPUT_GAMEPAD_RIGHT_SHOULDER, RIGHT_SHOULDER),
        (XINPUT_GAMEPAD_LEFT_THUMB, LEFT_STICK),
        (XINPUT_GAMEPAD_RIGHT_THUMB, RIGHT_STICK),
    ];

    /// Raw values, no dead zone: titles apply their own.
    pub(super) fn convert(pad: &XINPUT_GAMEPAD) -> PadState {
        let stick = |v: i16| (f32::from(v) / 32767.0).max(-1.0);
        PadState {
            buttons: BUTTONS
                .iter()
                .filter(|(host, _)| pad.wButtons & host != 0)
                .fold(0, |all, (_, bit)| all | bit),
            left_trigger: f32::from(pad.bLeftTrigger) / 255.0,
            right_trigger: f32::from(pad.bRightTrigger) / 255.0,
            left_x: stick(pad.sThumbLX),
            left_y: stick(pad.sThumbLY),
            right_x: stick(pad.sThumbRX),
            right_y: stick(pad.sThumbRY),
        }
    }

    pub(super) struct Host {
        empty_until: [Option<Instant>; PADS],
        /// Where scripted mouse motion goes (diagnostics).
        #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
        counts: Arc<Counts>,
        was_captured: bool,
        clipped: bool,
        /// Mouse buttons held when the cursor was captured (the click that
        /// captured it): no shots until let go.
        swallowed: Vec<u8>,
        #[cfg(feature = "diagnostics")]
        kbm: kbm::Script,
    }

    impl Host {
        pub(super) fn new(counts: Arc<Counts>) -> Host {
            Host {
                empty_until: [None; PADS],
                counts,
                was_captured: false,
                clipped: false,
                swallowed: Vec::new(),
                #[cfg(feature = "diagnostics")]
                kbm: kbm::Script::from_env(),
            }
        }

        /// Frees the cursor when the title window loses focus, and holds a
        /// captured one inside the window.
        fn cursor(&mut self, focus: bool) {
            cursor::FOCUSED.store(focus, Ordering::Relaxed);
            if !focus {
                cursor::release();
            }
            if !cursor::clips(focus) {
                if self.clipped {
                    // SAFETY: lets go of the clip we set.
                    unsafe { ClipCursor(std::ptr::null()) };
                    self.clipped = false;
                }
                return;
            }
            // SAFETY: plain queries on the foreground window, ours.
            unsafe {
                let hwnd = GetForegroundWindow();
                let mut client = RECT::default();
                GetClientRect(hwnd, &mut client);
                let mut corners = [
                    POINT {
                        x: client.left,
                        y: client.top,
                    },
                    POINT {
                        x: client.right,
                        y: client.bottom,
                    },
                ];
                ClientToScreen(hwnd, &mut corners[0]);
                ClientToScreen(hwnd, &mut corners[1]);
                let [a, b] = corners;
                if let Some((left, top, right, bottom)) = cursor::clip_rect((a.x, a.y, b.x, b.y)) {
                    let rect = RECT {
                        left,
                        top,
                        right,
                        bottom,
                    };
                    ClipCursor(&rect);
                    self.clipped = true;
                } else if self.clipped {
                    ClipCursor(std::ptr::null());
                    self.clipped = false;
                }
            }
        }
    }

    /// Mouse motion and the wheel from raw input (no pointer acceleration),
    /// on a thread and message-only window of its own. Counted only while
    /// the cursor is captured in the focused title window.
    pub(super) fn raw_mouse(counts: Arc<Counts>) {
        let spawned = std::thread::Builder::new()
            .name("victory mouse".into())
            .spawn(move || {
                let class: Vec<u16> = "victory.mouse\0".encode_utf16().collect();
                // SAFETY: a window class and a message-only window of this
                // thread; the registration names that window.
                let hwnd = unsafe {
                    let instance = GetModuleHandleW(std::ptr::null());
                    let wc = WNDCLASSW {
                        lpfnWndProc: Some(DefWindowProcW),
                        hInstance: instance,
                        lpszClassName: class.as_ptr(),
                        ..std::mem::zeroed()
                    };
                    RegisterClassW(&wc);
                    CreateWindowExW(
                        0,
                        class.as_ptr(),
                        std::ptr::null(),
                        0,
                        0,
                        0,
                        0,
                        0,
                        HWND_MESSAGE,
                        std::ptr::null_mut(),
                        instance,
                        std::ptr::null(),
                    )
                };
                let device = RAWINPUTDEVICE {
                    usUsagePage: 1, // generic desktop
                    usUsage: 2,     // mouse
                    dwFlags: RIDEV_INPUTSINK,
                    hwndTarget: hwnd,
                };
                // SAFETY: one valid device description.
                let ok = !hwnd.is_null()
                    && unsafe {
                        RegisterRawInputDevices(&device, 1, size_of::<RAWINPUTDEVICE>() as u32)
                    } != 0;
                if !ok {
                    tracing::warn!("raw mouse input unavailable: no mouse look");
                    return;
                }
                tracing::info!("raw mouse input on");
                let mut wheel = 0;
                // SAFETY: the standard message loop of this thread's window.
                unsafe {
                    let mut msg: MSG = std::mem::zeroed();
                    while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
                        if msg.message == WM_INPUT {
                            raw_input(msg.lParam as HRAWINPUT, &counts, &mut wheel);
                        }
                        DispatchMessageW(&msg);
                    }
                }
            });
        if spawned.is_err() {
            tracing::warn!("no raw mouse thread: no mouse look");
        }
    }

    /// One raw input event; `wheel` keeps the part of a step a fine-grained
    /// wheel has turned.
    fn raw_input(handle: HRAWINPUT, counts: &Counts, wheel: &mut i32) {
        /// Relative motion is the default; absolute comes from tablets,
        /// remote desktops and virtual machines.
        const MOUSE_MOVE_ABSOLUTE: u16 = 0x1;
        const RI_MOUSE_WHEEL: u16 = 0x0400;
        // SAFETY: zeroed is a valid RAWINPUT; `size` is its room.
        let mut raw: RAWINPUT = unsafe { std::mem::zeroed() };
        let mut size = size_of::<RAWINPUT>() as u32;
        // SAFETY: the handle of a WM_INPUT this thread received.
        let got = unsafe {
            GetRawInputData(
                handle,
                RID_INPUT,
                (&raw mut raw).cast(),
                &mut size,
                size_of::<RAWINPUTHEADER>() as u32,
            )
        };
        if got == u32::MAX || raw.header.dwType != RIM_TYPEMOUSE {
            return;
        }
        if !(cursor::FOCUSED.load(Ordering::Relaxed) && cursor::captured()) {
            *wheel = 0;
            return;
        }
        // SAFETY: a mouse event, so the mouse member is the one written.
        let mouse = unsafe { raw.data.mouse };
        if mouse.usFlags & MOUSE_MOVE_ABSOLUTE == 0 {
            counts.add(i64::from(mouse.lLastX), i64::from(mouse.lLastY));
        }
        // SAFETY: both members of the button union are plain integers.
        let buttons = unsafe { mouse.Anonymous.Anonymous };
        if buttons.usButtonFlags & RI_MOUSE_WHEEL != 0 {
            *wheel += i32::from(buttons.usButtonData as i16);
            let steps = *wheel / 120;
            *wheel -= steps * 120;
            if steps != 0 {
                counts.add_wheel(steps);
            }
        }
    }

    impl Source for Host {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            if self.empty_until[slot].is_some_and(|until| Instant::now() < until) {
                return None;
            }
            let mut state = XINPUT_STATE::default();
            // SAFETY: room for the state.
            if unsafe { XInputGetState(slot as u32, &mut state) } != 0 {
                self.empty_until[slot] = Some(Instant::now() + EMPTY_SLOT_RECHECK);
                return None;
            }
            self.empty_until[slot] = None;
            Some(convert(&state.Gamepad))
        }

        /// A for `VICTORY_AUTO_START` and `VICTORY_AUTO_SKIP`, on the pad
        /// the keyboard plays on whatever Enter is bound to.
        fn pressed(&mut self) -> u32 {
            if auto_start::pressing() || auto_skip::pressing() {
                A
            } else {
                0
            }
        }

        fn keys(&mut self) -> Vec<KeyState> {
            let focus = focused();
            self.cursor(focus);
            // The scripted keys (`VICTORY_KEYS`) are keys like the others:
            // the keyboard has them and the bindings make them pad input.
            #[cfg(feature = "diagnostics")]
            let mut down = {
                let mut down = scripted();
                down.extend(self.kbm.keys(&self.counts));
                down
            };
            #[cfg(not(feature = "diagnostics"))]
            let mut down = Vec::new();
            // Enter for the keyboard (A is pressed on the pad apart);
            // with or without focus.
            if (auto_start::pressing() || auto_skip::pressing()) && !down.contains(&0x0D) {
                down.push(0x0D);
            }
            if focus {
                // Left/right modifier keys only, as devices report them; no
                // mouse buttons (below 0x08).
                let held: Vec<u8> = (0x08..=0xFE_u8)
                    .filter(|vk| !(0x10..=0x12).contains(vk) && !down.contains(vk))
                    // SAFETY: plain queries.
                    .filter(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) } < 0)
                    .collect();
                down.extend(held);
            }
            down.into_iter()
                .map(|vk| KeyState {
                    // SAFETY: plain query.
                    scan_code: unsafe { MapVirtualKeyW(u32::from(vk), MAPVK_VK_TO_VSC_EX) },
                    virtual_key: vk,
                    // shortcut: no characters (code point 0);
                    // the title reads scan codes and virtual keys only.
                    ..Default::default()
                })
                .collect()
        }

        fn mouse_buttons(&mut self) -> Vec<u8> {
            let captured = cursor::FOCUSED.load(Ordering::Relaxed) && cursor::captured();
            let mut down = Vec::new();
            if captured {
                let held: Vec<u8> = MOUSE_BUTTONS
                    .into_iter()
                    // SAFETY: plain queries.
                    .filter(|&vk| unsafe { GetAsyncKeyState(i32::from(vk)) } < 0)
                    .collect();
                if !self.was_captured {
                    self.swallowed.clone_from(&held);
                }
                self.swallowed.retain(|b| held.contains(b));
                down = held
                    .into_iter()
                    .filter(|b| !self.swallowed.contains(b))
                    .collect();
            }
            self.was_captured = captured;
            #[cfg(feature = "diagnostics")]
            down.extend(self.kbm.buttons());
            down
        }

        fn rumble(&mut self, slot: usize, low: u16, high: u16) {
            let speeds = XINPUT_VIBRATION {
                wLeftMotorSpeed: low,
                wRightMotorSpeed: high,
            };
            // SAFETY: a valid vibration; an empty slot just fails.
            unsafe { XInputSetState(slot as u32, &speeds) };
        }
    }

    /// Debugging aid: `VICTORY_KEYS=<seconds>:<key>,...` holds each key for
    /// 0.2 s that long after input starts, with or without focus. A key is a
    /// letter, a digit, or `Enter`, `Esc`, `Tab`, `Back`, `Up`, `Down`,
    /// `Left`, `Right` (the key bindings map them to pad input; by default
    /// as the stand-in pad always did).
    #[cfg(feature = "diagnostics")]
    fn scripted() -> Vec<u8> {
        static SCRIPT: LazyLock<(Instant, Vec<(Duration, u8)>)> = LazyLock::new(|| {
            let list = std::env::var("VICTORY_KEYS").unwrap_or_default();
            let keys = list.split(',').filter_map(|entry| {
                let (at, key) = entry.split_once(':')?;
                let vk = match key.trim() {
                    "Enter" => 0x0D,
                    "Esc" => 0x1B,
                    "Tab" => 0x09,
                    "Back" => 0x08,
                    "Up" => 0x26,
                    "Down" => 0x28,
                    "Left" => 0x25,
                    "Right" => 0x27,
                    k if k.len() == 1 && k.as_bytes()[0].is_ascii_alphanumeric() => {
                        k.as_bytes()[0].to_ascii_uppercase()
                    }
                    _ => return None,
                };
                Some((
                    Duration::try_from_secs_f64(at.trim().parse().ok()?).ok()?,
                    vk,
                ))
            });
            (Instant::now(), keys.collect())
        });
        let (start, keys) = &*SCRIPT;
        let now = start.elapsed();
        keys.iter()
            .filter(|&&(at, _)| (at..at + Duration::from_millis(200)).contains(&now))
            .map(|&(_, vk)| vk)
            .collect()
    }

    /// Debugging aid: `VICTORY_KBM=<seconds>:<key>[*<hold seconds>],...`
    /// plays keyboard and mouse input into the controls as if held (with or
    /// without focus or capture). A key is a `[controls]` key name (`W`,
    /// `LShift`, `MouseLeft`...) held 0.2 s unless a hold is given;
    /// `MouseX=<counts>` / `MouseY=<counts>` move the mouse that far, spread
    /// over the hold; `Wheel=<steps>` turns the wheel (up positive).
    #[cfg(feature = "diagnostics")]
    mod kbm {
        use std::time::{Duration, Instant};

        use super::super::Counts;
        use crate::modules::controls::key_code;

        enum What {
            Key(u8),
            Mouse(f64, f64),
            Wheel(i32),
        }

        pub(super) struct Script {
            start: Instant,
            last: Duration,
            entries: Vec<(Duration, Duration, What)>,
            buttons: Vec<u8>,
        }

        impl Script {
            pub(super) fn from_env() -> Script {
                let list = std::env::var("VICTORY_KBM").unwrap_or_default();
                let entries = list
                    .split(',')
                    .filter(|e| !e.trim().is_empty())
                    .filter_map(|entry| {
                        let parsed = parse(entry);
                        if parsed.is_none() {
                            tracing::warn!(entry, "VICTORY_KBM entry not understood");
                        }
                        parsed
                    })
                    .collect();
                Script {
                    start: Instant::now(),
                    last: Duration::ZERO,
                    entries,
                    buttons: Vec::new(),
                }
            }

            /// Keys held now; adds the mouse motion since the last poll.
            pub(super) fn keys(&mut self, counts: &Counts) -> Vec<u8> {
                let now = self.start.elapsed();
                let last = std::mem::replace(&mut self.last, now);
                let mut keys = Vec::new();
                self.buttons.clear();
                for (at, hold, what) in &self.entries {
                    let end = *at + *hold;
                    match *what {
                        What::Key(vk) if (*at..end).contains(&now) => {
                            if vk <= 0x06 {
                                self.buttons.push(vk);
                            } else {
                                keys.push(vk);
                            }
                        }
                        What::Key(_) => {}
                        What::Mouse(x, y) => {
                            let from = last.max(*at);
                            let to = now.min(end);
                            if to > from && !hold.is_zero() {
                                let part = (to - from).as_secs_f64() / hold.as_secs_f64();
                                counts.add((x * part).round() as i64, (y * part).round() as i64);
                            }
                        }
                        What::Wheel(steps) => {
                            if last < *at && *at <= now {
                                counts.add_wheel(steps);
                            }
                        }
                    }
                }
                keys
            }

            /// Mouse buttons held now (from the last `keys`).
            pub(super) fn buttons(&self) -> Vec<u8> {
                self.buttons.clone()
            }
        }

        fn parse(entry: &str) -> Option<(Duration, Duration, What)> {
            let (at, rest) = entry.split_once(':')?;
            let (what, hold) = match rest.split_once('*') {
                Some((what, hold)) => (what, hold.trim().parse::<f64>().ok()?),
                None => (rest, 0.2),
            };
            let what = match what.trim().split_once('=') {
                Some((axis, n)) => {
                    let n: f64 = n.trim().parse().ok()?;
                    match axis.trim() {
                        "MouseX" => What::Mouse(n, 0.0),
                        "MouseY" => What::Mouse(0.0, n),
                        "Wheel" => What::Wheel(n as i32),
                        _ => return None,
                    }
                }
                None => What::Key(key_code(what)?),
            };
            Some((
                Duration::try_from_secs_f64(at.trim().parse().ok()?).ok()?,
                Duration::try_from_secs_f64(hold).ok()?,
                what,
            ))
        }
    }

    /// The foreground window is one of ours (the title's or our fallback).
    fn focused() -> bool {
        let mut process = 0;
        // SAFETY: plain queries; a null window gives process 0.
        unsafe {
            GetWindowThreadProcessId(GetForegroundWindow(), &mut process);
            process == GetCurrentProcessId()
        }
    }

    #[test]
    fn host_pads_convert_to_console_pads() {
        let pad = XINPUT_GAMEPAD {
            wButtons: XINPUT_GAMEPAD_A | XINPUT_GAMEPAD_START | XINPUT_GAMEPAD_DPAD_LEFT,
            bLeftTrigger: 255,
            sThumbLX: -32768,
            sThumbLY: 32767,
            ..Default::default()
        };
        let state = convert(&pad);
        assert_eq!(state.buttons, A | MENU | LEFT);
        assert_eq!(
            (
                state.left_trigger,
                state.left_x,
                state.left_y,
                state.right_x
            ),
            (1.0, -1.0, 1.0, 0.0)
        );
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::HleResolver;
    use crate::HostPolicy;
    use victory_loader::Provenance;
    use victory_pe::ImportSymbol;

    #[derive(Default)]
    struct FakeHost {
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
        pressed: u32,
        buttons: Vec<u8>,
        rumble: Vec<(usize, u16, u16)>,
    }

    struct Fake(Arc<Mutex<FakeHost>>);

    impl Source for Fake {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            self.0.lock().unwrap().pads[slot]
        }
        fn keys(&mut self) -> Vec<KeyState> {
            self.0.lock().unwrap().keys.clone()
        }
        fn pressed(&mut self) -> u32 {
            self.0.lock().unwrap().pressed
        }
        fn mouse_buttons(&mut self) -> Vec<u8> {
            self.0.lock().unwrap().buttons.clone()
        }
        fn rumble(&mut self, slot: usize, low: u16, high: u16) {
            self.0.lock().unwrap().rumble.push((slot, low, high));
        }
    }

    /// A hub with the default bindings and a linear mouse stick.
    fn fake() -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        fake_with(Settings {
            deadzone: 0.0,
            ..Settings::default()
        })
    }

    fn fake_with(settings: Settings) -> (&'static Hub, Arc<Mutex<FakeHost>>) {
        let host = Arc::new(Mutex::new(FakeHost::default()));
        let counts = Arc::new(Counts::default());
        let hub = Hub::new(Box::new(Fake(Arc::clone(&host))), counts, settings);
        // Whatever the environment says: one test turns it off itself.
        hub.state().latch = true;
        hub.poll();
        (hub, host)
    }

    fn key(virtual_key: u8) -> KeyState {
        KeyState {
            scan_code: u32::from(virtual_key) + 0x100,
            virtual_key,
            ..Default::default()
        }
    }

    fn slot<F: Copy, T>(object: *const T, offset: usize) -> F {
        // SAFETY: our objects start with their table, which has the slot.
        let f = unsafe { *(*(object as *const *const *const ())).add(offset / 8) };
        // SAFETY: the caller names the slot's signature.
        unsafe { std::mem::transmute_copy(&f) }
    }

    /// Device callbacks seen: device index, current and previous status.
    type Log = Mutex<Vec<(usize, u32, u32)>>;

    extern "system" fn record(
        _token: u64,
        context: *mut c_void,
        device: *const Device,
        _timestamp: u64,
        current: u32,
        previous: u32,
    ) {
        // SAFETY: the test passes its log and one of the hub's devices.
        let (log, index) = unsafe { (&*(context as *const Log), (*device).index) };
        log.lock().unwrap().push((index, current, previous));
    }

    fn register(hub: &Hub, kind: u32, enumeration: u32, log: &Log) -> u64 {
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let mut token = 0;
        let context = log as *const Log as *mut c_void;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let hr = register(
            &hub.object,
            std::ptr::null(),
            kind,
            1,
            enumeration,
            context,
            Some(record),
            &mut token,
        );
        assert_eq!(hr, S_OK);
        assert_ne!(token, 0);
        token
    }

    fn result(hr: HRESULT, out: *const c_void) -> Result<*const Reading, HRESULT> {
        if hr == S_OK {
            Ok(out.cast())
        } else {
            assert!(out.is_null());
            Err(hr)
        }
    }

    fn current(hub: &Hub, kind: u32, device: usize) -> Result<*const Reading, HRESULT> {
        type Current = extern "system" fn(This, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let hr = slot::<Current, _>(&hub.object, 0x20)(
            &hub.object,
            kind,
            &hub.devices[device],
            &mut out,
        );
        result(hr, out)
    }

    fn walk(hub: &Hub, offset: usize, from: *const Reading) -> Result<*const Reading, HRESULT> {
        type Walk = extern "system" fn(This, *const Reading, u32, *const Device, Out) -> HRESULT;
        let mut out = std::ptr::null();
        let device = reading(from).device;
        let hr =
            slot::<Walk, _>(&hub.object, offset)(&hub.object, from, GAMEPAD_KIND, device, &mut out);
        result(hr, out)
    }

    fn pad(reading: *const Reading) -> PadState {
        let mut state = PadState::default();
        let get =
            slot::<extern "system" fn(*const Reading, *mut PadState) -> bool, _>(reading, 0xb0);
        assert!(get(reading, &mut state));
        state
    }

    fn stamp(reading: *const Reading) -> (u64, u64) {
        let time = slot::<extern "system" fn(*const Reading) -> u64, _>(reading, 0x28);
        let sequence = slot::<extern "system" fn(*const Reading, u32) -> u64, _>(reading, 0x20);
        (time(reading), sequence(reading, GAMEPAD_KIND))
    }

    fn release(reading: *const Reading) {
        slot::<extern "system" fn(*const Reading) -> u32, _>(reading, 0x10)(reading);
    }

    #[test]
    fn ordinal_one_resolves_to_create() {
        let r = HleResolver::new(HostPolicy::none());
        assert_eq!(
            r.classify("GameInput.dll", &ImportSymbol::Ordinal(1)),
            Some(Provenance::Hle)
        );
        assert_eq!(r.classify("GameInput.dll", &ImportSymbol::Ordinal(2)), None);
    }

    #[test]
    fn pads_connect_report_and_disconnect() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        // Blocking enumeration ran before registration returned: with no
        // host pad the keyboard's stand-in is the only one.
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        host.lock().unwrap().pads[1] = Some(PadState {
            buttons: A,
            left_x: 0.5,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[1..],
            [(1, STATUS_CONNECTED, 0), (FALLBACK, 0, STATUS_CONNECTED)]
        );

        let first = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(first).buttons, A);
        assert_eq!(pad(first).left_x, 0.5);
        // No change, no new reading.
        hub.poll();
        let again = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(again, first);
        release(again);

        std::thread::sleep(Duration::from_millis(2));
        host.lock().unwrap().pads[1].as_mut().unwrap().buttons = B;
        hub.poll();
        let second = current(hub, GAMEPAD_KIND, 1).unwrap();
        assert_eq!(pad(second).buttons, B);
        let (t1, s1) = stamp(first);
        let (t2, s2) = stamp(second);
        assert!(t2 > t1 && s2 == s1 + 1);
        assert!(t2 <= now());

        // The title's edge check: previous of current, and back again.
        assert_eq!(walk(hub, 0x30, second), Ok(first));
        assert_eq!(walk(hub, 0x28, first), Ok(second));
        assert_eq!(walk(hub, 0x28, second), Err(READING_NOT_FOUND));
        assert_eq!(walk(hub, 0x30, first), Err(READING_NOT_FOUND));
        for r in [first, first, second, second] {
            release(r);
        }

        host.lock().unwrap().pads[1] = None;
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[3..],
            [(1, 0, STATUS_CONNECTED), (FALLBACK, STATUS_CONNECTED, 0)]
        );
        assert_eq!(current(hub, GAMEPAD_KIND, 1), Err(DEVICE_DISCONNECTED));
        // The stand-in is back, at rest.
        let rest = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(rest), PadState::default());
        release(rest);
    }

    #[test]
    fn with_no_device_readings_are_walked_across_devices_in_time_order() {
        type Walk = extern "system" fn(This, *const Reading, u32, *const Device, Out) -> HRESULT;
        let any = |hub: &Hub, offset, from, kind| {
            let mut out = std::ptr::null();
            let walk = slot::<Walk, _>(&hub.object, offset);
            result(
                walk(&hub.object, from, kind, std::ptr::null(), &mut out),
                out,
            )
        };
        let both = GAMEPAD_KIND | KEYBOARD_KIND;
        let tick = || std::thread::sleep(Duration::from_millis(2));
        let (hub, host) = fake();
        // The first poll: the stand-in pad and the keyboard, at one time.
        let stand_in = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        let no_keys = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        tick();
        host.lock().unwrap().pads[1] = Some(PadState {
            buttons: A,
            ..Default::default()
        });
        hub.poll();
        let pad_a = current(hub, GAMEPAD_KIND, 1).unwrap();
        tick();
        // A key bound to nothing: no pad reading beside the keyboard's.
        host.lock().unwrap().keys = vec![key(b'P')];
        hub.poll();
        let w = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        tick();
        host.lock().unwrap().pads[1].as_mut().unwrap().buttons = B;
        hub.poll();
        let pad_b = current(hub, GAMEPAD_KIND, 1).unwrap();

        let order = [stand_in, no_keys, pad_a, w, pad_b];
        for pair in order.windows(2) {
            assert_eq!(any(hub, 0x28, pair[0], both), Ok(pair[1]));
            assert_eq!(any(hub, 0x30, pair[1], both), Ok(pair[0]));
        }
        assert_eq!(any(hub, 0x30, stand_in, both), Err(READING_NOT_FOUND));
        assert_eq!(any(hub, 0x28, pad_b, both), Err(READING_NOT_FOUND));
        // Only gamepads: the keyboard's readings are skipped.
        assert_eq!(any(hub, 0x28, stand_in, GAMEPAD_KIND), Ok(pad_a));
        assert_eq!(any(hub, 0x28, pad_a, GAMEPAD_KIND), Ok(pad_b));
        assert_eq!(any(hub, 0x30, pad_a, GAMEPAD_KIND), Ok(stand_in));
        // Naming the device keeps the walk on it.
        assert_eq!(walk(hub, 0x30, pad_a), Err(READING_NOT_FOUND));
        for r in order {
            release(r);
        }
    }

    #[test]
    fn keys_feed_the_keyboard_and_the_stand_in_pad() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, KEYBOARD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(KEYBOARD, STATUS_CONNECTED, 0)]);

        let keys = vec![key(b'W'), key(0x0D), key(b'L'), key(b'C')];
        host.lock().unwrap().keys = keys.clone();
        hub.poll();
        let r = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        let count = slot::<extern "system" fn(*const Reading) -> u32, _>(r, 0x70);
        let state =
            slot::<extern "system" fn(*const Reading, u32, *mut KeyState) -> u32, _>(r, 0x78);
        assert_eq!(count(r), 4);
        let mut got = [KeyState::default(); 8];
        assert_eq!(state(r, 8, got.as_mut_ptr()), 4);
        assert_eq!(got[..4], keys[..]);
        release(r);

        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(
            pad(r),
            PadState {
                buttons: A,
                left_y: 1.0,
                right_x: 1.0,
                right_trigger: 1.0,
                ..Default::default()
            }
        );
        release(r);
        // A keyboard reading is no gamepad reading.
        assert_eq!(current(hub, GAMEPAD_KIND, KEYBOARD), Err(READING_NOT_FOUND));
    }

    #[test]
    fn keyboard_and_mouse_drive_the_stand_in_pad_without_a_host_pad() {
        let (hub, host) = fake();
        {
            let mut h = host.lock().unwrap();
            h.keys = vec![key(b'W'), key(0x0D)];
            h.buttons = vec![0x01]; // left mouse
        }
        hub.poll();
        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        let p = pad(r);
        assert_eq!((p.buttons, p.left_y, p.right_trigger), (A, 1.0, 1.0));
        release(r);
        // The keyboard device still has its keys, and no mouse button.
        let r = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        let count = slot::<extern "system" fn(*const Reading) -> u32, _>(r, 0x70);
        assert_eq!(count(r), 2);
        release(r);
    }

    /// An AZERTY keyboard: the key at the US W place types Z. By its place
    /// (`keys_by_position`) it moves the stick as W does; by letter (the
    /// default) it is the left trigger, as Z.
    #[test]
    fn keys_go_by_their_place_when_asked() {
        let azerty_z = KeyState {
            scan_code: 0x11,
            virtual_key: b'Z',
            ..Default::default()
        };
        let (hub, host) = fake_with(Settings {
            by_position: true,
            ..Settings::default()
        });
        host.lock().unwrap().keys = vec![azerty_z];
        hub.poll();
        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!((pad(r).left_y, pad(r).left_trigger), (1.0, 0.0), "the W place");
        release(r);
        let (hub, host) = fake();
        host.lock().unwrap().keys = vec![azerty_z];
        hub.poll();
        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!((pad(r).left_y, pad(r).left_trigger), (0.0, 1.0), "by letter");
        release(r);
    }

    #[test]
    fn keyboard_and_mouse_ride_on_a_connected_host_pad() {
        let (hub, host) = fake();
        {
            let mut h = host.lock().unwrap();
            h.pads[1] = Some(PadState {
                buttons: B,
                left_trigger: 0.5,
                ..Default::default()
            });
            h.keys = vec![key(b'D'), key(b'X')];
            h.buttons = vec![0x02]; // right mouse
        }
        hub.poll();
        let r = current(hub, GAMEPAD_KIND, 1).unwrap();
        let p = pad(r);
        assert_eq!(
            (p.buttons, p.left_x, p.left_trigger),
            (B | X, 1.0, 1.0),
            "pad and keyboard together"
        );
        release(r);
        assert_eq!(
            current(hub, GAMEPAD_KIND, FALLBACK),
            Err(DEVICE_DISCONNECTED),
            "no stand-in pad beside a host pad"
        );
        // The automatic presses go to that pad too.
        host.lock().unwrap().pressed = A;
        hub.poll();
        assert_eq!(buttons(hub, 1) & A, A);
    }

    #[test]
    fn a_title_pad_read_carries_the_mouse_motion_since_its_last_read() {
        let (hub, _host) = fake();
        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        let first = stamp(r).0;
        release(r);
        std::thread::sleep(Duration::from_millis(5));
        // Below a full stick (no carry into the next read).
        hub.counts.add(4, -2);
        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        let p = pad(r);
        assert!(stamp(r).0 > first, "a new reading at the read");
        assert!(p.right_x > 0.0 && p.right_y > 0.0, "{p:?}");
        assert!((p.right_x - 2.0 * p.right_y).abs() < 1e-4, "{p:?}");
        release(r);
        std::thread::sleep(Duration::from_millis(5));
        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(r).right_x, 0.0, "no motion, no turn");
        release(r);
    }

    #[test]
    fn async_enumeration_waits_for_the_poll_and_unregistering_stops_callbacks() {
        let (hub, host) = fake();
        let log = Log::default();
        let token = register(hub, GAMEPAD_KIND, ASYNC_ENUMERATION, &log);
        assert!(log.lock().unwrap().is_empty());
        hub.poll();
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(unregister(&hub.object, token, 5000));
        host.lock().unwrap().pads[0] = Some(PadState::default());
        hub.poll();
        assert_eq!(log.lock().unwrap().len(), 1);
    }

    #[test]
    fn unregistering_waits_for_a_running_callback() {
        use std::sync::atomic::AtomicBool;
        static INSIDE: AtomicBool = AtomicBool::new(false);
        static LEAVE: AtomicBool = AtomicBool::new(false);
        static FROM_INSIDE: AtomicU32 = AtomicU32::new(0);
        extern "system" fn slow(
            token: u64,
            context: *mut c_void,
            _: *const Device,
            _: u64,
            _: u32,
            _: u32,
        ) {
            // From inside its own callback, unregistering fails at once.
            let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(context, 0x68);
            if !unregister(context.cast(), token, 0) {
                FROM_INSIDE.fetch_add(1, Ordering::SeqCst);
            }
            INSIDE.store(true, Ordering::SeqCst);
            while !LEAVE.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        let (hub, host) = fake();
        type Register = extern "system" fn(
            This,
            *const Device,
            u32,
            u32,
            u32,
            *mut c_void,
            Option<DeviceCallback>,
            *mut u64,
        ) -> HRESULT;
        let register = slot::<Register, _>(&hub.object, 0x48);
        let mut token = 0;
        let context = (&raw const hub.object).cast_mut().cast();
        assert_eq!(
            register(&hub.object, std::ptr::null(), GAMEPAD_KIND, 1, 0, context, Some(slow), &mut token),
            S_OK
        );
        host.lock().unwrap().pads[1] = Some(PadState {
            buttons: A,
            ..Default::default()
        });
        let poller = std::thread::spawn(|| hub.poll());
        while !INSIDE.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(1));
        }
        let unregister = slot::<extern "system" fn(This, u64, u64) -> bool, _>(&hub.object, 0x68);
        assert!(!unregister(&hub.object, token, 5000), "times out while it runs");
        let leave = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            LEAVE.store(true, Ordering::SeqCst);
        });
        let start = Instant::now();
        assert!(unregister(&hub.object, token, u64::MAX), "waits for it");
        assert!(start.elapsed() >= Duration::from_millis(40));
        assert_eq!(FROM_INSIDE.load(Ordering::SeqCst), 1);
        poller.join().unwrap();
        leave.join().unwrap();
    }

    #[test]
    fn the_latch_is_on_unless_turned_off() {
        assert!(latch_asked(None));
        assert!(latch_asked(Some("1".as_ref())));
        assert!(latch_asked(Some("".as_ref())));
        assert!(!latch_asked(Some("0".as_ref())));
    }

    /// A host pad at rest that counts as connected: a little stick drift.
    fn resting() -> PadState {
        PadState {
            right_y: 0.001,
            ..Default::default()
        }
    }

    fn buttons(hub: &Hub, device: usize) -> u32 {
        let r = current(hub, GAMEPAD_KIND, device).unwrap();
        let b = pad(r).buttons;
        release(r);
        b
    }

    fn held_keys(hub: &Hub) -> Vec<u8> {
        let r = current(hub, KEYBOARD_KIND, KEYBOARD).unwrap();
        let mut got = [KeyState::default(); 8];
        let state =
            slot::<extern "system" fn(*const Reading, u32, *mut KeyState) -> u32, _>(r, 0x78);
        let n = state(r, 8, got.as_mut_ptr()) as usize;
        release(r);
        got[..n].iter().map(|k| k.virtual_key).collect()
    }

    #[test]
    fn a_tap_between_current_reads_is_pressed_in_one_read_and_released_in_the_next() {
        let (hub, host) = fake();
        // The title reads the stand-in pad every frame.
        assert_eq!(buttons(hub, FALLBACK), 0);
        host.lock().unwrap().keys = vec![key(0x0D)];
        hub.poll();
        host.lock().unwrap().keys.clear();
        hub.poll();
        let tap = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(tap).buttons, A);
        // The title's edge check sees the press.
        let before = walk(hub, 0x30, tap).unwrap();
        assert_eq!(pad(before).buttons, 0);
        release(before);
        // The next read, with or without a poll between, has it released.
        let next = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(next).buttons, 0);
        assert!(stamp(next).1 > stamp(tap).1);
        hub.poll();
        let after = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(after, next);
        for r in [tap, next, after] {
            release(r);
        }
    }

    #[test]
    fn a_held_button_stays_pressed_and_its_release_is_not_delayed() {
        let (hub, host) = fake();
        assert_eq!(buttons(hub, FALLBACK), 0);
        host.lock().unwrap().keys = vec![key(0x0D)];
        for _ in 0..3 {
            hub.poll();
            assert_eq!(buttons(hub, FALLBACK), A);
        }
        host.lock().unwrap().keys.clear();
        hub.poll();
        assert_eq!(buttons(hub, FALLBACK), 0);
    }

    /// Sticks are the latest value; a trigger tapped between two reads is
    /// held for the next (a mouse click shorter than a frame shoots), the
    /// keyboard and mouse's triggers by `controls`.
    #[test]
    fn sticks_are_the_latest_value_and_a_trigger_tap_is_held() {
        let (hub, host) = fake();
        assert_eq!(buttons(hub, FALLBACK), 0);
        host.lock().unwrap().keys = vec![key(b'W'), key(b'C'), key(0x0D)];
        hub.poll();
        host.lock().unwrap().keys = vec![key(b'D')];
        hub.poll();
        let r = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(
            pad(r),
            PadState {
                buttons: A,
                left_x: 1.0,
                right_trigger: 1.0,
                ..Default::default()
            }
        );
        release(r);
    }

    #[test]
    fn a_host_pad_tap_is_latched_too() {
        let (hub, host) = fake();
        host.lock().unwrap().pads[0] = Some(resting());
        hub.poll();
        assert_eq!(buttons(hub, 0), 0);
        host.lock().unwrap().pads[0] = Some(PadState {
            buttons: B | X,
            left_x: 0.5,
            ..Default::default()
        });
        hub.poll();
        // B let go, X still held.
        host.lock().unwrap().pads[0] = Some(PadState {
            buttons: X,
            ..Default::default()
        });
        hub.poll();
        let r = current(hub, GAMEPAD_KIND, 0).unwrap();
        assert_eq!(
            pad(r),
            PadState {
                buttons: B | X,
                ..Default::default()
            }
        );
        release(r);
        assert_eq!(buttons(hub, 0), X);
    }

    /// Different buttons tapped one after another between two reads are
    /// handed out one per read, not as a chord nobody pressed.
    #[test]
    fn taps_in_sequence_are_not_reported_as_a_chord() {
        let (hub, host) = fake();
        host.lock().unwrap().pads[0] = Some(resting());
        hub.poll();
        assert_eq!(buttons(hub, 0), 0);
        for b in [A, 0, B, 0] {
            host.lock().unwrap().pads[0] = Some(PadState {
                buttons: b,
                ..Default::default()
            });
            hub.poll();
        }
        assert_eq!(buttons(hub, 0), A, "the first tap");
        assert_eq!(buttons(hub, 0), B, "then the second, alone");
        assert_eq!(buttons(hub, 0), 0);
        // Pressed together in one poll: still one chord.
        host.lock().unwrap().pads[0] = Some(PadState {
            buttons: A | B,
            ..Default::default()
        });
        hub.poll();
        host.lock().unwrap().pads[0] = Some(PadState::default());
        hub.poll();
        assert_eq!(buttons(hub, 0), A | B);
        assert_eq!(buttons(hub, 0), 0);
    }

    #[test]
    fn key_taps_in_sequence_are_not_reported_together() {
        let (hub, host) = fake();
        assert_eq!(held_keys(hub), Vec::<u8>::new());
        for keys in [vec![key(b'Q')], vec![], vec![key(b'E')], vec![]] {
            host.lock().unwrap().keys = keys;
            hub.poll();
        }
        assert_eq!(held_keys(hub), [b'Q']);
        assert_eq!(held_keys(hub), [b'E']);
        assert_eq!(held_keys(hub), Vec::<u8>::new());
    }

    #[test]
    fn a_key_tap_between_current_reads_is_latched_on_the_keyboard() {
        let (hub, host) = fake();
        host.lock().unwrap().keys = vec![key(b'Q')];
        hub.poll();
        assert_eq!(held_keys(hub), [b'Q']);
        host.lock().unwrap().keys = vec![key(b'Q'), key(b'E')];
        hub.poll();
        host.lock().unwrap().keys = vec![key(b'Q')];
        hub.poll();
        assert_eq!(held_keys(hub), [b'Q', b'E']);
        assert_eq!(held_keys(hub), [b'Q']);
        host.lock().unwrap().keys.clear();
        hub.poll();
        assert_eq!(held_keys(hub), Vec::<u8>::new());
    }

    #[test]
    fn without_current_reads_or_with_the_latch_off_the_history_is_as_recorded() {
        for off in [false, true] {
            let (hub, host) = fake();
            let first = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
            if off {
                hub.state().latch = false;
            } else {
                // No current read for longer than the latch waits.
                hub.state().slots[FALLBACK].latch.read = Some(0);
                std::thread::sleep(Duration::from_micros(LATCH_IDLE.saturating_sub(now())));
            }
            host.lock().unwrap().keys = vec![key(0x0D)];
            hub.poll();
            host.lock().unwrap().keys.clear();
            hub.poll();
            let press = walk(hub, 0x28, first).unwrap();
            assert_eq!(pad(press).buttons, A);
            let up = walk(hub, 0x28, press).unwrap();
            assert_eq!(pad(up).buttons, 0);
            assert_eq!(walk(hub, 0x28, up), Err(READING_NOT_FOUND));
            assert_eq!(buttons(hub, FALLBACK), 0);
            for r in [first, press, up] {
                release(r);
            }
        }
    }

    /// A receiver whose pad is switched off answers with an all-zero state:
    /// it stays disconnected and the stand-in pad stays, also with presses
    /// made for the player. Its first input connects it; unplugging forgets
    /// that.
    #[test]
    fn a_switched_off_receiver_does_not_hide_the_stand_in_pad() {
        let (hub, host) = fake();
        let log = Log::default();
        register(hub, GAMEPAD_KIND, 2, &log);
        assert_eq!(*log.lock().unwrap(), [(FALLBACK, STATUS_CONNECTED, 0)]);

        host.lock().unwrap().pads[0] = Some(PadState::default());
        hub.poll();
        assert_eq!(log.lock().unwrap().len(), 1);
        assert_eq!(current(hub, GAMEPAD_KIND, 0), Err(DEVICE_DISCONNECTED));
        // Presses for the player do not count as the receiver's input.
        host.lock().unwrap().pressed = A;
        hub.poll();
        assert_eq!(current(hub, GAMEPAD_KIND, 0), Err(DEVICE_DISCONNECTED));
        host.lock().unwrap().pressed = 0;
        host.lock().unwrap().keys = vec![key(0x0D)];
        hub.poll();
        let stand_in = current(hub, GAMEPAD_KIND, FALLBACK).unwrap();
        assert_eq!(pad(stand_in).buttons, A);
        release(stand_in);
        host.lock().unwrap().keys.clear();

        // The pad switched on: its first press connects it, and its rest
        // after that keeps it.
        host.lock().unwrap().pads[0] = Some(PadState {
            buttons: B,
            ..Default::default()
        });
        hub.poll();
        assert_eq!(
            log.lock().unwrap()[1..],
            [(0, STATUS_CONNECTED, 0), (FALLBACK, 0, STATUS_CONNECTED)]
        );
        host.lock().unwrap().pads[0] = Some(PadState::default());
        host.lock().unwrap().pressed = A;
        hub.poll();
        let first = current(hub, GAMEPAD_KIND, 0).unwrap();
        assert_eq!(pad(first).buttons, A, "presses for the player go to it");
        release(first);
        host.lock().unwrap().pressed = 0;

        // Unplugged and back at rest: disconnected again.
        host.lock().unwrap().pads[0] = None;
        hub.poll();
        host.lock().unwrap().pads[0] = Some(PadState::default());
        hub.poll();
        assert_eq!(current(hub, GAMEPAD_KIND, 0), Err(DEVICE_DISCONNECTED));
        release(current(hub, GAMEPAD_KIND, FALLBACK).unwrap());
    }

    #[test]
    fn devices_are_found_by_id_and_rumble_reaches_the_host() {
        let (hub, host) = fake();
        host.lock().unwrap().pads[2] = Some(PadState::default());
        hub.poll();
        let device = &hub.devices[2] as *const Device;
        let info = slot::<extern "system" fn(*const Device) -> *const u8, _>(device, 0x18);
        // SAFETY: the info has the id at +0x20.
        let id = unsafe { *(info(device).add(0x20) as *const [u8; 32]) };
        assert_ne!(id, [0; 32]);

        type Find = extern "system" fn(This, *const [u8; 32], Out) -> HRESULT;
        let find = slot::<Find, _>(&hub.object, 0x80);
        let mut out = std::ptr::null();
        assert_eq!(find(&hub.object, &id, &mut out), S_OK);
        assert_eq!(out, device.cast());
        assert_eq!(find(&hub.object, &[1; 32], &mut out), DEVICE_NOT_FOUND);
        assert!(out.is_null());

        type Rumble = extern "system" fn(*const Device, *const [f32; 4]);
        let rumble = slot::<Rumble, _>(device, 0x50);
        rumble(device, &[0.5, 1.0, 1.0, 1.0]);
        rumble(&hub.devices[FALLBACK], &[1.0; 4]);
        rumble(device, std::ptr::null());
        assert_eq!(host.lock().unwrap().rumble, [(2, 32767, 65535), (2, 0, 0)]);
    }
}
