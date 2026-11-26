# Ray tracing in this fork: what the title does and what has been built

Status, 2026-11-25: **nothing is lit by ray tracing yet.** Most of the
machinery translates and is unit-tested; it has not yet run end to end in the
game. Everything here is off by default.

## What the title does

The title's indirect light (probes, bounce, reflections) and the
shadows of its lamps and sun are ray traced. Without them, anything the sun
does not hit directly is black: interiors, the shadowed side of a face, rooms
lit only by lamps.

The title does not use a ray-tracing API:

- Traversal is ordinary compute-shader code: a per-thread stack in shared
  memory, and loads through raw 40-bit addresses of title memory.
- The static world's acceleration structures ship pre-built in the title's
  data, in the console GPU's native node format (64-byte nodes: box nodes with
  four children and 16-bit float bounds, 128-byte box nodes with 32-bit float
  bounds, triangle nodes, instance nodes). They are unpacked on the GPU.
- A set of builder programs (from the title's `XboxBvhBuilder.dll`, 52 compute
  programs) builds and refits the dynamic structures on the GPU every frame.
- Only the per-node test is hardware: the instructions
  `image_bvh64_intersect_ray` and `image_bvh_intersect_ray`. One call tests a
  ray against a box node's four children (returning the hit children, nearest
  first) or against a triangle node (returning the intersection's numerator,
  denominator and barycentric numerators).

So mapping this onto the host's own ray tracing (DXR) is not practical: the
layer would have to know the geometry behind structures the title never
describes to it, and the traversal loops live in 64 of the title's own
programs. This fork runs the title's builder and traversal as translated
shaders and emulates the one instruction.

## What this fork adds

1. **The builder programs translate** (all 52). They pass buffer addresses in
   a spilled parameter table rather than in descriptors. Added: the spilled
   user-data table, root unordered-access views (translator and host binding),
   byte and 16-bit buffer accesses, and several control-flow shapes. On a GPU
   with 32-lane waves, three of them (prefix sums over groups of 256 and 512
   threads, whose 64-lane waves read lanes across their halves) run in the
   paired form: one host wave for each console wave, each thread two of its
   lanes.
2. **Title memory by raw address from shaders.** A directory with one entry per
   64 KiB of title address space leads a shader to the layer's host copy of
   that memory; loads, stores, atomic add/sub and float min/max go through it.
   Before a dispatch the host makes ready the memory its root views and root
   constants name.
3. **The ray test, emulated in DXIL.** Four slab tests and a sorting network
   for box nodes; a watertight ray/triangle test for triangle nodes; both forms
   of the instruction. A plain Rust reference lives in
   `crates/victory-shader/src/bvh.rs`; tests run the emitted DXIL on the GPU
   over 311 ray/node cases and compare with the reference bit for bit.
4. **The trace and shading programs translate** (63 of 64 trace programs, and
   the irradiance-cache shading program). This needed lane masks treated as
   plain numbers on some paths, and exact forms for programs written for
   64-lane waves running on a GPU with 32-lane waves.
5. **A safety rule on the host side.** The builder's stages run only while
   every builder pipeline the title has created has a host pipeline; otherwise
   none run (`VICTORY_BVH_BUILDER=1` asks for it; `VICTORY_BVH_QUARANTINE=1`
   also keeps the builder's stores out of title memory, for a first test). A
   partly working builder wrote garbage bounds into title memory and crashed
   the title.

## Known gaps

- With `VICTORY_BVH_BUILDER=1` the builder is not usable yet. Until the
  paired form, three of its programs did not translate on a 32-lane GPU,
  which kept it off; every builder pipeline is now created offline
  (`victory translate --pipelines` with `VICTORY_HOST_MAX_WAVE=32`). Switched
  on in game runs on an RTX 5090, with or without `VICTORY_BVH_QUARANTINE=1`,
  the title crashed about 80 s into the gameplay, reading a pointer of all
  ones; on an RTX 4070 Ti it faulted within seconds of the first 3D frame.
  Fixed since (2026-11-26, from the forks), not yet tried in a game here:
  the builder's `DispatchX` calls count threads, not groups, and ran up to
  64 times over, writing past its structures; the quarantine leaked through
  other shaders' write declarations in the same list; numbered copies taken
  by one list were missed by another; walkers declared every copy written;
  the upload memory the builder's records lie in was out of reach of
  addresses; free memory and the process heap could be guarded (a freeze).
  With the builder on, the walkers wait until it has run once, and again
  once the host turns it off.
  Fixed since in the translator (2026-11-26), not yet tried in a game: the
  last group of a dispatch that counts threads starts only the lanes up to
  the count (the builder's programs get two instructions in front that cut
  EXEC there, the count a host word); programs that mark loads and stores
  coherent (the bottom-up build and refit: a thread reads the box another
  group's thread stored before an atomic) get globally coherent views and
  device fences; the builder's loops are cut at 65,536 iterations, not the
  compute programs' 4,096; with the builder on, the directory of title
  memory by address follows buffers the title releases or replaces.
- Without the builder (no `VICTORY_BVH_BUILDER=1`) the dynamic structures and
  the top of the tree are never built, and a walk through them read whatever
  their memory held: dark tiles on hair and faces, bright squares in shadow,
  walks that ran to the trace loop limit. The node test now answers "no hit"
  then (a box node returns no child, a triangle node a miss) and reads no
  node, decided when the program is translated: a run with the switch whose
  builder the host turns off later (a builder pipeline that cannot be made)
  keeps the full test; its walkers are then held, not run.
- One shadow-mask trace program does not translate yet
  (`CS_TraceShadowRays_FullAdaptive`: a lane mask joined with numbers).
  Four others now take their structure's address from a table of view
  addresses by heap slot, which exists only with the builder on (without
  it their dispatches are dropped).
- An access to a title page that has no host copy is lost (a read gives zero, a
  store is dropped).
- Not publicly documented, and assumed: the instruction's rounding, the order
  of two boxes entered at the same distance, the result for a ray exactly
  through an edge, and that a child whose pointer is "none" is a miss
  whatever its box (sorted, it would otherwise stand before children that
  are there).
- Cost on the GPU is unmeasured.

## Where to look

- `crates/victory-shader/src/bvh.rs`: the reference for the node test.
- `crates/victory-shader/src/emit/lower.rs`: the lowering of the two
  instructions and of memory by address (`Lower::memory`).
- `crates/victory-shader/src/resource.rs`: address tracking for descriptors
  and raw addresses.
- `crates/victory-gpu/src/pipeline.rs`, `draw.rs`: the builder rule and the
  host side of root views and title memory.
