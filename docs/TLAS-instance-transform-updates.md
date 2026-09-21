B1: Device-side TLAS instance-transform updates — engine TetSolve kernel + LC `update_transforms_from_buffer`
=================================================================================================================
Goal: a cage frame becomes [user wind kernel → engine TetSolve kernel → TLAS PREFER_UPDATE build], all on the GPU stream. No readback, no CPU affine solve, no per-instance setShapeTransform. CPU cost per frame: O(1). Target: 10⁵–10⁶ dynamic cage instances.

Design decisions:
- LC API = stash-style `Accel::set_transform_buffer_on_update(first, BufferView<float4x4>)`, folded into the existing `AccelBuildCommand` (extending the command beats a new command type — a new Tag forces ~8 visitors incl. CUDA/Metal/rust/remote to implement it; `_update_instance_buffer_only` was added exactly this way). Buffer carried as a plain handle like `MeshBuildCommand::_vertex_buffer`.
- New dedicated builtin kernel (`accel_transform_copy.bytes`) rather than extending `accel_process.bytes` — old kernels keep their cache keys; gather layout (thread = instance index, read src, write p0/p1/p2 only, preserve InstanceID/Mask/Flags/AS pointer).
- Engine keeps the current CPU path as a runtime fallback (non-contiguous rows / registration failure / `NT_TETCAGE_CPU_PATH=1`), which doubles as the pixel-diff validation baseline.

-----------------------------------------------------------------------------------------------------
Part 1 — LuisaCompute fork (C:\Users\barca\Projects\LuisaCompute)
-----------------------------------------------------------------------------------------------------
1. Command: include/luisa/runtime/rhi/command.h (~587-677)
   - Add to `AccelBuildCommand`: `uint64_t _transform_src_buffer; uint64_t _transform_src_offset; uint32_t _transform_first_instance; uint32_t _transform_count;` + getters + a static factory or ctor default (0 handle = disabled). Mirror `MeshBuildCommand`'s plain-handle style (no BufferView type in the RHI command layer).

2. Runtime API: include/luisa/runtime/rtx/accel.h + src/runtime/rtx/accel.cpp
   - `void Accel::set_transform_buffer_on_update(uint first_instance, BufferView<float4x4> transforms) noexcept;` — stashes `{first, count, device handle+offset}` under the existing `_mtx`; validated (first+count ≤ _instance_count, else warn + clear).
   - `_build()` (~76-94): fold the stash into the command alongside modifications; clear stash on commit (same drain semantics as `_modifications`). Document: matrices are luisa column-major float4x4; the backend transposes; rows are updated verbatim (no visibility/AABB semantics).

3. Builtin kernel: src/backends/common/hlsl/builtin/accel_transform_copy.bytes (NEW)
   - `RWStructuredBuffer<MeshInst> _InstBuffer:u0` (reuse the MeshInst desc mirror struct), `StructuredBuffer<float4> _Src:t0` (stride 4 = one float4x4 per instance), `cbuffer b0 { uint dsp; uint count; uint first; }`, `[numthreads(256,1,1)]`: id<dsp, inst=first+id< count guard, read c0..c3, write rows `pN = float4(c0.N, c1.N, c2.N, c3.N)` (matches `Modification::set_transform` row-major encoding, command.h:613-627). Touch ONLY p0/p1/p2.
   - Register in the embed list: src/backends/common/hlsl/CMakeLists.txt.
   - Loader `BuiltinKernel::load_accel_transform_copy_kernel` (clone of load_accel_set_kernel, BuiltinKernel.cpp:5-43; 4 properties, cache key "accel_transform_copy_v1") + `Device::set_accel_transform_kernel` LazyLoadShader (DXRuntime/Device.h:84-85, Device.cpp:91).

4. DX backend: src/backends/dx/Resource/TopAccel.cpp / TopAccel.h
   - `PreProcessInst` + `PreProcess` accept the transform-source; next to the existing instBuffer `ComputeUAV` record (TopAccel.cpp:151-155 / 281-285) also record `tracker.Record(BufferView(src), Usage::ComputeRead)` for the source buffer.
   - `Build` (TopAccel.cpp:288-348): when a transform source is present, dispatch the new kernel (cbuffer {count, Length(), first}, src SRV, instBuffer UAV) before `BuildRaytracingAccelerationStructure`; the existing `AccelInstanceBuffer` record + `UpdateState` flush already sequences UAV→build. Dispatch extent = count; scratch/prebuild sizing unchanged.
   - src/backends/dx/DXApi/LCCmdBuffer.cpp (AccelBuildCommand visitors ~387-404 / ~996-1020): pass the new fields through to TopAccel.

5. Housekeeping in the fork:
   - Reorder layer: src/backends/common/command_reorder_visitor.h (~1024-1029) — register the source buffer as a READ dependency of AccelBuildCommand (else the reorder could hoist the build above the producer kernel; compare MeshBuildCommand at :1033-1045).
   - Validation backend: src/backends/validation/stream.cpp (~423-436) — accept the fields, validate bounds.
   - VK backend: explicit `LUISA_NOT_IMPLEMENTED` in the Tlas path if a transform source is set (engine never instantiates VK; guard for safety).
   - Remote/rust: OFF in this build — untouched.
   - Build: `ninja -C C:\Users\barca\Projects\LuisaCompute\build-dx` (DLLs land in build-dx\bin, exactly where the engine loads them). Suggest a fork branch `tlas-transform-from-buffer` on top of 8b6710d90.

-----------------------------------------------------------------------------------------------------
Part 2 — Engine: Geometry "GPU-owned transform rows"
-----------------------------------------------------------------------------------------------------
6. include/newtype/scene/Geometry.h + src/newtype/scene/Geometry.cpp
   - New state: `std::vector<std::pair<uint,uint>> _gpuTransformRanges; uint _topologyGeneration = 0; bool _gpuTransformsDirty = false;` (generation bumped by post-build add_instance/add_shape/remove_shape).
   - New public API:
     - `[[nodiscard]] uint tlas_index_of(ShapeId id) const noexcept;` (resolves via `_slots`, Geometry.h:51-54; returns ~0u for invalid/prototype slots).
     - `bool register_gpu_transform_rows(luisa::span<const ShapeId> ids)` — resolves rows, validates they form ONE contiguous run (v1 LC API constraint; multiple runs or holes → false + log). Records the range; registered rows are excluded from CPU ownership.
     - `void notify_gpu_transforms_dirty() noexcept;` — sets `_gpuTransformsDirty`.
     - `[[nodiscard]] uint topology_generation() const noexcept;`
   - Guards:
     - `set_transform` (Geometry.cpp:745-773) on a registered row: log warning + drop (the GPU writer owns it).
     - `upload_dirty_transforms` (Geometry.cpp:1006-1084): the whole-buffer `many` full-upload (1062) and the resize path (1015-1031) must not clobber GPU rows — when a GPU range exists, use per-index uploads skipping owned rows (topology-change frames are already the slow path).
     - Registration invalidation: any generation bump clears `_gpuTransformRanges` + logs; TetCageGeometry re-resolves and re-registers when it sees a new generation (swap-and-pop stale-row hazard, Geometry.cpp:865-957). Known v1 limitation, documented: a runtime removeShape interleaved with an animated cage can show one stale matrix frame for the swapped row before re-registration.
   - `Geometry::update` (Geometry.cpp:536-545): when `_gpuTransformsDirty` → clear flag, `_tlas.set_transform_buffer_on_update(range.first, _instance_transform_buffer.view(range.first, range.count))`, force `needs_update = true` → existing `_tlas.build(PREFER_UPDATE)` carries the device source. Prev-transform handling for GPU rows needs nothing here (TetSolve writes prev itself).

7. Pipeline: thin scene-facing wrappers in include/newtype/core/Pipeline.h + PipelineUpdate.cpp delegating to `_geom` (`registerGpuTransformRows`, `notifyGpuTransformsDirty`, `topologyGeneration`, `geometryTlasRow(ShapeId)`); `pipeline.geometry()` already exists (Pipeline.h:542).

-----------------------------------------------------------------------------------------------------
Part 3 — Engine: TetCageGeometry GPU solve path
-----------------------------------------------------------------------------------------------------
8. include/newtype/scene/TetCageGeometry.h / src/newtype/scene/TetCageGeometry.cpp
   - New LUISA_STRUCT'd GPU tables built in `build()`:
     - `TetSolveRow { uint4 cage_verts; float4 r0; float4 rinv[3]; uint tlas_row; }` (one per valid (copy,tet), packed 48 B),
     - `Buffer<float4x4> _copyWorldBuffer` (copies ≤ 256 typically),
     - `Buffer<float3> _restCageBuffer/_animCageBuffer/_cageStateBuffer` (existing, unchanged).
   - Internal engine kernel compiled in `build()` (device.compile<1>, ProceduralGeometry-constructor precedent — NOT ShaderManager; the affine solve is engine logic, the user customization point stays the wind shader):
     x = solve-row index → read 4 anim verts, A=[a1-a0,a2-a0,a3-a0], L.col(j)=A·rinv[j], det guard (kMinDet), world = copyWorld · [L|a0−L·r0], write `transforms.write(row.prev_slot… )`: prev[row] = old cur[row], cur[row] = world. Collapsed tet: write prev=cur only (frozen motion, keep matrix — same semantics as the CPU path).
   - `update(pipeline)` GPU path: dirty-state uploads → user wind dispatch (contract unchanged) → TetSolve dispatch(x = rows) → resolve/re-register rows if `topologyGeneration` changed → `pipeline.notifyGpuTransformsDirty()`. NO `copy_to`, NO `synchronize()`. `last_update_ms()` now measures dispatch-only time (≈0) — UI text updated to say so.
   - Registration: after `build()` (instances exist), resolve rows via ShapeIds, `register_gpu_transform_rows`; on failure (non-contiguous — e.g. another shape added between cage copies) log + fall back to the existing CPU path (readback + solve + per-instance setShapeTransform), also forceable via `NT_TETCAGE_CPU_PATH=1` for validation/A-B.
   - `bake_initial_pose()` keeps the build-time one-shot (wind dispatch + single readback at build time is acceptable; CPU solve seeds the initial addPrototypeInstance matrices — current code, unchanged).
   - Motion-blur note: build seeds prev=cur; TetSolve maintains prev per frame; when the cage settles we stop dispatching and prev converges to cur naturally.

9. src/tests/TetCageScene.cpp: no API changes; UI stat line tweak ("dispatch-only"), mention NT_TETCAGE_CPU_PATH in the header comment.

10. Docs: TetCageGeometry.h header block + docs/tetrahedral-cage-prototype.md follow-up section (the doc already anticipates this at :138-148): describe the GPU path, the LC API dependency, fallback, and the removeShape caveat.

-----------------------------------------------------------------------------------------------------
Part 4 — Verification
-----------------------------------------------------------------------------------------------------
11. LC builds clean (ninja -C build-dx); engine Release build links the new DLLs.
12. Correctness: `NT_TETCAGE_FREEZE_AFTER=3` freeze-mode screenshots via `--perf-shot`, GPU path vs `NT_TETCAGE_CPU_PATH=1` — pixel-identical (same matrices through either path). Wind animation + NT_TETCAGE_BAKE_WIND sanity; denoiser ON during motion to confirm prev-transform correctness (no smearing/ghosting regressions).
13. Scaling: NT_TETCAGE_COPIES=64 (≈98k instances) with NT_TETCAGE_LOG_UPDATE=1 — TetCage/Update CPU ms ≈ 0, fps comparison vs CPU path; verify TLAS refit time dominates and scales sublinearly.
14. Regression: default scene set (cornell etc.) unaffected — all Geometry changes gated on a registered GPU range existing; procedural reference path (NT_ENABLE_PROCEDURAL=1 build) still A/B-diffs against the cage.
15. Logs clean: no set_transform-on-owned-row warnings, no registration failures, LC validation-backend stream runs without complaints.

Risks / notes:
- LC embed list must include the new .bytes or the loader fails at runtime (build-time check via CMake target).
- Reorder-visitor read dependency (step 5) is cheap insurance even if the DX stream doesn't currently reorder accel builds.
- VK path intentionally raises not-implemented (engine never instantiates VK).
- One stale-matrix frame possible after runtime removeShape interleaving (documented; scene doesn't do this today).
- Engine keeps working against upstream LC (CPU fallback) — useful if the fork patch is later PR'd upstream.
- Commit strategy: LC changes on a fork branch; engine changes on the engine repo as usual (no commits unless you ask).