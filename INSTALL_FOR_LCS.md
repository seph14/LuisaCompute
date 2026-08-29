# Installing LuisaCompute for LCS (and other find_package consumers)

This doc covers the additional step needed when an external project (e.g.
[LuisaComputeSimulator](https://github.com/.../LuisaComputeSimulator)) wants to
consume LuisaCompute via `find_package(LuisaCompute CONFIG)` against a
**pre-built install tree**, instead of rebuilding LuisaCompute from source via
`add_subdirectory`.

If you only need to build LuisaCompute itself or its examples/tests, the
general [BUILD.md](BUILD.md) is enough — skip this file.

## Why this exists

`add_subdirectory` works but wastes time: every consumer recompiles LuisaCompute
from source. Worse, if the consumer's compile flags drift from how the host
(e.g. NewTypeEngine) builds LuisaCompute, COMDAT-folded inline functions can
disagree at runtime — that class of bug caused a Release-only
`luisa-ast.dll` access violation in NewTypeEngine's LCS integration
(see `session-2026-06-27-lcs-integration-wiring` in NewTypeEngine's memory).

The fix is `find_package(LuisaCompute CONFIG)` against a single canonical
install tree. Both LCS and NewTypeEngine link against the same set of DLLs and
import libs, compiled once with identical flags.

## Prerequisites

- A configured-and-built LuisaCompute tree at `build-dx` (Release) and/or
  `build-dx-debug` (Debug). See [BUILD.md](BUILD.md).
- Both configs are needed if you want to consume both — the CRT flavor
  (`/MD` vs `/MDd`) must match the consumer.

## Install

From the repo root:

```bat
:: Release
scripts\install_for_lcs.bat release

:: Debug
scripts\install_for_lcs.bat debug
```

This produces:

```
install-dx/                          (or install-dx-debug/)
├── bin/
│   ├── luisa-ast.dll
│   ├── luisa-core.dll
│   ├── luisa-dsl.dll
│   ├── luisa-runtime.dll
│   ├── luisa-xir.dll
│   ├── luisa-osl.dll
│   ├── luisa-backend-dx.dll
│   ├── luisa-backend-vk.dll
│   └── luisa-validation-layer.dll
├── lib/
│   ├── luisa-ast.lib    (etc. — import libs for the DLLs above)
│   └── cmake/LuisaCompute/
│       ├── LuisaComputeConfig.cmake            (script-generated)
│       ├── LuisaComputeConfigVersion.cmake     (script-generated)
│       ├── LuisaComputeTargets.cmake           (copied from build-dx)
│       └── LuisaComputeTargets-release.cmake   (or -debug.cake)
└── include/
    └── luisa/...        (public API + bundled ext headers under luisa/ext/)
```

## What the script works around

The script does three things on top of `cmake --install`:

1. **Tolerates the test-install failure.** `cmake --install` will exit non-zero
   in `src/tests/cmake_install.cmake` because not all `test_*.exe` are built.
   All DLLs, import libs, and headers install *before* that step, so the
   failure is harmless. The script verifies the critical files landed.

2. **Copies the EXPORT Targets file** from
   `build-dx/src/CMakeFiles/Export/<hash>/LuisaComputeTargets.cmake` into the
   install tree. The hash subdir is a CMake implementation detail; `cmake
   --install` is supposed to do this copy as its final step but never gets
   there because of (1).

3. **Writes `LuisaComputeConfig.cmake`** with two workarounds for upstream
   bugs:
   - `find_dependency(Threads)` — the exported `luisa::luisa-compute-ext`
     target references `Threads::Threads`, which is not auto-imported.
   - Restores `INTERFACE_INCLUDE_DIRECTORIES` on the bundled-ext targets
     (`EASTL-interface`, `EABase`, `spdlog_header_only`, `xxhash`,
     `magic_enum`, `half`, `luisa-compute-ext-marl-interface`,
     `luisa-compute-ext-stb-interface`, `reproc-interface`). LuisaCompute
     sets these with `$<BUILD_INTERFACE:>` only, so the include path is
     stripped at install time even though the headers are installed under
     `<prefix>/include/luisa/ext/<subdir>/`.

## Consumer workflow

After running the install, point your consumer project at the install tree:

```bat
:: In LCS:
cmake -S . -B build-dx -G "Visual Studio 17 2022" -A x64 ^
  -DLCS_LUISA_COMPUTE_INSTALL_DIR="<path-to>/LuisaCompute/install-dx" ^
  -DLCS_NO_INTERNAL_FIBER_SCHEDULER=ON ^
  -DLCS_BUILD_MAIN_APPLICATION=OFF -DLCS_BUILD_PYBINDINGS=OFF ^
  -DLCS_ENABLE_GUI=OFF -DLCS_ENABLE_TEST=OFF
```

In NewTypeEngine, the existing vcxproj link line already pulls
`luisa-*.lib` from `build-dx/lib/` — no change needed for NTE itself. The
install tree is purely for LCS's find_package consumption.

## Maintenance

- **Need to free disk space?** Both `install-dx/` and `install-dx-debug/` are
  100% reproducible from the build dirs. Safe to delete and regenerate.
- **After pulling LuisaCompute changes:** rebuild `build-dx` / `build-dx-debug`,
  then re-run `install_for_lcs.bat` to refresh the install trees.
- **The install trees are ABI-coupled to the consumers.** LuisaCompute passes
  plain structs across the runtime/backend DLL boundary (e.g.
  `BindlessArrayUpdateCommand::Modification` grew from 64 to 72 bytes in the
  Aug 2026 upstream sync). A consumer compiled against stale install headers
  but running a fresh backend DLL misparses those structs and crashes on
  `stream << bindless_array.update()` (garbage operation values hit the
  backend's validation asserts). After refreshing either install tree,
  **rebuild every consumer** (LCS, NewTypeEngine) before running it. Do not
  mix: engine binaries and `luisa-*.dll`s deployed next to them must come from
  the same install generation. **This includes `luisa-validation-layer.dll`**
  — it proxies the whole `DeviceInterface` ABI, and a stale one crashes with
  access violations in `Stream::_dispatch` even when every other DLL is fresh.
- **The script now builds what it installs.** Before installing, it
  incrementally builds `luisa-compute-coro`, `luisa-compute-osl`,
  `luisa-compute-backend-dx`, `luisa-compute-validation-layer` (plus
  `luisa-compute-backend-vk` in release), and fails loudly if any target
  fails to build. This exists because `cmake --install` used to abort at
  `src/coro` when coro wasn't built and then silently re-copy stale backend
  DLLs as "Up-to-date" (this bit us twice in the Aug 2026 sync: the debug
  backend and the debug validation layer both shipped stale).
- **If `cmake --install` ever stops failing at the tests step** (e.g. upstream
  fixes the test install rules, or you build all the test exes), the script
  will still work — the failure path is the expected one but not required.
