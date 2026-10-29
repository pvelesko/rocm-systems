# NCCL LTO IR parity gaps

What `librccl_device.ltoir` (`NCCL_EMIT_LTO_IR`) does **not** match about
NCCL's `libnccl_device.ltoir`, and why each divergence was left in place.

This is an internal engineering document. It is deliberately not listed in
`docs/sphinx/_toc.yml.in`, so it does not publish to rccl.readthedocs.io.

| Field | Value |
| --- | --- |
| Baseline | RCCL `develop` @ `de35f5278b` |
| Toolchain | ROCm 7.2.4, `gfx950` |
| Artifact | `$BUILDDIR/lib/librccl_device.ltoir` |
| Companion | `$BUILDDIR/lib/librccl_device.bc` (`EMIT_LLVM_IR`) |

Everything below was measured on that configuration. Numbers that a
toolchain bump could move are called out as such.

## Scope rule this document obeys

A capability that exists in RCCL and was not reaching the LTO IR surface
got wired up. A capability that exists in NCCL and not in RCCL gets
recorded here and nothing else. No gap below was closed, and none of them
should be closed as a side effect of LTO IR work — each is either a device
API question or a pre-existing property of the `.bc` path.

## 1. The export surface is 82 symbols, not NCCL's 96

Both artifacts export exactly 82 defined code symbols, and it is the same
set in both (`test_export_surfaces_match`). The arithmetic against
`bindings/ir/nccl_device_wrapper.h`:

| | count |
| --- | --- |
| `NCCL_IR_EXPORT` declarations in the header | 96 |
| of which fp8, `#if !defined(__HIPCC__)` | 14 |
| **AMD-visible, and measured in both artifacts** | **82** |
| `NCCL_IR_EXTERN_C __device__` declarations (not exported — see §3) | 16 |

So the whole export-surface delta against NCCL is the fp8 block. There is
no symbol that RCCL implements but fails to publish through the LTO IR
path; the LTO IR surface and the bitcode surface agree exactly, by two
independent mechanisms (§4).

## 2. The 14 fp8 entry points are a device-API gap, not an LTO IR gap

`bindings/ir/util.h:16` defines `NCCL_IR_FP8_TYPES(F)` as empty on
`__HIP_PLATFORM_AMD__`, with the comment "RCCL's device ReduceSum/Copy has
no fp8 support, so the IR exposes no FP8 entry points." The declarations in
`nccl_device_wrapper.h` are correspondingly guarded at lines 394, 419, 456,
481, 518, 543 and 580 — seven families, two types each:

```
ncclLsaReduceSum_{F8E4M3,F8E5M2}          ncclMultimemReduceSum_{F8E4M3,F8E5M2}
ncclLsaCopy_{F8E4M3,F8E5M2}               ncclMultimemCopy_{F8E4M3,F8E5M2}
ncclLsaReduceSumCopy_{F8E4M3,F8E5M2}      ncclMultimemReduceSumCopy_{F8E4M3,F8E5M2}
ncclLocalReduceSumCopy_{F8E4M3,F8E5M2}
```

The underlying device support is gated one level down on
`__CUDA_FP8_TYPES_EXIST__`, in
`src/include/nccl_device/impl/reduce_copy__types.h:73` (and in
`multimem__funcs.h`, `vector__funcs.h`, `vector__types.h`). That macro is
referenced in 14 places across the tree and **defined in none of them** —
nor anywhere under `/opt/rocm/include`. So those blocks are dead in every
RCCL build, CUDA-shaped or not.

This means the gap is in RCCL's *device* reduce/copy API, not in the IR
emission. Emitting the thunks from either artifact would produce calls into
specializations that do not exist. Closing it is device-header work.

Note that RCCL's **host** collectives do support `ncclFloat8e4m3` /
`ncclFloat8e5m2`; that is `src/device/reduce_kernel.h`'s own path and is
unrelated to the device API surfaced here.

## 3. 16 declared device entry points are not on either surface

`nccl_device_wrapper.h` declares 16 symbols with `NCCL_IR_EXTERN_C
__device__` rather than `NCCL_IR_EXPORT`:

```
ncclDevComm_{Rank,NRanks,LsaRank,LsaSize,LsaMultimem,LsaBarrier}
ncclDevComm_{RailGinBarrier,WorldGinBarrier,HybridLsaBarrier,HybridRailGinBarrier}
ncclBarrierSessionDestroy       ncclBarrierSession_C_size
ncclLsaBarrierSessionDestroy    ncclLsaBarrierSession_C_size
ncclGinBarrierSessionDestroy    ncclGinBarrierSession_C_size
```

`NCCL_IR_EXPORT` is what carries `used, retain, visibility("default")`.
Without it these symbols land in both artifacts as **local** (`llvm-nm`
type `t`, confirmed on both), so a consumer referencing them from its own
translation unit does not resolve against either file.

They are equally absent from the `.bc`, so this is a pre-existing property
of `develop` and not something the LTO IR path introduced or could fix
unilaterally — the two artifacts are *supposed* to publish the same
surface, and they do. Promoting these 16 would change
`librccl_device.bc`'s published ABI, which is a deliberate decision about
the device API and belongs in its own change.

Nothing in `test/ir-device` references any of the 16, so the suite passing
is not evidence against this.

## 4. The two artifacts reach the same surface by different routes

| | mechanism |
| --- | --- |
| `.bc` | `opt -internalize-public-api-list=<exact 82-name list>` (`bindings/ir/Makefile:154`, `bindings/ir/CMakeLists.txt:217`) |
| `.ltoir` | `opt --passes=internalize` with no list; the root set is `@llvm.used`, populated by `NCCL_IR_EXPORT`'s `used, retain` |

NCCL uses an `nccl*` glob for the list, which RCCL does not; RCCL's `.bc`
path enumerates names.

A consequence worth knowing before adding a thunk: the `.ltoir` surface
updates automatically from the `NCCL_IR_EXPORT` declaration, while the
`.bc` surface needs the name hand-added in **two** places that must stay in
sync. `test_export_surfaces_match` fails loudly if they drift, which is why
that test asserts set equality rather than just checking for mangled names.

## 5. No ftz / align IR cleanup is needed or performed

NCCL's recipe strips `nvvm-reflect-ftz` module metadata and `align`
attributes because libNVVM rejects them. Both are NVVM-specific. AMDGPU has
no analogue and the AMDGPU backend needs no equivalent pass, so the RCCL
recipes do nothing here; see the notes at `bindings/ir/Makefile:170,286`
and `bindings/ir/CMakeLists.txt:268`.

## 6. The `.ltoir` file holds LLVM bitcode, not NVIDIA LTO IR

`file(1)` reports "LLVM IR bitcode" and the magic is `42 43 c0 de` (`BC`),
identical in kind to `librccl_device.bc`. The two differ in *content*, not
format: the `.ltoir` is compiled `-O1 -Xclang -disable-llvm-passes`, so the
optimizer has not run and inlining is deferred to the consumer's device
link.

The extension is NCCL's for interface parity. A consumer globbing
`lib*_device.ltoir` in `$BUILDDIR/lib` works unchanged across the two
projects, and — unlike globbing `*.bc` — it cannot accidentally pick up
`librccl_device.bc`, which is the collapsed artifact with the opposite
inlining policy.

## 7. Install divergence

NCCL installs neither artifact. RCCL installs both:
`librccl_device.bc` under `if(EMIT_LLVM_IR)` and `librccl_device.ltoir`
under `if(NCCL_EMIT_LTO_IR)` (`CMakeLists.txt:934,938`), alongside the
`nccl_device` headers. Deliberate: RCCL has an in-tree device-LTO consumer
(`test/ir-device`) and ships the device API as a supported surface.

## 8. Incidental findings, recorded and not acted on

These surfaced while measuring the above. None is an LTO IR defect; all
predate this work.

**`IR_gin_mpi_test.cpp` does not compile against `develop`'s header.**
The `ncclGinBarrierSessionInit` call at line 160 and the
`ncclBarrierSessionInit` call at line 182 pass `net` by value (a
`ncclGin_C` built at lines 149 and 171) where
`nccl_device_wrapper.h:318,339` declare the parameter as `ncclGin_C const*
net`. This is the 2 pre-existing errors in the `test/ir-device` baseline
(`19 passed, 2 errors` at the end of this work, against `7 passed, 2
errors` before it), unchanged either way.

**Two dead LDS globals in the `.ltoir`.** The artifact contains four
`addrspace(3)` globals (the `.bc` has zero): the function-local stashes of
`ncclCoopBcast_WarpSpan_stash()` and `ncclCoopBcast_Cta_stash()`
(`src/include/nccl_device/coop.h:405,419`). They are internal-linkage and
unreachable from any exported thunk; they survive only because the LTO IR
recipe deliberately runs no DCE, and a consumer's link removes them.
`test_no_reachable_lds` asserts the reachability property rather than a
count, precisely so this expected residue does not read as a regression.

**The `llvm_ir` path rebuilds unconditionally, by two separate
mechanisms.** In `bindings/ir/Makefile`, `$(UNOPTIMIZED_BC)` lists the
`.PHONY` target `check_hipify` as a *normal* prerequisite
(`bindings/ir/Makefile:174`, and identically at `:140` on `develop`), which
makes it perpetually out of date — a no-op `make build` still recompiles
the `.bc`. In CMake, `cmake -E copy_if_different` does not bump the
destination mtime when the contents match, so once
`nccl_device_wrapper.h` is touched the staging rule re-runs on every
subsequent build. Both are pre-existing `develop` behaviour on the
`EMIT_LLVM_IR` path.

The LTO IR rules deliberately do not reproduce the first one:
`$(LTO_IR_RAW)` takes `check_hipify` as an **order-only** prerequisite
(`| check_hipify`), which still runs the staging check first without
marking the target stale. A no-op `make ltoir` regenerates nothing, and
touching `librccl_device.bc` does not regenerate `librccl_device.ltoir`.

**The artifacts are not byte-comparable across build systems.** clang
embeds the input path in `source_filename` and the output path in
`ModuleID`. The Makefile passes a relative input path and CMake an absolute
one, and each build directory gives a different output path, so md5
comparisons are only meaningful within one build system *and* one build
directory. This is worth stating because it cost a false regression alarm
once already.
