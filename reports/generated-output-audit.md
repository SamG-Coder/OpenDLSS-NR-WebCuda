# Audit of emitted WGSL

Inspected WebCuda 5aa80e3 output from renderer af26c75, alongside the locally generated upstream production shader at 9d08f41. No upstream shader is copied into this repository or used by the application. This is a WGSL source audit and paired GPU timing experiment, not inspection of the browser driver's final machine instructions.

## Measured changes to emitted shaders

RTX 5080, Edge 153, six alternating warm samples per configuration, same model and gradient, current native-half path on both sides. Full renderer wall time includes final output readback; cold setup is excluded. The audit harness intercepts half-kernel artifacts only when constructing the experimental renderer. Production artifacts are untouched.

| Diagnostic change | 720p baseline | 720p edited | 1080p baseline | 1080p edited |
| --- | ---: | ---: | ---: | ---: |
| Remove storage barriers inside half GEMMs | 408.05 ms | 408.10 ms | 953.70 ms | 950.35 ms |
| Replace leading-bit search loop with firstLeadingBit | 409.15 ms | 384.85 ms | 952.05 ms | 904.35 ms |

Every output hash matched within each six-sample comparison at both resolutions. The bit scan reduced time by 5.9% and 5.0%. Removing storage barriers did not produce a meaningful benefit in this run. These edits are diagnostics, not shipped compiler optimizations, and full intermediate parity testing is still required before promoting them.

## Findings and ownership

1. **Software leading-bit search survives into the inner arithmetic.** `kernels/numeric.cuh` implements `nr_fixed_half` using a shift loop to locate the most significant bit. The emitted `f_nr_fixed_half` retains that variable-length loop and runs once per output per 16-term group. Replacing that exact loop with `firstLeadingBit(v_mag)` is equivalent because zero returns earlier. The proper CUDA implementation would express this using `31u - __clz(mag)` and add a typed CUDA intrinsic mapping in WebCuda. This is a source/intrinsic coverage issue, not evidence of incorrect compilation.

2. **Conservative storage barriers.** Compiler storage-use analysis marks a buffer read/write anywhere in the kernel, then emits `storageBarrier()` beside every `__syncthreads()`. Output packing uses atomics after the reduction loop, yet this marks all shared-tile barriers too. These GEMMs need shared synchronization there, but have no storage dependency crossing those barriers. A general compiler fix must be control-flow aware; blanket removal would break other CUDA programs. The isolated test shows this is not the main bottleneck on the tested device.

3. **Packed stores contend between neighboring threads.** The selected 8x16 half GEMM has 128 threads and one output per thread. Its output and optional raw bindings are atomic u32 arrays. `nr_activation_store` uses compare/exchange to preserve neighboring packed lanes: four FP8 output threads, or two half-output threads, share a word. CUDA strong CAS additionally lowers to a WGSL weak-CAS retry loop, correctly preserving semantics. Upstream's representative tile128 shader assigns eight outputs per thread and packs whole words with one owner. Addressing this requires changing CUDA output ownership; merely dropping atomics is incorrect. Earlier multi-output CUDA variants were not dependable wins, so this must be tested with the new half/exponent layout.

4. **Rounding and publication perform extra work.** Ours preserves software half rounding and bit manipulation in each group, even though half products now use native f16. Packed output goes through `nr_quant` (encode E4, decode to float), then `nr_activation_store` encodes E4 again. Upstream's generated path directly packs codes, uses native half publication and precomputed activation/weight tables. Removing redundant encode/decode is a CUDA/storage-generator improvement. Replacing all software half arithmetic with f16 needs separate proofs for rounding, subnormals, overflow and signed zero; exact half products alone do not justify it.

5. **Different reuse and tile layouts.** Our inspected half kernel stages 8x16 outputs with K steps of 16, using 1,536 shared bytes. The upstream tile128 example stages 32x32 outputs with K steps of 32, four-lane half vectors, padded/transposed weight tiles, and 8,704 shared bytes. It reuses each operand across more outputs and synchronizes less often per output. The adapter's 32 KiB workgroup limit is distinct from the shader's actual shared allocation; simply requesting 32 KiB does not reproduce this layout. Bank-conflict or occupancy benefits cannot be established from WGSL alone.

6. **Constant folding and loop optimization are deferred to the driver.** Our emitted source retains literal-false branches such as partition-disabled paths, format branches inside helpers, fixed eight-iteration loops, repeated scale expressions, and unused helper arguments. WebCuda removes unreachable functions by dependency, but it does not fully specialize helper bodies or optimize these loops. Upstream explicitly unrolls reduction groups and specializes shapes. The browser driver may already remove many of our source-level redundancies, so source length or apparent repetition is not a speed measurement. The barrier result illustrates this distinction.

## Reproduction and next implementation

Set local `NR_DLL` and `NR_BROWSER`, run `npm run build`, then set `NR_AUDIT=bitscan` or `NR_AUDIT=barriers` and run `npm run audit:generated`. Raw JSON remains ignored. The harness deliberately fails if no eligible shader is modified. Output hash disagreement fails the run.

The first justified production change is CUDA `__clz` support plus the equivalent leading-bit replacement, followed by all intermediate parity checks. Next, separately test direct code publication and whole-word output ownership. Treat wider tiles, full half publication and table-based numerics as independent experiments. No finding supports attributing the entire upstream speed advantage to WebCuda compiler overhead.
