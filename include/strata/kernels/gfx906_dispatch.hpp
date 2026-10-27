// include/strata/kernels/gfx906_dispatch.hpp - where the gfx906 (wave64) kernel variants are compiled and launched.
//
// The STRATA_HIP_GFX906 build targets gfx906 alone, so its variants (one-wavefront MMVQ, the mode-7 experts, the
// one-wavefront router, ...) are chosen by `#if defined(STRATA_HIP_GFX906)`. A STRATA_ENABLE_HIP build with a wave64
// arch in its list (STRATA_HIP_WAVE64, e.g. gfx906;gfx1200) is one binary for a gfx906 card AND a wave32 card, so there
// each variant is compiled beside the wave32 code and chosen per device at run time:
//
//   - STRATA_GFX906_KERNELS: the variants exist in this build (either of the two above).
//   - STRATA_GFX906_BODY: this compilation pass may hold a variant's kernel body (the host pass, a gfx906 device pass,
//     or the gfx906 build). Other device passes compile the kernel empty: it is never launched on those cards.
//   - on_gfx906(): the current device is a gfx906 card (cached per device). Constant true in the gfx906 build, so its
//     dispatch stays as it was.
//
// A wave32-only HIP build and the CUDA build define neither macro and never call on_gfx906() (constant false).
#pragma once

#if defined(STRATA_HIP_GFX906) || defined(STRATA_HIP_WAVE64)
#define STRATA_GFX906_KERNELS 1
#endif

#if defined(STRATA_HIP_GFX906) || !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx906__)
#define STRATA_GFX906_BODY 1
#else
#define STRATA_GFX906_BODY 0
#endif

#if defined(STRATA_HIP_WAVE64) && !defined(STRATA_HIP_GFX906)
#include <cuda_runtime.h>

#include <atomic>
#include <cstring>
#endif

namespace strata::kernels {

#if defined(STRATA_HIP_GFX906)
inline bool on_gfx906_device(int) { return true; }
inline bool on_gfx906() { return true; }
#elif defined(STRATA_HIP_WAVE64)
/// gcnArchName of device `dev` starts with "gfx906" (cached; an unknown device counts as not gfx906).
inline bool on_gfx906_device(int dev) {
    static std::atomic<int> cache[64]{};   // 0 not probed, 1 other, 2 gfx906
    if (dev < 0 || dev >= 64) return false;
    int c = cache[dev].load(std::memory_order_relaxed);
    if (c == 0) {
        cudaDeviceProp p{};
        c = cudaGetDeviceProperties(&p, dev) == cudaSuccess && std::strncmp(p.gcnArchName, "gfx906", 6) == 0 ? 2 : 1;
        if (c == 1) (void) cudaGetLastError();
        cache[dev].store(c, std::memory_order_relaxed);
    }
    return c == 2;
}
/// the current device is a gfx906 card
inline bool on_gfx906() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { (void) cudaGetLastError(); return false; }
    return on_gfx906_device(dev);
}
#else
constexpr bool on_gfx906_device(int) { return false; }
constexpr bool on_gfx906() { return false; }
#endif

}  // namespace strata::kernels
