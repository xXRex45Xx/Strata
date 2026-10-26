#pragma once
// Included only by STRATA_ENABLE_HIP builds. CUDA builds use NVIDIA headers.
#if defined(_WIN32)
// hip_runtime.h pulls in <windows.h> on Windows: keep its min/max macros and the rarely used APIs out of the engine.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#ifdef STRATA_HIP_WAVE64
// On wave64 cards (gfx906, gfx908) HIP's own warp-sync builtins would run across the whole 64-lane wavefront and break
// every kernel's reduction order; disable them and let intrinsics.hpp supply the logical-32-lane-half mapping.
// Must come before hip_runtime.h. Wave32-only builds leave HIP's builtins alone: they compile exactly as before.
#ifndef HIP_DISABLE_WARP_SYNC_BUILTINS
#define HIP_DISABLE_WARP_SYNC_BUILTINS 1
#endif
#endif  // STRATA_HIP_WAVE64
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#ifdef STRATA_HIP_WAVE64
#include <atomic>
#include "host_lock.hpp"
#endif
// Do not let HIP's legacy macro corrupt libstdc++ attribute names.
#ifdef __noinline__
#undef __noinline__
#endif
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrClockRate hipDeviceAttributeClockRate
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMinor hipDeviceAttributeComputeCapabilityMinor
#define cudaDevAttrIntegrated hipDeviceAttributeIntegrated
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorStreamCaptureUnsupported hipErrorStreamCaptureUnsupported
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventBlockingSync hipEventBlockingSync
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#ifdef STRATA_HIP_WAVE64
#define cudaFreeHost(p) (::strata::hip_compat::host_free(p))
#elif HIP_VERSION_MAJOR < 7
#define cudaFreeHost hipHostFree
#else
#define cudaFreeHost hipFreeHost
#endif
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributePreferredSharedMemoryCarveout hipFuncAttributePreferredSharedMemoryCarveout
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphUpload hipGraphUpload
#define cudaGraph_t hipGraph_t
#ifdef STRATA_HIP_WAVE64
// Mapped host memory is how the GPU and the CPU talk inside a running graph (the doorbells, the CPU experts' results,
// the verify flags): a kernel spins on a word the CPU writes, and the CPU on a word the kernel writes. On gfx906 that
// needs FINE-GRAINED (coherent) memory - coarse-grained host memory may sit in the GPU's L2 for the whole kernel and the
// spin never sees the other side (the STRATA_HIP_GFX906 build makes every mapped allocation coherent for this reason).
// The wave32 cards keep the plain hipHostMalloc (the Strix Halo / RDNA and APU paths depend on it), and a wave64 arch
// beside wave32 ones (gfx906;gfx1200) is one binary, so this is decided per device at run time: a portable allocation
// (any GPU may read it) is coherent when any visible GPU is wave64, any other one when the current GPU is.
inline bool strata_hip_device_wave64(int dev) {
    static std::atomic<int> cache[64]{};   // 0 not probed yet, 1 wave32, 2 wave64
    if (dev < 0 || dev >= 64) return true;
    int c = cache[dev].load(std::memory_order_relaxed);
    if (!c) {
        int ws = 64;
        if (hipDeviceGetAttribute(&ws, hipDeviceAttributeWarpSize, dev) != hipSuccess) ws = 64;
        c = ws == 64 ? 2 : 1;
        cache[dev].store(c, std::memory_order_relaxed);
    }
    return c == 2;
}
inline bool strata_hip_any_wave64() {
    static const bool any = [] {
        int n = 0;
        if (hipGetDeviceCount(&n) != hipSuccess) return true;
        for (int d = 0; d < n; ++d)
            if (strata_hip_device_wave64(d)) return true;
        return false;
    }();
    return any;
}
inline hipError_t cudaHostAlloc(void** p, size_t bytes, unsigned int flags) {
    if (flags & hipHostMallocMapped) {
        int dev = 0;
        const bool wave64 = (flags & hipHostMallocPortable) ? strata_hip_any_wave64()
                            : hipGetDevice(&dev) != hipSuccess || strata_hip_device_wave64(dev);
        if (wave64) flags |= hipHostMallocCoherent;
    }
    return ::strata::hip_compat::host_malloc(p, bytes, flags);
}
template <typename T>
inline hipError_t cudaHostAlloc(T** p, size_t bytes, unsigned int flags) {
    return cudaHostAlloc(reinterpret_cast<void**>(p), bytes, flags);
}
#else
#define cudaHostAlloc hipHostMalloc
#endif  // STRATA_HIP_WAVE64
#if HIP_VERSION_MAJOR < 7
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaHostAllocPortable hipHostMallocPortable
#else
#define cudaHostAllocDefault hipHostAllocDefault
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaHostAllocPortable hipHostAllocPortable
#endif
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#ifdef STRATA_HIP_WAVE64
#define cudaHostRegister(p, bytes, flags) (::strata::hip_compat::host_register((p), (bytes), (flags)))
#else
#define cudaHostRegister hipHostRegister
#endif
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly
#ifdef STRATA_HIP_WAVE64
#define cudaHostUnregister(p) (::strata::hip_compat::host_unregister(p))
#else
#define cudaHostUnregister hipHostUnregister
#endif
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaMalloc hipMalloc
#define cudaMallocHost(...) (::strata::hip_compat::malloc_host(__VA_ARGS__))
#if defined(_WIN32)
// Windows: hipMemGetInfo counts this process's own allocations only, so the desktop's and other programs' share of
// the card is invisible to it, and VRAM sized by it overfills the card (WDDM then moves memory out to system RAM and
// the GPU slows down).  Its free figure is lowered by what Windows' video memory budget for this process withholds
// (src/core/device.cu; STRATA_WDDM_BUDGET=0: hipMemGetInfo as it is).
namespace strata::hip_compat {
hipError_t mem_get_info(size_t* free_bytes, size_t* total_bytes);
}
#define cudaMemGetInfo ::strata::hip_compat::mem_get_info
#else
#define cudaMemGetInfo hipMemGetInfo
#endif
#define cudaMemcpy hipMemcpy
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpyPeer hipMemcpyPeer
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaSetDevice hipSetDevice
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamCreate hipStreamCreate
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamQuery hipStreamQuery
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

namespace strata::hip_compat {
template <typename T>
inline hipError_t malloc_host(T** pointer, size_t bytes) {
#ifdef STRATA_HIP_WAVE64
    return host_malloc(reinterpret_cast<void**>(pointer), bytes, hipHostMallocDefault);
#else
    return hipHostMalloc(reinterpret_cast<void**>(pointer), bytes, hipHostMallocDefault);
#endif
}
}  // namespace strata::hip_compat

template <typename Kernel>
inline hipError_t cudaFuncSetAttribute(Kernel kernel, hipFuncAttribute attribute, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attribute, value);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph, unsigned long long flags) {
    return hipGraphInstantiateWithFlags(exec, graph, flags);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph,
                                     hipGraphNode_t* error, char* log, size_t size) {
    return hipGraphInstantiate(exec, graph, error, log, size);
}
#define __trap() __builtin_trap()   // the compiled-out sm_80 paths (never selected on AMD)
#define cudaMemcpyToSymbol(symbol, ...) hipMemcpyToSymbol(HIP_SYMBOL(symbol), __VA_ARGS__)
#define cudaMemcpyFromSymbol(dst, symbol, ...) hipMemcpyFromSymbol(dst, HIP_SYMBOL(symbol), __VA_ARGS__)

#include "intrinsics.hpp"
