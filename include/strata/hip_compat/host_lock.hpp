#pragma once
// Included by cuda_runtime.h after hip_runtime.h, in wave64 HIP builds only (STRATA_HIP_WAVE64: gfx906 / gfx908 in the
// arch list); wave32-only builds keep the plain HIP calls.
//
// On Linux, ROCm maps host memory for the GPU through a KFD userptr / SVM range and does not pin its pages:
// hipHostRegister, and hipHostMalloc with ROCr's default HSA_USERPTR_FOR_PAGED_MEM, leave them pageable. The engine
// assumes NVIDIA's semantics (a registered range is page-locked by the driver), but here a reclaim, swap or compaction
// of such a page evicts all of the process's GPU queues until amdgpu validates the whole range again - under memory
// pressure that keeps retrying and the engine reports "no progress for ... s". So every cudaHostRegister /
// cudaHostAlloc / cudaMallocHost of the HIP build also mlock()s its range and munlock()s it before it is unregistered
// or freed (Strata #3913). mlock needs `ulimit -l` to cover the ranges: the first failure is reported once and the
// later ranges are left as they were. STRATA_HIP_MLOCK=0 turns it off.
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#if defined(__linux__)
#include <sys/mman.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#endif

namespace strata::hip_compat {

#if defined(__linux__)
struct HostLocks {
    std::mutex mu;
    std::unordered_map<void*, size_t> locked;   // range start -> bytes mlock()ed
    bool off = false;                           // STRATA_HIP_MLOCK=0, or mlock failed once
};
inline HostLocks& host_locks() {
    static HostLocks* h = [] {
        auto* l = new HostLocks;   // (never destroyed: ranges may be freed during static destruction)
        const char* v = std::getenv("STRATA_HIP_MLOCK");
        l->off = v != nullptr && v[0] == '0';
        return l;
    }();
    return *h;
}
inline void host_lock(void* p, size_t bytes) {
    if (p == nullptr || bytes == 0) return;
    HostLocks& h = host_locks();
    std::lock_guard<std::mutex> lk(h.mu);
    if (h.off || h.locked.count(p)) return;
    if (mlock(p, bytes) != 0) {
        const int e = errno;
        h.off = true;
        std::fprintf(stderr,
                     "hip host memory: mlock of %.2f GB failed (%s) - registered host memory stays pageable, and memory "
                     "pressure can stall the GPUs; raise the locked-memory limit (ulimit -l / memlock in "
                     "/etc/security/limits.conf) to keep it resident (STRATA_HIP_MLOCK=0 silences this)\n",
                     (double) bytes / 1073741824.0, std::strerror(e));
        return;
    }
    h.locked.emplace(p, bytes);
}
inline void host_unlock(void* p) {
    if (p == nullptr) return;
    HostLocks& h = host_locks();
    std::lock_guard<std::mutex> lk(h.mu);
    const auto it = h.locked.find(p);
    if (it == h.locked.end()) return;
    munlock(it->first, it->second);
    h.locked.erase(it);
}
#else
inline void host_lock(void*, size_t) {}
inline void host_unlock(void*) {}
#endif

inline hipError_t host_register(void* p, size_t bytes, unsigned int flags) {
    const hipError_t e = hipHostRegister(p, bytes, flags);
    if (e == hipSuccess) host_lock(p, bytes);
    return e;
}
inline hipError_t host_unregister(void* p) {
    host_unlock(p);
    return hipHostUnregister(p);
}
inline hipError_t host_malloc(void** p, size_t bytes, unsigned int flags) {
    const hipError_t e = hipHostMalloc(p, bytes, flags);
    if (e == hipSuccess && p != nullptr) host_lock(*p, bytes);
    return e;
}
inline hipError_t host_free(void* p) {
    host_unlock(p);
#if HIP_VERSION_MAJOR < 7
    return hipHostFree(p);
#else
    return hipFreeHost(p);
#endif
}

}  // namespace strata::hip_compat
