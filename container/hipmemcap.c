/* hipmemcap.c - per-process device-memory cap for HIP applications (AMD).
 *
 * Loaded with LD_PRELOAD. Wraps hipMalloc/hipFree (and the async variants),
 * tracks live allocations and returns hipErrorOutOfMemory once the total would
 * exceed HIPMEM_LIMIT_BYTES. Plays the role of CUDA_MPS_PINNED_DEVICE_MEM_LIMIT.
 *
 * Build:  gcc -O2 -fPIC -shared -o libhipmemcap.so hipmemcap.c -ldl -lpthread
 * Use:    LD_PRELOAD=/opt/gpushare/lib/libhipmemcap.so HIPMEM_LIMIT_BYTES=8589934592 app
 *
 * Limits: allocations made through other entry points (hipMallocManaged,
 * hipExtMallocWithFlags, hipMallocPitch, symbols resolved via hipGetProcAddress)
 * are not counted. Verify coverage for each framework with tests/mem_cap_test.py.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef int hipError_t;
typedef void *hipStream_t;
#define hipSuccess 0
#define hipErrorOutOfMemory 2

#define TABLE_SIZE (1 << 16)                 /* open-addressing table of live blocks */
static void *keys[TABLE_SIZE];
static size_t sizes[TABLE_SIZE];
static size_t used = 0, limit = 0;
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;

static void init_limit(void) {
    if (limit) return;
    const char *s = getenv("HIPMEM_LIMIT_BYTES");
    limit = s ? strtoull(s, NULL, 10) : SIZE_MAX;
    if (!limit) limit = SIZE_MAX;
}

static size_t slot(void *p) { return ((uintptr_t)p >> 8) & (TABLE_SIZE - 1); }

static void track(void *p, size_t n) {
    for (size_t i = slot(p), k = 0; k < TABLE_SIZE; i = (i + 1) & (TABLE_SIZE - 1), k++)
        if (!keys[i] || keys[i] == (void *)1) { keys[i] = p; sizes[i] = n; return; }
}

static size_t untrack(void *p) {
    for (size_t i = slot(p), k = 0; k < TABLE_SIZE && keys[i]; i = (i + 1) & (TABLE_SIZE - 1), k++)
        if (keys[i] == p) { keys[i] = (void *)1; return sizes[i]; }   /* tombstone */
    return 0;
}

static int reserve(size_t n) {               /* 1 = allowed, 0 = over the cap */
    int ok;
    pthread_mutex_lock(&mu);
    init_limit();
    ok = (used + n <= limit);
    if (ok) used += n;
    pthread_mutex_unlock(&mu);
    if (!ok) fprintf(stderr, "[hipmemcap] denied %zu B (used %zu / limit %zu)\n", n, used, limit);
    return ok;
}

static void commit(void *p, size_t n, hipError_t rc) {
    pthread_mutex_lock(&mu);
    if (rc == hipSuccess && p) track(p, n); else used -= n;
    pthread_mutex_unlock(&mu);
}

static void release(void *p) {
    pthread_mutex_lock(&mu);
    used -= untrack(p);
    pthread_mutex_unlock(&mu);
}

typedef hipError_t (*malloc_fn)(void **, size_t);
typedef hipError_t (*free_fn)(void *);
typedef hipError_t (*malloc_async_fn)(void **, size_t, hipStream_t);
typedef hipError_t (*free_async_fn)(void *, hipStream_t);

/* Resolve the real HIP symbol (next in link order) once per wrapper. */
#define REAL(name, type) static type real_##name; \
    if (!real_##name) real_##name = (type)dlsym(RTLD_NEXT, #name)

hipError_t hipMalloc(void **ptr, size_t n) {
    REAL(hipMalloc, malloc_fn);
    if (!reserve(n)) return hipErrorOutOfMemory;
    hipError_t rc = real_hipMalloc(ptr, n);
    commit(rc == hipSuccess ? *ptr : NULL, n, rc);
    return rc;
}

hipError_t hipFree(void *p) {
    REAL(hipFree, free_fn);
    hipError_t rc = real_hipFree(p);
    if (rc == hipSuccess && p) release(p);
    return rc;
}

hipError_t hipMallocAsync(void **ptr, size_t n, hipStream_t s) {
    REAL(hipMallocAsync, malloc_async_fn);
    if (!reserve(n)) return hipErrorOutOfMemory;
    hipError_t rc = real_hipMallocAsync(ptr, n, s);
    commit(rc == hipSuccess ? *ptr : NULL, n, rc);
    return rc;
}

hipError_t hipFreeAsync(void *p, hipStream_t s) {
    REAL(hipFreeAsync, free_async_fn);
    hipError_t rc = real_hipFreeAsync(p, s);
    if (rc == hipSuccess && p) release(p);
    return rc;
}
