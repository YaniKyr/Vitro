/* vitro_observer.c - libvitroobs.so, the per-container GPU observer.
 *
 * Preloaded (LD_PRELOAD) into the inference server. It sits between the server
 * and the CUDA driver (libcuda), counts what passes through, and publishes the
 * container's own figures to a small memory-mapped stats file:
 *
 *   SM limit      asked from the driver once; under MPS this is the slice
 *   memory        bytes counted in the allocate/free calls (model memory),
 *                 footprint = memory cap - free (includes the context overhead),
 *                 overhead  = footprint - model memory
 *   GPU time      every kernel launch is bracketed by two GPU-side events on its
 *                 own stream; the gap between them is the time the GPU spent on
 *                 that kernel. Summed per inference and per container.
 *   utilisation   busy time / wall time. Per container the busy periods of
 *                 overlapping inferences are merged (1 ms buckets), so it stays
 *                 at or below 100 %.
 *   SM-weighted   each kernel's time scaled by min(blocks, SMs) / SMs. An
 *                 ESTIMATE from the launch shape, not a measurement.
 *
 * It only observes. Limits are enforced by MPS, not here.
 *
 * How the interception works: the CUDA runtime, cuDNN, cuBLAS (and ctypes) reach
 * libcuda through dlopen + dlsym, so this library overrides dlsym and hands out
 * its own wrappers for the driver calls it cares about. Newer CUDA runtimes
 * (12 and later) fetch most driver functions through cuGetProcAddress instead;
 * that call is wrapped too and substitutes the same wrappers. A program that
 * links libcuda directly, using neither path, is not observed.
 *
 * Inferences may overlap (no serialisation). A kernel launch is attributed to
 * the inference that is open on the calling thread; launches from any other
 * thread are counted as "unattributed".
 *
 * Build:  gcc -O2 -fPIC -shared -o libvitroobs.so vitro_observer.c -lpthread -ldl
 * Use:    LD_PRELOAD=/opt/vitro/libvitroobs.so python3 server.py
 * Env:    VITRO_STATS_FILE       stats file (default /run/vitro/stats)
 *         VITRO_MEM_CAP_BYTES    memory cap; default: taken from
 *                                CUDA_MPS_PINNED_DEVICE_MEM_LIMIT ("0=2G")
 *         VITRO_DEBUG            log every driver call that gets hooked
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef void *CUstream, *CUevent, *CUcontext, *CUfunction;
#define CUDA_ERROR_NOT_READY 600

/* ---- stats file layout (mirrored by vitro_stats.py; every field is 8 bytes
 *      after the 16-byte header, so there is no padding) ---------------------- */
#define VITRO_MAGIC 0x53424f4f52544956ULL            /* "VITROOBS" */
#define VITRO_VERSION 1
struct vitro_stats {
    uint64_t magic;
    uint32_t version;
    uint32_t seq;                    /* odd while being written (seqlock) */
    uint64_t pid;
    uint64_t updated_ns;             /* CLOCK_REALTIME */
    uint64_t sm_count;               /* SMs this context may use */
    uint64_t mem_cap_bytes;          /* 0 = no cap known */
    uint64_t mem_model_bytes;        /* counted allocations, now */
    uint64_t mem_model_peak_bytes;
    uint64_t mem_footprint_bytes;    /* cap - free; 0 if no cap */
    uint64_t mem_overhead_bytes;     /* footprint - model */
    uint64_t mem_free_bytes;         /* as reported by the driver */
    uint64_t mem_total_bytes;
    uint64_t kernels;                /* launches measured */
    uint64_t kernels_unattributed;   /* ... of which outside any inference */
    uint64_t kernels_unobserved;     /* launches seen but not measured */
    uint64_t pending;                /* launches not resolved yet */
    uint64_t inferences;
    uint64_t inflight;
    double gpu_ms;                   /* sum of kernel times */
    double smw_ms;                   /* SM-weighted sum (estimate) */
    double busy_ms;                  /* merged busy time (lags by 2 s) */
    double util_1s, util_10s;        /* merged busy time, % of window */
    double smw_util_1s, smw_util_10s;
    double inf_gpu_ms, inf_wall_ms;  /* totals over finished inferences */
    double last_gpu_ms, last_wall_ms, last_smw_ms;
    uint64_t last_kernels;
};

/* ---- what the server calls around each inference --------------------------- */
struct vitro_inference {
    double gpu_ms;                   /* GPU time of this inference's kernels */
    double wall_ms;                  /* begin -> end */
    double smw_ms;                   /* SM-weighted GPU time (estimate) */
    uint64_t kernels;
    uint64_t unresolved;             /* kernels that could not be timed */
};
void vitro_inference_begin(void);
int vitro_inference_end(struct vitro_inference *out);

/* ---- driver functions used by the observer itself --------------------------- */
static struct {
    CUresult (*CtxGetCurrent)(CUcontext *);
    CUresult (*CtxSetCurrent)(CUcontext);
    CUresult (*CtxGetDevice)(int *);
    CUresult (*DeviceGetAttribute)(int *, int, int);
    CUresult (*MemGetInfo)(size_t *, size_t *);
    CUresult (*StreamCreate)(CUstream *, unsigned);
    CUresult (*EventCreate)(CUevent *, unsigned);
    CUresult (*EventRecord)(CUevent, CUstream);
    CUresult (*EventQuery)(CUevent);
    CUresult (*EventSynchronize)(CUevent);
    CUresult (*EventElapsedTime)(float *, CUevent, CUevent);
} cu;

static void *(*real_dlsym)(void *, const char *);

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

/* ---- one bracketed launch --------------------------------------------------- */
struct pair {
    struct pair *next;
    CUevent a, b;                    /* before / after the launch */
    uint32_t ref, ref_gen;           /* time reference used (see refs[]) */
    float weight;                    /* min(blocks, SMs) / SMs */
};

/* ---- an open inference (slot 0 collects launches outside any inference) ----- */
#define NSLOT 65
struct rec {
    pthread_mutex_t mu;
    struct pair *head, *tail;
    int active;
    double t0;
    double gpu_ms, smw_ms;
    uint64_t kernels, unresolved;
};
static struct rec slots[NSLOT];
static __thread struct rec *tl_rec;

/* ---- time references: map GPU event time onto the host clock ---------------- */
#define NREF 8
#define REF_PERIOD_MS 5000.0
static struct { CUevent ev; double clock_ms; uint32_t gen; } refs[NREF];
static uint32_t ref_cur;

/* ---- busy timeline: 1 ms buckets, ~16 s ------------------------------------- */
#define NB 16384
static float tl_busy[NB], tl_smw[NB];
static int64_t tl_epoch[NB];
static int64_t tl_final;             /* buckets before this are in busy_total */

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;      /* totals, timeline */
static pthread_mutex_t pool_mu = PTHREAD_MUTEX_INITIALIZER;   /* free pairs */
static pthread_mutex_t launch_mu = PTHREAD_MUTEX_INITIALIZER; /* event-launch-event */
static pthread_mutex_t init_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t mem_mu = PTHREAD_MUTEX_INITIALIZER;

static struct pair *pool;
static int g_ready, g_stop;
static CUcontext g_ctx;
static CUstream g_stream;            /* observer's own stream, for refs */
static pthread_t g_thread;
static struct vitro_stats *g_stats;  /* mmap'd */
static struct vitro_stats g_tot;     /* working copy, under g_mu */
static uint64_t g_mem_now, g_mem_peak, g_unobserved;

/* ---- memory accounting: device pointer -> size ------------------------------ */
#define NPTR 65536
static struct { CUdeviceptr p; size_t n; } ptrs[NPTR];       /* p == 1: tombstone */

static void mem_add(CUdeviceptr p, size_t n)
{
    if (!p) return;
    pthread_mutex_lock(&mem_mu);
    for (unsigned i = (unsigned)(p >> 8), k = 0; k < NPTR; ++k, ++i) {
        unsigned s = i % NPTR;
        if (ptrs[s].p == 0 || ptrs[s].p == 1) {
            ptrs[s].p = p; ptrs[s].n = n;
            g_mem_now += n;
            if (g_mem_now > g_mem_peak) g_mem_peak = g_mem_now;
            break;
        }
    }
    pthread_mutex_unlock(&mem_mu);
}

static void mem_del(CUdeviceptr p)
{
    if (!p) return;
    pthread_mutex_lock(&mem_mu);
    for (unsigned i = (unsigned)(p >> 8), k = 0; k < NPTR; ++k, ++i) {
        unsigned s = i % NPTR;
        if (ptrs[s].p == 0) break;
        if (ptrs[s].p == p) {
            g_mem_now -= ptrs[s].n;
            ptrs[s].p = 1; ptrs[s].n = 0;
            break;
        }
    }
    pthread_mutex_unlock(&mem_mu);
}

/* ---- timeline ---------------------------------------------------------------- */
static void tl_add(double start, double end, double weight)      /* under g_mu */
{
    double now = now_ms();
    if (end > now) end = now;
    if (start < now - (NB - 64)) start = now - (NB - 64);
    for (int64_t ms = (int64_t)floor(start); ms <= (int64_t)floor(end); ++ms) {
        double lo = start > ms ? start : ms, hi = end < ms + 1 ? end : ms + 1;
        if (hi <= lo) continue;
        unsigned i = (unsigned)(ms & (NB - 1));
        if (tl_epoch[i] != ms) { tl_epoch[i] = ms; tl_busy[i] = tl_smw[i] = 0; }
        tl_busy[i] += (float)(hi - lo);
        tl_smw[i] += (float)((hi - lo) * weight);
    }
}

static void tl_window(int64_t from, int64_t to, double *busy, double *smw) /* g_mu */
{
    *busy = *smw = 0;
    for (int64_t ms = from; ms < to; ++ms) {
        unsigned i = (unsigned)(ms & (NB - 1));
        if (tl_epoch[i] != ms) continue;
        *busy += tl_busy[i] > 1 ? 1 : tl_busy[i];
        *smw += tl_smw[i] > 1 ? 1 : tl_smw[i];
    }
}

/* ---- resolve finished launches of one record --------------------------------- */
static void drain(struct rec *r, int limit)
{
    pthread_mutex_lock(&r->mu);
    while (r->head && limit-- > 0) {
        struct pair *p = r->head;
        CUresult q = cu.EventQuery(p->b);
        if (q == CUDA_ERROR_NOT_READY) break;
        float dur = 0, off = 0;
        int ok = q == 0 && cu.EventElapsedTime(&dur, p->a, p->b) == 0 && dur >= 0;
        r->head = p->next;
        if (!r->head) r->tail = NULL;
        if (ok) {
            r->gpu_ms += dur; r->smw_ms += dur * p->weight; r->kernels++;
        } else {
            r->unresolved++;
        }
        pthread_mutex_lock(&g_mu);
        if (ok) {
            g_tot.kernels++;
            g_tot.gpu_ms += dur; g_tot.smw_ms += dur * p->weight;
            if (r == &slots[0]) g_tot.kernels_unattributed++;
            if (refs[p->ref].gen == p->ref_gen &&
                cu.EventElapsedTime(&off, refs[p->ref].ev, p->a) == 0) {
                double start = refs[p->ref].clock_ms + off;
                tl_add(start, start + dur, p->weight);
            }
        } else {
            g_unobserved++;
        }
        g_tot.pending--;
        pthread_mutex_unlock(&g_mu);
        pthread_mutex_lock(&pool_mu);
        p->next = pool; pool = p;
        pthread_mutex_unlock(&pool_mu);
    }
    pthread_mutex_unlock(&r->mu);
}

static void new_ref(void)
{
    uint32_t i = (ref_cur + 1) % NREF;
    if (!refs[i].ev && cu.EventCreate(&refs[i].ev, 0)) return;
    pthread_mutex_lock(&g_mu);
    refs[i].gen++;                                   /* invalidate old users */
    pthread_mutex_unlock(&g_mu);
    if (cu.EventRecord(refs[i].ev, g_stream) || cu.EventSynchronize(refs[i].ev)) return;
    refs[i].clock_ms = now_ms();
    ref_cur = i;
}

static void publish(void)
{
    size_t free_b = 0, total_b = 0;
    cu.MemGetInfo(&free_b, &total_b);
    double now = now_ms();
    int64_t edge = (int64_t)now - 30;                /* let recent launches resolve */

    pthread_mutex_lock(&g_mu);
    struct vitro_stats s = g_tot;
    double b, w;
    tl_window(edge - 1000, edge, &b, &w);
    s.util_1s = b / 10.0; s.smw_util_1s = w / 10.0;
    tl_window(edge - 10000, edge, &b, &w);
    s.util_10s = b / 100.0; s.smw_util_10s = w / 100.0;
    if (!tl_final) tl_final = (int64_t)now - 2000;
    tl_window(tl_final, (int64_t)now - 2000, &b, &w);
    tl_final = (int64_t)now - 2000;
    g_tot.busy_ms += b;
    s.busy_ms = g_tot.busy_ms;
    s.kernels_unobserved = g_unobserved;
    pthread_mutex_unlock(&g_mu);

    pthread_mutex_lock(&mem_mu);
    s.mem_model_bytes = g_mem_now; s.mem_model_peak_bytes = g_mem_peak;
    pthread_mutex_unlock(&mem_mu);
    s.mem_free_bytes = free_b; s.mem_total_bytes = total_b;
    if (s.mem_cap_bytes) {
        s.mem_footprint_bytes = s.mem_cap_bytes > free_b ? s.mem_cap_bytes - free_b : 0;
        s.mem_overhead_bytes = s.mem_footprint_bytes > s.mem_model_bytes
                             ? s.mem_footprint_bytes - s.mem_model_bytes : 0;
    }
    s.inflight = 0;
    for (int i = 1; i < NSLOT; ++i) s.inflight += slots[i].active != 0;

    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    s.updated_ns = (uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec;
    if (!g_stats) return;
    uint32_t seq = g_stats->seq;
    __atomic_store_n(&g_stats->seq, seq + 1, __ATOMIC_SEQ_CST);
    s.seq = seq + 1;
    memcpy(g_stats, &s, sizeof s);
    __atomic_store_n(&g_stats->seq, seq + 2, __ATOMIC_SEQ_CST);
}

static void *worker(void *arg)
{
    (void)arg;
    sigset_t all;
    sigfillset(&all);
    pthread_sigmask(SIG_BLOCK, &all, NULL);
    if (cu.CtxSetCurrent(g_ctx)) return NULL;
    double last_pub = 0, last_ref = now_ms();
    while (!g_stop) {
        for (int i = 0; i < NSLOT; ++i)
            if (i == 0 || slots[i].active) drain(&slots[i], 512);
        double now = now_ms();
        if (now - last_ref > REF_PERIOD_MS) { new_ref(); last_ref = now; }
        if (now - last_pub > 100) { publish(); last_pub = now; }
        usleep(20000);
    }
    return NULL;
}

static void stop_worker(void)
{
    if (!g_ready) return;
    g_stop = 1;
    pthread_join(g_thread, NULL);
}

static uint64_t parse_size(const char *v)                  /* "2G", "512M", "1024K" or bytes */
{
    char *end;
    double n = strtod(v, &end);
    if (*end == 'G' || *end == 'g') n *= 1024.0 * 1024 * 1024;
    else if (*end == 'M' || *end == 'm') n *= 1024.0 * 1024;
    else if (*end == 'K' || *end == 'k') n *= 1024.0;
    return n > 0 ? (uint64_t)n : 0;
}

/* The memory limit of this client. The driver does not report it: cuMemGetInfo
 * gives the free memory under the limit, but the total of the whole GPU. With a
 * limit set in the environment it is read from there; with a limit that belongs
 * to an MPS v3 namespace whoever starts the container passes it in VITRO_MEM_CAP. */
static uint64_t mem_cap_from_env(void)
{
    const char *v = getenv("VITRO_MEM_CAP");
    if (v && *v) return parse_size(v);
    v = getenv("VITRO_MEM_CAP_BYTES");
    if (v && *v) return strtoull(v, NULL, 10);
    v = getenv("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT");          /* "0=2G" or "0=512M" */
    if (!v || !(v = strchr(v, '='))) return 0;
    return parse_size(v + 1);
}

static void open_stats(void)
{
    const char *path = getenv("VITRO_STATS_FILE");
    if (!path || !*path) path = "/run/vitro/stats";
    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0 || ftruncate(fd, sizeof(struct vitro_stats))) {
        fprintf(stderr, "[vitro] stats file %s not writable; metrics stay in-process\n", path);
        if (fd >= 0) close(fd);
        return;
    }
    void *m = mmap(NULL, sizeof(struct vitro_stats), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (m != MAP_FAILED) { g_stats = m; memset(g_stats, 0, sizeof *g_stats); }
}

/* First driver call with a context: set everything up. Returns 1 if the calling
 * thread's context is the observed one. */
static int ensure_ctx(void)
{
    CUcontext c = NULL;
    if (g_ready) return cu.CtxGetCurrent(&c) == 0 && c == g_ctx;
    pthread_mutex_lock(&init_mu);
    if (!g_ready) {
        void *h = dlopen("libcuda.so.1", RTLD_LAZY);
#define R(f) (cu.f = real_dlsym(h, "cu" #f))
        if (h && R(CtxGetCurrent) && R(CtxSetCurrent) && R(CtxGetDevice) && R(DeviceGetAttribute) &&
            (cu.MemGetInfo = real_dlsym(h, "cuMemGetInfo_v2")) && R(StreamCreate) && R(EventCreate) &&
            R(EventRecord) && R(EventQuery) && R(EventSynchronize) && R(EventElapsedTime) &&
            cu.CtxGetCurrent(&c) == 0 && c) {
#undef R
            int dev = 0, sms = 0;
            cu.CtxGetDevice(&dev);
            cu.DeviceGetAttribute(&sms, 16, dev);    /* 16 = MULTIPROCESSOR_COUNT */
            for (int i = 0; i < NSLOT; ++i) pthread_mutex_init(&slots[i].mu, NULL);
            g_ctx = c;
            g_tot.magic = VITRO_MAGIC; g_tot.version = VITRO_VERSION;
            g_tot.pid = (uint64_t)getpid();
            g_tot.sm_count = (uint64_t)sms;
            g_tot.mem_cap_bytes = mem_cap_from_env();
            if (cu.StreamCreate(&g_stream, 1) == 0) {            /* 1 = non-blocking */
                ref_cur = NREF - 1;
                new_ref();
                open_stats();
                if (pthread_create(&g_thread, NULL, worker, NULL) == 0) {
                    atexit(stop_worker);
                    g_ready = 1;
                    fprintf(stderr, "[vitro] observer active: %d SMs, cap %llu MiB\n", sms,
                            (unsigned long long)(g_tot.mem_cap_bytes >> 20));
                }
            }
        }
    }
    pthread_mutex_unlock(&init_mu);
    return g_ready && cu.CtxGetCurrent(&c) == 0 && c == g_ctx;
}

/* ---- bracket one launch: before() ... real launch ... after() --------------- */
static struct pair *before(CUstream stream, uint64_t blocks)
{
    if (!ensure_ctx()) { __atomic_add_fetch(&g_unobserved, 1, __ATOMIC_RELAXED); return NULL; }
    pthread_mutex_lock(&pool_mu);
    struct pair *p = pool;
    if (p) pool = p->next;
    pthread_mutex_unlock(&pool_mu);
    if (!p) {
        p = calloc(1, sizeof *p);
        if (!p || cu.EventCreate(&p->a, 0) || cu.EventCreate(&p->b, 0)) {
            free(p);                                  /* events leak only on this error path */
            __atomic_add_fetch(&g_unobserved, 1, __ATOMIC_RELAXED);
            return NULL;
        }
    }
    uint64_t sms = g_tot.sm_count ? g_tot.sm_count : 1;
    p->weight = blocks && blocks < sms ? (float)blocks / sms : 1.0f;
    p->next = NULL;
    /* Held until after(): keeps event-launch-event together on the stream even
     * when several inferences launch at once. */
    pthread_mutex_lock(&launch_mu);
    p->ref = ref_cur; p->ref_gen = refs[p->ref].gen;
    cu.EventRecord(p->a, stream);
    return p;
}

static void after(struct pair *p, CUstream stream, CUresult rc)
{
    if (!p) return;
    if (rc || cu.EventRecord(p->b, stream)) {
        pthread_mutex_unlock(&launch_mu);
        pthread_mutex_lock(&pool_mu);
        p->next = pool; pool = p;
        pthread_mutex_unlock(&pool_mu);
        return;
    }
    struct rec *r = tl_rec ? tl_rec : &slots[0];
    pthread_mutex_lock(&r->mu);
    if (r->tail) r->tail->next = p; else r->head = p;
    r->tail = p;
    pthread_mutex_unlock(&r->mu);
    pthread_mutex_lock(&g_mu);
    g_tot.pending++;
    pthread_mutex_unlock(&g_mu);
    pthread_mutex_unlock(&launch_mu);
}

/* ---- wrappers ---------------------------------------------------------------- */
typedef struct {
    unsigned gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes;
    CUstream hStream;
    void *attrs;
    unsigned numAttrs;
} CUlaunchConfig;

static CUresult (*r_LaunchKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned,
                                  unsigned, unsigned, CUstream, void **, void **);
static CUresult (*r_LaunchKernelEx)(const CUlaunchConfig *, CUfunction, void **, void **);
static CUresult (*r_LaunchCooperativeKernel)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                                             unsigned, unsigned, unsigned, CUstream, void **);
static CUresult (*r_GraphLaunch)(void *, CUstream);
static CUresult (*r_MemAlloc)(CUdeviceptr *, size_t);
static CUresult (*r_MemAllocManaged)(CUdeviceptr *, size_t, unsigned);
static CUresult (*r_MemAllocPitch)(CUdeviceptr *, size_t *, size_t, size_t, unsigned);
static CUresult (*r_MemAllocAsync)(CUdeviceptr *, size_t, CUstream);
static CUresult (*r_MemAllocFromPoolAsync)(CUdeviceptr *, size_t, void *, CUstream);
static CUresult (*r_MemFree)(CUdeviceptr);
static CUresult (*r_MemFreeAsync)(CUdeviceptr, CUstream);

static CUresult h_LaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx,
                               unsigned by, unsigned bz, unsigned shm, CUstream s, void **p, void **e)
{
    struct pair *pr = before(s, (uint64_t)gx * gy * gz);
    CUresult rc = r_LaunchKernel(f, gx, gy, gz, bx, by, bz, shm, s, p, e);
    after(pr, s, rc);
    return rc;
}

static CUresult h_LaunchKernelEx(const CUlaunchConfig *c, CUfunction f, void **p, void **e)
{
    if (!c) return r_LaunchKernelEx(c, f, p, e);
    struct pair *pr = before(c->hStream, (uint64_t)c->gridDimX * c->gridDimY * c->gridDimZ);
    CUresult rc = r_LaunchKernelEx(c, f, p, e);
    after(pr, c->hStream, rc);
    return rc;
}

static CUresult h_LaunchCooperativeKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
                                          unsigned bx, unsigned by, unsigned bz, unsigned shm,
                                          CUstream s, void **p)
{
    struct pair *pr = before(s, (uint64_t)gx * gy * gz);
    CUresult rc = r_LaunchCooperativeKernel(f, gx, gy, gz, bx, by, bz, shm, s, p);
    after(pr, s, rc);
    return rc;
}

static CUresult h_GraphLaunch(void *g, CUstream s)
{
    struct pair *pr = before(s, 0);                  /* shape unknown: weight 1 */
    CUresult rc = r_GraphLaunch(g, s);
    after(pr, s, rc);
    return rc;
}

static CUresult h_MemAlloc(CUdeviceptr *d, size_t n)
{
    CUresult rc = r_MemAlloc(d, n);
    if (!rc && d) { mem_add(*d, n); ensure_ctx(); }
    return rc;
}

static CUresult h_MemAllocManaged(CUdeviceptr *d, size_t n, unsigned flags)
{
    CUresult rc = r_MemAllocManaged(d, n, flags);
    if (!rc && d) { mem_add(*d, n); ensure_ctx(); }
    return rc;
}

static CUresult h_MemAllocPitch(CUdeviceptr *d, size_t *pitch, size_t w, size_t h, unsigned el)
{
    CUresult rc = r_MemAllocPitch(d, pitch, w, h, el);
    if (!rc && d && pitch) { mem_add(*d, *pitch * h); ensure_ctx(); }
    return rc;
}

static CUresult h_MemAllocAsync(CUdeviceptr *d, size_t n, CUstream s)
{
    CUresult rc = r_MemAllocAsync(d, n, s);
    if (!rc && d) { mem_add(*d, n); ensure_ctx(); }
    return rc;
}

static CUresult h_MemAllocFromPoolAsync(CUdeviceptr *d, size_t n, void *pool_, CUstream s)
{
    CUresult rc = r_MemAllocFromPoolAsync(d, n, pool_, s);
    if (!rc && d) { mem_add(*d, n); ensure_ctx(); }
    return rc;
}

static CUresult h_MemFree(CUdeviceptr d)
{
    CUresult rc = r_MemFree(d);
    if (!rc) mem_del(d);
    return rc;
}

static CUresult h_MemFreeAsync(CUdeviceptr d, CUstream s)
{
    CUresult rc = r_MemFreeAsync(d, s);
    if (!rc) mem_del(d);
    return rc;
}

static const struct { const char *name; void *wrapper; void **real; } hooks[] = {
    {"cuLaunchKernel", h_LaunchKernel, (void **)&r_LaunchKernel},
    {"cuLaunchKernelEx", h_LaunchKernelEx, (void **)&r_LaunchKernelEx},
    {"cuLaunchCooperativeKernel", h_LaunchCooperativeKernel, (void **)&r_LaunchCooperativeKernel},
    {"cuGraphLaunch", h_GraphLaunch, (void **)&r_GraphLaunch},
    {"cuMemAlloc_v2", h_MemAlloc, (void **)&r_MemAlloc},
    {"cuMemAllocManaged", h_MemAllocManaged, (void **)&r_MemAllocManaged},
    {"cuMemAllocPitch_v2", h_MemAllocPitch, (void **)&r_MemAllocPitch},
    {"cuMemAllocAsync", h_MemAllocAsync, (void **)&r_MemAllocAsync},
    {"cuMemAllocFromPoolAsync", h_MemAllocFromPoolAsync, (void **)&r_MemAllocFromPoolAsync},
    {"cuMemFree_v2", h_MemFree, (void **)&r_MemFree},
    {"cuMemFreeAsync", h_MemFreeAsync, (void **)&r_MemFreeAsync},
};

static int g_debug = -1;

/* Wrapper for `name`, remembering the real function; NULL if it is not hooked.
 * cuGetProcAddress may be asked for a name without its _v2 suffix. */
static void *hook_for(const char *name, void *real, int allow_unversioned)
{
    if (g_debug < 0) g_debug = getenv("VITRO_DEBUG") != NULL;
    for (unsigned i = 0; i < sizeof hooks / sizeof *hooks; ++i) {
        size_t n = strlen(name);
        if (!strcmp(name, hooks[i].name) ||
            (allow_unversioned && !strncmp(name, hooks[i].name, n) && !strcmp(hooks[i].name + n, "_v2"))) {
            if (g_debug) fprintf(stderr, "[vitro] hooked %s\n", name);
            *hooks[i].real = real;
            return hooks[i].wrapper;
        }
    }
    return NULL;
}

static CUresult (*r_GetProcAddress)(const char *, void **, int, uint64_t);
static CUresult (*r_GetProcAddress_v2)(const char *, void **, int, uint64_t, void *);
static CUresult h_GetProcAddress(const char *, void **, int, uint64_t);
static CUresult h_GetProcAddress_v2(const char *, void **, int, uint64_t, void *);

/* Replace a looked-up driver function by its wrapper. The runtime also asks
 * cuGetProcAddress for cuGetProcAddress itself, so that is wrapped as well. */
static void *substitute(const char *name, void *real, int allow_unversioned)
{
    if (!strcmp(name, "cuGetProcAddress")) { r_GetProcAddress = real; return h_GetProcAddress; }
    if (!strcmp(name, "cuGetProcAddress_v2")) { r_GetProcAddress_v2 = real; return h_GetProcAddress_v2; }
    void *w = hook_for(name, real, allow_unversioned);
    return w ? w : real;
}

static CUresult h_GetProcAddress(const char *name, void **fn, int ver, uint64_t flags)
{
    CUresult rc = r_GetProcAddress(name, fn, ver, flags);
    if (!rc && name && fn && *fn) *fn = substitute(name, *fn, 1);
    return rc;
}

static CUresult h_GetProcAddress_v2(const char *name, void **fn, int ver, uint64_t flags, void *st)
{
    CUresult rc = r_GetProcAddress_v2(name, fn, ver, flags, st);
    if (!rc && name && fn && *fn) *fn = substitute(name, *fn, 1);
    return rc;
}

/* The override. Everything that is not a hooked driver call passes straight
 * through. A caller that asks for RTLD_NEXT gets the search order after this
 * library instead of after itself; with a preloaded library that is the same
 * set of objects in practice. */
void *dlsym(void *handle, const char *name)
{
    if (!real_dlsym) {
        static const char *vers[] = {"GLIBC_2.34", "GLIBC_2.2.5", "GLIBC_2.17"};
        for (unsigned i = 0; i < 3 && !real_dlsym; ++i)
            real_dlsym = dlvsym(RTLD_NEXT, "dlsym", vers[i]);
        if (!real_dlsym) { fprintf(stderr, "[vitro] cannot find the real dlsym\n"); abort(); }
    }
    void *sym = real_dlsym(handle, name);
    if (!sym || name[0] != 'c' || name[1] != 'u') return sym;
    return substitute(name, sym, 0);
}

/* ---- inference boundaries ---------------------------------------------------- */
void vitro_inference_begin(void)
{
    tl_rec = NULL;
    if (!g_ready) return;
    pthread_mutex_lock(&g_mu);
    for (int i = 1; i < NSLOT; ++i)
        if (!slots[i].active) { slots[i].active = 1; tl_rec = &slots[i]; break; }
    pthread_mutex_unlock(&g_mu);
    if (!tl_rec) return;                             /* more than 64 at once: unattributed */
    tl_rec->gpu_ms = tl_rec->smw_ms = 0;
    tl_rec->kernels = tl_rec->unresolved = 0;
    tl_rec->t0 = now_ms();
}

int vitro_inference_end(struct vitro_inference *out)
{
    struct rec *r = tl_rec;
    if (out) memset(out, 0, sizeof *out);
    if (!r) return -1;
    double wall = now_ms() - r->t0;
    tl_rec = NULL;
    for (int tries = 0; tries < 1000; ++tries) {     /* wait for this inference's kernels */
        drain(r, 1 << 30);
        pthread_mutex_lock(&r->mu);
        CUevent last = r->tail ? r->tail->b : NULL;
        pthread_mutex_unlock(&r->mu);
        if (!last) break;
        cu.EventSynchronize(last);
    }
    struct vitro_inference m = {r->gpu_ms, wall, r->smw_ms, r->kernels, r->unresolved};
    pthread_mutex_lock(&g_mu);
    g_tot.inferences++;
    g_tot.inf_gpu_ms += m.gpu_ms; g_tot.inf_wall_ms += m.wall_ms;
    g_tot.last_gpu_ms = m.gpu_ms; g_tot.last_wall_ms = m.wall_ms;
    g_tot.last_smw_ms = m.smw_ms; g_tot.last_kernels = m.kernels;
    r->active = 0;
    pthread_mutex_unlock(&g_mu);
    if (out) *out = m;
    return 0;
}
