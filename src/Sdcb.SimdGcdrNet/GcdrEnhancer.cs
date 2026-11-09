using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using Sdcb.SimdGcdrNet.Kernels;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Imaging;
using Sdcb.SimdOnnx.Kernels;
using Sdcb.SimdOnnx.Onnx;
using VkBackend = Sdcb.SimdOnnx.Backends.Vulkan.VkBackend;
using VkDeviceLostException = Sdcb.SimdOnnx.Backends.Vulkan.VkDeviceLostException;
using MetalBackend = Sdcb.SimdOnnx.Backends.Metal.MetalBackend;

namespace Sdcb.SimdGcdrNet;

/// <summary>Compute backend selection for <see cref="GcdrEnhancer"/>.</summary>
public enum GcdrBackend
{
    /// <summary>CPU: Scalar → Vector&lt;float&gt; → AVX2 → AVX-512 dispatch on x64, NEON on arm64,
    /// PackedSimd (WASM SIMD128) on browser-wasm.</summary>
    Cpu,
    /// <summary>Vulkan compute (discrete GPU preferred; integrated and mobile GPUs otherwise).</summary>
    Vulkan,
    /// <summary>GPU (Vulkan or Metal) if a compute-capable device exists, else CPU.</summary>
    Auto,
    /// <summary>Metal compute on Apple Silicon (macOS arm64).</summary>
    Metal,
}

/// <summary>
/// GCDRNet two-stage document enhancer: GC-Net shading estimation → division
/// correction → DR-Net detail restoration. Interleaved uint8 in (BGR24 / RGB24 /
/// BGRA32 / RGBA32, see <see cref="GcdrPixelFormat"/>), same format out.
/// Thread-safe: size-agnostic workers (CPU arenas / Vulkan pipelines) are pooled
/// and rented per call; per-size plans are cached and share packed weights.
/// </summary>
public sealed class GcdrEnhancer : IDisposable
{
    private readonly OnnxModel _gcModel, _drModel;
    private readonly GcdrBackend _backend;

    // CPU: per-size plans are managed metadata (cached, at most MaxSizes; packed
    // weights shared across sizes through the kernel caches). Workers own grow-only
    // native arenas (CpuArenaBudget, see CpuWorker) and serve every size; idle
    // workers beyond the first are kept only while their arenas fit the budget.
    private const int MaxSizes = 8, MaxPerSize = 4;
    private const long CpuArenaBudget = 4L << 30;
    private readonly ConcurrentDictionary<long, (ExecPlan gc, ExecPlan dr)> _cpuPlans = new();
    private readonly CpuKernels.Cache _gcKernels = new(), _drKernels = new();
    // Cache Metal availability for the process: re-probing (incl. the MSL library
    // compile) per Enhance would re-throw on init failure for every frame.
    private static readonly Lazy<bool> s_metalOk = new(MetalOk);

    /// <summary>Whether the Metal backend can run here (what <see cref="GcdrBackend.Auto"/> checks).</summary>
    internal static bool MetalUsable => GpuPossible && s_metalOk.Value;
    private SizePool<CpuWorker>? _cpuPool;
    // Vulkan workers are size-agnostic (budgeted grow-only arenas, re-record on size
    // change): one pool serves every size; idle workers beyond the first are kept
    // only while their device memory fits the budget
    // (3 GB; integrated GPUs: at most a quarter of the shared device-local heap)
    private static long VkIdleBudget => (long)Backends.Vulkan.VkGcdrPipeline.BudgetFor(VkBackend.Default.Dev);
    private SizePool<Backends.Vulkan.VkGcdrPipeline>? _vkPool;
    private SizePool<Backends.Metal.MtlGcdrPipeline>? _mtlPool;
    private long _poolsCreated, _workersCreated, _poolsEvicted;

    /// <summary>Diagnostics: CPU size plans compiled (Vulkan: worker pools created), worker
    /// instances created, CPU size plans evicted.</summary>
    public (long pools, long workers, long evicted) PoolStats =>
        (_poolsCreated, _workersCreated, _poolsEvicted);

    /// <summary>Debug/test hook: invoked with the just-used vulkan pipeline after each Enhance.</summary>
    internal Action<Backends.Vulkan.VkGcdrPipeline>? DebugAfterEnhance;

    private GcdrEnhancer(OnnxModel gc, OnnxModel dr, GcdrBackend backend)
    {
        _gcModel = gc;
        _drModel = dr;
        _backend = backend;
    }

    /// <summary>Loads the GC-Net and DR-Net ONNX models from files (e.g.
    /// <c>gcdrnet_gcnet_sim.onnx</c> / <c>gcdrnet_drnet_sim.onnx</c>).</summary>
    public static GcdrEnhancer FromFiles(string gcnetOnnxPath, string drnetOnnxPath,
        GcdrBackend backend = GcdrBackend.Cpu) =>
        new(OnnxModel.Load(gcnetOnnxPath), OnnxModel.Load(drnetOnnxPath), backend);

    /// <summary>Parses both ONNX models directly from memory (no copy of the model bytes), e.g.
    /// <c>FromBytes(GcdrModels.GcNet, GcdrModels.DrNet)</c> with the Sdcb.SimdGcdrNet.Models package.
    /// The memory is only read during this call.</summary>
    public static GcdrEnhancer FromBytes(ReadOnlySpan<byte> gcnetOnnx, ReadOnlySpan<byte> drnetOnnx,
        GcdrBackend backend = GcdrBackend.Cpu) =>
        new(OnnxModel.Load(gcnetOnnx), OnnxModel.Load(drnetOnnx), backend);

    /// <summary>Loads both ONNX models from streams (read to the end, not disposed).</summary>
    public static GcdrEnhancer FromStreams(Stream gcnetOnnx, Stream drnetOnnx,
        GcdrBackend backend = GcdrBackend.Cpu)
    {
        ArgumentNullException.ThrowIfNull(gcnetOnnx);
        ArgumentNullException.ThrowIfNull(drnetOnnx);
        return new(OnnxModel.Load(gcnetOnnx), OnnxModel.Load(drnetOnnx), backend);
    }

    // pipeline sessions keep I/O in plan layout: the glue kernels below read/write
    // channels-last directly, so the plan has no boundary relayouts
    // the network sees B,G,R in [0, 1]
    private static readonly PixelNormalization s_unit = PixelNormalization.Unit();
    private static readonly CompileOptions CpuPlan = new(CompileOptions.DefaultNhwc, PhysicalIo: true);
    private static bool CpuNhwc => CpuPlan.Nhwc;

    private static long Key(int h, int w) => (long)h * 65536 + w;

    /// <summary>gc/dr plans for one padded size (compiled once, weights packed via the shared caches).</summary>
    private (ExecPlan gc, ExecPlan dr) CpuPlans(int h, int w)
    {
        long key = Key(h, w);
        if (_cpuPlans.TryGetValue(key, out var p)) return p;
        // the two nets compile independently: build them side by side
        var tgc = Task.Run(() =>
        {
            ExecPlan g = GraphCompiler.Compile(_gcModel, [1, 3, h, w], CpuPlan);
            CpuKernels.Pack(g, _gcKernels);
            return g;
        });
        ExecPlan dr = GraphCompiler.Compile(_drModel, [1, 6, h, w], CpuPlan);
        CpuKernels.Pack(dr, _drKernels);
        ExecPlan gc = tgc.GetAwaiter().GetResult();
        p = (gc, dr);
        if (!_cpuPlans.TryAdd(key, p)) return _cpuPlans.TryGetValue(key, out var won) ? won : p;
        Interlocked.Increment(ref _poolsCreated);
        // bound the cache (plans are managed; in-flight users keep theirs alive)
        while (_cpuPlans.Count > MaxSizes)
        {
            long victim = -1;
            foreach (long k in _cpuPlans.Keys) if (k != key) { victim = k; break; }
            if (victim < 0 || !_cpuPlans.TryRemove(victim, out _)) break;
            Interlocked.Increment(ref _poolsEvicted);
        }
        return p;
    }

    private SizePool<CpuWorker> CpuPool()
    {
        if (_cpuPool is { } p) return p;
        p = new SizePool<CpuWorker>(MaxPerSize,
            () => { Interlocked.Increment(ref _workersCreated); return new CpuWorker(); },
            w => w.Bytes, CpuArenaBudget);
        if (Interlocked.CompareExchange(ref _cpuPool, p, null) is { } won) { p.Dispose(); return won; }
        return p;
    }

    private SizePool<Backends.Vulkan.VkGcdrPipeline> VkPool()
    {
        if (_vkPool is { } p) return p;
        p = new SizePool<Backends.Vulkan.VkGcdrPipeline>(MaxPerSize,
            () => { Interlocked.Increment(ref _workersCreated); return new Backends.Vulkan.VkGcdrPipeline(VkBackend.Default, _gcModel, _drModel); },
            w => w.DeviceBytes, VkIdleBudget);
        if (Interlocked.CompareExchange(ref _vkPool, p, null) is { } won) { p.Dispose(); return won; }
        Interlocked.Increment(ref _poolsCreated);
        return p;
    }

    private SizePool<Backends.Metal.MtlGcdrPipeline> MtlPool()
    {
        if (_mtlPool is { } p) return p;
        p = new SizePool<Backends.Metal.MtlGcdrPipeline>(MaxPerSize,
            () => { Interlocked.Increment(ref _workersCreated); return new Backends.Metal.MtlGcdrPipeline(MetalBackend.Default, _gcModel, _drModel); },
            w => w.DeviceBytes, (long)Backends.Metal.MtlGcdrPipeline.ArenaBudget);
        if (Interlocked.CompareExchange(ref _mtlPool, p, null) is { } won) { p.Dispose(); return won; }
        Interlocked.Increment(ref _poolsCreated);
        return p;
    }


    /// <summary>
    /// Enhances an interleaved uint8 HWC image: reads <paramref name="src"/>
    /// (height rows of width pixels in <paramref name="format"/>, row stride
    /// <paramref name="srcStride"/> bytes) and writes <paramref name="dst"/> in the same
    /// format and dimensions (row stride <paramref name="dstStride"/> bytes). 4-byte formats
    /// copy alpha through unchanged. <paramref name="dst"/> may be the same memory as
    /// <paramref name="src"/> when the strides are equal.
    /// </summary>
    public void Enhance(ReadOnlySpan<byte> src, int width, int height, int srcStride,
        Span<byte> dst, int dstStride, GcdrPixelFormat format = GcdrPixelFormat.Bgr24)
    {
        int bpp = format.BytesPerPixel();
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(srcStride, width * bpp);
        ArgumentOutOfRangeException.ThrowIfLessThan(dstStride, width * bpp);
        if (src.Length < (long)(height - 1) * srcStride + width * bpp)
            throw new ArgumentException($"src holds {src.Length} bytes, {height} rows of stride {srcStride} need more.", nameof(src));
        if (dst.Length < (long)(height - 1) * dstStride + width * bpp)
            throw new ArgumentException($"dst holds {dst.Length} bytes, {height} rows of stride {dstStride} need more.", nameof(dst));

        bool useMtl = GpuPossible && (_backend == GcdrBackend.Metal
            || (_backend == GcdrBackend.Auto && s_metalOk.Value));
        if (useMtl)
        {
            var pool = MtlPool();
            var pipe = pool.Rent();
            try { pipe.Enhance(src, height, width, srcStride, dst, dstStride, format); }
            finally { pool.Return(pipe); }
            return;
        }
        bool useVk = GpuPossible && (_backend == GcdrBackend.Vulkan
            || (_backend == GcdrBackend.Auto && VulkanOk()));
        if (useVk)
        {
            for (int attempt = 0; ; attempt++)
            {
                var pool = VkPool();
                var pipe = pool.Rent();
                try
                {
                    pipe.Enhance(src, height, width, srcStride, dst, dstStride, format);
                    DebugAfterEnhance?.Invoke(pipe);
                    return;
                }
                catch (VkDeviceLostException) when (attempt == 0)
                {
                    // the backend re-creates its device on next use; this pool's pipelines are dead
                    Interlocked.CompareExchange(ref _vkPool, null, pool);
                }
                finally { pool.Return(pipe); }
            }
        }
        int ph = (-height) & 31, pw = (-width) & 31;
        int h = height + ph, w = width + pw;

        var (gcPlan, drPlan) = CpuPlans(h, w);
        var cpuPool = CpuPool();
        CpuWorker worker = cpuPool.Rent();
        try
        {
            var (gcOff, total) = StackArenas(gcPlan, drPlan);
            nint arena = worker.Arena(total * sizeof(float));
            using var gc = new InferenceSession(gcPlan, arena + (nint)(gcOff * sizeof(float)));
            using var dr = new InferenceSession(drPlan, arena);

            // stage 0: fill GC input — BGR u8 → float /255, top/left replicate pad
            Span<float> x = gc.Input;
            TensorLayout layout = CpuNhwc ? TensorLayout.Nhwc : TensorLayout.Nchw;
            ImageTensor.FromPixels(src, width, height, srcStride, (PixelFormat)format, x, layout, s_unit,
                w, h, padLeft: pw, padTop: ph, threads: Avx2Kernels.Threads);

            // stage 1: GC-Net → shading map
            gc.Run();
            ReadOnlySpan<float> shadow = gc.Output;
            int[] oshape = gc.OutputShape;
            if (oshape[2] != h || oshape[3] != w)
                throw new NotSupportedException($"GC-Net returned {oshape[2]}x{oshape[3]}, expected {h}x{w}.");

            // stage 2: corrected = clamp(x / max(shadow,1e-6), 0,1); concat into DR input
            Span<float> x6 = dr.Input;
            int hw = h * w;
            if (CpuNhwc) PixelOps.DivConcatNhwc(x, shadow, x6);
            else
            {
                x.CopyTo(x6.Slice(0, 3 * hw));
                PixelOps.DivClampInto(x, shadow, x6.Slice(3 * hw, 3 * hw));
            }

            // stage 3: DR-Net
            dr.Run();

            // stage 4: crop pad → u8 in the caller's format (alpha from src)
            ImageTensor.ToPixels(dr.Output, layout, w, h, dst, width, height, dstStride, (PixelFormat)format,
                cropLeft: pw, cropTop: ph, alphaSource: bpp == 4 ? src : default, alphaStride: srcStride,
                threads: Avx2Kernels.Threads);
        }
        finally { cpuPool.Return(worker); }
    }

    /// <summary>
    /// One arena for both stages (gc runs to completion before dr starts): the gc arena is
    /// stacked right past the dr input window, so gc's input/output survive the
    /// stage-2 glue that writes the dr input, and dr then reuses everything above.
    /// Returns the gc arena offset and the total size, in floats.
    /// </summary>
    internal static (long gcOff, long total) StackArenas(ExecPlan gc, ExecPlan dr, long align = 16) =>
        StackArenas(gc.ArenaFloats, dr, dr.ArenaFloats, align);

    /// <summary><see cref="StackArenas(ExecPlan, ExecPlan, long)"/> over each stage's full arena
    /// (GPU graphs append scratch / tile-read slack past the plan's values).</summary>
    internal static (long gcOff, long total) StackArenas(long gcArena, ExecPlan dr, long drArena, long align = 16)
    {
        CValue din = dr.Values[dr.InputValue];
        long o = dr.BufOffset[din.Buf], end = (o + din.ElementCount + align - 1) / align * align;
        long gcOff = o >= gcArena ? 0 : end;
        return (gcOff, Math.Max(drArena, gcOff + gcArena));
    }

    // browser/WASI: unlinked P/Invokes (vulkan-1, libobjc) trap the wasm module with
    // "null function" instead of throwing, so every backend (Auto/Vulkan/Metal) resolves to CPU
    private static bool GpuPossible => !OperatingSystem.IsBrowser() && !OperatingSystem.IsWasi();

    private static bool VulkanOk()
    {
        try { _ = VkBackend.Default.Dev.DeviceName; return true; }
        catch { return false; }
    }

    private static bool MetalOk()
    {
        try
        {
            var dev = MetalBackend.Default.Dev;
            // Apple7+ (M1+) for simdgroup matrix ops; Pipe() forces the embedded
            // MSL library compile now rather than mid-Enhance on macOS 13. Intel Macs can
            // claim Apple7 yet build the simdgroup-MMA conv with 0 threads/group (Pipe throws).
            return dev.SupportsApple7 && dev.Pipe("bgr2nhwc", Backends.GcdrShaders.Owner) is not null && dev.Pipe("conv_mm_32_0") is not null;
        }
        catch { return false; }
    }

    /// <summary>Convenience: enhance a tightly packed image in <paramref name="format"/>, returns a new array in the same format.</summary>
    public byte[] Enhance(byte[] src, int width, int height, GcdrPixelFormat format = GcdrPixelFormat.Bgr24)
    {
        int stride = width * format.BytesPerPixel();
        byte[] dst = new byte[stride * height];
        Enhance(src, width, height, stride, dst, stride, format);
        return dst;
    }

    /// <summary>Releases all pooled CPU sessions and GPU workers.</summary>
    public void Dispose()
    {
        Interlocked.Exchange(ref _cpuPool, null)?.Dispose();
        Interlocked.Exchange(ref _vkPool, null)?.Dispose();
        Interlocked.Exchange(ref _mtlPool, null)?.Dispose();
        _cpuPlans.Clear();
    }

    /// <summary>
    /// Size-agnostic CPU worker: one native arena holding both stages (see
    /// <see cref="StackArenas(ExecPlan, ExecPlan, long)"/>), borrowed by per-call sessions. Within
    /// <see cref="CpuArenaBudget"/> it only grows (25% headroom on regrowth), so mixed sizes
    /// neither reallocate nor re-fault pages; an over-budget arena (one huge image) drops
    /// to exact size once a size that fits comes along.
    /// </summary>
    private sealed unsafe class CpuWorker : IDisposable
    {
        private nint _p;
        private long _bytes;

        public long Bytes => _bytes;

        public nint Arena(long need)
        {
            need = Math.Max(need, 64);
            bool shrink = _p != 0 && _bytes > CpuArenaBudget && need <= CpuArenaBudget;
            if (_p != 0 && _bytes >= need && !shrink) return _p;
            bool headroom = _p != 0 && !shrink && need / 4 * 5 <= CpuArenaBudget;
            Release();
            long cap = headroom ? need / 4 * 5 : need;
            _p = (nint)NativeMemory.AllocZeroed((nuint)cap);
            _bytes = cap;
            GC.AddMemoryPressure(cap);
            return _p;
        }

        private void Release()
        {
            if (_p == 0) return;
            NativeMemory.Free((void*)_p);
            GC.RemoveMemoryPressure(_bytes);
            _p = 0; _bytes = 0;
        }

        public void Dispose()
        {
            Release();
            GC.SuppressFinalize(this);
        }

        ~CpuWorker() => Release();
    }

    /// <summary>Bounded rent/return pool of stateful workers (at most <c>max</c> idle and,
    /// with a <c>weight</c>, at most <c>budget</c> idle bytes beyond the first idle worker).</summary>
    private sealed class SizePool<T> : IDisposable where T : IDisposable
    {
        private readonly ConcurrentQueue<T> _idle = new();
        private readonly Func<T> _factory;
        private readonly int _max;
        private readonly Func<T, long>? _weight;
        private readonly long _budget;
        private int _idleCount;
        private long _idleBytes;
        private volatile bool _disposed;

        public SizePool(int max, Func<T> factory, Func<T, long>? weight = null, long budget = long.MaxValue)
        {
            _max = max; _factory = factory; _weight = weight; _budget = budget;
        }

        public T Rent()
        {
            if (_idle.TryDequeue(out T? t))
            {
                Interlocked.Decrement(ref _idleCount);
                if (_weight is not null) Interlocked.Add(ref _idleBytes, -_weight(t));
                return t;
            }
            return _factory();
        }

        public void Return(T t)
        {
            // pool may already be evicted/disposed — don't park into a dead pool
            if (_disposed) { t.Dispose(); return; }
            long w = _weight?.Invoke(t) ?? 0;
            if (Interlocked.Increment(ref _idleCount) <= _max
                && (Volatile.Read(ref _idleCount) == 1 || Interlocked.Read(ref _idleBytes) + w <= _budget))
            {
                Interlocked.Add(ref _idleBytes, w);
                _idle.Enqueue(t);
                // raced with Dispose after the check above: drain so nothing is parked in a dead pool
                if (_disposed) while (_idle.TryDequeue(out T? d)) d.Dispose();
                return;
            }
            Interlocked.Decrement(ref _idleCount);
            t.Dispose();
        }

        public void Dispose()
        {
            _disposed = true;
            while (_idle.TryDequeue(out T? t)) t.Dispose();
            _idleCount = 0;
            _idleBytes = 0;
        }
    }
}
