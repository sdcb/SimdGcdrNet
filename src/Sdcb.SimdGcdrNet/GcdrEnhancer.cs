using System.Collections.Concurrent;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Imaging;
using Sdcb.SimdOnnx.Kernels;
using Sdcb.SimdOnnx.Onnx;
using MetalBackend = Sdcb.SimdOnnx.Backends.Metal.MetalBackend;
using MtlImageSession = Sdcb.SimdOnnx.Backends.Metal.MtlImageSession;
using VkBackend = Sdcb.SimdOnnx.Backends.Vulkan.VkBackend;
using VkDeviceLostException = Sdcb.SimdOnnx.Backends.Vulkan.VkDeviceLostException;
using VkImageSession = Sdcb.SimdOnnx.Backends.Vulkan.VkImageSession;

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

    // CPU: one model for both nets and the glue between them (Compose), so one plan per padded
    // size (at most MaxSizes compiled; packed weights shared across sizes) and one arena per
    // session. Sessions serve every size (grow-only arenas); idle ones beyond the first are kept
    // only while their arenas fit the budget, and an over-budget one (a huge image) is dropped.
    private const int MaxSizes = 8, MaxPerSize = 4;
    private const long CpuArenaBudget = 4L << 30;
    private OnnxModel? _model;
    private CpuModel? _cpuModel;
    private readonly Dictionary<long, long> _cpuSizes = new();   // LRU mirror of the CpuModel's plan cache (PoolStats)
    private long _cpuClock;
    // Cache Metal availability for the process: re-probing (incl. the MSL library
    // compile) per Enhance would re-throw on init failure for every frame.
    private static readonly Lazy<bool> s_metalOk = new(MetalOk);

    /// <summary>Whether the Metal backend can run here (what <see cref="GcdrBackend.Auto"/> checks).</summary>
    internal static bool MetalUsable => GpuPossible && s_metalOk.Value;
    private SizePool<CpuSession>? _cpuPool;
    // GPU image sessions are size-agnostic (budgeted grow-only arenas, re-record on size
    // change): one pool serves every size; idle sessions beyond the first are kept
    // only while their device memory fits the budget
    // (3 GB; integrated GPUs: at most a quarter of the shared device-local heap)
    private static long VkIdleBudget
    {
        get
        {
            var d = VkBackend.Default.Device;
            return d.UnifiedMemory && d.DeviceLocalBytes > 0 ? Math.Min(3L << 30, (long)(d.DeviceLocalBytes / 4)) : 3L << 30;
        }
    }
    private const long MtlIdleBudget = 3L << 30;
    private SizePool<VkImageSession>? _vkPool;
    private SizePool<MtlImageSession>? _mtlPool;
    private long _poolsCreated, _workersCreated, _poolsEvicted;

    /// <summary>Diagnostics: CPU size plans compiled (Vulkan: worker pools created), worker
    /// instances (CPU sessions) created, CPU size plans evicted.</summary>
    public (long pools, long workers, long evicted) PoolStats =>
        (_poolsCreated, _workersCreated, _poolsEvicted);

    /// <summary>Debug/test hook: invoked with the just-used Vulkan session after each Enhance.</summary>
    internal Action<VkImageSession>? DebugAfterEnhance;

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

    // the network sees B,G,R in [0, 1]; sessions keep I/O in plan layout (channels-last), so the
    // plan has no boundary relayouts
    private static readonly PixelNormalization s_unit = PixelNormalization.Unit();
    // every run writes a fresh input, so the arena may reuse it once the glue has read it
    private static readonly CompileOptions CpuPlan = new(CompileOptions.DefaultNhwc, PhysicalIo: true) { ReleaseInputs = true };
    private static bool CpuNhwc => CpuPlan.Nhwc;

    /// <summary>
    /// GC-Net → corrected = clamp(x / max(shading, 1e-6), 0, 1) → DR-Net on concat(x, corrected), as
    /// one model over the [1, 3, h, w] input in [0, 1].
    /// </summary>
    /// <summary>The DR-Net input tensor of the composed model (tests cut the model there).</summary>
    internal const string DrInput = "dr_in";

    internal static OnnxModel Compose(OnnxModel gc, OnnxModel dr)
    {
        var b = new OnnxModelBuilder(gc.Opset);
        string x = b.Input("x", 1, 3, -1, -1);
        string shade = b.Inline(gc, "gc/", x)[0];
        string eps = b.Constant("eps", [1e-6f]), zero = b.Constant("zero", [0f]), one = b.Constant("one", [1f]);
        string q = b.Node("Div", [x, b.Node("Clip", [shade, eps, ""])]);
        string corrected = b.Node("Clip", [q, zero, one]);
        string x6 = b.Node("Concat", [x, corrected], DrInput, ("axis", 1));
        b.Output(b.Inline(dr, "dr/", x6)[0]);
        return b.Build();
    }

    /// <summary>The composed model (one instance: the backends cache graphs per model).</summary>
    private OnnxModel Model()
    {
        if (_model is { } m) return m;
        m = Compose(_gcModel, _drModel);
        return Interlocked.CompareExchange(ref _model, m, null) ?? m;
    }

    private CpuModel CpuModel()
    {
        if (_cpuModel is { } m) return m;
        m = new CpuModel(Model(), CpuPlan, MaxSizes);
        return Interlocked.CompareExchange(ref _cpuModel, m, null) ?? m;
    }

    /// <summary>GPU pixel I/O: B,G,R in [0, 1], padded to a multiple of 32 at the top/left. The staged
    /// tensor-core convs on SG16 (Intel Xe2) devices are twice as fast here as the direct family and
    /// stress-clean on this model (the OCR models are not; see CompileOptions.StagedCoopConv).</summary>
    private static readonly ImageSessionOptions s_vkImage = new()
    {
        PadMultiple = 32, Padding = ImagePadding.TopLeft,
        Compile = VkBackend.PipeOptions with { StagedCoopConv = true, ReleaseInputs = true },
    };
    private static readonly ImageSessionOptions s_mtlImage = new()
    {
        PadMultiple = 32, Padding = ImagePadding.TopLeft,
        Compile = MetalBackend.PipeOptions with { ReleaseInputs = true },
    };

    /// <summary>Counts a padded size the way the CPU model's LRU plan cache sees it.</summary>
    private void NoteCpuSize(int h, int w)
    {
        long key = (long)h * 65536 + w;
        lock (_cpuSizes)
        {
            bool added = !_cpuSizes.ContainsKey(key);
            _cpuSizes[key] = ++_cpuClock;
            if (!added) return;
            _poolsCreated++;
            if (_cpuSizes.Count <= MaxSizes) return;
            long oldest = _cpuSizes.Where(kv => kv.Key != key).OrderBy(kv => kv.Value).First().Key;
            _cpuSizes.Remove(oldest);
            _poolsEvicted++;
        }
    }

    private SizePool<CpuSession> CpuPool()
    {
        if (_cpuPool is { } p) return p;
        p = new SizePool<CpuSession>(MaxPerSize,
            () => { Interlocked.Increment(ref _workersCreated); return new CpuSession(CpuModel()); },
            w => w.ArenaBytes, CpuArenaBudget);
        if (Interlocked.CompareExchange(ref _cpuPool, p, null) is { } won) { p.Dispose(); return won; }
        return p;
    }

    private SizePool<VkImageSession> VkPool()
    {
        if (_vkPool is { } p) return p;
        p = new SizePool<VkImageSession>(MaxPerSize,
            () => { Interlocked.Increment(ref _workersCreated); return VkBackend.Default.CreateImageSession(Model(), s_vkImage); },
            w => w.DeviceBytes, VkIdleBudget);
        if (Interlocked.CompareExchange(ref _vkPool, p, null) is { } won) { p.Dispose(); return won; }
        Interlocked.Increment(ref _poolsCreated);
        return p;
    }

    private SizePool<MtlImageSession> MtlPool()
    {
        if (_mtlPool is { } p) return p;
        p = new SizePool<MtlImageSession>(MaxPerSize,
            () => { Interlocked.Increment(ref _workersCreated); return MetalBackend.Default.CreateImageSession(Model(), s_mtlImage); },
            w => w.DeviceBytes, MtlIdleBudget);
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
            try { pipe.Run(src, width, height, srcStride, dst, dstStride, (PixelFormat)format); }
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
                    pipe.Run(src, width, height, srcStride, dst, dstStride, (PixelFormat)format);
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

        var cpuPool = CpuPool();
        CpuSession session = cpuPool.Rent();
        bool keep = true;
        try
        {
            session.Reshape([1, 3, h, w], session.Model);
            NoteCpuSize(h, w);
            int[] oshape = session.OutputShape();
            if (oshape[2] != h || oshape[3] != w)
                throw new NotSupportedException($"The networks returned {oshape[2]}x{oshape[3]}, expected {h}x{w}.");
            TensorLayout layout = CpuNhwc ? TensorLayout.Nhwc : TensorLayout.Nchw;
            int threads = Environment.ProcessorCount;
            // BGR u8 → float /255, top/left replicate pad
            ImageTensor.FromPixels(src, width, height, srcStride, (PixelFormat)format, session.Input, layout, s_unit,
                w, h, padLeft: pw, padTop: ph, threads: threads);
            ReadOnlySpan<float> y = session.Run();
            // crop the pad → u8 in the caller's format (alpha from src)
            ImageTensor.ToPixels(y, layout, w, h, dst, width, height, dstStride, (PixelFormat)format,
                cropLeft: pw, cropTop: ph, alphaSource: bpp == 4 ? src : default, alphaStride: srcStride,
                threads: threads);
            // an arena past the budget (one huge image) is not kept for the sizes after it
            keep = session.ArenaBytes <= CpuArenaBudget;
        }
        catch { keep = false; throw; }
        finally
        {
            if (keep) cpuPool.Return(session);
            else session.Dispose();
        }
    }

    // browser/WASI: unlinked P/Invokes (vulkan-1, libobjc) trap the wasm module with
    // "null function" instead of throwing, so every backend (Auto/Vulkan/Metal) resolves to CPU
    private static bool GpuPossible => !OperatingSystem.IsBrowser() && !OperatingSystem.IsWasi();

    private static bool VulkanOk() => VkBackend.TryGetDefault(out _);

    private static bool MetalOk()
    {
        try
        {
            // runs a tiny conv stack end to end: the MSL library compile happens now rather than
            // mid-Enhance (macOS 13), and devices that cannot build the simdgroup-matrix convs (Intel
            // Macs can claim Apple7 yet build them with 0 threads per group) fail here
            if (!MetalBackend.TryGetDefault(out MetalBackend? mb)) return false;
            using MtlImageSession s = mb.CreateImageSession(ProbeModel(), s_mtlImage);
            byte[] img = new byte[32 * 32 * 3];
            s.Run(img, 32, 32, 96, img, 96);
            return true;
        }
        catch { return false; }
    }

    /// <summary>x -> 3x3 conv (3 to 32) -> relu -> 3x3 conv (32 to 3): the conv shapes the networks start with.</summary>
    private static OnnxModel ProbeModel()
    {
        var b = new OnnxModelBuilder(18);
        string x = b.Input("x", 1, 3, -1, -1);
        string w1 = b.Constant("w1", new float[32 * 3 * 9], 32, 3, 3, 3), w2 = b.Constant("w2", new float[3 * 32 * 9], 3, 32, 3, 3);
        string h = b.Node("Relu", [b.Node("Conv", [x, w1], null, ("pads", new[] { 1, 1, 1, 1 }))]);
        b.Output(b.Node("Conv", [h, w2], null, ("pads", new[] { 1, 1, 1, 1 })));
        return b.Build();
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
        lock (_cpuSizes) _cpuSizes.Clear();
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
