using Sdcb.SimdOnnx.Backends.Metal;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdGcdrNet.Backends.Metal;

/// <summary>
/// Fused GPU worker: BGR u8 upload → bgr2nhwc → GC-Net → div6 (division
/// correction, interleaved 6ch NHWC) → DR-Net → nhwc2bgr → u8 readback. Both
/// plans keep physical (NHWC) I/O, so neither net contains a Transpose. All
/// buffers are shared memory — upload is a memcpy into a mapped buffer, one
/// command buffer submit per frame.
///
/// Size-agnostic: per-size state is only the cached graphs (plan + dispatch
/// list over the model's shared const store). The worker owns grow-only
/// activation arenas and BGR staging buffers and re-binds when the source size
/// changes. Not thread-safe; callers pool workers.
/// </summary>
internal sealed unsafe class MtlGcdrPipeline : IDisposable
{
    private readonly MtlDevice _dev;
    private readonly MetalBackend _be;
    private readonly OnnxModel _gcModel, _drModel;
    // one arena for both stages: dr at 0, gc stacked past the dr input (GcdrEnhancer.StackArenas)
    private MtlBuffer? _arena, _bgrIn, _bgrOut;
    private ulong _gcBase;                 // bytes
    private MtlGraphModel? _gcG, _drG;     // graphs of the bound padded size (refs held)
    private int _h, _w;                    // padded dims
    private int _ih, _iw;                  // source dims
    private int _pt, _pl;
    private bool _bound;
    private const int Es = 2;              // fp16 storage

    internal MtlGcdrPipeline(MetalBackend be, OnnxModel gc, OnnxModel dr)
    {
        _be = be;
        _dev = be.Dev;
        _gcModel = gc;
        _drModel = dr;
    }

    private ExecPlan GcPlan => _gcG!.Plan;
    private ExecPlan DrPlan => _drG!.Plan;
    private static ulong InOff(ExecPlan p) => (ulong)p.BufOffset[p.Values[p.InputValue].Buf] * Es;
    private static ulong OutOff(ExecPlan p) => (ulong)p.BufOffset[p.Values[p.OutputValue].Buf] * Es;
    private static ulong OutBytes(ExecPlan p) => (ulong)p.Values[p.OutputValue].ElementCount * Es;
    private static ulong InBytes(ExecPlan p) => (ulong)p.Values[p.InputValue].ElementCount * Es;

    /// <summary>Bind the worker to a source size: fetch (cached) graphs for its padded
    /// size and grow buffers as needed.</summary>
    internal void Bind(int ih, int iw)
    {
        if (_bound && ih == _ih && iw == _iw) return;
        _bound = false;
        int pt = (-ih) & 31, pl = (-iw) & 31;
        int h = ih + pt, w = iw + pl;
        if (_gcG is null || h != _h || w != _w)
        {
            // a new size compiles two plans + graphs: build them side by side
            var tg = Task.Run(() => _be.AcquireGraph(_gcModel, [1, 3, h, w], physIo: true));
            MtlGraphModel gd;
            try { gd = _be.AcquireGraph(_drModel, [1, 6, h, w], physIo: true); }
            catch { try { tg.GetAwaiter().GetResult().Release(); } catch { } throw; }
            MtlGraphModel gg;
            try { gg = tg.GetAwaiter().GetResult(); }
            catch { gd.Release(); throw; }
            _gcG?.Release(); _drG?.Release();
            _gcG = gg; _drG = gd;
            _h = h; _w = w;
        }
        _ih = ih; _iw = iw; _pt = pt; _pl = pl;

        // gc base aligned to 256 B
        var (gcOff, total) = GcdrEnhancer.StackArenas(GcPlan, DrPlan, 256 / Es);
        _gcBase = (ulong)gcOff * Es;
        ulong need = (ulong)total * Es + 64;
        bool shrink = _arena is not null && _arena.Length > ArenaBudget && need <= ArenaBudget;
        ulong div = _arena is null ? 8UL : 4UL;
        ulong head = need / div * (div + 1) <= ArenaBudget ? div : 0UL;
        if (_arena is null || _arena.Length < need || shrink)
        {
            _arena?.Dispose();
            _arena = _dev.NewBuffer((nuint)(head != 0 && !shrink ? need / head * (head + 1) : need));
        }
        ulong bgrBytes = (ulong)ih * (ulong)iw * 3 + 4;
        if (_bgrIn is null || _bgrIn.Length < bgrBytes)
        {
            _bgrIn?.Dispose();
            _bgrIn = _dev.NewBuffer((nuint)(bgrBytes + bgrBytes / 4));
        }
        if (_bgrOut is null || _bgrOut.Length < bgrBytes)
        {
            _bgrOut?.Dispose();
            _bgrOut = _dev.NewBuffer((nuint)(bgrBytes + bgrBytes / 4));
        }
        _bound = true;
    }

    /// <summary>Per-worker device-arena budget (gc + dr): grow-only within it (12.5%
    /// headroom on first allocation, 25% on regrowth). 3 GB.</summary>
    internal static ulong ArenaBudget => 3UL << 30;

    /// <summary>Device bytes held by this worker (arena + BGR staging).</summary>
    internal long DeviceBytes => (long)((_arena?.Length ?? 0) + (_bgrIn?.Length ?? 0) + (_bgrOut?.Length ?? 0));

    private void RecordAll(MtlEncoder enc, int srcStride)
    {
        MtlGraphModel gcG = _gcG!, drG = _drG!;
        ExecPlan gcPlan = gcG.Plan, drPlan = drG.Plan;
        uint h = (uint)_h, w = (uint)_w;

        // bgr2nhwc -> gc arena input
        enc.SetPipeline(_dev.Pipe("bgr2nhwc", GcdrShaders.Owner));
        enc.SetBuffer(_bgrIn!, 0, 0);
        enc.SetBuffer(_arena!, _gcBase + InOff(gcPlan), 1);
        var pc = stackalloc uint[7] { h, w, (uint)_ih, (uint)_iw, (uint)_pt, (uint)_pl, (uint)srcStride };
        enc.SetPc(pc, 7 * 4, 2);
        enc.Dispatch((w + 255) / 256, h, 1, 256, 1, 1);

        foreach (var d in gcG.Dispatches)
            MtlInferenceSession.RecordOne(enc, gcG, d, _arena!, _gcBase);

        // div6: gc input + gc output -> dr arena input
        uint hw = h * w;
        enc.SetPipeline(_dev.Pipe("div6", GcdrShaders.Owner));
        enc.SetBuffer(_arena!, _gcBase + InOff(gcPlan), 0);
        enc.SetBuffer(_arena!, _gcBase + OutOff(gcPlan), 1);
        enc.SetBuffer(_arena!, InOff(drPlan), 2);
        var pc2 = stackalloc uint[1] { hw };
        enc.SetPc(pc2, 4, 3);
        enc.Dispatch((hw + 255) / 256, 1, 1, 256, 1, 1);

        foreach (var d in drG.Dispatches)
            MtlInferenceSession.RecordOne(enc, drG, d, _arena!);

        // nhwc2bgr -> bgrOut
        uint nwords = ((uint)_ih * (uint)_iw * 3u + 3u) / 4u;
        enc.SetPipeline(_dev.Pipe("nhwc2bgr", GcdrShaders.Owner));
        enc.SetBuffer(_arena!, OutOff(drPlan), 0);
        enc.SetBuffer(_bgrOut!, 0, 1);
        var pc3 = stackalloc uint[7] { h, w, (uint)_ih, (uint)_iw, (uint)_pt, (uint)_pl, nwords };
        enc.SetPc(pc3, 7 * 4, 2);
        enc.Dispatch((nwords + 255) / 256, 1, 1, 256, 1, 1);
    }

    /// <summary>Enhance one image of any size (re-binds first when the size changed).</summary>
    /// <remarks>Non-BGR24 formats are converted to/from BGR24 on the CPU while copying
    /// through the (unified-memory) staging buffers; the shaders only see BGR24.</remarks>
    public void Enhance(ReadOnlySpan<byte> bgrSrc, int ih, int iw, int srcStride, Span<byte> bgrDst, int dstStride,
        GcdrPixelFormat fmt = GcdrPixelFormat.Bgr24)
    {
        Bind(ih, iw);
        if (fmt == GcdrPixelFormat.Bgr24)
        {
            Enhance(bgrSrc, srcStride, bgrDst, dstStride);
            return;
        }
        int row = iw * 3;
        byte* srcMap = (byte*)_bgrIn!.Contents;
        for (int y = 0; y < ih; y++)
            Kernels.PixelOps.RowToBgr24(bgrSrc.Slice(y * srcStride), new Span<byte>(srcMap + y * row, row), iw, fmt);
        Run();
        byte* dstMap = (byte*)_bgrOut!.Contents;
        for (int y = 0; y < ih; y++)
            Kernels.PixelOps.RowFromBgr24(new ReadOnlySpan<byte>(dstMap + y * row, row), bgrDst.Slice(y * dstStride),
                iw, fmt, bgrSrc.Slice(y * srcStride));
    }

    private void Run()
    {
        using var pool = AutoReleasePool.Create();
        var enc = _dev.BeginCommands();
        RecordAll(enc, _iw * 3); // staging is always tightly packed
        enc.End();
        enc.CommitAndWait();
    }

    /// <summary>Enhance one BGR24 image of the currently bound size.</summary>
    public void Enhance(ReadOnlySpan<byte> bgrSrc, int srcStride, Span<byte> bgrDst, int dstStride)
    {
        if (!_bound) throw new InvalidOperationException("worker not bound to a size");
        ulong inBytes = (ulong)_ih * (ulong)_iw * 3;
        byte* srcMap = (byte*)_bgrIn!.Contents;
        if (srcStride == _iw * 3)
            bgrSrc.Slice(0, (int)inBytes).CopyTo(new Span<byte>(srcMap, (int)inBytes));
        else
            for (int y = 0; y < _ih; y++)
                bgrSrc.Slice(y * srcStride, _iw * 3).CopyTo(
                    new Span<byte>(srcMap + y * _iw * 3, _iw * 3));

        Run();

        byte* dstMap = (byte*)_bgrOut!.Contents;
        ulong outBytes = (ulong)_iw * 3;
        if (dstStride == _iw * 3)
            new ReadOnlySpan<byte>(dstMap, (int)(outBytes * (ulong)_ih)).CopyTo(bgrDst);
        else
            for (int y = 0; y < _ih; y++)
                new ReadOnlySpan<byte>(dstMap + y * _iw * 3, _iw * 3).CopyTo(
                    bgrDst.Slice(y * dstStride, _iw * 3));
    }

    /// <summary>Debug handle for one stage (gc or dr) of the bound size.</summary>
    internal sealed class DebugStage(MtlGcdrPipeline p, bool gc)
    {
        internal MtlGraphModel G => (gc ? p._gcG : p._drG)!;
        internal MtlBuffer Arena => p._arena!;
        internal ulong Base => gc ? p._gcBase : 0;
        public ExecPlan DebugPlan => G.Plan;
        public void DebugArena(Span<float> dst)
            => MtlInferenceSession.FromStorage((byte*)Arena.Contents + Base, dst.Slice(0, G.Plan.ArenaFloats));
        public (MtlBuffer consts, MtlBuffer arena) DebugBuffers => (G.Consts, Arena);
    }

    internal DebugStage DebugGc => new(this, true);
    internal DebugStage DebugDr => new(this, false);

    /// <summary>Debug/test: run the fused pipeline then download the gc input/output
    /// and the dr input windows.</summary>
    internal void DebugIntermediates(Span<float> gcIn, Span<float> gcOut, Span<float> drIn)
    {
        ExecPlan gcPlan = GcPlan, drPlan = DrPlan;
        int hw = _h * _w;
        MtlInferenceSession.FromStorage((byte*)_arena!.Contents + _gcBase + InOff(gcPlan), gcIn.Slice(0, 3 * hw));
        MtlInferenceSession.FromStorage((byte*)_arena!.Contents + _gcBase + OutOff(gcPlan), gcOut.Slice(0, 3 * hw));
        MtlInferenceSession.FromStorage((byte*)_arena!.Contents + InOff(drPlan), drIn.Slice(0, 6 * hw));
    }

    public void Dispose()
    {
        _arena?.Dispose(); _bgrIn?.Dispose(); _bgrOut?.Dispose();
        _gcG?.Release(); _drG?.Release();
    }
}
