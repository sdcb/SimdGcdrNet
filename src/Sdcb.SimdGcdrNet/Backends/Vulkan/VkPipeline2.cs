using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdGcdrNet.Backends.Vulkan;

/// <summary>
/// Fused GPU worker: BGR u8 upload → bgr2nhwc shader → GC-Net → div6 (division
/// correction, interleaved 6ch NHWC) → DR-Net → nhwc2bgr → u8 readback. Both
/// plans keep physical (NHWC) I/O, so neither net contains a Transpose. One
/// command buffer, one submit per frame.
///
/// Size-agnostic: per-size state is only the cached graphs (plan + dispatch
/// list over the model's shared const store). The worker owns grow-only
/// activation arenas and BGR staging buffers and re-records its command buffer
/// when the source size changes, so a new size costs a re-record instead of a
/// pipeline build (arena allocation, weight upload, first-touch). Not
/// thread-safe; callers pool workers.
/// </summary>
internal sealed unsafe class VkGcdrPipeline : IDisposable
{
    private readonly VkDevice _dev;
    private readonly VkBackend _be;
    private readonly OnnxModel _gcModel, _drModel;
    private readonly IntPtr _cmdPool, _cmd, _fence;
    private IntPtr _descPool;              // !PushDescriptors: rebuilt on each record
    // one arena for both stages: dr at 0, gc stacked past the dr input (GcdrEnhancer.StackArenas)
    private VkBuffer? _arena, _bgrIn, _bgrOut;
    private ulong _gcBase;                 // bytes
    private byte* _bgrInMap, _bgrOutMap;
    private VkGraphModel? _gcG, _drG;      // graphs of the bound padded size (refs held)
    private int _h, _w;                    // padded dims
    private int _ih, _iw;                  // source dims
    private int _pt, _pl;
    private GcdrPixelFormat _fmt;
    private int Bpp => _fmt.BytesPerPixel();
    private bool _recorded;
    private readonly bool _f16;
    private readonly VkPipeline _pPre, _pDiv6, _pPost;

    internal VkGcdrPipeline(VkBackend be, OnnxModel gc, OnnxModel dr)
    {
        _be = be;
        _dev = be.Dev;
        _gcModel = gc;
        _drModel = dr;
        _f16 = be.F16;
        string sfx = _f16 ? "_h" : "";
        _pPre = _dev.GetPipeline("bgr2nhwc" + sfx, 2, 9 * 4, owner: GcdrShaders.Owner);
        _pDiv6 = _dev.GetPipeline("div6" + sfx, 3, 4, owner: GcdrShaders.Owner);
        _pPost = _dev.GetPipeline("nhwc2bgr" + sfx, 3, 9 * 4, owner: GcdrShaders.Owner);
        _cmdPool = _dev.NewCommandPool();
        try
        {
            _cmd = _dev.AllocateCommandBuffer(_cmdPool);
            _fence = _dev.NewFence();
        }
        catch
        {
            _dev.DestroyCommandPool(_cmdPool);
            throw;
        }
    }

    private int Es => _f16 ? 2 : 4;
    private ExecPlan GcPlan => _gcG!.Plan;
    private ExecPlan DrPlan => _drG!.Plan;
    private ulong InOff(ExecPlan p) => (ulong)p.BufOffset[p.Values[p.InputValue].Buf] * (ulong)Es;
    private ulong OutOff(ExecPlan p) => (ulong)p.BufOffset[p.Values[p.OutputValue].Buf] * (ulong)Es;
    private ulong OutBytes(ExecPlan p) => (ulong)p.Values[p.OutputValue].ElementCount * (ulong)Es;
    private ulong InBytes(ExecPlan p) => (ulong)p.Values[p.InputValue].ElementCount * (ulong)Es;

    /// <summary>Bind the worker to a source size: fetch (cached) graphs for its padded
    /// size, grow buffers as needed, re-record when anything changed.</summary>
    internal void Bind(int ih, int iw, GcdrPixelFormat fmt = GcdrPixelFormat.Bgr24)
    {
        if (_recorded && ih == _ih && iw == _iw && fmt == _fmt) return;
        _recorded = false;   // stays false if anything below throws (buffers may be gone)
        int pt = (-ih) & 31, pl = (-iw) & 31;
        int h = ih + pt, w = iw + pl;
        if (_gcG is null || h != _h || w != _w)
        {
            // a new size compiles two plans + graphs: build them side by side
            var tg = Task.Run(() => _be.AcquireGraph(_gcModel, [1, 3, h, w], GraphOptions));
            VkGraphModel gd;
            try { gd = _be.AcquireGraph(_drModel, [1, 6, h, w], GraphOptions); }
            catch { try { tg.GetAwaiter().GetResult().Release(); } catch { } throw; }
            VkGraphModel gg;
            try { gg = tg.GetAwaiter().GetResult(); }
            catch { gd.Release(); throw; }
            if (gg.F16 != _f16 || gd.F16 != _f16)
            {
                gg.Release(); gd.Release();
                throw new InvalidOperationException("gc/dr storage precision mismatch");
            }
            _gcG?.Release(); _drG?.Release();
            _gcG = gg; _drG = gd;
            _h = h; _w = w;
        }
        _ih = ih; _iw = iw; _pt = pt; _pl = pl; _fmt = fmt;

        ulong es = (ulong)Es;
        // gc base aligned to 256 B (any minStorageBufferOffsetAlignment)
        // the graphs' full arenas: the plan's values plus scratch / tile-read slack past them
        var (gcOff, total) = GcdrEnhancer.StackArenas((long)((_gcG!.ArenaBytes + es - 1) / es), DrPlan,
            (long)((_drG!.ArenaBytes + es - 1) / es), (long)(256 / es));
        _gcBase = (ulong)gcOff * es;
        ulong need = (ulong)total * es + 64;
        // the arena only grows while within ArenaBudget; an over-budget one (huge image)
        // is dropped to exact size as soon as a size that fits the budget comes along
        bool shrink = _arena is not null && _arena.Size > ArenaBudget && need <= ArenaBudget;
        // headroom (within budget): 1/8 on the first allocation so neighbouring sizes fit,
        // 1/4 on regrowth — a fresh multi-GB arena costs ~50-70 ms to allocate and first-touch
        ulong div = _arena is null ? 8UL : 4UL;
        ulong head = need / div * (div + 1) <= ArenaBudget ? div : 0UL;
        Arena(ref _arena, need, shrink, head);
        ulong bgrBytes = (ulong)ih * (ulong)iw * (ulong)Bpp + 4;
        Staging(ref _bgrIn, ref _bgrInMap, bgrBytes, preferHost: false);
        Staging(ref _bgrOut, ref _bgrOutMap, bgrBytes, preferHost: true);
        Record((uint)(_iw * Bpp));   // _bgrIn is always tightly packed (we de-stride on upload)
        _recorded = true;
    }

    /// <summary>Fused-pipeline graphs: physical I/O and the staged
    /// tensor-core convs on SG16 (Intel Xe2) devices — twice as fast here as the direct family and
    /// stress-clean on this model (the OCR models are not; see CompileOptions.StagedCoopConv).</summary>
    private static CompileOptions GraphOptions { get; } =
        VkBackend.PipeOptions with { StagedCoopConv = true };

    /// <summary>Per-worker device-arena budget (gc + dr). Within it the arena is grow-only
    /// (12.5% headroom on the first allocation, 25% on regrowth) so mixed sizes don't reallocate.
    /// 3 GB; on integrated GPUs the "device-local" heap is system RAM shared with the CPU, so
    /// the budget is also capped at a quarter of that heap.</summary>
    internal ulong ArenaBudget => BudgetFor(_dev);

    internal static ulong BudgetFor(VkDevice dev) =>
        dev.UnifiedMemory && dev.DeviceLocalHeap > 0 ? Math.Min(3UL << 30, dev.DeviceLocalHeap / 4) : 3UL << 30;

    /// <summary>Device bytes held by this worker (arena + BGR staging).</summary>
    internal long DeviceBytes => (long)((_arena?.Size ?? 0) + (_bgrIn?.Size ?? 0) + (_bgrOut?.Size ?? 0));

    /// <param name="buf">Buffer to (re)allocate; kept when already large enough (unless <paramref name="shrink"/>).</param>
    /// <param name="need">Required size in bytes.</param>
    /// <param name="shrink">Always reallocate at exactly <paramref name="need"/> bytes (releases an oversized buffer).</param>
    /// <param name="headDiv">0 = exact size, else allocate need * (1 + 1/headDiv).</param>
    private void Arena(ref VkBuffer? buf, ulong need, bool shrink, ulong headDiv)
    {
        if (buf is not null && buf.Size >= need && !shrink) return;
        buf?.Free();
        buf = null;
        buf = _dev.NewStorageBuffer(headDiv != 0 && !shrink ? need / headDiv * (headDiv + 1) : need, hostVisible: false);
    }

    /// <summary>Grow-only mapped staging buffer; the map pointer is cleared with the buffer
    /// it points into and re-established whenever missing.</summary>
    private void Staging(ref VkBuffer? buf, ref byte* map, ulong need, bool preferHost)
    {
        if (buf is not null && buf.Size < need)
        {
            if (map != null) buf.Unmap();
            map = null;
            buf.Free();
            buf = null;
        }
        buf ??= _dev.NewStorageBuffer(need + need / 4, hostVisible: true, preferHost);
        if (map == null) map = (byte*)buf.Map();
    }

    private static byte[] Pc(params uint[] u)
    {
        byte[] b = new byte[u.Length * 4];
        Buffer.BlockCopy(u, 0, b, 0, b.Length);
        return b;
    }

    private void Record(uint srcStride)
    {
        VkGraphModel gcG = _gcG!, drG = _drG!;
        ExecPlan gcPlan = gcG.Plan, drPlan = drG.Plan;
        if (!_dev.PushDescriptors)
        {
            if (_descPool != IntPtr.Zero) Vk.vkDestroyDescriptorPool(_dev.Device, _descPool, null);
            _descPool = IntPtr.Zero;
            _descPool = _dev.NewDescriptorPool((uint)(gcG.Dispatches.Length + drG.Dispatches.Length + 3));
        }

        uint h = (uint)_h, w = (uint)_w;
        uint bpp = (uint)Bpp, swap = _fmt.SwapRB() ? 1u : 0u;
        Vk.Check(Vk.vkResetCommandBuffer(_cmd, 0), "vkResetCommandBuffer");
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(_cmd, &begin), "vkBeginCommandBuffer");

        if (Environment.GetEnvironmentVariable("GCDR_ZERO_ARENA") == "1")
        {
            Vk.vkCmdFillBuffer(_cmd, _arena!.Buffer, 0, _arena.Size, 0);
            Vk.VkMemoryBarrier fb = new()
            {
                SType = VkConst.StMemoryBarrier,
                SrcAccessMask = VkConst.AccessTransferWrite, DstAccessMask = VkConst.AccessShaderRead | VkConst.AccessShaderWrite,
            };
            Vk.vkCmdPipelineBarrier(_cmd, VkConst.PipelineStageTransfer, VkConst.PipelineStageComputeShader, 0,
                1, &fb, 0, null, 0, null);
        }

        // bgr2nhwc -> gc arena input
        Dispatch(_pPre,
            [(_bgrIn!, 0UL, _bgrIn!.Size), (_arena!, _gcBase + InOff(gcPlan), InBytes(gcPlan))],
            Pc(h, w, (uint)_ih, (uint)_iw, (uint)_pt, (uint)_pl, srcStride, bpp, swap),
            (w + 255) / 256, h, 1);
        _dev.CmdComputeBarrier(_cmd);
        RecordGraph(gcG, _arena!, _gcBase);

        // div6: gc input + gc output -> dr arena input
        uint hw = h * w;
        ulong es = (ulong)Es;
        _dev.CmdComputeBarrier(_cmd);
        Dispatch(_pDiv6,
            [(_arena!, _gcBase + InOff(gcPlan), 3UL * hw * es),
             (_arena!, _gcBase + OutOff(gcPlan), 3UL * hw * es),
             (_arena!, InOff(drPlan), 6UL * hw * es)],
            Pc(hw), (hw + 255) / 256, 1, 1);
        _dev.CmdComputeBarrier(_cmd);
        RecordGraph(drG, _arena!, 0);

        // nhwc2bgr -> bgrOut
        _dev.CmdComputeBarrier(_cmd);
        uint nwords = bpp == 4 ? (uint)_ih * (uint)_iw : ((uint)_ih * (uint)_iw * 3u + 3u) / 4u;
        Dispatch(_pPost,
            [(_arena!, OutOff(drPlan), OutBytes(drPlan)), (_bgrOut!, 0UL, _bgrOut!.Size), (_bgrIn!, 0UL, _bgrIn!.Size)],
            Pc(h, w, (uint)_ih, (uint)_iw, (uint)_pt, (uint)_pl, nwords, bpp, swap),   // PC layout must match nhwc2bgr.comp
            (nwords + 255) / 256, 1, 1);
        Vk.Check(Vk.vkEndCommandBuffer(_cmd), "vkEndCommandBuffer");
    }

    /// <summary>All graph dispatches with barriers between them (none before the first).</summary>
    private void RecordGraph(VkGraphModel g, VkBuffer arena, ulong arenaBase)
    {
        var binds = new (VkBuffer buf, ulong off, ulong size)[8];
        for (int i = 0; i < g.Dispatches.Length; i++)
        {
            VkDispatch d = g.Dispatches[i];
            if (i > 0) _dev.CmdComputeBarrier(_cmd);
            if (binds.Length < d.Binds.Length) binds = new (VkBuffer, ulong, ulong)[d.Binds.Length];
            for (int b = 0; b < d.Binds.Length; b++)
            {
                VkBind vb = d.Binds[b];
                VkBuffer buf = vb.Slot switch
                {
                    VkSlot.Arena => arena,
                    VkSlot.Consts => g.Consts,
                    VkSlot.Dummy => g.Dummy,
                    VkSlot.Direct => vb.Buf!,
                    _ => throw new InvalidOperationException($"pipeline graph dispatch binds {vb.Slot}"),
                };
                binds[b] = (buf, vb.Slot == VkSlot.Arena ? arenaBase + vb.Offset : vb.Offset, vb.Size);
            }
            Dispatch(d.Pipe, binds.AsSpan(0, d.Binds.Length), d.Pc, d.Gx, d.Gy, d.Gz);
        }
    }

    private void Dispatch(VkPipeline pipe, ReadOnlySpan<(VkBuffer buf, ulong off, ulong size)> binds,
        byte[] pc, uint gx, uint gy, uint gz)
    {
        Vk.vkCmdBindPipeline(_cmd, VkConst.BindPointCompute, pipe.Pipeline);
        if (_dev.PushDescriptors)
        {
            var writes = stackalloc Vk.VkWriteDescriptorSet[8];
            var infos = stackalloc Vk.VkDescriptorBufferInfo[8];
            for (int b = 0; b < binds.Length; b++)
            {
                infos[b] = new Vk.VkDescriptorBufferInfo { Buffer = binds[b].buf.Buffer, Offset = binds[b].off, Range = binds[b].size };
                writes[b] = new Vk.VkWriteDescriptorSet
                {
                    SType = VkConst.StWriteDescriptorSet, DstBinding = (uint)b,
                    DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer,
                    PBufferInfo = infos + b,
                };
            }
            _dev.CmdPushDescriptors(_cmd, pipe.Layout, writes, (uint)binds.Length);
        }
        else
        {
            IntPtr set = _dev.AllocateDescriptorSet(_descPool, pipe.SetLayout);
            for (int b = 0; b < binds.Length; b++)
                _dev.BindBuffer(set, (uint)b, binds[b].buf, binds[b].off, binds[b].size);
            Vk.vkCmdBindDescriptorSets(_cmd, VkConst.BindPointCompute, pipe.Layout, 0, 1, &set, 0, null);
        }
        if (pc.Length > 0)
            fixed (byte* p = pc)
                Vk.vkCmdPushConstants(_cmd, pipe.Layout, VkConst.StageComputeShader, 0, (uint)pc.Length, p);
        Vk.vkCmdDispatch(_cmd, gx, gy, gz);
    }

    /// <summary>Enhance one image of any size (binds/re-records first when the size changed).</summary>
    public void Enhance(ReadOnlySpan<byte> bgrSrc, int ih, int iw, int srcStride, Span<byte> bgrDst, int dstStride,
        GcdrPixelFormat fmt = GcdrPixelFormat.Bgr24)
    {
        Bind(ih, iw, fmt);
        Enhance(bgrSrc, srcStride, bgrDst, dstStride);
    }

    /// <summary>Enhance one image of the currently bound size and pixel format.</summary>
    public void Enhance(ReadOnlySpan<byte> bgrSrc, int srcStride, Span<byte> bgrDst, int dstStride)
    {
        if (!_recorded) throw new InvalidOperationException("worker not bound to a size");
        int row = _iw * Bpp;
        ulong inBytes = (ulong)_ih * (ulong)row;
        if (srcStride == row)
            bgrSrc.Slice(0, (int)inBytes).CopyTo(new Span<byte>(_bgrInMap, (int)inBytes));
        else
            for (int y = 0; y < _ih; y++)
                bgrSrc.Slice(y * srcStride, row).CopyTo(
                    new Span<byte>(_bgrInMap + y * row, row));
        _bgrIn!.Flush(0, inBytes);

        _dev.Submit(_cmd, _fence);
        _dev.WaitFence(_fence);
        _bgrOut!.Invalidate(0, inBytes);

        if (dstStride == row)
            new ReadOnlySpan<byte>(_bgrOutMap, (int)inBytes).CopyTo(bgrDst);
        else
            for (int y = 0; y < _ih; y++)
                new ReadOnlySpan<byte>(_bgrOutMap + y * row, row)
                    .CopyTo(bgrDst.Slice(y * dstStride));
    }

    /// <summary>Debug view of one stage (gc or dr) of the bound size.</summary>
    internal sealed class DebugStage(VkGcdrPipeline p, bool gc)
    {
        private VkGraphModel G => (gc ? p._gcG : p._drG) ?? throw new InvalidOperationException("worker not bound");
        private VkBuffer Arena => p._arena!;
        private ulong Base => gc ? p._gcBase : 0;
        public ExecPlan DebugPlan => G.Plan;
        /// <summary>Download the plan's arena window (ArenaFloats elements) into dst.</summary>
        public void DebugArena(Span<float> dst) => p.Download(Arena, Base, dst.Slice(0, G.Plan.ArenaFloats));
        public (VkBuffer consts, VkBuffer arena) DebugBuffers => (G.Consts, Arena);
    }

    internal DebugStage DebugGc => new(this, true);
    internal DebugStage DebugDr => new(this, false);

    private void Download(VkBuffer buf, ulong off, Span<float> dst)
    {
        ulong bytes = (ulong)dst.Length * (ulong)Es;
        byte[] tmp = new byte[bytes];
        fixed (byte* t = tmp)
        {
            _dev.Download(buf, t, bytes, off);
            VkInferenceSession.FromStorage(t, dst, _f16);
        }
    }

    /// <summary>Debug: download intermediate tensors (gc input, gc shadow out, dr 6ch input — padded, NHWC).</summary>
    internal void DebugIntermediates(Span<float> gcIn, Span<float> gcOut, Span<float> drIn)
    {
        int hw = _h * _w;
        // gc's window is overwritten by dr: only valid while dr has not clobbered it
        // (dr input + whatever dr left untouched); kept for bisecting
        Download(_arena!, _gcBase + InOff(GcPlan), gcIn.Slice(0, 3 * hw));
        Download(_arena!, _gcBase + OutOff(GcPlan), gcOut.Slice(0, 3 * hw));
        Download(_arena!, InOff(DrPlan), drIn.Slice(0, 6 * hw));
    }

    public void Dispose()
    {
        // Enhance() is synchronous (already waits the fence); callers must not race Dispose
        _dev.DestroyCommandPool(_cmdPool);
        if (_descPool != IntPtr.Zero) { Vk.vkDestroyDescriptorPool(_dev.Device, _descPool, null); _descPool = IntPtr.Zero; }
        _dev.DestroyFence(_fence);
        if (_bgrIn is not null) { if (_bgrInMap != null) _bgrIn.Unmap(); _bgrIn.Free(); _bgrIn = null; _bgrInMap = null; }
        if (_bgrOut is not null) { if (_bgrOutMap != null) _bgrOut.Unmap(); _bgrOut.Free(); _bgrOut = null; _bgrOutMap = null; }
        _arena?.Free(); _arena = null;
        _gcG?.Release(); _gcG = null;
        _drG?.Release(); _drG = null;
        _recorded = false;
    }
}
