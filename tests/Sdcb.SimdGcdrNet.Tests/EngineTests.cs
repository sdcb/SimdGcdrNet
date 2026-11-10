using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Kernels;
using Sdcb.SimdOnnx.Onnx;
using Xunit;

namespace Sdcb.SimdGcdrNet.Tests;

/// <summary>Stage-2 glue math and the CPU pipeline paths (engine fusions are tested in SimdOnnx).</summary>
public class EngineTests
{
    internal static string GcPath => Path.Combine(Golden.ModelDir, "gcdrnet_gcnet_sim.onnx");
    internal static string DrPath => Path.Combine(Golden.ModelDir, "gcdrnet_drnet_sim.onnx");

    [Theory]
    [InlineData("small")]
    [InlineData("pad")]
    [InlineData("mid")]
    public void DivConcat_MatchesOracle_Float(string tag)
    {
        // stage-2 math verified at float precision, not just through u8 output: the composed
        // model (GC-Net + glue) cut at the DR-Net input, on the oracle input, vs the oracle dr_in
        float[] x = Golden.F32(tag, "input");
        int[] shape = Golden.ShapeOf(tag, "input");
        float[] expect = Golden.F32(tag, "dr_in");

        OnnxModel m = GcdrEnhancer.Compose(OnnxModel.Load(GcPath), OnnxModel.Load(DrPath));
        using var sess = new CpuSession(new CpuModel(m).WithOutput(GcdrEnhancer.DrInput));
        sess.Reshape(shape);
        x.CopyTo(sess.Input);
        float[] drIn = sess.Run().ToArray();

        var (max, rmse, _) = Golden.Diff(drIn, expect);
        Assert.True(max < 1e-3f, $"dr_in[{tag}] max diff {max} rmse {rmse}");
    }

    [Fact]
    public void Pipeline_SizeSwitching_MatchesFresh_Cpu()
    {
        // one size-agnostic worker reuses (dirty, larger) arenas across sizes and the
        // per-size plans share packed weights: every size must match a fresh enhancer
        string[] tags = ["mid", "small", "pad", "mid", "small"];
        var fresh = new Dictionary<string, byte[]>();
        foreach (string tag in tags.Distinct())
        {
            var (_, _, _, _, oh, ow) = Golden.Dims(tag);
            using var e = GcdrEnhancer.FromFiles(GcPath, DrPath);
            fresh[tag] = e.Enhance(Golden.Bgr(tag), ow, oh);
        }

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath);
        foreach (string tag in tags)
        {
            var (_, _, _, _, oh, ow) = Golden.Dims(tag);
            byte[] dst = enh.Enhance(Golden.Bgr(tag), ow, oh);
            Assert.True(dst.AsSpan().SequenceEqual(fresh[tag]), $"size-switch[{tag}] differs from a fresh worker");
        }
        Assert.Equal(1, enh.PoolStats.workers);
    }

    [Theory]
    [InlineData("small", 13, 17)]
    [InlineData("pad", 8, 5)]
    public void Pipeline_Strided_Cpu(string tag, int srcPad, int dstPad)
    {
        var (_, _, _, _, oh, ow) = Golden.Dims(tag);
        byte[] tight = Golden.Bgr(tag);
        byte[] expectU8 = Golden.U8(tag);

        int ss = ow * 3 + srcPad, ds = ow * 3 + dstPad;
        byte[] src = new byte[ss * oh], dst = new byte[ds * oh];
        for (int y = 0; y < oh; y++)
            Array.Copy(tight, y * ow * 3, src, y * ss, ow * 3);

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Cpu);
        enh.Enhance(src, ow, oh, ss, dst, ds);

        int maxd = 0;
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow * 3; x++)
                maxd = Math.Max(maxd, Math.Abs(dst[y * ds + x] - expectU8[y * ow * 3 + x]));
        Assert.True(maxd <= 2, $"pipeline-cpu-strided[{tag}] max u8 diff {maxd}");
    }

    [Fact]
    public void Pipeline_Mid_U8()
    {
        var (_, _, _, _, oh, ow) = Golden.Dims("mid");
        byte[] src = Golden.Bgr("mid");
        byte[] expectU8 = Golden.U8("mid");

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Cpu);
        byte[] dst = enh.Enhance(src, ow, oh);

        int maxd = 0;
        for (int i = 0; i < dst.Length; i++) maxd = Math.Max(maxd, Math.Abs(dst[i] - expectU8[i]));
        Assert.True(maxd <= 2, $"pipeline-mid max u8 diff {maxd}");
    }

    [Fact]
    public void Concurrent_Enhance_SameSize()
    {
        // the per-size session pool must return *distinct* sessions to concurrent callers
        var (_, _, _, _, oh, ow) = Golden.Dims("small");
        byte[] src = Golden.Bgr("small");
        byte[] expectU8 = Golden.U8("small");

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Cpu);
        int[] maxDiffs = new int[8];
        Parallel.For(0, 8, t =>
        {
            byte[] dst = enh.Enhance(src, ow, oh);
            int maxd = 0;
            for (int i = 0; i < dst.Length; i++) maxd = Math.Max(maxd, Math.Abs(dst[i] - expectU8[i]));
            maxDiffs[t] = maxd;
        });
        Assert.All(maxDiffs, d => Assert.True(d <= 2, $"concurrent enhance diff {d}"));
    }

    [Fact]
    public void Pipeline_MinSize_32x32()
    {
        int w = 32, h = 32;
        byte[] src = new byte[w * h * 3];
        var rng = new Random(3);
        rng.NextBytes(src);

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Cpu);
        byte[] dst = enh.Enhance(src, w, h);
        Assert.Equal(w * h * 3, dst.Length);
        Assert.All(dst, v => Assert.InRange(v, (byte)0, (byte)255));
    }
}

internal static class RngExt
{
    public static float[] Array(this Random r, int n)
        => Enumerable.Range(0, n).Select(_ => (float)r.NextDouble() * 2f - 1f).ToArray();
}
