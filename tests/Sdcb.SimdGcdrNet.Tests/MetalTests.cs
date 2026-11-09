using Sdcb.SimdOnnx.Backends.Metal;
using Sdcb.SimdGcdrNet.Backends.Metal;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;
using Xunit;

namespace Sdcb.SimdGcdrNet.Tests;

public class MetalTests
{
    private static string GcPath => Path.Combine(Golden.Dir, "models", "gcdrnet_gcnet_sim.onnx");
    private static string DrPath => Path.Combine(Golden.Dir, "models", "gcdrnet_drnet_sim.onnx");

    [SkippableTheory]
    [InlineData("small")]
    [InlineData("pad")]
    [InlineData("mid")]
    public void GcNet_MatchesGolden_Metal(string tag)
    {
        Skip.IfNot(MetalAvailable());
        float[] input = Golden.F32(tag, "input");
        int[] shape = Golden.ShapeOf(tag, "input");
        float[] expect = Golden.F32(tag, "shadow");

        OnnxModel m = OnnxModel.Load(GcPath);
        using var sess = MetalBackend.Default.NewSession(m, shape);
        input.CopyTo(sess.Input);
        sess.Run();
        var (max, rmse, _) = Golden.Diff(sess.Output, expect);
        float tol = 2e-2f;
        Assert.True(max < tol, $"gcnet-mtl[{tag}] max diff {max} rmse {rmse} (tol {tol})");
    }

    [SkippableTheory]
    [InlineData("small")]
    [InlineData("pad")]
    [InlineData("mid")]
    public void DrNet_MatchesGolden_Metal(string tag)
    {
        Skip.IfNot(MetalAvailable());
        float[] input = Golden.F32(tag, "dr_in");
        int[] shape = Golden.ShapeOf(tag, "dr_in");
        float[] expect = Golden.F32(tag, "output");

        OnnxModel m = OnnxModel.Load(DrPath);
        using var sess = MetalBackend.Default.NewSession(m, shape);
        input.CopyTo(sess.Input);
        sess.Run();
        var (max, rmse, _) = Golden.Diff(sess.Output, expect);
        float tol = 2e-2f;
        Assert.True(max < tol, $"drnet-mtl[{tag}] max diff {max} rmse {rmse} (tol {tol})");
    }

    [SkippableTheory]
    [InlineData("small")]
    [InlineData("pad")]
    [InlineData("mid")]
    public void Pipeline_MatchesGoldenU8_Metal(string tag)
    {
        Skip.IfNot(MetalAvailable());
        var (_, _, _, _, oh, ow) = Golden.Dims(tag);
        byte[] src = Golden.Bgr(tag);
        byte[] expectU8 = Golden.U8(tag);

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Metal);
        byte[] dst = new byte[ow * oh * 3];
        enh.Enhance(src, ow, oh, ow * 3, dst, ow * 3);

        int diffs = 0, maxd = 0;
        for (int i = 0; i < dst.Length; i++)
        {
            int d = Math.Abs(dst[i] - expectU8[i]);
            if (d > 0) diffs++;
            if (d > maxd) maxd = d;
        }
        int u8tol = 8;
        Assert.True(maxd <= u8tol, $"pipeline-mtl[{tag}] max u8 diff {maxd} (tol {u8tol}), changed px {diffs}/{dst.Length}");
    }

    [SkippableTheory]
    [InlineData("small", 13, 17)]
    [InlineData("pad", 8, 5)]
    public void Pipeline_Strided_Metal(string tag, int srcPad, int dstPad)
    {
        Skip.IfNot(MetalAvailable());
        var (_, _, _, _, oh, ow) = Golden.Dims(tag);
        byte[] tight = Golden.Bgr(tag);
        byte[] expectU8 = Golden.U8(tag);

        int ss = ow * 3 + srcPad, ds = ow * 3 + dstPad;
        byte[] src = new byte[ss * oh];
        byte[] dst = new byte[ds * oh];
        for (int y = 0; y < oh; y++)
            Array.Copy(tight, y * ow * 3, src, y * ss, ow * 3);

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Metal);
        enh.Enhance(src, ow, oh, ss, dst, ds);

        int maxd = 0;
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow * 3; x++)
                maxd = Math.Max(maxd, Math.Abs(dst[y * ds + x] - expectU8[y * ow * 3 + x]));
        int u8tol = 8;
        Assert.True(maxd <= u8tol, $"pipeline-mtl-strided[{tag}] max u8 diff {maxd} (tol {u8tol})");
    }

    [SkippableFact]
    public void Pipeline_SizeSwitching_MatchesFresh_Metal()
    {
        // one size-agnostic worker re-records across sizes and reuses (dirty,
        // larger) arenas: every size must match a fresh enhancer byte for byte
        Skip.IfNot(MetalAvailable());
        string[] tags = ["mid", "small", "pad", "mid", "small"];
        var fresh = new Dictionary<string, byte[]>();
        foreach (string tag in tags.Distinct())
        {
            var (_, _, _, _, oh, ow) = Golden.Dims(tag);
            using var e = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Metal);
            fresh[tag] = e.Enhance(Golden.Bgr(tag), ow, oh);
        }

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Metal);
        foreach (string tag in tags)
        {
            var (_, _, _, _, oh, ow) = Golden.Dims(tag);
            byte[] dst = enh.Enhance(Golden.Bgr(tag), ow, oh);
            Assert.True(dst.AsSpan().SequenceEqual(fresh[tag]), $"size-switch[{tag}] differs from a fresh worker");
        }
        Assert.Equal(1, enh.PoolStats.workers);
    }

    private static bool MetalAvailable()
    {
        try { return GcdrEnhancer.MetalUsable; }
        catch { return false; }
    }
}
