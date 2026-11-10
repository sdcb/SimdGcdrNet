using Sdcb.SimdOnnx.Backends.Vulkan;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;
using Xunit;

namespace Sdcb.SimdGcdrNet.Tests;

public class VulkanTests
{
    private static string GcPath => Path.Combine(Golden.Dir, "models", "gcdrnet_gcnet_sim.onnx");
    private static string DrPath => Path.Combine(Golden.Dir, "models", "gcdrnet_drnet_sim.onnx");

    [SkippableTheory]
    [InlineData("small")]
    [InlineData("pad")]
    [InlineData("mid")]
    public void GcNet_MatchesGolden_Vulkan(string tag)
    {
        Skip.IfNot(VkAvailable());
        float[] input = Golden.F32(tag, "input");
        int[] shape = Golden.ShapeOf(tag, "input");
        float[] expect = Golden.F32(tag, "shadow");

        OnnxModel m = OnnxModel.Load(GcPath);
        using var sess = VkBackend.Default.NewSession(m, shape);
        input.CopyTo(sess.Input);
        sess.Run();
        var (max, rmse, _) = Golden.Diff(sess.Output, expect);
        float tol = VkBackend.Default.F16 ? 2e-2f : 5e-4f;
        Assert.True(max < tol, $"gcnet-vk[{tag}] max diff {max} rmse {rmse} (tol {tol})");
    }

    [SkippableTheory]
    [InlineData("small")]
    [InlineData("pad")]
    [InlineData("mid")]
    public void DrNet_MatchesGolden_Vulkan(string tag)
    {
        Skip.IfNot(VkAvailable());
        float[] input = Golden.F32(tag, "dr_in");
        int[] shape = Golden.ShapeOf(tag, "dr_in");
        float[] expect = Golden.F32(tag, "output");

        OnnxModel m = OnnxModel.Load(DrPath);
        using var sess = VkBackend.Default.NewSession(m, shape);
        input.CopyTo(sess.Input);
        sess.Run();
        var (max, rmse, _) = Golden.Diff(sess.Output, expect);
        float tol = VkBackend.Default.F16 ? 2e-2f : 5e-4f;
        Assert.True(max < tol, $"drnet-vk[{tag}] max diff {max} rmse {rmse} (tol {tol})");
    }

    [SkippableTheory]
    [InlineData("small")]
    [InlineData("pad")]
    [InlineData("mid")]
    public void Pipeline_MatchesGoldenU8_Vulkan(string tag)
    {
        Skip.IfNot(VkAvailable());
        var (_, _, _, _, oh, ow) = Golden.Dims(tag);
        byte[] src = Golden.Bgr(tag);
        byte[] expectU8 = Golden.U8(tag);

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Vulkan);
        byte[] dst = new byte[ow * oh * 3];
        enh.Enhance(src, ow, oh, ow * 3, dst, ow * 3);

        int diffs = 0, maxd = 0;
        for (int i = 0; i < dst.Length; i++)
        {
            int d = Math.Abs(dst[i] - expectU8[i]);
            if (d > 0) diffs++;
            if (d > maxd) maxd = d;
        }
        int u8tol = VkBackend.Default.F16 ? 8 : 2;
        Assert.True(maxd <= u8tol, $"pipeline-vk[{tag}] max u8 diff {maxd} (tol {u8tol}), changed px {diffs}/{dst.Length}");
    }

    [SkippableTheory]
    [InlineData("small", 13, 17)]
    [InlineData("pad", 8, 5)]
    public void Pipeline_Strided_Vulkan(string tag, int srcPad, int dstPad)
    {
        Skip.IfNot(VkAvailable());
        var (_, _, _, _, oh, ow) = Golden.Dims(tag);
        byte[] tight = Golden.Bgr(tag);
        byte[] expectU8 = Golden.U8(tag);

        int ss = ow * 3 + srcPad, ds = ow * 3 + dstPad;
        byte[] src = new byte[ss * oh];
        byte[] dst = new byte[ds * oh];
        for (int y = 0; y < oh; y++)
            Array.Copy(tight, y * ow * 3, src, y * ss, ow * 3);

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Vulkan);
        enh.Enhance(src, ow, oh, ss, dst, ds);

        int maxd = 0;
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow * 3; x++)
                maxd = Math.Max(maxd, Math.Abs(dst[y * ds + x] - expectU8[y * ow * 3 + x]));
        int u8tol = VkBackend.Default.F16 ? 8 : 2;
        Assert.True(maxd <= u8tol, $"pipeline-vk-strided[{tag}] max u8 diff {maxd} (tol {u8tol})");
    }

    [SkippableFact]
    public void Pipeline_SizeSwitching_MatchesFresh_Vulkan()
    {
        // one size-agnostic worker re-records across sizes and reuses (dirty,
        // larger) arenas: every size must match a fresh enhancer byte for byte
        Skip.IfNot(VkAvailable());
        string[] tags = ["mid", "small", "pad", "mid", "small"];
        var fresh = new Dictionary<string, byte[]>();
        foreach (string tag in tags.Distinct())
        {
            var (_, _, _, _, oh, ow) = Golden.Dims(tag);
            using var e = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Vulkan);
            fresh[tag] = e.Enhance(Golden.Bgr(tag), ow, oh);
        }

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Vulkan);
        foreach (string tag in tags)
        {
            var (_, _, _, _, oh, ow) = Golden.Dims(tag);
            byte[] dst = enh.Enhance(Golden.Bgr(tag), ow, oh);
            Assert.True(dst.AsSpan().SequenceEqual(fresh[tag]), $"size-switch[{tag}] differs from a fresh worker");
        }
        Assert.Equal(1, enh.PoolStats.workers);
    }

    private static bool VkAvailable()
    {
        try { _ = VkBackend.Default.Dev.DeviceName; return true; }
        catch { return false; }
    }
}

/// <summary>The register-tiled wide-wave route (<see cref="VkDevice.ConvDot"/>: conv_dot, gemm_dot,
/// convpool_pi, uphead_sc2), forced on whatever device runs the tests: the shaders use no
/// subgroup operations, so any fp16-storage GPU validates their math against the golden data.</summary>
[Collection(TierToggle.Name)]
public class VulkanConvDotTests
{
    private static string GcPath => Path.Combine(Golden.Dir, "models", "gcdrnet_gcnet_sim.onnx");
    private static string DrPath => Path.Combine(Golden.Dir, "models", "gcdrnet_drnet_sim.onnx");

    [SkippableTheory]
    [InlineData("gc", "small", "input", "shadow")]
    [InlineData("gc", "pad", "input", "shadow")]
    [InlineData("gc", "mid", "input", "shadow")]
    [InlineData("dr", "small", "dr_in", "output")]
    [InlineData("dr", "pad", "dr_in", "output")]
    [InlineData("dr", "mid", "dr_in", "output")]
    public void ConvDotRoute_MatchesGolden(string which, string tag, string inName, string outName)
    {
        VkBackend? be = s_dotBackend.Value;
        Skip.If(be is null || !be.F16, "needs a Vulkan device with fp16 storage");

        float[] input = Golden.F32(tag, inName);
        int[] shape = Golden.ShapeOf(tag, inName);
        float[] expect = Golden.F32(tag, outName);
        OnnxModel m = OnnxModel.Load(which == "gc" ? GcPath : DrPath);
        using var sess = be.NewSession(m, shape);
        Assert.Contains(be.GraphFor(m, shape).Dispatches, d => d.Name!.StartsWith("conv_dot", StringComparison.Ordinal));
        input.CopyTo(sess.Input);
        sess.Run();
        var (max, rmse, _) = Golden.Diff(sess.Output, expect);
        Assert.True(max < 2e-2f, $"{which}-convdot[{tag}] max diff {max} rmse {rmse}");
    }

    // a private device with the conv_dot route forced on: flipping it on the shared default device
    // would re-route graphs other tests compile concurrently (byte-exact comparisons then fail)
    private static readonly Lazy<VkBackend?> s_dotBackend = new(() =>
    {
        try
        {
            var be = new VkBackend();
            be.Dev.ConvDot = true;
            return be;
        }
        catch { return null; }
    });
}

/// <summary>Tests that flip a process-wide kernel tier switch must not overlap any other test.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class TierToggle
{
    public const string Name = "TierToggle";
}
