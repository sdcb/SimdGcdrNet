using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;
using Xunit;

namespace Sdcb.SimdGcdrNet.Tests;

public class GoldenTests
{
    private static string GcPath => Path.Combine(Golden.Dir, "models", "gcdrnet_gcnet_sim.onnx");
    private static string DrPath => Path.Combine(Golden.Dir, "models", "gcdrnet_drnet_sim.onnx");

    [Theory]
    [InlineData("small")]
    [InlineData("pad")]
    [InlineData("mid")]
    public void GcNet_MatchesOnnxRuntime(string tag)
    {
        var (_, _, _, _, _, _) = Golden.Dims(tag);
        float[] input = Golden.F32(tag, "input");
        int[] shape = Golden.ShapeOf(tag, "input");
        float[] expect = Golden.F32(tag, "shadow");

        OnnxModel m = OnnxModel.Load(GcPath);
        CompiledModel cm = CompiledModel.Compile(m, shape);
        using var sess = cm.CreateSession();
        input.CopyTo(sess.Input);
        sess.Run();
        var (max, rmse, _) = Golden.Diff(sess.Output, expect);
        Assert.True(max < 1e-4f, $"gcnet[{tag}] max diff {max} rmse {rmse} (shape {string.Join('x', sess.OutputShape)})");
    }

    [Theory]
    [InlineData("small")]
    [InlineData("pad")]
    [InlineData("mid")]
    public void DrNet_MatchesOnnxRuntime(string tag)
    {
        float[] input = Golden.F32(tag, "dr_in");
        int[] shape = Golden.ShapeOf(tag, "dr_in");
        float[] expect = Golden.F32(tag, "output");

        OnnxModel m = OnnxModel.Load(DrPath);
        CompiledModel cm = CompiledModel.Compile(m, shape);
        using var sess = cm.CreateSession();
        input.CopyTo(sess.Input);
        sess.Run();
        var (max, rmse, _) = Golden.Diff(sess.Output, expect);
        Assert.True(max < 1e-4f, $"drnet[{tag}] max diff {max} rmse {rmse}");
    }

    [Theory]
    [InlineData("small")]
    [InlineData("pad")]
    public void Pipeline_MatchesGoldenU8(string tag)
    {
        var (h, w, ph, pw, oh, ow) = Golden.Dims(tag);
        byte[] src = Golden.Bgr(tag);
        byte[] expectU8 = Golden.U8(tag);

        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath);
        byte[] dst = new byte[ow * oh * 3];
        enh.Enhance(src, ow, oh, ow * 3, dst, ow * 3);

        int diffs = 0, maxd = 0;
        for (int i = 0; i < dst.Length; i++)
        {
            int d = Math.Abs(dst[i] - expectU8[i]);
            if (d > 0) diffs++;
            if (d > maxd) maxd = d;
        }
        Assert.True(maxd <= 2, $"pipeline[{tag}] max u8 diff {maxd}, changed px {diffs}/{dst.Length}");
    }
}
