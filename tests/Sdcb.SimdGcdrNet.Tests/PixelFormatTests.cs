using Sdcb.SimdOnnx.Backends.Metal;
using Sdcb.SimdOnnx.Backends.Vulkan;
using Xunit;

namespace Sdcb.SimdGcdrNet.Tests;

/// <summary>
/// Every <see cref="GcdrPixelFormat"/> must give exactly the BGR24 result of the same backend
/// (channels permuted), copy alpha through, honour strides and leave row padding untouched.
/// </summary>
public class PixelFormatTests
{
    private static string GcPath => Path.Combine(Golden.Dir, "models", "gcdrnet_gcnet_sim.onnx");
    private static string DrPath => Path.Combine(Golden.Dir, "models", "gcdrnet_drnet_sim.onnx");

    public static TheoryData<GcdrBackend, GcdrPixelFormat> Cases()
    {
        var d = new TheoryData<GcdrBackend, GcdrPixelFormat>();
        foreach (GcdrBackend b in new[] { GcdrBackend.Cpu, GcdrBackend.Vulkan, GcdrBackend.Metal })
            foreach (GcdrPixelFormat f in Enum.GetValues<GcdrPixelFormat>())
                d.Add(b, f);
        return d;
    }

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void Format_MatchesBgr24(GcdrBackend backend, GcdrPixelFormat format)
    {
        SkipUnavailable(backend);
        var (_, _, _, _, oh, ow) = Golden.Dims("pad");
        byte[] bgr = Golden.Bgr("pad");
        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, backend);
        byte[] expect = enh.Enhance(bgr, ow, oh);

        int bpp = format is GcdrPixelFormat.Bgra32 or GcdrPixelFormat.Rgba32 ? 4 : 3;
        int ss = ow * bpp + 11, ds = ow * bpp + 7;
        byte[] src = Pack(bgr, ow, oh, format, ss);
        byte[] dst = new byte[ds * oh];
        Array.Fill(dst, (byte)0xA5);
        enh.Enhance(src, ow, oh, ss, dst, ds, format);

        for (int y = 0; y < oh; y++)
        {
            for (int x = 0; x < ow; x++)
            {
                int e = (y * ow + x) * 3, d = y * ds + x * bpp, s = y * ss + x * bpp;
                bool swap = format is GcdrPixelFormat.Rgb24 or GcdrPixelFormat.Rgba32;
                byte b = dst[d + (swap ? 2 : 0)], g = dst[d + 1], r = dst[d + (swap ? 0 : 2)];
                if (b != expect[e] || g != expect[e + 1] || r != expect[e + 2])
                    Assert.Fail($"{backend}/{format} pixel ({x},{y}): got {b},{g},{r} expect {expect[e]},{expect[e + 1]},{expect[e + 2]}");
                if (bpp == 4) Assert.Equal(src[s + 3], dst[d + 3]);
            }
            for (int p = ow * bpp; p < ds; p++)
                Assert.Equal(0xA5, dst[y * ds + p]);
        }
    }

    [SkippableTheory]
    [InlineData(GcdrBackend.Cpu)]
    [InlineData(GcdrBackend.Vulkan)]
    [InlineData(GcdrBackend.Metal)]
    public void InPlace_Rgba32(GcdrBackend backend)
    {
        SkipUnavailable(backend);
        var (_, _, _, _, oh, ow) = Golden.Dims("small");
        byte[] bgr = Golden.Bgr("small");
        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, backend);
        byte[] src = Pack(bgr, ow, oh, GcdrPixelFormat.Rgba32, ow * 4);
        byte[] expect = enh.Enhance(src, ow, oh, GcdrPixelFormat.Rgba32);
        enh.Enhance(src, ow, oh, ow * 4, src, ow * 4, GcdrPixelFormat.Rgba32);
        Assert.Equal(expect, src);
    }

    [Fact]
    public void Enhance_RejectsShortBuffers()
    {
        using var enh = GcdrEnhancer.FromFiles(GcPath, DrPath, GcdrBackend.Cpu);
        byte[] buf = new byte[32 * 32 * 4];
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            enh.Enhance(buf, 32, 32, 32 * 3, buf, 32 * 4, GcdrPixelFormat.Bgra32));
        Assert.Throws<ArgumentException>(() =>
            enh.Enhance(buf, 32, 33, 32 * 4, buf, 32 * 4, GcdrPixelFormat.Bgra32));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            enh.Enhance(buf, 32, 32, 32 * 3, buf, 32 * 3, (GcdrPixelFormat)99));
    }

    /// <summary>Repacks tight BGR24 into <paramref name="format"/> with row stride
    /// <paramref name="stride"/>; alpha gets a per-pixel pattern, row padding 0x5A.</summary>
    private static byte[] Pack(byte[] bgr, int w, int h, GcdrPixelFormat format, int stride)
    {
        int bpp = format is GcdrPixelFormat.Bgra32 or GcdrPixelFormat.Rgba32 ? 4 : 3;
        bool swap = format is GcdrPixelFormat.Rgb24 or GcdrPixelFormat.Rgba32;
        byte[] dst = new byte[stride * h];
        Array.Fill(dst, (byte)0x5A);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                int s = (y * w + x) * 3, d = y * stride + x * bpp;
                dst[d + (swap ? 2 : 0)] = bgr[s];
                dst[d + 1] = bgr[s + 1];
                dst[d + (swap ? 0 : 2)] = bgr[s + 2];
                if (bpp == 4) dst[d + 3] = (byte)(x * 7 + y * 13);
            }
        return dst;
    }

    private static void SkipUnavailable(GcdrBackend backend)
    {
        if (backend == GcdrBackend.Vulkan)
        {
            bool ok;
            try { _ = VkBackend.Default.Dev.DeviceName; ok = true; } catch { ok = false; }
            Skip.IfNot(ok, "no Vulkan device");
        }
        else if (backend == GcdrBackend.Metal)
        {
            bool ok;
            try { ok = GcdrEnhancer.MetalUsable; } catch { ok = false; }
            Skip.IfNot(ok, "no Metal device");
        }
    }
}
