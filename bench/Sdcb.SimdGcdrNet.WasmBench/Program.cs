using System.Diagnostics;
using System.Net;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.Wasm;
using System.Text.Json;
using System.Text.Json.Nodes;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Kernels;
using Sdcb.SimdGcdrNet.Kernels;
using Sdcb.SimdOnnx.Onnx;
using Sdcb.SimdGcdrNet.Testing;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Sdcb.SimdGcdrNet.WasmBench;

/// <summary>
/// Browser entry point (driven by index.html / server.mjs). Same modes and output format as
/// bench/Sdcb.SimdGcdrNet.Benchmark, with testdata fetched over HTTP from <c>/testdata/</c>:
/// <c>--corpus [cpu|auto]</c>, <c>--bench h w reps</c>, <c>--benchcpu h w reps</c>, <c>--check [tag]</c>,
/// <c>--tiers [h] [w]</c> (PackedSimd vs Vector&lt;float&gt; in one process). Flags: <c>--vector</c>
/// (Vector&lt;float&gt; tier, PackedSimd off), <c>--count N</c> (corpus: first N timed images),
/// <c>--out path</c> (JSON result, saved by the server), <c>--base-url</c>.
/// </summary>
public static class Program
{
    private static HttpClient s_http = null!;

    public static async Task<int> Main(string[] args)
    {
        string? baseUrl = null, outPath = null;
        int count = 0;
        var rest = new List<string>();
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--base-url" when i + 1 < args.Length: baseUrl = args[++i]; break;
                case "--out" when i + 1 < args.Length: outPath = args[++i]; break;
                case "--count" when i + 1 < args.Length && int.TryParse(args[i + 1], out int n): count = n; i++; break;
                case "--vector": Nhwc.UsePacked = false; break;
                default: rest.Add(args[i]); break;
            }
        }
        s_http = new HttpClient();
        if (baseUrl is not null) s_http.BaseAddress = new Uri(baseUrl);
        s_http.DefaultRequestHeaders.Add("X-WasmBench", "1");

        string isa = Nhwc.UsePacked ? "packedsimd" : Vector.IsHardwareAccelerated ? "vector128" : "scalar";
        Console.WriteLine($"[WASM] {RuntimeInformation.FrameworkDescription} {RuntimeInformation.RuntimeIdentifier}; " +
            $"PackedSimd.IsSupported={PackedSimd.IsSupported} Vector<float>={Vector<float>.Count} " +
            $"(accelerated {Vector.IsHardwareAccelerated}) isa={isa} cpus={Environment.ProcessorCount} " +
            $"threads={Avx2Kernels.Threads} GCDRNET_PROF={Environment.GetEnvironmentVariable("GCDRNET_PROF") ?? "-"}");
        Console.WriteLine($"[WASM] args: {string.Join(' ', args)}");

        var meta = new JsonObject
        {
            ["args"] = string.Join(' ', args),
            ["framework"] = RuntimeInformation.FrameworkDescription,
            ["rid"] = RuntimeInformation.RuntimeIdentifier,
            ["isa"] = isa,
            ["packedSimdSupported"] = PackedSimd.IsSupported,
            ["vectorCount"] = Vector<float>.Count,
            ["processorCount"] = Environment.ProcessorCount,
            ["threads"] = Avx2Kernels.Threads,
            ["timestamp"] = DateTimeOffset.Now.ToString("yyyy-MM-ddTHH:mm:sszzz"),
        };
        var doc = new JsonObject { ["meta"] = meta };
        int rc;
        try
        {
            string mode = rest.Count > 0 ? rest[0] : "--corpus";
            int Arg(int i, int def) => rest.Count > i ? int.Parse(rest[i]) : def;
            rc = mode switch
            {
                "--corpus" => await Corpus(rest.Count > 1 ? rest[1] : "cpu", count, doc),
                "--bench" => await Bench(Arg(1, 1152), Arg(2, 1600), Arg(3, 3), doc),
                "--benchcpu" => await BenchE2e(Arg(1, 1152), Arg(2, 1600), Arg(3, 3), doc),
                "--check" => await Check(rest.Count > 1 ? rest[1] : "small", doc),
                "--tiers" => await Tiers(Arg(1, 384), Arg(2, 512), doc),
                _ => Usage(),
            };
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WASM] FATAL {ex}");
            rc = 99;
        }
        meta["exitCode"] = rc;
        if (outPath is not null)
        {
            using var content = new StringContent(doc.ToJsonString(new JsonSerializerOptions { WriteIndented = true }),
                System.Text.Encoding.UTF8, "application/json");
            using var resp = await s_http.PostAsync("/api/save-benchmark?out=" + Uri.EscapeDataString(outPath), content);
            Console.WriteLine(resp.IsSuccessStatusCode ? $"[WASM] saved {outPath}" : $"[WASM] save failed: {resp.StatusCode}");
            if (!resp.IsSuccessStatusCode && rc == 0) rc = 5;
        }
        Console.WriteLine($"[WASM] exit {rc}");
        return rc;
    }

    private static int Usage()
    {
        Console.WriteLine("usage: --corpus [cpu|auto] | --bench [h] [w] [reps] | --benchcpu [h] [w] [reps] | --check [tag] | --tiers [h] [w]" +
            "  [--vector] [--count N] [--out path]");
        return 1;
    }

    private static async Task<byte[]> Fetch(string rel) =>
        await TryFetch(rel) ?? throw new FileNotFoundException("/testdata/" + rel);

    // browser fetch occasionally fails with "TypeError: Failed to fetch" on a reused keep-alive socket
    private static async Task<byte[]?> TryFetch(string rel)
    {
        for (int attempt = 1; ; attempt++)
        {
            try
            {
                using var resp = await s_http.GetAsync("/testdata/" + rel);
                if (resp.StatusCode == HttpStatusCode.NotFound) return null;
                resp.EnsureSuccessStatusCode();
                return await resp.Content.ReadAsByteArrayAsync();
            }
            catch (HttpRequestException ex) when (attempt < 4)
            {
                Console.WriteLine($"[WASM] fetch {rel} failed ({ex.Message}), retry {attempt}");
                await Task.Delay(200 * attempt);
            }
        }
    }

    private static async Task<(byte[] gc, byte[] dr)> Models()
    {
        long t = Stopwatch.GetTimestamp();
        var gc = Fetch("models/gcdrnet_gcnet_sim.onnx");
        var dr = Fetch("models/gcdrnet_drnet_sim.onnx");
        var r = (await gc, await dr);
        Console.WriteLine($"[WASM] models fetched ({(r.Item1.Length + r.Item2.Length) / 1048576.0:F1} MB) in {Ms(t):F0} ms");
        return r;
    }

    private static double Ms(long t0) => (Stopwatch.GetTimestamp() - t0) * 1000.0 / Stopwatch.Frequency;

    /// <summary>Decodes into a grow-only BGR24 buffer; returns (width, height).</summary>
    private static (int w, int h) DecodeBgr(byte[] encoded, ref byte[]? buf)
    {
        using var img = Image.Load<Bgr24>(encoded);
        int len = img.Width * img.Height * 3;
        if (buf is null || buf.Length < len) buf = new byte[len];
        img.CopyPixelDataTo(MemoryMarshal.Cast<byte, Bgr24>(buf.AsSpan(0, len)));
        return (img.Width, img.Height);
    }

    // same sampled FNV-1a as the desktop --corpus: hashes are comparable across hosts of one ISA tier
    private static ulong Hash(ReadOnlySpan<byte> d)
    {
        ulong h = 1469598103934665603UL;
        for (int i = 0; i < d.Length; i += 977) { h ^= d[i]; h *= 1099511628211UL; }
        return h;
    }

    // RealDAE corpus, desktop --corpus methodology: first image warms up (untimed), every other
    // image is timed once (cold per-size setup included, decode excluded), then a second pass
    // re-times all of them and cross-checks the output hashes.
    private static async Task<int> Corpus(string backend, int count, JsonObject doc)
    {
        var be = backend == "auto" ? GcdrBackend.Auto : GcdrBackend.Cpu;
        using var manifest = JsonDocument.Parse(await Fetch("realdae/manifest.json"));
        var images = manifest.RootElement.GetProperty("images").EnumerateArray()
            .Select(e => (name: e.GetProperty("name").GetString()!, w: e.GetProperty("w").GetInt32(), h: e.GetProperty("h").GetInt32()))
            .ToArray();
        var run = images.Skip(1).ToArray();
        if (count > 0) run = run.Take(count).ToArray();

        var (gcBytes, drBytes) = await Models();
        long t0 = Stopwatch.GetTimestamp();
        using var enh = GcdrEnhancer.FromBytes(gcBytes, drBytes, be);
        Console.WriteLine($"[WASM] engine loaded in {Ms(t0):F0} ms; managed {GC.GetTotalMemory(false) / 1048576.0:F0} MB");

        // compressed bytes stay cached for pass 2 (decoded BGR of 50 images would not fit the wasm heap)
        var encoded = new Dictionary<string, byte[]>();
        byte[]? src = null, dst = null, gt = null;
        {
            var (name, _, _) = images[0];
            var (w, h) = DecodeBgr(await Fetch("realdae/" + name), ref src);
            dst = new byte[src!.Length];
            t0 = Stopwatch.GetTimestamp();
            enh.Enhance(src, w, h, w * 3, dst, w * 3);
            Console.WriteLine($"{name + " (warmup)",-52} {w,5}x{h,-5} {Ms(t0),9:F1} ms");
        }

        var rows = new JsonArray();
        var hashes = new Dictionary<string, ulong>();
        double totalMs = 0, totalMp = 0, psnrSum = 0, managedPeak = 0;
        int psnrN = 0;
        for (int idx = 0; idx < run.Length; idx++)
        {
            var (name, _, _) = run[idx];
            byte[] enc = encoded[name] = await Fetch("realdae/" + name);
            int us = name.IndexOf('_');
            byte[]? gtEnc = us > 0 ? await TryFetch("realdae/" + name[..us] + "_gt.webp") : null;
            var (w, h) = DecodeBgr(enc, ref src);
            int len = w * h * 3;
            if (dst.Length < len) dst = new byte[len];
            Span<byte> o = dst.AsSpan(0, len);
            long t = Stopwatch.GetTimestamp();
            enh.Enhance(src!.AsSpan(0, len), w, h, w * 3, o, w * 3);
            double ms = Ms(t), mp = w * h / 1e6;
            totalMs += ms; totalMp += mp;
            managedPeak = Math.Max(managedPeak, GC.GetTotalMemory(false) / 1048576.0);
            ulong hsh = hashes[name] = Hash(o);
            var row = new JsonObject { ["name"] = name, ["w"] = w, ["h"] = h, ["ms"] = Math.Round(ms, 1), ["hash"] = $"{hsh:x16}" };
            string quality = $"  {hsh:x8}";
            if (gtEnc is not null)
            {
                var (gw, gh) = DecodeBgr(gtEnc, ref gt);
                if (gw == w && gh == h)
                {
                    double se = 0, ae = 0;
                    for (int i = 0; i < len; i++) { int d = o[i] - gt![i]; se += (double)d * d; ae += Math.Abs(d); }
                    double psnr = 10 * Math.Log10(255.0 * 255.0 / (se / len));
                    psnrSum += psnr; psnrN++;
                    row["psnr"] = Math.Round(psnr, 3);
                    quality += $" {psnr,6:F2} dB  mae {ae / len,5:F2}";
                }
            }
            rows.Add((JsonNode)row);
            Console.WriteLine($"[WASM][{idx + 1}/{run.Length}] {name,-40} {w,5}x{h,-5} {mp,5:F2} MP {ms,9:F1} ms{quality}");
        }

        double warmMs = 0;
        int flaky = 0;
        foreach (var (name, _, _) in run)
        {
            var (w, h) = DecodeBgr(encoded[name], ref src);
            int len = w * h * 3;
            Span<byte> o = dst.AsSpan(0, len);
            long t = Stopwatch.GetTimestamp();
            enh.Enhance(src!.AsSpan(0, len), w, h, w * 3, o, w * 3);
            warmMs += Ms(t);
            ulong h2 = Hash(o);
            if (h2 != hashes[name]) { flaky++; Console.WriteLine($"  HASH MISMATCH pass1={hashes[name]:x8} pass2={h2:x8}  {name}"); }
        }
        if (flaky == 0) Console.WriteLine("  pass1/pass2 output hashes all identical");

        Console.WriteLine($"corpus[{backend}] {run.Length} timed images: {totalMs:F1} ms total, " +
            $"{totalMp / (totalMs / 1000):F2} MP/s, mean {totalMs / run.Length:F1} ms");
        Console.WriteLine($"  2nd pass: {warmMs:F1} ms total, {totalMp / (warmMs / 1000):F2} MP/s, mean {warmMs / run.Length:F1} ms " +
            $"(delta = per-size setup {totalMs - warmMs:F0} ms, {100.0 * (totalMs - warmMs) / totalMs:F0}%)");
        if (psnrN > 0) Console.WriteLine($"  psnr mean {psnrSum / psnrN:F2} dB over {psnrN} images");
        var (pools, workers, evicted) = enh.PoolStats;
        Console.WriteLine($"  pools: {pools} pools created ({workers} workers), {evicted} evicted; managed peak {managedPeak:F0} MB");

        doc["images"] = rows;
        doc["summary"] = new JsonObject
        {
            ["backend"] = backend,
            ["timedImages"] = run.Length,
            ["megapixels"] = Math.Round(totalMp, 2),
            ["pass1TotalMs"] = Math.Round(totalMs, 1),
            ["pass1MeanMs"] = Math.Round(totalMs / run.Length, 1),
            ["pass1MpPerSec"] = Math.Round(totalMp / (totalMs / 1000), 3),
            ["pass2TotalMs"] = Math.Round(warmMs, 1),
            ["pass2MeanMs"] = Math.Round(warmMs / run.Length, 1),
            ["psnrMean"] = psnrN > 0 ? Math.Round(psnrSum / psnrN, 3) : null,
            ["psnrImages"] = psnrN,
            ["hashMismatches"] = flaky,
            ["managedPeakMb"] = Math.Round(managedPeak, 1),
        };
        return flaky == 0 ? 0 : 2;
    }

    // gc + dr inference on random input, best of reps; GCDRNET_PROF=1 adds the per-op split
    private static async Task<int> Bench(int h, int w, int reps, JsonObject doc)
    {
        var (gcBytes, drBytes) = await Models();
        var gc = new CompiledModel(GraphCompiler.Compile(OnnxModel.Load(gcBytes), [1, 3, h, w], CompileOptions.Default));
        var dr = new CompiledModel(GraphCompiler.Compile(OnnxModel.Load(drBytes), [1, 6, h, w], CompileOptions.Default));
        using var s1 = gc.CreateSession();
        using var s2 = dr.CreateSession();
        new Random(0).NextBytes(MemoryMarshal.AsBytes(s1.Input));
        new Random(1).NextBytes(MemoryMarshal.AsBytes(s2.Input));
        s1.Run(); s2.Run();
        s1.ResetProfile(); s2.ResetProfile();
        double best1 = double.MaxValue, best2 = double.MaxValue;
        for (int i = 0; i < reps; i++)
        {
            long t = Stopwatch.GetTimestamp(); s1.Run(); best1 = Math.Min(best1, Ms(t));
            t = Stopwatch.GetTimestamp(); s2.Run(); best2 = Math.Min(best2, Ms(t));
        }
        Console.WriteLine($"gc {best1:F1} ms, dr {best2:F1} ms, total {best1 + best2:F1} ms");
        var summary = new JsonObject { ["h"] = h, ["w"] = w, ["reps"] = reps, ["gcMs"] = Math.Round(best1, 1), ["drMs"] = Math.Round(best2, 1) };
        if (Environment.GetEnvironmentVariable("GCDRNET_PROF") == "1")
            foreach (var (tag, ss) in new[] { ("gc", s1), ("dr", s2) })
            {
                Console.WriteLine($"--- {tag} profile ---");
                var prof = new JsonObject();
                foreach (var kv in ss.Profile().OrderByDescending(k => k.Value))
                {
                    double ms = kv.Value.TotalMilliseconds / reps;
                    Console.WriteLine($"  {kv.Key,-16} {ms,8:F1} ms/run");
                    prof[kv.Key.ToString()] = Math.Round(ms, 2);
                }
                summary[tag + "Profile"] = prof;
            }
        doc["summary"] = summary;
        return 0;
    }

    // the real Enhance path (BGR u8 -> u8), best of reps after one warmup
    private static async Task<int> BenchE2e(int h, int w, int reps, JsonObject doc)
    {
        var (gcBytes, drBytes) = await Models();
        using var enh = GcdrEnhancer.FromBytes(gcBytes, drBytes, GcdrBackend.Cpu);
        byte[] src = new byte[w * h * 3], dst = new byte[w * h * 3];
        new Random(0).NextBytes(src);
        enh.Enhance(src, w, h, w * 3, dst, w * 3);
        double best = double.MaxValue;
        for (int i = 0; i < reps; i++)
        {
            long t = Stopwatch.GetTimestamp();
            enh.Enhance(src, w, h, w * 3, dst, w * 3);
            best = Math.Min(best, Ms(t));
        }
        Console.WriteLine($"cpu e2e {w}x{h}: {best:F1} ms");
        doc["summary"] = new JsonObject { ["h"] = h, ["w"] = w, ["reps"] = reps, ["bestMs"] = Math.Round(best, 1) };
        return 0;
    }

    // onnxruntime golden check (tests/testdata/{tag}_*), same tolerance as the desktop --check
    private static async Task<int> Check(string tag, JsonObject doc)
    {
        var (gcBytes, drBytes) = await Models();
        using var manifest = JsonDocument.Parse(await Fetch("manifest.json"));
        int[] Shape(string name) => manifest.RootElement.GetProperty(tag).GetProperty("shapes").GetProperty(name)
            .EnumerateArray().Select(e => e.GetInt32()).ToArray();
        async Task<float[]> F32(string name) => MemoryMarshal.Cast<byte, float>((await Fetch($"{tag}_{name}.f32")).AsSpan()).ToArray();

        float MaxDiff(byte[] model, int[] shape, float[] input, float[] expect)
        {
            using var s = CompiledModel.Compile(OnnxModel.Load(model), shape).CreateSession();
            input.CopyTo(s.Input);
            s.Run();
            float max = 0;
            ReadOnlySpan<float> o = s.Output;
            for (int i = 0; i < o.Length; i++) max = MathF.Max(max, MathF.Abs(o[i] - expect[i]));
            return max;
        }
        // input / dr_in are derived from {tag}_src.png like the tests do (GoldenDerive)
        var dims = manifest.RootElement.GetProperty(tag);
        byte[]? src = null;
        var (ow, oh) = DecodeBgr(await Fetch($"{tag}_src.png"), ref src);
        float[] input = GoldenDerive.Input(src!, oh, ow, dims.GetProperty("ph").GetInt32(), dims.GetProperty("pw").GetInt32());
        float[] shadow = await F32("shadow");
        float gcMax = MaxDiff(gcBytes, Shape("input"), input, shadow);
        float drMax = MaxDiff(drBytes, Shape("dr_in"), GoldenDerive.DrIn(input, shadow), await F32("output"));
        Console.WriteLine($"gcnet[{tag}] max diff = {gcMax:E3}");
        Console.WriteLine($"drnet[{tag}] max diff = {drMax:E3}");
        doc["summary"] = new JsonObject { ["tag"] = tag, ["gcMaxDiff"] = gcMax, ["drMaxDiff"] = drMax };
        return gcMax < 2e-3 && drMax < 2e-3 ? 0 : 3;
    }

    // PackedSimd vs Vector<float> in one process: gc / dr float outputs on u[0,1) input and the
    // e2e u8 output on a random image
    private static async Task<int> Tiers(int h, int w, JsonObject doc)
    {
        bool saved = Nhwc.UsePacked;
        try { return await TiersCore(h, w, doc); }
        finally { Nhwc.UsePacked = saved; }
    }

    private static async Task<int> TiersCore(int h, int w, JsonObject doc)
    {
        if (!PackedSimd.IsSupported)
        {
            Console.WriteLine("PackedSimd not supported: nothing to compare");
            return 1;
        }
        var (gcBytes, drBytes) = await Models();
        var summary = new JsonObject();
        bool ok = true;
        foreach (var (tag, model, c) in new[] { ("gc", gcBytes, 3), ("dr", drBytes, 6) })
        {
            var cm = new CompiledModel(GraphCompiler.Compile(OnnxModel.Load(model), [1, c, h, w], CompileOptions.Default));
            var outs = new float[2][];
            for (int pass = 0; pass < 2; pass++)
            {
                Nhwc.UsePacked = pass == 0;
                using var s = cm.CreateSession();
                var rng = new Random(7);
                Span<float> inp = s.Input;
                for (int i = 0; i < inp.Length; i++) inp[i] = rng.NextSingle();
                s.Run();
                outs[pass] = s.Output.ToArray();
            }
            double maxd = 0, se = 0, ss = 0;
            int same = 0;
            for (int i = 0; i < outs[0].Length; i++)
            {
                double d = Math.Abs(outs[0][i] - outs[1][i]);
                if (outs[0][i] == outs[1][i]) same++;
                maxd = Math.Max(maxd, d); se += d * d; ss += (double)outs[1][i] * outs[1][i];
            }
            double rel = Math.Sqrt(se / Math.Max(ss, 1e-30));
            Console.WriteLine($"{tag} {w}x{h}: max|d| {maxd:E3}  rel-rms {rel:E3}  bit-identical {same}/{outs[0].Length}");
            summary[tag] = new JsonObject { ["maxDiff"] = maxd, ["relRms"] = rel, ["identical"] = same, ["count"] = outs[0].Length };
            ok &= rel < 2e-4;
        }
        byte[] src = new byte[w * h * 3];
        new Random(0).NextBytes(src);
        var dsts = new byte[2][];
        for (int pass = 0; pass < 2; pass++)
        {
            Nhwc.UsePacked = pass == 0;
            using var enh = GcdrEnhancer.FromBytes(gcBytes, drBytes, GcdrBackend.Cpu);
            dsts[pass] = enh.Enhance(src, w, h);
        }
        int diff = 0, maxb = 0;
        for (int i = 0; i < src.Length; i++)
        {
            int d = Math.Abs(dsts[0][i] - dsts[1][i]);
            if (d > 0) diff++;
            maxb = Math.Max(maxb, d);
        }
        Console.WriteLine($"e2e u8 {w}x{h}: {diff}/{src.Length} bytes differ, max {maxb}");
        summary["e2e"] = new JsonObject { ["bytesDiffer"] = diff, ["maxByteDiff"] = maxb };
        doc["summary"] = summary;
        return ok && maxb <= 2 ? 0 : 4;
    }
}
