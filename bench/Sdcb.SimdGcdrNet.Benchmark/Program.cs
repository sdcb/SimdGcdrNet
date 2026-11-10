using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.SimdOnnx.Engine;
using Sdcb.SimdOnnx.Onnx;

namespace Sdcb.SimdGcdrNet.Benchmark;

public static partial class Program
{
    public static int Main(string[] args)
    {
        string dir = FindTestdata();
        string gcPath = Path.Combine(dir, "models", "gcdrnet_gcnet_sim.onnx");
        string drPath = Path.Combine(dir, "models", "gcdrnet_drnet_sim.onnx");

        if (args.Length > 0 && args[0] == "--ci")
            return Ci(Path.Combine(dir, "realdae"), gcPath, drPath, args);
        if (args.Length > 0 && args[0] == "--check")
        {
            string tag = args.Length > 1 ? args[1] : "small";
            return Check(dir, gcPath, drPath, tag);
        }
        if (args.Length > 0 && args[0] == "--bench")
        {
            int h = args.Length > 1 ? int.Parse(args[1]) : 1152;
            int w = args.Length > 2 ? int.Parse(args[2]) : 1600;
            int reps = args.Length > 3 ? int.Parse(args[3]) : 10;
            return Bench(gcPath, drPath, h, w, reps);
        }
        if (args.Length > 0 && args[0] == "--benchvk")
        {
            int h = args.Length > 1 ? int.Parse(args[1]) : 1152;
            int w = args.Length > 2 ? int.Parse(args[2]) : 1600;
            int reps = args.Length > 3 ? int.Parse(args[3]) : 10;
            return BenchVk(gcPath, drPath, h, w, reps);
        }
        if (args.Length > 0 && args[0] == "--mtrace")
        {
            // per-node CPU-vs-Metal bisect: first node whose output diverges
            string which = args.Length > 1 ? args[1] : "gc";
            int h = args.Length > 2 ? int.Parse(args[2]) : 96;
            int w = args.Length > 3 ? int.Parse(args[3]) : 128;
            string mpath = which == "gc" ? gcPath : drPath;
            var m = OnnxModel.Load(mpath);
            int c = which == "gc" ? 3 : 6;
            // run the SAME (gpu-layout) plan on CPU and Metal so node indices align
            var cpuPlan = GraphCompiler.Compile(m, [1, c, h, w], Sdcb.SimdOnnx.Backends.Metal.MetalBackend.PlanOptions);
            using var cpu = new InferenceSession(cpuPlan);
            var rng = new Random(0);
            if (args.Length > 4)
            {
                var gb = File.ReadAllBytes(args[4]);
                var gf = new float[gb.Length / 4];
                Buffer.BlockCopy(gb, 0, gf, 0, gb.Length);
                gf.CopyTo(cpu.Input);
            }
            else
                for (int i = 0; i < cpu.Input.Length; i++) cpu.Input[i] = rng.NextSingle();
            var mbe = Sdcb.SimdOnnx.Backends.Metal.MetalBackend.Default;
            var mPlan = GraphCompiler.Compile(m, [1, c, h, w], Sdcb.SimdOnnx.Backends.Metal.MetalBackend.PlanOptions);
            using var mtl = mbe.NewSession(m, [1, c, h, w]);
            cpu.Input.CopyTo(mtl.Input);
            mtl.DebugUploadInput();
            if (Environment.GetEnvironmentVariable("MTRACE_NODES") == "1")
            {
                for (int i = 0; i < Math.Max(cpuPlan.Nodes.Length, mPlan.Nodes.Length); i++)
                {
                    string cn = i < cpuPlan.Nodes.Length ? $"{cpuPlan.Nodes[i].Op} {cpuPlan.Nodes[i].Name}" : "-";
                    string mn = i < mPlan.Nodes.Length ? $"{mPlan.Nodes[i].Op} {mPlan.Nodes[i].Name}" : "-";
                    Console.WriteLine($"{i,3}  cpu: {cn,-46}  mtl: {mn}");
                }
                return 0;
            }
            int bad = -1;
            for (int i = 0; i < cpuPlan.Nodes.Length; i++)
            {
                cpu.ExecNode(i);
                mtl.DebugRunNode(i);
                var want = cpu.ValueSpan(cpuPlan.Nodes[i].Out[0]);
                var got = mtl.DebugValue(cpuPlan.Nodes[i].Out[0]);
                if (got.Length != want.Length) Console.WriteLine($"    !! len got {got.Length} want {want.Length}");
                double rmse = 0, mx = 0; int nd = 0, nan = 0, nanw = 0;
                int nl = Math.Min(want.Length, got.Length);
                for (int k = 0; k < nl; k++)
                {
                    if (!float.IsFinite(got[k])) nan++;
                    if (!float.IsFinite(want[k])) nanw++;
                    if (!float.IsFinite(got[k]) || !float.IsFinite(want[k])) continue;
                    double d = Math.Abs(want[k] - got[k]);
                    rmse += d * d;
                    if (d > 1e-3) nd++;
                    if (d > mx) mx = d;
                }
                rmse = Math.Sqrt(rmse / want.Length);
                var nd_ = cpuPlan.Nodes[i];
                string io = $"{nd_.Op} {nd_.Name}";
                if (nd_.P is ConvP cpx)
                    io += $" k{cpx.Kernel[0]}x{cpx.Kernel[1]} s{cpx.Strides[0]}x{cpx.Strides[1]} p[{string.Join(',',cpx.Pads)}] {cpx.PadMode} padv={cpx.PadValue} act={cpx.Act} g={cpx.Group} pool={(cpx.Pool!=null)} w={string.Join('x',mPlan.Values[nd_.In[1]].Shape)} in={string.Join('x',mPlan.Values[nd_.In[0]].PhysShape)} out={string.Join('x',mPlan.Values[nd_.Out[0]].PhysShape)}";
                Console.WriteLine($"[{i,3}] {io,-46} rmse {rmse,10:E3} max {mx,10:E3} nd {nd} nan {nan} nanw {nanw}");
                if (i <= 4 && nd_.P is ConvP)
                {
                    // weight packing check: expected PackConv [K][oc] vs metal const
                    var wv = mPlan.Values[nd_.In[1]];
                    var wsrc = wv.Floats!;
                    int oc_ = wv.Shape[0], ic_ = wv.Shape[1], taps = wv.Shape[2] * wv.Shape[3];
                    var exp = new float[wsrc.Length];
                    for (int o2 = 0; o2 < oc_; o2++) for (int c2 = 0; c2 < ic_; c2++) for (int t = 0; t < taps; t++)
                        exp[(t * ic_ + c2) * oc_ + o2] = wsrc[(o2 * ic_ + c2) * taps + t];
                    var mw = mtl.DebugConst(nd_.In[1], ConstKind.ConvW, wsrc.Length);
                    int wb = -1; for (int k = 0; k < exp.Length; k++) if (Math.Abs(mw[k] - exp[k]) > 1e-3) { wb = k; break; }
                    Console.WriteLine($"    w-check: pack len {exp.Length} off {mtl.DebugConstOff(nd_.In[1], ConstKind.ConvW)} first-diff {wb} head {string.Join(' ', mw.Take(8).Select(x => $"{x:F3}"))} vs exp {string.Join(' ', exp.Take(8).Select(x => $"{x:F3}"))}");
                    if (wb >= 0) Console.WriteLine($"      @{wb}: got {mw[wb]:F4} exp {exp[wb]:F4}; around got {string.Join(' ', mw.Skip(wb).Take(6).Select(x => $"{x:F3}"))} exp {string.Join(' ', exp.Skip(wb).Take(6).Select(x => $"{x:F3}"))}");
                }
                if (bad < 0 && (rmse > 1e-2 || nan > 0 || nanw > 0))
                {
                    bad = i;
                    foreach (var dn in mtl.DebugNodeDispatches(i)) Console.WriteLine($"    disp: {dn}");
                    var sb = new System.Text.StringBuilder();
                    for (int k = 0; k < Math.Min(16, nl); k++) sb.Append($"{want[k]:F4}/{got[k]:F4}  ");
                    Console.WriteLine($"    head: {sb}");
                    // per-channel max diff (NHWC last-dim channels)
                    int ch = mPlan.Values[nd_.Out[0]].PhysShape[^1];
                    if (ch > 0)
                    {
                        var cd = new double[ch];
                        for (int k = 0; k < nl; k++) cd[k % ch] = Math.Max(cd[k % ch], Math.Abs(want[k] - got[k]));
                        Console.WriteLine($"    chmax: {string.Join(' ', cd.Take(Math.Min(ch,32)).Select(v => $"{v:F2}"))}");
                        // worst index
                        double bd = 0; int bi = -1;
                        for (int k = 0; k < nl; k++) { double d = Math.Abs(want[k]-got[k]); if (d > bd) { bd = d; bi = k; } }
                        if (bi >= 0) Console.WriteLine($"    worst@{bi} (pix {bi/ch} ch {bi%ch}): want {want[bi]:F4} got {got[bi]:F4}");
                    }
                }
            }
            Console.WriteLine(bad < 0 ? "all nodes match (rmse<1e-2)" : $"first bad node: {bad}");
            return bad < 0 ? 0 : 1;
        }
        if (args.Length > 0 && args[0] == "--checkgpu")
        {
            // exactness vs CPU (vector<float> baseline): random src -> max |byte diff|
            int h = args.Length > 1 ? int.Parse(args[1]) : 384;
            int w = args.Length > 2 ? int.Parse(args[2]) : 512;
            GcdrBackend be = (args.Length > 3 ? args[3] : "metal") == "metal" ? GcdrBackend.Metal : GcdrBackend.Vulkan;
            using var enhC = GcdrEnhancer.FromFiles(gcPath, drPath, GcdrBackend.Cpu);
            using var enhG = GcdrEnhancer.FromFiles(gcPath, drPath, be);
            byte[] src = new byte[w * h * 3];
            new Random(0).NextBytes(src);
            byte[] dc = new byte[w * h * 3], dg = new byte[w * h * 3];
            enhC.Enhance(src, w, h, w * 3, dc, w * 3);
            enhG.Enhance(src, w, h, w * 3, dg, w * 3);
            int dif = 0, maxd = 0; long sumd = 0;
            for (int i = 0; i < dc.Length; i++)
            {
                int d = Math.Abs(dc[i] - dg[i]);
                if (d > 0) { dif++; sumd += d; }
                if (d > maxd) maxd = d;
            }
            Console.WriteLine($"{be} vs cpu {w}x{h}: diff {dif}/{dc.Length} px, maxd {maxd}, meand {(dif > 0 ? (double)sumd / dif : 0):F3}");
            return maxd > 4 ? 1 : 0;
        }
        if (args.Length > 0 && args[0] == "--benchmtl")
        {
            int h = args.Length > 1 ? int.Parse(args[1]) : 1152;
            int w = args.Length > 2 ? int.Parse(args[2]) : 1600;
            int reps = args.Length > 3 ? int.Parse(args[3]) : 10;
            return BenchE2e(gcPath, drPath, h, w, reps, GcdrBackend.Metal, "metal");
        }
        if (args.Length > 0 && args[0] == "--benchcpu")
        {
            int h = args.Length > 1 ? int.Parse(args[1]) : 1152;
            int w = args.Length > 2 ? int.Parse(args[2]) : 1600;
            int reps = args.Length > 3 ? int.Parse(args[3]) : 10;
            return BenchE2e(gcPath, drPath, h, w, reps, GcdrBackend.Cpu, "cpu");
        }
        if (args.Length > 0 && args[0] == "--node")
        {
            // --node gc|dr h w idx reps : time one node in isolation (after a full warmup run)
            bool isDr = args[1] == "dr";
            int nh = int.Parse(args[2]), nw = int.Parse(args[3]), idx = int.Parse(args[4]), nreps = int.Parse(args[5]);
            var cm = new CompiledModel(GraphCompiler.Compile(OnnxModel.Load(isDr ? drPath : gcPath), [1, isDr ? 6 : 3, nh, nw], CompileOptions.Default));
            var ss = cm.CreateSession();
            new Random(0).NextBytes(MemoryMarshal.AsBytes(ss.Input));
            if (args.Length > 6 && args[6] == "u") { var rr = new Random(0); var inp = ss.Input; for (int i = 0; i < inp.Length; i++) inp[i] = rr.NextSingle(); }
            ss.Run();
            for (int i = 0; i < 3; i++) ss.ExecNode(idx);
            double bestN = double.MaxValue;
            for (int i = 0; i < nreps; i++)
            {
                long t0 = Stopwatch.GetTimestamp();
                ss.ExecNode(idx);
                bestN = Math.Min(bestN, Stopwatch.GetTimestamp() - t0);
            }
            Console.WriteLine($"node #{idx}: best {bestN / Stopwatch.Frequency * 1000:F2} ms");
            return 0;
        }
        if (args.Length > 0 && args[0] == "--vkcheck")
        {
            // Vulkan graph vs CPU graph on the same random input (u[0,1)): max |diff| and RMS
            int ch = args.Length > 1 ? int.Parse(args[1]) : 256;
            int cw = args.Length > 2 ? int.Parse(args[2]) : 384;
            int bad = 0;
            foreach ((string tag, string mp, int cin) in new[] { ("gc", gcPath, 3), ("dr", drPath, 6) })
            {
                var m = OnnxModel.Load(mp);
                var cpu = new CompiledModel(GraphCompiler.Compile(m, [1, cin, ch, cw], CompileOptions.Default)).CreateSession();
                using var vk = Sdcb.SimdOnnx.Backends.Vulkan.VkBackend.Default.NewSession(m, [1, cin, ch, cw]);
                var rng = new Random(7);
                for (int i = 0; i < cpu.Input.Length; i++) cpu.Input[i] = vk.Input[i] = rng.NextSingle();
                cpu.Run(); vk.Run();
                double maxd = 0, se = 0, ss = 0;
                var a = cpu.Output; var b = vk.Output;
                for (int i = 0; i < a.Length; i++)
                {
                    double d = Math.Abs(a[i] - b[i]);
                    maxd = Math.Max(maxd, d); se += d * d; ss += (double)a[i] * a[i];
                }
                double rel = Math.Sqrt(se / Math.Max(ss, 1e-30));
                Console.WriteLine($"{tag} {cw}x{ch}: max|d| {maxd:E3}  rel-rms {rel:E3}");
                if (rel > 5e-3 || double.IsNaN(rel)) bad++;
            }
            return bad == 0 ? 0 : 4;
        }
        if (args.Length > 0 && args[0] == "--vknodes")
        {
            // node-by-node GPU vs CPU on one plan (errors accumulate; a broken kernel shows as a jump)
            bool isDr = args.Length > 1 && args[1] == "dr";
            int nh = args.Length > 2 ? int.Parse(args[2]) : 128, nw = args.Length > 3 ? int.Parse(args[3]) : 192;
            var m = OnnxModel.Load(isDr ? drPath : gcPath);
            int[] shape = [1, isDr ? 6 : 3, nh, nw];
            var plan = GraphCompiler.Compile(m, shape, Sdcb.SimdOnnx.Backends.Vulkan.VkBackend.PlanOptions);
            using var cpu = new InferenceSession(plan);
            using var gpu = Sdcb.SimdOnnx.Backends.Vulkan.VkBackend.Default.NewSession(m, shape);
            var rng = new Random(7);
            for (int i = 0; i < cpu.Input.Length; i++) cpu.Input[i] = gpu.Input[i] = rng.NextSingle();
            gpu.DebugUploadInput();
            var ranges = Sdcb.SimdOnnx.Backends.Vulkan.VkBackend.Default.GraphFor(m, shape).NodeRanges;
            var gd = Sdcb.SimdOnnx.Backends.Vulkan.VkBackend.Default.GraphFor(m, shape).Dispatches;
            for (int i = 0; i < plan.Nodes.Length; i++)
            {
                var nd = plan.Nodes[i];
                cpu.ExecNode(i);
                var (ds, dc) = ranges[i];
                gpu.DebugRunRange(ds, dc);
                int ov = nd.Out[0];
                var v = plan.Values[ov];
                float[] got = new float[v.ElementCount];
                gpu.DebugDownload(ov, got);
                var want = cpu.ValueSpan(ov);
                double maxd = 0, se = 0, ss = 0;
                for (int j = 0; j < got.Length; j++)
                {
                    double d = Math.Abs(got[j] - want[j]);
                    maxd = Math.Max(maxd, d); se += d * d; ss += (double)want[j] * want[j];
                }
                string shaders = string.Join(",", Enumerable.Range(ds, dc).Select(k => gd[k].Name?.Split('(')[0]));
                double relv = Math.Sqrt(se / Math.Max(ss, 1e-30));
                Console.WriteLine($"[{i,3}] {nd.Op,-14} {string.Join('x', v.Shape),-16} rel {relv:E2} max {maxd:E2}  {shaders}");
                if (relv > 0.01 && Environment.GetEnvironmentVariable("VKNODES_DUMP") == "1")
                {
                    int[] ps = v.PhysShape;
                    int pc = ps[^1], pw = ps.Length >= 2 ? ps[^2] : 1;
                    string pp = nd.P is ConvP cpp ? $"k{cpp.Kernel[0]} s{cpp.Strides[0]} g{cpp.Group} pads {string.Join(',', cpp.Pads)} mode {cpp.PadMode} act {cpp.Act} bias {cpp.BiasValue >= 0} add {cpp.AddValue >= 0}" : "";
                    Console.WriteLine($"   phys {string.Join('x', ps)} {pp}");
                    foreach (int iv in nd.In)
                    {
                        var vi = plan.Values[iv];
                        if (vi.Buf < 0) { Console.WriteLine($"   in v{iv} const {string.Join('x', vi.Shape)}"); continue; }
                        float[] gi = new float[vi.ElementCount];
                        gpu.DebugDownload(iv, gi);
                        var ci = cpu.ValueSpan(iv);
                        double se2 = 0, ss2 = 0;
                        for (int j = 0; j < gi.Length; j++) { double d = gi[j] - ci[j]; se2 += d * d; ss2 += (double)ci[j] * ci[j]; }
                        Console.WriteLine($"   in v{iv} {string.Join('x', vi.PhysShape)} buf {vi.Buf} off {plan.BufOffset[vi.Buf]} rel {Math.Sqrt(se2 / Math.Max(ss2, 1e-30)):E2}");
                    }
                    Console.WriteLine($"   out v{ov} buf {v.Buf} off {plan.BufOffset[v.Buf]}");
                    int shown = 0, badN = 0;
                    var badC = new int[pc]; var badX = new int[pw];
                    for (int j = 0; j < got.Length; j++)
                        if (Math.Abs(got[j] - want[j]) > 0.05 + 0.05 * Math.Abs(want[j]))
                        {
                            badN++; badC[j % pc]++; badX[(j / pc) % pw]++;
                            if (shown++ < 12) Console.WriteLine($"   y{j / pc / pw} x{(j / pc) % pw} c{j % pc}: gpu {got[j]:G5} cpu {want[j]:G5}");
                        }
                    Console.WriteLine($"   bad {badN}/{got.Length}; per c: {string.Join(' ', badC.Take(64))}");
                    Console.WriteLine($"   per x: {string.Join(' ', badX.Take(96))}");
                    break;
                }
            }
            return 0;
        }
        if (args.Length > 0 && args[0] == "--trace")
        {
            int h = args.Length > 1 ? int.Parse(args[1]) : 96;
            int w = args.Length > 2 ? int.Parse(args[2]) : 128;
            return Trace(gcPath, drPath, h, w);
        }
        if (args.Length > 0 && args[0] == "--vkprof")
        {
            int h = args.Length > 1 ? int.Parse(args[1]) : 1152;
            int w = args.Length > 2 ? int.Parse(args[2]) : 1600;
            return VkProf(gcPath, drPath, h, w);
        }
        if (args.Length > 0 && args[0] == "--mtlprof")
        {
            int h = args.Length > 1 ? int.Parse(args[1]) : 1152;
            int w = args.Length > 2 ? int.Parse(args[2]) : 1600;
            var be = Sdcb.SimdOnnx.Backends.Metal.MetalBackend.Default;
            foreach ((string tag, string mp, int cin) in new[] { ("gc", gcPath, 3), ("dr", drPath, 6) })
            {
                var m = OnnxModel.Load(mp);
                using var s = be.NewSession(m, [1, cin, h, w]);
                s.Run(); // warm
                var agg = s.DebugProfile();
                Console.WriteLine($"--- {tag} metal per-shader (total {agg.Values.Sum():F1} ms) ---");
                foreach (var kv in agg.OrderByDescending(k => k.Value)) Console.WriteLine($"  {kv.Key,-14} {kv.Value,8:F2} ms");
            }
            return 0;
        }
        if (args.Length > 0 && args[0] == "--corpus")
        {
            string backend = args.Length > 1 ? args[1] : "auto";
            if (!File.Exists(Path.Combine(dir, "realdae", "manifest.json")))
            {
                Console.Error.WriteLine("RealDAE corpus not found: run tools/fetch-fixture.ps1");
                return 4;
            }
            return Corpus(gcPath, drPath, Path.Combine(dir, "realdae"), backend);
        }
        if (args.Length > 0 && args[0] == "--flip")
        {
            // nondeterminism hunt: enhance one image repeatedly while interleaving
            // another size to force pool eviction + pipeline recreation.
            return Flip(gcPath, drPath, Path.Combine(dir, "realdae"),
                args.Length > 1 ? args[1] : "18_shadow.webp");
        }
        if (args.Length > 0 && args[0] == "--analyze")
        {
            int ah = args.Length > 1 ? int.Parse(args[1]) : 2592;
            int aw = args.Length > 2 ? int.Parse(args[2]) : 2464;
            Analyze(gcPath, drPath, ah, aw);
            return 0;
        }
        if (args.Length > 0 && args[0] == "--vkdup")
        {
            int dh = args.Length > 1 ? int.Parse(args[1]) : 2560;
            int dw = args.Length > 2 ? int.Parse(args[2]) : 2464;
            int dr_ = args.Length > 3 ? int.Parse(args[3]) : 0;
            return VkDup(gcPath, drPath, dh, dw, dr_ != 0);
        }
        if (args.Length > 0 && args[0] == "--planchk")
        {
            // do two Compile() calls produce byte-identical buf layouts?
            for (int rep = 0; rep < 3; rep++)
            {
                var m = OnnxModel.Load(gcPath);
                int[] sh = [1, 3, 2592, 2464];
                var p1 = GraphCompiler.Compile(m, sh);
                var p2 = GraphCompiler.Compile(m, sh);
                int bad = 0;
                for (int i = 0; i < p1.BufOffset.Length; i++)
                    if (p1.BufOffset[i] != p2.BufOffset[i]) bad++;
                int nb = 0;
                for (int i = 0; i < p1.Nodes.Length; i++)
                    if (!p1.Nodes[i].In.AsSpan().SequenceEqual(p2.Nodes[i].In) ||
                        !p1.Nodes[i].Out.AsSpan().SequenceEqual(p2.Nodes[i].Out)) nb++;
                Console.WriteLine($"rep{rep}: bufs={p1.BufOffset.Length} diffOff={bad} arena={p1.ArenaFloats}/{p2.ArenaFloats} nodeDiff={nb} " +
                    $"inOff={p1.BufOffset[p1.Values[p1.InputValue].Buf]}/{p2.BufOffset[p2.Values[p2.InputValue].Buf]} " +
                    $"outOff={p1.BufOffset[p1.Values[p1.OutputValue].Buf]}/{p2.BufOffset[p2.Values[p2.OutputValue].Buf]}");
            }
            return 0;
        }
        Console.WriteLine("usage: --ci smoke|bench [options] | --check [tag] | --bench [h] [w] [reps] | --node gc|dr h w idx reps [u] | --benchvk [h] [w] [reps] | --vkprof [h] [w] | --vkcheck [h] [w] | --vknodes gc|dr [h] [w] | --corpus [cpu|vulkan|auto] | --flip [img] | --analyze [h] [w] | --vkdup [h] [w] [dr]");
        return 1;
    }

    // determinism probe on a standalone VkInferenceSession: same input, run twice,
    // destroy, recreate — any hash change = stale-read/nondeterminism at session level.
    private static int VkDup(string gcPath, string drPath, int h, int w, bool useDr)
    {
        var be = Sdcb.SimdOnnx.Backends.Vulkan.VkBackend.Default;
        string path = useDr ? drPath : gcPath;
        int c = useDr ? 6 : 3;
        var m = OnnxModel.Load(path);
        static ulong Fhash(ReadOnlySpan<float> d)
        {
            ulong hsh = 1469598103934665603UL;
            foreach (float f in d) { hsh ^= BitConverter.SingleToUInt32Bits(f); hsh *= 1099511628211UL; }
            return hsh;
        }
        for (int round = 0; round < 4; round++)
        {
            using var s = be.NewSession(m, [1, c, h, w]);
            var rng = new Random(42);
            for (int i = 0; i < s.Input.Length; i++) s.Input[i] = rng.NextSingle();
            s.Run();
            Console.WriteLine($"session#{round} run1 hash={Fhash(s.Output):x16}");
            s.Run();
            Console.WriteLine($"session#{round} run2 hash={Fhash(s.Output):x16}");
            // interleave: run smaller sessions to churn memory
            foreach (int cs in new[] { 1024, 1312, 1152 })
            {
                using var s2 = be.NewSession(m, [1, c, cs, cs]);
                new Random(0).NextBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(s2.Input));
                s2.Run();
            }
        }
        return 0;
    }

    // read-before-write audit: replay the plan's arena writes in node order and flag
    // any input value whose byte range isn't fully covered by prior writes.
    private static void Analyze(string gcPath, string drPath, int h, int w)
    {
        foreach (var (path, c) in new[] { (gcPath, 3), (drPath, 6) })
        {
            var plan = GraphCompiler.Compile(OnnxModel.Load(path), [1, c, h, w], CompileOptions.Default);
            var written = new List<(long s, long e)>();
            bool Covered(long s, long e) =>
                written.Any(r => r.s <= s && r.e >= e);
            // graph input counts as written by the host copy
            var iv = plan.Values[plan.InputValue];
            if (iv.Buf >= 0)
                written.Add((plan.BufOffset[iv.Buf] * 4L, plan.BufOffset[iv.Buf] * 4L + iv.Shape.Aggregate(1L, (a, x) => a * x) * 4));
            int flags = 0;
            for (int i = 0; i < plan.Nodes.Length; i++)
            {
                var nd = plan.Nodes[i];
                foreach (int vid in nd.In)
                {
                    var v = plan.Values[vid];
                    if (v.Buf < 0 || v.Shape.Length == 0) continue;
                    long s = plan.BufOffset[v.Buf] * 4L, e = s + v.Shape.Aggregate(1L, (a, x) => a * x) * 4;
                    if (!Covered(s, e))
                    {
                        flags++;
                        Console.WriteLine($"  read-unwritten node[{i}] {nd.Op} {nd.Name} vid={vid} buf={v.Buf} [{s},{e}) shape={string.Join('x', v.Shape)}");
                    }
                }
                foreach (int vid in nd.Out)
                {
                    var v = plan.Values[vid];
                    if (v.Buf < 0 || v.Shape.Length == 0) continue;
                    long s = plan.BufOffset[v.Buf] * 4L, e = s + v.Shape.Aggregate(1L, (a, x) => a * x) * 4;
                    if (!Covered(s, e)) written.Add((s, e));
                }
            }
            Console.WriteLine($"{Path.GetFileName(path)} [{c}x{h}x{w}]: {flags} unwritten-read flags, arena {plan.ArenaFloats * 4L / 1048576.0:F0} MB");
        }
    }

    // bisect nondeterminism: two same-size pipelines with churn pipelines created
    // between them; hash gc output + dr input + final dst per run to see which
    // stage first diverges.
    private static int Flip(string gcPath, string drPath, string corpusDir, string imgName)
    {
        var manifest = System.Text.Json.JsonDocument.Parse(
            File.ReadAllText(Path.Combine(corpusDir, "manifest.json")));
        var images = manifest.RootElement.GetProperty("images").EnumerateArray()
            .Select(e => (name: e.GetProperty("name").GetString()!,
                          w: e.GetProperty("w").GetInt32(),
                          h: e.GetProperty("h").GetInt32())).ToArray();
        var a = images.First(i => i.name == imgName);
        var others = images.Where(i => i.name != imgName && (long)i.w * i.h > 3_000_000)
            .OrderBy(i => (long)i.w * i.h).ToArray();
        byte[] bgrA = DecodeBgr(Path.Combine(corpusDir, a.name));
        var bgrO = others.Select(o => (o, DecodeBgr(Path.Combine(corpusDir, o.name)))).ToArray();

        static ulong Fnv(byte[] d)
        {
            ulong hsh = 1469598103934665603UL;
            for (int i = 0; i < d.Length; i += 977) { hsh ^= d[i]; hsh *= 1099511628211UL; }
            return hsh;
        }

        using var enh = GcdrEnhancer.FromFiles(gcPath, drPath, GcdrBackend.Vulkan);
        byte[] dst = new byte[bgrA.Length], tmp = new byte[bgrA.Length];
        // CPU reference (slow, one shot) — the oracle-correct output
        using var cpu = GcdrEnhancer.FromFiles(gcPath, drPath, GcdrBackend.Cpu);
        byte[] dstCpu = new byte[bgrA.Length];
        cpu.Enhance(bgrA, a.w, a.h, a.w * 3, dstCpu, a.w * 3);
        Console.WriteLine($"cpu-ref       hash={Fnv(dstCpu):x16}");
        // pass 1: same pipeline twice (pool hit) — should be identical
        for (int k = 0; k < 2; k++)
        {
            enh.Enhance(bgrA, a.w, a.h, a.w * 3, dst, a.w * 3);
            Console.WriteLine($"same-pipe[{k}] hash={Fnv(dst):x16} firstDiff={FirstDiff(dst, dstCpu)}");
        }
        // pass 2: evict between runs — fresh pipeline each time, varying predecessor sizes
        for (int k = 0; k < Math.Min(bgrO.Length, 20); k++)
        {
            var (o, bgr) = bgrO[k % bgrO.Length];
            byte[] t2 = tmp.Length == bgr.Length ? tmp : new byte[bgr.Length];
            enh.Enhance(bgr, o.w, o.h, o.w * 3, t2, o.w * 3); // evicts a's pool (size differs)
            enh.Enhance(bgrA, a.w, a.h, a.w * 3, dst, a.w * 3);
            Console.WriteLine($"fresh-pipe[{k}] after {o.name[..Math.Min(20, o.name.Length)]} hash={Fnv(dst):x16} firstDiff={FirstDiff(dst, dstCpu)}");
        }
        return 0;

        static int FirstDiff(byte[] x, byte[] y)
        {
            for (int i = 0; i < x.Length; i++) if (x[i] != y[i]) return i;
            return -1;
        }
    }

    // Iterate the RealDAE corpus (manifest.json + *.webp; *_gt.webp for PSNR when present).
    // First image warms up global init (model load/JIT); every remaining image is timed once —
    // cold e2e including per-size setup (plans, graphs, buffer growth), excluding decode.
    private static int Corpus(string gcPath, string drPath, string corpusDir, string backend)
    {
        var be = backend.ToLowerInvariant() switch
        {
            "cpu" => GcdrBackend.Cpu,
            "vulkan" or "vk" => GcdrBackend.Vulkan,
            "metal" or "mtl" => GcdrBackend.Metal,
            _ => GcdrBackend.Auto,
        };
        var manifest = System.Text.Json.JsonDocument.Parse(
            File.ReadAllText(Path.Combine(corpusDir, "manifest.json")));
        // manifest order (01..50): sizes interleave like real workloads — pool
        // eviction/churn is exercised instead of a sorted best case.
        var images = manifest.RootElement.GetProperty("images").EnumerateArray()
            .Select(e => (name: e.GetProperty("name").GetString()!,
                          w: e.GetProperty("w").GetInt32(),
                          h: e.GetProperty("h").GetInt32()))
            .ToArray();

        // Working set (MB, Refresh()ed): sampled before/after engine init and after every
        // Enhance (warmup + pass1 + pass2). ΔWS = last pass1 image − loaded; peak = max sample.
        var proc = System.Diagnostics.Process.GetCurrentProcess();
        double WsMb() { proc.Refresh(); return proc.WorkingSet64 / 1048576.0; }
        double wsInit0 = WsMb();
        using var enh = GcdrEnhancer.FromFiles(gcPath, drPath, be);
        double wsLoaded = WsMb(), wsPeak = wsLoaded;
        Console.WriteLine($"engine loaded: ws {wsInit0:F1} -> {wsLoaded:F1} MB ({wsLoaded - wsInit0:+0.0;-0.0} MB)");
        {   // global warmup: model load, JIT, first pipeline — untimed
            var (name, w, h) = images[0];
            byte[] bgr = DecodeBgr(Path.Combine(corpusDir, name));
            enh.Enhance(bgr, w, h, w * 3, new byte[bgr.Length], w * 3);
            double ws = WsMb(); wsPeak = Math.Max(wsPeak, ws);
            Console.WriteLine($"{name + " (warmup)",-52} {w,5}x{h,-5} ws {ws,7:F1} MB");
        }

        Sdcb.SimdOnnx.Backends.Vulkan.VkDevice.ResetPeakDeviceBytes();
        long gc0 = GC.GetTotalMemory(true);
        double wsLast = wsLoaded;
        var hashes = new Dictionary<string, ulong>();
        double totalMs = 0, totalMp = 0, psnrSum = 0; int psnrN = 0;
        int from = int.TryParse(Environment.GetEnvironmentVariable("GCDR_FROM"), out int f0) ? f0 : 1;
        string? only = Environment.GetEnvironmentVariable("GCDR_ONLY");
        var run = images.Skip(from).Where(i => only == null || i.name.Contains(only)).ToArray();
        // grow-only image buffers: per-image LOH garbage would otherwise pile up in the
        // working set (~45 MB per 5 MP image until a gen2 GC) and drown the engine's share
        byte[]? srcBuf = null, dstBuf = null, gtBuf = null;
        foreach ((string name, int w, int h) in run)
        {
            int len = DecodeBgr(Path.Combine(corpusDir, name), ref srcBuf);
            Span<byte> bgr = srcBuf.AsSpan(0, len), dst = Grow(ref dstBuf, len);
            long t = Stopwatch.GetTimestamp();
            enh.Enhance(bgr, w, h, w * 3, dst, w * 3);
            double ms = (Stopwatch.GetTimestamp() - t) * 1000.0 / Stopwatch.Frequency;
            double mp = w * h / 1e6;
            totalMs += ms; totalMp += mp;
            wsLast = WsMb(); wsPeak = Math.Max(wsPeak, wsLast);
            ulong hsh = 1469598103934665603UL;
            for (int i = 0; i < dst.Length; i += 977) { hsh ^= dst[i]; hsh *= 1099511628211UL; }
            hashes[name] = hsh;
            string? dumpDir = Environment.GetEnvironmentVariable("GCDR_DUMP");
            if (dumpDir != null)
                File.WriteAllBytes(Path.Combine(dumpDir, name.Replace(".webp", ".bgr")), dst.ToArray());
            string quality = $"  {hsh:x8}";
            string gt = Path.Combine(corpusDir, name[..name.IndexOf('_')] + "_gt.webp");
            if (File.Exists(gt))
            {
                int glen = DecodeBgr(gt, ref gtBuf);
                ReadOnlySpan<byte> refBgr = gtBuf.AsSpan(0, glen);
                if (refBgr.Length == dst.Length)
                {
                    double se = 0, ae = 0;
                    for (int i = 0; i < dst.Length; i++)
                    { int d = dst[i] - refBgr[i]; se += (double)d * d; ae += Math.Abs(d); }
                    double psnr = 10 * Math.Log10(255.0 * 255.0 / (se / dst.Length));
                    psnrSum += psnr; psnrN++;
                    quality += $" {psnr,6:F2} dB  mae {ae / dst.Length,5:F2}";
                }
            }
            Console.WriteLine($"{name,-52} {w,5}x{h,-5} {mp,5:F2} MP {ms,9:F1} ms{quality}  ws {wsLast,7:F1} MB");
        }
        long gcD = GC.GetTotalMemory(true) - gc0;

        // second pass: measures pool-hit throughput AND cross-checks output hashes
        double warmMs = 0; int flaky = 0;
        foreach ((string name, int w, int h) in run)
        {
            int len = DecodeBgr(Path.Combine(corpusDir, name), ref srcBuf);
            Span<byte> bgr = srcBuf.AsSpan(0, len), dst = Grow(ref dstBuf, len);
            long t = Stopwatch.GetTimestamp();
            enh.Enhance(bgr, w, h, w * 3, dst, w * 3);
            warmMs += (Stopwatch.GetTimestamp() - t) * 1000.0 / Stopwatch.Frequency;
            wsPeak = Math.Max(wsPeak, WsMb());
            ulong h2 = 1469598103934665603UL;
            for (int i = 0; i < dst.Length; i += 977) { h2 ^= dst[i]; h2 *= 1099511628211UL; }
            if (h2 != hashes[name]) { flaky++; Console.WriteLine($"  HASH MISMATCH pass1={hashes[name]:x8} pass2={h2:x8}  {name}"); }
        }
        if (flaky == 0) Console.WriteLine("  pass1/pass2 output hashes all identical");

        Console.WriteLine($"corpus[{backend}] {run.Length} timed images: {totalMs:F1} ms total, " +
            $"{totalMp / (totalMs / 1000):F2} MP/s, mean {totalMs / run.Length:F1} ms");
        Console.WriteLine($"  2nd pass: {warmMs:F1} ms total, {totalMp / (warmMs / 1000):F2} MP/s " +
            $"(delta = per-size setup {totalMs - warmMs:F0} ms, {100.0 * (totalMs - warmMs) / totalMs:F0}%)");
        if (psnrN > 0) Console.WriteLine($"  psnr mean {psnrSum / psnrN:F2} dB over {psnrN} images");
        var (pools, workers, evicted) = enh.PoolStats;
        Console.WriteLine($"  pools: {pools} pools created ({workers} workers), {evicted} evicted");
        Console.WriteLine($"  ws: init {wsInit0:F0} -> loaded {wsLoaded:F0} MB ({wsLoaded - wsInit0:+0;-0}), " +
            $"loaded -> last {wsLast:F0} MB (delta {wsLast - wsLoaded:+0;-0} MB), peak {wsPeak:F0} MB");
        Console.WriteLine($"  mem: managed {gcD / 1048576.0:+0;-0} MB (after forced GC), " +
            $"vkDevice live {Sdcb.SimdOnnx.Backends.Vulkan.VkDevice.LiveDeviceBytes / 1048576.0:F0} MB " +
            $"(peak {Sdcb.SimdOnnx.Backends.Vulkan.VkDevice.PeakDeviceBytes / 1048576.0:F0} MB)");
        return 0;
    }

    private static byte[] DecodeBgr(string path)
    {
        byte[]? buf = null;
        int len = DecodeBgr(path, ref buf);
        return buf!.Length == len ? buf : buf[..len];
    }

    private static Span<byte> Grow(ref byte[]? buf, int len)
    {
        if (buf is null || buf.Length < len) buf = new byte[len];
        return buf.AsSpan(0, len);
    }

    /// <summary>Decodes into a grow-only buffer; returns the BGR byte count.</summary>
    private static int DecodeBgr(string path, ref byte[]? buf)
    {
#if ANDROID
        using var bmp = Android.Graphics.BitmapFactory.DecodeFile(path)
            ?? throw new InvalidDataException($"cannot decode {path}");
        int w = bmp.Width, h = bmp.Height;
        var bgr = Grow(ref buf, w * h * 3);
        int[] row = new int[w];
        for (int y = 0; y < h; y++)
        {
            bmp.GetPixels(row, 0, w, 0, y, w, 1);
            int d = y * w * 3;
            foreach (int argb in row)
            {
                bgr[d++] = (byte)argb; bgr[d++] = (byte)(argb >> 8); bgr[d++] = (byte)(argb >> 16);
            }
        }
        return w * h * 3;
#else
        using var decoded = SkiaSharp.SKBitmap.Decode(path) ?? throw new InvalidDataException($"cannot decode {path}");
        // the platform color type is RGBA on Linux: force BGRA so the raw bytes below are B,G,R
        using var bmp = decoded.ColorType == SkiaSharp.SKColorType.Bgra8888 ? decoded.Copy()
            : decoded.Copy(SkiaSharp.SKColorType.Bgra8888) ?? throw new InvalidDataException($"cannot convert {path}");
        int w = bmp.Width, h = bmp.Height;
        var bgr = Grow(ref buf, w * h * 3);
        IntPtr px = bmp.GetPixels();
        int rb = bmp.RowBytes;
        unsafe
        {
            for (int y = 0; y < h; y++)
            {
                byte* s = (byte*)px + y * rb;
                int d = y * w * 3;
                for (int x = 0; x < w; x++, s += 4, d += 3)
                {
                    bgr[d] = s[0]; bgr[d + 1] = s[1]; bgr[d + 2] = s[2];
                }
            }
        }
        return w * h * 3;
#endif
    }

    private static (int W, int H) ImageSize(string path)
    {
#if ANDROID
        var o = new Android.Graphics.BitmapFactory.Options { InJustDecodeBounds = true };
        Android.Graphics.BitmapFactory.DecodeFile(path, o);
        return (o.OutWidth, o.OutHeight);
#else
        using var codec = SkiaSharp.SKCodec.Create(path);
        return (codec.Info.Width, codec.Info.Height);
#endif
    }

    private static int Check(string dir, string gcPath, string drPath, string tag)
    {
        // input / dr_in are derived from {tag}_src.png like the tests do (GoldenDerive)
        var dims = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "manifest.json"))).RootElement.GetProperty(tag);
        int oh = dims.GetProperty("orig_h").GetInt32(), ow = dims.GetProperty("orig_w").GetInt32();
        int ph = dims.GetProperty("ph").GetInt32(), pw = dims.GetProperty("pw").GetInt32();
        float[] input = Testing.GoldenDerive.Input(DecodeBgr(Path.Combine(dir, $"{tag}_src.png")), oh, ow, ph, pw);
        int[] shape = ReadShape(dir, tag, "input");
        float[] expect = ToF32(File.ReadAllBytes(Path.Combine(dir, $"{tag}_shadow.f32")));

        var m = OnnxModel.Load(gcPath);
        var cm = CompiledModel.Compile(m, shape);
        Console.WriteLine($"gcnet: {cm.NodeCount} nodes, arena {cm.ArenaFloats * 4L / 1024.0 / 1024:F1} MB");
        var sess = cm.CreateSession();
        input.CopyTo(sess.Input);
        try { sess.Run(); }
        catch (Exception ex)
        {
            Console.WriteLine($"RUN FAILED: {ex}");
            return 2;
        }
        float max = 0; int argmax = 0;
        var outp = sess.Output;
        for (int i = 0; i < outp.Length; i++)
        {
            float d = MathF.Abs(outp[i] - expect[i]);
            if (d > max) { max = d; argmax = i; }
        }
        Console.WriteLine($"gcnet[{tag}] max diff = {max:E3} at {argmax} (got {outp[argmax]}, want {expect[argmax]})");

        // drnet
        float[] drIn = Testing.GoldenDerive.DrIn(input, expect);
        float[] drExpect = ToF32(File.ReadAllBytes(Path.Combine(dir, $"{tag}_output.f32")));
        var m2 = OnnxModel.Load(drPath);
        var cm2 = CompiledModel.Compile(m2, ReadShape(dir, tag, "dr_in"));
        var sess2 = cm2.CreateSession();
        drIn.CopyTo(sess2.Input);
        sess2.Run();
        var o2 = sess2.Output;
        float max2 = 0;
        for (int i = 0; i < o2.Length; i++) max2 = MathF.Max(max2, MathF.Abs(o2[i] - drExpect[i]));
        Console.WriteLine($"drnet[{tag}] max diff = {max2:E3}");
        return max < 2e-3 && max2 < 2e-3 ? 0 : 3;
    }

    // run each node with progress printed — for hunting hangs at odd sizes
    private static int Trace(string gcPath, string drPath, int h, int w)
    {
        foreach (var (path, c) in new[] { (gcPath, 3), (drPath, 6) })
        {
            var plan = GraphCompiler.Compile(OnnxModel.Load(path), [1, c, h, w], CompileOptions.Default);
            long maxT = plan.Values.Max(v => v.Shape.Aggregate(1L, (a, x) => a * x));
            Console.WriteLine($"max tensor {maxT * 4 / 1048576.0:F0} MB, arena {plan.ArenaFloats * 4L / 1048576.0:F0} MB");
            using var s = new InferenceSession(plan);
            new Random(0).NextBytes(MemoryMarshal.AsBytes(s.Input));
            Console.WriteLine($"== {Path.GetFileName(path)} [{c}x{h}x{w}] {plan.Nodes.Length} nodes ==");
            for (int i = 0; i < plan.Nodes.Length; i++)
            {
                var nd = plan.Nodes[i];
                string io = $"{string.Join('x', plan.Values[nd.In[0]].Shape)}->{string.Join('x', plan.Values[nd.Out[0]].Shape)}";
                if (nd.P is ConvP cp) io += $" w={string.Join('x', plan.Values[nd.In[1]].Shape)} g={cp.Group}";
                Console.Write($"[{i,3}] {nd.Op} {nd.Name} {io}  ");
                Console.Out.Flush();
                long t = Stopwatch.GetTimestamp();
                s.ExecNode(i);
                Console.WriteLine($"{(Stopwatch.GetTimestamp() - t) * 1000.0 / Stopwatch.Frequency:F1} ms");
                Console.Out.Flush();
            }
        }
        return 0;
    }

    private static int Bench(string gcPath, string drPath, int h, int w, int reps)
    {
        var gc = new CompiledModel(GraphCompiler.Compile(OnnxModel.Load(gcPath), [1, 3, h, w], CompileOptions.Default));
        var dr = new CompiledModel(GraphCompiler.Compile(OnnxModel.Load(drPath), [1, 6, h, w], CompileOptions.Default));
        var s1 = gc.CreateSession(); var s2 = dr.CreateSession();
        new Random(0).NextBytes(MemoryMarshal.AsBytes(s1.Input));
        new Random(1).NextBytes(MemoryMarshal.AsBytes(s2.Input));
        // warmup
        s1.Run(); s2.Run();
        s1.ResetProfile(); s2.ResetProfile();
        double best1 = double.MaxValue, best2 = double.MaxValue;
        for (int i = 0; i < reps; i++)
        {
            long t = Stopwatch.GetTimestamp(); s1.Run();
            best1 = Math.Min(best1, Stopwatch.GetTimestamp() - t);
            t = Stopwatch.GetTimestamp(); s2.Run();
            best2 = Math.Min(best2, Stopwatch.GetTimestamp() - t);
        }
        Console.WriteLine($"gc {best1 / Stopwatch.Frequency * 1000:F1} ms, dr {best2 / Stopwatch.Frequency * 1000:F1} ms, total {(best1 + best2) / Stopwatch.Frequency * 1000:F1} ms");
        if (Environment.GetEnvironmentVariable("GCDRNET_PROF") == "1")
        {
            Console.WriteLine("--- gc profile ---");
            foreach (var kv in s1.Profile().OrderByDescending(k => k.Value)) Console.WriteLine($"  {kv.Key,-16} {kv.Value.TotalMilliseconds / reps,8:F1} ms/run");
            Console.WriteLine("--- dr profile ---");
            foreach (var kv in s2.Profile().OrderByDescending(k => k.Value)) Console.WriteLine($"  {kv.Key,-16} {kv.Value.TotalMilliseconds / reps,8:F1} ms/run");
            if (Environment.GetEnvironmentVariable("GCDRNET_PROF_NODES") == "1")
                foreach (var (tag, ss) in new[] { ("gc", s1), ("dr", s2) })
                {
                    Console.WriteLine($"--- {tag} top nodes (ms/run) ---");
                    foreach (var (i, n, t) in ss.NodeProfile().OrderByDescending(x => x.t).Take(int.TryParse(Environment.GetEnvironmentVariable("GCDRNET_PROF_TOP"), out int top) ? top : 14))
                    {
                        string extra = n.P is ConvP cp ? $" k{cp.Kernel[0]} s{cp.Strides[0]} g{cp.Group}" : "";
                        Console.WriteLine($"  #{i,-3} {n.Op,-14}{extra,-12} {string.Join('x', ss.ShapeOf(n.In[0]))}->{string.Join('x', ss.ShapeOf(n.Out[0]))}  {t.TotalMilliseconds / reps,7:F2}");
                    }
                }
        }
        return 0;
    }

    private static int BenchVk(string gcPath, string drPath, int h, int w, int reps) =>
        BenchE2e(gcPath, drPath, h, w, reps, GcdrBackend.Vulkan, "vulkan");

    private static int BenchE2e(string gcPath, string drPath, int h, int w, int reps, GcdrBackend backend, string tag)
    {
        // full pipeline: BGR u8 -> enhance -> BGR u8 (Vulkan: everything on GPU)
        using var enh = GcdrEnhancer.FromFiles(gcPath, drPath, backend);
        byte[] src = new byte[w * h * 3];
        new Random(0).NextBytes(src);
        byte[] dst = new byte[w * h * 3];
        enh.Enhance(src, w, h, w * 3, dst, w * 3); // warmup + JIT + upload
        double best = double.MaxValue;
        for (int i = 0; i < reps; i++)
        {
            long t = Stopwatch.GetTimestamp();
            enh.Enhance(src, w, h, w * 3, dst, w * 3);
            best = Math.Min(best, Stopwatch.GetTimestamp() - t);
        }
        Console.WriteLine($"{tag} e2e {w}x{h}: {best / Stopwatch.Frequency * 1000:F1} ms" + (backend == GcdrBackend.Vulkan ? " (incl. upload/download)" : ""));
        return 0;
    }

    private static int VkProf(string gcPath, string drPath, int h, int w)
    {
        var be = Sdcb.SimdOnnx.Backends.Vulkan.VkBackend.Default;
        foreach ((string tag, string mp, int cin) in new[] { ("gc", gcPath, 3), ("dr", drPath, 6) })
        {
            var m = OnnxModel.Load(mp);
            using var s = be.NewSession(m, [1, cin, h, w]);
            s.Run(); // warm
            var agg = s.DebugProfile();
            double tot = agg.Values.Sum();
            Console.WriteLine($"--- {tag} gpu per-shader (total {tot:F1} ms) ---");
            foreach (var kv in agg.OrderByDescending(k => k.Value)) Console.WriteLine($"  {kv.Key,-14} {kv.Value,8:F2} ms");
            if (Environment.GetEnvironmentVariable("VKPROF_NODES") == "1")
                foreach (var (name, ms) in s.PerDispatch!.OrderByDescending(x => x.Ms).Take(15))
                    Console.WriteLine($"    {ms,8:F2} ms  {name}");
        }
        return 0;
    }

    private static float[] ToF32(byte[] raw)
    {
        var f = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, f, 0, raw.Length);
        return f;
    }

    private static int[] ReadShape(string dir, string tag, string name)
    {
        var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "manifest.json")));
        return doc.RootElement.GetProperty(tag).GetProperty("shapes").GetProperty(name)
            .EnumerateArray().Select(e => e.GetInt32()).ToArray();
    }

    private static string FindTestdata()
    {
        if (Environment.GetEnvironmentVariable("GCDRNET_TESTDATA") is { Length: > 0 } env) return env;
        var d = new DirectoryInfo(AppContext.BaseDirectory);
        while (d != null)
        {
            string cand = Path.Combine(d.FullName, "tests", "testdata");
            if (File.Exists(Path.Combine(cand, "manifest.json"))) return cand;
            cand = Path.Combine(d.FullName, "testdata");
            if (File.Exists(Path.Combine(cand, "manifest.json"))) return cand;
            d = d.Parent;
        }
        throw new DirectoryNotFoundException("testdata not found");
    }
}
