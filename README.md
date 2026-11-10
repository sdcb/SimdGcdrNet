# Sdcb.SimdGcdrNet

[English](#english) | [中文](#中文)

---

## English

**Sdcb.SimdGcdrNet** is a pure C# implementation of [GCDRNet](https://github.com/ZZZHANG-jx/GCDRNet)
(*Appearance Enhancement for Camera-captured Document Images in the Wild*, IEEE TAI 2023) —
document-image shadow removal and appearance enhancement **without onnxruntime or any other
third-party inference engine**. Inference runs on [Sdcb.SimdOnnx](https://github.com/sdcb/SimdOnnx),
a pure C# engine shared with `Sdcb.SimdPaddleOCR` (it grew out of this repository's own engine).

The model is a two-stage pipeline:

1. **GC-Net** predicts a per-pixel multiplicative shading/illumination map `I_sm`.
2. **Division correction**: `I_GC = clamp(I_s / I_sm, 0, 1)`.
3. **DR-Net** consumes the 6-channel input `[I_s; I_GC]` and produces the enhanced image.

### Features

- **Zero inference-engine dependency** — self-contained ONNX proto reader, graph compiler
  (shape specialization, constant folding, operator fusion, buffer-arena planning).
- **CPU backend** with ISA tiered dispatch: AVX-512 → AVX2 (+FMA, multithreaded) → `Vector<float>` → Scalar on x64;
  NEON (AArch64 AdvSimd) → Scalar on arm64; PackedSimd (WASM SIMD128) → `Vector<float>` → Scalar on browser-wasm.
- **Vulkan compute backend** (Windows/Linux/Android `libvulkan`): push-descriptors, ReBAR-aware
  staging, single-recorded-command-buffer graph execution: pixels → GC → division → DR → pixels as one
  model and one submit per image, with arbitrary source/destination strides.
- **Metal compute backend** (macOS, Apple Silicon): pure C# ObjC P/Invoke (no Xamarin/Metal.NET),
  fp16 storage + fp32 math, simdgroup-MMA implicit-GEMM convolutions — see
  [docs/apple-m4-metal.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/apple-m4-metal.md) (~7.5× CPU on M4).
- **Any interleaved 8-bit layout** (`Bgr24` / `Rgb24` / `Bgra32` / `Rgba32`), arbitrary strides, in place;
  thread-safe `GcdrEnhancer`; Native AOT compatible.

### Performance (RTX 3080 Ti, 1600×1152)

| Path | Time | vs PyTorch baseline |
|---|---|---|
| Vulkan e2e (incl. upload/download + pre/post) | **~9.6 ms** | PyTorch pure inference 30.5 ms; with pre/post 57.2 ms |
| Vulkan corpus (49 RealDAE, seq order) | **~28 ms/img** | methodology: [docs/corpus.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/corpus.md), current numbers: [docs/perf.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/perf.md) |
| CPU AVX2 corpus (same set) | ~722 ms/img | peak working set ~0.8 GB (Vulkan: 0.33 GB + 0.42 GB VRAM) |
| CPU AVX2 (16 threads) gc+dr inference | ~365 ms | — |

Ryzen AI 9 H 365 (AVX-512) + Radeon 880M (RDNA 3.5 iGPU): see [docs/amd-h365-880m.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/amd-h365-880m.md)
(Vulkan ~100 ms/img, CPU AVX-512 ~285 ms/img on the same corpus).

Ryzen 9 5950X (AVX2, 16C/32T) + Intel Arc B580 (Xe2, XMX via 8x16x16 coopmat): see [docs/amd-5950x-b580.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/amd-5950x-b580.md)
(Vulkan ~36 ms/img, CPU AVX2 ~414 ms/img on the same corpus).

Rockchip RK3399 (2× A72 + 4× A53, NEON, passively cooled): see [docs/rockchip-rk3399.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/rockchip-rk3399.md)
(CPU NEON ~9.0 s/img on the same corpus, 2.6× the generic `Vector<float>` path at a locked 1 GHz).

Snapdragon 8 Gen 3 (realme GT5 Pro, Android 16; Adreno 750 without coopmat, 64/128-lane waves): see [docs/snapdragon-8gen3.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/snapdragon-8gen3.md)
(Vulkan ~486 ms/img with register-tiled conv/GEMM kernels, 2.6× the shared-memory path; CPU NEON ~1.03 s/img on the same corpus).

Browser (browser-wasm, WASM SIMD128, Web Workers threads, LLVM AOT): see [docs/wasm.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/wasm.md)
(Edge on a 5950X: CPU PackedSimd ~696 ms/img on the same corpus, 1.55× the generic `Vector<float>` path in the same browser).

### Quick start

Requires **.NET 10**: both packages target `net10.0` only.

| Package | Contents |
|---|---|
| `Sdcb.SimdGcdrNet` | the engine (CPU / Vulkan / Metal); ships LINQPad samples (System.Drawing, ImageSharp, OpenCvSharp, SkiaSharp + SimdPaddleOCR) |
| `Sdcb.SimdGcdrNet.Models` | the GC-Net + DR-Net ONNX models (~21 MB) as embedded resources; no dependencies, versioned independently of the engine |

```csharp
using Sdcb.SimdGcdrNet;
using Sdcb.SimdGcdrNet.Models;

// Auto: Metal on macOS Apple Silicon, else Vulkan, else CPU. Also: Cpu (default) / Vulkan / Metal.
using GcdrEnhancer enh = GcdrEnhancer.FromBytes(GcdrModels.GcNet, GcdrModels.DrNet, GcdrBackend.Auto);
// without the models package: GcdrEnhancer.FromFiles("gcdrnet_gcnet_sim.onnx", "gcdrnet_drnet_sim.onnx", ...)
// (models in tests/testdata/models), or FromBytes(span, span) / FromStreams(stream, stream)

byte[] dst = enh.Enhance(bgr, width: 1600, height: 1152);   // tightly packed BGR24, HWC
// strided, any interleaved 8-bit layout, output in the same layout (alpha copied through):
enh.Enhance(srcSpan, w, h, srcStride, dstSpan, dstStride, GcdrPixelFormat.Rgba32);
```

Pixel formats: `Bgr24` (OpenCV `Mat`, GDI+ 24bpp), `Rgb24`, `Bgra32` (SkiaSharp, GDI+ 32bpp),
`Rgba32` (ImageSharp, browser canvas). `dst` may be the same buffer as `src` (same stride) for in-place enhancement.

Input is padded to a multiple of 32 with top/left **replicate** padding and cropped back —
matching the reference Python implementation.

### Examples

The three UI examples open `examples/example.jpg` (a phone photo of a hand-drawn poster with a hard hand shadow):

- `examples/SimdCodecs.AspNetCore` — ASP.NET Core + SimdCodecs (`ImageCodecs.Decode` with `ApplyOrientation` into one `Rgba32` buffer, enhanced in place; a bad upload returns the codec's own error message): upload / drag-and-drop,
  before-after slider, `auto` / `cpu` backend switch, timings; `POST /api/enhance` returns the enhanced image
  (API docs at `/scalar`).
- `examples/SkiaSharp.Avalonia` — cross-platform desktop (Windows / Linux / macOS, `Bgra32` `SKBitmap`):
  side-by-side preview, `auto` / `cpu` / `vulkan` (`metal` on macOS) switch, save as JPEG / PNG.
- `examples/OpenCvSharp5.Wpf` — WPF + OpenCvSharp5 (`Bgr24` `Mat`, any `Step()`): same UI, plus drag-and-drop.
- `examples/PaddleOCR.Console` — enhance with SimdGcdrNet, then OCR the original and the enhanced image with
  [SimdPaddleOCR](https://github.com/sdcb/SimdPaddleOCR); both take the same decoded RGBA buffer. Its default
  `examples/ocr-example.jpg` (from RealDAE) has heavy show-through from the back of the page: OCR on the
  original returns 56 lines (mostly garbage from the back side), on the enhanced image the 14 real lines.

```bash
dotnet run --project examples/SimdCodecs.AspNetCore        # http://localhost:5190
dotnet run --project examples/SkiaSharp.Avalonia -- [image]
dotnet run --project examples/OpenCvSharp5.Wpf -- [image]
dotnet run --project examples/PaddleOCR.Console -- [image] [output.jpg]
```

### Layout

The ONNX engine (protobuf reader, graph compiler, CPU SIMD kernels, Vulkan / Metal backends) is
[Sdcb.SimdOnnx](https://github.com/sdcb/SimdOnnx); this library holds the GCDRNet-specific parts:

- `src/Sdcb.SimdGcdrNet/GcdrEnhancer.cs` — public API: backend selection, session pools, and the
  composed model (GC-Net → `clamp(x / max(shading, 1e-6), 0, 1)` → DR-Net, built with
  `OnnxModelBuilder`) that runs on SimdOnnx's `CpuSession` and `VkImageSession` / `MtlImageSession`.
  The library uses only SimdOnnx's public API; the fusions it needs (UpHead, ChannelShift, SafeDiv)
  are engine built-ins.
- `bench/Sdcb.SimdGcdrNet.Benchmark` — `--check`, `--bench` (CPU), `--benchvk` (Vulkan e2e),
  `--benchmtl` / `--checkgpu` / `--mtlprof` / `--mtrace` (Metal e2e, diff check, per-shader
  profile, CPU-vs-Metal per-node trace), `--vkprof` (per-shader/per-dispatch GPU timing), `--vkcheck` / `--vknodes`
  (Vulkan vs CPU, whole graph / per node)
- `bench/Sdcb.SimdGcdrNet.AndroidBench` — the same modes inside an Android APK (`run.ps1`: build → adb install → push testdata → run → pull logs)
- `bench/Sdcb.SimdGcdrNet.WasmBench` — `--corpus` / `--bench` / `--check` / `--tiers` in a browser (`run.ps1`: publish with LLVM AOT → `server.mjs` with COOP/COEP → headless Edge/Chrome); needs the `wasm-tools-net10` workload and Node.js
- `src/Sdcb.SimdGcdrNet.Models` — `GcdrModels` over the embedded ONNX files (taken from `tests/testdata/models`)
- `src/Sdcb.SimdGcdrNet/linqpad-samples` — LINQPad samples packed into `Sdcb.SimdGcdrNet` (tag `linqpad-samples`)
- `tests/Sdcb.SimdGcdrNet.Tests` — xUnit tests; `tests/GoldenDerive.cs` recomputes the derived golden data;
  `tests/Sdcb.SimdGcdrNet.TestReport` merges CI results into a markdown report
- `tools/fetch-fixture.ps1` — downloads the RealDAE corpus; `tools/GenGolden` regenerates the golden data
- `build/` — LINQPad scripts to pack (`01-build-packages`) and publish (`02-publish-proget-nuget`) the
  packages; versions live in `00-common.linq` and `Directory.Build.props`

### Tests

```bash
./tools/fetch-fixture.ps1                          # RealDAE corpus (~36 MB) -> tests/testdata/realdae
dotnet test tests/Sdcb.SimdGcdrNet.Tests -c Release
```

Golden data comes from onnxruntime runs of the simplified models (CPU ≤ 1e-4 /
Vulkan ≤ 5e-4): only the source `{tag}_src.png` and the two oracle outputs (`_shadow.f32`, `_output.f32`)
are committed; the GC-Net input, the DR-Net input and the final u8 image are recomputed bit-exactly by
`tests/GoldenDerive.cs`. Regenerate with `dotnet run --project tools/GenGolden -- --models tests/testdata/models`
(pure C#: OnnxRuntime oracle + SkiaSharp decode).

The 50-image RealDAE corpus (`NN_<color|shadow|bleed>.webp` + paired `NN_gt.webp` ground truths) lives in
[sdcb/DocGainFixture](https://github.com/sdcb/DocGainFixture); without it `CorpusTests` are skipped.
The benchmark's `--corpus [cpu|vulkan|metal]` mode reports per-image ms, MP/s, memory and PSNR/MAE vs the `_gt` pairs.

Kernel dispatch follows the runtime's `IsSupported` flags, so ISA tiers are exercised via the standard
runtime knobs: `DOTNET_EnableAVX512=0` / `DOTNET_EnableAVX512F=0` (AVX2), `DOTNET_EnableAVX2=0` (128-bit
`Vector<float>`), `DOTNET_EnableAVX=0` (SSE, no FMA) and `DOTNET_EnableHWIntrinsic=0` (scalar).

CI ([.github/workflows/test.yml](https://github.com/sdcb/SimdGcdrNet/blob/main/.github/workflows/test.yml))
runs two waves in parallel, then merges them into one report (the run's summary page):

- **smoke**: `dotnet test` plus a corpus smoke (`Benchmark --ci smoke`: sane, deterministic output that is
  closer to the ground truth than the input; GPU output close to CPU) on win-x64, win-arm64, linux-x64,
  linux-arm64, osx-x64 and osx-arm64 (CPU and Metal), plus the x64 ISA tiers above and arm64 scalar;
- **bench**: `Benchmark --ci bench` on 20 corpus images, several replicas per case: win-x64 (default,
  no AVX-512, no AVX2), linux-x64, linux-arm64 (NEON) and osx-arm64 (CPU, Metal).

### Notes

- Vulkan path requires a compute-capable device; `GcdrBackend.Auto` falls back to CPU.
- Metal path requires macOS on Apple Silicon with Apple7+ (simdgroup-MMA, i.e. M1+; an x64 process
  under Rosetta also qualifies); `Auto` checks `SupportsApple7` before choosing it,
  `GcdrBackend.Metal` selects it unconditionally. Set `GCDRNET_MTL_VERBOSE=1` to print the chosen device.
- Thread-safety: `GcdrEnhancer` is thread-safe (CPU: per-size session pools; Vulkan: size-agnostic worker pool).
- 2.0 moved the engine out: `InferenceSession`, `CompiledModel`, `OnnxModel`, `VkInferenceSession` and
  `MtlInferenceSession` are no longer in `Sdcb.SimdGcdrNet.*`; use the same types from
  `Sdcb.SimdOnnx.*` (a dependency of this package). `GcdrEnhancer`'s API is unchanged.
- Not covered: batched inference (N>1), fp16/int8.
- Native AOT works (`IsAotCompatible`, no trim/AOT warnings; CPU and Vulkan output verified identical to JIT). On **x64** the executable
  project must set the instruction set, otherwise ILC compiles for the SSE baseline, folds
  `Avx2.IsSupported` / `Avx512F.IsSupported` to `false` and the CPU backend silently runs the 128-bit
  `Vector<float>` tier (~1.3× slower):

  ```xml
  <IlcInstructionSet>x86-64-v3</IlcInstructionSet>   <!-- AVX2+FMA; x86-64-v4 for AVX-512 -->
  ```

  The binary then refuses to start on CPUs without those extensions. arm64 needs nothing (NEON is in
  the baseline); GPU backends are unaffected.

### License

The inference engine (`Sdcb.SimdGcdrNet`, everything outside `src/Sdcb.SimdGcdrNet.Models`) is Apache-2.0.
`Sdcb.SimdGcdrNet.Models` is MIT, following the upstream [GCDRNet](https://github.com/ZZZHANG-jx/GCDRNet)
weights it embeds (Copyright (c) 2024 Jiaxin Zhang); see
[THIRD-PARTY-NOTICES.txt](https://github.com/sdcb/SimdGcdrNet/blob/main/THIRD-PARTY-NOTICES.txt), also shipped in both packages.
The test corpus ([sdcb/DocGainFixture](https://github.com/sdcb/DocGainFixture)) and `examples/ocr-example.jpg`
are from the RealDAE dataset of the same paper.

---

## 中文

**Sdcb.SimdGcdrNet** 是 [GCDRNet](https://github.com/ZZZHANG-jx/GCDRNet)（相机拍摄文档去阴影/外观增强，
IEEE TAI 2023）的**纯 C# 推理库**——不依赖 onnxruntime 等第三方推理引擎，与 `Sdcb.SimdPaddleOCR`
共用纯 C# 推理引擎 [Sdcb.SimdOnnx](https://github.com/sdcb/SimdOnnx)（2.0 起引擎类型移到 `Sdcb.SimdOnnx.*`）。

模型为两级串联：**GC-Net** 估计逐像素光照/阴影图 `I_sm` → `I_GC = clamp(I_s/I_sm)` 除法校正 →
**DR-Net** 消费 6 通道 `[I_s; I_GC]` 输出增强图。

### 特性

- **零推理引擎依赖**：自带 ONNX proto 读取 + 图编译（shape 特化、常量折叠、算子融合、arena 规划）
- **CPU 后端**：x64 上 AVX-512 → AVX2+FMA 多线程 → `Vector<float>` → Scalar 四档 ISA 自动分发；arm64 上 NEON（AdvSimd）→ Scalar；browser-wasm 上 PackedSimd（WASM SIMD128）→ `Vector<float>` → Scalar
- **Vulkan 后端**：push-descriptor、ReBAR 感知 staging、单命令缓冲图执行、
  像素 → GC → 除法校正 → DR → 像素合成一个模型、每张图一次提交，支持任意 src/dst stride
- **Metal 后端**（macOS，Apple Silicon）：纯 C# ObjC P/Invoke（不依赖 Xamarin/Metal.NET），fp16 存储 + fp32 计算，
  simdgroup-MMA 隐式 GEMM 卷积，见 [docs/apple-m4-metal.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/apple-m4-metal.md)（M4 上约为 CPU 的 7.5 倍）
- **任意 8 位交错像素格式**（`Bgr24` / `Rgb24` / `Bgra32` / `Rgba32`），任意 stride，支持原地增强；
  `GcdrEnhancer` 线程安全；兼容 Native AOT

### 性能（RTX 3080 Ti，1600×1152）

| 路径 | 耗时 | 对比 PyTorch |
|---|---|---|
| Vulkan 端到端（含上下行+前后处理） | **~9.6 ms** | PyTorch 纯推理 30.5 ms / 含前后处理 57.2 ms |
| Vulkan corpus（49 张 RealDAE，序号序） | **~28 ms/张** | 口径见 [docs/corpus.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/corpus.md)，最新数据见 [docs/perf.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/perf.md) |
| CPU AVX2 corpus（同集） | ~722 ms/张 | 峰值 working set ~0.8 GB（Vulkan：0.33 GB + 显存 0.42 GB） |
| CPU AVX2（16 线程）gc+dr 推理 | ~365 ms | — |

锐龙 AI 9 H 365（AVX-512）+ Radeon 880M（RDNA 3.5 核显）见 [docs/amd-h365-880m.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/amd-h365-880m.md)
（同一语料 Vulkan ~100 ms/张，CPU AVX-512 ~285 ms/张）。

锐龙 9 5950X（AVX2，16C/32T）+ Intel Arc B580（Xe2，8x16x16 coopmat 走 XMX）见 [docs/amd-5950x-b580.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/amd-5950x-b580.md)
（同一语料 Vulkan ~36 ms/张，CPU AVX2 ~414 ms/张）。

瑞芯微 RK3399（2× A72 + 4× A53，NEON，被动散热）见 [docs/rockchip-rk3399.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/rockchip-rk3399.md)
（同一语料 CPU NEON ~9.0 s/张；锁 1 GHz 时是通用 `Vector<float>` 路径的 2.6 倍）。

骁龙 8 Gen 3（真我 GT5 Pro，Android 16；Adreno 750 无 coopmat，64/128 lane wave）见 [docs/snapdragon-8gen3.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/snapdragon-8gen3.md)
（同一语料 Vulkan ~486 ms/张，寄存器分块卷积/GEMM 内核比共享内存路径快 2.6 倍；CPU NEON ~1.03 s/张）。

浏览器（browser-wasm，WASM SIMD128，Web Workers 多线程，LLVM AOT）见 [docs/wasm.md](https://github.com/sdcb/SimdGcdrNet/blob/main/docs/wasm.md)
（5950X 上的 Edge：同一语料 CPU PackedSimd ~696 ms/张，是同浏览器通用 `Vector<float>` 路径的 1.55 倍）。

### 快速上手

需要 **.NET 10**：两个包都只面向 `net10.0`。

| 包 | 内容 |
|---|---|
| `Sdcb.SimdGcdrNet` | 推理引擎（CPU / Vulkan / Metal）；附带 LINQPad 示例（System.Drawing、ImageSharp、OpenCvSharp、SkiaSharp + SimdPaddleOCR） |
| `Sdcb.SimdGcdrNet.Models` | GC-Net + DR-Net 两个 ONNX 模型（约 21 MB），以嵌入资源形式提供；无任何依赖，版本独立于引擎 |

```csharp
using Sdcb.SimdGcdrNet;
using Sdcb.SimdGcdrNet.Models;

// Auto：macOS Apple Silicon 优先 Metal，否则 Vulkan，最后 CPU；也可显式 Cpu（默认）/ Vulkan / Metal
using GcdrEnhancer enh = GcdrEnhancer.FromBytes(GcdrModels.GcNet, GcdrModels.DrNet, GcdrBackend.Auto);
// 不用模型包：GcdrEnhancer.FromFiles("gcdrnet_gcnet_sim.onnx", "gcdrnet_drnet_sim.onnx", ...)
// （模型见 tests/testdata/models），或 FromBytes(span, span) / FromStreams(stream, stream)

byte[] dst = enh.Enhance(bgrBytes, width: 1600, height: 1152);   // 紧密排列的 BGR24，HWC
// 带 stride、任意 8 位交错格式，输出与输入同格式（alpha 原样拷回）：
enh.Enhance(src, w, h, srcStride, dst, dstStride, GcdrPixelFormat.Rgba32);
```

像素格式：`Bgr24`（OpenCV `Mat`、GDI+ 24bpp）、`Rgb24`、`Bgra32`（SkiaSharp、GDI+ 32bpp）、
`Rgba32`（ImageSharp、浏览器 canvas）。`dst` 可以和 `src` 是同一块内存（stride 相同），即原地增强。

输入补齐到 32 倍数（上/左 replicate padding），输出自动裁回，与参考 Python 实现一致。

### 示例

三个界面示例默认打开 `examples/example.jpg`（手机拍摄的手抄报，带一块很硬的手影）：

- `examples/SimdCodecs.AspNetCore`：ASP.NET Core + SimdCodecs（`ImageCodecs.Decode` 加 `ApplyOrientation` 解码到一块 `Rgba32` 缓冲区并原地增强；上传文件有问题时直接返回解码器给出的错误原因），上传/拖拽图片、前后对比滑块、
  `auto` / `cpu` 后端切换与耗时；`POST /api/enhance` 直接返回增强后的图片，API 文档在 `/scalar`。
- `examples/SkiaSharp.Avalonia`：跨平台桌面（Windows / Linux / macOS，`Bgra32` 的 `SKBitmap`），左右对比预览，
  可切换 `auto` / `cpu` / `vulkan`（macOS 上为 `metal`），结果可存为 JPEG / PNG。
- `examples/OpenCvSharp5.Wpf`：WPF + OpenCvSharp5（`Bgr24` 的 `Mat`，按 `Step()` 传 stride），界面同上，支持拖拽。
- `examples/PaddleOCR.Console`：先用 SimdGcdrNet 增强，再用 [SimdPaddleOCR](https://github.com/sdcb/SimdPaddleOCR)
  分别识别原图和增强图；两者共用同一块解码后的 RGBA 像素。默认图片 `examples/ocr-example.jpg`（取自 RealDAE）
  背面透字严重：原图 OCR 出 56 行（大部分是背面透过来的乱码），增强后只剩正文的 14 行。

```bash
dotnet run --project examples/SimdCodecs.AspNetCore        # http://localhost:5190
dotnet run --project examples/SkiaSharp.Avalonia -- [图片]
dotnet run --project examples/OpenCvSharp5.Wpf -- [图片]
dotnet run --project examples/PaddleOCR.Console -- [图片] [输出.jpg]
```

### 构建 shader

本库不再自带 shader：所有 GPU 内核（包括 UpHead / ChannelShift 融合和像素 I/O）都在
[Sdcb.SimdOnnx](https://github.com/sdcb/SimdOnnx) 里。

### 测试

```bash
./tools/fetch-fixture.ps1                          # RealDAE 语料（约 36 MB）-> tests/testdata/realdae
dotnet test tests/Sdcb.SimdGcdrNet.Tests -c Release
```

golden 由 onnxruntime 对简化模型跑出（CPU 阈值 ≤1e-4 / Vulkan ≤5e-4）：仓库里只提交源图 `{tag}_src.png`
和两份 oracle 输出（`_shadow.f32`、`_output.f32`），GC-Net 输入、DR-Net 输入和最终 u8 图由
`tests/GoldenDerive.cs` 逐位一致地现算。重新生成用
`dotnet run --project tools/GenGolden -- --models tests/testdata/models`（纯 C#，OnnxRuntime oracle + SkiaSharp 解码）。

50 张 RealDAE 语料（`NN_<color|shadow|bleed>.webp` 及配对 `NN_gt.webp` 基准图）放在
[sdcb/DocGainFixture](https://github.com/sdcb/DocGainFixture)，没下载时 `CorpusTests` 自动跳过；
benchmark `--corpus [cpu|vulkan|metal]` 按图报耗时、MP/s、内存及对照 `_gt` 的 PSNR/MAE。

内核分发直接跟随 runtime 的 `IsSupported` 标志，ISA 档位用 runtime 开关强制降级：
`DOTNET_EnableAVX512=0` / `DOTNET_EnableAVX512F=0`（AVX2）、`DOTNET_EnableAVX2=0`（128 位 Vector）、
`DOTNET_EnableAVX=0`（SSE 无 FMA）、`DOTNET_EnableHWIntrinsic=0`（纯标量）。

CI（[.github/workflows/test.yml](https://github.com/sdcb/SimdGcdrNet/blob/main/.github/workflows/test.yml)）
两波并行，最后汇总成一份报告（显示在 run 的 Summary 页）：

- **smoke（正确性）**：win-x64、win-arm64、linux-x64、linux-arm64、osx-x64、osx-arm64（CPU 与 Metal）上跑
  `dotnet test` 和语料冒烟（`Benchmark --ci smoke`：输出正常、两次运行一致、比输入更接近 GT；GPU 与 CPU 输出接近），
  另加 x64 上述各 ISA 档位与 arm64 纯标量；
- **bench（性能）**：`Benchmark --ci bench` 跑 20 张语料图、每个 case 多个 replica：win-x64（默认、无 AVX-512、
  无 AVX2）、linux-x64、linux-arm64（NEON）、osx-arm64（CPU、Metal）。

### Native AOT

支持 Native AOT（`IsAotCompatible`，裁剪 / AOT 无警告；CPU、Vulkan 输出与 JIT 逐字节一致）。**x64** 上可执行项目必须指定指令集，
否则 ILC 按 SSE 基线编译，`Avx2.IsSupported` / `Avx512F.IsSupported` 被折成 `false`，CPU 后端会静默退到
128 位 `Vector<float>` 档（慢约 1.3 倍）：

```xml
<IlcInstructionSet>x86-64-v3</IlcInstructionSet>   <!-- AVX2+FMA；x86-64-v4 才有 AVX-512 -->
```

这样编出的程序在不支持这些扩展的 CPU 上无法启动。arm64 不用设（NEON 在基线里）；GPU 后端不受影响。

### 许可证

推理引擎（`Sdcb.SimdGcdrNet`，即 `src/Sdcb.SimdGcdrNet.Models` 以外的全部代码）为 Apache-2.0；
`Sdcb.SimdGcdrNet.Models` 沿用其内嵌的上游 [GCDRNet](https://github.com/ZZZHANG-jx/GCDRNet) 模型权重的 MIT 许可
（Copyright (c) 2024 Jiaxin Zhang），详见 [THIRD-PARTY-NOTICES.txt](https://github.com/sdcb/SimdGcdrNet/blob/main/THIRD-PARTY-NOTICES.txt)（两个包内均附带）。
测试语料（[sdcb/DocGainFixture](https://github.com/sdcb/DocGainFixture)）和 `examples/ocr-example.jpg`
取自同一论文的 RealDAE 数据集。
