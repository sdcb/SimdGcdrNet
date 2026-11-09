using System.Diagnostics;
using Sdcb.SimdGcdrNet;
using Sdcb.SimdGcdrNet.Models;
using Sdcb.SimdPaddleOCR;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using Sdcb.SimdCodecs;
using Sdcb.SimdCodecs.Abstractions;
using Sdcb.SimdCodecs.Webp;

// usage: PaddleOCR.Console [image] [output]
// Cleans a photographed page with SimdGcdrNet (shadows, uneven lighting, show-through from the
// back side), then runs SimdPaddleOCR on the original and the enhanced image. The default
// ocr-example.jpg has heavy show-through, which OCR reads as garbage lines on the original.
// Both libraries take raw interleaved pixels, so SimdCodecs decodes straight into one RGBA array they share.
string input = args.Length > 0 ? args[0] : Path.Combine(AppContext.BaseDirectory, "ocr-example.jpg");
string output = args.Length > 1 ? args[1] : Path.ChangeExtension(Path.GetFileName(input), null) + "-enhanced.jpg";
Console.OutputEncoding = System.Text.Encoding.UTF8;

// ApplyOrientation bakes the EXIF rotation in; Decode throws with the codec's reason when the file is bad.
using PixelBuffer decoded = ImageCodecs.Decode(File.ReadAllBytes(input), PixelFormat.Rgba32, new DecodeOptions { ApplyOrientation = true });
int width = decoded.Width, height = decoded.Height;
byte[] original = decoded.Pixels.ToArray(); // a span cannot live across the await below
byte[] enhanced = (byte[])original.Clone();
int stride = width * 4;
Console.WriteLine($"{input}: {width}x{height}");

using GcdrEnhancer enhancer = GcdrEnhancer.FromBytes(GcdrModels.GcNet, GcdrModels.DrNet, GcdrBackend.Auto);
using PaddleOcrAll ocr = await PaddleOcrAll.LoadAsync(ChineseV6TinyModels.Default);

Stopwatch sw = Stopwatch.StartNew();
enhancer.Enhance(enhanced, width, height, stride, enhanced, stride, GcdrPixelFormat.Rgba32);
Console.WriteLine($"enhanced in {sw.ElapsedMilliseconds} ms (first call includes plan compilation)");
ImageFormat outputFormat = Path.GetExtension(output).ToLowerInvariant() switch
{
    ".png" => ImageFormat.Png,
    ".webp" => ImageFormat.Webp,
    _ => ImageFormat.Jpeg,
};
using (FileStream fs = File.Create(output))
    ImageCodecs.Encode(enhanced, width, height, PixelFormat.Rgba32, outputFormat, fs, stride, outputFormat == ImageFormat.Webp ? new WebpLossyEncodeOptions { Quality = 90 } : null, decoded.Metadata);
Console.WriteLine($"saved {Path.GetFullPath(output)}");

PaddleOcrResult before = ocr.Run(original, width, height, stride, ImagePixelFormat.Rgba32);
PaddleOcrResult after = ocr.Run(enhanced, width, height, stride, ImagePixelFormat.Rgba32);
Print("original", before);
Print("enhanced", after);

Console.WriteLine();
Console.WriteLine($"{"",-10}{"lines",8}{"chars",8}");
Summary("original", before);
Summary("enhanced", after);

static void Print(string title, PaddleOcrResult result)
{
    Console.WriteLine();
    Console.WriteLine($"===== {title} =====");
    foreach (PaddleOcrLine line in result.Lines)
        Console.WriteLine(line.Text);
}

static void Summary(string title, PaddleOcrResult result) =>
    Console.WriteLine($"{title,-10}{result.Lines.Length,8}{result.Lines.Sum(l => l.Text.Length),8}");
