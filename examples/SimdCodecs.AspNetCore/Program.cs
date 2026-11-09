using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using SimdCodecs.AspNetCore;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Mvc;
using Scalar.AspNetCore;
using Sdcb.SimdGcdrNet;
using Sdcb.SimdCodecs;
using Sdcb.SimdCodecs.Abstractions;
using Sdcb.SimdCodecs.Jpeg;
using Sdcb.SimdCodecs.Webp;

const long maxUploadBytes = 20 * 1024 * 1024;
string[] timingHeaders = ["X-Decode-Ms", "X-Enhance-Ms", "X-Encode-Ms", "X-Image-Size", "X-Backend"];

WebApplicationBuilder builder = WebApplication.CreateBuilder(args);
builder.WebHost.ConfigureKestrel(options => options.Limits.MaxRequestBodySize = maxUploadBytes);
builder.Services.Configure<FormOptions>(options => options.MultipartBodyLengthLimit = maxUploadBytes);
builder.Services.AddSingleton<GcdrEngine>();
builder.Services.AddOpenApi();
builder.Services.AddCors(options => options.AddDefaultPolicy(policy =>
    policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod().WithExposedHeaders(timingHeaders)));

WebApplication app = builder.Build();
app.UseCors();
app.UseDefaultFiles();
app.UseStaticFiles();
app.MapOpenApi();
app.MapScalarApiReference();

app.MapGet("/api/backends", () => Results.Ok(new BackendList(GcdrEngine.Backends, GcdrEngine.DefaultBackend)))
    .WithName("ListBackends")
    .WithTags("Enhance")
    .WithSummary("列出可用计算后端")
    .Produces<BackendList>();

app.MapGet("/example.jpg", (IWebHostEnvironment env) =>
{
    string? path = new[]
    {
        Path.Combine(AppContext.BaseDirectory, "example.jpg"),
        Path.GetFullPath(Path.Combine(env.ContentRootPath, "..", "example.jpg")),
    }.FirstOrDefault(File.Exists);
    return path is null ? Results.NotFound() : Results.File(path, "image/jpeg");
}).ExcludeFromDescription();

app.MapPost("/api/enhance", EnhanceAsync)
    .DisableAntiforgery()
    .WithName("EnhanceImage")
    .WithTags("Enhance")
    .WithSummary("文档图片去阴影 / 外观增强")
    .WithDescription("上传图片（表单字段 file），可选 backend=auto|cpu、format=jpeg|webp|png（默认 jpeg）。" +
        "返回增强后的图片；耗时与尺寸见响应头 X-Decode-Ms、X-Enhance-Ms、X-Encode-Ms、X-Image-Size、X-Backend。" +
        "每个新尺寸的第一次调用包含计划编译，耗时偏高。")
    .Produces(StatusCodes.Status200OK, contentType: "image/jpeg")
    .Produces<EnhanceError>(StatusCodes.Status400BadRequest)
    .Produces<EnhanceError>(StatusCodes.Status500InternalServerError)
    .Accepts<EnhanceUpload>("multipart/form-data");

app.Run();

static async Task<IResult> EnhanceAsync(
    [FromForm] EnhanceUpload upload,
    GcdrEngine engine,
    HttpResponse response,
    CancellationToken cancellationToken)
{
    IFormFile? file = upload.File;
    if (file is null || file.Length == 0)
        return Results.BadRequest(new EnhanceError("请上传图片文件（表单字段 file）。"));
    if (file.Length > maxUploadBytes)
        return Results.BadRequest(new EnhanceError("图片超过 20 MB 上限。"));
    if (!GcdrEngine.TryParseBackend(upload.Backend, out GcdrBackend backend))
        return Results.BadRequest(new EnhanceError("backend 必须是 auto 或 cpu。"));
    if (!TryGetEncoder(upload.Format, out ImageFormat format, out EncodeOptions? encodeOptions, out string contentType))
        return Results.BadRequest(new EnhanceError("format 必须是 jpeg、webp 或 png。"));

    Stopwatch stage = Stopwatch.StartNew();
    try
    {
        // The facade decodes from memory: read the upload (at most 20 MB) into one array.
        byte[] data = new byte[file.Length];
        await using (Stream stream = file.OpenReadStream())
            await stream.ReadExactlyAsync(data, cancellationToken);

        // The header gives the size, so an oversized image is refused before any pixel is allocated.
        // A file whose header cannot be read goes on to Decode, which throws with the codec's reason.
        if (ImageCodecs.TryReadInfo(data, out ImageInfo info) && (long)info.Width * info.Height > GcdrEngine.MaxImagePixels)
            return Results.BadRequest(new EnhanceError("图片像素超过上限（1600 万）。"));

        // Phone photos usually carry an EXIF rotation: ApplyOrientation bakes it in and resets the tag in Metadata.
        // Metadata (ICC profile, Exif) is passed on to the encoder.
        using PixelBuffer buffer = ImageCodecs.Decode(data, PixelFormat.Rgba32,
            new DecodeOptions { ApplyOrientation = true }, cancellationToken);
        int width = buffer.Width, height = buffer.Height, stride = buffer.Stride;
        double decodeMs = stage.Elapsed.TotalMilliseconds;

        stage.Restart();
        GcdrEnhancer enhancer = engine.Get(backend);
        // In place: src and dst are the same RGBA buffer with the same stride.
        enhancer.Enhance(buffer.Pixels, width, height, stride, buffer.WritablePixels, stride, GcdrPixelFormat.Rgba32);
        double enhanceMs = stage.Elapsed.TotalMilliseconds;

        stage.Restart();
        using MemoryStream output = new();
        ImageCodecs.Encode(buffer.Pixels, width, height, PixelFormat.Rgba32, format, output, stride, encodeOptions, buffer.Metadata, cancellationToken);
        double encodeMs = stage.Elapsed.TotalMilliseconds;

        response.Headers["X-Decode-Ms"] = FormatMs(decodeMs);
        response.Headers["X-Enhance-Ms"] = FormatMs(enhanceMs);
        response.Headers["X-Encode-Ms"] = FormatMs(encodeMs);
        response.Headers["X-Image-Size"] = $"{width}x{height}";
        response.Headers["X-Backend"] = backend.ToString().ToLowerInvariant();
        return Results.File(output.ToArray(), contentType);
    }
    catch (NotSupportedException ex)
    {
        return Results.BadRequest(new EnhanceError(ex.Message));
    }
    catch (InvalidDataException ex)
    {
        return Results.BadRequest(new EnhanceError(ex.Message));
    }
    catch (Exception ex)
    {
        return Results.Json(new EnhanceError(ex.Message), statusCode: StatusCodes.Status500InternalServerError);
    }
}

static bool TryGetEncoder(string? format, out ImageFormat imageFormat, out EncodeOptions? options, out string contentType)
{
    (imageFormat, options, contentType) = (string.IsNullOrWhiteSpace(format) ? "jpeg" : format.Trim().ToLowerInvariant()) switch
    {
        "webp" => (ImageFormat.Webp, (EncodeOptions?)new WebpLossyEncodeOptions { Quality = 90 }, "image/webp"),
        "jpeg" or "jpg" => (ImageFormat.Jpeg, new JpegEncodeOptions { Quality = 90 }, "image/jpeg"),
        "png" => (ImageFormat.Png, (EncodeOptions?)null, "image/png"),
        _ => (ImageFormat.Unknown, null, ""),
    };
    return imageFormat != ImageFormat.Unknown;
}

static string FormatMs(double ms) => ms.ToString("0.0", CultureInfo.InvariantCulture);

internal sealed class EnhanceUpload
{
    public IFormFile? File { get; init; }

    [Description("auto（GPU 优先：Vulkan / Metal，无 GPU 时回退 CPU）或 cpu，默认 auto。")]
    public string? Backend { get; init; }

    [Description("输出格式：jpeg（默认）、webp 或 png。")]
    public string? Format { get; init; }
}

internal sealed record BackendList(string[] Backends, string Default);

internal sealed record EnhanceError(string Error);
