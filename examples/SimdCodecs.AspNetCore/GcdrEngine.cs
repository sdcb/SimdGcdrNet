using System.Collections.Concurrent;
using Sdcb.SimdGcdrNet;
using Sdcb.SimdGcdrNet.Models;

namespace SimdCodecs.AspNetCore;

/// <summary>One thread-safe <see cref="GcdrEnhancer"/> per backend, created on first use.</summary>
public sealed class GcdrEngine : IDisposable
{
    public const long MaxImagePixels = 16_000_000;
    public const string DefaultBackend = "auto";
    public static readonly string[] Backends = ["auto", "cpu"];

    private readonly ConcurrentDictionary<GcdrBackend, Lazy<GcdrEnhancer>> _enhancers = new();

    public static bool TryParseBackend(string? value, out GcdrBackend backend)
    {
        backend = (string.IsNullOrWhiteSpace(value) ? DefaultBackend : value.Trim().ToLowerInvariant()) switch
        {
            "auto" => GcdrBackend.Auto,
            "cpu" => GcdrBackend.Cpu,
            _ => (GcdrBackend)(-1),
        };
        return backend >= 0;
    }

    public GcdrEnhancer Get(GcdrBackend backend) => _enhancers.GetOrAdd(backend, static b => new Lazy<GcdrEnhancer>(
        () => GcdrEnhancer.FromBytes(GcdrModels.GcNet, GcdrModels.DrNet, b),
        LazyThreadSafetyMode.ExecutionAndPublication)).Value;

    public void Dispose()
    {
        foreach (Lazy<GcdrEnhancer> lazy in _enhancers.Values)
            if (lazy.IsValueCreated) lazy.Value.Dispose();
        _enhancers.Clear();
    }
}
