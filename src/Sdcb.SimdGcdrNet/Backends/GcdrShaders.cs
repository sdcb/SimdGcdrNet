using System.Reflection;

namespace Sdcb.SimdGcdrNet.Backends;

/// <summary>The pipeline glue shaders (pixel conversion, division correction) embedded in this assembly.</summary>
internal static class GcdrShaders
{
    public static Assembly Owner { get; } = typeof(GcdrShaders).Assembly;
}
