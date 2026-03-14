"""Apple GPU Runtime - Unified API for GPU operations on Apple Silicon."""

try:
    from applegpu_runtime.applegpu_runtime import version  # native module
except ImportError:
    # Fallback when native extension is not built
    def version() -> str:
        return "0.1.0-stub"

__version__ = version()
__all__ = ["version"]
