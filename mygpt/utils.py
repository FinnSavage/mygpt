"""Hardware detection and the portability glue, so the rest of the code doesn't have to care."""

import contextlib
import platform

import torch


def get_device() -> str:
    if torch.cuda.is_available():          # NVIDIA, or AMD via ROCm
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel Arc / Core Ultra
        return "xpu"
    if torch.backends.mps.is_available():  # Apple Silicon
        return "mps"
    return "cpu"


def device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "xpu":
        return torch.xpu.get_device_name(0)
    if device == "mps":
        return "Apple Silicon GPU"
    return platform.processor() or platform.machine() or "CPU"


def vram_gb(device: str) -> float | None:
    try:
        if device == "cuda":
            return torch.cuda.get_device_properties(0).total_memory / 1e9
        if device == "xpu":
            return torch.xpu.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
    return None


_precision_override: dict[str, torch.dtype] = {}


def set_precision(device: str, dtype: torch.dtype):
    """Let autotune swap in whichever precision benchmarked fastest on this machine."""
    _precision_override[device] = dtype


def reset_precision(device: str):
    _precision_override.pop(device, None)


def precision(device: str) -> tuple[torch.dtype, bool]:
    """Return (compute dtype, needs_grad_scaler)."""
    if device in _precision_override:
        dtype = _precision_override[device]
        return dtype, dtype == torch.float16
    return default_precision(device)


def default_precision(device: str) -> tuple[torch.dtype, bool]:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        return torch.float16, True       # older NVIDIA cards
    if device == "xpu":
        return torch.bfloat16, False
    return torch.float32, False          # cpu and mps: keep it simple and safe


def autocast(device: str):
    dtype, _ = precision(device)
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device, dtype=dtype)


def enable_fast_math(device: str):
    """Free speedups that don't change results in any way that matters here."""
    if device == "cuda":
        # TF32 runs float32 matrix math on tensor cores (Ampere/RTX 30 and newer)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def sync(device: str):
    """Wait for queued GPU work to finish, so timings are real."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    elif device == "xpu":
        torch.xpu.synchronize()


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def maybe_compile(model, device: str, example_x, example_y, enabled: bool = True):
    """torch.compile if it actually works here, otherwise plain eager mode.

    Compilation happens lazily on the first forward/backward pass, so we run
    one tiny test step inside a try block instead of trusting torch.compile().
    """
    if not enabled:
        return model, "off"
    if device not in ("cuda", "xpu") or not triton_available():
        return model, "unavailable (needs a GPU and Triton)"
    try:
        compiled = torch.compile(model)
        with autocast(device):
            _, loss = compiled(example_x, example_y)
        loss.backward()
        model.zero_grad(set_to_none=True)
        return compiled, "on"
    except Exception as e:
        model.zero_grad(set_to_none=True)
        return model, f"failed, using eager ({type(e).__name__})"


def is_oom(err: BaseException) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(err, RuntimeError) and "out of memory" in str(err).lower()


def free_memory(device: str):
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "xpu":
        torch.xpu.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def doctor() -> list[tuple[str, str, bool, str]]:
    """System check rows: (label, value, ok, hint)."""
    rows = []
    rows.append(("Python", platform.python_version(), True, ""))
    rows.append(("OS", f"{platform.system()} {platform.release()}", True, ""))
    rows.append(("PyTorch", torch.__version__, True, ""))

    device = get_device()
    gpu_ok = device != "cpu"
    rows.append((
        "Device", f"{device} ({device_name(device)})", gpu_ok,
        "" if gpu_ok else "No GPU detected. Training works but will be slow. "
        "If you have a GPU, rerun install.py, which picks the right PyTorch build for it.",
    ))

    vram = vram_gb(device)
    if vram is not None:
        rows.append(("GPU memory", f"{vram:.1f} GB", True, ""))

    dtype, _ = precision(device)
    rows.append(("Precision", str(dtype).replace("torch.", ""), True, ""))

    if device in ("cuda", "xpu"):
        tri = triton_available()
        rows.append((
            "torch.compile", "available" if tri else "not available", tri,
            "" if tri else "Optional speedup. On Windows with NVIDIA, try: pip install triton-windows",
        ))

    from .config import auto_preset
    rows.append(("Suggested model size", auto_preset(device, vram), True, ""))
    return rows


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"
