#!/usr/bin/env python3
"""One-step installer for mygpt.

Creates a .venv, picks the right PyTorch build for your hardware, installs
mygpt into it, runs a system check, and adds a global `mygpt` command so you
never have to activate the environment. Run with:

    python install.py [--cpu] [--skip-torch] [--no-shortcut]
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

# Keeps our prints in order relative to the subprocess calls below (pip, the
# re-exec'd venv python, etc.) when stdout isn't a terminal, e.g. redirected
# to a log file. Without this, each process's buffer can flush at a
# different time and interleave output out of order.
sys.stdout.reconfigure(line_buffering=True)

MIN_PY = (3, 10)
MAX_PY = (3, 14)

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"

NVIDIA_CUDA_TAGS = ["cu130", "cu126", "cu124", "cu121"]
AMD_ROCM_TAGS = ["rocm6.4", "rocm6.3", "rocm6.2"]


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def run(cmd):
    print("> " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def check_python():
    v = sys.version_info
    if not (MIN_PY <= v[:2] <= MAX_PY):
        print(f"mygpt needs Python {MIN_PY[0]}.{MIN_PY[1]} to {MAX_PY[0]}.{MAX_PY[1]}, found {v.major}.{v.minor}.")
        print("Install a supported Python from https://www.python.org/downloads/ and run this again.")
        sys.exit(1)


def ensure_venv_and_reexec(argv):
    """Create .venv if missing, then re-run this script with the venv's python."""
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        return
    if not VENV_DIR.exists():
        print(f"Creating virtual environment at {VENV_DIR}...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    py = venv_python()
    print("Switching to the virtual environment...")
    result = subprocess.run([str(py), str(Path(__file__).resolve()), *argv])
    sys.exit(result.returncode)


def detect_gpu():
    """Return one of 'nvidia', 'amd', 'intel', 'mps', or None."""
    system = platform.system()
    machine = platform.machine()

    if system == "Darwin" and machine == "arm64":
        return "mps"

    out = ""
    if system == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=15,
            ).stdout.lower()
        except Exception:
            pass
    else:
        try:
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=15).stdout.lower()
        except Exception:
            pass
        if not out:
            try:
                lspci_out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=15).stdout.lower()
                # lspci lists every PCI device, including chipset/platform entries like
                # "Host bridge: Advanced Micro Devices, Inc. [AMD] ..." on any Ryzen
                # machine, GPU or not. Keep only the actual display-adapter lines.
                out = "\n".join(
                    line for line in lspci_out.splitlines()
                    if "vga" in line or "3d" in line or "display" in line
                )
            except Exception:
                pass

    if "nvidia" in out:
        return "nvidia"
    if "amd" in out or "radeon" in out or "advanced micro devices" in out:
        return "amd"
    if "intel" in out and "arc" in out:
        return "intel"
    return None


def _sees_gpu(python) -> bool:
    check = subprocess.run(
        [str(python), "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True,
    )
    return check.stdout.strip() == "True"


def _uninstall_torch(python):
    """pip treats an unpinned "torch" as already satisfied no matter which
    --index-url comes next, so a retry with a different CUDA/ROCm tag (or the
    final CPU fallback) needs the previous build removed first, or it's a no-op."""
    print("> " + " ".join([str(python), "-m", "pip", "uninstall", "-y", "torch"]))
    subprocess.run([str(python), "-m", "pip", "uninstall", "-y", "torch"])


def install_torch(python, gpu, force_cpu):
    cpu_cmd = [str(python), "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"]

    if force_cpu or gpu is None:
        run(cpu_cmd)
        return
    if gpu == "mps":
        run([str(python), "-m", "pip", "install", "torch"])
        return
    if gpu == "intel":
        run([str(python), "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/xpu"])
        return
    if gpu == "amd":
        if platform.system() == "Windows":
            print("AMD GPU detected. RX 7000/9000 and Ryzen AI cards run PyTorch natively on Windows")
            print("through AMD's ROCm wheels. See AMD's \"Use ROCm on Radeon and Ryzen\" Windows how-to:")
            print("  https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/howto_windows.html")
            print("Installing the CPU build for now; pass --skip-torch if you'd rather install the ROCm build yourself.")
            run(cpu_cmd)
            return
        # Linux: PyTorch publishes ROCm wheels directly, same pattern as the CUDA tags below.
        if _sees_gpu(python):
            print("PyTorch already set up for your GPU, skipping.")
            return
        for tag in AMD_ROCM_TAGS:
            _uninstall_torch(python)
            try:
                run([str(python), "-m", "pip", "install", "torch", "--index-url", f"https://download.pytorch.org/whl/{tag}"])
            except subprocess.CalledProcessError:
                continue
            if _sees_gpu(python):  # ROCm builds still report through torch.cuda
                return
            print(f"torch ({tag}) installed but can't see the GPU, trying an older ROCm build.")
        print("No ROCm build could see the GPU. Installing the CPU build instead.")
        _uninstall_torch(python)
        run(cpu_cmd)
        return

    # nvidia: try the newest CUDA build first, but a bad install can still succeed on pip's end
    # while silently running on CPU (e.g. an old driver that the new CUDA build doesn't support),
    # so actually check torch.cuda.is_available() before accepting each one.
    if _sees_gpu(python):
        print("PyTorch already set up for your GPU, skipping.")
        return
    for tag in NVIDIA_CUDA_TAGS:
        _uninstall_torch(python)
        try:
            run([str(python), "-m", "pip", "install", "torch", "--index-url", f"https://download.pytorch.org/whl/{tag}"])
        except subprocess.CalledProcessError:
            continue
        if _sees_gpu(python):
            return
        print(f"torch ({tag}) installed but can't see the GPU, trying an older CUDA build.")
    print("No CUDA build could see the GPU. Installing the CPU build instead.")
    print("If you have an NVIDIA GPU, updating your driver and re-running this script may fix it.")
    _uninstall_torch(python)
    run(cpu_cmd)


SHORTCUT_MARKER = "# added by mygpt install.py"


def _on_path(folder: Path) -> bool:
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry and os.path.normcase(os.path.abspath(os.path.expanduser(entry))) == os.path.normcase(str(folder)):
            return True
    return False


def _shortcut_unix() -> bool:
    """Link ~/.local/bin/mygpt to the venv's mygpt. Returns True if usable in this terminal right now."""
    home = Path.home()
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = VENV_DIR / "bin" / "mygpt"
    link = bin_dir / "mygpt"

    if link.is_symlink() or link.exists():
        if not link.is_symlink():
            raise RuntimeError(f"{link} already exists and isn't a mygpt shortcut, so it was left alone")
        link.unlink()  # an old shortcut, maybe from a previous copy of the project
    link.symlink_to(target)
    print(f"\nAdded the 'mygpt' command: {link} -> {target}")

    if _on_path(bin_dir):
        return True

    # ~/.local/bin isn't on PATH (the default on macOS), so add it to the shell's startup file
    shell = Path(os.environ.get("SHELL", "")).name
    if shell == "zsh":
        profile = home / ".zprofile"
    elif shell == "bash":
        profile = home / (".bash_profile" if platform.system() == "Darwin" else ".bashrc")
    elif shell == "fish":
        print('Fish shell: run  fish_add_path ~/.local/bin  once to finish setting up the command.')
        return False
    else:
        profile = home / ".profile"
    existing = profile.read_text(encoding="utf-8") if profile.exists() else ""
    if SHORTCUT_MARKER not in existing:
        with profile.open("a", encoding="utf-8") as f:
            f.write(f'\n{SHORTCUT_MARKER}\nexport PATH="$HOME/.local/bin:$PATH"\n')
        print(f"Added ~/.local/bin to your PATH in {profile}")
    return False


def _shortcut_windows() -> bool:
    """Put mygpt.cmd in a per-user folder and add that folder to the user PATH."""
    import ctypes
    import winreg

    bin_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "mygpt" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe = VENV_DIR / "Scripts" / "mygpt.exe"
    (bin_dir / "mygpt.cmd").write_text(f'@echo off\r\n"{exe}" %*\r\n', encoding="utf-8")
    print(f"\nAdded the 'mygpt' command: {bin_dir / 'mygpt.cmd'}")

    if _on_path(bin_dir):
        return True

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_READ | winreg.KEY_WRITE) as key:
        try:
            current, kind = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            current, kind = "", winreg.REG_EXPAND_SZ
        if kind not in (winreg.REG_SZ, winreg.REG_EXPAND_SZ):
            raise RuntimeError("your user PATH has an unexpected format, so it was left alone")
        parts = [p for p in current.split(";") if p]
        if not any(os.path.normcase(p) == os.path.normcase(str(bin_dir)) for p in parts):
            parts.append(str(bin_dir))
            winreg.SetValueEx(key, "Path", 0, kind, ";".join(parts))
            print(f"Added {bin_dir} to your user PATH")

    # tell Windows the environment changed, so newly opened terminals pick it up
    result = ctypes.c_ulong()
    ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "Environment", 0x0002, 5000, ctypes.byref(result))
    return False


def install_shortcut() -> bool:
    if platform.system() == "Windows":
        return _shortcut_windows()
    return _shortcut_unix()


def main():
    parser = argparse.ArgumentParser(description="Install mygpt and the right PyTorch build for this machine.")
    parser.add_argument("--cpu", action="store_true", help="Force the CPU build of PyTorch, even if a GPU is detected.")
    parser.add_argument("--skip-torch", action="store_true", help="Don't install PyTorch (install it yourself first).")
    parser.add_argument("--no-shortcut", action="store_true",
                        help="Don't add a global 'mygpt' command; you'll activate .venv yourself instead.")
    args = parser.parse_args()

    check_python()
    ensure_venv_and_reexec(sys.argv[1:])

    python = sys.executable
    run([python, "-m", "pip", "install", "--upgrade", "pip"])

    gpu = None
    if not args.skip_torch:
        gpu = detect_gpu()
        print(f"Detected GPU: {gpu or 'none'}")
        install_torch(python, gpu, args.cpu)
    else:
        print("Skipping PyTorch install (--skip-torch). Install it yourself, then re-run this script.")

    run([python, "-m", "pip", "install", "-e", str(ROOT)])

    if platform.system() == "Windows" and gpu == "nvidia":
        try:
            run([python, "-m", "pip", "install", "triton-windows"])
        except subprocess.CalledProcessError:
            print("triton-windows install failed; torch.compile will fall back to eager mode. Continuing.")

    print()
    run([python, "-m", "mygpt", "doctor"])

    ready_now = False
    if not args.no_shortcut:
        try:
            ready_now = install_shortcut()
        except Exception as e:  # never fail the whole install over the convenience shortcut
            print(f"\nCouldn't set up the global 'mygpt' command ({e}).")
            args.no_shortcut = True

    print("\nSetup complete.")
    if args.no_shortcut:
        activate = r".venv\Scripts\activate" if platform.system() == "Windows" else "source .venv/bin/activate"
        print(f"Activate the environment in each new terminal first:\n  {activate}\nthen run: mygpt train\n")
    elif ready_now:
        print("Run this from any folder:\n  mygpt train\n")
    else:
        print("Open a NEW terminal window (this one doesn't know about the command yet), then run:\n  mygpt train\n")


if __name__ == "__main__":
    main()
