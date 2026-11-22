#!/usr/bin/env python3
"""
model_catalog.py — Recommended local models and whether this machine can run them.

The app shows this list in "Get more models". Each model has its download size,
the RAM it needs while answering, and a short note on what it is good for.
A model is offered only if it fits in this machine's RAM (next to the app
itself) or in the memory of an NVIDIA GPU, and there is enough free disk space
for the download.

The same checks choose the first model on a new machine: run.sh and Docker call
`python -m rag.model_catalog --recommend`, which prints the best model in
QUALITY_ORDER that fits. Weak laptops get a small model, machines with more RAM
or a GPU get a stronger one.
"""

import argparse
import ctypes
import functools
import os
import platform
import shutil
import subprocess

# ram_gb: memory the model uses while answering (4-bit, about 4k context).
# Values marked "thesis" were measured in the bachelor thesis benchmark.
CATALOG = [
    {
        "name": "qwen2.5:0.5b", "size_gb": 0.4, "ram_gb": 1.2,   # thesis: 1.15 GB
        "good_for": "Fastest and smallest. Short answers in English on old or slow laptops. Weak in other languages.",
    },
    {
        "name": "gemma2:2b", "size_gb": 1.6, "ram_gb": 2.1,
        "good_for": "Best choice for a normal 8 GB laptop. Fast and follows the language of the question well.",
    },
    {
        "name": "qwen2.5:3b", "size_gb": 1.9, "ram_gb": 3.2,     # thesis: 3.22 GB
        "good_for": "Careful with numbers and English text. The model chosen in the thesis benchmark. Often answers in English.",
    },
    {
        "name": "llama3.2:3b", "size_gb": 2.0, "ram_gb": 3.0,
        "good_for": "General assistant for English questions. Less reliable in other languages.",
    },
    {
        "name": "qwen2.5:7b", "size_gb": 4.7, "ram_gb": 5.5,
        "good_for": "Better at longer documents, reasoning and many languages. Needs about 10 GB RAM or a 6 GB GPU.",
    },
    {
        "name": "llama3:8b", "size_gb": 4.7, "ram_gb": 6.9,      # thesis: 6.85 GB
        "good_for": "Largest model from the thesis benchmark. Detailed English answers, slow without a GPU.",
    },
]

# Best first, for multilingual, grounded answers (from my comparison after the
# thesis: gemma2:2b beat qwen2.5:3b and llama3.2:3b on language and facts).
QUALITY_ORDER = ["qwen2.5:7b", "gemma2:2b", "llama3:8b", "qwen2.5:3b", "llama3.2:3b", "qwen2.5:0.5b"]

APP_RAM_GB = 2.5     # embedding model + app (measured with BAAI/bge-m3)
SYSTEM_RAM_GB = 1.5  # left for the operating system and browser
DISK_MARGIN_GB = 1.0


def total_ram_gb() -> float:
    """Installed RAM in GB (Linux, macOS, Windows), 0 if unknown."""
    try:
        system = platform.system()
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024 ** 2
        elif system == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            return int(out.stdout.strip()) / 1024 ** 3
        elif system == "Windows":
            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MemStatus()
            stat.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / 1024 ** 3
    except Exception:
        pass
    return 0.0


@functools.lru_cache(maxsize=1)
def gpu_memory_gb() -> float:
    """Memory of the largest NVIDIA GPU in GB, 0 if there is none.
    Apple Silicon shares RAM with the GPU, so the RAM check already covers it."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        values = [float(v) for v in out.stdout.split() if v.strip()]
        return max(values) / 1024 if out.returncode == 0 and values else 0.0
    except Exception:
        return 0.0


def models_dir() -> str:
    """Folder where Ollama stores models (used for the free disk space check)."""
    candidates = [
        os.environ.get("OLLAMA_MODELS"),
        os.path.expanduser("~/.ollama/models"),
        "/usr/share/ollama/.ollama/models",
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return os.path.expanduser("~")


def free_disk_gb() -> float:
    return shutil.disk_usage(models_dir()).free / 1024 ** 3


def check(model: dict, installed: set, ram_gb: float = None, disk_gb: float = None,
          gpu_gb: float = None) -> tuple:
    """
    Returns (status, message). status is one of:
    "installed", "ok", "ram", "disk".
    """
    if model["name"] in installed:
        return "installed", "Installed"
    ram_gb = total_ram_gb() if ram_gb is None else ram_gb
    disk_gb = free_disk_gb() if disk_gb is None else disk_gb
    gpu_gb = gpu_memory_gb() if gpu_gb is None else gpu_gb
    needed_ram = model["ram_gb"] + APP_RAM_GB + SYSTEM_RAM_GB
    fits_gpu = gpu_gb >= model["ram_gb"] + 0.5
    if ram_gb and needed_ram > ram_gb and not fits_gpu:
        return "ram", f"Needs about {needed_ram:.0f} GB RAM, this computer has {ram_gb:.0f} GB"
    if model["size_gb"] + DISK_MARGIN_GB > disk_gb:
        return "disk", f"Needs {model['size_gb'] + DISK_MARGIN_GB:.1f} GB free disk, {disk_gb:.1f} GB available"
    return "ok", "Can be installed"


def _by_name(name: str) -> dict:
    return next(m for m in CATALOG if m["name"] == name)


def recommended(ram_gb: float = None, gpu_gb: float = None, disk_gb: float = None,
                installed: set = frozenset()) -> str:
    """Best model in QUALITY_ORDER that this machine can run (and download, if missing)."""
    for name in QUALITY_ORDER:
        if check(_by_name(name), set(installed), ram_gb, disk_gb, gpu_gb)[0] in ("installed", "ok"):
            return name
    return QUALITY_ORDER[-1]


def best_installed(installed: list, ram_gb: float = None, gpu_gb: float = None) -> str:
    """Model the app opens with when config.json says "auto": the best installed
    catalog model that fits, else the first installed model."""
    for name in QUALITY_ORDER:
        if name in installed and check(_by_name(name), set(), ram_gb, 1e9, gpu_gb)[0] == "ok":
            return name
    return installed[0] if installed else QUALITY_ORDER[1]


def _main():
    parser = argparse.ArgumentParser(description="Hardware check and model choice")
    parser.add_argument("--recommend", action="store_true", help="print the best model for this machine")
    parser.add_argument("--pull", action="store_true", help="download the model from config.json, or the recommended one")
    parser.add_argument("--ram", type=float, help="pretend this much RAM (GB), for testing")
    parser.add_argument("--gpu", type=float, help="pretend this much GPU memory (GB), for testing")
    parser.add_argument("--disk", type=float, help="pretend this much free disk (GB), for testing")
    args = parser.parse_args()

    import json
    with open(os.path.join(os.path.dirname(__file__), "..", "config.json"), encoding="utf-8") as f:
        chosen = json.load(f).get("chosen_model", "auto")
    ram = args.ram if args.ram is not None else total_ram_gb()
    gpu = args.gpu if args.gpu is not None else gpu_memory_gb()
    model = chosen if chosen and chosen != "auto" else recommended(ram, gpu, args.disk)

    if args.pull:
        from rag.ollama_client import list_models, pull_model
        print(f"This machine: {ram:.0f} GB RAM, {gpu:.0f} GB GPU memory -> {model}", flush=True)
        if model in list_models():
            print(f"{model} is already installed")
            return
        last = None
        for text, fraction in pull_model(model):
            step = text if fraction is None else f"{text} {int(fraction * 10) * 10}%"
            if step != last:
                print(step, flush=True)
                last = step
    else:
        print(model)


if __name__ == "__main__":
    _main()
