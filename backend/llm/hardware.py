"""Hardware detection and training-profile recommendation (section 46)."""
from __future__ import annotations

import os
import platform

from .backend import selected_backend, torch_available
from .config import MODEL_PROFILES, TwinSimConfig, count_parameters_analytic


def detect_hardware() -> dict:
    info = {
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": None,
        "gpu": None,
        "vram_gb": None,
        "cuda_available": False,
        "torch_installed": torch_available(),
        "backend": selected_backend(),
    }
    try:
        import psutil  # optional
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                kb = int(f.readline().split()[1])
            info["ram_gb"] = round(kb / 2**20, 1)
        except Exception:
            pass
    if info["torch_installed"]:
        try:
            import torch
            if torch.cuda.is_available():
                info["cuda_available"] = True
                props = torch.cuda.get_device_properties(0)
                info["gpu"] = props.name
                info["vram_gb"] = round(props.total_memory / 2**30, 1)
        except Exception:
            pass
    return info


def recommend_profile(hw: dict | None = None) -> dict:
    """Pick the largest profile whose rough memory estimate fits; explain why."""
    hw = hw or detect_hardware()
    vram = hw.get("vram_gb")
    if hw.get("cuda_available") and vram:
        if vram >= 16:
            name, batch, accum = "research-large", 16, 8
        elif vram >= 6:
            name, batch, accum = "research", 8 if vram < 10 else 32, 8 if vram < 10 else 4
        else:
            name, batch, accum = "dev", 16, 1
        reason = f"CUDA GPU with {vram} GB VRAM detected"
    else:
        name, batch, accum = "dev", 16, 1
        reason = "no CUDA GPU detected - CPU training, small development model"
    cfg = TwinSimConfig(**MODEL_PROFILES[name])
    return {
        "profile": name,
        "reason": reason,
        "layers": cfg.num_layers,
        "heads": cfg.num_heads,
        "embedding_dim": cfg.embedding_dim,
        "context_length": cfg.context_length,
        "vocab_size": cfg.vocab_size,
        "batch_size": batch,
        "grad_accum_steps": accum,
        "approx_parameters": count_parameters_analytic(cfg),
        "mixed_precision": bool(hw.get("cuda_available")),
    }
