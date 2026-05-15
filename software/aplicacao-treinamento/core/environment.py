"""
Detecção de ambiente de execução.

Detecta:
  - Sistema operacional
  - Ambiente de execução: local, Google Colab, RunPod/pod
  - GPU dedicada: fabricante, nome, VRAM
  - RAM disponível
  - Versão de Python e PyTorch
"""
from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Estrutura de dados do relatório de ambiente
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    index:        int
    name:         str
    vendor:       str    # "NVIDIA" | "AMD" | "Intel" | "Outro"
    vram_total:   int    # bytes
    vram_free:    int    # bytes
    cuda_support: bool   # True se acessível via torch.cuda (NVIDIA e AMD ROCm)
    backend:      str = "unknown"  # "cuda" | "xpu" | "unknown"

    @property
    def vram_total_gb(self) -> float:
        return round(self.vram_total / (1024 ** 3), 2)

    @property
    def vram_livre_gb(self) -> float:
        return round(self.vram_free / (1024 ** 3), 2)

    @property
    def training_backend_available(self) -> bool:
        """True se há backend funcional de treino disponível (cuda ou xpu)."""
        return self.backend in ("cuda", "xpu")


@dataclass
class EnvironmentReport:
    # Sistema
    os_name:     str          # "Windows" | "Linux" | "macOS"
    os_version:  str
    python_ver:  str
    torch_ver:   Optional[str]

    # Ambiente de execução
    runtime:     str          # "local" | "colab" | "runpod" | "pod"

    # Hardware
    ram_total:   int          # bytes
    ram_free:    int          # bytes
    gpus:        list[GPUInfo] = field(default_factory=list)

    # Capacidade inferida
    cuda_available: bool = False
    training_viable: bool = False  # True se há GPU NVIDIA com CUDA

    @property
    def ram_total_gb(self) -> float:
        return round(self.ram_total / (1024 ** 3), 2)

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        return self.gpus[0] if self.gpus else None

    def imprimir_resumo(self):
        print(f"  OS       : {self.os_name} {self.os_version}")
        print(f"  Ambiente : {self.runtime}")
        print(f"  Python   : {self.python_ver}")
        print(f"  PyTorch  : {self.torch_ver or 'não instalado'}")
        print(f"  RAM      : {self.ram_total_gb} GB total")
        if self.gpus:
            for g in self.gpus:
                backend_tag = g.backend.upper() if g.training_backend_available else "sem backend"
                print(f"  GPU [{g.index}] : {g.vendor} {g.name} — "
                      f"{g.vram_total_gb} GB VRAM ({backend_tag})")
        else:
            print("  GPU      : nenhuma GPU dedicada detectada")
        status = (
            "Pronto para treino (NVIDIA + CUDA + ≥4 GB VRAM)"
            if self.training_viable
            else "Bloqueado: requer GPU NVIDIA com CUDA e mínimo de 4 GB de VRAM"
        )
        print(f"  Status   : {status}")


# ─────────────────────────────────────────────────────────────────────────────
# Funções de detecção
# ─────────────────────────────────────────────────────────────────────────────

def _detect_runtime() -> str:
    """Detecta se está rodando em Colab, RunPod ou ambiente local."""
    # Google Colab
    if "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
        return "colab"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass

    # RunPod
    if "RUNPOD_POD_ID" in os.environ or "RUNPOD_API_KEY" in os.environ:
        return "runpod"

    # Pods genéricos (detecção heurística por variáveis de ambiente comuns)
    pod_hints = ("VAST_CONTAINERLABEL", "VAST_API_KEY", "PAPERSPACE_CLUSTER_ID",
                 "POD_ID", "JUPYTER_IMAGE_SPEC")
    if any(h in os.environ for h in pod_hints):
        return "pod"

    return "local"


def _detect_ram() -> tuple[int, int]:
    """Retorna (total, free) em bytes. Requer psutil."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total, vm.available
    except ImportError:
        return 0, 0


def _detect_gpus() -> list[GPUInfo]:
    """
    Detecta GPUs via PyTorch.

    Backends cobertos:
      - torch.cuda  : NVIDIA (CUDA) e AMD (ROCm — mapeia a mesma API)
      - torch.xpu   : Intel Arc/Xe (PyTorch 2.x com suporte Intel)

    Retorna lista vazia se nenhum backend estiver disponível.
    """
    gpus: list[GPUInfo] = []

    # ── Backend CUDA (cobre NVIDIA e AMD ROCm) ────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                name       = props.name
                vram_total = props.total_memory
                try:
                    torch.cuda.set_device(i)
                    vram_free = torch.cuda.mem_get_info(i)[0]
                except Exception:
                    vram_free = vram_total  # fallback conservador

                vendor = _infer_vendor(name)
                gpus.append(GPUInfo(
                    index        = i,
                    name         = name,
                    vendor       = vendor,
                    vram_total   = vram_total,
                    vram_free    = vram_free,
                    cuda_support = True,
                    backend      = "cuda",
                ))
    except ImportError:
        pass

    # ── Backend XPU (Intel Arc/Xe via torch.xpu, PyTorch 2.x+) ──────────────
    # Só tenta se nenhuma GPU foi detectada via CUDA para evitar duplicatas
    if not gpus:
        try:
            import torch
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                n = torch.xpu.device_count()
                for i in range(n):
                    try:
                        props      = torch.xpu.get_device_properties(i)
                        name       = getattr(props, "name", f"Intel XPU {i}")
                        vram_total = getattr(props, "total_memory", 0)
                    except Exception:
                        name       = f"Intel XPU {i}"
                        vram_total = 0
                    gpus.append(GPUInfo(
                        index        = i,
                        name         = name,
                        vendor       = "Intel",
                        vram_total   = vram_total,
                        vram_free    = vram_total,  # XPU não expõe mem_get_info
                        cuda_support = False,
                        backend      = "xpu",
                    ))
        except (ImportError, AttributeError, RuntimeError):
            pass

    return gpus


def _infer_vendor(gpu_name: str) -> str:
    name_upper = gpu_name.upper()
    if any(k in name_upper for k in ("NVIDIA", "GEFORCE", "QUADRO", "TESLA", "RTX", "GTX")):
        return "NVIDIA"
    if any(k in name_upper for k in ("AMD", "RADEON", "RX ")):
        return "AMD"
    if "INTEL" in name_upper:
        return "Intel"
    return "Outro"


# ─────────────────────────────────────────────────────────────────────────────

def detect_environment() -> EnvironmentReport:
    """Ponto de entrada principal: detecta e retorna o relatório completo."""
    torch_ver: Optional[str] = None
    cuda_available = False
    try:
        import torch
        torch_ver = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    ram_total, ram_free = _detect_ram()
    gpus = _detect_gpus()

    training_viable = any(
        g.vendor == "NVIDIA" and g.backend == "cuda" and g.vram_total_gb >= 4.0
        for g in gpus
    )

    return EnvironmentReport(
        os_name          = platform.system(),
        os_version       = platform.version(),
        python_ver       = sys.version.split()[0],
        torch_ver        = torch_ver,
        runtime          = _detect_runtime(),
        ram_total        = ram_total,
        ram_free         = ram_free,
        gpus             = gpus,
        cuda_available   = cuda_available,
        training_viable  = training_viable,
    )
