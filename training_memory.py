# SPDX-License-Identifier: Apache-2.0
"""Conservative admission plus the existing fixed device safety limits."""
import gc
from pathlib import Path

GIB = 1024**3


def violation(memory, elapsed):
    if memory['available'] < 4*GIB:
        return 'available_memory_below_4_gib'
    if memory['swap_delta'] > 512*1024**2:
        return 'swap_growth_above_512_mib'
    if elapsed > 900:
        return 'segment_above_900_seconds'
    return None


def admission_bytes(base_dir, resume_path=None):
    # The mapped backbone still needs resident GPU storage; keep the existing 4 GiB reserve
    # plus optimizer loading and a small workspace before admitting any model allocation.
    backbone = sum(p.stat().st_size for p in Path(base_dir).glob('*.safetensors'))
    if not backbone:
        raise ValueError('base_weight_sizes_unavailable')
    resume = Path(resume_path).stat().st_size if resume_path else 0
    return backbone + 4*GIB + resume + 512*1024**2


def restore_optimizer(torch, optimizer, path, step, protocol_sha256):
    # Mapping avoids an eager full CPU copy while optimizer.load_state_dict migrates moments.
    resume = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    try:
        if resume.get('step') != step or resume.get('protocol_sha256') != protocol_sha256:
            raise ValueError('resume_state_mismatch')
        optimizer.load_state_dict(resume['optimizer'])
        torch.set_rng_state(resume['cpu_rng'])
        torch.mps.set_rng_state(resume['mps_rng'])
    finally:
        del resume
        gc.collect()
