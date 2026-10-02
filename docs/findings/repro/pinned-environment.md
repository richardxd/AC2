# E1 pinned environment on Ada

## Purpose

Record how the exact paper software stack was installed without altering a system toolkit or other project's environment.

The project venv is `/scratch/richard1xur/ac2/venv` (Python 3.12.14). `source repro/env.sh; INSTALL_FA=skip bash install.sh` installed the original pinned stack. The system has CUDA 12.1 and 13.0, so a local CUDA 12.9 toolkit was assembled from NVIDIA redistributables `cuda_nvcc` 12.9.86, `cuda_cudart` 12.9.79 and `cuda_cccl` 12.9.27, each SHA256-checked against [NVIDIA's 12.9.1 manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json). Components and receipt live in `/scratch/richard1xur/ac2/cuda-12.9/`; the `toolkit/` directory links their files.

Build command after sourcing the environment: `FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=16 NVCC_THREADS=2 INSTALL_FA=sdist FA_ONLY=1 bash install.sh`. Flash-attn's setup reads its own architecture variable; the sm80 cubins work on Ada sm89. The first build was interrupted by the supervisor; the second succeeded. Logs: `runs/e1/flash-attn-build-02.log` and `runs/e1/install-base.log`.

Acceptance: [verification script](../../../repro/verify_environment.py), [UUID-verified GPU receipt](../../../repro/receipts/e1-acceptance-uuid.json), [package freeze](../../../repro/receipts/e1-freeze.txt). Real forward/backward kernel checks passed on physical GPUs 1–7, maximum absolute difference versus Torch SDPA 0.00048828125; all gradients finite. Before the final run, Torch UUIDs were matched to nvidia-smi GPU1–7 UUIDs, excluding GPU0. The first receipt inferred that mapping; independent review caught the gap and triggered this explicit verification. This verifies the library/kernel path, not training or distributed execution (E6 remains separate).

Research assessment: no model or algorithm changes. ML-engineering assessment: native driver 580.95.05 supports this CUDA stack; project-local compiler avoids changing shared system state. Project scratch cap 45 GB; measured 26 GB after build.
