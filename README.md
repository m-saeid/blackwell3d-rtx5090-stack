# Blackwell3D Stack ⚡ RTX 5090 Edition

A reproducible, carefully tuned installation guide for running:

* PyTorch (CUDA 13.1)
* PyTorch3D (compiled from source)
* pointnet2_ops (patched for compute capability 12.0)

on:

* NVIDIA GeForce RTX 5090
* Driver 590.48.01
* CUDA Toolkit 13.1 (nvcc 13.1.115)
* Python 3.11 (virtual environment)

This guide is specifically written for cutting edge Blackwell GPUs (Compute Capability 12.0).

---

# 1. System Reference (Tested Configuration)

```
nvidia-smi
Driver Version: 590.48.01
CUDA Version: 13.1
GPU: NVIDIA GeForce RTX 5090
Memory: 32GB
```

```
nvcc --version
Cuda compilation tools, release 13.1, V13.1.115
```

Why this matters:

* RTX 5090 = Compute Capability 12.0
* CUDA 13.1 is required for proper Blackwell support
* Older CUDA toolkits will fail to compile extensions

---

# 2. Install NVIDIA Driver

Install latest production driver supporting CUDA 13.x.

Verify:

```
nvidia-smi
```

It must show CUDA Version 13.1 support.

---

# 3. Install CUDA 13.1 (Local Toolkit)

Download CUDA 13.1 from NVIDIA.

Install locally (example path used here):

```
$HOME/Desktop/saeid/venv/cuda-13.1
```

After installation, add this to your `~/.bashrc`:

```
export CUDA_HOME=$HOME/Desktop/saeid/venv/cuda-13.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Reload shell:

```
source ~/.bashrc
```

Verify:

```
nvcc --version
```

---

# 4. Create Python Environment

```
python3.11 -m venv envpy311c131
source envpy311c131/bin/activate
pip install --upgrade pip setuptools wheel
```

---

# 5. Install PyTorch (CUDA 13.1 Compatible)

Install the official PyTorch build that matches CUDA 13.x.

Example:

```
pip install torch torchvision torchaudio
```

Verify:

```
python -c "import torch; print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

---

# 6. Important: Set Compute Capability

RTX 5090 = Compute Capability 12.0

Set:

```
export TORCH_CUDA_ARCH_LIST="12.0"
```

Why?

* Prevents PyTorch from attempting unsupported architectures
* Avoids build failures for CUDA extensions
* Ensures compiled kernels target sm_120 only

---

# 7. Install PyTorch3D (From Source)

Clone PyTorch3D:

```
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
```

Build:

```
export TORCH_CUDA_ARCH_LIST="12.0"
pip install --no-build-isolation -e . 2>&1 | tee ../pytorch3d-build.log
```

Why `--no-build-isolation`?

It ensures PyTorch headers and CUDA from the current environment are used.

---

# 8. Install pointnet2_ops (Blackwell Patch Required)

This repository requires modification to build on RTX 5090.

## Step 1 — Update setup.py

Remove any reference to compute_130.

Use only:

```
extra_compile_args={
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "-gencode=arch=compute_120,code=sm_120"
    ],
}
```

Why remove compute_130?

RTX 5090 compute capability is 12.0. Adding 13.0/13.1 causes:

* Unknown CUDA arch errors
* nvcc failures
* PyTorch validation errors

## Step 2 — Update JIT Loader

In the load() call:

```
_ext = load(
    "_ext",
    sources=_ext_sources,
    extra_include_paths=[...],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_120,code=sm_120"
    ],
    with_cuda=True,
)
```

## Step 3 — Clean & Build

```
export TORCH_CUDA_ARCH_LIST="12.0"

rm -rf build pointnet2_ops/_ext pointnet2_ops.egg-info
find . -name '*.o' -delete
find . -name '*.so' -delete

pip install --no-build-isolation -e .
```

If done correctly, extension compiles successfully.

---

# 9. Smoke Tests

## PyTorch

```
python - <<EOF
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
EOF
```

## PyTorch3D

```
python - <<EOF
import pytorch3d
print("PyTorch3D OK")
EOF
```

## pointnet2_ops

Run your custom test script (fps, knn, grouping, interpolation).
All operations must execute on CUDA without errors.

---

# 10. Full Installation Script

Create `install_blackwell_stack.sh`:

```
#!/bin/bash
set -e

export CUDA_HOME=$HOME/Desktop/saeid/venv/cuda-13.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

python3.11 -m venv envpy311c131
source envpy311c131/bin/activate

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio

export TORCH_CUDA_ARCH_LIST="12.0"

# Install PyTorch3D
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install --no-build-isolation -e .
cd ..

# Install pointnet2_ops (assumes patched source)
git clone <your-pointnet2-ops-repo>
cd <your-pointnet2-ops-repo>
pip install --no-build-isolation -e .


echo "Blackwell 3D stack installation complete."
```

Note:

* CUDA driver installation cannot be scripted here.
* setup.py modifications must be applied before running this script.

---

# 11. Final Notes

This stack is intentionally tuned for:

* RTX 5090
* CUDA 13.1
* Compute Capability 12.0
* PyTorch compiled with CUDA 13.x

If you use a different GPU, adjust TORCH_CUDA_ARCH_LIST accordingly.

---

If you follow this guide precisely, you can reproduce a fully working 3D deep learning stack optimized for Blackwell GPUs.
