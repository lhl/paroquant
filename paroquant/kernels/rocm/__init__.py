from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline


_ROCM_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdint>

namespace {

template <typename scalar_t>
__device__ inline float scalar_to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <typename scalar_t>
__device__ inline scalar_t float_to_scalar(float v) {
  return static_cast<scalar_t>(v);
}

template <typename scalar_t>
__global__ void paroquant_rotate_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ out,
    const int16_t* __restrict__ pairs,
    const scalar_t* __restrict__ theta,
    const scalar_t* __restrict__ scales,
    int64_t tokens,
    int64_t hidden,
    int group_size,
    int krot,
    bool has_scales) {
  const int row = blockIdx.x;
  const int group = blockIdx.y;
  const int lane = threadIdx.x;
  const int half_group = group_size / 2;
  extern __shared__ float buf[];

  if (row >= tokens || lane >= half_group) {
    return;
  }

  const int base = group * group_size;
  const int col0 = base + lane;
  const int col1 = col0 + half_group;
  const int64_t row_base = static_cast<int64_t>(row) * hidden;

  float v0 = scalar_to_float(x[row_base + col0]);
  float v1 = scalar_to_float(x[row_base + col1]);
  if (has_scales) {
    v0 *= scalar_to_float(scales[col0]);
    v1 *= scalar_to_float(scales[col1]);
  }
  buf[lane] = v0;
  buf[lane + half_group] = v1;
  __syncthreads();

  for (int r = 0; r < krot; ++r) {
    const int pair_base = r * static_cast<int>(hidden) + base + 2 * lane;
    const int i = static_cast<int>(pairs[pair_base + 0]);
    const int j = static_cast<int>(pairs[pair_base + 1]);
    const float angle = scalar_to_float(theta[r * (hidden / 2) + group * half_group + lane]);
    float s, c;
    sincosf(angle, &s, &c);
    const float xi = buf[i];
    const float xj = buf[j];
    buf[i] = fmaf(xj, s, xi * c);
    buf[j] = fmaf(xi, -s, xj * c);
    __syncthreads();
  }

  out[row_base + col0] = float_to_scalar<scalar_t>(buf[lane]);
  out[row_base + col1] = float_to_scalar<scalar_t>(buf[lane + half_group]);
}

} // namespace

torch::Tensor rotate(
    torch::Tensor x,
    torch::Tensor pairs,
    torch::Tensor theta,
    c10::optional<torch::Tensor> scales_opt,
    int64_t group_size) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor");
  TORCH_CHECK(pairs.is_cuda(), "pairs must be a CUDA/HIP tensor");
  TORCH_CHECK(theta.is_cuda(), "theta must be a CUDA/HIP tensor");
  TORCH_CHECK(x.dim() >= 2, "x must have a trailing hidden dimension");
  TORCH_CHECK(pairs.dim() == 2, "pairs must be [krot, hidden]");
  TORCH_CHECK(theta.dim() == 2, "theta must be [krot, hidden // 2]");
  TORCH_CHECK(pairs.scalar_type() == at::kShort, "pairs must be int16");
  TORCH_CHECK(x.scalar_type() == theta.scalar_type(), "x and theta dtypes must match");
  TORCH_CHECK(group_size > 0 && group_size % 2 == 0, "group_size must be a positive even number");

  const int64_t hidden = x.size(-1);
  const int64_t tokens = x.numel() / hidden;
  const int64_t krot = pairs.size(0);
  TORCH_CHECK(hidden % group_size == 0, "hidden must be divisible by group_size");
  TORCH_CHECK(pairs.size(1) == hidden, "pairs second dimension must equal hidden");
  TORCH_CHECK(theta.size(0) == krot, "theta first dimension must equal pairs first dimension");
  TORCH_CHECK(theta.size(1) == hidden / 2, "theta second dimension must equal hidden // 2");

  bool has_scales = scales_opt.has_value() && scales_opt.value().defined() && scales_opt.value().numel() > 0;
  torch::Tensor scales;
  if (has_scales) {
    scales = scales_opt.value();
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA/HIP tensor");
    TORCH_CHECK(scales.numel() == hidden, "scales must contain one value per hidden channel");
    TORCH_CHECK(scales.scalar_type() == x.scalar_type(), "scales dtype must match x");
  }

  at::cuda::CUDAGuard device_guard(x.device());
  auto x_contig = x.contiguous().view({tokens, hidden});
  auto pairs_contig = pairs.contiguous();
  auto theta_contig = theta.contiguous();
  torch::Tensor scales_contig = has_scales ? scales.contiguous().view({hidden}) : torch::Tensor();
  auto out_2d = torch::empty_like(x_contig);

  const dim3 grid(tokens, hidden / group_size);
  const dim3 block(group_size / 2);
  const size_t shared_bytes = static_cast<size_t>(group_size) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x_contig.scalar_type(), "paroquant_rocm_rotate", [&] {
    const scalar_t* scales_ptr = has_scales ? scales_contig.data_ptr<scalar_t>() : nullptr;
    paroquant_rotate_kernel<scalar_t><<<grid, block, shared_bytes, stream>>>(
        x_contig.data_ptr<scalar_t>(),
        out_2d.data_ptr<scalar_t>(),
        pairs_contig.data_ptr<int16_t>(),
        theta_contig.data_ptr<scalar_t>(),
        scales_ptr,
        tokens,
        hidden,
        static_cast<int>(group_size),
        static_cast<int>(krot),
        has_scales);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out_2d.view(x.sizes());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rotate", &rotate, "ParoQuant pairwise rotation (ROCm/HIP)");
}
'''


def _rotation_build_directory(arch: str) -> Path:
    cache_root = Path.home() / ".cache" / "paroquant" / "torch_extensions"
    abi_tag = (
        f"py{sys.version_info.major}{sys.version_info.minor}_"
        f"torch{torch.__version__}_"
        f"hip{torch.version.hip or 'none'}_"
        f"{arch}"
    )
    abi_tag = "".join(c if c.isalnum() else "_" for c in abi_tag)
    build_dir = cache_root / "paroquant_rotation_rocm" / abi_tag
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir


def _load_rotation_extension():
    if torch.version.hip is None:
        raise RuntimeError("ParoQuant ROCm rotation requested, but this PyTorch build has no HIP support")
    if not torch.cuda.is_available():
        raise RuntimeError("ParoQuant ROCm rotation requires a visible HIP device")

    arch = os.environ.get("PAROQUANT_HIP_ARCH")
    if arch is None:
        # A visible HIP device is guaranteed by the checks above.
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    os.environ.setdefault("PYTORCH_ROCM_ARCH", arch)
    build_dir = _rotation_build_directory(arch)
    # On NixOS the bundled pip-ROCm clang can't find the standard C++/glibc and
    # ROCm Thrust headers. Discover them from the nix store (each overridable via
    # the matching PAROQUANT_* env var) and pass them as -isystem includes. On
    # non-NixOS systems nothing matches and the toolchain's default search paths
    # are used unchanged.
    import glob as _glob

    def _discover(env_var: str, pattern: str) -> str | None:
        override = os.environ.get(env_var)
        if override:
            return override
        hits = _glob.glob(pattern)
        return sorted(hits)[-1] if hits else None

    nix_cxx = _discover("PAROQUANT_GCC_CXX_INCLUDE", "/nix/store/*gcc-prefix*/include/c++")
    nix_glibc = _discover("PAROQUANT_GLIBC_INCLUDE", "/nix/store/*glibc*-dev/include")
    nix_thrust = _discover("PAROQUANT_ROCTHRUST_INCLUDE", "/nix/store/*rocthrust*/include")
    nix_rocprim = _discover("PAROQUANT_ROCPRIM_INCLUDE", "/nix/store/*rocprim*/include")

    # Stub headers for ROCm libs not installed on NixOS (hipsparse, hipblas,
    # hipblaslt, hipsolver), needed by ATen/hip/HIPContextLight.h but not by the
    # paroquant kernel itself.
    _stubs = Path(__file__).parents[3] / "rocm_stubs"
    nix_rocm_stubs = str(_stubs) if _stubs.is_dir() else None

    extra_cxx_includes: list[str] = []
    if nix_cxx:
        extra_cxx_includes.append(f"-isystem{nix_cxx}")
        # Target-triple subdir holds bits/c++config.h (triple varies by host).
        triples = _glob.glob(f"{nix_cxx}/*-linux-gnu")
        if triples:
            extra_cxx_includes.append(f"-isystem{sorted(triples)[-1]}")
    for inc in (nix_glibc, nix_thrust, nix_rocprim, nix_rocm_stubs):
        if inc:
            extra_cxx_includes.append(f"-isystem{inc}")

    # Link-time search path for system ROCm libs (not under /opt/rocm on NixOS).
    extra_ldflags: list[str] = []
    lib_dir = os.environ.get("PAROQUANT_ROCM_LIB_DIR", "/run/current-system/sw/lib")
    if Path(lib_dir).is_dir():
        extra_ldflags += [f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"]

    # Ensure ninja (installed in the venv) is on PATH for torch's cpp_extension.
    _this_python_bin = str(Path(sys.executable).parent)
    _cur_path = os.environ.get("PATH", "")
    if _this_python_bin not in _cur_path.split(os.pathsep):
        os.environ["PATH"] = _this_python_bin + os.pathsep + _cur_path

    # torch.utils.cpp_extension.ROCM_HOME is resolved at import time via
    # shutil.which('hipcc').  When the venv's bin isn't in PATH at import time,
    # it falls back to the system NixOS hipcc which lacks the ROCm device library.
    # Patch ROCM_HOME to point at the venv root so the venv's hipcc wrapper is used.
    import torch.utils.cpp_extension as _tce
    _venv_root = str(Path(sys.executable).parent.parent)
    _venv_hipcc = Path(_venv_root) / "bin" / "hipcc"
    if _venv_hipcc.exists() and _tce.ROCM_HOME != _venv_root:
        _tce.ROCM_HOME = _venv_root
        _tce.HIP_HOME = os.path.join(_venv_root, "hip")
        _tce.IS_HIP_EXTENSION = True

    load_kwargs = dict(
        name=f"paroquant_rotation_rocm_{arch}",
        cpp_sources="",
        cuda_sources=_ROCM_SRC,
        build_directory=str(build_dir),
        extra_cflags=["-O3", "-std=c++17"] + extra_cxx_includes,
        extra_cuda_cflags=["-O3", f"--offload-arch={arch}", "-mcumode"] + extra_cxx_includes,
        extra_ldflags=extra_ldflags,
        with_cuda=True,
        verbose=bool(int(os.environ.get("PAROQUANT_VERBOSE_BUILD", "0"))),
    )
    try:
        return load_inline(**load_kwargs)
    except Exception:
        shutil.rmtree(build_dir, ignore_errors=True)
        build_dir.mkdir(parents=True, exist_ok=True)
        return load_inline(**load_kwargs)


_C = _load_rotation_extension()

try:
    _DEF_LIB = torch.library.Library("rotation", "DEF")
    _DEF_LIB.define("rotate(Tensor x, Tensor idx_ij, Tensor theta, Tensor? scales=None, int group_size=128) -> Tensor")
except RuntimeError:
    _DEF_LIB = None

def _rotate_impl(x, idx_ij, theta, scales=None, group_size=128):
    return _C.rotate(x, idx_ij, theta, scales, int(group_size))


_IMPL_LIB = torch.library.Library("rotation", "IMPL", "CUDA")
_IMPL_LIB.impl("rotate", _rotate_impl)

try:

    @torch.library.register_fake("rotation::rotate")
    def _fake_kernel(x, idx_ij, theta, scales=None, group_size=128):
        return torch.empty_like(x)

except RuntimeError:
    pass

from .autograd import RotateTensorFunc, scaled_pairwise_rotation  # noqa: E402

__all__ = ["scaled_pairwise_rotation", "RotateTensorFunc"]
