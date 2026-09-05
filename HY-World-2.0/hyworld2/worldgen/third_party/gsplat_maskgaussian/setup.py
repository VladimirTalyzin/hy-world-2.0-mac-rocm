import glob
import os
import os.path as osp
import pathlib
import platform
import sys

from setuptools import find_packages, setup

__version__ = None
exec(open("gsplat/version.py", "r").read())

URL = "https://github.com/nerfstudio-project/gsplat"

BUILD_NO_CUDA = os.getenv("BUILD_NO_CUDA", "0") == "1"
WITH_SYMBOLS = os.getenv("WITH_SYMBOLS", "0") == "1"
LINE_INFO = os.getenv("LINE_INFO", "0") == "1"
MAX_JOBS = os.getenv("MAX_JOBS")
need_to_unset_max_jobs = False
if not MAX_JOBS:
    need_to_unset_max_jobs = True
    os.environ["MAX_JOBS"] = "10"
    print(f"Setting MAX_JOBS to {os.environ['MAX_JOBS']}")


def get_ext():
    from torch.utils.cpp_extension import BuildExtension

    return BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)


def get_extensions():
    import torch
    from torch.__config__ import parallel_info
    from torch.utils.cpp_extension import CUDAExtension

    extensions_dir = osp.join("gsplat", "cuda")
    sources = glob.glob(osp.join(extensions_dir, "csrc", "*.cu")) + glob.glob(
        osp.join(extensions_dir, "csrc", "*.cpp")
    )
    sources += [osp.join(extensions_dir, "ext.cpp")]

    undef_macros = []
    define_macros = []

    extra_compile_args = {"cxx": ["-O3"]}
    if not os.name == "nt":  # Not on Windows:
        extra_compile_args["cxx"] += ["-Wno-sign-compare"]
    extra_link_args = [] if WITH_SYMBOLS else ["-s"]

    info = parallel_info()
    if (
        "backend: OpenMP" in info
        and "OpenMP not found" not in info
        and sys.platform != "darwin"
    ):
        extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
        if sys.platform == "win32":
            extra_compile_args["cxx"] += ["/openmp"]
        else:
            extra_compile_args["cxx"] += ["-fopenmp"]
    else:
        print("Compiling without OpenMP...")

    # Compile for mac arm64
    if sys.platform == "darwin" and platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

    nvcc_flags = os.getenv("NVCC_FLAGS", "")
    nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
    nvcc_flags += ["-O3", "-std=c++17"]

    if torch.version.hip:
        # ROCm path. hipcc drives clang, which rejects the nvcc spellings the
        # CUDA branch below uses (--use_fast_math, -diag-suppress,
        # -allow-unsupported-compiler, --expt-relaxed-constexpr, -lineinfo).
        #
        # USE_ROCM was added to later versions of PyTorch. Define here to
        # support older PyTorch versions as well:
        define_macros += [("USE_ROCM", None)]
        undef_macros += ["__HIP_NO_HALF_CONVERSIONS__"]
        nvcc_flags += ["-ffast-math"]
        if LINE_INFO:
            nvcc_flags += ["-g", "-gline-tables-only"]

        # The `rocm-sdk` wheels keep the AMDGCN device bitcode under
        # lib/llvm/amdgcn instead of <root>/amdgcn, so clang's default probe
        # from --rocm-path misses it and the build dies with "cannot find ROCm
        # device library". Point it at the real directory when we can find one.
        rocm_home = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH")
        if not rocm_home:
            try:
                import _rocm_sdk_core
                rocm_home = osp.dirname(_rocm_sdk_core.__file__)
            except ImportError:
                rocm_home = None
        if rocm_home:
            for cand in (osp.join(rocm_home, "lib", "llvm", "amdgcn"),
                         osp.join(rocm_home, "amdgcn")):
                if osp.isdir(osp.join(cand, "bitcode")):
                    nvcc_flags += ["--rocm-device-lib-path=" + osp.join(cand, "bitcode")]
                    break

        extra_compile_args["nvcc"] = nvcc_flags
        if sys.platform == "win32":
            extra_compile_args["nvcc"] += ["-DWIN32_LEAN_AND_MEAN"]
    else:
        nvcc_flags += ["--use_fast_math", "--expt-relaxed-constexpr"]
        if LINE_INFO:
            nvcc_flags += ["-lineinfo"]
        # GLM/Torch has spammy and very annoyingly verbose warnings that this suppresses
        nvcc_flags += ["-diag-suppress", "20012,186"]
        extra_compile_args["nvcc"] = nvcc_flags
        if sys.platform == "win32":
            extra_compile_args["nvcc"] += [
                "-DWIN32_LEAN_AND_MEAN",
                "-allow-unsupported-compiler",
            ]

    current_dir = pathlib.Path(__file__).parent.resolve()
    include_dirs = [osp.join(current_dir, "gsplat", "cuda", "include")]

    # Headers fetched by tools/setup_rocm_thirdparty.sh. HYWORLD_ROCM_THIRD_PARTY
    # overrides the default location of `<repo>/third_party_rocm`.
    tp = os.environ.get("HYWORLD_ROCM_THIRD_PARTY")
    if not tp:
        tp = osp.join(str(current_dir), "..", "..", "..", "..", "..", "third_party_rocm")
    tp = osp.abspath(tp)

    def _add_include(path, what):
        if osp.isdir(path):
            include_dirs.append(path)
        else:
            print(f"WARNING: {what} headers not found at {path} — "
                  f"run tools/setup_rocm_thirdparty.sh")

    # glm. Upstream vendors it as a git submodule under
    # gsplat/cuda/csrc/third_party/glm, but HY-World-2.0 ships only the empty
    # directory, so the build fails identically on CUDA. It is kept outside the
    # gsplat tree because torch's hipify mirrors gsplat/cuda -> gsplat/hip and
    # copies only the source extensions it recognises, silently dropping glm's
    # .inl files.
    _add_include(osp.join(tp, "glm"), "glm")

    if torch.version.hip:
        # --- ROCm: supply rocThrust / rocPRIM / hipCUB headers --------------
        # torch/headeronly/util/complex.h does `#include <thrust/complex.h>`
        # whenever __HIPCC__ is defined, but the `rocm-sdk` wheels ship no
        # rocThrust headers at all (rocm-sdk-devel has an empty include/).
        _add_include(osp.join(tp, "rocThrust"), "rocThrust")
        _add_include(osp.join(tp, "rocPRIM", "rocprim", "include"), "rocPRIM")
        _add_include(osp.join(tp, "hipCUB", "hipcub", "include"), "hipCUB")

        # hipcc defines __CUDACC__, so glm/simd/platform.h takes its CUDA
        # branch (which precedes the __HIP__ branch) and then errors with
        # "GLM requires CUDA 7.0 or higher" because CUDA_VERSION is undefined.
        # Declaring a CUDA 8 level selects GLM_COMPILER_CUDA80, whose only
        # effect is the __device__ __host__ decoration that hipcc wants anyway.
        extra_compile_args["nvcc"] += ["-DCUDA_VERSION=8000"]

        if sys.platform == "win32":
            # --- ROCm on Windows: host sources must go through hipcc --------
            # torch.utils.cpp_extension compiles .cpp with MSVC cl.exe, but
            # these host files include HIP headers (amd_hip_vector_types.h),
            # which use GCC/clang __attribute__ syntax that cl cannot parse.
            # Generate a .cu shim per .cpp so the HIP toolchain (clang) picks
            # them up instead. hipify mirrors gsplat/cuda -> gsplat/hip before
            # compiling, and the relative include below resolves inside that
            # mirror as well.
            shim_dir = osp.join(str(current_dir), "gsplat", "cuda", "csrc", "_hip_host")
            os.makedirs(shim_dir, exist_ok=True)
            shimmed = []
            for src in sources:
                if src.endswith(".cpp"):
                    stem = osp.splitext(osp.basename(src))[0]
                    shim = osp.join(shim_dir, stem + "_host.cu")
                    rel = osp.relpath(osp.abspath(src), shim_dir).replace("\\", "/")
                    with open(shim, "w") as fh:
                        fh.write("// Generated by setup.py: routes a host-side .cpp through hipcc.\n")
                        fh.write('#include "%s"\n' % rel)
                    shimmed.append(shim)
                else:
                    shimmed.append(src)
            sources = shimmed

    extension = CUDAExtension(
        "gsplat.csrc",
        sources,
        include_dirs=include_dirs,
        define_macros=define_macros,
        undef_macros=undef_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
    return [extension]


_packages = find_packages(where=".", include=["gsplat", "gsplat.*"])

setup(
    name="gsplat",
    version=__version__,
    description=" Python package for differentiable rasterization of gaussians",
    keywords="gaussian, splatting, cuda",
    url=URL,
    download_url=f"{URL}/archive/gsplat-{__version__}.tar.gz",
    python_requires=">=3.7",
    install_requires=[
        "ninja",
        "numpy",
        "jaxtyping",
        "rich>=12",
        "torch",
        "typing_extensions; python_version<'3.8'",
    ],
    extras_require={
        # dev dependencies. Install them by `pip install gsplat[dev]`
        "dev": [
            "black[jupyter]==22.3.0",
            "isort==5.10.1",
            "pylint==2.13.4",
            "pytest==7.1.2",
            "pytest-xdist==2.5.0",
            "typeguard>=2.13.3",
            "pyyaml==6.0",
            "build",
            "twine",
        ],
    },
    ext_modules=get_extensions() if not BUILD_NO_CUDA else [],
    cmdclass={"build_ext": get_ext()} if not BUILD_NO_CUDA else {},
    packages=_packages,
    # https://github.com/pypa/setuptools/issues/1461#issuecomment-954725244
    include_package_data=True,
)

if need_to_unset_max_jobs:
    print("Unsetting MAX_JOBS")
    os.environ.pop("MAX_JOBS")
