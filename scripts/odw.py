#!/usr/bin/env python3

import argparse
import os
import shlex
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PARAM_FILE = REPO_ROOT / "scripts" / "opendwarf_application_parameters.py"
namespace = {}
exec(PARAM_FILE.read_text(), namespace)

APP_PATHS = {
    "nqueens": {
        "cuda": "branch-and-bound/nqueens/cuda",
        "opencl": "branch-and-bound/nqueens/opencl",
        "hip": "branch-and-bound/nqueens/hip",
    },
    "crc": {
        "cuda": "combinational-logic/crc/cuda",
        "opencl": "combinational-logic/crc/opencl",
        "hip": "combinational-logic/crc/hip",
    },
    "lud": {
        "cuda": "dense-linear-algebra/lud/cuda",
        "opencl": "dense-linear-algebra/lud/opencl",
        "hip": "dense-linear-algebra/lud/hip",
    },
    "needle": {
        "cuda": "dynamic-programming/nw/cuda",
        "opencl": "dynamic-programming/nw/opencl",
        "hip": "dynamic-programming/nw/hip",
    },
    "swat": {
        "cuda": "dynamic-programming/swat/cuda",
        "opencl": "dynamic-programming/swat/opencl",
        "hip": "dynamic-programming/swat/hip",
    },
    "tdm": {
        "cuda": "finite-state-machine/tdm/cuda",
        "opencl": "finite-state-machine/tdm/opencl",
        "hip": "finite-state-machine/tdm/hip",
    },
    "bwa_hmm": {
        "cuda": "graphical-models/hmm/cuda",
        "opencl": "graphical-models/hmm/opencl",
        "hip": "graphical-models/hmm/hip",
    },
    "bfs": {
        "cuda": "graph-traversal/bfs/cuda",
        "opencl": "graph-traversal/bfs/opencl",
        "hip": "graph-traversal/bfs/hip",
    },
    "kmeans": {
        "cuda": "mapreduce/kmeans/cuda",
        "opencl": "mapreduce/kmeans/opencl",
        "hip": "mapreduce/kmeans/hip",
    },
    "gemnoui": {
        "cuda": "n-body-methods/gem/cuda",
        "opencl": "n-body-methods/gem/opencl",
        "hip": "n-body-methods/gem/hip",
    },
    "csr": {
        "cuda": "sparse-linear-algebra/SPMV/cuda",
        "opencl": "sparse-linear-algebra/SPMV/opencl",
        "hip": "sparse-linear-algebra/SPMV/hip",
    },
    "srad": {
        "cuda": "structured-grids/srad/cuda",
        "opencl": "structured-grids/srad/opencl",
        "hip": "structured-grids/srad/hip",
    },
    "cfd": {
        "cuda": "unstructured-grids/cfd/cuda",
        "opencl": "unstructured-grids/cfd/opencl",
        "hip": "unstructured-grids/cfd/hip",
    },
    "dwt2d": {
        "cuda": "spectral-methods/dwt2d/cuda",
        "opencl": "spectral-methods/dwt2d/opencl",
        "hip": "spectral-methods/dwt2d/hip",
    },
    "fft": {
        "cuda": "spectral-methods/fft/cuda",
        "opencl": "spectral-methods/fft/opencl",
        "hip": "spectral-methods/fft/hip",
    },
    "cwt": {
        "cuda": "spectral-methods/cwt/cuda",
        "opencl": "spectral-methods/cwt/opencl",
        "hip": "spectral-methods/cwt/hip",
    },
}

COMPILERS = {
    "cuda": {
        "nvcc": "cuda-nvcc",
        "scale-nvidia": "cuda-scale-nvidia",
        "scale-amd": "cuda-scale-amd",
    },
    "opencl": {
        "opencl": "opencl",
    },
    "hip": {
        "hipcc": "hip-hipcc",
    },
}


def shell(cmd, cwd=None):
    print("+", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def host_name():
    return os.uname().nodename.split(".")[0]

def opencl_args():
    return os.environ.get("OPENCL_ARGS", "-p 0 -d 0 -t 1 --").strip()

def make_vars(backend, compiler):
    top = REPO_ROOT
    lsb = Path(
        os.environ.get(
            "LSB_INSTALL_ROOT",
            str(top / "external" / "liblsb-install" / host_name()),
        )
    )

    common = [
        f"TOP_LEVEL={top}",
        f"LSB_INSTALL_ROOT={lsb}",
        f"CPPFLAGS=-I{lsb}/include",
        f"LDFLAGS=-L{lsb}/lib -Xlinker -rpath -Xlinker {lsb}/lib",
        "LDLIBS=-llsb",
    ]

    if "SCALE_ROOT" in os.environ:
        common.append(f"SCALE_ROOT={os.environ['SCALE_ROOT']}")

    if backend == "cuda":
        vars_ = common + [
            f"CUDA_NVCC={os.environ.get('CUDA_NVCC', os.environ.get('NVCC', 'nvcc'))}",
            f"CUDA_PATH={os.environ.get('CUDA_PATH', '')}",
            f"NVCCFLAGS={os.environ.get('NVCCFLAGS', '-O3 -std=c++17')}",
        ]

        if compiler in ("nvcc", "scale-nvidia"):
            cuda_dev_target = os.environ.get("CUDA_DEV_TARGET", "")
            if not cuda_dev_target:
                raise SystemExit(
                    f"COMPILER={compiler} requires CUDA_DEV_TARGET, but it is not set."
                )

            cuda_arch = os.environ.get(
                "CUDA_ARCH",
                cuda_dev_target.removeprefix("sm_"),
            )

            vars_ += [
                f"CUDA_DEV_TARGET={cuda_dev_target}",
                f"CUDA_ARCH={cuda_arch}",
            ]

        elif compiler == "scale-amd":
            hip_dev_target = os.environ.get("HIP_DEV_TARGET", "")
            if not hip_dev_target:
                raise SystemExit(
                    "COMPILER=scale-amd requires HIP_DEV_TARGET, but it is not set."
                )

            hip_arch = os.environ.get("HIP_ARCH", hip_dev_target)

            vars_ += [
                f"HIP_DEV_TARGET={hip_dev_target}",
                f"HIP_ARCH={hip_arch}",
            ]

        else:
            raise SystemExit(f"Unknown CUDA compiler: {compiler}")

        return vars_

    if backend == "hip":
        hip_dev_target = os.environ.get("HIP_DEV_TARGET", "")
        if not hip_dev_target:
            raise SystemExit("BACKEND=hip requires HIP_DEV_TARGET, but it is not set.")

        hip_arch = os.environ.get("HIP_ARCH", hip_dev_target)

        return common + [
            f"HIPCC={os.environ.get('HIPCC', 'hipcc')}",
            f"HIP_DEV_TARGET={hip_dev_target}",
            f"HIP_ARCH={hip_arch}",
            f"HIPFLAGS={os.environ.get('HIPFLAGS', '-O3 -std=c++17')}",
        ]

    if backend == "opencl":
        opencl_cppflags = os.environ.get(
            "OPENCL_CPPFLAGS",
            "-DOPENCL -DCL_TARGET_OPENCL_VERSION=120",
        )

        required_defs = "-D_GNU_SOURCE -D_DEFAULT_SOURCE"
        for flag in required_defs.split():
            if flag not in opencl_cppflags.split():
                opencl_cppflags = f"{flag} {opencl_cppflags}"

        return common + [
            "CC=/usr/bin/gcc",
            "CXX=/usr/bin/g++",
            f"CFLAGS={os.environ.get('CFLAGS', '-O3 -std=c99')}",
            f"CXXFLAGS={os.environ.get('CXXFLAGS', '-O3 -std=c++17')}",
            f"OCD_COMMON_ARGS_SRC={os.environ.get('OCD_COMMON_ARGS_SRC', str(top / 'include' / 'common_args.c'))}",
            f"OCD_OPTS_SRC={os.environ.get('OCD_OPTS_SRC', str(top / 'opts' / 'opts.c'))}",
            f"OCD_RDTSC_SRC={os.environ.get('OCD_RDTSC_SRC', str(top / 'include' / 'rdtsc.c'))}",
            f"OPENCL_CPPFLAGS={opencl_cppflags}",
            f"OPENCL_LDFLAGS={os.environ.get('OPENCL_LDFLAGS', '')}",
            f"OPENCL_LDLIBS={os.environ.get('OPENCL_LDLIBS', '-lOpenCL')}",
        ]
    raise SystemExit(f"Unknown backend: {backend}")


def get_application(app_name):
    if app_name not in namespace:
        raise SystemExit(f"Unknown application variable: {app_name}")

    app = namespace[app_name]

    if not isinstance(app, dict):
        raise SystemExit(f"{app_name} exists, but is not an application dictionary")

    return app


def get_app_dir(app, backend):
    name = app["name"]

    if name not in APP_PATHS:
        raise SystemExit(f"No APP_PATHS entry for application '{name}'")

    if backend not in APP_PATHS[name]:
        valid = ", ".join(sorted(APP_PATHS[name]))
        raise SystemExit(f"No backend '{backend}' for application '{name}'. Valid: {valid}")

    path = REPO_ROOT / APP_PATHS[name][backend]

    if not path.exists():
        raise SystemExit(f"Backend path does not exist: {path}")

    return path


def get_suffix(backend, compiler):
    if backend not in COMPILERS:
        raise SystemExit(f"Unknown backend '{backend}'")

    if compiler not in COMPILERS[backend]:
        valid = ", ".join(sorted(COMPILERS[backend]))
        raise SystemExit(
            f"Unknown compiler '{compiler}' for backend '{backend}'. Valid: {valid}"
        )

    return COMPILERS[backend][compiler]


def make_target(app, backend, compiler):
    return f"{app['name']}-{get_suffix(backend, compiler)}"


def run_target(app, backend, compiler):
    return f"run-{app['name']}-{get_suffix(backend, compiler)}"


def binary_name(app, backend, compiler):
    return f"{app['name']}-{get_suffix(backend, compiler)}"


def run_binary(app_dir, app, backend, compiler, final_args, lsb_name):
    target = binary_name(app, backend, compiler)
    exe = app_dir / target
    if not exe.exists():
        raise SystemExit(f"Executable does not exist: {exe}")

    results = REPO_ROOT / "results"
    results.mkdir(exist_ok=True)

    if backend == "opencl":
        for kernel in app_dir.rglob("*.cl"):
            dst = results / kernel.name
            dst.write_bytes(kernel.read_bytes())

    env_cmd = f"ODW_LSB_NAME={shlex.quote(lsb_name)} {shlex.quote(str(exe))}"
    if final_args:
        env_cmd += " " + final_args

    if backend == "cuda" and compiler.startswith("scale-"):
        scale_root = os.environ.get("SCALE_ROOT", str(REPO_ROOT / "scale-1.7.1-Linux"))

        if compiler == "scale-amd":
            target = os.environ.get("HIP_ARCH", os.environ.get("HIP_DEV_TARGET", "gfx942"))
        else:
            cuda_arch = os.environ.get("CUDA_ARCH", os.environ.get("CUDA_DEV_TARGET", "sm_70").removeprefix("sm_"))
            target = f"sm_{cuda_arch}"

        env_cmd = f"source {shlex.quote(scale_root + '/bin/scaleenv')} {shlex.quote(target)} && " + env_cmd

    shell(["bash", "-lc", env_cmd], cwd=results)




def selected_apps(app_arg):
    if app_arg is None or app_arg == "" or app_arg == "all":
        return ["nqueens"]
    return [app_arg]


def build(args):
    for app_name in selected_apps(args.app):
        app = get_application(app_name)
        app_dir = get_app_dir(app, args.backend)

        shell(
            ["make", make_target(app, args.backend, args.compiler)]
            + make_vars(args.backend, args.compiler),
            cwd=app_dir,
        )


def run(args):
    for app_name in selected_apps(args.app):
        app = get_application(app_name)
        app_dir = get_app_dir(app, args.backend)

        if args.size not in app:
            valid = [k for k in ("tiny", "small", "medium", "large", "default") if k in app]
            msg = (
                f"Application '{app_name}' has no size '{args.size}'. "
                f"Available sizes: {', '.join(valid)}"
            )

            if os.environ.get("ODW_SKIP_MISSING_SIZE", "0") == "1":
                print(f"Skipping: {msg}")
                continue

            raise SystemExit(msg)

        problem_args = app[args.size]
        extra_args = os.environ.get("ARGS", "").strip()

        if args.backend == "opencl":
            selector_args = opencl_args()
            final_args = f"{extra_args} {selector_args} {problem_args}".strip()
        else:
            # Runs execute from REPO_ROOT/results. Convert app-relative test paths
            # to repository-absolute paths so dataset loaders can find them.
            parts = shlex.split(problem_args)
            fixed = []
            for part in parts:
                if part.startswith("../test/"):
                    fixed.append(str(REPO_ROOT / part[3:]))
                else:
                    fixed.append(part)
            problem_args = " ".join(shlex.quote(x) for x in fixed)
            final_args = f"{extra_args} {problem_args}".strip()

        lsb_app_name = app_name
        device_label = os.environ.get("DEVICE_LABEL", "").strip()
        lsb_parts = [lsb_app_name, args.backend, args.compiler, args.size]
        if device_label:
            lsb_parts.append(device_label)
        lsb_name = "_".join(lsb_parts).replace("-", "_")

        for i in range(args.iterations):
            print(
                f"==> {app['name']} backend={args.backend} compiler={args.compiler} "
                f"size={args.size} iter={i + 1}/{args.iterations}"
            )

            run_binary(
                app_dir,
                app,
                args.backend,
                args.compiler,
                final_args,
                lsb_name,
            )


def clean(args):
    for app_name in selected_apps(args.app):
        app = get_application(app_name)
        app_dir = get_app_dir(app, args.backend)
        shell(["make", "clean"], cwd=app_dir)


def add_common_args(p):
    p.add_argument("--app", default="nqueens")
    p.add_argument("--backend", default="cuda", choices=["cuda", "opencl", "hip"])
    p.add_argument("--compiler", default=None)


def normalize_compiler(args):
    if not hasattr(args, "compiler"):
        return

    if args.compiler is None:
        if args.backend == "cuda":
            args.compiler = "nvcc"
        elif args.backend == "opencl":
            args.compiler = "opencl"
        elif args.backend == "hip":
            args.compiler = "hipcc"


def main():
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build")
    add_common_args(p)
    p.set_defaults(func=build)

    p = sub.add_parser("run")
    add_common_args(p)
    p.add_argument("--size", default="tiny")
    p.add_argument("--iterations", type=int, default=1)
    p.set_defaults(func=run)

    p = sub.add_parser("clean")
    p.add_argument("--app", default="nqueens")
    p.add_argument("--backend", default="cuda", choices=["cuda", "opencl", "hip"])
    p.set_defaults(func=clean)

    args = parser.parse_args()
    normalize_compiler(args)
    args.func(args)


if __name__ == "__main__":
    main()
