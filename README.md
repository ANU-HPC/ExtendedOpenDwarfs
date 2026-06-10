# ExtendedOpenDwarfs

> Originally developed as **OpenDwarfs** at Virginia Tech and CHREC, modernized at the **Australian National University (2017–2019)**, and further extended for contemporary heterogeneous programming environments with native **CUDA**, **HIP**, **OpenCL**, and **SCALE** support.

---

## Overview

ExtendedOpenDwarfs is a modern heterogeneous benchmarking suite derived from the original OpenDwarfs project developed at Virginia Tech and the NSF Center for High-Performance Reconfigurable Computing (CHREC).

The original OpenDwarfs benchmark suite provided one of the first comprehensive OpenCL benchmark collections based on the [Berkeley Dwarfs](http://view.eecs.berkeley.edu/wiki/Dwarf_Mine), enabling performance characterization across CPUs, GPUs, FPGAs, and accelerator architectures.

This repository extends that foundation through two major generations of development:

1. **OpenDwarfs Modernization (ANU, 2017–2019)**
2. **ExtendedOpenDwarfs (2026–Present)**

The result is a portable benchmark suite suitable for evaluating:

- Accelerator architectures
- Programming models
- Runtime systems
- Compilers
- Performance portability frameworks
- Scheduling systems
- Heterogeneous execution environments

while retaining the Berkeley Dwarfs methodology as a representative collection of scientific and engineering workloads.

---

## Project Evolution

### OpenDwarfs (Virginia Tech / CHREC)

The original OpenDwarfs suite was implemented primarily in OpenCL and designed to characterize computational motifs derived from the Berkeley Dwarfs.

Key publications include:

- *OpenCL and the 13 Dwarfs: A Work in Progress* (ICPE 2012)
- *On the Characterization of OpenCL Dwarfs on Fixed and Reconfigurable Platforms* (ASAP 2014)
- *OpenDwarfs: Characterization of Dwarf-based Benchmarks on Fixed and Reconfigurable Architectures* (JSPS 2015)

---

### Enhanced OpenDwarfs (ANU, 2017–2019)

As part of research into accelerator characterization, performance portability, and heterogeneous scheduling, the benchmark suite underwent extensive modernization and curation.

Enhancements included:

- Standardized workload scales:
  - Tiny
  - Small
  - Medium
  - Large

- Memory-hierarchy-aware problem sizing

- Improved benchmark correctness and validation

- Repair or replacement of legacy benchmarks

- Addition of new workloads and benchmark coverage

- Integration with LibSciBench

- PAPI hardware counter support

- Energy measurement support through:
  - Intel RAPL
  - NVIDIA NVML

- Statistical benchmarking methodology

- Automated repeated execution

- Reproducible datasets and execution workflows

- Improved portability across contemporary accelerator architectures

This work is described in:

- Beau Johnston and Josh Milthorpe,
  *Dwarfs on Accelerators: Enhancing OpenCL Benchmarking for Heterogeneous Computing Architectures*,
  ICPP Workshops 2018.
  DOI: https://doi.org/10.1145/3229710.3229729

- Beau Johnston,
  *Characterizing and Predicting Scientific Workloads for Heterogeneous Computing Systems*,
  PhD Thesis, Australian National University, 2019.
  https://openresearch-repository.anu.edu.au/handle/1885/162792

---

### ExtendedOpenDwarfs (2026–Present)

This repository extends the benchmark suite beyond OpenCL and introduces a unified workflow for evaluating modern heterogeneous systems.

New capabilities include:

- Native CUDA implementations
- Native HIP implementations
- OpenCL baselines
- Spectral Compute's [SCALE](https://docs.scale-lang.com/stable/) compiler support
- Unified build infrastructure
- Unified execution infrastructure
- Automated benchmarking workflows
- Automated result aggregation
- Automated plotting and visualization
- Modern accelerator support

Target architectures include:

| Vendor | Architectures |
|----------|-------------|
| NVIDIA | V100, A100, H100 and newer |
| AMD | MI100, MI300A and newer |
| Intel | OpenCL-capable accelerators |
| CPU | x86 and future host architectures |

---

## Supported Backends

Depending on the benchmark, implementations may be available for:

| Backend | Compiler |
|----------|----------|
| OpenCL | Vendor OpenCL SDK |
| CUDA | NVCC |
| HIP | HIPCC |
| SCALE (CUDA→AMD) | SCALE AMD |
| SCALE (CUDA→NVIDIA) | SCALE NVIDIA |

Availability varies by benchmark and platform.

---

## Benchmark Status

### Stable

- gem

### Beta

- bfs
- cfd
- crc
- fft
- kmeans
- lud
- nw
- spmv
- srad
- swat
- bwa_hmm
- nqueens

### Alpha

- tdm

---

# Installation

## Clone Repository

```bash
git clone git@github.com:ANU-HPC/ExtendedOpenDwarfs.git 
cd ExtendedOpenDwarfs
```

---

## Install SCALE (Optional)

Required only for SCALE compiler experiments.

Follow installation instructions from the [SCALE project](https://docs.scale-lang.com/stable/manual/tutorials/how-to-install/).

---

## Install LibSciBench

ExtendedOpenDwarfs uses LibSciBench for:

- Timing
- Region instrumentation
- Statistical analysis
- Energy measurement

Building and installing LibSciBench is integrated into the make workflow.

---

## Install Pixi

Pixi is used to provide a reproducible analysis environment.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Then install project dependencies:

```bash
pixi install
```

This provides:

- R
- ggplot2
- tidyverse
- plotting dependencies

---

# System Configuration

Platform-specific configuration is handled through:

```bash
setup-backends.sh
```

You'll need to update according to your system.
These are reference/samples are from systems provided by Experimental Computing Laboratory (ExCL) at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725

Each host automatically configures:

- CUDA
- HIP
- OpenCL
- SCALE

toolchains and library paths.

---

# Getting Started --- Automated Benchmark Sweeps

A helper script is provided for running complete backend sweeps.
This also compiles all benchmarks and dependencies.

Single size:

```bash
SIZE=medium ./runner.sh
```

Specific benchmark:

```bash
APP=nqueens SIZE=large ./runner.sh
```

Multiple repetitions:

```bash
APP=nqueens SIZE=large ITERS=20 ./runner.sh
```

Full size sweep:

```bash
SWEEP=1 ./runner.sh
```

The runner automatically executes all supported implementations for the current system:

- OpenCL
- CUDA
- HIP
- SCALE AMD
- SCALE NVIDIA

where available.


# Building Benchmarks

Build a single benchmark:

```bash
make build \
    APP=nqueens \
    BACKEND=opencl \
    COMPILER=opencl
```

Examples:

```bash
make build APP=nqueens BACKEND=cuda COMPILER=nvcc

make build APP=nqueens BACKEND=hip COMPILER=hipcc

make build APP=nqueens BACKEND=cuda COMPILER=scale-amd
```

---

# Running Benchmarks

General form:

```bash
make run \
    APP=<benchmark> \
    BACKEND=<backend> \
    COMPILER=<compiler> \
    SIZE=<size> \
    ITERS=<repetitions>
```

Example:

```bash
ARGS="-p 0 -d 0 -t 1 --" \
make run \
    APP=nqueens \
    BACKEND=opencl \
    COMPILER=opencl \
    SIZE=tiny \
    ITERS=5
```

---

## OpenCL Device Selection

OpenCL benchmarks use the original OpenDwarfs device-selection interface:

```bash
-p <platform>
-d <device>
-t <type>
```

where:

| Type | Meaning |
|--------|----------|
| 0 | CPU |
| 1 | GPU |
| 2 | MIC |
| 3 | FPGA |

Example:

```bash
-p 0 -d 0 -t 1 --
```

selects GPU device 0 on platform 0.

---

# Results

LibSciBench outputs timing data into:

```text
results/
```

Typical files:

```text
lsb.<benchmark>.<backend>.r0
lsb.<benchmark>.<backend>.r1
...
```

Each file contains:

- Runtime
- Region timings
- Transfer costs
- Kernel execution times
- Setup overheads

depending on benchmark instrumentation.

---

# Plotting

Generate plots from all benchmark results:

```bash
pixi run plot-lsb
```

Plots are written to:

```text
results/plots/
```

Current plots include:

- Runtime distributions
- Region breakdowns
- Benchmark comparisons
- Machine comparisons
- Normalized performance views

Plots are automatically grouped by:

- Benchmark
- Machine
- Backend
- Compiler

to simplify comparison across heterogeneous systems.

---

# Energy Measurements

LibSciBench supports energy collection through:

- Intel RAPL
- NVIDIA NVML

When available, energy measurements are recorded alongside timing information.

For RAPL access:

```bash
sudo modprobe msr
sudo chmod 666 /dev/cpu/*/msr

echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

Verify availability:

```bash
papi_native_avail -e rapl:::PP0_ENERGY:PACKAGE0
```

---

# Berkeley Dwarfs Coverage

The benchmark suite is organized around the Berkeley Dwarfs:

- Dense Linear Algebra
- Sparse Linear Algebra
- Spectral Methods
- Structured Grids
- Dynamic Programming
- Graphical Models
- N-Body Methods
- Backtrack and Branch-and-Bound
- Combinational Logic
- MapReduce
- and others

providing a representative collection of scientific computing workloads.

---

## Citation

If you use ExtendedOpenDwarfs in academic work, please cite the relevant publications below.

### ExtendedOpenDwarfs

Publication Pending---for now just cite this repository.

### Original OpenDwarfs

#### OpenCL and the 13 Dwarfs: A Work in Progress (ICPE 2012)

```bibtex
@inproceedings{feng2012opendwarfs,
  author    = {Wu-chun Feng and Heshan Lin and Thomas Scogland and Jing Zhang},
  title     = {OpenCL and the 13 Dwarfs: A Work in Progress},
  booktitle = {Proceedings of the 3rd ACM/SPEC International Conference on Performance Engineering (ICPE)},
  year      = {2012},
  pages     = {291--294},
  doi       = {10.1145/2188286.2188341}
}
```

#### On the Characterization of OpenCL Dwarfs on Fixed and Reconfigurable Platforms (ASAP 2014)

```bibtex
@inproceedings{krommydas2014characterization,
  author    = {Konstantinos Krommydas and Wu-chun Feng and Muhsen Owaida and Christos Antonopoulos and Nikolaos Bellas},
  title     = {On the Characterization of OpenCL Dwarfs on Fixed and Reconfigurable Platforms},
  booktitle = {IEEE 25th International Conference on Application-Specific Systems, Architectures and Processors (ASAP)},
  year      = {2014},
  pages     = {153--160},
  doi       = {10.1109/ASAP.2014.6868650}
}
```

#### OpenDwarfs: Characterization of Dwarf-Based Benchmarks on Fixed and Reconfigurable Architectures (JSPS 2016)

```bibtex
@article{krommydas2016opendwarfs,
  author  = {Konstantinos Krommydas and Wu-chun Feng and Christos Antonopoulos and Nikolaos Bellas},
  title   = {OpenDwarfs: Characterization of Dwarf-Based Benchmarks on Fixed and Reconfigurable Architectures},
  journal = {Journal of Signal Processing Systems},
  volume  = {85},
  number  = {3},
  pages   = {373--392},
  year    = {2016},
  doi     = {10.1007/s11265-015-1051-z}
}
```

### Enhanced OpenDwarfs (ANU Modernization)

#### Dwarfs on Accelerators: Enhancing OpenCL Benchmarking for Heterogeneous Computing Architectures (ICPPW 2018)

```bibtex
@inproceedings{johnston2018dwarfs,
  author    = {Beau Johnston and Josh Milthorpe},
  title     = {Dwarfs on Accelerators: Enhancing OpenCL Benchmarking for Heterogeneous Computing Architectures},
  booktitle = {47th International Conference on Parallel Processing Workshops (ICPPW)},
  year      = {2018},
  doi       = {10.1145/3229710.3229729}
}
```

#### Characterizing and Predicting Scientific Workloads for Heterogeneous Computing Systems (PhD Thesis, ANU 2019)

```bibtex
@phdthesis{johnston2019thesis,
  author = {Beau Johnston},
  title  = {Characterizing and Predicting Scientific Workloads for Heterogeneous Computing Systems},
  school = {Australian National University},
  year   = {2019},
  url    = {https://openresearch-repository.anu.edu.au/handle/1885/162792}
}
```

---

# Acknowledgements

**OpenDwarfs was originally developed at Virginia Tech and CHREC:**

This project has been supported in part by Air Force Research Lab, Altera, AMD, Department of Defense, Harris, Los Alamos National Laboratory, and Xilinx via the NSF Center for High-Performance Reconfigurable Computing (CHREC) under NSF grant IIP-0804155 and indirectly by AFOSR grant FA9550-12-1-0442 and NSF grants CNS-0916719 and MRI-0960081.

Integration for Altera FPGA support for crc and csr, as well as extensions for these
benchmarks, have been contributed by Tyler Kenney at IBM.

Part of the OpenDwarfs benchmark suite (as acknowledged in the respective benchmarks' READMEs) was ported to OpenCL from the corresponding CUDA implementations in earlier implementations of the Rodinia benchmark suite (http://www.cs.virginia.edu/~skadron/wiki/rodinia/index.php/Main_Page).

**The modernization and benchmarking extensions were developed at the Australian National University.**

ExtendedOpenDwarfs continues this work by providing a modern framework for heterogeneous benchmark evaluation across OpenCL, CUDA, HIP, and SCALE ecosystems.

