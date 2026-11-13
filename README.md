# CUDA video processing exercise

Process a video with a CPU or CUDA implementation using GStreamer.
Edit `src/Compute.cpp` and `src/Compute.cu` for the exercise.

## Requirements

- CMake 3.18 or newer, pkg-config, and a C++20 compiler with `<format>` support.
- NVIDIA GPU, driver, and CUDA toolkit (`nvcc`).
- GStreamer development packages and runtime plugins for video decoding and MP4 encoding.

On Ubuntu, install the build tools and GStreamer packages with:

```bash
sudo apt install build-essential cmake pkg-config \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly gstreamer1.0-libav
```

The NVIDIA driver and CUDA toolkit must also be installed.

## Build on the course GPU machine

Tested on `gpgpu.image.lrde.iaas.epita.fr` with GCC 13, CUDA 13.4,
and a GeForce GTX 1650 (compute capability 7.5):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j4
```

or on Nixos

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j4
```

Use `Debug` instead of `Release` for a debug build. On another machine, adapt
the CUDA compiler path and architecture to its installed toolkit and GPU.
If working on AFS, choose a build directory outside AFS and use that path
instead of `build` throughout the commands.

## Run

For SSH sessions without a display, write an output video:

```bash
build/stream --mode=gpu samples/ACET.mp4 --output=gpu-output.mp4
build/stream --mode=cpu samples/ACET.mp4 --output=cpu-output.mp4
```

With a graphical display, omit `--output` to view the processed video:

```bash
build/stream --mode=gpu "samples/Nuits Blanches .mp4"
```

The CPU implementation includes a deliberate 50 ms delay per frame.
