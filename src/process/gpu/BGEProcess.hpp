#pragma once

#include <array>
#include <cstdint>
#include <cuda/std/array>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "Process.hpp"

namespace process::gpu
{
    inline constexpr std::size_t SEED = 868686ULL;
    inline constexpr std::size_t RESERVOIR_K_MAX = 32;
    inline constexpr std::uint8_t RGB_DIFF_THRESHOLD = 10;
    inline constexpr std::size_t MAX_WEIGHT = 25;

    struct Reservoir
    {
        Image<uint32_t> d_weights;
        Image<uchar4> d_colors;
        std::size_t k;

        Reservoir(std::size_t width, std::size_t height, std::size_t k);
    };

    class BGEProcess : public Process
    {
    public:
        BGEProcess(ImageView<rgb8> d_frame, ImageView<rgb8> d_bg,
                   std::size_t reservoir_size);

        ~BGEProcess() = default;

        void run() override;

    private:
        ImageView<rgb8> d_bg_;
        std::size_t k_;
        Image<curandState> d_rng_;

        Reservoir d_rs_;
    };

    CUDA_HOST_DEVICE inline bool matches(uchar4 c, const rgb8& rgb);
    CUDA_HOST_DEVICE inline uchar4 to_uchar4(const rgb8& c);
    CUDA_HOST_DEVICE inline rgb8 to_rgb8(const uchar4& c);
} /* namespace process::gpu */

#include "BGEProcess.hxx"
