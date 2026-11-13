#include <algorithm>
#include <chrono>
#include <cstddef>

#include "Compute.hpp"
#include "Image.hpp"
#include "logo.h"
#include "CudaUtils.cuh"
#include "process/Process.hpp"
#include "process/gpu/BGEProcess.hpp"
#include "process/gpu/MCProcess.hpp"

void compute_cu(ImageView<rgb8> in)
{
    // Background given on the command line, loaded once
    static Image<rgb8> bg_file =
        g_params.bg ? Image<rgb8>(g_params.bg) : Image<rgb8>();

    if (g_params.bg && !bg_file.buffer)
        throw std::runtime_error("cannot load background image");
    if (bg_file.buffer
        && (bg_file.width != in.width || bg_file.height != in.height))
        throw std::runtime_error("background and video sizes differ");

    const std::size_t row_bytes = in.width * sizeof(rgb8);

    // Buffer used for the estimated background
    static Image<rgb8> d_frame(in.width, in.height, true);
    static Image<rgb8> d_bg(in.width, in.height, true);

    static bool bg_uploaded = false;
    if (bg_file.buffer && !bg_uploaded)
    {
        CUDA_CHECK(cudaMemcpy2D(d_bg.buffer, d_bg.stride, bg_file.buffer,
                                bg_file.stride, row_bytes, in.height,
                                cudaMemcpyHostToDevice));
        bg_uploaded = true;
    }

    CUDA_CHECK(cudaMemcpy2D(d_frame.buffer, d_frame.stride, in.buffer,
                            in.stride, row_bytes, in.height,
                            cudaMemcpyHostToDevice));

    if (!bg_file.buffer)
    {
        using clock = std::chrono::steady_clock;
        static process::gpu::BGEProcess bge(d_frame, d_bg, g_params.bg_number_frame);
        static clock::time_point last_sample{};

        const auto now = clock::now();

        if (now - last_sample
            >= std::chrono::milliseconds(g_params.bg_sampling_rate))
        {
            last_sample = now;
            bge.run();
        }
    }

    // Compute motion mask (frame vs bg)
    static process::gpu::MCProcess mc(d_frame, d_bg, g_params.opening_size,
                                      g_params.th_low, g_params.th_high);
    mc.run();

    CUDA_CHECK(cudaMemcpy2D(in.buffer, in.stride, d_frame.buffer,
                            d_frame.stride, row_bytes, in.height,
                            cudaMemcpyDeviceToHost));
}
