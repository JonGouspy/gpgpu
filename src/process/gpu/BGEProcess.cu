#include <iterator>

#include "BGEProcess.hpp"
#include "CudaUtils.cuh"

namespace process::gpu
{
    namespace
    {
        __global__ void init_curand_kernel(ImageView<curandState> d_states,
                                           size_t seed)
        {
            size_t x = blockDim.x * blockIdx.x + threadIdx.x;
            size_t y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x >= d_states.width || y >= d_states.height)
                return;

            curandState* row = d_states.getline(y);
            curand_init(seed, y * d_states.width + x, 0, &row[x]);
        }

        __device__ static int
        find_matching_reservoir_device(const ImageView<rgb8> in, std::size_t x,
                                       std::size_t y, const Reservoir& rs,
                                       std::size_t k)
        {
            ssize_t m_idx = -1;

            for (std::size_t i = 0; i < k; ++i)
            {
                const std::size_t row = y + i * in.height;
                const uint32_t w = rs.d_weights.at(x, row);
                if (w > 0)
                {
                    const uchar4 c = rs.d_colors.at(x, row);
                    if (matches(c, in.at(x, y)))
                    {
                        m_idx = i;
                        break;
                    }
                }
                else
                    m_idx = i;
            }

            return m_idx;
        }

        // upgrades idea: have 4 bytes instead of 3 (padding)
        // or rrrrrrr gggggg bbbbbbb
        __global__ static void bge_process_kernel(ImageView<rgb8> in,
                                                  ImageView<rgb8> bg_out,
                                                  ImageView<curandState> rng,
                                                  Reservoir& d_rs,
                                                  std::size_t k)
        {
            size_t x = blockDim.x * blockIdx.x + threadIdx.x;
            size_t y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x >= in.width || y >= in.height)
                return;

            const rgb8 prgb = in.at(x, y);
            const int m_idx = find_matching_reservoir_device(in, x, y, d_rs, k);

            if (m_idx != -1)
            {
                const uint32_t row = y + m_idx * in.height;
                uint32_t w = d_rs.d_weights.at(x, row);
                if (w > 0)
                {
                    d_rs.d_weights.at(x, row) =
                        w + 1 < MAX_WEIGHT ? w + 1 : MAX_WEIGHT;
                    w = d_rs.d_weights.at(x, row);

                    d_rs.d_colors.at(x, row).x =
                        ((w - 1) * d_rs.d_colors.at(x, row).x + prgb.r) / w;
                    d_rs.d_colors.at(x, row).y =
                        ((w - 1) * d_rs.d_colors.at(x, row).y + prgb.g) / w;
                    d_rs.d_colors.at(x, row).z =
                        ((w - 1) * d_rs.d_colors.at(x, row).z + prgb.b) / w;
                }
                else
                {
                    d_rs.d_weights.at(x, row) = 1;
                    d_rs.d_colors.at(x, row) = to_uchar4(prgb);
                }
            }
            else // no match
            {
                size_t min_row = y;
                size_t total_weights = 0;
                for (size_t i = 0; i < k; ++i)
                {
                    const uint32_t row = y + i * in.height;
                    const uint32_t w = d_rs.d_weights.at(x, row);
                    if (d_rs.d_weights.at(x, min_row) > w)
                        min_row = row;
                    total_weights += w;
                }

                const float rand_val = curand_uniform(&rng.at(x, y));
                if (rand_val * total_weights >= d_rs.d_weights.at(x, min_row))
                {
                    d_rs.d_weights.at(x, min_row) = 1;
                    d_rs.d_colors.at(x, min_row) = to_uchar4(prgb);
                }
            }

            uint32_t max_idx = 0;
            uint32_t max_weight = d_rs.d_weights.at(x, y);

            for (uint32_t i = 1; i < d_rs.k; ++i)
            {
                const uint32_t w = d_rs.d_weights.at(x, y + i * in.height);
                if (w > max_weight)
                {
                    max_weight = w;
                    max_idx = i;
                }
            }

            const uchar4 best_color =
                d_rs.d_colors.at(x, y + max_idx * in.height);
            bg_out.at(x, y) = to_rgb8(best_color);
        }
    } // namespace

    Reservoir::Reservoir(std::size_t width, std::size_t height, std::size_t k)
        : d_weights(width, height * k, true)
        , d_colors(width, height * k, true)
        , k(k)
    {
        CUDA_CHECK(cudaMemset2D(d_weights.buffer, d_weights.stride, 0,
                                d_weights.width * sizeof(uint32_t),
                                d_weights.height));
        CUDA_CHECK(cudaMemset2D(d_colors.buffer, d_colors.stride, 0,
                                d_colors.width * sizeof(uchar4),
                                d_colors.height));
    }

    BGEProcess::BGEProcess(ImageView<rgb8> d_frame, ImageView<rgb8> d_bg,
                           std::size_t reservoir_size)
        : Process(d_frame)
        , d_bg_(d_bg)
        , k_(cuda::std::clamp<std::size_t>(reservoir_size, 1, RESERVOIR_K_MAX))
        , d_rng_(d_frame.width, d_frame.height, true)
        , d_rs_(d_frame.width, d_frame.height, k_)
    {
        init_curand_kernel<<<utils::grid_2d(d_frame.width, d_frame.height),
                             utils::block_2d()>>>(d_rng_, SEED);
        CUDA_CHECK_LAST("init_curand_kernel");
    }

    // No allocation, no copy: the frame is already on the GPU (d_frame) and
    // the background stays there (d_bg_) for MCProcess
    void BGEProcess::run()
    {
        const ImageView<rgb8> d_frame{ buffer_, width_, height_, stride_ };

        bge_process_kernel<<<utils::grid_2d(this->width_, this->height_),
                             utils::block_2d()>>>(d_frame, d_bg_, d_rng_, d_rs_,
                                                  k_);
        CUDA_CHECK_LAST("bge_process_kernel");
    }
} // namespace process::gpu
