#pragma once

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

#define CUDA_CHECK(expr)                                                       \
    ::process::gpu::utils::detail::check((expr), #expr, __FILE__, __LINE__)

#define CUDA_CHECK_LAST(kernel_name)                                           \
    ::process::gpu::utils::detail::check(cudaPeekAtLastError(), (kernel_name), \
                                         __FILE__, __LINE__)

namespace process::gpu::utils
{
    inline constexpr int BLOCK_SIZE_2D = 32;
    inline constexpr int BLOCK_SIZE_1D = 256;

    inline dim3 block_2d()
    {
        return dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
    }

    inline dim3 grid_2d(int width, int height)
    {
        return dim3((width + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D,
                    (height + BLOCK_SIZE_2D - 1) / BLOCK_SIZE_2D);
    }

    inline dim3 block_1d()
    {
        return dim3(BLOCK_SIZE_1D);
    }

    inline dim3 grid_1d(int size)
    {
        return dim3((size + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D);
    }

    namespace detail
    {
        inline void check(cudaError_t rc, const char* what, const char* file,
                          int line)
        {
            if (rc == cudaSuccess)
                return;

            std::ostringstream msg;
            msg << "[CUDA Error] " << cudaGetErrorName(rc) << "\n"
                << "  Details: " << cudaGetErrorString(rc) << "\n"
                << "  Call:    " << what << "\n"
                << "  At:      " << file << ":" << line;
            throw std::runtime_error(msg.str());
        }
    } // namespace detail
} // namespace process::gpu::utils
