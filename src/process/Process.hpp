#pragma once

#include <cstddef>

#include "Image.hpp"

#if defined(__CUDACC__) || defined(__nvcc__)
#    define CUDA_HOST_DEVICE __host__ __device__
#else
#    define CUDA_HOST_DEVICE
#endif

namespace process
{
    class Process
    {
    public:
        Process(ImageView<rgb8> img)
            : buffer_(img.buffer)
            , width_(img.width)
            , height_(img.height)
            , stride_(img.stride)
        {}

        virtual ~Process() = default;

        virtual void run() = 0;

        template <typename T>
        CUDA_HOST_DEVICE static inline T* line_get(ImageView<T>& view, int y);
        template <typename T>
        CUDA_HOST_DEVICE static inline const T*
        line_get(const ImageView<T>& view, int y);

    protected:
        CUDA_HOST_DEVICE inline rgb8* input_line_get(int y);
        CUDA_HOST_DEVICE inline const rgb8* input_line_get(int y) const;

        rgb8* buffer_;
        int width_;
        int height_;
        std::ptrdiff_t stride_;
    };
} // namespace process

#include "Process.hxx"
