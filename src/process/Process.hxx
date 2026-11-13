#pragma once

#include <iostream>
#include <stdexcept>

#include "Process.hpp"

namespace process
{
    CUDA_HOST_DEVICE inline rgb8* Process::input_line_get(int y)
    {
        return (rgb8*)((std::byte*)this->buffer_ + y * this->stride_);
    }

    CUDA_HOST_DEVICE inline const rgb8* Process::input_line_get(int y) const
    {
        return (const rgb8*)((const std::byte*)this->buffer_
                             + y * this->stride_);
    }

    template <typename T>
    CUDA_HOST_DEVICE inline T* Process::line_get(ImageView<T>& view, int y)
    {
        return (T*)((std::byte*)view.buffer + y * view.stride);
    }

    template <typename T>
    CUDA_HOST_DEVICE inline const T* Process::line_get(const ImageView<T>& view,
                                                       int y)
    {
        return (const T*)((const std::byte*)view.buffer + y * view.stride);
    }
} // namespace process
