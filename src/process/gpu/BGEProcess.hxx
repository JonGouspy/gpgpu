#pragma once

#include <cstddef>
#include <cuda/std/algorithm>

#include "BGEProcess.hpp"

namespace process::gpu
{
    CUDA_HOST_DEVICE inline bool matches(uchar4 c, const rgb8& rgb)
    {
        return abs(int(rgb.r) - c.x) < RGB_DIFF_THRESHOLD
            && abs(int(rgb.g) - c.y) < RGB_DIFF_THRESHOLD
            && abs(int(rgb.b) - c.z) < RGB_DIFF_THRESHOLD;
    }

    CUDA_HOST_DEVICE inline uchar4 to_uchar4(const rgb8& c)
    {
        return make_uchar4(c.r, c.g, c.b, 0);
    }

    CUDA_HOST_DEVICE inline rgb8 to_rgb8(const uchar4& c)
    {
        return rgb8{ c.x, c.y, c.z };
    }
} // namespace process::gpu
