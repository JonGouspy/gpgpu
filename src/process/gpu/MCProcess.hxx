#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "MCProcess.hpp"

namespace process::gpu
{
    inline const Image<uint8_t>& MCProcess::mask_get() const
    {
        return this->mask_;
    }

    inline float MCProcess::grayscale(const rgb8& rgb) const
    {
        return 0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b;
    }

    inline rgb8 MCProcess::viridis(float t)
    {
        static const rgb8 pal[] = { { 68, 1, 84 },
                                    { 59, 82, 139 },
                                    { 33, 145, 140 },
                                    { 94, 201, 98 },
                                    { 253, 231, 37 } };

        t = std::clamp(t, 0.f, 1.f) * 4.f;
        const int i = std::min(int(t), 3);
        const float f = t - i;
        const rgb8 a = pal[i], b = pal[i + 1];

        return { uint8_t(a.r + f * (b.r - a.r)), uint8_t(a.g + f * (b.g - a.g)),
                 uint8_t(a.b + f * (b.b - a.b)) };
    }
} // namespace process::gpu
