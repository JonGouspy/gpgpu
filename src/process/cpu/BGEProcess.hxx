#pragma once

#include "BGEProcess.hpp"

namespace process::cpu
{
    inline void ReservoirState::set(std::size_t w, const rgb8& rgb)
    {
        this->weight = w;
        this->color = rgb;
    }

    inline bool ReservoirState::matches(const rgb8& rgb) const
    {
        return abs(int(rgb.r) - this->color.r) < RGB_DIFF_THRESHOLD
            && abs(int(rgb.g) - this->color.g) < RGB_DIFF_THRESHOLD
            && abs(int(rgb.b) - this->color.b) < RGB_DIFF_THRESHOLD;
    }

    inline void ReservoirState::update(const rgb8& rgb)
    {
        this->weight = std::min(this->weight + 1, MAX_WEIGHT);
        const uint8_t w = this->weight;

        this->color.r = ((w - 1) * this->color.r + rgb.r) / w;
        this->color.g = ((w - 1) * this->color.g + rgb.g) / w;
        this->color.b = ((w - 1) * this->color.b + rgb.b) / w;
    }

    inline float BGEProcess::randfloat()
    {
        return this->dis_(this->gen_);
    }
} // namespace process::cpu
