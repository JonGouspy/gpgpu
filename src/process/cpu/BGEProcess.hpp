#pragma once

#include <array>
#include <random>
#include <vector>

#include "Process.hpp"

namespace process::cpu
{
    constexpr std::size_t RESERVOIR_K_MAX = 32;
    constexpr uint8_t RGB_DIFF_THRESHOLD = 10;
    constexpr std::size_t MAX_WEIGHT = 200;

    struct ReservoirState
    {
        std::size_t weight = 0;
        rgb8 color;

        inline void set(std::size_t w, const rgb8& rgb);
        inline bool matches(const rgb8& rgb) const;
        inline void update(const rgb8& rgb);
    };

    class BGEProcess : public Process
    {
    private:
        using Reservoir = std::array<ReservoirState, RESERVOIR_K_MAX>;

    public:
        BGEProcess(ImageView<rgb8> img, std::size_t reservoir_size);

        void run() override;

    private:
        ssize_t find_matching_reservoir(const rgb8& rgb,
                                        const Reservoir& rs) const;
        rgb8 get_rgb_max_weight(const Reservoir& rs) const;
        inline float randfloat();

        std::random_device rd_;
        std::mt19937 gen_;
        std::uniform_real_distribution<float> dis_;
        std::size_t k_;
        std::vector<Reservoir> rs_;
    };
} // namespace process::cpu

#include "BGEProcess.hxx"
