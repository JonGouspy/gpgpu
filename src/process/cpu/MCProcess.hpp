#pragma once

#include <vector>

#include "Process.hpp"

namespace process::cpu
{
    class MCProcess : public Process
    {
    public:
        MCProcess(ImageView<rgb8> img, ImageView<rgb8> bg, int opening_size,
                  int th_low, int th_high);

        const inline std::vector<float>& mask_get() const;

        void run() override;

    private:
        void compute_mask();

        template <typename Compare>
        void compute_kernel(float dvalue, Compare fun);

        void compute_erosion();

        void compute_dilation();

        void compute_hysteresis();

        inline float grayscale(const rgb8& rgb) const;

        static inline rgb8 viridis(float t);

        void mask_draw(ImageView<rgb8> out);

        ImageView<rgb8> bg_;
        std::vector<float> mask_;
        int opening_size_;
        int th_low_;
        int th_high_;
    };
} // namespace process::cpu

#include "MCProcess.hxx"
