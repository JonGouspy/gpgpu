#pragma once

#include <cstdint>
#include <vector>

#include "Process.hpp"

namespace process::gpu
{
    class MCProcess : public Process
    {
    public:
        MCProcess(ImageView<rgb8> d_frame, ImageView<rgb8> d_bg,
                  int opening_size, int th_low, int th_high);

        const inline Image<uint8_t>& mask_get() const;

        void run() override;

        ~MCProcess();

    private:
        inline float grayscale(const rgb8& rgb) const;

        static inline rgb8 viridis(float t);

        void mask_draw(ImageView<rgb8> out);

        Image<uint8_t> mask_; // host copy, debug only

        // Device work buffers
        ImageView<rgb8> d_bg_;
        Image<uint8_t> d_mask_;
        Image<uint8_t> d_tmp_;
        Image<uint8_t> d_marker_;
        std::uint8_t* d_act_in_;
        std::uint8_t* d_act_out_;
        Image<int> d_changed_;

        // Parameters
        int opening_size_;
        int th_low_;
        int th_high_;
        int nblocks_;
    };
} // namespace process::gpu

#include "MCProcess.hxx"
