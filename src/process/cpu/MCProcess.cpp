#include "MCProcess.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

#include "Image.hpp"

namespace process::cpu
{
    MCProcess::MCProcess(ImageView<rgb8> img, ImageView<rgb8> bg,
                         int opening_size, int th_low, int th_high)
        : Process(img)
        , bg_(bg)
        , mask_(img.width * img.height, 0.f)
        , opening_size_(
              opening_size > 0
                  ? opening_size
                  : std::max(1, std::min(img.width, img.height) / 240))
        , th_low_(th_low)
        , th_high_(th_high)
    {}

    void MCProcess::compute_mask()
    {
        for (int y = 0; y < this->height_; ++y)
        {
            const rgb8* frame_lineptr = this->input_line_get(y);
            const rgb8* bg_lineptr = this->line_get<rgb8>(this->bg_, y);

            for (int x = 0; x < this->width_; ++x)
            {
                const int dr =
                    std::abs(int(frame_lineptr[x].r) - int(bg_lineptr[x].r));
                const int dg =
                    std::abs(int(frame_lineptr[x].g) - int(bg_lineptr[x].g));
                const int db =
                    std::abs(int(frame_lineptr[x].b) - int(bg_lineptr[x].b));
                mask_[y * this->width_ + x] = std::max({ dr, dg, db });
            }
        }
    }

    // FIXME: We should optimized it.
    template <typename Compare>
    void MCProcess::compute_kernel(float dvalue, Compare fun)
    {
        const int w = this->width_;
        const int h = this->height_;
        const int r = this->opening_size_;

        std::vector<float> res(w * h);

        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
            {
                float value = dvalue;
                for (int i = y - r; i <= y + r; ++i)
                {
                    if (i < 0 || i >= h)
                        continue;

                    for (int j = x - r; j <= x + r; ++j)
                    {
                        if (j < 0 || j >= w)
                            continue;

                        const float pixel = this->mask_[i * w + j];

                        if (fun(pixel, value))
                            value = pixel;
                    }
                }

                res[y * w + x] = value;
            }

        this->mask_ = std::move(res);
    }

    void MCProcess::compute_erosion()
    {
        compute_kernel(255, [](float a, float b) { return a < b; });
    }

    void MCProcess::compute_dilation()
    {
        compute_kernel(0, [](float a, float b) { return a > b; });
    }

    // Hysteresis
    constexpr int NEIGHBORS[4][2] = { { -1, 0 },
                                      { 1, 0 },
                                      { 0, -1 },
                                      { 0, 1 } };

    // CHORE: CUDA version in slides
    void MCProcess::compute_hysteresis()
    {
        const int w = this->width_;
        const int h = this->height_;

        std::vector<uint8_t> out(w * h, 0);

        bool has_changed = true;
        while (has_changed)
        {
            has_changed = false;

            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                {
                    const int p = y * w + x;

                    if (out[p] || mask_[p] < th_low_)
                        continue;

                    if (mask_[p] >= th_high_)
                    {
                        out[p] = 1;
                        has_changed = true;
                        continue;
                    }

                    for (const auto& n : NEIGHBORS)
                    {
                        const int qy = y + n[0];
                        const int qx = x + n[1];
                        if (qy < 0 || qy >= h || qx < 0 || qx >= w)
                            continue;

                        if (out[qy * w + qx])
                        {
                            out[p] = 1;
                            has_changed = true;
                            break;
                        }
                    }
                }
        }

        for (int p = 0; p < w * h; ++p)
            mask_[p] = out[p] ? 255.f : 0.f;
    }

    void MCProcess::run()
    {
        compute_mask();
        compute_erosion();
        compute_dilation();
        compute_hysteresis();

        for (int y = 0; y < this->height_; ++y)
        {
            rgb8* lineptr = this->input_line_get(y);

            for (int x = 0; x < this->width_; ++x)
            {
                const float m = mask_[y * this->width_ + x] / 255.f;
                const float r = lineptr[x].r + 0.5f * 255.f * m;
                lineptr[x].r = std::min(r, 255.f);
            }
        }
    }

    void MCProcess::mask_draw(ImageView<rgb8> out)
    {
        for (int y = 0; y < out.height; ++y)
        {
            rgb8* lineptr = this->line_get<rgb8>(out, y);
            for (int x = 0; x < out.width; ++x)
            {
                // lineptr[x] = viridis(mask_[y * out.width + x] / 50.f);
                const uint8_t value = mask_[y * out.width + x];
                lineptr[x] = { value, value, value };
            }
        }
    }
} // namespace process::cpu
