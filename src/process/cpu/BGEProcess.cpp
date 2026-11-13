#include "BGEProcess.hpp"

#include <algorithm>
#include <cstddef>

namespace process::cpu
{
    BGEProcess::BGEProcess(ImageView<rgb8> img, std::size_t reservoir_size)
        : Process(img)
        , gen_(rd_())
        , dis_(0.0f, 1.f)
        , k_(std::clamp<std::size_t>(reservoir_size, 1, RESERVOIR_K_MAX))
        , rs_(img.width * img.height)
    {}

    rgb8 BGEProcess::get_rgb_max_weight(const Reservoir& rs) const
    {
        auto it =
            std::max_element(rs.begin(), rs.begin() + k_, [](auto& a, auto& b) {
                return a.weight < b.weight;
            });

        return it->color;
    }

    ssize_t BGEProcess::find_matching_reservoir(const rgb8& rgb,
                                                const Reservoir& rs) const
    {
        ssize_t m_idx = -1;

        for (std::size_t i = 0; i < k_; ++i)
        {
            const ReservoirState& state = rs[i];
            if (state.weight > 0)
            {
                if (state.matches(rgb))
                {
                    m_idx = i;
                    break;
                }
            }
            else
                m_idx = i;
        }

        return m_idx;
    }

    void BGEProcess::run()
    {
        for (int y = 0; y < this->height_; ++y)
        {
            rgb8* lineptr = this->input_line_get(y);

            for (int x = 0; x < this->width_; ++x)
            {
                const size_t index = y * this->width_ + x;
                Reservoir& rs = rs_[index];
                const rgb8 prgb = lineptr[x];
                const ssize_t m_idx = find_matching_reservoir(prgb, rs);

                if (m_idx != -1 && rs[m_idx].weight > 0)
                {
                    rs[m_idx].update(prgb);
                }
                else if (m_idx != -1 && rs[m_idx].weight == 0) // empty slot
                {
                    rs[m_idx].set(1, prgb);
                }
                else // no match
                {
                    size_t min_idx = 0;
                    size_t total_weights = 0;
                    for (size_t k = 0; k < rs.size(); ++k)
                    {
                        if (rs[min_idx].weight > rs[k].weight)
                            min_idx = k;
                        total_weights += rs[k].weight;
                    }
                    if (randfloat() * total_weights >= rs[min_idx].weight)
                    {
                        rs[min_idx].set(1, prgb);
                    }
                }

                // cap weight (done with std::min before)
                // set pixel to max weight
                lineptr[x] = get_rgb_max_weight(rs);
            }
        }
    }
} // namespace process::cpu
