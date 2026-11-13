#include "Compute.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>

#include "Image.hpp"
#include "cpu/BGEProcess.hpp"
#include "cpu/MCProcess.hpp"
#include "logo.h"

/// Your cpp version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cpp(ImageView<rgb8> in);

/// Your CUDA version of the algorithm
/// This function is called by cpt_process_frame for each frame
void compute_cu(ImageView<rgb8> in);

/// CPU Single threaded version of the Method
void compute_cpp(ImageView<rgb8> in)
{
    // Background given on the command line, loaded once
    static Image<rgb8> bg_file =
        g_params.bg ? Image<rgb8>(g_params.bg) : Image<rgb8>();

    if (g_params.bg && !bg_file.buffer)
        throw std::runtime_error("cannot load background image");
    if (bg_file.buffer
        && (bg_file.width != in.width || bg_file.height != in.height))
        throw std::runtime_error("background and video sizes differ");

    // Buffer used for the estimated background
    static std::vector<rgb8> data(in.width * in.height);
    static ImageView<rgb8> bg = bg_file.buffer
        ? ImageView<rgb8>(bg_file)
        : ImageView<rgb8>{ data.data(), in.width, in.height,
                           std::ptrdiff_t(in.width * sizeof(rgb8)) };

    if (!bg_file.buffer)
    {
        using clock = std::chrono::steady_clock;
        static process::cpu::BGEProcess bge(bg, g_params.bg_number_frame);
        static clock::time_point last_sample{};

        const auto now = clock::now();

        if (now - last_sample
            >= std::chrono::milliseconds(g_params.bg_sampling_rate))
        {
            last_sample = now;

            // Copy the frame: BGE turns it in place into the background
            for (int y = 0; y < in.height; ++y)
                std::copy_n(process::Process::line_get(in, y), in.width,
                            process::Process::line_get(bg, y));

            bge.run();
        }
    }

    // Compute motion mask (frame vs bg)
    static process::cpu::MCProcess mc(in, bg, g_params.opening_size,
                                      g_params.th_low, g_params.th_high);
    mc.run();
}

extern "C"
{
    Parameters g_params;

    void cpt_init(Parameters* params)
    {
        g_params = *params;
    }

    void cpt_set_colorspace(e_colorspace_t colorspace)
    {
        int low = 3;
        int high = 30;

        // Limited range: differences are scaled by 219 / 255
        if (colorspace == RGB_LIMITED)
        {
            low = low * 219 / 255;
            high = high * 219 / 255;
        }

        if (g_params.th_low < 0)
            g_params.th_low = low;
        if (g_params.th_high < 0)
            g_params.th_high = high;
    }

    void cpt_process_frame(uint8_t* buffer, int width, int height, int stride)
    {
        auto img = ImageView<rgb8>{ (rgb8*)buffer, width, height, stride };
        if (g_params.device == e_device_t::CPU)
            compute_cpp(img);
        else if (g_params.device == e_device_t::GPU)
            compute_cu(img);
    }
}
