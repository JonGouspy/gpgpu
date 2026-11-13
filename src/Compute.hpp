#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    // Execution parameters taken from the command line
    typedef enum
    {
        CPU,
        GPU
    } e_device_t;

    typedef enum
    {
        RGB_FULL,
        RGB_LIMITED
    } e_colorspace_t;

    typedef struct
    {
        e_device_t device;
        /* Noise suppression parameters */
        const char* bg;
        int opening_size;
        int th_low;
        int th_high;
        /* Background estimation parameters */
        int bg_sampling_rate;
        int bg_number_frame;
    } Parameters;

    /// Global state initialization
    /// This function is called once before any other cpt_* function at the
    /// beginning of the program
    void cpt_init(Parameters* params);

    /// Called by gstreamer once the video format is known, before any frame
    void cpt_set_colorspace(e_colorspace_t colorspace);

    /// Function called by gstreamer to process the incoming frame
    void cpt_process_frame(uint8_t* buffer, int width, int height, int stride);

    extern Parameters g_params;
#ifdef __cplusplus
}
#endif
