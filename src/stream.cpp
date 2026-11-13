// Create a GStreamer pipeline to stream a video through our plugin

#include <cstdlib>
#include <format>
#include <gst/gst.h>

#include "argh.h"
#include "gstfilter.h"

static gboolean plugin_init(GstPlugin* plugin)
{
    /* FIXME Remember to set the rank if it's an element that is meant
       to be autoplugged by decodebin. */
    return gst_element_register(plugin, "myfilter", GST_RANK_NONE,
                                GST_TYPE_MYFILTER);
}

static void my_code_init()
{
    gst_plugin_register_static(GST_VERSION_MAJOR, GST_VERSION_MINOR, "myfilter",
                               "Private elements of my application",
                               plugin_init, "1.0", "LGPL", "", "", "");
}

int main(int argc, char* argv[])
{
    argh::parser cmdl(argc, argv);
    if (cmdl[{ "-h", "--help" }])
    {
        g_printerr("Usage: %s --mode=[gpu,cpu] <filename> "
                   "[--output=output.mp4] "
                   "[--bg=<image>] "
                   "[--opening_size=3] [--low=3] [--high=30] "
                   "[--bg_sampling_rate=500] [--bg_number_frame=10]\n",
                   argv[0]);
        return 0;
    }

    Parameters params;
    auto method = cmdl("mode", "cpu").str();
    auto filename = cmdl(1).str();
    auto output = cmdl({ "-o", "--output" }, "").str();
    if (method == "cpu")
        params.device = e_device_t::CPU;
    else if (method == "gpu")
        params.device = e_device_t::GPU;
    else
    {
        g_printerr("Invalid method: %s\n", method.c_str());
        return 1;
    }

    // Optional background image: if set, skips background estimation
    auto bg = cmdl("bg", "").str();
    params.bg = bg.empty() ? nullptr : bg.c_str();

    // Opening radius (window (2r+1)x(2r+1)), negatif = derived from image size
    params.opening_size = atoi(cmdl("opening_size", -1).str().c_str());

    // Hysteresis thresholds, scale depends on the color space used
    params.th_low = atoi(cmdl("low", -1).str().c_str());
    params.th_high = atoi(cmdl("high", -1).str().c_str());

    // Background estimation: sampling interval and number of frames kept
    params.bg_sampling_rate = atoi(cmdl("bg_sampling_rate", 500).str().c_str());
    params.bg_number_frame = atoi(cmdl("bg_number_frame", 10).str().c_str());

    g_debug("Using method: %s", method.c_str());
    gst_init(&argc, &argv);
    my_code_init();
    cpt_init(&params);

    // Create a GStreamer pipeline to stream a video through our plugin
    const char* pipe_str;
    g_print("Output: %s\n", output.c_str());
    if (output.empty())
        pipe_str = "filesrc name=fsrc ! decodebin ! videoconvert ! "
                   "video/x-raw, format=(string)RGB ! myfilter ! videoconvert "
                   "! fpsdisplaysink sync=false";
    else
        pipe_str =
            "filesrc name=fsrc ! decodebin ! videoconvert ! video/x-raw, "
            "format=(string)RGB ! myfilter ! videoconvert ! video/x-raw, "
            "format=I420 ! x264enc ! mp4mux ! filesink name=fdst";

    GError* error = NULL;
    auto pipeline = gst_parse_launch(pipe_str, &error);
    if (!pipeline)
    {
        g_printerr("Failed to create pipeline: %s\n", error->message);
        return 1;
    }

    auto filesrc = gst_bin_get_by_name(GST_BIN(pipeline), "fsrc");
    g_object_set(filesrc, "location", filename.c_str(), NULL);
    g_object_unref(filesrc);

    if (!output.empty())
    {
        auto filesink = gst_bin_get_by_name(GST_BIN(pipeline), "fdst");
        g_object_set(filesink, "location", output.c_str(), NULL);
        g_object_unref(filesink);
    }

    // Start the pipeline
    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    // Wait until error or EOS
    GstBus* bus = gst_element_get_bus(pipeline);
    GstMessage* msg = gst_bus_timed_pop_filtered(
        bus, GST_CLOCK_TIME_NONE,
        (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    // Free resources
    if (msg != nullptr)
        gst_message_unref(msg);
    gst_object_unref(bus);
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return 0;
}
