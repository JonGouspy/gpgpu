#include <cstdint>
#include <vector>

#include "CudaUtils.cuh"
#include "Image.hpp"
#include "MCProcess.hpp"

namespace process::gpu
{
    namespace
    {
        __global__ void mask_kernel(ImageView<rgb8> in, ImageView<rgb8> bg,
                                    ImageView<uint8_t> mask)
        {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x >= in.width || y >= in.height)
                return;

            const rgb8* frame_lineptr = Process::line_get(in, y);
            const rgb8* bg_lineptr = Process::line_get(bg, y);

            const int dr =
                std::abs(int(frame_lineptr[x].r) - int(bg_lineptr[x].r));
            const int dg =
                std::abs(int(frame_lineptr[x].g) - int(bg_lineptr[x].g));
            const int db =
                std::abs(int(frame_lineptr[x].b) - int(bg_lineptr[x].b));

            mask.at(x, y) = max(dr, max(dg, db));
        }

        struct MinOp
        {
            __device__ bool operator()(uint8_t a, uint8_t b) const
            {
                return a < b;
            }
        };

        struct MaxOp
        {
            __device__ bool operator()(uint8_t a, uint8_t b) const
            {
                return a > b;
            }
        };

        template <typename Compare>
        __global__ void
        morpho_kernel(const ImageView<uint8_t> in, ImageView<uint8_t> out,
                      int r, uint8_t dvalue, Compare fun, int dx, int dy)
        {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x >= in.width || y >= in.height)
                return;

            uint8_t value = dvalue;
            for (int k = -r; k <= r; ++k)
            {
                const int i = y + k * dy;
                const int j = x + k * dx;

                if (i < 0 || i >= in.height || j < 0 || j >= in.width)
                    continue;

                const uint8_t pixel = in.at(j, i);

                if (fun(pixel, value))
                    value = pixel;
            }

            out.at(x, y) = value;
        }

        // Separable erosion / dilation done entirely in shared memory.
        // Same result as the (2r+1)x(2r+1) square, bit for bit.
        // Requires a square block (blockDim.x == blockDim.y).
        //
        // Shared memory layout (example: B = 4, r = 2, T = 8)
        //
        //   in_tile : T x T                 h_tile : T x B      out : B x B
        //   (tile + halo)                   (horizontal)        (vertical)
        //
        //      <-------- T --------->          <-- B --->       <-- B --->
        //      <r->  <-- B --->  <r->
        //   ^  h  h  h  h  h  h  h  h       ^  h  h  h  h
        //   |  h  h  h  h  h  h  h  h       |  h  h  h  h
        //   |  h  h  #  #  #  #  h  h       |  v  v  v  v       o  o  o  o
        //   |  h  h  #  #  #  #  h  h       |  v  v  v  v       o  o  o  o
        //   T  h  h  #  #  #  #  h  h  ==>  T  v  v  v  v  ==>  o  o  o  o
        //   |  h  h  #  #  #  #  h  h       |  v  v  v  v       o  o  o  o
        //   |  h  h  h  h  h  h  h  h       |  h  h  h  h
        //   v  h  h  h  h  h  h  h  h       v  h  h  h  h
        //
        //   h = halo
        //   # = tile pixels
        //   v = horizontal results
        //   o = output pixels
        //
        //   (x0, y0) = image position of in_tile[0]
        //            = (blockIdx.x * B - r, blockIdx.y * B - r)
        //
        //   1) load:       in_tile[i]     = in(x0 + i % T, y0 + i / T)
        //   2) horizontal: h_tile[row][c] = op(in_tile[row][c .. c + 2r])
        //   3) vertical:   out(x, y)      = op(h_tile[ty .. ty + 2r][tx])
        //
        //   mem: one dynamic buffer, mem_bytes = T * T + T * B
        //   +--------------------------------+----------------------+
        //   | in_tile : T * T bytes          | h_tile : T * B       |
        //   +--------------------------------+----------------------+
        //   ^ mem + 0                        ^ mem + T * T
        template <typename Compare>
        __global__ void morpho_shared_kernel(const ImageView<uint8_t> in,
                                             ImageView<uint8_t> out, int r,
                                             uint8_t dvalue, Compare fun)
        {
            extern __shared__ uint8_t mem[];

            const int B = blockDim.x;
            const int T = B + 2 * r;
            uint8_t* in_tile = mem;
            uint8_t* h_tile = mem + T * T;

            const int x0 = blockIdx.x * B - r;
            const int y0 = blockIdx.y * B - r;
            const int tid = threadIdx.y * B + threadIdx.x;
            const int nbthreads = B * blockDim.y;

            for (int i = tid; i < T * T; i += nbthreads)
            {
                const int gx = x0 + i % T;
                const int gy = y0 + i / T;

                in_tile[i] =
                    (gx >= 0 && gx < in.width && gy >= 0 && gy < in.height)
                    ? in.at(gx, gy)
                    : dvalue;
            }
            __syncthreads();

            for (int i = tid; i < T * B; i += nbthreads)
            {
                const uint8_t* row = in_tile + (i / B) * T + (i % B);
                uint8_t value = dvalue;

                for (int k = 0; k <= 2 * r; ++k)
                    if (fun(row[k], value))
                        value = row[k];

                h_tile[i] = value;
            }
            __syncthreads();

            const int x = blockIdx.x * B + threadIdx.x;
            const int y = blockIdx.y * B + threadIdx.y;

            if (x >= out.width || y >= out.height)
                return;

            uint8_t value = dvalue;

            for (int k = 0; k <= 2 * r; ++k)
            {
                const int idx = (threadIdx.y + k) * B + threadIdx.x;
                uint8_t pixel = h_tile[idx];
                if (fun(pixel, value))
                    value = pixel;
            }

            out.at(x, y);
        }

        template <typename Compare>
        void compute_morpho(ImageView<uint8_t>& d_mask,
                            ImageView<uint8_t>& d_tmp, int r, uint8_t dvalue,
                            Compare fun)
        {
            const dim3 grid = utils::grid_2d(d_mask.width, d_mask.height);
            const dim3 block = utils::block_2d();

            constexpr int B = utils::BLOCK_SIZE_2D;
            const int T = B + 2 * r;
            const int mem_bytes = T * T + T * B;

            int max_smem = 0;
            cudaDeviceGetAttribute(&max_smem,
                                   cudaDevAttrMaxSharedMemoryPerBlock, 0);

            if (max_smem >= mem_bytes)
            {
                morpho_shared_kernel<<<grid, block, mem_bytes>>>(
                    d_mask, d_tmp, r, dvalue, fun);
                CUDA_CHECK_LAST("morpho_shared_kernel");

                std::swap(d_mask, d_tmp);
            }
            else
            {
                morpho_kernel<<<grid, block>>>(d_mask, d_tmp, r, dvalue, fun, 1,
                                               0);
                morpho_kernel<<<grid, block>>>(d_tmp, d_mask, r, dvalue, fun, 0,
                                               1);
                CUDA_CHECK_LAST("morpho_kernel");
            }
        }

        // Hysteresis thresholding by block-wise propagation in shared memory.
        // A pixel is marked if mask >= th_high, or if mask >= th_low and it is
        // 4-connected (through pixels >= th_low) to a marked pixel.
        // Requires a square block (blockDim.x == blockDim.y == BS).
        //
        // Each launch: a block loads its tile of `marker` (+1 halo from the
        // neighbors, 0 outside the image), propagates inside it until nothing
        // changes, then writes back. Only block borders need another launch.
        //
        //   tile : (BS + 2) x (BS + 2)
        //        .  h  h  h  h  .       h = halo (neighbor markers, global
        //        memory) h  #  #  #  #  h       # = tile pixels (one per
        //        thread) h  #  #  #  #  h       . = unused corner
        //        (4-connectivity) .  h  h  h  h  .
        //
        // Per-thread state:
        //   candidate  : mask >= th_low   (may be marked by propagation)
        //   strong     : mask >= th_high  (seed, marked right away)
        //   was_marked : marker already 1 at the start of this launch
        //
        // Steps:
        //   0) skip:      the block runs only if active_in[b] or one of its 4
        //                 neighbor blocks is set (something changed there last
        //                 launch). Otherwise tile and halo are unchanged, so is
        //                 the result. First launch: everything is active.
        //   1) load:      tile = was_marked || strong, plus halo on edge
        //   threads 2) propagate: unmarked candidates with a marked 4-neighbor
        //   become
        //                 marked, until no thread marks anything
        //                 (__syncthreads_or: all threads exit on the same
        //                 round)
        //   3) write:     new marks -> marker = 1, has_changed = 1,
        //   active_out[b] = 1
        //
        // Notes:
        //   - The early return of step 0 depends only on blockIdx and precedes
        //     every barrier: no deadlock.
        //   - Marks only go 0 -> 1, so races on tile/marker are harmless: a
        //   missed
        //     mark is seen at the next round or the next launch.
        //   - The host loop stops when a launch changes nothing.
        __global__ void hysteresis_block_kernel(const ImageView<uint8_t> mask,
                                                ImageView<std::uint8_t> marker,
                                                int th_low, int th_high,
                                                int* has_changed,
                                                std::uint8_t* active_in,
                                                std::uint8_t* active_out)
        {
            // Remove useless iteration: bloc where no adjacent pixel was
            // modified
            const int bx = blockIdx.x, by = blockIdx.y;
            const int gw = gridDim.x, gh = gridDim.y;
            const int b = by * gw + bx;

            const bool act = active_in[b] || (bx > 0 && active_in[b - 1])
                || (bx < gw - 1 && active_in[b + 1])
                || (by > 0 && active_in[b - gw])
                || (by < gh - 1 && active_in[b + gw]);

            if (!act)
                return;

            constexpr int BS = utils::BLOCK_SIZE_2D;

            // +2 for neightboors on most left/top/bottom/right pixel
            __shared__ std::uint8_t tile[BS + 2][BS + 2];

            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            // +1 to recenter after adding extremity neightbors
            const int tile_x = threadIdx.x + 1;
            const int tile_y = threadIdx.y + 1;

            const bool inside = x < mask.width && y < mask.height;

            const uint8_t m = inside ? mask.at(x, y) : 0;
            const bool candidate = inside && m >= th_low;
            const bool strong = candidate && m >= th_high;
            const bool was_marked = inside && marker.at(x, y);

            auto get_safe_pixel = [&](int pos_x, int pos_y) -> std::uint8_t {
                return (pos_x >= 0 && pos_x < mask.width && pos_y >= 0
                        && pos_y < mask.height)
                    ? marker.at(pos_x, pos_y)
                    : 0;
            };

            tile[tile_y][tile_x] = was_marked || strong;

            if (threadIdx.x == 0)
                tile[tile_y][0] = get_safe_pixel(x - 1, y);

            if (threadIdx.x == BS - 1)
                tile[tile_y][BS + 1] = get_safe_pixel(x + 1, y);

            if (threadIdx.y == 0)
                tile[0][tile_x] = get_safe_pixel(x, y - 1);

            if (threadIdx.y == BS - 1)
                tile[BS + 1][tile_x] = get_safe_pixel(x, y + 1);

            __syncthreads();

            bool marked = tile[tile_y][tile_x];
            bool newly;
            do
            {
                newly = false;
                if (candidate && !marked
                    && (tile[tile_y - 1][tile_x] | tile[tile_y + 1][tile_x]
                        | tile[tile_y][tile_x - 1] | tile[tile_y][tile_x + 1]))
                {
                    tile[tile_y][tile_x] = 1;
                    marked = true;
                    newly = true;
                }

            } while (__syncthreads_or(newly));

            if (marked && !was_marked)
            {
                marker.at(x, y) = 1;
                *has_changed = true;
                active_out[b] = 1;
            }
        }

        __global__ void binarize_kernel(ImageView<uint8_t> mask,
                                        const ImageView<std::uint8_t> marker)
        {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x >= mask.width || y >= mask.height)
                return;

            mask.at(x, y) = marker.at(x, y) ? 255 : 0;
        }

        void compute_hysteresis(ImageView<uint8_t> d_mask,
                                ImageView<uint8_t> d_marker, int* d_changed,
                                std::uint8_t* d_act_in, std::uint8_t* d_act_out,
                                int th_low, int th_high)
        {
            // Buffers are allocated once in MCProcess: only reset them here
            CUDA_CHECK(cudaMemset2D(d_marker.buffer, d_marker.stride, 0,
                                    d_mask.width * sizeof(std::uint8_t),
                                    d_marker.height));

            const dim3 grid = utils::grid_2d(d_mask.width, d_mask.height);
            const dim3 block = utils::block_2d();
            const int nblocks = grid.x * grid.y;

            CUDA_CHECK(cudaMemset(d_act_in, 1, nblocks));

            int has_changed = 1;
            while (has_changed)
            {
                CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
                CUDA_CHECK(cudaMemset(d_act_out, 0, nblocks));

                hysteresis_block_kernel<<<grid, block>>>(
                    d_mask, d_marker, th_low, th_high, d_changed, d_act_in,
                    d_act_out);
                CUDA_CHECK_LAST("hysteresis_kernel");

                CUDA_CHECK(cudaMemcpy(&has_changed, d_changed, sizeof(int),
                                      cudaMemcpyDeviceToHost));

                std::swap(d_act_in, d_act_out);
            }

            binarize_kernel<<<grid, block>>>(d_mask, d_marker);
            CUDA_CHECK_LAST("binarize_kernel");
        }

        __global__ void overlay_kernel(ImageView<rgb8> img,
                                       const ImageView<uint8_t> mask)
        {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x >= img.width || y >= img.height)
                return;

            rgb8* lineptr = Process::line_get(img, y);

            const uint8_t m = mask.at(x, y);
            rgb8& pixel = img.at(x, y);
            pixel.r = min(pixel.r + (m >> 1), 255);
        }
    } // namespace

    MCProcess::MCProcess(ImageView<rgb8> d_frame, ImageView<rgb8> d_bg,
                         int opening_size, int th_low, int th_high)
        : Process(d_frame)
        , mask_(d_frame.width, d_frame.height)
        , d_bg_(d_bg)
        , d_mask_(d_frame.width, d_frame.height, true)
        , d_tmp_(d_frame.width, d_frame.height, true)
        , d_marker_(d_frame.width, d_frame.height, true)
        , d_act_in_(nullptr)
        , d_act_out_(nullptr)
        , d_changed_(1, 1, true)
        , opening_size_(
              opening_size > 0
                  ? opening_size
                  : std::max(1, std::min(d_frame.width, d_frame.height) / 240))
        , th_low_(th_low)
        , th_high_(th_high)
    {
        const dim3 grid = utils::grid_2d(d_frame.width, d_frame.height);
        nblocks_ = grid.x * grid.y;

        CUDA_CHECK(cudaMalloc(&d_act_in_, nblocks_));
        CUDA_CHECK(cudaMalloc(&d_act_out_, nblocks_));
    }

    MCProcess::~MCProcess()
    {
        cudaFree(d_act_in_);
        cudaFree(d_act_out_);
    }

    void MCProcess::run()
    {
        const int w = this->width_;
        const int h = this->height_;
        const int os = this->opening_size_;

        const ImageView<rgb8> d_frame{ this->buffer_, w, h, this->stride_ };

        const dim3 block = utils::block_2d();
        const dim3 grid = utils::grid_2d(w, h);

        mask_kernel<<<grid, block>>>(d_frame, d_bg_, d_mask_);
        CUDA_CHECK_LAST("mask_kernel");

        compute_morpho(d_mask_, d_tmp_, os, 255, MinOp{});

        compute_morpho(d_mask_, d_tmp_, os, 0, MaxOp{});

        compute_hysteresis(d_mask_, d_marker_, d_changed_.buffer, d_act_in_,
                           d_act_out_, th_low_, th_high_);

        overlay_kernel<<<grid, block>>>(d_frame, d_mask_);
        CUDA_CHECK_LAST("overlay_kernel");

        // DEBUG
        // CUDA_CHECK(cudaMemcpy(this->mask_.buffer, d_mask_.buffer,
        // w * h * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    }

    void MCProcess::mask_draw(ImageView<rgb8> out)
    {
        for (int y = 0; y < out.height; ++y)
        {
            for (int x = 0; x < out.width; ++x)
            {
                out.at(x, y) = viridis(mask_.at(x, y) / 50.f);
            }
        }
    }
} // namespace process::gpu
