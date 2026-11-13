#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string_view>

#include "cuda_runtime.h"
#include "stb_image.h"

#if defined(__CUDACC__) || defined(__nvcc__)
#    define CUDA_HOST_DEVICE __host__ __device__
#else
#    define CUDA_HOST_DEVICE
#endif

struct rgb8
{
    uint8_t r, g, b;
};

// View over a 2D buffer
template <class T>
struct ImageView
{
    T* buffer = nullptr;
    int width = 0;
    int height = 0;
    std::ptrdiff_t stride = 0;

    CUDA_HOST_DEVICE T* getline(int y);
    CUDA_HOST_DEVICE T& at(int x, int y);
    CUDA_HOST_DEVICE const T* getline(int y) const;
    CUDA_HOST_DEVICE const T& at(int x, int y) const;
};

// Class that owns data
template <class T>
struct Image : ImageView<T>
{
    void (*deleter)(void*) = nullptr;

    Image() = default;
    Image(int width, int height, bool device = false);
    Image(const char* path);
    Image(std::string_view path)
        : Image(path.data())
    {}
    ~Image();

    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    Image(const Image& other) = delete;
    Image& operator=(const Image& other) = delete;

    Image clone() const;
};

template <class T>
Image<T>::Image(const char* path)
{
    int w, h, n;
    this->buffer = (T*)stbi_load(path, &w, &h, &n, sizeof(T));
    this->width = w;
    this->height = h;
    this->stride = w * sizeof(T);
    this->deleter = stbi_image_free;
}

template <class T>
Image<T>::Image(int width, int height, bool device)
{
    static auto cudaDelete = [](void* ptr) { cudaFree(ptr); };
    this->width = width;
    this->height = height;
    if (device)
    {
        size_t pitch;
        cudaError_t rc = cudaMallocPitch((void**)&this->buffer, &pitch,
                                         this->width * sizeof(T), this->height);
        if (rc)
            throw std::runtime_error("Image cudaMallocPitch failed");
        this->stride = pitch;
        this->deleter = cudaDelete;
    }
    else
    {
        this->stride = width * sizeof(T);
        this->buffer = (T*)malloc(this->height * this->stride);
        this->deleter = free;
    }
}

template <class T>
Image<T>::~Image()
{
    if (this->buffer && this->deleter)
        this->deleter(this->buffer);
    this->buffer = nullptr;
}

template <class T>
Image<T>::Image(Image&& other) noexcept
{
    std::swap((ImageView<T>&)(*this), (ImageView<T>&)(other));
}

template <class T>
Image<T>& Image<T>::operator=(Image&& other) noexcept
{
    std::swap((ImageView<T>&)(*this), (ImageView<T>&)(other));
    return *this;
}

template <class T>
Image<T> Image<T>::clone() const
{
    Image<T> out(this->width, this->height);
    for (int y = 0; y < this->height; ++y)
        std::memcpy((char*)out.buffer + y * out.stride, //
                    (char*)this->buffer + y * this->stride, //
                    this->width * sizeof(T));
    return out;
}

template <class T>
T* ImageView<T>::getline(int y)
{
    return (T*)((std::byte*)this->buffer + y * this->stride);
}

template <class T>
T& ImageView<T>::at(int x, int y)
{
    return this->getline(y)[x];
}

template <class T>
const T* ImageView<T>::getline(int y) const
{
    return (T*)((std::byte*)this->buffer + y * this->stride);
}

template <class T>
const T& ImageView<T>::at(int x, int y) const
{
    return this->getline(y)[x];
}
