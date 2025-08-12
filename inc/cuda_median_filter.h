#pragma once
#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif
//显存缓冲池句柄
typedef void* CudaBufferPool;
//初始化显存缓冲池
extern "C" DLL_EXPORT
CudaBufferPool cuda_init_buffer_pool(int max_width, int max_height);
//清理显存缓冲池
extern "C" DLL_EXPORT
int cuda_cleanup_buffer_pool(CudaBufferPool pool);
//使用缓冲池的中值滤波（高性能版本）
extern "C" DLL_EXPORT
int cuda_median_filter_with_pool(
    CudaBufferPool pool,
    const unsigned char*
    src
    int width,
    int height,
    int kernel_size,
    unsigned char* dst);
//原始接口（保持兼容性）
extern "C" DLL_EXPORT
int cuda_median_filter(
    const unsigned char* src,
    int width,
    int height,
    int kernel_size,
    unsigned char* dst);
