#include "cuda_median_filter.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <map>

// 显存缓冲池结构
struct BufferPool {
    unsigned char* d_src;
    unsigned char* d_dst;
    int allocated_width;
    int allocated_height;
    bool is_allocated;
};

// 全局缓冲池映射
static std::map<CudaBufferPool, BufferPool> g_buffer_pools;

// CUDA核函数：中值滤波
__global__ void median_filter_kernel(const unsigned char src, unsigned char dst, int width, int height, int kernel_size)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half = kernel_size / 2;
    if (x >= width || y >= height) return;

    unsigned char window[49]; // 最大支持7x7
    int count = 0;
    for (int dy = -half; dy <= half; ++dy) {
        for (int dx = -half; dx <= half; ++dx) {
            int ix = min(max(x + dx, 0), width - 1);
            int iy = min(max(y + dy, 0), height - 1);
            window[count++] = src[iy * width + ix];
        }
    }
  // 排序取中值
    for (int i = 0; i < count - 1; ++i) {
        for (int j = 0; j < count - i - 1; ++j) {
            if (window[j] > window[j + 1]) {
                unsigned char tmp = window[j];
                window[j] = window[j + 1];
                window[j + 1] = tmp;
            }
        }
    }
    dst[y * width + x] = window[count / 2];
}

// 初始化显存缓冲池
extern "C" DLL_EXPORT 
int CudaBufferPool cuda_init_buffer_pool(int max_width, int max_height)
{
    BufferPool* pool = new BufferPool();
    pool->d_src = nullptr;
    pool->d_dst = nullptr;
    pool->allocated_width = 0;
    pool->allocated_height = 0;
    pool->is_allocated = false;
    
    g_buffer_pools[pool] = *pool;
    return pool;
}

// 清理显存缓冲池
extern "C" DLL_EXPORT
int cuda_cleanup_buffer_pool(CudaBufferPool pool)
{
    if (!pool) return -1;
    
    auto it = g_buffer_pools.find(pool);
    if (it == g_buffer_pools.end()) return -2;
    
    BufferPool& buffer_pool = it->second;
    if (buffer_pool.is_allocated) {
        cudaFree(buffer_pool.d_src);
        cudaFree(buffer_pool.d_dst);
    }
    
    g_buffer_pools.erase(it);
    delete (BufferPool*)pool;
    return 0;
}

// 使用缓冲池的中值滤波（高性能版本）
extern "C" DLL_EXPORT
int cuda_median_filter_with_pool(
    CudaBufferPool pool,
    const unsigned char* src,
    int width,
    int height,
    int kernel_size,
    unsigned char* dst)
{
    if (!pool || !src || !dst || width <= 0 || height <= 0 || 
        kernel_size < 3 || (kernel_size % 2) == 0 || kernel_size > 7)
        return -1;
    
    auto it = g_buffer_pools.find(pool);
    if (it == g_buffer_pools.end()) return -2;
    
    BufferPool& buffer_pool = it->second;
    size_t img_size = width  height  sizeof(unsigned char);
    
    // 检查是否需要重新分配显存
    if (!buffer_pool.is_allocated || 
        buffer_pool.allocated_width < width || 
        buffer_pool.allocated_height < height) {
        
        // 释放旧的显存
        if (buffer_pool.is_allocated) {
            cudaFree(buffer_pool.d_src);
            cudaFree(buffer_pool.d_dst);
        }
        
        // 分配新的显存
        cudaError_t err = cudaMalloc(&buffer_pool.d_src, img_size);
        if (err != cudaSuccess) return -3;
        
        err = cudaMalloc(&buffer_pool.d_dst, img_size);
        if (err != cudaSuccess) {
            cudaFree(buffer_pool.d_src);
            return -4;
        }
        
        buffer_pool.allocated_width = width;
        buffer_pool.allocated_height = height;
        buffer_pool.is_allocated = true;
    }
    
    // 复制数据到GPU
    cudaError_t err = cudaMemcpy(buffer_pool.d_src, src, img_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) return -5;
    
    // 执行滤波
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    median_filter_kernel<<<grid, block>>>(buffer_pool.d_src, buffer_pool.d_dst, width, height, kernel_size);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return -6;
    
    // 复制结果回CPU
    err = cudaMemcpy(dst, buffer_pool.d_dst, img_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) return -7;
    
    return 0;
}

// 原始接口（保持兼容性）
extern "C" DLL_EXPORT
int cuda_median_filter(const unsigned char src, int width, int height, int kernel_size, unsigned char dst)
{
    if (!src || !dst || width <= 0 || height <= 0 || kernel_size < 3 || (kernel_size % 2) == 0 || kernel_size > 7)
        return -1;
    size_t img_size = width  height  sizeof(unsigned char);
    unsigned char d_src = nullptr, d_dst = nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_src, img_size);
    if (err != cudaSuccess) return -2;
    err = cudaMalloc(&d_dst, img_size);
    if (err != cudaSuccess) { cudaFree(d_src); return -3; }
    err = cudaMemcpy(d_src, src, img_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_src); cudaFree(d_dst); return -4; }

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    median_filter_kernel<<<grid, block>>>(d_src, d_dst, width, height, kernel_size);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { cudaFree(d_src); cudaFree(d_dst); return -5; }
    err = cudaMemcpy(dst, d_dst, img_size, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
    if (err != cudaSuccess) return -6;
    return 0;
}
