/**
 * CUDA Histogram Implementation
 * Features:
 * - Uses shared memory for efficiency
 * - Implements atomic operations
 * - Saturates counts at 127
 * - Handles up to 4096 bins
 */

#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <cuda_runtime.h>

#define NUM_BINS 4096
#define BLOCK_SIZE 256
#define MAX_COUNT 127

// 检查CUDA错误的辅助函数
#define checkCudaErrors(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

/**
 * 直方图计算的核心kernel
 * 使用共享内存和原子操作来计算直方图
 */
__global__ void histogram_kernel(unsigned int *input, 
                               unsigned int *bins,
                               unsigned int num_elements,
                               unsigned int num_bins) {
    // 声明共享内存
    __shared__ unsigned int shared_bins[NUM_BINS];
    
    // 获取线程ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // 初始化共享内存
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        shared_bins[i] = 0;
    }
    __syncthreads();
    
    // 处理输入数据
    for (int i = tid; i < num_elements; i += stride) {
        atomicAdd(&shared_bins[input[i]], 1);
    }
    __syncthreads();
    
    // 将共享内存中的结果原子性地累加到全局内存
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
        if (shared_bins[i] > 0) {
            atomicAdd(&bins[i], shared_bins[i]);
        }
    }
}

/**
 * 将bin的值限制在127的kernel
 */
__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < num_bins; i += stride) {
        if (bins[i] > MAX_COUNT) {
            bins[i] = MAX_COUNT;
        }
    }
}

/**
 * 计时辅助函数
 */
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

/**
 * 主函数
 */
int main(int argc, char **argv) {
    // 变量声明
    int inputLength;
    unsigned int *hostInput;
    unsigned int *hostBins;
    unsigned int *resultRef;
    unsigned int *deviceInput;
    unsigned int *deviceBins;
    
    // 从命令行参数读取输入长度
    if (argc > 1) {
        inputLength = atoi(argv[1]);
    } else {
        inputLength = 1000000; // 默认值
    }
    printf("The input length is %d\n", inputLength);
    
    // 分配主机内存
    hostInput = (unsigned int*)malloc(inputLength * sizeof(unsigned int));
    hostBins = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    resultRef = (unsigned int*)malloc(NUM_BINS * sizeof(unsigned int));
    
    // 检查内存分配
    if (hostInput == NULL || hostBins == NULL || resultRef == NULL) {
        printf("Error: Host memory allocation failed\n");
        return -1;
    }
    
    // 初始化输入数据
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, NUM_BINS - 1);
    
    for (int i = 0; i < inputLength; i++) {
        hostInput[i] = dis(gen);
    }
    
    // 创建CPU参考结果
    memset(resultRef, 0, NUM_BINS * sizeof(unsigned int));
    for (int i = 0; i < inputLength; i++) {
        if (resultRef[hostInput[i]] < MAX_COUNT) {
            resultRef[hostInput[i]]++;
        }
    }
    
    // 分配GPU内存
    checkCudaErrors(cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int)));
    
    // 将数据复制到GPU
    checkCudaErrors(cudaMemcpy(deviceInput, hostInput, 
                              inputLength * sizeof(unsigned int), 
                              cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int)));
    
    // 计算网格和块的维度
    int blockSize = BLOCK_SIZE;
    int gridSize = (inputLength + blockSize - 1) / blockSize;
    
    // 限制网格大小
    if (gridSize > 1024) gridSize = 1024;
    
    printf("Grid Size: %d, Block Size: %d\n", gridSize, blockSize);
    
    // 记录开始时间
    double start = cpuSecond();
    
    // 启动核函数
    histogram_kernel<<<gridSize, blockSize>>>(deviceInput, deviceBins,
                                            inputLength, NUM_BINS);
    convert_kernel<<<gridSize, blockSize>>>(deviceBins, NUM_BINS);
    
    // 检查kernel执行错误
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    // 记录结束时间
    double end = cpuSecond();
    printf("Kernel execution time: %f seconds\n", end - start);
    
    // 将结果复制回CPU
    checkCudaErrors(cudaMemcpy(hostBins, deviceBins, 
                              NUM_BINS * sizeof(unsigned int), 
                              cudaMemcpyDeviceToHost));
    
    // 验证结果
    bool correct = true;
    int errors = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        if (hostBins[i] != resultRef[i]) {
            if (errors < 10) {
                printf("Mismatch at bin %d: GPU = %u, CPU = %u\n", 
                       i, hostBins[i], resultRef[i]);
            }
            errors++;
            correct = false;
        }
    }
    
    if (correct) {
        printf("Results match!\n");
    } else {
        printf("Total %d mismatches found!\n", errors);
    }
    
    // 释放GPU内存
    checkCudaErrors(cudaFree(deviceInput));
    checkCudaErrors(cudaFree(deviceBins));
    
    // 释放CPU内存
    free(hostInput);
    free(hostBins);
    free(resultRef);
    
    return 0;
}